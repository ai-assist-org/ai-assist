"""Shared AWL script execution for action engine, task runner, CLI, and agent tools."""

import atexit
import fcntl
import hashlib
import json
import logging
import os
import shutil
import tempfile
from datetime import UTC, datetime
from io import TextIOWrapper
from pathlib import Path
from typing import Any

from .config import get_config_dir, get_reports_dir, load_local_variables

logger = logging.getLogger(__name__)

_session_tmpdir: str | None = None


def init_session_tmpdir() -> str:
    """Create a session-scoped temporary directory. Returns the path."""
    global _session_tmpdir  # noqa: PLW0603
    if _session_tmpdir is None:
        _session_tmpdir = tempfile.mkdtemp(prefix="ai-assist-")
        logger.info("Session tmpdir created: %s", _session_tmpdir)
    return _session_tmpdir


def cleanup_session_tmpdir() -> None:
    """Remove the session temporary directory if it exists."""
    global _session_tmpdir  # noqa: PLW0603
    if _session_tmpdir is not None:
        shutil.rmtree(_session_tmpdir, ignore_errors=True)
        logger.info("Session tmpdir removed: %s", _session_tmpdir)
        _session_tmpdir = None


def _get_session_tmpdir() -> str:
    """Get or lazily create the session tmpdir."""
    if _session_tmpdir is not None:
        return _session_tmpdir
    # Lazy init with atexit cleanup for non-CLI usage
    path = init_session_tmpdir()
    atexit.register(cleanup_session_tmpdir)
    return path


def _generate_builtin_variables() -> dict[str, str]:
    """Generate built-in date/time and path variables for AWL execution."""
    now_local = datetime.now().astimezone()
    now_utc = datetime.now(UTC)

    config_dir = get_config_dir()

    return {
        "date": now_local.strftime("%Y-%m-%d"),
        "time": now_local.strftime("%H:%M:%S"),
        "datetime": now_local.strftime("%Y-%m-%d %H:%M:%S"),
        "utc_date": now_utc.strftime("%Y-%m-%d"),
        "utc_time": now_utc.strftime("%H:%M:%S"),
        "utc_datetime": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": now_local.strftime("%Z"),
        "config_dir": str(config_dir),
        "reports_dir": str(get_reports_dir()),
        "logs_dir": str(config_dir / "logs"),
        "tmpdir": _get_session_tmpdir(),
    }


def _merge_all_variables(variables: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge builtin, local, and caller-provided variables (caller wins)."""
    merged = _generate_builtin_variables()
    merged.update(load_local_variables())
    if variables:
        merged.update(variables)
    return merged


def load_awl_workflow(prompt: str):
    """Parse an AWL script file and return the workflow AST.

    Resolves ~ paths. Relative paths are tried against cwd first,
    then the config dir.
    Raises FileNotFoundError or ParseError on failure.
    """
    from .awl_parser import AWLParser

    awl_path = Path(prompt).expanduser()
    if not awl_path.is_absolute():
        cwd_path = Path.cwd() / prompt
        if cwd_path.exists():
            awl_path = cwd_path
        else:
            awl_path = get_config_dir() / prompt

    if not awl_path.exists():
        raise FileNotFoundError(f"AWL script not found: {awl_path}")

    source = awl_path.read_text()
    return AWLParser(source).parse(), awl_path


def _try_lock_script(resolved_path: Path) -> TextIOWrapper | None:
    """Try to acquire a cross-process exclusive lock for an AWL script.

    Returns the open file handle (keeping the lock alive) on success,
    or None if the script is already locked by another execution.
    """
    locks_dir = get_config_dir() / "locks"
    locks_dir.mkdir(parents=True, exist_ok=True)

    path_hash = hashlib.sha256(str(resolved_path).encode()).hexdigest()[:16]
    lock_file = locks_dir / f"{path_hash}.lock"

    try:
        fh = lock_file.open("w")
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fh.write(str(resolved_path))
        fh.flush()
        return fh
    except OSError:
        return None


def get_missing_variables(workflow, variables: dict[str, Any] | None = None) -> set[str]:
    """Return input variables required by the workflow but absent from the provided dict."""
    from .awl_runtime import _compute_input_variables

    required = _compute_input_variables(workflow)
    provided = set(_merge_all_variables(variables).keys())
    return required - provided


async def run_awl_script(
    prompt: str, agent: object, variables: dict[str, Any] | None = None, verbose: bool = False
) -> str:
    """Execute an AWL script, handling both @goal and @start workflows.

    This is the single entry point for all AWL execution paths:
    - CLI /run command
    - Action engine (scheduled actions)
    - Task runner
    - Agent tool (introspection__execute_awl_script)

    Variables are merged from three sources (highest precedence wins):
    1. Caller-provided variables (CLI key=value, agent tool)
    2. User-defined variables from ~/.ai-assist/variables.yaml
    3. Built-in variables (date/time, paths)
    """
    from .awl_ast import GoalNode

    merged = _merge_all_variables(variables)

    workflow, _awl_path = load_awl_workflow(prompt)

    lock_fh = _try_lock_script(_awl_path)
    if lock_fh is None:
        logger.warning(
            "AWL script lock rejected: %s",
            json.dumps(
                {
                    "event": "awl_lock_rejected",
                    "script": prompt,
                    "resolved_path": str(_awl_path),
                    "pid": os.getpid(),
                }
            ),
        )
        return f"Skipped: '{prompt}' is already running in another session."

    import time

    _lock_log_ctx = {
        "script": prompt,
        "resolved_path": str(_awl_path),
        "pid": os.getpid(),
    }
    logger.info(
        "AWL script lock acquired: %s",
        json.dumps({"event": "awl_lock_acquired", **_lock_log_ctx}),
    )
    _lock_start = time.monotonic()
    try:
        has_goal = any(isinstance(n, GoalNode) for n in workflow.body)

        if has_goal:
            from .goal_runner import GoalRunner
            from .goal_state import GoalStateManager

            state_manager = GoalStateManager(get_config_dir() / "state")
            runner = GoalRunner(_awl_path, agent, state_manager, initial_variables=merged)
            runner.load()
            await runner.run_cycle()

            lines = [f"Goal '{runner.goal_id}' cycle completed."]
            state = state_manager.load(runner.goal_id)
            lines.append(f"Status: {state.status} | Cycles: {state.cycle_count}")
            if state.success_reason:
                lines.append(f"Success: {state.success_reason}")
            return "\n".join(lines)

        from .awl_runtime import AWLRuntime, AWLRuntimeError

        runtime = AWLRuntime(agent, verbose=verbose)
        try:
            result = await runtime.execute(workflow, variables=merged)
        except AWLRuntimeError as e:
            logger.error("AWL workflow aborted: %s", e)
            raise RuntimeError(f"AWL workflow aborted: {e}") from e
        if not result.success:
            raise RuntimeError(
                f"AWL workflow failed: {result.task_outcomes[-1].summary if result.task_outcomes else 'unknown error'}"
            )
        return result.return_value or "Workflow completed successfully."
    finally:
        duration = round(time.monotonic() - _lock_start, 2)
        lock_fh.close()
        logger.info(
            "AWL script lock released: %s",
            json.dumps({"event": "awl_lock_released", "duration_s": duration, **_lock_log_ctx}),
        )
