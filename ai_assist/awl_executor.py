"""Shared AWL script execution for action engine, task runner, CLI, and agent tools."""

from pathlib import Path
from typing import Any

from .config import get_config_dir


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


def get_missing_variables(workflow, variables: dict[str, Any] | None = None) -> set[str]:
    """Return input variables required by the workflow but absent from the provided dict."""
    from .awl_runtime import _compute_input_variables

    required = _compute_input_variables(workflow)
    provided = set(variables.keys()) if variables else set()
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
    """
    from .awl_ast import GoalNode

    workflow, _awl_path = load_awl_workflow(prompt)
    has_goal = any(isinstance(n, GoalNode) for n in workflow.body)

    if has_goal:
        from .goal_runner import GoalRunner
        from .goal_state import GoalStateManager

        state_manager = GoalStateManager(get_config_dir() / "state")
        runner = GoalRunner(_awl_path, agent, state_manager)
        runner.load()
        await runner.run_cycle()

        lines = [f"Goal '{runner.goal_id}' cycle completed."]
        state = state_manager.load(runner.goal_id)
        lines.append(f"Status: {state.status} | Cycles: {state.cycle_count}")
        if state.success_reason:
            lines.append(f"Success: {state.success_reason}")
        return "\n".join(lines)

    from .awl_runtime import AWLRuntime

    runtime = AWLRuntime(agent, verbose=verbose)
    result = await runtime.execute(workflow, variables=variables)
    if not result.success:
        raise RuntimeError(
            f"AWL workflow failed: {result.task_outcomes[-1].summary if result.task_outcomes else 'unknown error'}"
        )
    return result.return_value or "Workflow completed successfully."
