"""internal__transform: run a short Python payload over one file, instead of `python3 -c` or awk.

ai-assist validates and reads the file, redacts it, and hands the text to a child process
(`transform_harness`) that runs the payload with RestrictedPython and preloaded modules only, after
confining itself with the kernel (Landlock + seccomp on Linux; on macOS it is started under
sandbox-exec): the payload cannot import, open a file, write, spawn or reach the network.
"""

import asyncio
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from .filesystem_tools import FilesystemTools, _kill_process_group, _limit_output
from .redaction import Redactor
from .transform_harness import PRELOADED

logger = logging.getLogger(__name__)

INPUT_CAP = 50 * 1024 * 1024
HARNESS = [sys.executable, "-I", "-m", "ai_assist.transform_harness"]


def _sandbox_profile() -> str:
    """macOS sandbox-exec profile: no network, process, write, nor read in $HOME outside Python."""

    def quoted(path: str) -> str:
        return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'

    readable = {sys.prefix, sys.base_prefix, str(Path(__file__).resolve().parent.parent)}
    allowed = " ".join(f"(subpath {quoted(p)})" for p in sorted(readable))
    return (
        "(version 1)\n(allow default)\n(deny network*)\n(deny process-exec*)\n(deny process-fork)\n"
        f"(deny file-write*)\n(deny file-read* (subpath {quoted(str(Path.home()))}))\n"
        f"(allow file-read* {allowed})\n"
    )


def child_command() -> tuple[list[str], bool]:
    """The command starting the harness, and whether it runs under sandbox-exec (macOS)."""
    sandbox_exec = shutil.which("sandbox-exec") if sys.platform == "darwin" else None
    if sandbox_exec:
        return [sandbox_exec, "-p", _sandbox_profile(), *HARNESS], True
    return list(HARNESS), False


async def run_child(request: dict[str, Any], timeout: float) -> dict | str:
    """Run the harness in a child process; its JSON result, or an error text."""
    command, sandboxed = child_command()
    with tempfile.TemporaryDirectory(prefix="ai-assist-transform-") as cwd:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env={},
            start_new_session=True,
        )
        payload = json.dumps({**request, "sandboxed": sandboxed}).encode()
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(payload), timeout)
        except TimeoutError:
            _kill_process_group(proc.pid)
            await proc.wait()
            return f"transform error: timed out after {timeout:g} seconds"
    try:
        return json.loads(stdout)
    except ValueError:
        logger.error("transform child failed (exit %s): %s", proc.returncode, stderr.decode(errors="replace"))
        return f"transform error: the transform process failed (exit code {proc.returncode})"


async def confinement_status() -> str:
    """How the transform child is confined on this system (for /doctor)."""
    result = await run_child({"code": "", "text": "", "path": "-"}, TransformTools.timeout_seconds)
    return result["confinement"] if isinstance(result, dict) else result


class TransformTools:
    """Tool definition and execution of internal__transform."""

    timeout_seconds = 30
    _warned = False  # the weaker-layers warning is logged once per process

    def __init__(self, filesystem_tools: FilesystemTools, redactor: Redactor) -> None:
        self.filesystem_tools = filesystem_tools
        self.redactor = redactor

    def get_tool_definitions(self) -> list[dict]:
        modules = ", ".join(PRELOADED)
        return [
            {
                "name": "internal__transform",
                "description": (
                    "Run a short Python payload over one file to parse, count, filter or reshape it: use "
                    "this instead of `python3 -c`, awk, or `| cut | sort | uniq` pipelines. The payload "
                    "gets `text` (the file content), `lines` (text.splitlines()) and `path` (the file "
                    f"name), and these preloaded modules: {modules} (io: StringIO only; yaml: safe_load, "
                    "safe_load_all, safe_dump); importing them is optional, any other import fails. "
                    "Print the result. The payload cannot open another file, write or spawn "
                    "processes, and cannot use names or attributes "
                    "starting with `_` or str.format (use f-strings). Not available: open (the content "
                    "is in `text`), eval/exec, setattr, vars, and the statements match, nonlocal, "
                    "async and except*. To process another tool's result, save it first with "
                    "`__save_to_file`."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string", "description": "Path of the file to process"},
                        "code": {"type": "string", "description": "Python payload; print the result"},
                        "max_output_lines": {"type": "integer", "description": "Show only this many lines"},
                        "from_end": {
                            "type": "boolean",
                            "description": "With max_output_lines: show the last lines",
                        },
                        "max_line_chars": {
                            "type": "integer",
                            "description": "Cut each output line to this many characters",
                        },
                    },
                    "required": ["file", "code"],
                },
                "_server": "internal",
                "_original_name": "transform",
                "_readonly": True,
            }
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if tool_name != "transform":
            return f"Error: unknown transform tool '{tool_name}'"
        path = Path(arguments["file"]).expanduser()
        path_error = await self.filesystem_tools._validate_path(str(path))
        if path_error:
            return path_error
        if not path.is_file():
            return f"Error: File not found: {arguments['file']}"
        if path.stat().st_size > INPUT_CAP:
            return f"Error: {arguments['file']} is larger than {INPUT_CAP // (1024 * 1024)} MB"
        text = self.redactor.redact(path.read_text(errors="replace"))

        result = await run_child({"code": arguments["code"], "text": text, "path": path.name}, self.timeout_seconds)
        if isinstance(result, str):
            return result
        output = result["output"]
        if result["error"]:
            prefix = output.rstrip("\n") + "\n" if output else ""
            return f"{prefix}transform error: {result['error']}" + self._confinement_note(result)
        return (self._shape(output, arguments) or "(no output)") + self._confinement_note(result)

    def _confinement_note(self, result: dict) -> str:
        """Say when the kernel confinement was missing or partial (the payload ran anyway)."""
        confinement = result.get("confinement", "")
        if not (confinement.startswith("unavailable") or "not blocked" in confinement):
            return ""
        if not TransformTools._warned:
            TransformTools._warned = True
            logger.warning("internal__transform runs without full kernel confinement: %s", confinement)
        return (
            f"\n[note: kernel confinement incomplete ({confinement}); ran with the Python guard and " "resource limits]"
        )

    def _shape(self, output: str, arguments: dict[str, Any]) -> str:
        """Apply the optional output limits, saving the whole output when lines are cut."""
        max_lines, max_chars = arguments.get("max_output_lines"), arguments.get("max_line_chars")
        if not (max_lines or max_chars):
            return output
        shown, note = _limit_output(output, max_lines, bool(arguments.get("from_end", False)), max_chars)
        if note.startswith("showing"):
            saved = self.filesystem_tools._save_full_output(output)
            note += f"; full output saved to {saved}: read it with internal__read_file"
        return shown + (f"[{note}]\n" if note else "")
