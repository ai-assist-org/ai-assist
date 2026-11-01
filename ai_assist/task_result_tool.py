"""Finish tool for AWL tasks — how the agent reports a task's outcome.

The runtime offers ``internal__finish_task`` only while an AWL task runs. The
agent calls it once at the end with ``status`` "success" (and the exposed
values) or "failed" (and a reason), so a task that could not be done is never
mistaken for one that produced empty results. The schema is static so the tool
list, and the prompt cache, stay the same across tasks.
"""

from typing import Any

_STATUSES = ("success", "failed")


class TaskResultTool:
    """Records the outcome the agent reports for the current AWL task."""

    def __init__(self) -> None:
        self._active = False
        self._result: dict[str, Any] | None = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def reported(self) -> bool:
        """Whether the current task has called internal__finish_task yet."""
        return self._result is not None

    def begin(self) -> tuple[bool, dict[str, Any] | None]:
        """Start recording for a task; returns the previous state for ``end``."""
        previous = (self._active, self._result)
        self._active = True
        self._result = None
        return previous

    def end(self, previous: tuple[bool, dict[str, Any] | None]) -> dict[str, Any] | None:
        """Return what the task recorded (None if never called) and restore ``previous``."""
        result = self._result
        self._active, self._result = previous
        return result

    def get_tool_definitions(self) -> list[dict]:
        return [
            {
                "name": "internal__finish_task",
                "description": (
                    "Report the outcome of the current workflow task. Call it once, when you are done. "
                    "Use status 'success' with the requested values, or status 'failed' with a reason "
                    "if the goal could not be achieved (a required step failed, a tool returned an "
                    "error, needed data is missing). Never report empty or made-up values as success."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": list(_STATUSES)},
                        "reason": {
                            "type": "string",
                            "description": "Why the task failed (required when status is 'failed').",
                        },
                        "values": {
                            "type": "object",
                            "description": "The values the task asks you to expose, keyed by name.",
                        },
                    },
                    "required": ["status"],
                },
                "_server": "internal",
                "_original_name": "finish_task",
            },
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if tool_name != "finish_task":
            return f"Error: unknown task result tool '{tool_name}'"
        if not self._active:
            return "Error: internal__finish_task is only available inside a workflow task"
        status = arguments.get("status")
        if status not in _STATUSES:
            return f"Error: status must be one of {', '.join(_STATUSES)}"
        reason = str(arguments.get("reason") or "").strip()
        if status == "failed" and not reason:
            return "Error: status 'failed' requires a reason"
        values = arguments.get("values") or {}
        if not isinstance(values, dict):
            return "Error: values must be an object keyed by variable name"
        self._result = {"status": status, "reason": reason, "values": values}
        return "Task result recorded. Reply with a one-line summary and stop."
