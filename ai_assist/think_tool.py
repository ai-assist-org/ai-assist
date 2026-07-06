"""Think tool — a planning/reasoning scratchpad for the agent.

Gives Claude a structured place to plan multi-step tasks, track progress,
and reason about intermediate results. The tool does nothing externally;
it just returns a minimal acknowledgement so Claude can continue.
"""

from typing import Any


class ThinkTool:
    """A no-op tool that lets the agent think, plan, and reason explicitly."""

    def get_tool_definitions(self) -> list[dict]:
        """Get tool definitions for the agent."""
        return [
            {
                "name": "internal__think",
                "description": (
                    "Use this tool to plan and reason about complex tasks. "
                    "Write out your step-by-step plan, track what you've done, "
                    "note intermediate results, or decide what to do next. "
                    "This tool has no side effects — use it freely before acting."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "thought": {
                            "type": "string",
                            "description": "Your reasoning, plan, or notes about the current task.",
                        },
                    },
                    "required": ["thought"],
                },
                "_server": "internal",
                "_original_name": "think",
                "_readonly": True,
            },
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute the think tool. Returns a minimal acknowledgement."""
        if tool_name != "think":
            return f"Error: unknown think tool '{tool_name}'"
        return "Thought recorded."
