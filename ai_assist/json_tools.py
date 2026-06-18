"""JSON query tool — wraps jq for efficient JSON file processing."""

import os
import shutil
import subprocess
from typing import Any


class JsonTools:
    """Internal JSON query tool using jq.

    Only exposes tools when jq is installed on the system.
    """

    def __init__(self, filesystem_tools):
        self.filesystem_tools = filesystem_tools
        self.jq_path: str | None = shutil.which("jq")

    def get_tool_definitions(self) -> list[dict]:
        if not self.jq_path:
            return []

        return [
            {
                "name": "internal__json_query",
                "description": (
                    "Query or transform a JSON file using a jq filter expression. "
                    "Use this instead of writing Python scripts for JSON processing. "
                    "Common filters: `.key` (extract field), `.[] | {id, status}` (project fields), "
                    '`[.[] | select(.status == "failed")]` (filter array), '
                    "`length` (count elements), `map(.field)` (transform array), "
                    "`group_by(.key)` (group elements), `sort_by(.key)` (sort), "
                    '`[.[] | select(.name | test("pattern"))]` (regex match).'
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file": {
                            "type": "string",
                            "description": "Path to the JSON file to query.",
                        },
                        "filter": {
                            "type": "string",
                            "description": "jq filter expression.",
                        },
                        "raw_output": {
                            "type": "boolean",
                            "description": "Use -r for raw string output (no JSON quotes).",
                            "default": False,
                        },
                        "slurp": {
                            "type": "boolean",
                            "description": "Use --slurp to read all inputs into an array. Auto-enabled for .jsonl files.",
                            "default": False,
                        },
                    },
                    "required": ["file", "filter"],
                },
                "_server": "internal",
                "_original_name": "json_query",
            },
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if tool_name != "json_query":
            return f"Error: unknown json tool '{tool_name}'"

        file_path = os.path.expanduser(arguments["file"])
        path_error = await self.filesystem_tools._validate_path(file_path)
        if path_error:
            return path_error

        filter_expr = arguments["filter"]
        raw_output = arguments.get("raw_output", False)
        slurp = arguments.get("slurp", False)

        jq = self.jq_path
        if not jq:
            return "Error: jq is not installed"

        if not slurp and file_path.endswith(".jsonl"):
            slurp = True

        args = [jq]
        if slurp:
            args.append("--slurp")
        if raw_output:
            args.append("-r")
        args.extend([filter_expr, file_path])

        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=30, check=False)
        except subprocess.TimeoutExpired:
            return "jq error: query timed out after 30 seconds"

        if result.returncode != 0:
            return f"jq error: {result.stderr.strip()}"

        return result.stdout

    def filter_string(self, input_text: str, filter_expr: str) -> str:
        """Run a jq filter on a string (piped via stdin)."""
        if not self.jq_path:
            return "Error: jq is not installed. __jq_filter requires jq."

        try:
            result = subprocess.run(
                [self.jq_path, filter_expr],
                input=input_text,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "jq error: filter timed out after 30 seconds"

        if result.returncode != 0:
            return f"jq error: {result.stderr.strip()}"

        return result.stdout
