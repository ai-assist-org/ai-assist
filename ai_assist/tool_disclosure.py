"""Progressive disclosure of tools: send the core tools, list the others, call them through a proxy.

Tool input schemas are most of the prompt (MCP servers bring dozens of tools each), so only the
tools used in nearly every conversation are sent in full. Every other tool is listed by name and
first sentence in the description of the proxy tool: the model reads its schema with
introspection__get_tool_help, then calls it through introspection__call_tool. The agent unwraps the
proxied call before running it, so caching, write and loop detection, plan mode, the audit trail
and the tool-call log all see the real tool.

The tool list stays the same within and across queries: nothing is loaded on demand, so the
prompt cache and the thinking-block signatures (bound to the exact prefix) are never disturbed.
"""

from typing import Any

CALL_TOOL = "introspection__call_tool"

# The tools behind ~98% of the calls in real use (audit trails), plus the ones the loop relies on.
CORE_TOOLS = frozenset(
    {
        "internal__execute_command",
        "internal__read_file",
        "internal__search_in_file",
        "internal__edit_file",
        "internal__write_file",
        "internal__list_directory",
        "internal__transform",
        "internal__json_query",
        "internal__think",
        "internal__web_search",
        "internal__web_fetch",
        "internal__read_report",
        "internal__write_report",
        "internal__append_to_report",
        "internal__list_reports",
        "internal__save_knowledge",
        "internal__search_knowledge",
        "internal__execute_skill_script",
        "internal__finish_task",
        "introspection__get_skill_help",
        "introspection__get_tool_help",
    }
)


def first_sentence(description: str, max_length: int = 200) -> str:
    """The first sentence (or line) of a description, at most about max_length characters."""
    description = description.strip()
    for sep in (". ", ".\n", "\n"):
        idx = description.find(sep)
        if 0 < idx <= max_length:
            return description[: idx + 1] if sep.startswith(".") else description[:idx]
    if len(description) <= max_length:
        return description
    return description[:max_length].rsplit(" ", 1)[0] + "..."


def split_tools(tools: list[dict]) -> tuple[list[dict], list[dict]]:
    """The tools to send, and the ones to list in the proxy (both in their original order)."""
    return [t for t in tools if t["name"] in CORE_TOOLS], [t for t in tools if t["name"] not in CORE_TOOLS]


def call_tool_definition(deferred: list[dict]) -> dict:
    """The proxy tool, whose description lists the deferred tools."""
    catalog = "".join(
        f"- {t['name']}: {first_sentence(t.get('_full_description') or t['description'])}\n" for t in deferred
    )
    return {
        "name": CALL_TOOL,
        "description": (
            "Call one of the tools listed below, which are not sent in full to save context. "
            "Read its parameters first with introspection__get_tool_help, then pass them in `arguments` "
            "(special parameters such as __jq_filter or __save_to_file go there too).\n\n"
            f"Tools:\n{catalog}"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the tool to call, as listed"},
                "arguments": {"type": "object", "description": "The tool's arguments"},
            },
            "required": ["name"],
        },
    }


def unwrap(name: str, arguments: Any, known: set[str]) -> tuple[str, dict, str | None]:
    """Resolve a proxied call to (tool name, arguments, error); other calls pass through."""
    if name != CALL_TOOL:
        return name, arguments, None
    target = arguments.get("name") if isinstance(arguments, dict) else None
    if not isinstance(target, str) or not target:
        return name, {}, f"Error: {CALL_TOOL} needs the `name` of the tool to call."
    if target == CALL_TOOL or target not in known:
        return target, {}, f"Error: unknown tool {target}; the tools you can call are listed in {CALL_TOOL}."
    inner = arguments.get("arguments") or {}
    if not isinstance(inner, dict):
        return target, {}, "Error: `arguments` must be an object."
    return target, inner, None
