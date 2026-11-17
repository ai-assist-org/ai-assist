"""Send only the core tools; list the others in a proxy tool that calls them by name.

The tool list never changes, so the prompt cache and thinking-block signatures stay valid.
"""

import logging
from typing import Any

import jsonschema

logger = logging.getLogger(__name__)

CALL_TOOL = "introspection__call_tool"

# The tools behind ~98% of the calls in the audit trails, plus those the agent loop relies on.
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
    description = description.strip()
    for sep in (". ", ".\n", "\n"):
        idx = description.find(sep)
        if 0 < idx <= max_length:
            return description[: idx + 1] if sep.startswith(".") else description[:idx]
    if len(description) <= max_length:
        return description
    return description[:max_length].rsplit(" ", 1)[0] + "..."


def split_tools(tools: list[dict]) -> tuple[list[dict], list[dict]]:
    return [t for t in tools if t["name"] in CORE_TOOLS], [t for t in tools if t["name"] not in CORE_TOOLS]


def call_tool_definition(deferred: list[dict]) -> dict:
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


def tool_documentation(tool: dict, schema: dict) -> dict:
    return {
        "tool_name": tool["name"],
        "description": tool.get("_full_description", tool["description"]),
        "input_schema": schema,
    }


def check_arguments(schema: dict, arguments: dict) -> list[str]:
    """The problems with arguments, unknown parameters included (strict); empty when valid."""
    strict = {**schema, "additionalProperties": schema.get("additionalProperties", False)}
    validator = jsonschema.validators.validator_for(strict, default=jsonschema.Draft202012Validator)
    try:
        validator.check_schema(strict)
    except jsonschema.SchemaError:
        # MCP servers ship their own schemas: a broken one must not block the tool
        logger.exception("Invalid JSON Schema, arguments not checked")
        return []
    return [f"{error.json_path}: {error.message}" for error in validator(strict).iter_errors(arguments)]
