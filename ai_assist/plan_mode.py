"""Plan mode — explore-then-execute workflow with user approval."""

from dataclasses import dataclass

# Tool name prefixes that indicate read operations (heuristic fallback for
# MCP tools without a _readonly annotation; anything else counts as a write)
_READ_PREFIXES = (
    "get_",
    "list_",
    "search_",
    "query_",
    "find_",
    "count_",
    "read_",
    "describe_",
    "show_",
    "fetch_",
    "lookup_",
    "inspect_",
    "view_",
)


def is_planning_tool(tool: dict) -> bool:
    """Whether a tool is read-only and so usable in plan mode.

    Uses the ``_readonly`` annotation when present on a tool dict. Falls back
    to read-style name prefixes for MCP tools without it (configure
    ``readonly_tools`` to allow others).
    """
    if "_readonly" in tool:
        return bool(tool["_readonly"])
    return tool["name"].split("__", 1)[-1].startswith(_READ_PREFIXES)


def get_planning_tools(available_tools: list[dict]) -> list[dict]:
    """Filter tools to read-only for plan mode."""
    return [tool for tool in available_tools if is_planning_tool(tool)]


@dataclass
class PlanSession:
    task: str
    plan_text: str | None = None
    status: str = "planning"


def get_plan_system_prompt(task: str, revision_feedback: str | None = None) -> str:
    """Build the planning-phase prompt for the agent."""
    prompt = f"""You are in PLAN MODE. Explore and analyze before proposing any changes.

## Task
{task}

## Instructions
1. Use read-only tools (search, query, read, list, introspection, think) to understand the current state
2. Identify what needs to be done
3. Produce a structured plan with these sections:

### Context
Brief summary of what you found during exploration.

### Steps
Numbered list of concrete actions to take. For each step specify what tool to use and the expected outcome.

### Resources Affected
List of tickets, reports, or other resources that will be created or modified.

### Verification
How to verify the plan was executed correctly.

### Risks
Any risks or things that might go wrong.

## Constraints
- You can ONLY use read-only tools
- Do NOT modify, create, write, or delete anything
- Focus on understanding and planning, not acting"""

    if revision_feedback:
        prompt += f"""

## Revision Request
The user reviewed a previous version of this plan and asked for changes:
{revision_feedback}

Please produce an updated plan addressing this feedback."""

    return prompt


def get_execution_prompt(plan_text: str) -> str:
    """Wrap an approved plan into an execution prompt."""
    return f"""The user has approved the following plan. Execute it now.

{plan_text}

After completing each step, briefly confirm what was done. If a step cannot be completed, explain why."""
