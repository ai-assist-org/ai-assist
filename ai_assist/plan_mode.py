"""Plan mode — explore-then-execute workflow with user approval."""

from dataclasses import dataclass

# MCP tool prefixes that indicate write operations (heuristic fallback)
_WRITE_PREFIXES = ("create_", "update_", "delete_", "add_", "write_")


def get_planning_tools(available_tools: list[dict]) -> list[dict]:
    """Filter tools to read-only for plan mode.

    Uses the ``_readonly`` annotation when present on a tool dict.
    Falls back to a name-prefix heuristic for MCP tools without it.
    """
    result = []
    for tool in available_tools:
        if "_readonly" in tool:
            if tool["_readonly"]:
                result.append(tool)
            continue

        # Heuristic fallback for MCP tools without _readonly annotation
        name = tool["name"]
        if "__" in name:
            _, tool_part = name.split("__", 1)
            if tool_part.startswith(_WRITE_PREFIXES):
                continue

        result.append(tool)

    return result


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
