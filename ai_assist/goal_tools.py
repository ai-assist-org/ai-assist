"""Internal tools for managing autonomous goals (AWL-based)"""

import logging
import re
from pathlib import Path

from ai_assist.config import get_config_dir
from ai_assist.goal_state import GoalStateManager

logger = logging.getLogger(__name__)


def _slugify(title: str) -> str:
    """Convert title to a filesystem-safe slug"""
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return slug[:50] if slug else "goal"


class GoalTools:
    """Internal tools for creating and managing autonomous goals via AWL"""

    def __init__(self, agent, goals_file: Path | None = None):
        self.agent = agent
        self.goals_dir = get_config_dir() / "goals"
        self.goals_dir.mkdir(parents=True, exist_ok=True)
        self.state_manager = GoalStateManager(get_config_dir() / "state")

    def get_tool_definitions(self) -> list[dict]:
        """Return tool definitions for agent"""
        return [
            {
                "name": "goal__create",
                "_readonly": False,
                "description": (
                    "Create a new autonomous goal as an AWL script (.awl file). "
                    "The generated file uses the @goal directive with a @task inside. "
                    "The user can then run it from CLI (ai-assist /run <file>) or "
                    "schedule it via internal__create_action. Each cycle, the goal body executes "
                    "and success criteria are evaluated — when met, the goal completes. "
                    "Variables persist between cycles."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Short title for the goal (used as filename)",
                        },
                        "description": {
                            "type": "string",
                            "description": "Detailed description of what to achieve (becomes the task Goal)",
                        },
                        "success_criteria": {
                            "type": "string",
                            "description": "When is the goal complete? (becomes the Success: field)",
                        },
                        "max_actions": {
                            "type": "integer",
                            "description": "Maximum tool calls per cycle. Default: 5.",
                        },
                    },
                    "required": ["title", "description", "success_criteria"],
                },
            },
            {
                "name": "goal__list",
                "_readonly": True,
                "description": "List all goals with their current status",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "status_filter": {
                            "type": "string",
                            "enum": ["all", "active", "paused", "completed", "cancelled"],
                            "description": "Filter goals by status. Default: all.",
                        },
                    },
                },
            },
            {
                "name": "goal__update",
                "_readonly": False,
                "description": ("Update a goal's status. Use this to pause, resume, or cancel a goal."),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "goal_id": {
                            "type": "string",
                            "description": "ID of the goal to update",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["active", "paused", "completed", "cancelled"],
                            "description": "New status for the goal",
                        },
                    },
                    "required": ["goal_id", "status"],
                },
            },
        ]

    async def execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a goal management tool"""
        if tool_name == "goal__create":
            return await self._create_goal(arguments)
        elif tool_name == "goal__list":
            return await self._list_goals(arguments)
        elif tool_name == "goal__update":
            return await self._update_goal(arguments)

        return f"Unknown tool: {tool_name}"

    async def _create_goal(self, args: dict) -> str:
        """Create a new goal as an AWL file"""
        title = args["title"]
        description = args["description"]
        success_criteria = args["success_criteria"]
        max_actions = args.get("max_actions", 5)

        goal_id = _slugify(title)
        awl_path = self.goals_dir / f"{goal_id}.awl"

        # Avoid overwriting existing goals
        if awl_path.exists():
            return f"Error: Goal '{goal_id}' already exists at {awl_path}"

        awl_content = f"""\
@start

@goal {goal_id} max_actions={max_actions}
  Success: {success_criteria}

  @task {goal_id}_check @no-history
  Goal: {description}
  Expose: result
  @end

@end

@end
"""
        awl_path.write_text(awl_content)

        return (
            f"Goal created: {title}\n"
            f"  ID: {goal_id}\n"
            f"  File: {awl_path}\n"
            f"  Max actions: {max_actions}\n\n"
            f"To run once: ai-assist /run {awl_path}\n"
            f"To schedule: use internal__create_action with prompt={awl_path}\n"
            f"Edit {awl_path} to customize the AWL workflow."
        )

    async def _list_goals(self, args: dict) -> str:
        """List goals by scanning AWL files and state"""
        from .awl_ast import GoalNode
        from .awl_parser import AWLParser

        status_filter = args.get("status_filter", "all")
        awl_files = sorted(self.goals_dir.glob("*.awl"))

        if not awl_files:
            return "No goals found."

        lines = []
        count = 0
        for awl_path in awl_files:
            try:
                source = awl_path.read_text()
                workflow = AWLParser(source).parse()
                goal_nodes = [n for n in workflow.body if isinstance(n, GoalNode)]
                if not goal_nodes:
                    continue

                goal_node = goal_nodes[0]
                state = self.state_manager.load(goal_node.goal_id)

                if status_filter not in ("all", state.status):
                    continue

                status_icon = {
                    "active": "[ACTIVE]",
                    "paused": "[PAUSED]",
                    "completed": "[DONE]",
                    "cancelled": "[CANCELLED]",
                }.get(state.status, f"[{state.status}]")

                lines.append(
                    f"  {status_icon} {goal_node.goal_id}\n"
                    f"    Success: {goal_node.success_criteria}\n"
                    f"    Cycles: {state.cycle_count} | File: {awl_path.name}"
                )
                count += 1
            except Exception as e:
                lines.append(f"  [ERROR] {awl_path.name}: {e}")

        if not lines:
            return f"No goals found{' with status ' + status_filter if status_filter != 'all' else ''}."

        return f"Goals ({count}):\n" + "\n".join(lines)

    async def _update_goal(self, args: dict) -> str:
        """Update a goal's status via state file"""
        goal_id = args["goal_id"]
        new_status = args["status"]

        # Check the goal AWL file exists
        awl_path = self.goals_dir / f"{goal_id}.awl"
        if not awl_path.exists():
            return f"Error: Goal not found with ID '{goal_id}'"

        state = self.state_manager.load(goal_id)
        state.status = new_status
        self.state_manager.save(goal_id, state)

        return f"Goal updated: {goal_id}\n  Status: {new_status}"
