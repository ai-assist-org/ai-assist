"""Agent tools for managing actions in event-schedules.json"""

import json
import logging
from pathlib import Path
from typing import Any

from .action_loader import ActionLoader
from .action_model import ActionDefinition
from .config import get_config_dir

logger = logging.getLogger(__name__)


class ActionTools:
    """Agent tools for CRUD operations on actions"""

    def __init__(self, schedules_file: Path | None = None, known_mcp_servers: set[str] | None = None) -> None:
        if schedules_file is None:
            schedules_file = get_config_dir() / "event-schedules.json"
        self.schedules_file = Path(schedules_file)
        self.schedules_file.parent.mkdir(parents=True, exist_ok=True)
        self.known_mcp_servers = known_mcp_servers

    def get_tool_definitions(self) -> list[dict]:
        return [
            {
                "name": "internal__create_action",
                "description": (
                    "Create a new action with a trigger. Trigger types: "
                    "'interval' (every: '5m'), 'schedule' (at: '9:00', days: 'weekdays'), "
                    "'interval_range' (every: '1h', between: '9:00', and: '18:00'), "
                    "'mqtt' (topic: 'alerts/#'), 'dbus' (interface, signal), "
                    "'once' (at: '2026-05-07T14:00:00')."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Unique action name"},
                        "trigger": {
                            "type": "object",
                            "description": "Trigger configuration with 'type' field",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Prompt to execute. Formats: 'mcp://server/prompt', path.awl, or natural language.",
                        },
                        "prompt_arguments": {"type": "object", "description": "Arguments for MCP prompts"},
                        "enabled": {"type": "boolean", "description": "Whether action is enabled (default: true)"},
                        "notify": {"type": "boolean", "description": "Send notification on completion"},
                        "conditions": {"type": "array", "items": {"type": "object"}},
                    },
                    "required": ["name", "trigger", "prompt"],
                },
                "_server": "internal",
            },
            {
                "name": "internal__list_actions",
                "description": "List all configured actions with their triggers and status.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "trigger_type": {
                            "type": "string",
                            "description": "Filter by trigger type (interval, schedule, mqtt, dbus, once, etc.)",
                        },
                    },
                },
                "_server": "internal",
            },
            {
                "name": "internal__update_action",
                "description": "Update an existing action's properties.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the action to update"},
                        "trigger": {"type": "object", "description": "New trigger configuration"},
                        "prompt": {"type": "string", "description": "New prompt"},
                        "prompt_arguments": {"type": "object"},
                        "enabled": {"type": "boolean"},
                        "notify": {"type": "boolean"},
                        "conditions": {"type": "array", "items": {"type": "object"}},
                    },
                    "required": ["name"],
                },
                "_server": "internal",
            },
            {
                "name": "internal__delete_action",
                "description": "Delete an action by name.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the action to delete"},
                    },
                    "required": ["name"],
                },
                "_server": "internal",
            },
            {
                "name": "internal__enable_action",
                "description": "Enable or disable an action.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Action name"},
                        "enabled": {"type": "boolean", "description": "true to enable, false to disable"},
                    },
                    "required": ["name", "enabled"],
                },
                "_server": "internal",
            },
            {
                "name": "internal__get_action_status",
                "description": "Get execution history and current state of an action.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Action name"},
                        "history_limit": {"type": "integer", "description": "Max history entries (default: 5)"},
                    },
                    "required": ["name"],
                },
                "_server": "internal",
            },
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        handlers = {
            "internal__create_action": self._create_action,
            "internal__list_actions": self._list_actions,
            "internal__update_action": self._update_action,
            "internal__delete_action": self._delete_action,
            "internal__enable_action": self._enable_action,
            "internal__get_action_status": self._get_action_status,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return f"Unknown tool: {tool_name}"
        return await handler(arguments)

    async def _create_action(self, args: dict[str, Any]) -> str:
        try:
            action = ActionDefinition.from_dict(args)
            action.validate_definition()
        except (KeyError, ValueError) as e:
            return f"Invalid action: {e}"

        loader = ActionLoader(self.schedules_file)
        actions = loader.load_actions()

        if any(a.name == action.name for a in actions):
            return f"Action '{action.name}' already exists. Use update_action to modify it."

        actions.append(action)
        loader.save_actions(actions)
        return f"Created action '{action.name}' with trigger type '{action.trigger_type}'"

    async def _list_actions(self, args: dict[str, Any]) -> str:
        loader = ActionLoader(self.schedules_file)
        actions = loader.load_actions()

        trigger_filter = args.get("trigger_type")
        if trigger_filter:
            actions = [a for a in actions if a.trigger_type == trigger_filter]

        if not actions:
            return "No actions configured."

        lines = []
        for a in actions:
            status = "enabled" if a.enabled else "disabled"
            trigger_info = json.dumps(a.trigger, separators=(",", ":"))
            lines.append(f"- {a.name} [{status}] trigger={trigger_info} prompt={a.prompt[:60]}")

        return "\n".join(lines)

    async def _update_action(self, args: dict[str, Any]) -> str:
        name = args["name"]
        loader = ActionLoader(self.schedules_file)
        actions = loader.load_actions()

        target = None
        for a in actions:
            if a.name == name:
                target = a
                break

        if target is None:
            return f"Action '{name}' not found."

        if "trigger" in args:
            target.trigger = args["trigger"]
        if "prompt" in args:
            target.prompt = args["prompt"]
        if "prompt_arguments" in args:
            target.prompt_arguments = args["prompt_arguments"]
        if "enabled" in args:
            target.enabled = args["enabled"]
        if "notify" in args:
            target.notify = args["notify"]
        if "conditions" in args:
            target.conditions = args["conditions"]

        try:
            target.validate_definition()
        except ValueError as e:
            return f"Invalid update: {e}"

        loader.save_actions(actions)
        return f"Updated action '{name}'"

    async def _delete_action(self, args: dict[str, Any]) -> str:
        name = args["name"]
        loader = ActionLoader(self.schedules_file)
        actions = loader.load_actions()

        new_actions = [a for a in actions if a.name != name]
        if len(new_actions) == len(actions):
            return f"Action '{name}' not found."

        loader.save_actions(new_actions)
        return f"Deleted action '{name}'"

    async def _enable_action(self, args: dict[str, Any]) -> str:
        name = args["name"]
        enabled = args["enabled"]
        loader = ActionLoader(self.schedules_file)
        actions = loader.load_actions()

        for a in actions:
            if a.name == name:
                a.enabled = enabled
                loader.save_actions(actions)
                state = "enabled" if enabled else "disabled"
                return f"Action '{name}' {state}"

        return f"Action '{name}' not found."

    async def _get_action_status(self, args: dict[str, Any]) -> str:
        from .action_engine import ActionEngine
        from .state import StateManager

        name = args["name"]
        limit = args.get("history_limit", 5)

        loader = ActionLoader(self.schedules_file)
        actions = loader.load_actions()

        target = None
        for a in actions:
            if a.name == name:
                target = a
                break

        if target is None:
            return f"Action '{name}' not found."

        state_key = ActionEngine._state_key(target)
        state_manager = StateManager()

        state = state_manager.get_monitor_state(state_key)
        history = state_manager.get_history(state_key, limit=limit)

        lines = [f"Action: {name}", f"Trigger: {json.dumps(target.trigger)}", f"Enabled: {target.enabled}"]

        if state.last_check:
            lines.append(f"Last run: {state.last_check.isoformat()}")
            lines.append(f"Last success: {state.last_results.get('last_success', 'unknown')}")

        if history:
            lines.append(f"\nRecent history ({len(history)} entries):")
            for entry in history:
                result = entry.get("result", entry)
                ts = entry.get("timestamp", "?")
                success = result.get("success", "?")
                lines.append(f"  {ts}: success={success}")

        return "\n".join(lines)
