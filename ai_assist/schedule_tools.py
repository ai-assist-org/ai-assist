"""Internal schedule management tools for ai-assist"""

import json
from pathlib import Path

from .config import get_config_dir
from .tasks import TaskDefinition


class ScheduleTools:
    """Internal tools for managing monitor and task schedules"""

    def __init__(self, schedules_file: Path | None = None, known_mcp_servers: set[str] | None = None):
        """Initialize schedule tools

        Args:
            schedules_file: Path to schedules JSON file (defaults to ~/.ai-assist/schedules.json)
            known_mcp_servers: Set of configured MCP server names for validation
        """
        if schedules_file is None:
            schedules_file = get_config_dir() / "schedules.json"

        self.schedules_file = Path(schedules_file)
        self.schedules_file.parent.mkdir(parents=True, exist_ok=True)
        self.known_mcp_servers = known_mcp_servers

    def get_tool_definitions(self) -> list[dict]:
        """Get tool definitions for the AI agent"""
        return [
            {
                "name": "internal__create_monitor",
                "description": "Create a new monitor schedule with knowledge graph integration. When user references an MCP prompt (like '/server/prompt'), convert it to 'mcp://server/prompt' format and extract any arguments.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Monitor name (must be unique across all schedules)",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Monitoring prompt to execute. Three formats supported: (1) MCP prompts: 'mcp://server/prompt' (e.g., 'mcp://dci/rca'). (2) AWL scripts: use the filesystem path ending in .awl (e.g., '/home/user/scripts/ci_duty.awl' or 'goals/my_goal.awl'). Do NOT prefix AWL paths with 'mcp://'. (3) Natural language: plain text (e.g., 'Find failures').",
                        },
                        "prompt_arguments": {
                            "type": "object",
                            "description": 'Arguments for MCP prompts. Extract from user\'s request (e.g., \'for Semih\' → {"for": "Semih"}, \'last 7 days\' → {"days": "7"}). Leave empty for natural language prompts.',
                        },
                        "interval": {
                            "type": "string",
                            "description": "Interval (e.g., '5m', '1h', 'morning on weekdays')",
                        },
                        "description": {
                            "type": "string",
                            "description": "Optional description of what this monitor does",
                        },
                        "enabled": {
                            "type": "boolean",
                            "description": "Whether monitor is enabled (default: true)",
                        },
                        "conditions": {
                            "type": "array",
                            "description": "Optional list of condition dictionaries",
                            "items": {"type": "object"},
                        },
                        "knowledge_graph": {
                            "type": "object",
                            "description": "Optional knowledge graph configuration",
                        },
                    },
                    "required": ["name", "prompt", "interval"],
                },
                "_server": "internal",
            },
            {
                "name": "internal__create_task",
                "description": "Create a new periodic task schedule. When user references an MCP prompt (like '/server/prompt'), convert it to 'mcp://server/prompt' format and extract any arguments.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Task name (must be unique across all schedules)",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Task prompt to execute. Three formats supported: (1) MCP prompts: 'mcp://server/prompt' (e.g., 'mcp://dci/rca'). (2) AWL scripts: use the filesystem path ending in .awl (e.g., '/home/user/scripts/ci_duty.awl' or 'goals/my_goal.awl'). Do NOT prefix AWL paths with 'mcp://'. (3) Natural language: plain text (e.g., 'Find failures').",
                        },
                        "prompt_arguments": {
                            "type": "object",
                            "description": 'Arguments for MCP prompts. Extract from user\'s request (e.g., \'for Semih\' → {"for": "Semih"}, \'last 7 days\' → {"days": "7"}). Leave empty for natural language prompts.',
                        },
                        "interval": {
                            "type": "string",
                            "description": "Interval (e.g., '30s', '5m', '1h', 'morning on weekdays', '9:00 on monday,friday')",
                        },
                        "description": {
                            "type": "string",
                            "description": "Optional description of what this task does",
                        },
                        "enabled": {
                            "type": "boolean",
                            "description": "Whether task is enabled (default: true)",
                        },
                        "conditions": {
                            "type": "array",
                            "description": "Optional list of condition dictionaries",
                            "items": {"type": "object"},
                        },
                    },
                    "required": ["name", "prompt", "interval"],
                },
                "_server": "internal",
            },
            {
                "name": "internal__list_schedules",
                "description": "List all schedules (monitors and tasks) with their status",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filter_type": {
                            "type": "string",
                            "enum": ["monitor", "task"],
                            "description": "Optional filter by type (monitor or task)",
                        }
                    },
                },
                "_server": "internal",
            },
            {
                "name": "internal__update_schedule",
                "description": "Update an existing schedule's properties",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Schedule name",
                        },
                        "schedule_type": {
                            "type": "string",
                            "enum": ["monitor", "task"],
                            "description": "Schedule type (monitor or task)",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "New prompt",
                        },
                        "interval": {
                            "type": "string",
                            "description": "New interval",
                        },
                        "description": {
                            "type": "string",
                            "description": "New description",
                        },
                        "enabled": {
                            "type": "boolean",
                            "description": "New enabled status",
                        },
                        "conditions": {
                            "type": "array",
                            "description": "New conditions list",
                            "items": {"type": "object"},
                        },
                        "knowledge_graph": {
                            "type": "object",
                            "description": "New knowledge graph config (monitors only)",
                        },
                    },
                    "required": ["name", "schedule_type"],
                },
                "_server": "internal",
            },
            {
                "name": "internal__delete_schedule",
                "description": "Delete a schedule permanently",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Schedule name",
                        },
                        "schedule_type": {
                            "type": "string",
                            "enum": ["monitor", "task"],
                            "description": "Schedule type (monitor or task)",
                        },
                    },
                    "required": ["name", "schedule_type"],
                },
                "_server": "internal",
            },
            {
                "name": "internal__enable_schedule",
                "description": "Enable or disable a schedule",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Schedule name",
                        },
                        "schedule_type": {
                            "type": "string",
                            "enum": ["monitor", "task"],
                            "description": "Schedule type (monitor or task)",
                        },
                        "enabled": {
                            "type": "boolean",
                            "description": "Whether to enable or disable",
                        },
                    },
                    "required": ["name", "schedule_type", "enabled"],
                },
                "_server": "internal",
            },
            {
                "name": "internal__get_schedule_status",
                "description": "Get detailed status of a schedule including last run information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Schedule name",
                        },
                        "schedule_type": {
                            "type": "string",
                            "enum": ["monitor", "task"],
                            "description": "Schedule type (monitor or task)",
                        },
                    },
                    "required": ["name", "schedule_type"],
                },
                "_server": "internal",
            },
        ]

    async def execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a schedule tool

        Args:
            tool_name: Name of the tool (without internal__ prefix)
            arguments: Tool arguments

        Returns:
            str: Tool result as text
        """
        if tool_name == "create_monitor":
            return self._create_monitor(
                name=arguments["name"],
                prompt=arguments["prompt"],
                interval=arguments["interval"],
                description=arguments.get("description"),
                enabled=arguments.get("enabled", True),
                conditions=arguments.get("conditions", []),
                knowledge_graph=arguments.get("knowledge_graph"),
                prompt_arguments=arguments.get("prompt_arguments"),
            )

        elif tool_name == "create_task":
            return self._create_task(
                name=arguments["name"],
                prompt=arguments["prompt"],
                interval=arguments["interval"],
                description=arguments.get("description"),
                enabled=arguments.get("enabled", True),
                conditions=arguments.get("conditions", []),
                prompt_arguments=arguments.get("prompt_arguments"),
            )

        elif tool_name == "list_schedules":
            return self._list_schedules(filter_type=arguments.get("filter_type"))

        elif tool_name == "update_schedule":
            return self._update_schedule(
                name=arguments["name"],
                schedule_type=arguments["schedule_type"],
                **{k: v for k, v in arguments.items() if k not in ["name", "schedule_type"]},
            )

        elif tool_name == "delete_schedule":
            return self._delete_schedule(name=arguments["name"], schedule_type=arguments["schedule_type"])

        elif tool_name == "enable_schedule":
            return self._enable_schedule(
                name=arguments["name"], schedule_type=arguments["schedule_type"], enabled=arguments["enabled"]
            )

        elif tool_name == "get_schedule_status":
            return self._get_schedule_status(name=arguments["name"], schedule_type=arguments["schedule_type"])

        else:
            raise ValueError(f"Unknown schedule tool: {tool_name}")

    def _create_monitor(
        self,
        name: str,
        prompt: str,
        interval: str,
        description: str | None = None,
        enabled: bool = True,
        conditions: list | None = None,
        knowledge_graph: dict | None = None,
        prompt_arguments: dict | None = None,
    ) -> str:
        """Create a new monitor schedule"""
        if conditions is None:
            conditions = []

        # Validate prompt references (AWL paths, MCP servers)
        prompt_error = self._validate_prompt(prompt)
        if prompt_error:
            return f"Error: {prompt_error}"

        # Validate monitor definition (now just a task with notification support)
        try:
            monitor_def = TaskDefinition(
                name=name,
                prompt=prompt,
                interval=interval,
                description=description,
                enabled=enabled,
                conditions=conditions,
                prompt_arguments=prompt_arguments,
                notify=False,  # Default notify for monitors
                notification_channels=["console"],
            )
            monitor_def.validate()
        except ValueError as e:
            return f"Error: Invalid monitor definition: {e}"

        # Load existing schedules
        schedules = self._load_schedules()

        # Check for duplicate name
        if any(m["name"] == name for m in schedules.get("monitors", [])):
            return f"Error: Monitor with name '{name}' already exists"
        if any(t["name"] == name for t in schedules.get("tasks", [])):
            return f"Error: Task with name '{name}' already exists (names must be unique across all schedules)"

        # Add monitor
        if "monitors" not in schedules:
            schedules["monitors"] = []

        monitor_data = {
            "name": name,
            "prompt": prompt,
            "interval": interval,
            "enabled": enabled,
        }
        if description:
            monitor_data["description"] = description
        if conditions:
            monitor_data["conditions"] = conditions
        if knowledge_graph:
            monitor_data["knowledge_graph"] = knowledge_graph
        if prompt_arguments:
            monitor_data["prompt_arguments"] = prompt_arguments

        schedules["monitors"].append(monitor_data)

        # Save schedules
        self._save_schedules(schedules)

        return f"Monitor '{name}' created successfully (interval: {interval}, enabled: {enabled})"

    def _create_task(
        self,
        name: str,
        prompt: str,
        interval: str,
        description: str | None = None,
        enabled: bool = True,
        conditions: list | None = None,
        prompt_arguments: dict | None = None,
    ) -> str:
        """Create a new task schedule"""
        if conditions is None:
            conditions = []

        # Validate prompt references (AWL paths, MCP servers)
        prompt_error = self._validate_prompt(prompt)
        if prompt_error:
            return f"Error: {prompt_error}"

        # Validate task definition
        try:
            task_def = TaskDefinition(
                name=name,
                prompt=prompt,
                interval=interval,
                description=description,
                enabled=enabled,
                conditions=conditions,
                prompt_arguments=prompt_arguments,
            )
            task_def.validate()
        except ValueError as e:
            return f"Error: Invalid task definition: {e}"

        # Load existing schedules
        schedules = self._load_schedules()

        # Check for duplicate name
        if any(m["name"] == name for m in schedules.get("monitors", [])):
            return f"Error: Monitor with name '{name}' already exists (names must be unique across all schedules)"
        if any(t["name"] == name for t in schedules.get("tasks", [])):
            return f"Error: Task with name '{name}' already exists"

        # Add task
        if "tasks" not in schedules:
            schedules["tasks"] = []

        task_data = {
            "name": name,
            "prompt": prompt,
            "interval": interval,
            "enabled": enabled,
        }
        if description:
            task_data["description"] = description
        if conditions:
            task_data["conditions"] = conditions
        if prompt_arguments:
            task_data["prompt_arguments"] = prompt_arguments

        schedules["tasks"].append(task_data)

        # Save schedules
        self._save_schedules(schedules)

        return f"Task '{name}' created successfully (interval: {interval}, enabled: {enabled})"

    def _list_schedules(self, filter_type: str | None = None) -> str:
        """List all schedules with their status"""
        schedules = self._load_schedules()

        result = []

        if filter_type != "task":
            monitors = schedules.get("monitors", [])
            if monitors:
                result.append("## Monitors\n")
                for monitor in monitors:
                    status = "enabled" if monitor.get("enabled", True) else "disabled"
                    result.append(f"- **{monitor['name']}** ({status})")
                    result.append(f"  - Interval: {monitor['interval']}")
                    if monitor.get("description"):
                        result.append(f"  - Description: {monitor['description']}")
                    result.append(f"  - Prompt: {monitor['prompt']}")
                    if monitor.get("knowledge_graph"):
                        result.append("  - Knowledge Graph: enabled")
                    result.append("")

        if filter_type != "monitor":
            tasks = schedules.get("tasks", [])
            if tasks:
                result.append("## Tasks\n")
                for task in tasks:
                    status = "enabled" if task.get("enabled", True) else "disabled"
                    result.append(f"- **{task['name']}** ({status})")
                    result.append(f"  - Interval: {task['interval']}")
                    if task.get("description"):
                        result.append(f"  - Description: {task['description']}")
                    result.append(f"  - Prompt: {task['prompt']}")
                    result.append("")

        if not result:
            return "No schedules found"

        return "\n".join(result)

    def _update_schedule(self, name: str, schedule_type: str, **updates) -> str:
        """Update an existing schedule"""
        schedules = self._load_schedules()
        collection_name = "monitors" if schedule_type == "monitor" else "tasks"
        collection = schedules.get(collection_name, [])

        # Find schedule
        schedule = None
        for s in collection:
            if s["name"] == name:
                schedule = s
                break

        if not schedule:
            return f"Error: {schedule_type.capitalize()} '{name}' not found"

        # Validate prompt if being updated
        if "prompt" in updates and updates["prompt"] is not None:
            prompt_error = self._validate_prompt(updates["prompt"])
            if prompt_error:
                return f"Error: {prompt_error}"

        # Apply updates
        for key, value in updates.items():
            if value is not None:  # Only update if value provided
                # Merge dicts (like prompt_arguments) instead of replacing
                if key == "prompt_arguments" and isinstance(value, dict) and isinstance(schedule.get(key), dict):
                    schedule[key].update(value)
                else:
                    schedule[key] = value

        # Validate updated schedule
        try:
            # Both monitors and tasks are now TaskDefinition
            task_def = TaskDefinition(
                name=schedule["name"],
                prompt=schedule["prompt"],
                interval=schedule["interval"],
                description=schedule.get("description"),
                enabled=schedule.get("enabled", True),
                conditions=schedule.get("conditions", []),
                prompt_arguments=schedule.get("prompt_arguments"),
                notify=schedule.get("notify", False),
                notification_channels=schedule.get("notification_channels", ["console"]),
            )
            task_def.validate()
        except (ValueError, TypeError) as e:
            return f"Error: Invalid update: {e}"

        # Save schedules
        self._save_schedules(schedules)

        updated_fields = ", ".join(updates.keys())
        return f"{schedule_type.capitalize()} '{name}' updated successfully (updated: {updated_fields})"

    def _delete_schedule(self, name: str, schedule_type: str) -> str:
        """Delete a schedule"""
        schedules = self._load_schedules()
        collection_name = "monitors" if schedule_type == "monitor" else "tasks"
        collection = schedules.get(collection_name, [])

        # Find and remove schedule
        initial_count = len(collection)
        schedules[collection_name] = [s for s in collection if s["name"] != name]

        if len(schedules[collection_name]) == initial_count:
            return f"Error: {schedule_type.capitalize()} '{name}' not found"

        # Save schedules
        self._save_schedules(schedules)

        return f"{schedule_type.capitalize()} '{name}' deleted successfully"

    def _enable_schedule(self, name: str, schedule_type: str, enabled: bool) -> str:
        """Enable or disable a schedule"""
        schedules = self._load_schedules()
        collection_name = "monitors" if schedule_type == "monitor" else "tasks"
        collection = schedules.get(collection_name, [])

        # Find schedule
        schedule = None
        for s in collection:
            if s["name"] == name:
                schedule = s
                break

        if not schedule:
            return f"Error: {schedule_type.capitalize()} '{name}' not found"

        # Update enabled status
        schedule["enabled"] = enabled

        # Save schedules
        self._save_schedules(schedules)

        status = "enabled" if enabled else "disabled"
        return f"{schedule_type.capitalize()} '{name}' {status} successfully"

    def _get_schedule_status(self, name: str, schedule_type: str) -> str:
        """Get detailed status of a schedule"""
        schedules = self._load_schedules()
        collection_name = "monitors" if schedule_type == "monitor" else "tasks"
        collection = schedules.get(collection_name, [])

        # Find schedule
        schedule = None
        for s in collection:
            if s["name"] == name:
                schedule = s
                break

        if not schedule:
            return f"Error: {schedule_type.capitalize()} '{name}' not found"

        # Format status
        result = [
            f"## {schedule_type.capitalize()}: {name}\n",
            f"**Status:** {'enabled' if schedule.get('enabled', True) else 'disabled'}",
            f"**Interval:** {schedule['interval']}",
            f"**Prompt:** {schedule['prompt']}",
        ]

        if schedule.get("description"):
            result.append(f"**Description:** {schedule['description']}")

        if schedule.get("conditions"):
            result.append(f"**Conditions:** {len(schedule['conditions'])} condition(s)")

        if schedule_type == "monitor" and schedule.get("knowledge_graph"):
            result.append("**Knowledge Graph:** enabled")

        # TODO: Add last run time from state manager
        result.append("\n*Note: Last run information will be available when monitoring mode is running*")

        return "\n".join(result)

    def _validate_prompt(self, prompt: str) -> str | None:
        """Validate the prompt field and return an error message if invalid, None if valid.

        Checks:
        - AWL file paths (.awl): file must exist on disk
        - MCP prompts (mcp://server/prompt): server must be in known_mcp_servers if provided
        """
        if prompt.endswith(".awl"):
            awl_path = Path(prompt).expanduser()
            if not awl_path.is_absolute():
                awl_path = get_config_dir() / prompt
            if not awl_path.exists():
                return f"AWL script does not exist: {awl_path}"

        if prompt.startswith("mcp://") and self.known_mcp_servers is not None:
            # Extract server name from mcp://server/prompt
            ref = prompt[6:]
            if "/" in ref:
                server_name = ref.split("/", 1)[0]
                if server_name not in self.known_mcp_servers:
                    available = ", ".join(sorted(self.known_mcp_servers))
                    return f"MCP server '{server_name}' is not configured. Available servers: {available}"

        return None

    def _load_schedules(self) -> dict:
        """Load schedules from JSON file"""
        if not self.schedules_file.exists():
            return {"version": "1.0", "monitors": [], "tasks": []}

        try:
            with open(self.schedules_file) as f:
                data = json.load(f)

            # Ensure required keys exist
            if "version" not in data:
                data["version"] = "1.0"
            if "monitors" not in data:
                data["monitors"] = []
            if "tasks" not in data:
                data["tasks"] = []

            return data

        except json.JSONDecodeError:
            # Backup corrupted file
            backup_file = self.schedules_file.with_suffix(".json.backup")
            if self.schedules_file.exists():
                self.schedules_file.rename(backup_file)
                print(f"Warning: Corrupted schedules file backed up to {backup_file}")

            # Return fresh schedules
            return {"version": "1.0", "monitors": [], "tasks": []}

    def _save_schedules(self, schedules: dict):
        """Save schedules to JSON file"""
        # Ensure version is set
        if "version" not in schedules:
            schedules["version"] = "1.0"

        # Write atomically
        temp_file = self.schedules_file.with_suffix(".json.tmp")
        with open(temp_file, "w") as f:
            json.dump(schedules, f, indent=2)

        # Atomic rename
        temp_file.rename(self.schedules_file)
