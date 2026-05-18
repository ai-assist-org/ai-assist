"""Load schedules from JSON file"""

import json
import logging
from pathlib import Path

from .tasks import TaskDefinition

logger = logging.getLogger(__name__)


class ScheduleLoader:
    """Load monitor and task schedules from JSON file"""

    def __init__(self, json_file: Path):
        """Initialize loader

        Args:
            json_file: Path to schedules.json file
        """
        self.json_file = Path(json_file)

    def load_monitors(self) -> list[TaskDefinition]:
        """Load monitor definitions from JSON file as regular tasks

        Returns:
            List of TaskDefinition objects
        """
        data = self._load_json()
        tasks = []

        for monitor_data in data.get("monitors", []):
            try:
                task = TaskDefinition(
                    name=monitor_data["name"],
                    prompt=monitor_data["prompt"],
                    interval=monitor_data.get("interval"),
                    trigger=monitor_data.get("trigger"),
                    description=monitor_data.get("description"),
                    enabled=monitor_data.get("enabled", True),
                    conditions=monitor_data.get("conditions", []),
                    prompt_arguments=monitor_data.get("prompt_arguments"),
                    notify=monitor_data.get("notify", False),
                    notification_channels=monitor_data.get("notification_channels", ["console"]),
                )
                task.validate()
                tasks.append(task)
            except (KeyError, ValueError) as e:
                logger.warning("Skipping invalid monitor '%s': %s", monitor_data.get("name", "unknown"), e)

        return tasks

    def load_event_source_configs(self) -> dict:
        """Load event source configurations from JSON file

        Returns:
            Dict of source_type -> config, e.g. {"mqtt": {"broker": "localhost"}}
        """
        data = self._load_json()
        return data.get("event_sources", {})

    def load_tasks(self) -> list[TaskDefinition]:
        """Load task definitions from JSON file

        Returns:
            List of TaskDefinition objects
        """
        data = self._load_json()
        tasks = []

        for task_data in data.get("tasks", []):
            try:
                task = TaskDefinition.from_dict(task_data)
                task.validate()
                tasks.append(task)
            except (KeyError, ValueError) as e:
                logger.warning("Skipping invalid task '%s': %s", task_data.get("name", "unknown"), e)

        return tasks

    # Default tasks that are ensured to exist in schedules.json
    DEFAULT_TASKS = [
        {
            "name": "kg-synthesis",
            "prompt": "__builtin__:kg_synthesis",
            "interval": "night on weekdays",
            "description": "Synthesize knowledge and discover connections between entities",
            "enabled": True,
        },
    ]

    def ensure_default_tasks(self):
        """Ensure default tasks exist in schedules.json

        Adds any missing default tasks to the file so the agent can see and edit them.
        Matches by prompt (functional identifier), not name (user-editable label).
        """
        data = self._load_json()
        existing_prompts = {t.get("prompt") for t in data.get("tasks", [])}

        added = []
        for default_task in self.DEFAULT_TASKS:
            if default_task["prompt"] not in existing_prompts:
                data["tasks"].append(default_task)
                added.append(default_task["name"])

        if added:
            self._save_json(data)
            for name in added:
                print(f"Added default task to schedules: {name}")

    def _save_json(self, data: dict):
        """Save data to JSON file"""
        self.json_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.json_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_json(self) -> dict:
        """Load and parse JSON file

        Returns:
            Parsed JSON data or empty structure if file doesn't exist
        """
        if not self.json_file.exists():
            return {"version": "1.0", "monitors": [], "tasks": []}

        try:
            with open(self.json_file) as f:
                data = json.load(f)

            # Ensure required keys exist
            if "monitors" not in data:
                data["monitors"] = []
            if "tasks" not in data:
                data["tasks"] = []

            return data

        except json.JSONDecodeError as e:
            logger.error("Failed to parse %s: %s; returning empty schedules", self.json_file, e)
            return {"version": "1.0", "monitors": [], "tasks": []}
