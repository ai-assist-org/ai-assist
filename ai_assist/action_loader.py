"""Load and save unified actions from event-schedules.json"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from .action_model import ActionDefinition

logger = logging.getLogger(__name__)


class ActionLoader:
    """Load and persist actions from event-schedules.json"""

    DEFAULT_ACTIONS = [
        {
            "name": "kg-synthesis",
            "trigger": {"type": "schedule", "at": "night", "days": "weekdays"},
            "prompt": "__builtin__:kg_synthesis",
            "enabled": True,
        },
    ]

    def __init__(self, json_file: Path) -> None:
        self.json_file = Path(json_file)

    def load_actions(self) -> list[ActionDefinition]:
        data = self._load_json()
        actions = []
        for action_data in data.get("actions", []):
            try:
                action = ActionDefinition.from_dict(action_data)
                action.validate_definition()
                actions.append(action)
            except (KeyError, ValueError) as e:
                logger.warning("Skipping invalid action '%s': %s", action_data.get("name", "unknown"), e)
        return actions

    def load_event_source_configs(self) -> dict[str, Any]:
        data = self._load_json()
        return data.get("event_sources", {})

    def save_actions(self, actions: list[ActionDefinition]) -> None:
        data = self._load_json()
        data["actions"] = [a.to_dict() for a in actions]
        self._save_json(data)

    def ensure_defaults(self) -> None:
        data = self._load_json()
        existing_prompts = {a.get("prompt") for a in data.get("actions", [])}

        added = []
        for default in self.DEFAULT_ACTIONS:
            if default["prompt"] not in existing_prompts:
                data.setdefault("actions", []).append(default)
                added.append(default["name"])

        if added:
            self._save_json(data)
            for name in added:
                print(f"Added default action: {name}")

    def migrate_from_old_format(self, old_schedules_file: Path) -> None:
        if not old_schedules_file.exists():
            return

        try:
            with open(old_schedules_file) as f:
                old_data = json.load(f)
        except json.JSONDecodeError, OSError:
            logger.exception("Failed to read old schedules file for migration")
            return

        data = self._load_json()
        existing_names = {a.get("name") for a in data.get("actions", [])}

        new_actions: list[dict[str, Any]] = []
        for entry in old_data.get("monitors", []) + old_data.get("tasks", []):
            if entry.get("name") in existing_names:
                continue
            action = self._convert_old_task(entry)
            if action:
                new_actions.append(action)
                existing_names.add(action["name"])

        if not new_actions:
            return

        event_sources = old_data.get("event_sources", {})
        if event_sources and "event_sources" not in data:
            data["event_sources"] = event_sources

        data["actions"].extend(new_actions)
        self._save_json(data)
        print(f"Migrated {len(new_actions)} action(s) from {old_schedules_file}")

    def migrate_scheduled_actions(self, old_actions_file: Path) -> None:
        if not old_actions_file.exists():
            return

        try:
            with open(old_actions_file) as f:
                old_data = json.load(f)
        except json.JSONDecodeError, OSError:
            logger.exception("Failed to read old scheduled actions file for migration")
            return

        data = self._load_json()
        existing_prompts = {
            (a.get("prompt"), a.get("trigger", {}).get("at"))
            for a in data.get("actions", [])
            if a.get("trigger", {}).get("type") == "once"
        }
        count = 0

        for old_action in old_data.get("actions", []):
            if old_action.get("status") in ("completed", "failed"):
                continue
            key = (old_action["prompt"], old_action["scheduled_at"])
            if key in existing_prompts:
                continue
            action = {
                "name": old_action.get("description") or old_action.get("prompt", "")[:50],
                "trigger": {"type": "once", "at": old_action["scheduled_at"]},
                "prompt": old_action["prompt"],
                "enabled": True,
                "notify": old_action.get("notify", True),
                "created_by": old_action.get("created_by"),
                "created_at": old_action.get("created_at"),
                "status": old_action.get("status", "pending"),
            }
            data.setdefault("actions", []).append(action)
            count += 1

        if count > 0:
            self._save_json(data)
            print(f"Migrated {count} scheduled action(s) from {old_actions_file}")

    @staticmethod
    def _convert_old_task(task_data: dict[str, Any]) -> dict[str, Any] | None:
        interval = task_data.get("interval", "")
        if not interval:
            return None

        trigger = _parse_old_interval_to_trigger(interval)
        if trigger is None:
            return None

        action: dict[str, Any] = {
            "name": task_data["name"],
            "trigger": trigger,
            "prompt": task_data["prompt"],
            "enabled": task_data.get("enabled", True),
        }

        if task_data.get("conditions"):
            action["conditions"] = task_data["conditions"]
        if task_data.get("prompt_arguments"):
            action["prompt_arguments"] = task_data["prompt_arguments"]
        if task_data.get("notify"):
            action["notify"] = True
        if task_data.get("notification_channels", ["console"]) != ["console"]:
            action["notification_channels"] = task_data["notification_channels"]
        if task_data.get("description"):
            action["description"] = task_data["description"]

        return action

    def _load_json(self) -> dict[str, Any]:
        if not self.json_file.exists():
            return {"version": "2.0", "actions": []}
        try:
            with open(self.json_file) as f:
                data = json.load(f)
            if "actions" not in data:
                data["actions"] = []
            return data
        except json.JSONDecodeError:
            logger.error("Failed to parse %s; returning empty actions", self.json_file)
            return {"version": "2.0", "actions": []}

    def _save_json(self, data: dict[str, Any]) -> None:
        self.json_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.json_file.with_suffix(".json.tmp")
        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2)
        temp_file.rename(self.json_file)


def _parse_old_interval_to_trigger(interval: str) -> dict[str, Any] | None:
    interval = interval.strip()

    range_pattern = r"^(.+?)\s+between\s+(\d{1,2}:\d{2})\s+and\s+(\d{1,2}:\d{2})(?:\s+on\s+(.+))?$"
    range_match = re.match(range_pattern, interval, re.IGNORECASE)
    if range_match:
        trigger: dict[str, Any] = {
            "type": "interval_range",
            "every": range_match.group(1).strip(),
            "between": range_match.group(2),
            "and": range_match.group(3),
        }
        if range_match.group(4):
            trigger["days"] = range_match.group(4).strip()
        return trigger

    if " on " in interval.lower():
        parts = interval.split(" on ", 1)
        return {
            "type": "schedule",
            "at": parts[0].strip(),
            "days": parts[1].strip(),
        }

    return {"type": "interval", "every": interval}
