"""Unified action model for event-driven scheduling"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .event_sources import EventContext
from .tasks import TaskLoader

TRIGGER_TYPES_TIME = {"interval", "schedule", "interval_range", "once"}
TRIGGER_TYPES_EVENT = {"mqtt", "dbus", "file"}
TRIGGER_TYPES_ALL = TRIGGER_TYPES_TIME | TRIGGER_TYPES_EVENT


class ActionDefinition(BaseModel):
    """Unified action definition — replaces TaskDefinition and ScheduledAction"""

    model_config = ConfigDict(ser_json_timedelta="iso8601")

    name: str
    trigger: dict[str, Any]
    prompt: str
    enabled: bool = True
    prompt_arguments: dict[str, Any] | None = None
    max_turns: int = 100
    conditions: list[dict[str, Any]] = Field(default_factory=list)
    notify: bool = False
    notification_channels: list[str] = Field(default_factory=lambda: ["console"])
    created_by: str | None = None
    created_at: datetime | None = None
    status: str | None = None
    executed_at: datetime | None = None

    @property
    def trigger_type(self) -> str:
        return self.trigger.get("type", "")

    @property
    def is_time_based(self) -> bool:
        return self.trigger_type in TRIGGER_TYPES_TIME

    @property
    def is_event_based(self) -> bool:
        return self.trigger_type in TRIGGER_TYPES_EVENT

    @property
    def is_mcp_prompt(self) -> bool:
        from .mcp_prompt import is_mcp_prompt

        return is_mcp_prompt(self.prompt)

    def parse_mcp_prompt(self) -> tuple[str, str]:
        from .mcp_prompt import parse_mcp_prompt

        return parse_mcp_prompt(self.prompt)

    def validate_definition(self) -> None:
        if not self.name:
            raise ValueError("Action name is required")
        if not self.prompt:
            raise ValueError("Action prompt is required")
        if "type" not in self.trigger:
            raise ValueError("Trigger must have a 'type' field")

        trigger_type = self.trigger_type
        if trigger_type not in TRIGGER_TYPES_ALL:
            raise ValueError(f"Unknown trigger type '{trigger_type}'")

        if trigger_type == "interval":
            if "every" not in self.trigger:
                raise ValueError("Interval trigger requires 'every' field")
            TaskLoader.parse_interval(self.trigger["every"])

        elif trigger_type == "schedule":
            if "at" not in self.trigger:
                raise ValueError("Schedule trigger requires 'at' field")
            if "days" not in self.trigger:
                raise ValueError("Schedule trigger requires 'days' field")
            schedule_str = f"{self.trigger['at']} on {self.trigger['days']}"
            TaskLoader.parse_time_schedule(schedule_str)

        elif trigger_type == "interval_range":
            if "every" not in self.trigger:
                raise ValueError("Interval range trigger requires 'every' field")
            if "between" not in self.trigger:
                raise ValueError("Interval range trigger requires 'between' field")
            if "and" not in self.trigger:
                raise ValueError("Interval range trigger requires 'and' field")
            range_str = f"{self.trigger['every']} between {self.trigger['between']} and {self.trigger['and']}"
            if "days" in self.trigger:
                range_str += f" on {self.trigger['days']}"
            TaskLoader.parse_interval_with_range(range_str)

        elif trigger_type == "once":
            if "at" not in self.trigger:
                raise ValueError("Once trigger requires 'at' field")

        elif trigger_type == "mqtt":
            if "topic" not in self.trigger:
                raise ValueError("MQTT trigger requires 'topic' field")

        elif trigger_type == "dbus":
            if "interface" not in self.trigger:
                raise ValueError("D-Bus trigger requires 'interface' field")
            if "signal" not in self.trigger:
                raise ValueError("D-Bus trigger requires 'signal' field")

        elif trigger_type == "file":
            if "path" not in self.trigger:
                raise ValueError("File trigger requires 'path' field")

        if self.is_mcp_prompt:
            try:
                self.parse_mcp_prompt()
            except ValueError as e:
                raise ValueError(f"Invalid MCP prompt reference: {e}") from e

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionDefinition:
        return cls(
            name=data["name"],
            trigger=data["trigger"],
            prompt=data["prompt"],
            enabled=data.get("enabled", True),
            prompt_arguments=data.get("prompt_arguments"),
            max_turns=data.get("max_turns", 100),
            conditions=data.get("conditions", []),
            notify=data.get("notify", False),
            notification_channels=data.get("notification_channels", ["console"]),
            created_by=data.get("created_by"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            status=data.get("status"),
            executed_at=datetime.fromisoformat(data["executed_at"]) if data.get("executed_at") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "trigger": self.trigger,
            "prompt": self.prompt,
            "enabled": self.enabled,
        }
        if self.prompt_arguments is not None:
            d["prompt_arguments"] = self.prompt_arguments
        if self.max_turns != 100:
            d["max_turns"] = self.max_turns
        if self.conditions:
            d["conditions"] = self.conditions
        if self.notify:
            d["notify"] = self.notify
        if self.notification_channels != ["console"]:
            d["notification_channels"] = self.notification_channels
        if self.created_by is not None:
            d["created_by"] = self.created_by
        if self.created_at is not None:
            d["created_at"] = self.created_at.isoformat()
        if self.status is not None:
            d["status"] = self.status
        if self.executed_at is not None:
            d["executed_at"] = self.executed_at.isoformat()
        return d


class TriggerMatcher:
    """Matches incoming events against event-based trigger configurations"""

    def matches(self, event: EventContext, trigger: dict[str, Any]) -> bool:
        trigger_type = trigger.get("type", "")

        if trigger_type not in TRIGGER_TYPES_EVENT:
            return False

        if trigger_type != event.source_type:
            return False

        if trigger_type == "mqtt":
            return self._match_mqtt(event, trigger)
        if trigger_type == "dbus":
            return self._match_dbus(event, trigger)
        if trigger_type == "file":
            return self._match_file(event, trigger)

        return False

    def _match_mqtt(self, event: EventContext, trigger: dict[str, Any]) -> bool:
        event_topic = event.metadata.get("topic", "")
        trigger_topic = trigger.get("topic", "")

        if not self._mqtt_topic_matches(event_topic, trigger_topic):
            return False

        return self._match_payload_filters(event, trigger)

    def _match_dbus(self, event: EventContext, trigger: dict[str, Any]) -> bool:
        if event.metadata.get("interface") != trigger.get("interface"):
            return False
        if event.metadata.get("signal") != trigger.get("signal"):
            return False
        if "path" in trigger and event.metadata.get("path") != trigger["path"]:
            return False

        return self._match_payload_filters(event, trigger)

    def _match_file(self, event: EventContext, trigger: dict[str, Any]) -> bool:
        from .event_source_file import _path_matches_pattern

        event_path = event.metadata.get("path", "")
        trigger_path = str(Path(trigger.get("path", "")).expanduser())
        if not _path_matches_pattern(event_path, trigger_path):
            return False
        return self._match_payload_filters(event, trigger)

    @staticmethod
    def _match_payload_filters(event: EventContext, trigger: dict[str, Any]) -> bool:
        if "payload_contains" in trigger:
            if trigger["payload_contains"] not in event.payload:
                return False
        if "payload_regex" in trigger:
            if not re.search(trigger["payload_regex"], event.payload):
                return False
        return True

    @staticmethod
    def _mqtt_topic_matches(event_topic: str, pattern: str) -> bool:
        if pattern == event_topic:
            return True
        if "#" not in pattern and "+" not in pattern:
            return False

        pattern_parts = pattern.split("/")
        topic_parts = event_topic.split("/")

        for i, p in enumerate(pattern_parts):
            if p == "#":
                return True
            if i >= len(topic_parts):
                return False
            if p == "+":
                continue
            if p != topic_parts[i]:
                return False

        return len(pattern_parts) == len(topic_parts)
