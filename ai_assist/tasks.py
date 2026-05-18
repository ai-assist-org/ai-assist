"""User-defined task definitions and YAML loader"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TaskDefinition:
    """Definition of a user-defined periodic task or event-triggered task"""

    name: str
    prompt: str
    interval: str | None = None  # e.g., "5m", "1h", "morning on weekdays"
    trigger: dict[str, Any] | None = None  # e.g., {"type": "mqtt", "topic": "alerts/#"}
    description: str | None = None
    enabled: bool = True
    conditions: list[dict] = field(default_factory=list)
    prompt_arguments: dict[str, Any] | None = None
    max_turns: int = 100  # Maximum agentic turns (safety limit, loop detection usually triggers first)

    # Notification configuration
    notify: bool = False
    notification_channels: list[str] = field(default_factory=lambda: ["console"])

    @property
    def is_event_triggered(self) -> bool:
        return self.trigger is not None

    @property
    def interval_seconds(self) -> int:
        """Convert interval string to seconds (for simple intervals)"""
        if self.interval is None:
            raise ValueError("Event-triggered tasks have no interval")
        return TaskLoader.parse_interval(self.interval)

    @property
    def is_interval_with_range(self) -> bool:
        """Check if this is an interval-with-range schedule (e.g. '1h between 9:00 and 23:00')"""
        if self.interval is None:
            return False
        lower = self.interval.lower()
        return " between " in lower and " and " in lower

    @property
    def is_time_based(self) -> bool:
        """Check if this is a time-based schedule"""
        if self.interval is None:
            return False
        return " on " in self.interval.lower() and not self.is_interval_with_range

    @property
    def is_mcp_prompt(self) -> bool:
        from .mcp_prompt import is_mcp_prompt

        return is_mcp_prompt(self.prompt)

    def parse_mcp_prompt(self) -> tuple[str, str]:
        from .mcp_prompt import parse_mcp_prompt

        return parse_mcp_prompt(self.prompt)

    @classmethod
    def from_dict(cls, task_data: dict[str, Any]) -> TaskDefinition:
        """Create a TaskDefinition from a dictionary, using defaults for missing optional fields."""
        return cls(
            name=task_data["name"],
            prompt=task_data["prompt"],
            interval=task_data.get("interval"),
            trigger=task_data.get("trigger"),
            description=task_data.get("description"),
            enabled=task_data.get("enabled", True),
            conditions=task_data.get("conditions", []),
            prompt_arguments=task_data.get("prompt_arguments"),
            notify=task_data.get("notify", False),
            notification_channels=task_data.get("notification_channels", ["console"]),
        )

    def validate(self):
        """Validate task definition"""
        if not self.name:
            raise ValueError("Task name is required")
        if not self.prompt:
            raise ValueError("Task prompt is required")

        if self.interval is None and self.trigger is None:
            raise ValueError("Task must have either 'interval' or 'trigger'")
        if self.interval is not None and self.trigger is not None:
            raise ValueError("Task cannot have both 'interval' and 'trigger'")

        if self.trigger is not None:
            if "type" not in self.trigger:
                raise ValueError("Trigger must have a 'type' field")
            return

        # Validate MCP prompt format if applicable
        if self.is_mcp_prompt:
            try:
                self.parse_mcp_prompt()
            except ValueError as e:
                raise ValueError(f"Invalid MCP prompt reference: {e}") from e

        # Validate interval format (trigger tasks return early above, so interval is set here)
        assert self.interval is not None
        try:
            if self.is_interval_with_range:
                TaskLoader.parse_interval_with_range(self.interval)
            elif self.is_time_based:
                TaskLoader.parse_time_schedule(self.interval)
            else:
                _ = self.interval_seconds
        except ValueError as e:
            raise ValueError(f"Invalid interval '{self.interval}': {e}") from e


class TaskLoader:
    """Load and parse task definitions from YAML"""

    # Time presets
    TIME_PRESETS = {
        "morning": "9:00",
        "afternoon": "14:00",
        "evening": "18:00",
        "night": "22:00",
    }

    # Day groups
    DAY_GROUPS = {
        "weekdays": [0, 1, 2, 3, 4],  # Monday-Friday
        "weekends": [5, 6],  # Saturday-Sunday
    }

    @staticmethod
    def parse_time_schedule(schedule_str: str) -> dict:
        """Parse time-based schedule string

        Formats:
        - "morning on weekdays" -> 9:00 AM Monday-Friday
        - "9:00 on weekdays" -> 9:00 Monday-Friday
        - "14:30 on monday,wednesday,friday" -> 2:30 PM on specific days

        Returns:
            dict with 'time' (datetime.time) and 'days' (list of weekday numbers)
        """
        schedule_str = schedule_str.strip().lower()

        if " on " not in schedule_str:
            raise ValueError(
                f"Time-based schedule must include 'on': '{schedule_str}'. "
                "Examples: 'morning on weekdays', '9:00 on monday,friday'"
            )

        time_part, days_part = schedule_str.split(" on ", 1)
        time_part = time_part.strip()
        days_part = days_part.strip()

        # Parse time
        if time_part in TaskLoader.TIME_PRESETS:
            time_str = TaskLoader.TIME_PRESETS[time_part]
        else:
            time_str = time_part

        # Parse time string (HH:MM format)
        try:
            hour, minute = map(int, time_str.split(":"))
            if not (0 <= hour < 24 and 0 <= minute < 60):
                raise ValueError("Time out of range")
            schedule_time = dt_time(hour, minute)
        except (ValueError, AttributeError) as e:
            raise ValueError(
                f"Invalid time format: '{time_part}'. "
                "Use HH:MM (24-hour) or presets: morning, afternoon, evening, night"
            ) from e

        # Parse days
        if days_part in TaskLoader.DAY_GROUPS:
            days = TaskLoader.DAY_GROUPS[days_part]
        else:
            # Parse individual days
            day_names = {
                "monday": 0,
                "mon": 0,
                "tuesday": 1,
                "tue": 1,
                "wednesday": 2,
                "wed": 2,
                "thursday": 3,
                "thu": 3,
                "friday": 4,
                "fri": 4,
                "saturday": 5,
                "sat": 5,
                "sunday": 6,
                "sun": 6,
            }

            day_parts = [d.strip() for d in days_part.split(",")]
            days = []
            for day in day_parts:
                if day in day_names:
                    days.append(day_names[day])
                else:
                    raise ValueError(
                        f"Invalid day: '{day}'. " "Use day names (monday, tuesday, etc.) or groups (weekdays, weekends)"
                    )

            if not days:
                raise ValueError("At least one day must be specified")

        return {"time": schedule_time, "days": sorted(set(days))}

    @staticmethod
    def calculate_next_run(schedule: dict, from_time: datetime | None = None) -> datetime:
        """Calculate next run time for a time-based schedule

        Args:
            schedule: Dict with 'time' and 'days'
            from_time: Calculate from this time (default: now)

        Returns:
            Next datetime when task should run
        """
        if from_time is None:
            from_time = datetime.now()

        schedule_time = schedule["time"]
        allowed_days = set(schedule["days"])

        # Start from the same day at the scheduled time
        next_run = datetime.combine(from_time.date(), schedule_time)

        # If time has passed today, start from tomorrow
        if next_run <= from_time:
            next_run += timedelta(days=1)

        # Find next allowed day
        max_attempts = 7  # Don't loop forever
        attempts = 0
        while next_run.weekday() not in allowed_days and attempts < max_attempts:
            next_run += timedelta(days=1)
            attempts += 1

        if attempts >= max_attempts:
            raise ValueError(f"Could not find next run time for schedule: {schedule}")

        return next_run

    @staticmethod
    def parse_interval_with_range(schedule_str: str) -> dict:
        """Parse interval-with-range schedule string

        Formats:
        - "1h between 9:00 and 23:00"
        - "30m between 8:00 and 18:00 on weekdays"

        Returns:
            dict with 'interval_seconds' (int), 'start' (time), 'end' (time), 'days' (list|None)
        """
        schedule_str = schedule_str.strip().lower()

        pattern = r"^(.+?)\s+between\s+(\d{1,2}:\d{2})\s+and\s+(\d{1,2}:\d{2})(?:\s+on\s+(.+))?$"
        match = re.match(pattern, schedule_str)
        if not match:
            raise ValueError(
                f"Invalid interval-with-range format: '{schedule_str}'. "
                "Use format like '1h between 9:00 and 23:00' or '30m between 8:00 and 18:00 on weekdays'"
            )

        duration_str = match.group(1).strip()
        start_str = match.group(2)
        end_str = match.group(3)
        days_str = match.group(4)

        # Parse duration using existing parse_interval
        interval_seconds = TaskLoader.parse_interval(duration_str)

        # Parse start time
        try:
            sh, sm = map(int, start_str.split(":"))
            if not (0 <= sh < 24 and 0 <= sm < 60):
                raise ValueError("Start time out of range")
            start_time = dt_time(sh, sm)
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid start time: '{start_str}'") from e

        # Parse end time
        try:
            eh, em = map(int, end_str.split(":"))
            if not (0 <= eh < 24 and 0 <= em < 60):
                raise ValueError("End time out of range")
            end_time = dt_time(eh, em)
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid end time: '{end_str}'") from e

        # Parse optional days
        days = None
        if days_str:
            days_str = days_str.strip()
            if days_str in TaskLoader.DAY_GROUPS:
                days = TaskLoader.DAY_GROUPS[days_str]
            else:
                day_names = {
                    "monday": 0,
                    "mon": 0,
                    "tuesday": 1,
                    "tue": 1,
                    "wednesday": 2,
                    "wed": 2,
                    "thursday": 3,
                    "thu": 3,
                    "friday": 4,
                    "fri": 4,
                    "saturday": 5,
                    "sat": 5,
                    "sunday": 6,
                    "sun": 6,
                }
                day_parts = [d.strip() for d in days_str.split(",")]
                days = []
                for day in day_parts:
                    if day in day_names:
                        days.append(day_names[day])
                    else:
                        raise ValueError(f"Invalid day: '{day}'")
                days = sorted(set(days))

        return {
            "interval_seconds": interval_seconds,
            "start": start_time,
            "end": end_time,
            "days": days,
        }

    @staticmethod
    def calculate_next_interval_run(schedule: dict, from_time: datetime | None = None) -> datetime:
        """Calculate next run time for an interval-with-range schedule

        Args:
            schedule: Dict with 'interval_seconds', 'start', 'end', 'days'
            from_time: Calculate from this time (default: now)

        Returns:
            Next datetime when task should run
        """
        if from_time is None:
            from_time = datetime.now()

        start = schedule["start"]
        end = schedule["end"]
        interval = schedule["interval_seconds"]
        allowed_days = set(schedule["days"]) if schedule["days"] is not None else None

        current_time = from_time.time()
        today = from_time.date()

        def _is_allowed_day(d: datetime) -> bool:
            if allowed_days is None:
                return True
            return d.weekday() in allowed_days

        def _next_allowed_day_at_start(base_date) -> datetime:
            """Find next allowed day starting from base_date, at the start time."""
            candidate = datetime.combine(base_date, start)
            max_attempts = 7
            for _ in range(max_attempts):
                if _is_allowed_day(candidate):
                    return candidate
                candidate += timedelta(days=1)
            raise ValueError(f"Could not find next allowed day for schedule: {schedule}")

        # Check if we're within the active range today
        if current_time < start:
            # Before range: next run is today at start (if allowed day)
            candidate = datetime.combine(today, start)
            if _is_allowed_day(candidate):
                return candidate
            return _next_allowed_day_at_start(today + timedelta(days=1))

        if current_time >= end:
            # After range: next run is tomorrow at start (next allowed day)
            return _next_allowed_day_at_start(today + timedelta(days=1))

        # Within range: next run is from_time + interval
        next_run = from_time + timedelta(seconds=interval)
        if next_run.time() >= end:
            # Would exceed range: next run is tomorrow at start
            return _next_allowed_day_at_start(today + timedelta(days=1))

        # Still within range and on an allowed day
        if _is_allowed_day(next_run):
            return next_run

        return _next_allowed_day_at_start(today + timedelta(days=1))

    @staticmethod
    def parse_interval(interval_str: str) -> int:
        """Convert interval string to seconds

        Supported formats:
        - "30s" -> 30 seconds
        - "5m" -> 300 seconds
        - "1h" -> 3600 seconds
        - "2h30m" -> 9000 seconds
        """
        if not interval_str:
            raise ValueError("Interval cannot be empty")

        interval_str = interval_str.strip().lower()
        total_seconds = 0

        # Parse hours
        hours_match = re.search(r"(\d+)h", interval_str)
        if hours_match:
            total_seconds += int(hours_match.group(1)) * 3600

        # Parse minutes
        minutes_match = re.search(r"(\d+)m", interval_str)
        if minutes_match:
            total_seconds += int(minutes_match.group(1)) * 60

        # Parse seconds
        seconds_match = re.search(r"(\d+)s", interval_str)
        if seconds_match:
            total_seconds += int(seconds_match.group(1))

        if total_seconds == 0:
            raise ValueError(
                f"Invalid interval format: '{interval_str}'. " "Use formats like '30s', '5m', '1h', or '2h30m'"
            )

        return total_seconds

    def load_from_yaml(self, path: Path) -> list[TaskDefinition]:
        """Load task definitions from YAML file"""
        if not path.exists():
            return []

        try:
            with open(path) as f:
                data = yaml.safe_load(f)

            if not data or "tasks" not in data:
                return []

            tasks = []
            for task_data in data["tasks"]:
                task = TaskDefinition.from_dict(task_data)
                task.validate()
                tasks.append(task)

            return tasks

        except yaml.YAMLError as e:
            raise ValueError(f"YAML parsing error: {e}") from e
        except KeyError as e:
            raise ValueError(f"Missing required field in task definition: {e}") from e

    def load_from_yaml_string(self, yaml_content: str) -> list[TaskDefinition]:
        """Load task definitions from YAML string (for testing)"""
        try:
            data = yaml.safe_load(yaml_content)

            if not data or "tasks" not in data:
                return []

            tasks = []
            for task_data in data["tasks"]:
                task = TaskDefinition.from_dict(task_data)
                task.validate()
                tasks.append(task)

            return tasks

        except yaml.YAMLError as e:
            raise ValueError(f"YAML parsing error: {e}") from e
        except KeyError as e:
            raise ValueError(f"Missing required field in task definition: {e}") from e
