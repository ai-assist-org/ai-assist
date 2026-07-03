"""Internal tools for scheduling one-shot future actions"""

import re
from datetime import datetime, timedelta

from ai_assist.config import get_config_dir
from ai_assist.scheduled_actions import ScheduledAction, ScheduledActionManager


class ScheduleActionTools:
    """Internal tools for scheduling one-shot future actions"""

    def __init__(self, agent):
        self.agent = agent
        self.action_file = get_config_dir() / "scheduled-actions.json"
        self.manager = ScheduledActionManager(self.action_file, agent)

    def get_tool_definitions(self) -> list[dict]:
        """Return tool definitions for agent"""
        return [
            {
                "name": "internal__schedule_action",
                "_readonly": False,
                "description": (
                    "Schedule a one-time action to execute at a future time. "
                    "You decide whether the action needs to execute a query or is just a simple reminder. "
                    "Actions execute in the background /monitor process."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The action/query to execute (e.g., 'Check unread Gmail emails' or 'Time to watch TV')",
                        },
                        "time_spec": {
                            "type": "string",
                            "description": (
                                "When to execute the action. Examples: "
                                "'in 2 hours', 'in 30 minutes', 'at 5pm', '5pm', 'at 17:00', "
                                "'today at 5pm', 'tomorrow at 9am', 'next monday 10:00', 'in 1 day'"
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": "Optional human-readable description of why this is scheduled",
                        },
                        "execute_query": {
                            "type": "boolean",
                            "description": (
                                "Whether to execute the prompt via the agent to get results. "
                                "Set to true for queries that need data (e.g., 'check Gmail', 'search DCI jobs'). "
                                "Set to false for simple reminders (e.g., 'time to watch TV'). "
                                "Default: false (simple reminder)."
                            ),
                        },
                        "notify": {
                            "type": "boolean",
                            "description": ("Whether to send a notification when complete. " "Default: true."),
                        },
                        "create_report": {
                            "type": "boolean",
                            "description": ("Whether to save results to a report file. " "Default: false."),
                        },
                    },
                    "required": ["prompt", "time_spec"],
                },
            }
        ]

    async def execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute schedule action tool"""
        if tool_name == "internal__schedule_action":
            return await self.schedule_action(
                prompt=arguments["prompt"],
                time_spec=arguments["time_spec"],
                description=arguments.get("description"),
                execute_query=arguments.get("execute_query", False),
                notify=arguments.get("notify", True),
                create_report=arguments.get("create_report", False),
            )

        return f"Unknown tool: {tool_name}"

    async def schedule_action(
        self,
        prompt: str,
        time_spec: str,
        description: str | None = None,
        execute_query: bool = False,
        notify: bool = True,
        create_report: bool = False,
    ) -> str:
        """Schedule a future action based on agent's decision"""

        # Parse time specification
        scheduled_at = parse_time_spec(time_spec)
        if not scheduled_at:
            return (
                f"Error: Could not parse time specification '{time_spec}'\n\n"
                "Supported formats:\n"
                "  - Relative: 'in 2 hours', 'in 30 minutes', 'in 1 day'\n"
                "  - Today: 'at 5pm', '5pm', 'at 17:00', '17:00', 'today at 5pm'\n"
                "  - Tomorrow: 'tomorrow at 9am', 'tomorrow at 14:30'\n"
                "  - Next week: 'next monday 10:00', 'next friday 14:30'"
            )

        # Agent has decided the behavior via parameters (no inference needed)

        # Determine notification channels
        if notify:
            notification_channels = ["desktop", "file"]
        else:
            notification_channels = []

        # Create action
        action = ScheduledAction(
            id=f"action-{int(datetime.now().timestamp() * 1000)}",
            prompt=prompt,
            scheduled_at=scheduled_at,
            created_at=datetime.now(),
            created_by="agent",
            description=description or prompt[:100],
            execute_query=execute_query,
            notify=notify,
            create_report=create_report,
            notification_channels=notification_channels,
            status="pending",
            result=None,
            executed_at=None,
        )

        # Save
        await self.manager.save_action(action)

        # Format response
        time_until = scheduled_at - datetime.now()
        hours = int(time_until.total_seconds() / 3600)
        minutes = int((time_until.total_seconds() % 3600) / 60)

        output_mode = []
        if notify:
            output_mode.append("Desktop notification + file log")
        if create_report:
            output_mode.append("Create report")
        if not output_mode:
            output_mode.append("Silent execution")

        return (
            f"✓ Scheduled action for {scheduled_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  ({hours}h {minutes}m from now)\n\n"
            f"Action: {prompt}\n"
            f"Output: {', '.join(output_mode)}\n"
            f"(notify={notify}, create_report={create_report} based on intent)\n\n"
            f"The /monitor process will execute this and send notifications.\n"
            f"⚠️  Note: /monitor process must be running for scheduled actions to execute.\n"
            f"   Start it with: ai-assist /monitor"
        )


def parse_time_spec(spec: str) -> datetime | None:
    """Parse natural language time specification into datetime

    Supports:
    - "in X hours/minutes/days"
    - "at 5pm", "at 17:00", "5pm", "17:00"
    - "today at 5pm"
    - "tomorrow at HH:MM"
    - "next monday HH:MM"
    - Relative times
    """
    now = datetime.now()
    spec = spec.lower().strip()

    # Pattern: "in X hours/minutes/days"
    relative_match = re.match(r"in (\d+)\s*(hour|minute|day|h|m|d)s?", spec)
    if relative_match:
        amount = int(relative_match.group(1))
        unit = relative_match.group(2)

        if unit in ["hour", "h"]:
            return now + timedelta(hours=amount)
        elif unit in ["minute", "m"]:
            return now + timedelta(minutes=amount)
        elif unit in ["day", "d"]:
            return now + timedelta(days=amount)

    # Pattern: "at 5pm", "5pm", "at 17:00", "17:00" (today)
    # Also "today at 5pm"
    time_today_match = re.match(r"(?:today\s+)?(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", spec)
    if time_today_match:
        hour = int(time_today_match.group(1))
        minute = int(time_today_match.group(2)) if time_today_match.group(2) else 0
        meridiem = time_today_match.group(3)

        # Convert 12-hour to 24-hour if am/pm specified
        if meridiem:
            if meridiem == "pm" and hour != 12:
                hour += 12
            elif meridiem == "am" and hour == 12:
                hour = 0

        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

        # If the time has already passed today, assume they mean tomorrow
        if target <= now:
            target += timedelta(days=1)

        return target

    # Pattern: "tomorrow at 9am", "tomorrow at HH:MM", "tomorrow HH:MM"
    tomorrow_match = re.match(r"tomorrow\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", spec)
    if tomorrow_match:
        hour = int(tomorrow_match.group(1))
        minute = int(tomorrow_match.group(2)) if tomorrow_match.group(2) else 0
        meridiem = tomorrow_match.group(3)

        # Convert 12-hour to 24-hour if am/pm specified
        if meridiem:
            if meridiem == "pm" and hour != 12:
                hour += 12
            elif meridiem == "am" and hour == 12:
                hour = 0

        target = now + timedelta(days=1)
        return target.replace(hour=hour, minute=minute, second=0, microsecond=0)

    # Pattern: "next monday HH:MM"
    weekday_match = re.match(
        r"next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+(\d{1,2}):(\d{2})", spec
    )
    if weekday_match:
        weekdays = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
        target_weekday = weekdays[weekday_match.group(1)]
        hour = int(weekday_match.group(2))
        minute = int(weekday_match.group(3))

        # Calculate days until next occurrence
        current_weekday = now.weekday()
        days_ahead = (target_weekday - current_weekday) % 7
        if days_ahead == 0:
            days_ahead = 7  # Next week, not today

        target = now + timedelta(days=days_ahead)
        return target.replace(hour=hour, minute=minute, second=0, microsecond=0)

    return None
