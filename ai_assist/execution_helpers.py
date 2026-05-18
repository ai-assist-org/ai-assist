"""Shared helpers for task/action execution — notifications and state tracking"""

from datetime import datetime
from typing import Any

from .event_sources import EventContext
from .notification_dispatcher import Notification, NotificationDispatcher


async def send_failure_notification(action_name: str, error_summary: str, timestamp: datetime) -> None:
    notification = Notification(
        id=f"action-error-{action_name}-{int(timestamp.timestamp() * 1000)}",
        action_id=action_name,
        title=f"Action failed: {action_name}",
        message=error_summary[:500] if error_summary else "Unknown error",
        level="error",
        timestamp=timestamp,
        channels=["desktop", "console"],
        delivered={},
    )
    dispatcher = NotificationDispatcher()
    await dispatcher.dispatch(notification)


def build_history_entry(
    task_name: str,
    success: bool,
    timestamp: datetime,
    metadata: dict[str, Any],
    event_context: EventContext | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "task_name": task_name,
        "success": success,
        "timestamp": timestamp.isoformat(),
    }
    if success:
        entry["metadata"] = metadata
    else:
        entry["error"] = error or "Unknown error"
    if event_context is not None:
        entry["event_source"] = event_context.source_type
        entry["event_metadata"] = event_context.metadata
    return entry
