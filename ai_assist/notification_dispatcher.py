"""Notification dispatcher for multi-channel delivery"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from ai_assist.config import get_config_dir

logger = logging.getLogger(__name__)


class Notification(BaseModel):
    """A notification to be delivered to one or more channels"""

    model_config = ConfigDict(
        ser_json_timedelta="iso8601",
    )

    id: str
    action_id: str
    title: str
    message: str
    level: str  # "info", "success", "warning", "error"
    timestamp: datetime
    channels: list[str]
    delivered: dict[str, bool] = {}


class NotificationDispatcher:
    """Dispatch notifications to multiple channels"""

    def __init__(self, notification_log: Path | None = None):
        from ai_assist.notification_channels import (
            ConsoleNotificationChannel,
            DesktopNotificationChannel,
            FileNotificationChannel,
        )

        self.notification_log = notification_log or (get_config_dir() / "notifications.jsonl")

        self.channels: dict[str, Any] = {
            "console": ConsoleNotificationChannel(),
            "file": FileNotificationChannel(self.notification_log),
            "desktop": DesktopNotificationChannel(),
        }

    async def dispatch(self, notification: Notification) -> dict[str, bool]:
        """Dispatch notification to all specified channels"""
        results = {}

        for channel_name in notification.channels:
            channel = self.channels.get(channel_name)
            if not channel:
                results[channel_name] = False
                continue

            try:
                success = await channel.send(notification)
                results[channel_name] = success
                notification.delivered[channel_name] = success
            except Exception as e:
                logger.error("Error delivering to %s: %s", channel_name, e)
                results[channel_name] = False

        return results
