"""Notification channel implementations"""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_assist.notification_dispatcher import Notification


class ConsoleNotificationChannel:
    """Console-based notifications using Rich"""

    async def send(self, notification: Notification) -> bool:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()

        # Icon based on level
        icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
        }
        icon = icons.get(notification.level, "🔔")

        # Color based on level
        colors = {
            "info": "cyan",
            "success": "green",
            "warning": "yellow",
            "error": "red",
        }
        color = colors.get(notification.level, "white")

        panel = Panel(
            f"{notification.message}\n\n[dim]{notification.timestamp.strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
            title=f"{icon} {notification.title}",
            border_style=color,
        )

        console.print("\n")
        console.print(panel)
        console.print("\n")

        return True


class FileNotificationChannel:
    """File-based notification log (append-only)"""

    def __init__(self, log_file: Path):
        self.log_file = log_file

    async def send(self, notification: Notification) -> bool:
        # Ensure directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        log_entry = {
            "timestamp": notification.timestamp.isoformat(),
            "level": notification.level,
            "title": notification.title,
            "message": notification.message,
            "action_id": notification.action_id,
            "pid": os.getpid(),
        }

        # Append as JSONL
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return True


class DesktopNotificationChannel:
    """Desktop notifications via D-Bus (Linux) or native APIs"""

    async def send(self, notification: Notification) -> bool:
        try:
            import platform

            system = platform.system()

            if system == "Linux":
                return await self._send_dbus(notification)
            elif system == "Darwin":  # macOS
                return await self._send_macos(notification)
            elif system == "Windows":
                return await self._send_windows(notification)
            else:
                print(f"Desktop notifications not supported on {system}")
                return False
        except Exception as e:
            print(f"Desktop notification failed: {e}")
            return False

    async def _send_dbus(self, notification: Notification) -> bool:
        """Send via D-Bus (Linux libnotify)"""
        try:
            import subprocess

            # Use notify-send if available
            urgency = "normal"
            if notification.level == "error":
                urgency = "critical"

            # Desktop notifications have size limits, truncate if needed
            # For query results, show more context than simple reminders
            max_length = 500
            message = notification.message
            if len(message) > max_length:
                message = message[:max_length] + "...\n\n(See full result in TUI or notifications report)"

            subprocess.run(
                [
                    "notify-send",
                    "-u",
                    urgency,
                    notification.title,
                    message,
                ],
                check=True,
            )

            return True
        except subprocess.CalledProcessError, FileNotFoundError:
            return False

    async def _send_macos(self, notification: Notification) -> bool:
        """Send via osascript (macOS)"""
        try:
            import subprocess

            # macOS notifications also have size limits
            max_length = 500
            message = notification.message
            if len(message) > max_length:
                message = message[:max_length] + "..."

            # Escape for AppleScript string literals: backslashes first, then quotes
            message = message.replace("\\", "\\\\").replace('"', '\\"')
            title = notification.title.replace("\\", "\\\\").replace('"', '\\"')
            # Strip control characters that could break AppleScript parsing
            message = "".join(c for c in message if c >= " " or c in "\t")
            title = "".join(c for c in title if c >= " " or c in "\t")

            subprocess.run(["osascript", "-e", f'display notification "{message}" with title "{title}"'], check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    async def _send_windows(self, notification: Notification) -> bool:
        """Send via Windows toast notifications"""
        # Placeholder - would use win10toast or similar
        return False
