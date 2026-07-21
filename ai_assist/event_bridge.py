"""Event bridge for monitor→interactive IPC via events.jsonl"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.panel import Panel

from .file_watchdog import FileWatchdog

logger = logging.getLogger(__name__)


class BridgeEvent(BaseModel):
    """An event published by /monitor for /interactive to display"""

    model_config = ConfigDict(ser_json_timedelta="iso8601")

    id: str
    timestamp: datetime
    type: str  # "notify", "context"
    source_task: str
    source_type: str  # "mqtt", "dbus", "timer"
    title: str
    body: str
    event_data: dict = {}
    level: str = "info"  # "info", "success", "warning", "error"
    pid: int = Field(default_factory=os.getpid)


class BridgePublisher:
    """Appends BridgeEvents to a JSONL file (used by /monitor)"""

    def __init__(self, events_file: Path) -> None:
        self.events_file = events_file

    async def publish(self, event: BridgeEvent) -> None:
        self.events_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.events_file, "a") as f:
            f.write(event.model_dump_json() + "\n")

    async def cleanup(self, max_age_days: int = 7) -> int:
        if not self.events_file.exists():
            return 0

        cutoff = datetime.now() - timedelta(days=max_age_days)
        kept: list[str] = []
        removed = 0

        with open(self.events_file) as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    ts = datetime.fromisoformat(data["timestamp"])
                    if ts >= cutoff:
                        kept.append(line)
                    else:
                        removed += 1
                except json.JSONDecodeError, KeyError, ValueError:
                    kept.append(line)

        if removed > 0:
            with open(self.events_file, "w") as f:
                for line in kept:
                    f.write(line + "\n")

        return removed


class BridgeWatcher:
    """Watches events.jsonl and displays new events in the TUI (used by /interactive)"""

    def __init__(self, console: Console, events_file: Path) -> None:
        self.console = console
        self.events_file = events_file
        self.watchdog: FileWatchdog | None = None
        self.last_position = 0

        if self.events_file.exists():
            with open(self.events_file) as f:
                f.seek(0, 2)
                self.last_position = f.tell()

    async def on_file_change(self) -> None:
        if not self.events_file.exists():
            return

        try:
            with open(self.events_file) as f:
                f.seek(self.last_position)
                new_lines = f.readlines()
                self.last_position = f.tell()

                for line in new_lines:
                    try:
                        event = BridgeEvent.model_validate_json(line.strip())
                        self._display_event(event)
                    except Exception:
                        pass
        except Exception:
            logger.exception("Error reading events file")

    def _display_event(self, event: BridgeEvent) -> None:
        icons = {
            "info": "📡",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
        }
        colors = {
            "info": "blue",
            "success": "green",
            "warning": "yellow",
            "error": "red",
        }
        icon = icons.get(event.level, "📡")
        color = colors.get(event.level, "white")

        panel = Panel(
            f"{event.body}\n\n[dim]{event.source_type} | {event.timestamp}[/dim]",
            title=f"{icon} {event.title}",
            subtitle=f"[dim]{event.source_task}[/dim]",
            border_style=color,
        )

        self.console.print("\n")
        self.console.print(panel)
        self.console.print("\n")

    async def start(self) -> None:
        self.watchdog = FileWatchdog(self.events_file, self.on_file_change, debounce_seconds=0.1)
        await self.watchdog.start()

    async def stop(self) -> None:
        if self.watchdog:
            await self.watchdog.stop()
