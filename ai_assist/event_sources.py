"""Pluggable event source system for reactive task triggering"""

import asyncio
import importlib.util
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EventContext:
    """Context data from an event source, passed to TaskRunner"""

    source_type: str
    event_type: str
    payload: str
    metadata: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class EventSource(ABC):
    """Base class for pluggable event sources (MQTT, DBUS, etc.)"""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def subscribe(self, task_name: str, trigger_config: dict[str, Any]) -> None: ...

    @abstractmethod
    def unsubscribe_all(self) -> None: ...

    @abstractmethod
    async def start(self, dispatch: Callable[[str, EventContext], Awaitable[None]]) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @staticmethod
    async def _cancel_task(task: asyncio.Task | None) -> None:
        """Cancel an asyncio task and wait for it to finish."""
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


class EventSourceManager:
    """Registry and lifecycle manager for event sources"""

    def __init__(self) -> None:
        self._sources: dict[str, EventSource] = {}
        self._runners: dict[str, Any] = {}
        self._event_handler: Callable[[EventContext], Awaitable[None]] | None = None

    def register_source(self, name: str, source: EventSource) -> None:
        self._sources[name] = source

    def get_source(self, name: str) -> EventSource | None:
        return self._sources.get(name)

    def register_available_sources(self, configs: dict[str, Any]) -> None:
        """Register event sources based on config and available dependencies"""
        if "mqtt" in configs:
            if importlib.util.find_spec("aiomqtt") is not None:
                from .event_source_mqtt import MqttEventSource

                mqtt_source = MqttEventSource(configs["mqtt"])
                self.register_source("mqtt", mqtt_source)
                print(f"Event source enabled: MQTT (broker: {mqtt_source.broker}:{mqtt_source.port})")
            else:
                print(
                    "WARNING: MQTT event source configured but aiomqtt not installed. "
                    "Install with: pip install ai-assist[mqtt]"
                )

        if "dbus" in configs:
            if importlib.util.find_spec("dbus_next") is not None:
                from .event_source_dbus import DbusEventSource

                dbus_source = DbusEventSource(configs["dbus"])
                self.register_source("dbus", dbus_source)
                print(f"Event source enabled: D-Bus ({dbus_source.default_bus_type} bus)")
            else:
                print(
                    "WARNING: D-Bus event source configured but dbus-next not installed. "
                    "Install with: pip install ai-assist[dbus]"
                )

        if "file" in configs:
            from .event_source_file import FileEventSource

            file_source = FileEventSource(configs["file"])
            self.register_source("file", file_source)
            print("Event source enabled: File watcher")

    def configure(self, triggered_tasks: list[tuple[str, dict[str, Any], Any]]) -> None:
        for task_name, trigger_config, runner in triggered_tasks:
            source_type = trigger_config.get("type", "")
            source = self._sources.get(source_type)
            if source is None:
                logger.warning("No event source registered for type '%s' (task '%s')", source_type, task_name)
                continue
            source.subscribe(task_name, trigger_config)
            self._runners[task_name] = runner

    async def start(self) -> None:
        for source in self._sources.values():
            await source.start(self._dispatch_event)

    async def stop(self) -> None:
        for source in self._sources.values():
            await source.stop()

    async def _dispatch_event(self, task_name: str, event: EventContext) -> None:
        if self._event_handler is not None:
            try:
                await self._event_handler(event)
            except Exception:
                logger.exception("Error in event handler for '%s'", event.source_type)
            return

        runner = self._runners.get(task_name)
        if runner is None:
            logger.warning("Unknown task '%s' for event from '%s'", task_name, event.source_type)
            return
        try:
            await runner.run(event_context=event)
        except Exception:
            logger.exception("Error dispatching event to task '%s'", task_name)
