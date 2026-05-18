"""D-Bus event source — requires dbus-next (pip install ai-assist[dbus])"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from .event_sources import EventContext, EventSource

logger = logging.getLogger(__name__)


class DbusEventSource(EventSource):
    """Event source that subscribes to D-Bus signals on session and/or system bus"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.default_bus_type = config.get("bus", "session")
        self._subscriptions: list[tuple[str, dict[str, Any]]] = []
        self._dispatch: Callable[[str, EventContext], Awaitable[None]] | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._buses: list[Any] = []

    @property
    def name(self) -> str:
        return "dbus"

    def subscribe(self, task_name: str, trigger_config: dict[str, Any]) -> None:
        self._subscriptions.append((task_name, trigger_config))

    def unsubscribe_all(self) -> None:
        self._subscriptions.clear()

    async def start(self, dispatch: Callable[[str, EventContext], Awaitable[None]]) -> None:
        self._dispatch = dispatch

        needed_buses = set()
        for _task_name, trigger in self._subscriptions:
            bus = trigger.get("bus", self.default_bus_type)
            needed_buses.add(bus)

        for bus_type_str in needed_buses:
            subs = [
                (name, trigger)
                for name, trigger in self._subscriptions
                if trigger.get("bus", self.default_bus_type) == bus_type_str
            ]
            task = asyncio.create_task(self._listen(bus_type_str, subs))
            self._tasks.append(task)

    async def stop(self) -> None:
        for task in self._tasks:
            await self._cancel_task(task)
        self._tasks.clear()
        for bus in self._buses:
            bus.disconnect()
        self._buses.clear()

    async def _listen(self, bus_type_str: str, subscriptions: list[tuple[str, dict[str, Any]]]) -> None:
        from dbus_next import BusType, Message, MessageType
        from dbus_next.aio import MessageBus

        bus_type = BusType.SYSTEM if bus_type_str == "system" else BusType.SESSION
        bus = await MessageBus(bus_type=bus_type).connect()
        self._buses.append(bus)

        for _task_name, trigger in subscriptions:
            rule = self._build_match_rule(trigger)
            await bus.call(
                Message(
                    destination="org.freedesktop.DBus",
                    path="/org/freedesktop/DBus",
                    interface="org.freedesktop.DBus",
                    member="AddMatch",
                    signature="s",
                    body=[rule],
                )
            )

        def on_message(msg: Message) -> bool:
            if msg.message_type != MessageType.SIGNAL:
                return False
            for task_name, trigger in subscriptions:
                if self._signal_matches(msg, trigger):
                    event = EventContext(
                        source_type="dbus",
                        event_type=msg.member or "signal",
                        payload=str(msg.body) if msg.body else "",
                        metadata={
                            "interface": msg.interface,
                            "signal": msg.member,
                            "sender": msg.sender,
                            "path": msg.path,
                            "bus": bus_type_str,
                        },
                        timestamp=datetime.now(),
                    )
                    if self._dispatch is not None:
                        asyncio.ensure_future(self._dispatch(task_name, event))
            return False

        bus.add_message_handler(on_message)
        logger.info("D-Bus connected (%s bus), watching %d signal(s)", bus_type_str, len(subscriptions))

        try:
            await bus.wait_for_disconnect()
        except asyncio.CancelledError:
            pass

    @staticmethod
    def _signal_matches(msg: Any, trigger: dict[str, Any]) -> bool:
        if msg.interface != trigger.get("interface"):
            return False
        if msg.member != trigger.get("signal"):
            return False
        if "path" in trigger and msg.path != trigger["path"]:
            return False
        return True

    @staticmethod
    def _build_match_rule(trigger: dict[str, Any]) -> str:
        parts = ["type='signal'"]
        if "interface" in trigger:
            parts.append(f"interface='{trigger['interface']}'")
        if "signal" in trigger:
            parts.append(f"member='{trigger['signal']}'")
        if "path" in trigger:
            parts.append(f"path='{trigger['path']}'")
        return ",".join(parts)
