"""D-Bus event source — requires dbus-next (pip install ai-assist[dbus])"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from .event_sources import EventContext, EventSource

logger = logging.getLogger(__name__)


class DbusEventSource(EventSource):
    """Event source that subscribes to D-Bus signals and dispatches them as events"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.default_bus_type = config.get("bus", "session")
        self._subscriptions: list[tuple[str, dict[str, Any]]] = []
        self._dispatch: Callable[[str, EventContext], Awaitable[None]] | None = None
        self._task: asyncio.Task | None = None
        self._bus: Any = None

    @property
    def name(self) -> str:
        return "dbus"

    def subscribe(self, task_name: str, trigger_config: dict[str, Any]) -> None:
        self._subscriptions.append((task_name, trigger_config))

    def unsubscribe_all(self) -> None:
        self._subscriptions.clear()

    async def start(self, dispatch: Callable[[str, EventContext], Awaitable[None]]) -> None:
        self._dispatch = dispatch
        self._task = asyncio.create_task(self._listen())

    async def stop(self) -> None:
        await self._cancel_task(self._task)
        if self._bus:
            self._bus.disconnect()
            self._bus = None

    async def _listen(self) -> None:
        from dbus_next import BusType, Message, MessageType
        from dbus_next.aio import MessageBus

        bus_type_str = self.default_bus_type
        bus_type = BusType.SYSTEM if bus_type_str == "system" else BusType.SESSION
        self._bus = await MessageBus(bus_type=bus_type).connect()

        for _task_name, trigger in self._subscriptions:
            rule = self._build_match_rule(trigger)
            await self._bus.call(
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
            for task_name, trigger in self._subscriptions:
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
                        },
                        timestamp=datetime.now(),
                    )
                    if self._dispatch is not None:
                        asyncio.ensure_future(self._dispatch(task_name, event))
            return False

        self._bus.add_message_handler(on_message)
        logger.info("D-Bus connected (%s bus), watching %d signal(s)", bus_type_str, len(self._subscriptions))

        try:
            await self._bus.wait_for_disconnect()
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
