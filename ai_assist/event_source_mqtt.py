"""MQTT event source — requires aiomqtt (pip install ai-assist[mqtt])"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from .event_sources import EventContext, EventSource

logger = logging.getLogger(__name__)


class MqttEventSource(EventSource):
    """Event source that subscribes to MQTT topics and dispatches messages as events"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.broker = config.get("broker", "localhost")
        self.port = config.get("port", 1883)
        self.client_id = config.get("client_id", "ai-assist")
        self.username = config.get("username")
        self.password = config.get("password")
        self._subscriptions: dict[str, list[str]] = {}  # topic_pattern -> [task_name, ...]
        self._dispatch: Callable[[str, EventContext], Awaitable[None]] | None = None
        self._task: asyncio.Task | None = None
        self._running = False

    @property
    def name(self) -> str:
        return "mqtt"

    def subscribe(self, task_name: str, trigger_config: dict[str, Any]) -> None:
        topic = trigger_config["topic"]
        self._subscriptions.setdefault(topic, []).append(task_name)

    def unsubscribe_all(self) -> None:
        self._subscriptions.clear()

    async def start(self, dispatch: Callable[[str, EventContext], Awaitable[None]]) -> None:
        self._dispatch = dispatch
        self._running = True
        self._task = asyncio.create_task(self._listen())

    async def stop(self) -> None:
        self._running = False
        await self._cancel_task(self._task)

    async def _listen(self) -> None:
        import aiomqtt

        while self._running:
            try:
                async with aiomqtt.Client(
                    self.broker,
                    port=self.port,
                    identifier=self.client_id,
                    username=self.username,
                    password=self.password,
                ) as client:
                    for topic_pattern in self._subscriptions:
                        await client.subscribe(topic_pattern)
                    logger.info("MQTT connected to %s:%d", self.broker, self.port)

                    async for message in client.messages:
                        event = EventContext(
                            source_type="mqtt",
                            event_type="message",
                            payload=message.payload.decode("utf-8", errors="replace"),
                            metadata={
                                "topic": str(message.topic),
                                "qos": message.qos,
                                "retain": message.retain,
                            },
                            timestamp=datetime.now(),
                        )
                        assert self._dispatch is not None
                        for pattern, task_names in self._subscriptions.items():
                            if message.topic.matches(pattern):
                                for task_name in task_names:
                                    await self._dispatch(task_name, event)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("MQTT connection error, reconnecting in 5s")
                await asyncio.sleep(5.0)
