"""MQTT event source — requires aiomqtt (pip install ai-assist[mqtt])"""

import asyncio
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from .event_sources import EventContext, EventSource

logger = logging.getLogger(__name__)

_INITIAL_BACKOFF = 5.0
_MAX_BACKOFF = 300.0
# A connection must stay up this long before the backoff starts over: a broker
# that accepts then drops at once (e.g. another client took over the session)
# must slow the retries down, not loop every 5 s.
_STABLE_CONNECTION_SECONDS = 60.0

# Seams for tests
_sleep = asyncio.sleep
_now = time.monotonic


class MqttEventSource(EventSource):
    """Event source that subscribes to MQTT topics and dispatches messages as events"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.broker = config.get("broker", "localhost")
        self.port = config.get("port", 1883)
        # Unique per listener: two listeners with the same id knock each other
        # off the broker (it drops the existing session) in an endless loop.
        default_id = f"ai-assist-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.client_id = config.get("client_id", default_id)
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
        await self._cancel_task(self._task)  # never two listeners for one source
        self._dispatch = dispatch
        self._running = True
        self._task = asyncio.create_task(self._listen())

    async def stop(self) -> None:
        self._running = False
        await self._cancel_task(self._task)

    async def _listen(self) -> None:
        import aiomqtt

        backoff = _INITIAL_BACKOFF

        while self._running:
            connected_at: float | None = None
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
                    connected_at = _now()

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
                if connected_at is not None and _now() - connected_at >= _STABLE_CONNECTION_SECONDS:
                    backoff = _INITIAL_BACKOFF
                logger.exception("MQTT connection error, reconnecting in %.0fs", backoff)
                await _sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF)
