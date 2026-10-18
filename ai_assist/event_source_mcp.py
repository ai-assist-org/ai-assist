"""MCP notification event source.

The source deliberately knows only the local event-bridge extension, not any
particular upstream service.  A bridge owns its remote transport and emits a
small, reference-only envelope over the existing MCP connection.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from mcp import types
from pydantic import BaseModel, ConfigDict, Field

from .event_sources import EventContext, EventSource

if TYPE_CHECKING:
    from .agent import AiAssistAgent

logger = logging.getLogger(__name__)

EVENT_NOTIFICATION_METHOD = "notifications/event-bridge/event"


class MCPEventEnvelope(BaseModel):
    """Stable local event-bridge envelope sent as an MCP notification."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(alias="eventId")
    source: str
    event: str
    timestamp: datetime
    data: dict[str, Any] = Field(default_factory=dict)


class _AnyResult(types.Result):
    """Accept extension results while retaining their extra fields for callers."""

    model_config = ConfigDict(extra="allow")


class MCPEventSource(EventSource):
    """Subscribe to local MCP Events and turn bridge notifications into actions."""

    def __init__(self, config: dict[str, Any], agent: AiAssistAgent) -> None:
        self.config = config
        self.agent = agent
        self._subscriptions: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._dispatch: Callable[[str, EventContext], Awaitable[None]] | None = None
        self._running = False

    @property
    def name(self) -> str:
        return "mcp"

    def subscribe(self, task_name: str, trigger_config: dict[str, Any]) -> None:
        server = trigger_config["server"]
        event = trigger_config["event"]
        arguments = trigger_config.get("arguments", {})
        key = (server, event, json.dumps(arguments, sort_keys=True, separators=(",", ":")))
        entry = self._subscriptions.setdefault(
            key, {"server": server, "event": event, "arguments": arguments, "tasks": []}
        )
        entry["tasks"].append(task_name)

    def unsubscribe_all(self) -> None:
        self._subscriptions.clear()

    async def start(self, dispatch: Callable[[str, EventContext], Awaitable[None]]) -> None:
        self._dispatch = dispatch
        self._running = True
        self.agent.add_mcp_event_listener(self._on_notification)
        self.agent.add_mcp_server_listener(self._on_server_connected)
        for server in tuple(self.agent.sessions):
            await self._subscribe_server(server)

    async def stop(self) -> None:
        self._running = False
        self.agent.remove_mcp_event_listener(self._on_notification)
        self.agent.remove_mcp_server_listener(self._on_server_connected)
        for entry in tuple(self._subscriptions.values()):
            await self._request(entry["server"], "events/unsubscribe", entry, quiet=True)

    async def _on_server_connected(self, server: str) -> None:
        if self._running:
            await self._subscribe_server(server)

    async def _subscribe_server(self, server: str) -> None:
        for entry in self._subscriptions.values():
            if entry["server"] == server:
                await self._request(server, "events/subscribe", entry)

    async def _request(self, server: str, method: str, entry: dict[str, Any], quiet: bool = False) -> None:
        session = self.agent.sessions.get(server)
        if session is None:
            if not quiet:
                logger.warning("MCP event server '%s' is not connected", server)
            return
        params = {
            "name": entry["event"],
            "arguments": entry["arguments"],
            "delivery": {"mode": "mcp"},
        }
        try:
            await session.send_request(types.Request(method=method, params=params), _AnyResult)
        except Exception:
            # agentmail-mcp's Python bridge is pinned to an MCP SDK that cannot
            # parse extension request methods. Its private control tools retain
            # the same local-only semantics until that SDK gains the API.
            helper = "_event_bridge_subscribe" if method == "events/subscribe" else "_event_bridge_unsubscribe"
            try:
                arguments = {"event": entry["event"]}
                if "inboxId" in entry["arguments"]:
                    arguments["inboxId"] = entry["arguments"]["inboxId"]
                await session.call_tool(helper, arguments)
                return
            except Exception:
                pass
            if quiet:
                logger.debug("Unable to unsubscribe MCP event from %s", server, exc_info=True)
            else:
                logger.exception("Unable to subscribe to MCP event '%s' from %s", entry["event"], server)

    async def _on_notification(self, server: str, envelope: MCPEventEnvelope) -> None:
        if not self._running or self._dispatch is None:
            return
        event = EventContext(
            source_type="mcp",
            event_type=envelope.event,
            payload=json.dumps(envelope.data, sort_keys=True),
            metadata={
                "server": server,
                "source": envelope.source,
                "event_id": envelope.event_id,
                "data": envelope.data,
            },
            timestamp=envelope.timestamp,
        )
        # The ActionScheduler owns the matching and therefore needs each event once.
        await self._dispatch("mcp", event)
