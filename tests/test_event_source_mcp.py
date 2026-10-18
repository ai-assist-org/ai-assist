from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_assist.event_source_mcp import MCPEventEnvelope, MCPEventSource


@pytest.fixture
def agent():
    agent = MagicMock()
    agent.sessions = {"agentmail": MagicMock()}
    agent.sessions["agentmail"].send_request = AsyncMock()
    agent.sessions["agentmail"].call_tool = AsyncMock()
    return agent


@pytest.mark.asyncio
async def test_subscribes_and_dispatches_reference_only_event(agent):
    source = MCPEventSource({}, agent)
    source.subscribe(
        "new-email",
        {"type": "mcp", "server": "agentmail", "event": "message.received", "arguments": {"inboxId": "a@example.com"}},
    )
    dispatch = AsyncMock()

    await source.start(dispatch)
    agent.sessions["agentmail"].send_request.assert_awaited_once()

    await source._on_notification(
        "agentmail",
        MCPEventEnvelope(
            eventId="evt_1",
            source="agentmail",
            event="message.received",
            timestamp=datetime(2026, 1, 1),
            data={"inboxId": "a@example.com", "messageId": "msg_1"},
        ),
    )

    _, event = dispatch.await_args.args
    assert event.source_type == "mcp"
    assert event.event_type == "message.received"
    assert event.metadata["server"] == "agentmail"
    assert event.metadata["data"] == {"inboxId": "a@example.com", "messageId": "msg_1"}
    await source.stop()


@pytest.mark.asyncio
async def test_falls_back_to_legacy_bridge_control_tool(agent):
    agent.sessions["agentmail"].send_request = AsyncMock(side_effect=RuntimeError("unknown method"))
    source = MCPEventSource({}, agent)
    source.subscribe(
        "new-email",
        {"type": "mcp", "server": "agentmail", "event": "message.received", "arguments": {"inboxId": "a@example.com"}},
    )

    await source.start(AsyncMock())

    agent.sessions["agentmail"].call_tool.assert_awaited_once_with(
        "_event_bridge_subscribe", {"event": "message.received", "inboxId": "a@example.com"}
    )
    await source.stop()
