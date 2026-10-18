"""Tests for generic MCP channel-message routing."""

from unittest.mock import AsyncMock

import pytest

from ai_assist.config import MCPChannelConfig
from ai_assist.mcp_channel import MCPChannelEnvelope, MCPChannelRouter


@pytest.mark.asyncio
async def test_channel_message_spawns_isolated_reply_task():
    tasks = AsyncMock()
    router = MCPChannelRouter(tasks)
    message = MCPChannelEnvelope(content="hello", meta={"chat_id": "42", "user": "example"})

    await router._on_message("chat", MCPChannelConfig(), message)

    kwargs = tasks.spawn_task.call_args.kwargs
    assert kwargs["description"] == "Channel message from chat"
    assert kwargs["max_turns"] == 20
    assert "chat__reply" in kwargs["prompt"]
    assert "`42`" in kwargs["prompt"]
    assert "untrusted inbound message" in kwargs["prompt"]


@pytest.mark.asyncio
async def test_channel_message_without_reply_metadata_is_ignored():
    tasks = AsyncMock()
    router = MCPChannelRouter(tasks)

    await router._on_message("chat", MCPChannelConfig(), MCPChannelEnvelope(content="hello"))

    tasks.spawn_task.assert_not_awaited()
