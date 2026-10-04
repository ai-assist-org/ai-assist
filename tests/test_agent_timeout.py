"""Tests for agent query timeout enforcement."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig


@pytest.fixture
def agent():
    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"])
    return AiAssistAgent(config=config)


@pytest.mark.parametrize("streaming", [False, True])
async def test_timeout_interrupts_tool_and_cleans_query(make_replay_agent, streaming):
    agent = await make_replay_agent(
        [
            {
                "content": [{"type": "tool_use", "id": "t1", "name": "internal__think", "input": {"thought": "wait"}}],
                "stop_reason": "tool_use",
            },
        ]
    )
    cancelled = False
    captured_deadline = None

    async def slow_tools(*args, **kwargs):
        nonlocal cancelled, captured_deadline
        captured_deadline = agent._query_deadline
        try:
            await asyncio.sleep(100)
        finally:
            cancelled = True

    with patch.object(agent, "_execute_tools_concurrently", side_effect=slow_tools):
        start = time.monotonic()
        if streaming:
            events = [event async for event in agent.query_streaming("test", max_time_seconds=1)]
            result = events[-1]["message"]
        else:
            result = await agent.query("test", max_time_seconds=1)

    assert "timeout" in result.lower()
    assert "1 seconds" in result
    assert time.monotonic() - start < 5
    assert cancelled
    assert captured_deadline is not None
    assert agent._query_depth == 0
    assert agent._query_deadline is None
    assert agent._mlflow_root_span is None


async def test_query_default_deadline_and_cleanup(make_replay_agent):
    agent = await make_replay_agent(
        [
            {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"},
        ]
    )
    deadlines = []
    with patch.object(agent, "_execute_tools_concurrently", new=AsyncMock(return_value=([], False))) as execute:

        async def capture(*args, **kwargs):
            deadlines.append(agent._query_deadline)
            return [], False

        execute.side_effect = capture
        before = time.time()
        assert await agent.query("test") == "done"
    assert before + 599 <= deadlines[0] <= time.time() + 600
    assert agent._query_deadline is None


async def test_query_deadline_not_overwritten_by_nested_query(make_replay_agent):
    agent = await make_replay_agent(
        [
            {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"},
        ]
    )
    outer_deadline = time.time() + 300
    agent._query_deadline = outer_deadline
    agent._query_depth = 1
    assert await agent.query("nested test", max_time_seconds=10) == "done"
    assert agent._query_deadline == outer_deadline
    assert agent._query_depth == 1


@pytest.mark.asyncio
async def test_execute_mcp_prompt_inherits_deadline(agent):
    """execute_mcp_prompt should use remaining time from _query_deadline."""
    agent.sessions["test"] = MagicMock()
    mock_prompt_def = MagicMock()
    mock_prompt_def.arguments = []
    agent.available_prompts["test"] = {"my_prompt": mock_prompt_def}

    mock_result = MagicMock()
    mock_msg = MagicMock()
    mock_msg.role = "user"
    mock_msg.content = MagicMock()
    mock_msg.content.text = "test prompt"
    mock_result.messages = [mock_msg]

    agent.sessions["test"].get_prompt = AsyncMock(return_value=mock_result)
    agent._query_deadline = time.time() + 2

    captured_timeout = None

    async def capture_streaming(*args, **kwargs):
        nonlocal captured_timeout
        captured_timeout = kwargs.get("max_time_seconds")
        yield "test response"

    with patch.object(agent, "query_streaming", side_effect=capture_streaming):
        await agent.execute_mcp_prompt("test", "my_prompt")

    assert captured_timeout is not None
    assert captured_timeout <= 2
    assert captured_timeout >= 1


@pytest.mark.asyncio
async def test_execute_mcp_prompt_explicit_timeout_overrides_deadline(agent):
    """Explicit max_time_seconds should override _query_deadline."""
    agent.sessions["test"] = MagicMock()
    mock_prompt_def = MagicMock()
    mock_prompt_def.arguments = []
    agent.available_prompts["test"] = {"my_prompt": mock_prompt_def}

    mock_result = MagicMock()
    mock_msg = MagicMock()
    mock_msg.role = "user"
    mock_msg.content = MagicMock()
    mock_msg.content.text = "test prompt"
    mock_result.messages = [mock_msg]

    agent.sessions["test"].get_prompt = AsyncMock(return_value=mock_result)
    agent._query_deadline = time.time() + 300

    captured_timeout = None

    async def capture_streaming(*args, **kwargs):
        nonlocal captured_timeout
        captured_timeout = kwargs.get("max_time_seconds")
        yield "test response"

    with patch.object(agent, "query_streaming", side_effect=capture_streaming):
        await agent.execute_mcp_prompt("test", "my_prompt", max_time_seconds=42)

    assert captured_timeout == 42
