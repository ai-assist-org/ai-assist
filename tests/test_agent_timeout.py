"""Tests for agent query timeout enforcement."""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig


@pytest.fixture
def agent():
    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"])
    return AiAssistAgent(config=config)


def test_default_query_timeout_from_env(monkeypatch):
    monkeypatch.setenv("AI_ASSIST_QUERY_TIMEOUT", "1200")
    assert AiAssistConfig.from_env().default_query_timeout == 1200


def test_default_query_timeout_defaults_to_600():
    assert AiAssistConfig(anthropic_api_key="test-key").default_query_timeout == 600


@pytest.mark.asyncio
async def test_query_streaming_honors_configured_default_timeout(make_replay_agent):
    """With no explicit max_time_seconds, the deadline must use config.default_query_timeout,
    not the hard-coded 600s — this is what lets long RCA-style sessions raise their budget."""
    agent = await make_replay_agent(
        [
            {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"},
        ]
    )
    agent.config.default_query_timeout = 1234
    deadlines = []
    with patch.object(agent, "_execute_tools_concurrently", new=AsyncMock(return_value=([], False))) as execute:

        async def capture(*args, **kwargs):
            deadlines.append(agent._query_deadline)
            return [], False

        execute.side_effect = capture
        before = time.time()
        assert await agent.query("test") == "done"
    assert before + 1233 <= deadlines[0] <= time.time() + 1234


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


async def test_query_records_live_diagnostic_events(make_replay_agent, tmp_path):
    agent = await make_replay_agent([{"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"}])
    assert await agent.query("test") == "done"

    events = [
        __import__("json").loads(line)
        for line in (tmp_path / ".ai-assist" / "traces" / "query_events.jsonl").read_text().splitlines()
    ]
    assert [event["phase"] for event in events] == [
        "query_started",
        "turn_started",
        "model_stream_opened",
        "model_first_event",
        "model_completed",
        "query_finished",
    ]
    assert all("query_text" not in event for event in events)


async def test_model_stream_idle_timeout_is_enforced(make_replay_agent):
    """A blocked synchronous SDK iterator must not block query cancellation forever."""
    agent = await make_replay_agent([])
    agent.config.model_stream_first_event_timeout_seconds = 1
    agent.config.model_stream_idle_timeout_seconds = 1
    release_stream = threading.Event()

    class SlowStream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def __iter__(self):
            yield SimpleNamespace(type="message_start")
            release_stream.wait()

    client = MagicMock()
    client.messages.stream.return_value = SlowStream()
    agent.anthropic = client

    started = time.monotonic()
    try:
        events = [event async for event in agent.query_streaming("test")]
    finally:
        release_stream.set()

    assert time.monotonic() - started < 1.8
    assert events[-1]["type"] == "error"
    assert "timed out waiting for progress" in events[-1]["message"]


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
