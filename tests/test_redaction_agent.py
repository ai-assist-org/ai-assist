"""Tests that tool results are redacted before they reach the LLM."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig

SECRET = "s3cr3t-value-for-tests"


@pytest.fixture
def agent(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_SERVICE_TOKEN", SECRET)
    config = AiAssistConfig(anthropic_api_key="test-key", allowed_paths=[str(tmp_path)])
    return AiAssistAgent(config=config)


def _mock_mcp(agent, text):
    session = AsyncMock()
    result = MagicMock()
    result.content = [MagicMock(text=text)]
    session.call_tool.return_value = result
    agent.sessions = {"test": session}


@pytest.mark.asyncio
async def test_mcp_tool_result_is_redacted(agent):
    _mock_mcp(agent, f'{{"token": "{SECRET}"}}')

    result = await agent._execute_tool("test__some_tool", {})

    assert SECRET not in result
    assert "[REDACTED:TEST_SERVICE_TOKEN]" in result
    assert SECRET not in agent.last_tool_calls[-1]["result"]


@pytest.mark.asyncio
async def test_internal_tool_result_is_redacted(agent, tmp_path):
    secret_file = tmp_path / "service-config.txt"
    secret_file.write_text(f"TEST_SERVICE_TOKEN={SECRET}\n")

    result = await agent._execute_tool("internal__read_file", {"path": str(secret_file)})

    assert SECRET not in result
    assert "[REDACTED:TEST_SERVICE_TOKEN]" in result


@pytest.mark.asyncio
async def test_saved_file_is_redacted(agent, tmp_path):
    _mock_mcp(agent, f"key {SECRET}")
    target = tmp_path / "out.txt"

    await agent._execute_tool("test__some_tool", {"__save_to_file": str(target)})

    assert SECRET not in target.read_text()


@pytest.mark.asyncio
async def test_mcp_prompt_is_redacted(agent):
    message = MagicMock()
    message.role = "user"
    message.content.text = f"use {SECRET}"
    prompt_result = MagicMock(messages=[message])
    session = AsyncMock()
    session.get_prompt.return_value = prompt_result
    agent.sessions = {"test": session}
    agent.available_prompts = {"test": {"p": MagicMock(arguments=[])}}
    sent = []

    async def fake_query_streaming(messages, **kwargs):
        sent.extend(messages)
        yield "done"

    agent.query_streaming = fake_query_streaming

    await agent.execute_mcp_prompt("test", "p", {})

    assert sent
    assert SECRET not in str(sent)
