"""Tests for __jq_filter parameter in tool execution."""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig

requires_jq = pytest.mark.skipif(not shutil.which("jq"), reason="jq not installed")


def _make_agent_with_mcp(result_text):
    """Create an agent with a mocked MCP session returning result_text."""
    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"])
    agent = AiAssistAgent(config=config)

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.content = [MagicMock(text=result_text)]
    mock_session.call_tool.return_value = mock_result
    agent.sessions = {"test": mock_session}
    return agent


@requires_jq
@pytest.mark.asyncio
async def test_jq_filter_mcp_tool():
    """Test __jq_filter applies jq filter to MCP tool result."""
    data = json.dumps({"hits": [{"id": "job1", "status": "ok"}, {"id": "job2", "status": "failed"}]})
    agent = _make_agent_with_mcp(data)

    result = await agent._execute_tool(
        "test__some_tool",
        {"query": "test", "__jq_filter": '[.hits[] | select(.status == "failed")]'},
    )

    parsed = json.loads(result.strip())
    assert len(parsed) == 1
    assert parsed[0]["id"] == "job2"


@requires_jq
@pytest.mark.asyncio
async def test_jq_filter_with_save_to_file():
    """Test __jq_filter + __save_to_file saves filtered (not raw) result."""
    data = json.dumps({"items": [1, 2, 3], "total": 3})
    agent = _make_agent_with_mcp(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "filtered.json"
        result = await agent._execute_tool(
            "test__some_tool",
            {"__jq_filter": ".items", "__save_to_file": str(output_file)},
        )

        assert "Result saved to" in result
        saved = json.loads(output_file.read_text())
        assert saved == [1, 2, 3]


@requires_jq
@pytest.mark.asyncio
async def test_jq_filter_with_write_to_report():
    """Test __jq_filter + __write_to_report writes filtered result."""
    data = json.dumps({"name": "hello", "extra": "ignored"})
    agent = _make_agent_with_mcp(data)

    result = await agent._execute_tool(
        "test__some_tool",
        {"__jq_filter": ".name", "__write_to_report": "test-report"},
    )

    assert "test-report" in result


@requires_jq
@pytest.mark.asyncio
async def test_jq_filter_invalid_expression():
    """Test __jq_filter with invalid jq expression returns error."""
    agent = _make_agent_with_mcp('{"a": 1}')

    result = await agent._execute_tool(
        "test__some_tool",
        {"__jq_filter": ".["},
    )

    assert "jq error:" in result


@requires_jq
@pytest.mark.asyncio
async def test_jq_filter_with_collect_to_report_ignored():
    """Test __jq_filter is ignored when __collect_to_report is also set."""
    data = json.dumps({"items": [1, 2, 3]})
    agent = _make_agent_with_mcp(data)

    result = await agent._execute_tool(
        "test__some_tool",
        {"__jq_filter": ".items", "__collect_to_report": "test:jsonl"},
    )

    assert "not configured" in result.lower() or "error" in result.lower() or "report" in result.lower()


@requires_jq
@pytest.mark.asyncio
async def test_jq_filter_on_execute_command():
    """Test __jq_filter on execute_command applies to stdout, not the wrapper."""
    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"])
    agent = AiAssistAgent(config=config)
    await agent.connect_to_servers()

    result = await agent._execute_tool(
        "internal__execute_command",
        {
            "command": 'echo \'[{"name":"alpha","score":10},{"name":"beta","score":20}]\'',
            "__jq_filter": "[.[].name]",
        },
    )

    parsed = json.loads(result.strip())
    assert parsed == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_without_jq_filter_unchanged():
    """Test that tools without __jq_filter return unmodified results."""
    data = '{"data": "test"}'
    agent = _make_agent_with_mcp(data)

    result = await agent._execute_tool("test__some_tool", {"query": "test"})

    assert result == data
