"""Tests for __write_to_report and __append_to_report parameters in tool execution."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig
from ai_assist.report_tools import ReportTools


@pytest.fixture
def temp_reports_dir(tmp_path):
    """Temporary reports directory."""
    return tmp_path / "reports"


@pytest.fixture
def agent_with_mcp(temp_reports_dir):
    """Agent with a mocked MCP session and real report tools."""
    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"], allowed_paths=[tempfile.gettempdir()])
    agent = AiAssistAgent(config=config)
    agent.report_tools = ReportTools(reports_dir=temp_reports_dir)

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.content = [MagicMock(text='{"id": "job1", "status": "failure"}')]
    mock_session.call_tool.return_value = mock_result
    agent.sessions = {"test": mock_session}

    return agent


# --- _parse_report_param tests ---


def test_parse_report_param_name_only():
    assert AiAssistAgent._parse_report_param("my-report") == ("my-report", "md")


def test_parse_report_param_with_jsonl():
    assert AiAssistAgent._parse_report_param("data:jsonl") == ("data", "jsonl")


def test_parse_report_param_with_csv():
    assert AiAssistAgent._parse_report_param("export:csv") == ("export", "csv")


def test_parse_report_param_invalid_format():
    with pytest.raises(ValueError, match="Unsupported format"):
        AiAssistAgent._parse_report_param("data:xml")


def test_parse_report_param_empty_name():
    with pytest.raises(ValueError, match="cannot be empty"):
        AiAssistAgent._parse_report_param(":jsonl")


# --- __write_to_report tests ---


@pytest.mark.asyncio
async def test_write_to_report_mcp_tool(agent_with_mcp, temp_reports_dir):
    """__write_to_report saves MCP tool result as a report."""
    result = await agent_with_mcp._execute_tool(
        "test__some_tool",
        {"query": "test", "__write_to_report": "test-report:jsonl"},
    )

    assert "test-report" in result
    report_file = temp_reports_dir / "test-report.jsonl"
    assert report_file.exists()
    assert "job1" in report_file.read_text()


@pytest.mark.asyncio
async def test_write_to_report_default_format(agent_with_mcp, temp_reports_dir):
    """__write_to_report defaults to md format."""
    result = await agent_with_mcp._execute_tool(
        "test__some_tool",
        {"query": "test", "__write_to_report": "test-report"},
    )

    assert "test-report" in result
    report_file = temp_reports_dir / "test-report.md"
    assert report_file.exists()


# --- __append_to_report tests ---


@pytest.mark.asyncio
async def test_append_to_report_creates_file(agent_with_mcp, temp_reports_dir):
    """__append_to_report creates the report if it doesn't exist."""
    result = await agent_with_mcp._execute_tool(
        "test__some_tool",
        {"query": "test", "__append_to_report": "new-report:jsonl"},
    )

    assert "new-report" in result
    report_file = temp_reports_dir / "new-report.jsonl"
    assert report_file.exists()


@pytest.mark.asyncio
async def test_append_to_report_accumulates(agent_with_mcp, temp_reports_dir):
    """Multiple __append_to_report calls accumulate data."""
    # First call
    await agent_with_mcp._execute_tool(
        "test__some_tool",
        {"query": "test", "__append_to_report": "acc-report:jsonl"},
    )

    # Change mock result for second call
    mock_result2 = MagicMock()
    mock_result2.content = [MagicMock(text='{"id": "job2", "status": "success"}')]
    agent_with_mcp.sessions["test"].call_tool.return_value = mock_result2

    # Second call
    await agent_with_mcp._execute_tool(
        "test__some_tool",
        {"query": "test2", "__append_to_report": "acc-report:jsonl"},
    )

    report_file = temp_reports_dir / "acc-report.jsonl"
    content = report_file.read_text()
    assert "job1" in content
    assert "job2" in content


# --- Combined parameters ---


@pytest.mark.asyncio
async def test_save_to_file_and_write_to_report_combined(agent_with_mcp, temp_reports_dir):
    """Both __save_to_file and __write_to_report can be used together."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "result.json"

        result = await agent_with_mcp._execute_tool(
            "test__some_tool",
            {
                "query": "test",
                "__save_to_file": str(output_file),
                "__write_to_report": "combo-report:jsonl",
            },
        )

        # Both should be created
        assert output_file.exists()
        report_file = temp_reports_dir / "combo-report.jsonl"
        assert report_file.exists()

        # Summary should mention both
        assert "saved" in result.lower() or "Result" in result
        assert "combo-report" in result


# --- Existing behavior preserved ---


@pytest.mark.asyncio
async def test_without_redirection_returns_normal(agent_with_mcp):
    """Without redirection params, tool returns result as-is."""
    result = await agent_with_mcp._execute_tool(
        "test__some_tool",
        {"query": "test"},
    )

    assert result == '{"id": "job1", "status": "failure"}'


# --- Error handling ---


@pytest.mark.asyncio
async def test_write_to_report_invalid_format(agent_with_mcp):
    """__write_to_report with invalid format returns error."""
    result = await agent_with_mcp._execute_tool(
        "test__some_tool",
        {"query": "test", "__write_to_report": "report:xml"},
    )

    assert "Error" in result or "Unsupported" in result
