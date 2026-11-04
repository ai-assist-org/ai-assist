"""MCP tool results flagged with isError must reach the model as errors, not data."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig, MCPServerConfig, PaginationConfig
from ai_assist.report_tools import ReportTools

ERROR_TEXT = "Authentication failed: invalid API token"


def _mcp_result(text: str, is_error: bool) -> MagicMock:
    result = MagicMock()
    result.content = [MagicMock(text=text)]
    result.is_error = is_error
    return result


def _agent(tmp_path, pagination: PaginationConfig | None = None) -> AiAssistAgent:
    config = AiAssistConfig(
        anthropic_api_key="test-key",
        working_dirs=["/tmp"],
        allowed_paths=[str(tmp_path)],
        mcp_servers={"test": MCPServerConfig(command="echo", pagination=pagination)},
    )
    agent = AiAssistAgent(config=config)
    agent.report_tools = ReportTools(reports_dir=tmp_path / "reports")
    session = AsyncMock()
    session.call_tool.return_value = _mcp_result(ERROR_TEXT, is_error=True)
    agent.sessions = {"test": session}
    return agent


@pytest.mark.asyncio
async def test_mcp_error_returned_as_error(tmp_path):
    agent = _agent(tmp_path)

    result = await agent._execute_tool("test__list_videos", {"channel": "x"})

    assert result.startswith("Error:")
    assert ERROR_TEXT in result


@pytest.mark.asyncio
async def test_mcp_error_not_saved_to_file(tmp_path):
    agent = _agent(tmp_path)
    output_file = tmp_path / "list.json"

    result = await agent._execute_tool("test__list_videos", {"channel": "x", "__save_to_file": str(output_file)})

    assert result.startswith("Error:")
    assert ERROR_TEXT in result
    assert not output_file.exists()


@pytest.mark.asyncio
async def test_mcp_error_flagged_is_error_to_model(tmp_path):
    agent = _agent(tmp_path)
    # Initialize per-query state normally set in the streaming loop
    agent._tool_result_cache = {}
    agent._duplicate_tool_call_count = 0
    agent._recent_tool_calls_for_loop = []
    block = SimpleNamespace(type="tool_use", id="toolu_1", name="test__list_videos", input={"channel": "x"})

    tool_results, _ = await agent._execute_tools_concurrently([block])

    assert tool_results[0]["is_error"] is True
    assert ERROR_TEXT in tool_results[0]["content"]


@pytest.mark.asyncio
async def test_mcp_error_audited_as_failed_and_not_stored(tmp_path, caplog):
    agent = _agent(tmp_path)
    agent.knowledge_graph = MagicMock()
    agent.kg_save_enabled = True

    with caplog.at_level(logging.INFO, logger="ai_assist.audit"):
        await agent._execute_tool("test__list_videos", {"channel": "x"})

    assert "-> FAILED" in caplog.text
    agent.knowledge_graph.upsert_entity.assert_not_called()


@pytest.mark.asyncio
async def test_mcp_error_not_collected_to_report_without_pagination(tmp_path, caplog):
    agent = _agent(tmp_path)

    with caplog.at_level(logging.INFO, logger="ai_assist.audit"):
        result = await agent._execute_tool("test__list_videos", {"channel": "x", "__collect_to_report": "videos:jsonl"})

    assert result.startswith("Error:")
    assert ERROR_TEXT in result
    assert not (tmp_path / "reports" / "videos.jsonl").exists()
    assert "-> FAILED" in caplog.text


@pytest.mark.asyncio
async def test_mcp_error_not_collected_to_report_with_pagination(tmp_path):
    pagination = PaginationConfig(
        offset_param="offset", limit_param="limit", default_page_size=2, total_field="_meta.count"
    )
    agent = _agent(tmp_path, pagination=pagination)

    result = await agent._execute_tool("test__list_videos", {"channel": "x", "__collect_to_report": "videos:jsonl"})

    assert result.startswith("Error:")
    assert ERROR_TEXT in result
    assert not (tmp_path / "reports" / "videos.jsonl").exists()


@pytest.mark.asyncio
async def test_mcp_success_unchanged(tmp_path):
    agent = _agent(tmp_path)
    agent.sessions["test"].call_tool.return_value = _mcp_result('{"videos": []}', is_error=False)

    result = await agent._execute_tool("test__list_videos", {"channel": "x"})

    assert result == '{"videos": []}'


@pytest.mark.asyncio
async def test_real_call_tool_result_error_detected(tmp_path):
    """Use the SDK's own model so a mock can't hide the attribute name."""
    from mcp.types import CallToolResult

    agent = _agent(tmp_path)
    agent.sessions["test"].call_tool.return_value = CallToolResult.model_validate(
        {"content": [{"type": "text", "text": ERROR_TEXT}], "isError": True}
    )

    result = await agent._execute_tool("test__list_videos", {"channel": "x"})

    assert result == f"Error: list_videos failed: {ERROR_TEXT}"


@pytest.mark.asyncio
async def test_real_call_tool_result_success_unchanged(tmp_path):
    from mcp.types import CallToolResult

    agent = _agent(tmp_path)
    agent.sessions["test"].call_tool.return_value = CallToolResult.model_validate(
        {"content": [{"type": "text", "text": '{"videos": []}'}]}
    )

    result = await agent._execute_tool("test__list_videos", {"channel": "x"})

    assert result == '{"videos": []}'
