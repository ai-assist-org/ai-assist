"""Plan mode is enforced when tools run, not only when the tool list is built."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig
from ai_assist.plan_mode import get_planning_tools


def _mcp_tool(name):
    return {"name": f"test__{name}", "description": name, "input_schema": {"type": "object"}, "_server": "test"}


@pytest.fixture
def agent(tmp_path):
    agent = AiAssistAgent(config=AiAssistConfig(anthropic_api_key="test-key", allowed_paths=[str(tmp_path)]))
    agent.available_tools = agent.filesystem_tools.get_tool_definitions() + [
        _mcp_tool("get_issue"),
        _mcp_tool("transition_issue"),
    ]
    session = AsyncMock()
    result = MagicMock()
    result.content = [MagicMock(text='{"ok": true}')]
    session.call_tool.return_value = result
    agent.sessions = {"test": session}
    agent.plan_mode = True
    return agent


@pytest.mark.asyncio
async def test_internal_write_tool_refused(agent, tmp_path):
    target = tmp_path / "x.txt"

    result = await agent._execute_tool("internal__write_file", {"path": str(target), "content": "x"})

    assert "plan mode" in result.lower()
    assert not target.exists()


@pytest.mark.asyncio
async def test_unannotated_mcp_write_tool_refused(agent):
    result = await agent._execute_tool("test__transition_issue", {})

    assert "plan mode" in result.lower()
    agent.sessions["test"].call_tool.assert_not_called()


@pytest.mark.asyncio
async def test_read_tool_allowed(agent):
    result = await agent._execute_tool("test__get_issue", {})

    assert result == '{"ok": true}'


@pytest.mark.asyncio
async def test_file_redirection_refused(agent, tmp_path):
    target = tmp_path / "out.json"

    result = await agent._execute_tool("test__get_issue", {"__save_to_file": str(target)})

    assert "plan mode" in result.lower()
    assert not target.exists()


@pytest.mark.asyncio
async def test_unknown_tool_refused(agent):
    result = await agent._execute_tool("test__not_listed", {})

    assert "plan mode" in result.lower()


@pytest.mark.asyncio
async def test_write_tools_allowed_outside_plan_mode(agent, tmp_path):
    agent.plan_mode = False
    target = tmp_path / "x.txt"

    await agent._execute_tool("internal__write_file", {"path": str(target), "content": "x"})

    assert target.exists()


@pytest.mark.parametrize("name", ["transition_issue", "convert_report_to_doc", "download_file", "now", "run_job"])
def test_heuristic_denies_unrecognized_names(name):
    assert get_planning_tools([_mcp_tool(name)]) == []


@pytest.mark.parametrize("name", ["get_issue", "list_boards", "search_jobs", "query_components", "count_tickets"])
def test_heuristic_allows_read_names(name):
    assert len(get_planning_tools([_mcp_tool(name)])) == 1
