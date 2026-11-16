"""Every executed tool call leaves one audit record carrying its authorization verdict."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_assist.agent import AiAssistAgent
from ai_assist.audit import AuditLogger
from ai_assist.config import AiAssistConfig
from ai_assist.filesystem_tools import is_audit_protected_path


def _mcp_tool(name):
    return {"name": f"test__{name}", "description": name, "input_schema": {"type": "object"}, "_server": "test"}


@pytest.fixture
def agent(tmp_path):
    agent = AiAssistAgent(
        config=AiAssistConfig(anthropic_api_key="test-key", allowed_paths=[str(tmp_path)], allowed_commands=["echo"])
    )
    agent.available_tools = agent.filesystem_tools.get_tool_definitions() + [_mcp_tool("get_issue")]
    session = AsyncMock()
    result = MagicMock()
    result.content = [MagicMock(text='{"ok": true}')]
    result.is_error = False
    session.call_tool.return_value = result
    agent.sessions = {"test": session}
    agent.audit_logger = AuditLogger(audit_dir=tmp_path / "audit")
    return agent


def _entries(agent):
    return [json.loads(line) for line in agent.audit_logger.log_file.read_text().splitlines()]


@pytest.mark.asyncio
async def test_mcp_call_is_recorded_as_autonomous(agent):
    await agent._execute_tool("test__get_issue", {})

    [entry] = _entries(agent)
    assert entry["authorization"] == {"decision": "autonomous", "rule": "MCP tool: no approval gate"}
    assert entry["correlation_id"]


@pytest.mark.asyncio
async def test_plan_mode_refusal_is_recorded(agent):
    agent.plan_mode = True

    await agent._execute_tool("internal__write_file", {"path": "/tmp/x", "content": "x"})

    [entry] = _entries(agent)
    assert entry["success"] is False
    assert entry["authorization"]["decision"] == "refused"
    assert entry["authorization"]["rule"] == "plan-mode"


@pytest.mark.asyncio
async def test_command_policy_refusal_is_recorded(agent):
    result = await agent._execute_tool("internal__execute_command", {"command": "rm -rf /nonexistent"})

    assert result.startswith("Error")
    [entry] = _entries(agent)
    assert entry["authorization"]["decision"] == "refused"


@pytest.mark.asyncio
async def test_path_refusal_is_recorded(agent, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "x.txt"

    await agent._execute_tool("internal__write_file", {"path": str(outside), "content": "x"})

    [entry] = _entries(agent)
    assert entry["authorization"]["decision"] == "refused"
    assert not outside.exists()


@pytest.mark.asyncio
async def test_allowed_command_is_autonomous(agent):
    await agent._execute_tool("internal__execute_command", {"command": "echo hi"})

    [entry] = _entries(agent)
    assert entry["authorization"]["decision"] == "autonomous"
    assert entry["authorization"]["rule"] == "internal tool, permission_mode=manual"


@pytest.mark.asyncio
async def test_approval_is_recorded_with_its_question(agent):
    asked = []

    async def approve(question):
        asked.append(question)
        return True

    agent.filesystem_tools.confirmation_callback = approve
    await agent._execute_tool("internal__execute_command", {"command": "ls /"})

    [entry] = _entries(agent)
    assert asked
    assert entry["authorization"]["decision"] == "approved"
    assert entry["authorization"]["approver"] == "user"
    assert entry["authorization"]["questions"] == [asked[0][:500]]


@pytest.mark.asyncio
async def test_concurrent_calls_keep_their_own_verdicts(agent, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "x.txt"

    await asyncio.gather(
        agent._execute_tool("internal__write_file", {"path": str(outside), "content": "x"}),
        agent._execute_tool("test__get_issue", {}),
        agent._execute_tool("internal__execute_command", {"command": "echo hi"}),
    )

    entries = {e["tool_name"]: e for e in _entries(agent)}
    assert len(entries) == 3
    assert entries["internal__write_file"]["authorization"]["decision"] == "refused"
    assert entries["test__get_issue"]["authorization"]["decision"] == "autonomous"
    assert entries["internal__execute_command"]["authorization"]["decision"] == "autonomous"
    assert len({e["correlation_id"] for e in entries.values()}) == 3


def test_sink_is_protected_from_agent_writes(tmp_path, monkeypatch):
    sink = tmp_path / "remote" / "audit.jsonl"
    monkeypatch.setenv("AI_ASSIST_AUDIT_SINK", str(sink))

    assert is_audit_protected_path(sink)
    assert not is_audit_protected_path(tmp_path / "remote" / "other.txt")
