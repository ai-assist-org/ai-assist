"""Tests for the AWL finish tool (internal__finish_task)."""

import pytest

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig
from ai_assist.task_result_tool import TaskResultTool


@pytest.mark.asyncio
async def test_records_success_with_values():
    tool = TaskResultTool()
    prev = tool.begin()

    reply = await tool.execute_tool("finish_task", {"status": "success", "values": {"count": 3}})

    assert not reply.startswith("Error:")
    assert tool.end(prev) == {"status": "success", "reason": "", "values": {"count": 3}}


@pytest.mark.asyncio
async def test_records_failure_with_reason():
    tool = TaskResultTool()
    prev = tool.begin()

    await tool.execute_tool("finish_task", {"status": "failed", "reason": "listing returned null"})

    assert tool.end(prev) == {"status": "failed", "reason": "listing returned null", "values": {}}


@pytest.mark.asyncio
async def test_failure_requires_reason():
    tool = TaskResultTool()
    prev = tool.begin()

    reply = await tool.execute_tool("finish_task", {"status": "failed"})

    assert reply.startswith("Error:")
    assert tool.end(prev) is None


@pytest.mark.asyncio
async def test_rejects_unknown_status():
    tool = TaskResultTool()
    prev = tool.begin()

    reply = await tool.execute_tool("finish_task", {"status": "partial"})

    assert reply.startswith("Error:")
    assert tool.end(prev) is None


@pytest.mark.asyncio
async def test_last_call_wins():
    tool = TaskResultTool()
    prev = tool.begin()

    await tool.execute_tool("finish_task", {"status": "failed", "reason": "first try"})
    await tool.execute_tool("finish_task", {"status": "success", "values": {"x": 1}})

    assert tool.end(prev)["status"] == "success"


@pytest.mark.asyncio
async def test_not_callable_outside_a_task():
    tool = TaskResultTool()

    reply = await tool.execute_tool("finish_task", {"status": "success"})

    assert reply.startswith("Error:")


@pytest.mark.asyncio
async def test_nested_task_does_not_leak_into_outer():
    tool = TaskResultTool()
    outer = tool.begin()
    await tool.execute_tool("finish_task", {"status": "success", "values": {"outer": 1}})

    inner = tool.begin()
    await tool.execute_tool("finish_task", {"status": "failed", "reason": "inner broke"})
    assert tool.end(inner)["reason"] == "inner broke"

    assert tool.active
    assert tool.end(outer)["values"] == {"outer": 1}
    assert not tool.active


def test_schema_is_static():
    definition = TaskResultTool().get_tool_definitions()[0]

    assert definition["name"] == "internal__finish_task"
    schema = definition["input_schema"]
    assert schema["required"] == ["status"]
    assert schema["properties"]["status"]["enum"] == ["success", "failed"]
    assert schema["properties"]["values"]["type"] == "object"


def _agent() -> AiAssistAgent:
    return AiAssistAgent(config=AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"]))


def test_agent_offers_tool_only_during_a_task():
    agent = _agent()

    assert "internal__finish_task" not in [t["name"] for t in agent._build_api_tools()]
    prev = agent.task_result_tool.begin()
    assert "internal__finish_task" in [t["name"] for t in agent._build_api_tools()]
    agent.task_result_tool.end(prev)
    assert "internal__finish_task" not in [t["name"] for t in agent._build_api_tools()]


@pytest.mark.asyncio
async def test_agent_dispatches_tool_call():
    agent = _agent()
    prev = agent.task_result_tool.begin()

    await agent._execute_tool("internal__finish_task", {"status": "failed", "reason": "token expired"})

    assert agent.task_result_tool.end(prev) == {"status": "failed", "reason": "token expired", "values": {}}
