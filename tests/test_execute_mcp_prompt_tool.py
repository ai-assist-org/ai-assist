"""Tests for executing MCP prompts via introspection tool"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig
from ai_assist.introspection_tools import IntrospectionTools


@pytest.fixture
def mock_config():
    """Create mock config"""
    config = MagicMock(spec=AiAssistConfig)
    config.use_vertex = False
    config.anthropic_api_key = "test-key"
    config.model = "claude-3-5-sonnet-20241022"
    config.mcp_servers = {}
    config.allow_skill_script_execution = False
    config.allowed_commands = ["grep", "find", "wc", "sort", "head", "tail", "ls", "cat", "diff", "file", "stat"]
    config.allowed_paths = ["~/.ai-assist", "/tmp/ai-assist"]
    config.confirm_tools = ["internal__create_directory"]
    return config


@pytest.fixture
def agent_with_prompts(mock_config):
    """Create agent with mock prompts"""
    agent = AiAssistAgent(mock_config)

    # Add mock session
    agent.sessions["test_server"] = MagicMock()

    # Create mock prompt
    mock_prompt = MagicMock()
    mock_prompt.name = "test_prompt"
    mock_prompt.description = "Test prompt"
    mock_prompt.arguments = []

    agent.available_prompts["test_server"] = {"test_prompt": mock_prompt}

    return agent


def test_introspection_tools_has_agent_reference(mock_config):
    """Test that introspection tools get agent reference"""
    agent = AiAssistAgent(mock_config)

    # Agent reference should be set
    assert agent.introspection_tools.agent is agent


@pytest.mark.asyncio
async def test_execute_mcp_prompt_tool_added_with_agent(mock_config):
    """Test that execute_mcp_prompt tool is added when agent is available"""
    agent = AiAssistAgent(mock_config)
    await agent.connect_to_servers()

    # Find execute_mcp_prompt tool
    execute_tool = None
    for tool in agent.available_tools:
        if tool.get("name") == "introspection__execute_mcp_prompt":
            execute_tool = tool
            break

    # Should exist
    assert execute_tool is not None
    assert execute_tool["_server"] == "introspection"
    assert "execute an mcp prompt directly" in execute_tool["description"].lower()


@pytest.mark.asyncio
async def test_execute_mcp_prompt_tool_not_added_without_agent():
    """Test that execute_mcp_prompt tool is NOT added when agent is None"""
    introspection_tools = IntrospectionTools(agent=None)

    tool_defs = introspection_tools.get_tool_definitions()
    tool_names = [t["name"] for t in tool_defs]

    # Should NOT have execute_mcp_prompt
    assert "introspection__execute_mcp_prompt" not in tool_names
    # Should still have inspect_mcp_prompt
    assert "introspection__inspect_mcp_prompt" in tool_names


@pytest.mark.asyncio
async def test_execute_mcp_prompt_success(agent_with_prompts):
    """Test successful MCP prompt execution"""
    # Mock the execute_mcp_prompt method to return a result
    agent_with_prompts.execute_mcp_prompt = AsyncMock(return_value="Test result from prompt")

    # Execute via the tool
    result_json = await agent_with_prompts.introspection_tools.execute_tool(
        "execute_mcp_prompt", {"server": "test_server", "prompt": "test_prompt", "arguments": {}}
    )

    result = json.loads(result_json)

    # Should succeed
    assert result["success"] is True
    assert result["server"] == "test_server"
    assert result["prompt"] == "test_prompt"
    assert result["result"] == "Test result from prompt"

    # Verify execute_mcp_prompt was called correctly
    agent_with_prompts.execute_mcp_prompt.assert_called_once_with("test_server", "test_prompt", {})


@pytest.mark.asyncio
async def test_execute_mcp_prompt_with_arguments(agent_with_prompts):
    """Test MCP prompt execution with arguments"""
    agent_with_prompts.execute_mcp_prompt = AsyncMock(return_value="Report for Semih")

    # Execute with arguments
    result_json = await agent_with_prompts.introspection_tools.execute_tool(
        "execute_mcp_prompt",
        {"server": "test_server", "prompt": "test_prompt", "arguments": {"for": "Semih", "days": "7"}},
    )

    result = json.loads(result_json)

    assert result["success"] is True
    assert "Semih" in result["result"]

    # Verify arguments were passed
    agent_with_prompts.execute_mcp_prompt.assert_called_once_with(
        "test_server", "test_prompt", {"for": "Semih", "days": "7"}
    )


@pytest.mark.asyncio
async def test_execute_mcp_prompt_error_handling(agent_with_prompts):
    """Test error handling in MCP prompt execution"""
    # Mock execute_mcp_prompt to raise an error
    agent_with_prompts.execute_mcp_prompt = AsyncMock(side_effect=ValueError("Server not found"))

    # Execute the tool
    result_json = await agent_with_prompts.introspection_tools.execute_tool(
        "execute_mcp_prompt", {"server": "nonexistent", "prompt": "test"}
    )

    result = json.loads(result_json)

    # Should have error
    assert result["success"] is False
    assert "error" in result
    assert "Server not found" in result["error"]


@pytest.mark.asyncio
async def test_execute_mcp_prompt_without_agent_reference():
    """Test that execute_mcp_prompt fails gracefully without agent"""
    introspection_tools = IntrospectionTools(agent=None)

    result_json = await introspection_tools.execute_tool("execute_mcp_prompt", {"server": "test", "prompt": "test"})

    result = json.loads(result_json)

    # Should have error about missing agent
    assert "error" in result
    assert "agent reference not available" in result["error"].lower()


@pytest.mark.asyncio
async def test_execute_mcp_prompt_via_agent_execute_tool(agent_with_prompts):
    """Test executing MCP prompt through agent's _execute_tool"""
    agent_with_prompts.execute_mcp_prompt = AsyncMock(return_value="Weekly report generated")

    # Execute via agent's _execute_tool (with full prefixed name)
    result_json = await agent_with_prompts._execute_tool(
        "introspection__execute_mcp_prompt",
        {"server": "test_server", "prompt": "weekly_report", "arguments": {"for": "Peri"}},
    )

    result = json.loads(result_json)

    assert result["success"] is True
    assert result["result"] == "Weekly report generated"
    agent_with_prompts.execute_mcp_prompt.assert_called_once_with("test_server", "weekly_report", {"for": "Peri"})
