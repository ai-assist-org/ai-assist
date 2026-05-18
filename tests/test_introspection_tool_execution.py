"""Test that introspection tools can be executed successfully via _execute_tool"""

import json
from unittest.mock import MagicMock

import pytest

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig


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


@pytest.mark.asyncio
async def test_execute_introspection_tool_via_agent(mock_config):
    """Test that introspection tools can be executed via _execute_tool"""
    agent = AiAssistAgent(mock_config)
    await agent.connect_to_servers()

    # Add a mock prompt so inspect_mcp_prompt has something to inspect
    mock_prompt = MagicMock()
    mock_prompt.name = "test_prompt"
    mock_prompt.description = "Test prompt"
    mock_prompt.arguments = []

    agent.available_prompts["test_server"] = {"test_prompt": mock_prompt}

    # Execute the tool using the full prefixed name
    result = await agent._execute_tool(
        "introspection__inspect_mcp_prompt", {"server": "test_server", "prompt": "test_prompt"}
    )

    # Should get valid JSON response
    data = json.loads(result)
    assert "server" in data
    assert "prompt" in data
    assert data["server"] == "test_server"
    assert data["prompt"] == "test_prompt"
    assert "next_step" in data


@pytest.mark.asyncio
async def test_execute_introspection_tool_error_handling(mock_config):
    """Test that introspection tools handle errors correctly"""
    agent = AiAssistAgent(mock_config)
    await agent.connect_to_servers()

    # Try to inspect non-existent server
    result = await agent._execute_tool("introspection__inspect_mcp_prompt", {"server": "nonexistent", "prompt": "test"})

    # Should get error in JSON
    data = json.loads(result)
    assert "error" in data
    assert "nonexistent" in data["error"].lower()


@pytest.mark.asyncio
async def test_all_introspection_tools_executable(mock_config):
    """Test that all introspection tools are properly formatted and executable"""
    agent = AiAssistAgent(mock_config)
    await agent.connect_to_servers()

    introspection_tools = [t for t in agent.available_tools if t.get("_server") == "introspection"]

    # All should be executable (no "Invalid tool name format" errors)
    for tool in introspection_tools:
        # Verify name format
        assert "__" in tool["name"], f"Tool {tool['name']} missing __ separator"
        parts = tool["name"].split("__", 1)
        assert len(parts) == 2, f"Tool {tool['name']} doesn't split into 2 parts"
        assert parts[0] == "introspection", f"Tool {tool['name']} has wrong prefix"

        # Verify it can be executed (will fail with appropriate error, not "Invalid tool name format")
        result = await agent._execute_tool(tool["name"], {})
        assert "Invalid tool name format" not in result
