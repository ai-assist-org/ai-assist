"""Tests for MCP prompt discovery features"""

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


@pytest.fixture
def agent_with_prompts(mock_config):
    """Create agent with mock prompts"""
    agent = AiAssistAgent(mock_config)

    # Add mock session
    agent.sessions["tpci"] = MagicMock()

    # Create mock prompts with different argument configurations
    # Prompt 1: Has required and optional arguments
    mock_arg_required = MagicMock()
    mock_arg_required.name = "for"
    mock_arg_required.required = True
    mock_arg_required.description = "Name of person to generate report for"

    mock_arg_optional = MagicMock()
    mock_arg_optional.name = "period"
    mock_arg_optional.required = False
    mock_arg_optional.description = "Time period for report (default: last week)"

    mock_prompt_with_args = MagicMock()
    mock_prompt_with_args.name = "weekly_report"
    mock_prompt_with_args.description = "Generate weekly status report"
    mock_prompt_with_args.arguments = [mock_arg_required, mock_arg_optional]

    # Prompt 2: No arguments
    mock_prompt_no_args = MagicMock()
    mock_prompt_no_args.name = "status"
    mock_prompt_no_args.description = "Get current status"
    mock_prompt_no_args.arguments = []

    agent.available_prompts["tpci"] = {
        "weekly_report": mock_prompt_with_args,
        "status": mock_prompt_no_args,
    }

    return agent


def test_agent_has_prompts_with_arguments(agent_with_prompts):
    """Test that agent correctly stores prompts with argument information"""
    assert "tpci" in agent_with_prompts.available_prompts
    assert "weekly_report" in agent_with_prompts.available_prompts["tpci"]

    prompt = agent_with_prompts.available_prompts["tpci"]["weekly_report"]
    assert hasattr(prompt, "arguments")
    assert len(prompt.arguments) == 2

    # Check required argument
    required_arg = prompt.arguments[0]
    assert required_arg.name == "for"
    assert required_arg.required is True
    assert required_arg.description == "Name of person to generate report for"

    # Check optional argument
    optional_arg = prompt.arguments[1]
    assert optional_arg.name == "period"
    assert optional_arg.required is False


def test_prompt_without_arguments(agent_with_prompts):
    """Test that prompts without arguments work correctly"""
    prompt = agent_with_prompts.available_prompts["tpci"]["status"]
    assert hasattr(prompt, "arguments")
    assert len(prompt.arguments) == 0


def test_can_identify_required_vs_optional_arguments(agent_with_prompts):
    """Test that we can distinguish required from optional arguments"""
    prompt = agent_with_prompts.available_prompts["tpci"]["weekly_report"]

    required_args = [arg for arg in prompt.arguments if arg.required]
    optional_args = [arg for arg in prompt.arguments if not arg.required]

    assert len(required_args) == 1
    assert required_args[0].name == "for"

    assert len(optional_args) == 1
    assert optional_args[0].name == "period"


@pytest.mark.asyncio
async def test_inspect_mcp_prompt_introspection_tool(agent_with_prompts):
    """Test that agent can introspect MCP prompts via tool"""
    import json

    # Inspect the weekly_report prompt
    result_json = await agent_with_prompts.introspection_tools.execute_tool(
        "inspect_mcp_prompt", {"server": "tpci", "prompt": "weekly_report"}
    )

    result = json.loads(result_json)

    # Verify prompt info
    assert result["server"] == "tpci"
    assert result["prompt"] == "weekly_report"
    assert "next_step" in result

    # Verify arguments
    assert len(result["arguments"]) == 2

    # Check required argument
    req_arg = next(arg for arg in result["arguments"] if arg["name"] == "for")
    assert req_arg["required"] is True
    assert "person to generate report for" in req_arg["description"]

    # Check optional argument
    opt_arg = next(arg for arg in result["arguments"] if arg["name"] == "period")
    assert opt_arg["required"] is False

    # Verify example usage
    assert result["example_usage"]["server"] == "tpci"
    assert result["example_usage"]["prompt"] == "weekly_report"
    assert result["example_usage"]["arguments"] == {"for": "<for>"}


@pytest.mark.asyncio
async def test_inspect_unknown_server(agent_with_prompts):
    """Test inspection of unknown server"""
    import json

    result_json = await agent_with_prompts.introspection_tools.execute_tool(
        "inspect_mcp_prompt", {"server": "unknown", "prompt": "test"}
    )

    result = json.loads(result_json)
    assert "error" in result
    assert "unknown" in result["error"].lower()
    assert "available_servers" in result


@pytest.mark.asyncio
async def test_inspect_unknown_prompt(agent_with_prompts):
    """Test inspection of unknown prompt"""
    import json

    result_json = await agent_with_prompts.introspection_tools.execute_tool(
        "inspect_mcp_prompt", {"server": "tpci", "prompt": "nonexistent"}
    )

    result = json.loads(result_json)
    assert "error" in result
    assert "nonexistent" in result["error"].lower()
    assert "available_prompts" in result
