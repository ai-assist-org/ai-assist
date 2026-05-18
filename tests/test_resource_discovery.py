"""Tests for MCP resource discovery and introspection"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig


@pytest.fixture
def mock_config():
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


def _make_resource(
    uri="dci://elasticsearch/mapping",
    name="get_es_mapping",
    description="ES mapping",
    mime_type="text/plain",
    size=None,
):
    res = MagicMock()
    res.uri = uri
    res.name = name
    res.description = description
    res.mimeType = mime_type
    res.size = size
    res.annotations = None
    return res


def _make_resource_template(
    uri_template="dci://jobs/{job_id}", name="get_job", description="Get a job by ID", mime_type="application/json"
):
    tpl = MagicMock()
    tpl.uriTemplate = uri_template
    tpl.name = name
    tpl.description = description
    tpl.mimeType = mime_type
    tpl.annotations = None
    return tpl


@pytest.fixture
def agent_with_resources(mock_config):
    agent = AiAssistAgent(mock_config)
    agent.sessions["dci"] = MagicMock()

    res1 = _make_resource()
    res2 = _make_resource(
        uri="dci://teams/list",
        name="teams_list",
        description="List all teams",
        mime_type="application/json",
        size=4096,
    )
    agent.available_resources["dci"] = [res1, res2]

    tpl1 = _make_resource_template()
    agent.available_resource_templates["dci"] = [tpl1]

    return agent


def test_agent_has_resource_attributes(mock_config):
    agent = AiAssistAgent(mock_config)
    assert hasattr(agent, "available_resources")
    assert hasattr(agent, "available_resource_templates")
    assert isinstance(agent.available_resources, dict)
    assert isinstance(agent.available_resource_templates, dict)


def test_agent_stores_resources(agent_with_resources):
    assert "dci" in agent_with_resources.available_resources
    assert len(agent_with_resources.available_resources["dci"]) == 2

    res = agent_with_resources.available_resources["dci"][0]
    assert res.uri == "dci://elasticsearch/mapping"
    assert res.name == "get_es_mapping"
    assert res.description == "ES mapping"


def test_agent_stores_resource_templates(agent_with_resources):
    assert "dci" in agent_with_resources.available_resource_templates
    assert len(agent_with_resources.available_resource_templates["dci"]) == 1

    tpl = agent_with_resources.available_resource_templates["dci"][0]
    assert tpl.uriTemplate == "dci://jobs/{job_id}"
    assert tpl.name == "get_job"


def test_disconnect_cleans_up_resources(agent_with_resources):
    agent_with_resources._server_tasks = []
    agent_with_resources._disconnect_server("dci")

    assert "dci" not in agent_with_resources.available_resources
    assert "dci" not in agent_with_resources.available_resource_templates


@pytest.mark.asyncio
async def test_list_mcp_resources_all_servers(agent_with_resources):
    result_json = await agent_with_resources.introspection_tools.execute_tool("list_mcp_resources", {})
    result = json.loads(result_json)

    assert "resources" in result
    assert len(result["resources"]) == 2
    assert result["resources"][0]["uri"] == "dci://elasticsearch/mapping"
    assert result["resources"][0]["server"] == "dci"

    assert "resource_templates" in result
    assert len(result["resource_templates"]) == 1
    assert result["resource_templates"][0]["uri_template"] == "dci://jobs/{job_id}"


@pytest.mark.asyncio
async def test_list_mcp_resources_filter_by_server(agent_with_resources):
    result_json = await agent_with_resources.introspection_tools.execute_tool("list_mcp_resources", {"server": "dci"})
    result = json.loads(result_json)
    assert len(result["resources"]) == 2

    result_json = await agent_with_resources.introspection_tools.execute_tool(
        "list_mcp_resources", {"server": "nonexistent"}
    )
    result = json.loads(result_json)
    assert len(result["resources"]) == 0
    assert len(result["resource_templates"]) == 0


@pytest.mark.asyncio
async def test_read_mcp_resource_text(agent_with_resources):
    mock_content = MagicMock()
    mock_content.text = '{"mappings": {"properties": {"id": {"type": "keyword"}}}}'
    mock_content.mimeType = "text/plain"
    type(mock_content).blob = property(lambda self: (_ for _ in ()).throw(AttributeError))

    mock_result = MagicMock()
    mock_result.contents = [mock_content]

    session = agent_with_resources.sessions["dci"]
    session.read_resource = AsyncMock(return_value=mock_result)

    result_json = await agent_with_resources.introspection_tools.execute_tool(
        "read_mcp_resource", {"server": "dci", "uri": "dci://elasticsearch/mapping"}
    )
    result = json.loads(result_json)

    assert "contents" in result
    assert len(result["contents"]) == 1
    assert "mappings" in result["contents"][0]["text"]


@pytest.mark.asyncio
async def test_read_mcp_resource_blob(agent_with_resources):
    mock_content = MagicMock()
    mock_content.blob = "base64data=="
    mock_content.mimeType = "application/octet-stream"
    mock_content.text = None

    mock_result = MagicMock()
    mock_result.contents = [mock_content]

    session = agent_with_resources.sessions["dci"]
    session.read_resource = AsyncMock(return_value=mock_result)

    result_json = await agent_with_resources.introspection_tools.execute_tool(
        "read_mcp_resource", {"server": "dci", "uri": "dci://some/binary"}
    )
    result = json.loads(result_json)

    assert result["contents"][0]["type"] == "blob"
    assert "base64" in result["contents"][0]["summary"]


@pytest.mark.asyncio
async def test_read_mcp_resource_unknown_server(agent_with_resources):
    result_json = await agent_with_resources.introspection_tools.execute_tool(
        "read_mcp_resource", {"server": "unknown", "uri": "x://y"}
    )
    result = json.loads(result_json)
    assert "error" in result


@pytest.mark.asyncio
async def test_read_mcp_resource_session_error(agent_with_resources):
    session = agent_with_resources.sessions["dci"]
    session.read_resource = AsyncMock(side_effect=Exception("Connection lost"))

    result_json = await agent_with_resources.introspection_tools.execute_tool(
        "read_mcp_resource", {"server": "dci", "uri": "dci://elasticsearch/mapping"}
    )
    result = json.loads(result_json)
    assert "error" in result


def test_introspection_tools_have_resource_attributes(mock_config):
    agent = AiAssistAgent(mock_config)
    assert hasattr(agent.introspection_tools, "available_resources")
    assert hasattr(agent.introspection_tools, "available_resource_templates")


def test_resource_tools_in_tool_definitions(mock_config):
    agent = AiAssistAgent(mock_config)
    tool_defs = agent.introspection_tools.get_tool_definitions()
    tool_names = [t["name"] for t in tool_defs]
    assert "introspection__list_mcp_resources" in tool_names
    assert "introspection__read_mcp_resource" in tool_names
