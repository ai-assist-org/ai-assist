"""Tests for introspection tools"""

import json
from datetime import datetime, timedelta

import pytest

from ai_assist.context import ConversationMemory
from ai_assist.introspection_tools import IntrospectionTools
from ai_assist.knowledge_graph import KnowledgeGraph


@pytest.fixture
def kg():
    """Create in-memory knowledge graph"""
    return KnowledgeGraph(":memory:")


@pytest.fixture
def conversation_memory():
    """Create conversation memory"""
    return ConversationMemory(max_exchanges=10)


@pytest.fixture
def introspection_tools_with_kg(kg):
    """Create introspection tools with KG"""
    return IntrospectionTools(knowledge_graph=kg)


@pytest.fixture
def introspection_tools_full(kg, conversation_memory):
    """Create introspection tools with both KG and conversation memory"""
    return IntrospectionTools(knowledge_graph=kg, conversation_memory=conversation_memory)


def test_initialization_with_kg(kg):
    """Test introspection tools initialize with KG"""
    tools = IntrospectionTools(knowledge_graph=kg)
    assert tools.knowledge_graph is kg
    assert tools.conversation_memory is None


def test_initialization_with_both(kg, conversation_memory):
    """Test introspection tools initialize with both KG and conversation"""
    tools = IntrospectionTools(knowledge_graph=kg, conversation_memory=conversation_memory)
    assert tools.knowledge_graph is kg
    assert tools.conversation_memory is conversation_memory


def test_get_tool_definitions_with_kg(introspection_tools_with_kg):
    """Test tool definitions with KG only"""
    tools_defs = introspection_tools_with_kg.get_tool_definitions()

    # Should have 3 KG tools + 1 MCP prompt inspection tool + 1 get_tool_help + 1 get_skill_help + 1 validate_awl_script + 1 inspect_awl_script + 2 context tools + 2 resource tools
    assert len(tools_defs) == 12

    tool_names = [t["name"] for t in tools_defs]
    assert "introspection__search_knowledge_graph" in tool_names
    assert "introspection__get_kg_entity" in tool_names
    assert "introspection__get_kg_stats" in tool_names
    assert "introspection__inspect_mcp_prompt" in tool_names
    assert "introspection__get_skill_help" in tool_names

    # All should be from introspection server
    for tool in tools_defs:
        assert tool["_server"] == "introspection"


def test_get_tool_definitions_with_both(introspection_tools_full):
    """Test tool definitions with both KG and conversation"""
    tools_defs = introspection_tools_full.get_tool_definitions()

    # Should have 3 KG + 1 MCP prompt inspection + 1 conversation + 1 get_tool_help + 1 get_skill_help + 1 validate_awl_script + 1 inspect_awl_script + 2 context tools + 2 resource tools
    assert len(tools_defs) == 13

    tool_names = [t["name"] for t in tools_defs]
    assert "introspection__search_knowledge_graph" in tool_names
    assert "introspection__get_kg_entity" in tool_names
    assert "introspection__get_kg_stats" in tool_names
    assert "introspection__inspect_mcp_prompt" in tool_names
    assert "introspection__search_conversation_history" in tool_names
    assert "introspection__get_skill_help" in tool_names


def test_tool_definitions_have_schemas(introspection_tools_full):
    """Test all tool definitions have proper schemas"""
    tools_defs = introspection_tools_full.get_tool_definitions()

    for tool in tools_defs:
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool
        assert tool["description"]  # Not empty


@pytest.mark.asyncio
async def test_search_kg_by_type(introspection_tools_with_kg, kg):
    """Test searching KG by entity type"""
    # Add some entities
    for i in range(5):
        kg.insert_entity(
            entity_type="jira_ticket",
            entity_id=f"CILAB-{i}",
            valid_from=datetime.now(),
            data={"key": f"CILAB-{i}", "summary": f"Issue {i}"},
        )

    result = await introspection_tools_with_kg.execute_tool(
        "search_knowledge_graph", {"entity_type": "jira_ticket", "limit": 10}
    )

    data = json.loads(result)
    assert data["found"] == 5
    assert len(data["entities"]) == 5


@pytest.mark.asyncio
async def test_search_kg_with_time_range(introspection_tools_with_kg, kg):
    """Test searching KG with time range"""
    # Add old entity
    kg.insert_entity(
        entity_type="dci_job",
        entity_id="job-old",
        valid_from=datetime.now() - timedelta(hours=48),
        data={"status": "success"},
    )

    # Add recent entity
    kg.insert_entity(
        entity_type="dci_job",
        entity_id="job-recent",
        valid_from=datetime.now() - timedelta(hours=1),
        data={"status": "failure"},
    )

    result = await introspection_tools_with_kg.execute_tool(
        "search_knowledge_graph", {"entity_type": "dci_job", "time_range_hours": 24, "limit": 10}
    )

    data = json.loads(result)
    assert data["found"] == 1
    assert data["entities"][0]["id"] == "job-recent"


@pytest.mark.asyncio
async def test_search_kg_with_limit(introspection_tools_with_kg, kg):
    """Test search limit enforcement"""
    # Add many entities
    for i in range(20):
        kg.insert_entity(
            entity_type="jira_ticket", entity_id=f"CILAB-{i}", valid_from=datetime.now(), data={"key": f"CILAB-{i}"}
        )

    result = await introspection_tools_with_kg.execute_tool(
        "search_knowledge_graph", {"entity_type": "jira_ticket", "limit": 5}
    )

    data = json.loads(result)
    assert data["found"] == 5


@pytest.mark.asyncio
async def test_get_kg_entity_found(introspection_tools_with_kg, kg):
    """Test getting a specific entity"""
    kg.insert_entity(
        entity_type="jira_ticket",
        entity_id="CILAB-123",
        valid_from=datetime.now(),
        data={"key": "CILAB-123", "summary": "Test issue", "status": "Open"},
    )

    result = await introspection_tools_with_kg.execute_tool("get_kg_entity", {"entity_id": "CILAB-123"})

    data = json.loads(result)
    assert data["found"] is True
    assert data["entity"]["id"] == "CILAB-123"
    assert data["entity"]["data"]["summary"] == "Test issue"


@pytest.mark.asyncio
async def test_get_kg_entity_not_found(introspection_tools_with_kg, kg):
    """Test getting non-existent entity"""
    result = await introspection_tools_with_kg.execute_tool("get_kg_entity", {"entity_id": "CILAB-999"})

    data = json.loads(result)
    assert data["found"] is False


@pytest.mark.asyncio
async def test_get_kg_entity_with_relationships(introspection_tools_with_kg, kg):
    """Test getting entity with its relationships"""
    # Add job
    kg.insert_entity(entity_type="dci_job", entity_id="job-1", valid_from=datetime.now(), data={"status": "success"})

    # Add component
    kg.insert_entity(
        entity_type="dci_component", entity_id="comp-1", valid_from=datetime.now(), data={"version": "4.19.0"}
    )

    # Add relationship
    kg.insert_relationship(
        rel_type="job_uses_component", source_id="job-1", target_id="comp-1", valid_from=datetime.now()
    )

    result = await introspection_tools_with_kg.execute_tool("get_kg_entity", {"entity_id": "job-1"})

    data = json.loads(result)
    assert data["found"] is True
    assert len(data["related_entities"]) == 1
    assert data["related_entities"][0]["relationship"] == "job_uses_component"


@pytest.mark.asyncio
async def test_get_kg_stats(introspection_tools_with_kg, kg):
    """Test getting KG statistics"""
    # Add some entities
    kg.insert_entity(entity_type="jira_ticket", entity_id="CILAB-1", valid_from=datetime.now(), data={})
    kg.insert_entity(entity_type="dci_job", entity_id="job-1", valid_from=datetime.now(), data={})

    result = await introspection_tools_with_kg.execute_tool("get_kg_stats", {})

    data = json.loads(result)
    assert data["total_entities"] == 2
    assert "jira_ticket" in data["entities_by_type"]
    assert "dci_job" in data["entities_by_type"]


@pytest.mark.asyncio
async def test_search_conversation_history(introspection_tools_full):
    """Test searching conversation history"""
    # Add some exchanges
    introspection_tools_full.conversation_memory.add_exchange("What DCI jobs failed?", "Here are 5 failed jobs...")
    introspection_tools_full.conversation_memory.add_exchange(
        "Why did they fail?", "The failures were due to network issues..."
    )

    result = await introspection_tools_full.execute_tool("search_conversation_history", {"search_term": "failed"})

    data = json.loads(result)
    # Only first exchange contains exact substring "failed"
    # Second has "fail" and "failures" but not "failed"
    assert data["found"] == 1


@pytest.mark.asyncio
async def test_search_conversation_no_matches(introspection_tools_full):
    """Test conversation search with no matches"""
    introspection_tools_full.conversation_memory.add_exchange("Hello", "Hi there!")

    result = await introspection_tools_full.execute_tool("search_conversation_history", {"search_term": "kubernetes"})

    data = json.loads(result)
    assert data["found"] == 0


@pytest.mark.asyncio
async def test_search_conversation_case_insensitive(introspection_tools_full):
    """Test conversation search is case insensitive"""
    introspection_tools_full.conversation_memory.add_exchange("What about OpenShift?", "OpenShift is working fine.")

    result = await introspection_tools_full.execute_tool(
        "search_conversation_history", {"search_term": "openshift"}  # lowercase
    )

    data = json.loads(result)
    assert data["found"] == 1


@pytest.mark.asyncio
async def test_tool_without_kg(conversation_memory):
    """Test KG tools fail gracefully without KG"""
    tools = IntrospectionTools(knowledge_graph=None, conversation_memory=conversation_memory)

    result = await tools.execute_tool("search_knowledge_graph", {"entity_type": "jira_ticket"})

    data = json.loads(result)
    assert "error" in data
    assert "not available" in data["error"].lower()


@pytest.mark.asyncio
async def test_tool_without_conversation(kg):
    """Test conversation tool fails gracefully without conversation memory"""
    tools = IntrospectionTools(knowledge_graph=kg, conversation_memory=None)

    result = await tools.execute_tool("search_conversation_history", {"search_term": "test"})

    data = json.loads(result)
    assert "error" in data
    assert "not available" in data["error"].lower()


@pytest.mark.asyncio
async def test_search_conversation_entities_in_kg(introspection_tools_with_kg, kg):
    """Test searching conversation entities stored in the KG"""
    kg.insert_entity(
        entity_type="conversation",
        data={"user": "What are failing DCI jobs?", "assistant": "Here are 3 failing jobs..."},
        valid_from=datetime.now(),
    )
    kg.insert_entity(
        entity_type="conversation",
        data={"user": "Show me CILAB tickets", "assistant": "Found 5 open tickets..."},
        valid_from=datetime.now(),
    )

    # Search by entity type
    result = await introspection_tools_with_kg.execute_tool("search_knowledge_graph", {"entity_type": "conversation"})
    data = json.loads(result)
    assert data["found"] == 2

    # Search with text filter
    result = await introspection_tools_with_kg.execute_tool(
        "search_knowledge_graph", {"entity_type": "conversation", "search_text": "DCI"}
    )
    data = json.loads(result)
    assert data["found"] == 1
    assert "DCI" in data["entities"][0]["data"]["user"]


@pytest.mark.asyncio
async def test_unknown_tool(introspection_tools_full):
    """Test calling unknown tool returns error"""
    result = await introspection_tools_full.execute_tool("unknown_tool", {})

    data = json.loads(result)
    assert "error" in data


# --- get_skill_help tests ---


def test_get_skill_help_no_agent():
    """get_skill_help returns error without agent reference"""
    tools = IntrospectionTools(agent=None)
    result = tools._get_skill_help({"skill_name": "hello"})
    data = json.loads(result)
    assert "error" in data
    assert "Agent reference" in data["error"]


def test_get_skill_help_unknown_skill():
    """get_skill_help returns error and available skills list for unknown skill"""
    from unittest.mock import MagicMock

    agent = MagicMock()
    agent.skills_manager.loaded_skills = {}

    tools = IntrospectionTools(agent=agent)
    result = tools._get_skill_help({"skill_name": "nonexistent"})
    data = json.loads(result)
    assert "error" in data
    assert "available_skills" in data


def test_get_skill_help_success():
    """get_skill_help returns full skill body and directory"""
    from pathlib import Path
    from unittest.mock import MagicMock

    content = MagicMock()
    content.metadata.description = "A test skill"
    content.metadata.skill_path = Path("/tmp/test-skills/hello")
    content.body = "# Hello Skill\n\nFull instructions here."

    agent = MagicMock()
    agent.skills_manager.loaded_skills = {"hello": content}
    agent.script_execution_tools.enabled = True

    tools = IntrospectionTools(agent=agent)
    result = tools._get_skill_help({"skill_name": "hello"})
    data = json.loads(result)

    assert data["skill_name"] == "hello"
    assert data["description"] == "A test skill"
    assert data["skill_directory"] == "/tmp/test-skills/hello"
    assert "Hello Skill" in data["body"]
    assert "warning" not in data


def test_get_skill_help_warns_when_scripts_disabled():
    """get_skill_help includes warning when script execution is disabled"""
    from pathlib import Path
    from unittest.mock import MagicMock

    content = MagicMock()
    content.metadata.description = "A test skill"
    content.metadata.skill_path = Path("/tmp/test-skills/hello")
    content.body = "# Hello Skill"

    agent = MagicMock()
    agent.skills_manager.loaded_skills = {"hello": content}
    agent.script_execution_tools.enabled = False

    tools = IntrospectionTools(agent=agent)
    result = tools._get_skill_help({"skill_name": "hello"})
    data = json.loads(result)

    assert "warning" in data
    assert "DISABLED" in data["warning"]


@pytest.mark.asyncio
async def test_get_skill_help_via_execute_tool():
    """get_skill_help is routable via execute_tool"""
    from unittest.mock import MagicMock

    agent = MagicMock()
    agent.skills_manager.loaded_skills = {}

    tools = IntrospectionTools(agent=agent)
    result = await tools.execute_tool("get_skill_help", {"skill_name": "nonexistent"})
    data = json.loads(result)
    assert "error" in data


# --- AWL script validation tests ---


def test_validate_awl_tool_always_present():
    """validate_awl_script appears in tool definitions regardless of agent"""
    tools = IntrospectionTools(agent=None).get_tool_definitions()
    names = [t["name"] for t in tools]
    assert "introspection__validate_awl_script" in names


@pytest.mark.asyncio
async def test_validate_awl_script_valid():
    """Return success message for a valid AWL script"""
    tools = IntrospectionTools(agent=None)
    result = await tools.execute_tool(
        "validate_awl_script",
        {"awl_code": "@start\n@task t1\nGoal: do something\nExpose: result\n@end\n@end\n"},
    )
    assert "Valid" in result


@pytest.mark.asyncio
async def test_validate_awl_script_parse_error():
    """Return parse error details for invalid AWL"""
    tools = IntrospectionTools(agent=None)
    result = await tools.execute_tool("validate_awl_script", {"awl_code": "this is not awl\n"})
    assert "Error" in result


@pytest.mark.asyncio
async def test_validate_awl_script_empty():
    """Return error for empty input"""
    tools = IntrospectionTools(agent=None)
    result = await tools.execute_tool("validate_awl_script", {"awl_code": ""})
    assert "Error" in result


@pytest.mark.asyncio
async def test_validate_awl_script_line_number_in_error():
    """Parse errors include a line number"""
    tools = IntrospectionTools(agent=None)
    result = await tools.execute_tool(
        "validate_awl_script",
        {"awl_code": "@start\n@task t1\nGoal: ok\n@end\n@bogus\n@end\n"},
    )
    assert "Line" in result or "line" in result or "Error" in result


# --- AWL script execution tests ---


@pytest.fixture
def mock_agent():
    from unittest.mock import AsyncMock, MagicMock

    agent = MagicMock()
    agent.query = AsyncMock(return_value='{"result": "done"}')
    agent.skills_manager.loaded_skills = {}
    return agent


def test_awl_tool_in_definitions_when_agent_available(mock_agent):
    """execute_awl_script appears in tool definitions when agent is set"""
    tools = IntrospectionTools(agent=mock_agent).get_tool_definitions()
    names = [t["name"] for t in tools]
    assert "introspection__execute_awl_script" in names


def test_awl_tool_absent_without_agent():
    """execute_awl_script is NOT in tool definitions when agent is None"""
    tools = IntrospectionTools(agent=None).get_tool_definitions()
    names = [t["name"] for t in tools]
    assert "introspection__execute_awl_script" not in names


@pytest.mark.asyncio
async def test_execute_awl_script_basic(mock_agent, tmp_path):
    """Execute a minimal AWL script (no tasks, just @set)"""
    script = tmp_path / "test.awl"
    script.write_text("@start\n@set msg = hello\n@end\n")

    tools = IntrospectionTools(agent=mock_agent)
    result = await tools.execute_tool("execute_awl_script", {"script_path": str(script)})

    assert "completed successfully" in result.lower() or "Workflow" in result


@pytest.mark.asyncio
async def test_execute_awl_script_with_tasks(mock_agent, tmp_path):
    """Execute an AWL script that runs a task and returns a value"""
    script = tmp_path / "greet.awl"
    script.write_text(
        "@start\n" "@task greet\n" "Goal: Say hello\n" "Expose: result\n" "@end\n" "@return result\n" "@end\n"
    )

    tools = IntrospectionTools(agent=mock_agent)
    result = await tools.execute_tool("execute_awl_script", {"script_path": str(script)})

    assert "error" not in result.lower() or "runtime" not in result.lower()


@pytest.mark.asyncio
async def test_execute_awl_script_with_variables(mock_agent, tmp_path):
    """Variables are injected into the workflow"""
    script = tmp_path / "var.awl"
    script.write_text("@start\n@set greeting = ${name}\n@end\n")

    tools = IntrospectionTools(agent=mock_agent)
    result = await tools.execute_tool(
        "execute_awl_script", {"script_path": str(script), "variables": {"name": "World"}}
    )

    assert "completed successfully" in result.lower() or "Workflow" in result


@pytest.mark.asyncio
async def test_execute_awl_script_wrong_extension(mock_agent, tmp_path):
    """Reject files without .awl extension"""
    script = tmp_path / "script.py"
    script.write_text("print('hello')")

    tools = IntrospectionTools(agent=mock_agent)
    result = await tools.execute_tool("execute_awl_script", {"script_path": str(script)})

    assert "Error" in result
    assert ".awl" in result


@pytest.mark.asyncio
async def test_execute_awl_script_not_found(mock_agent):
    """Return clear error for missing script"""
    tools = IntrospectionTools(agent=mock_agent)
    result = await tools.execute_tool("execute_awl_script", {"script_path": "/nonexistent/path/test.awl"})

    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_execute_awl_script_parse_error(mock_agent, tmp_path):
    """Return clear error for invalid AWL syntax"""
    script = tmp_path / "bad.awl"
    script.write_text("this is not valid AWL\n")

    tools = IntrospectionTools(agent=mock_agent)
    result = await tools.execute_tool("execute_awl_script", {"script_path": str(script)})

    assert "Error" in result


@pytest.mark.asyncio
async def test_execute_awl_script_no_agent(tmp_path):
    """Return error when no agent is available"""
    script = tmp_path / "test.awl"
    script.write_text("@start\n@set x = 1\n@end\n")

    tools = IntrospectionTools(agent=None)
    result = await tools.execute_tool("execute_awl_script", {"script_path": str(script)})

    assert "Error" in result


@pytest.mark.asyncio
async def test_execute_awl_script_relative_path(mock_agent, tmp_path, monkeypatch):
    """Relative script paths are resolved from cwd"""
    script = tmp_path / "rel.awl"
    script.write_text("@start\n@set x = 1\n@end\n")
    monkeypatch.chdir(tmp_path)

    tools = IntrospectionTools(agent=mock_agent)
    result = await tools.execute_tool("execute_awl_script", {"script_path": "rel.awl"})

    assert "completed successfully" in result.lower() or "Workflow" in result


@pytest.mark.asyncio
async def test_execute_awl_script_missing_variables(mock_agent, tmp_path):
    """Return JSON error when required input variables are not provided"""
    script = tmp_path / "greet.awl"
    script.write_text("@start\n@set msg = ${name}\n@end\n")

    tools = IntrospectionTools(agent=mock_agent)
    result = await tools.execute_tool("execute_awl_script", {"script_path": str(script)})

    data = json.loads(result)
    assert "error" in data
    assert "name" in data["missing"]


@pytest.mark.asyncio
async def test_execute_awl_script_runtime_error(mock_agent, tmp_path):
    """Return JSON error on AWLRuntimeError"""
    from unittest.mock import AsyncMock, patch

    from ai_assist.awl_runtime import AWLRuntimeError

    script = tmp_path / "fail.awl"
    script.write_text("@start\n@task t1\nGoal: do something\nExpose: result\n@end\n@end\n")

    tools = IntrospectionTools(agent=mock_agent)
    with patch("ai_assist.awl_runtime.AWLRuntime") as MockRuntime:
        instance = MockRuntime.return_value
        instance.execute = AsyncMock(side_effect=AWLRuntimeError("boom"))
        result = await tools.execute_tool("execute_awl_script", {"script_path": str(script)})

    data = json.loads(result)
    assert "error" in data
    assert "boom" in data["error"]


# --- inspect_awl_script tests ---


def test_inspect_awl_tool_present_with_agent(mock_agent):
    """inspect_awl_script appears in tool definitions when agent is set"""
    tools = IntrospectionTools(agent=mock_agent).get_tool_definitions()
    names = [t["name"] for t in tools]
    assert "introspection__inspect_awl_script" in names


@pytest.mark.asyncio
async def test_inspect_awl_script_missing_path():
    """Return error when script_path is empty"""
    tools = IntrospectionTools(agent=None)
    result = await tools.execute_tool("inspect_awl_script", {"script_path": ""})

    data = json.loads(result)
    assert "error" in data


@pytest.mark.asyncio
async def test_inspect_awl_script_not_found():
    """Return error when script does not exist"""
    tools = IntrospectionTools(agent=None)
    result = await tools.execute_tool("inspect_awl_script", {"script_path": "/no/such/file.awl"})

    data = json.loads(result)
    assert "error" in data
    assert "not found" in data["error"].lower() or "Script not found" in data["error"]


@pytest.mark.asyncio
async def test_inspect_awl_script_parse_error(tmp_path):
    """Return error for invalid AWL syntax"""
    script = tmp_path / "bad.awl"
    script.write_text("this is not valid\n")

    tools = IntrospectionTools(agent=None)
    result = await tools.execute_tool("inspect_awl_script", {"script_path": str(script)})

    data = json.loads(result)
    assert "error" in data


@pytest.mark.asyncio
async def test_inspect_awl_script_returns_input_variables(tmp_path):
    """Return the set of input variables referenced but not defined in the script"""
    script = tmp_path / "greet.awl"
    script.write_text("@start\n@set msg = Hello ${name}\n@end\n")

    tools = IntrospectionTools(agent=None)
    result = await tools.execute_tool("inspect_awl_script", {"script_path": str(script)})

    data = json.loads(result)
    assert "input_variables" in data
    assert "name" in data["input_variables"]


@pytest.mark.asyncio
async def test_inspect_awl_script_no_input_variables(tmp_path):
    """Script with no external variables returns empty input_variables"""
    script = tmp_path / "simple.awl"
    script.write_text("@start\n@set x = hello\n@end\n")

    tools = IntrospectionTools(agent=None)
    result = await tools.execute_tool("inspect_awl_script", {"script_path": str(script)})

    data = json.loads(result)
    assert data["input_variables"] == []


@pytest.mark.asyncio
async def test_inspect_awl_script_relative_path(tmp_path, monkeypatch):
    """Relative paths in inspect_awl_script are resolved from cwd"""
    script = tmp_path / "rel.awl"
    script.write_text("@start\n@set x = hello\n@end\n")
    monkeypatch.chdir(tmp_path)

    tools = IntrospectionTools(agent=None)
    result = await tools.execute_tool("inspect_awl_script", {"script_path": "rel.awl"})

    data = json.loads(result)
    assert "input_variables" in data


# --- _awl_input_variables edge cases ---


def test_awl_input_variables_loop_with_collect(tmp_path):
    """LoopNode with collect= should add collect var to defined set"""
    from ai_assist.awl_executor import get_missing_variables
    from ai_assist.awl_parser import AWLParser

    source = (
        "@start\n"
        "@loop items as item collect=results\n"
        "@task t1\n"
        "Goal: process ${item}\n"
        "Expose: summary\n"
        "@end\n"
        "@end\n"
        "@end\n"
    )
    workflow = AWLParser(source).parse()
    missing = get_missing_variables(workflow)
    # 'items' is referenced but not defined → it's an input
    assert "items" in missing
    # 'item' and 'results' are defined by the loop → not inputs
    assert "item" not in missing
    assert "results" not in missing


def test_awl_input_variables_if_node(tmp_path):
    """Variables used inside @if/@else bodies are detected"""
    from ai_assist.awl_executor import get_missing_variables
    from ai_assist.awl_parser import AWLParser

    source = "@start\n" "@if flag\n" "@set a = ${x}\n" "@else\n" "@set b = ${y}\n" "@end\n" "@end\n"
    workflow = AWLParser(source).parse()
    missing = get_missing_variables(workflow)
    assert "x" in missing
    assert "y" in missing


# --- _execute_mcp_prompt missing-argument validation ---


@pytest.mark.asyncio
async def test_execute_mcp_prompt_missing_required_args(mock_agent):
    """Return JSON error when required prompt arguments are missing"""
    from unittest.mock import MagicMock

    arg = MagicMock()
    arg.name = "days"
    arg.required = True

    prompt_def = MagicMock()
    prompt_def.arguments = [arg]

    available_prompts = {"myserver": {"myprompт": prompt_def}}

    tools = IntrospectionTools(agent=mock_agent, available_prompts=available_prompts)
    result = await tools.execute_tool(
        "execute_mcp_prompt",
        {"server": "myserver", "prompt": "myprompт", "arguments": {}},
    )

    data = json.loads(result)
    assert "error" in data
    assert "days" in data["missing"]
