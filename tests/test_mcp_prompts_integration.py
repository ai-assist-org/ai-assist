"""Integration tests for MCP prompt execution in tasks"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig
from ai_assist.schedule_loader import ScheduleLoader
from ai_assist.state import StateManager
from ai_assist.task_runner import TaskRunner


@pytest.fixture
def temp_state_dir(tmp_path):
    """Create temporary state directory"""
    return tmp_path / "state"


@pytest.fixture
def temp_schedules_file(tmp_path):
    """Create temporary schedules file"""
    return tmp_path / "schedules.json"


@pytest.fixture
def state_manager(temp_state_dir):
    """Create state manager"""
    return StateManager(state_dir=temp_state_dir)


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
def agent_with_mcp_prompt(mock_config):
    """Create agent with mock MCP prompt"""
    agent = AiAssistAgent(mock_config)

    # Setup mock MCP server and prompt
    mock_session = MagicMock()
    agent.sessions["dci"] = mock_session

    # Create mock prompt definition
    mock_arg = MagicMock()
    mock_arg.name = "days"
    mock_arg.required = True
    mock_arg.description = "Number of days"

    mock_prompt = MagicMock()
    mock_prompt.name = "rca"
    mock_prompt.arguments = [mock_arg]

    agent.available_prompts["dci"] = {"rca": mock_prompt}

    # Mock get_prompt to return test result
    mock_message = MagicMock()
    mock_message.role = "user"
    mock_message.content = MagicMock()
    mock_message.content.text = "RCA analysis: Found 5 failures"

    mock_result = MagicMock()
    mock_result.messages = [mock_message]

    mock_session.get_prompt = AsyncMock(return_value=mock_result)

    # Mock query_streaming since execute_mcp_prompt uses streaming
    async def mock_streaming(**kwargs):
        yield "RCA analysis: Found 5 failures"
        yield {"type": "done", "turns": 1}

    agent.query_streaming = mock_streaming

    return agent


@pytest.mark.asyncio
async def test_mcp_prompt_task_end_to_end(temp_schedules_file, agent_with_mcp_prompt, state_manager):
    """Test complete flow: load schedule with MCP prompt → execute task → verify result"""

    # Create schedules.json with MCP prompt task
    schedules = {
        "version": "1.0",
        "tasks": [
            {
                "name": "Daily RCA",
                "prompt": "mcp://dci/rca",
                "prompt_arguments": {"days": "1"},
                "interval": "1h",
                "enabled": True,
            }
        ],
    }

    with open(temp_schedules_file, "w") as f:
        json.dump(schedules, f)

    # Load schedules
    loader = ScheduleLoader(temp_schedules_file)
    tasks = loader.load_tasks()

    assert len(tasks) == 1
    task_def = tasks[0]

    # Verify task is correctly loaded
    assert task_def.name == "Daily RCA"
    assert task_def.prompt == "mcp://dci/rca"
    assert task_def.prompt_arguments == {"days": "1"}
    assert task_def.is_mcp_prompt is True

    # Execute task
    runner = TaskRunner(task_def, agent_with_mcp_prompt, state_manager)
    result = await runner.run()

    # Verify execution
    assert result.success is True
    assert "RCA analysis: Found 5 failures" in result.output

    # Verify MCP prompt was called correctly
    agent_with_mcp_prompt.sessions["dci"].get_prompt.assert_called_once_with("rca", arguments={"days": "1"})


@pytest.mark.asyncio
async def test_mixed_tasks_execution(temp_schedules_file, agent_with_mcp_prompt, state_manager):
    """Test that MCP prompt tasks and natural language tasks can coexist"""

    # Create schedules with both types
    schedules = {
        "version": "1.0",
        "tasks": [
            {
                "name": "MCP Task",
                "prompt": "mcp://dci/rca",
                "prompt_arguments": {"days": "1"},
                "interval": "1h",
            },
            {
                "name": "Natural Task",
                "prompt": "Find system failures",
                "interval": "1h",
            },
        ],
    }

    with open(temp_schedules_file, "w") as f:
        json.dump(schedules, f)

    # Load and verify both tasks
    loader = ScheduleLoader(temp_schedules_file)
    tasks = loader.load_tasks()

    assert len(tasks) == 2

    # Verify MCP task
    mcp_task = tasks[0]
    assert mcp_task.is_mcp_prompt is True
    assert mcp_task.parse_mcp_prompt() == ("dci", "rca")

    # Verify natural language task
    natural_task = tasks[1]
    assert natural_task.is_mcp_prompt is False

    # Mock query_streaming - called by execute_mcp_prompt for MCP task
    async def mock_streaming(**kwargs):
        yield "Found 3 failures"
        yield {"type": "done", "turns": 1}

    agent_with_mcp_prompt.query_streaming = mock_streaming

    # Mock query() - called directly for natural language task
    agent_with_mcp_prompt.query = AsyncMock(return_value="Found 3 failures")

    # Execute both tasks
    runner1 = TaskRunner(mcp_task, agent_with_mcp_prompt, state_manager)
    result1 = await runner1.run()
    assert result1.success is True

    runner2 = TaskRunner(natural_task, agent_with_mcp_prompt, state_manager)
    result2 = await runner2.run()
    assert result2.success is True

    # Verify correct execution paths were used
    agent_with_mcp_prompt.sessions["dci"].get_prompt.assert_called_once()
    # query() called once for natural language task only
    assert agent_with_mcp_prompt.query.call_count == 1


@pytest.mark.asyncio
async def test_schedule_validation_rejects_invalid_mcp_prompts(temp_schedules_file):
    """Test that invalid MCP prompt formats are rejected during load"""

    # Create schedules with invalid MCP prompt
    schedules = {
        "version": "1.0",
        "tasks": [
            {
                "name": "Invalid Task",
                "prompt": "mcp://invalid",  # Missing /prompt part
                "interval": "1h",
            }
        ],
    }

    with open(temp_schedules_file, "w") as f:
        json.dump(schedules, f)

    # Load should skip invalid task with warning
    loader = ScheduleLoader(temp_schedules_file)
    tasks = loader.load_tasks()

    # Task should be skipped (validation fails)
    assert len(tasks) == 0


@pytest.mark.asyncio
async def test_backward_compatibility_with_existing_schedules(temp_schedules_file, state_manager):
    """Test that existing natural language schedules continue to work"""

    # Create traditional schedules
    schedules = {
        "version": "1.0",
        "tasks": [
            {
                "name": "Legacy Task 1",
                "prompt": "Check system health",
                "interval": "5m",
            },
            {
                "name": "Legacy Task 2",
                "prompt": "Find errors in logs",
                "interval": "10m",
                "description": "Error monitoring",
            },
        ],
    }

    with open(temp_schedules_file, "w") as f:
        json.dump(schedules, f)

    # Load should work exactly as before
    loader = ScheduleLoader(temp_schedules_file)
    tasks = loader.load_tasks()

    assert len(tasks) == 2

    # Verify both are natural language tasks
    for task in tasks:
        assert task.is_mcp_prompt is False
        assert task.prompt_arguments is None
