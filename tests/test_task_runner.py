"""Tests for task runner"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_assist.event_sources import EventContext
from ai_assist.state import StateManager
from ai_assist.task_runner import TaskResult, TaskRunner
from ai_assist.tasks import TaskDefinition


@pytest.fixture
def temp_state_dir(tmp_path):
    """Create temporary state directory"""
    return tmp_path / "state"


@pytest.fixture
def state_manager(temp_state_dir):
    """Create state manager with temp directory"""
    return StateManager(state_dir=temp_state_dir)


@pytest.fixture
def mock_agent():
    """Create mock agent"""
    agent = MagicMock()
    agent.query = AsyncMock()
    return agent


@pytest.fixture
def sample_task():
    """Create sample task definition"""
    return TaskDefinition(
        name="Test Task", prompt="Find failures in the system", interval="5m", description="A test task"
    )


@pytest.mark.asyncio
async def test_task_runner_success(mock_agent, state_manager, sample_task):
    """Test successful task execution"""
    mock_agent.query.return_value = "Found 5 failures in the last hour"

    runner = TaskRunner(sample_task, mock_agent, state_manager)
    result = await runner.run()

    assert result.success is True
    assert result.task_name == "Test Task"
    assert result.output == "Found 5 failures in the last hour"
    assert result.timestamp is not None
    assert mock_agent.query.called
    assert mock_agent.query.call_args[0][0] == "Find failures in the system"


@pytest.mark.asyncio
async def test_task_runner_failure(mock_agent, state_manager, sample_task):
    """Test task execution with error"""
    mock_agent.query.side_effect = Exception("API Error")

    with patch("ai_assist.task_runner.NotificationDispatcher") as mock_cls:
        mock_cls.return_value.dispatch = AsyncMock()
        runner = TaskRunner(sample_task, mock_agent, state_manager)
        result = await runner.run()

    assert result.success is False
    assert result.task_name == "Test Task"
    assert "API Error" in result.output
    assert result.timestamp is not None


@pytest.mark.asyncio
async def test_task_runner_state_tracking(mock_agent, state_manager, sample_task):
    """Test that task execution is tracked in state"""
    mock_agent.query.return_value = "Task completed successfully"

    runner = TaskRunner(sample_task, mock_agent, state_manager)
    await runner.run()

    # Check state was updated
    state = state_manager.get_monitor_state(runner.state_key)
    assert state.last_check is not None
    assert state.last_results["task_name"] == "Test Task"
    assert state.last_results["last_success"] is True


@pytest.mark.asyncio
async def test_task_runner_history(mock_agent, state_manager, sample_task):
    """Test that task execution is saved to history"""
    mock_agent.query.return_value = "Result 1"

    runner = TaskRunner(sample_task, mock_agent, state_manager)

    # Run task multiple times
    await runner.run()
    mock_agent.query.return_value = "Result 2"
    await runner.run()

    # Check history
    history = runner.get_history(limit=10)
    assert len(history) == 2
    assert history[0]["result"]["task_name"] == "Test Task"
    assert history[0]["result"]["success"] is True


@pytest.mark.asyncio
async def test_task_runner_state_key_sanitization(mock_agent, state_manager):
    """Test that task names are sanitized for state keys"""
    task = TaskDefinition(name="Test Task / With Special! Chars", prompt="Test", interval="5m")

    runner = TaskRunner(task, mock_agent, state_manager)
    state_key = runner._get_state_key()

    # Should only contain alphanumeric, dash, and underscore
    assert state_key == "task_Test_Task___With_Special__Chars"
    assert all(c.isalnum() or c in "-_" for c in state_key)


@pytest.mark.asyncio
async def test_task_runner_get_last_run(mock_agent, state_manager, sample_task):
    """Test getting last run timestamp"""
    mock_agent.query.return_value = "Test output"

    runner = TaskRunner(sample_task, mock_agent, state_manager)

    # Initially no last run
    assert runner.get_last_run() is None

    # After running, should have last run time
    await runner.run()
    last_run = runner.get_last_run()
    assert last_run is not None


@pytest.mark.asyncio
async def test_task_result_creation():
    """Test TaskResult dataclass creation"""
    from datetime import datetime

    result = TaskResult(
        task_name="Test", success=True, output="Test output", timestamp=datetime.now(), metadata={"count": 5}
    )

    assert result.task_name == "Test"
    assert result.success is True
    assert result.output == "Test output"
    assert result.metadata["count"] == 5


# Phase 4: MCP Prompt Task Execution Tests


@pytest.mark.asyncio
async def test_run_mcp_prompt_task(mock_agent, state_manager):
    """Test executing task with MCP prompt"""
    task = TaskDefinition(
        name="DCI RCA", prompt="mcp://dci/rca", interval="8:00 on weekdays", prompt_arguments={"days": "1"}
    )

    # Mock execute_mcp_prompt instead of query
    mock_agent.execute_mcp_prompt = AsyncMock(return_value="RCA analysis completed")

    runner = TaskRunner(task, mock_agent, state_manager)
    result = await runner.run()

    assert result.success is True
    assert result.output == "RCA analysis completed"
    mock_agent.execute_mcp_prompt.assert_called_once_with("dci", "rca", {"days": "1"}, max_turns=100)
    mock_agent.query.assert_not_called()


@pytest.mark.asyncio
async def test_run_mcp_prompt_without_arguments(mock_agent, state_manager):
    """Test executing MCP prompt task without arguments"""
    task = TaskDefinition(name="Simple Prompt", prompt="mcp://server/prompt", interval="1h")

    mock_agent.execute_mcp_prompt = AsyncMock(return_value="Prompt result")

    runner = TaskRunner(task, mock_agent, state_manager)
    result = await runner.run()

    assert result.success is True
    mock_agent.execute_mcp_prompt.assert_called_once_with("server", "prompt", None, max_turns=100)


@pytest.mark.asyncio
async def test_run_natural_language_task_unchanged(mock_agent, state_manager):
    """Test that natural language tasks still work after MCP changes"""
    task = TaskDefinition(name="Natural Task", prompt="Find failures in the system", interval="1h")

    mock_agent.query.return_value = "Found 3 failures"

    runner = TaskRunner(task, mock_agent, state_manager)
    result = await runner.run()

    assert result.success is True
    assert result.output == "Found 3 failures"
    mock_agent.query.assert_called_once_with("Find failures in the system", max_turns=100)


@pytest.mark.asyncio
async def test_run_mcp_prompt_invalid_server(mock_agent, state_manager):
    """Test handling of invalid MCP server"""
    task = TaskDefinition(name="Bad Server", prompt="mcp://invalid/prompt", interval="1h")

    mock_agent.execute_mcp_prompt = AsyncMock(side_effect=ValueError("MCP server 'invalid' not connected"))

    with patch("ai_assist.task_runner.NotificationDispatcher") as mock_cls:
        mock_cls.return_value.dispatch = AsyncMock()
        runner = TaskRunner(task, mock_agent, state_manager)
        result = await runner.run()

    assert result.success is False
    assert "MCP server 'invalid' not connected" in result.output


@pytest.mark.asyncio
async def test_run_mcp_prompt_invalid_arguments(mock_agent, state_manager):
    """Test handling of invalid MCP prompt arguments"""
    task = TaskDefinition(
        name="Bad Args", prompt="mcp://dci/rca", interval="1h", prompt_arguments={}  # Missing required args
    )

    mock_agent.execute_mcp_prompt = AsyncMock(side_effect=ValueError("Required argument 'days' missing"))

    with patch("ai_assist.task_runner.NotificationDispatcher") as mock_cls:
        mock_cls.return_value.dispatch = AsyncMock()
        runner = TaskRunner(task, mock_agent, state_manager)
        result = await runner.run()

    assert result.success is False
    assert "Required argument 'days' missing" in result.output


# Phase 0: Tests for Monitor/Task Unification


@pytest.mark.asyncio
async def test_task_runner_handles_monitors(mock_agent, state_manager):
    """TaskRunner should work with former monitor definitions"""
    task_def = TaskDefinition(
        name="test-monitor",
        interval="30m",
        prompt="Check DCI jobs",
    )

    mock_agent.query.return_value = "Found 3 failing jobs"

    runner = TaskRunner(task_def, mock_agent, state_manager)
    result = await runner.run()

    assert result.success is True
    assert result.output == "Found 3 failing jobs"


@pytest.mark.asyncio
async def test_notification_truncates_at_500_chars(mock_agent, state_manager):
    """Notifications should truncate at 500 chars consistently"""

    long_output = "x" * 1000
    task_def = TaskDefinition(name="test", interval="5m", prompt="test", notify=True)

    mock_agent.query.return_value = long_output

    with patch("ai_assist.task_runner.NotificationDispatcher") as mock_dispatcher_class:
        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch = AsyncMock(return_value={"console": True})
        mock_dispatcher_class.return_value = mock_dispatcher

        runner = TaskRunner(task_def, mock_agent, state_manager)
        await runner.run()

        # Notification should be truncated to 500 chars
        notification = mock_dispatcher.dispatch.call_args[0][0]
        assert len(notification.message) == 500


@pytest.mark.asyncio
async def test_failure_always_notifies(mock_agent, state_manager):
    """Task failures should always send a desktop+console notification, even without notify=True"""
    task_def = TaskDefinition(name="failing-task", interval="5m", prompt="test", notify=False)

    mock_agent.query.side_effect = Exception("Connection timeout")

    with patch("ai_assist.execution_helpers.NotificationDispatcher") as mock_dispatcher_class:
        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch = AsyncMock(return_value={"desktop": True, "console": True})
        mock_dispatcher_class.return_value = mock_dispatcher

        runner = TaskRunner(task_def, mock_agent, state_manager)
        result = await runner.run()

        assert result.success is False
        mock_dispatcher.dispatch.assert_called_once()
        notification = mock_dispatcher.dispatch.call_args[0][0]
        assert notification.level == "error"
        assert "failing-task" in notification.title
        assert "Connection timeout" in notification.message
        assert "desktop" in notification.channels
        assert "console" in notification.channels


@pytest.mark.asyncio
async def test_failure_notification_truncates_long_errors(mock_agent, state_manager):
    """Failure notification should truncate long error messages at 500 chars"""
    task_def = TaskDefinition(name="long-error-task", interval="5m", prompt="test")

    mock_agent.query.side_effect = Exception("E" * 1000)

    with patch("ai_assist.execution_helpers.NotificationDispatcher") as mock_dispatcher_class:
        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch = AsyncMock(return_value={"desktop": True, "console": True})
        mock_dispatcher_class.return_value = mock_dispatcher

        runner = TaskRunner(task_def, mock_agent, state_manager)
        await runner.run()

        notification = mock_dispatcher.dispatch.call_args[0][0]
        assert len(notification.message) == 500


@pytest.mark.asyncio
async def test_builtin_prompt_routes_to_synthesis(mock_agent, state_manager):
    """__builtin__:kg_synthesis prompt should call _run_synthesis_from_kg"""
    task = TaskDefinition(
        name="kg-synthesis",
        prompt="__builtin__:kg_synthesis",
        interval="night on weekdays",
    )

    mock_agent._run_synthesis_from_kg = AsyncMock(return_value="Synthesized 3 insights")

    runner = TaskRunner(task, mock_agent, state_manager)
    result = await runner.run()

    assert result.success is True
    assert result.output == "Synthesized 3 insights"
    mock_agent._run_synthesis_from_kg.assert_called_once()
    mock_agent.query.assert_not_called()


@pytest.mark.asyncio
async def test_awl_script_not_found_reports_failure(mock_agent, state_manager):
    """AWL script that doesn't exist should report failure, not success"""
    task = TaskDefinition(
        name="missing-awl",
        prompt="/nonexistent/path/script.awl",
        interval="15:00 on weekdays",
    )

    with patch("ai_assist.task_runner.NotificationDispatcher") as mock_cls:
        mock_cls.return_value.dispatch = AsyncMock()
        runner = TaskRunner(task, mock_agent, state_manager)
        result = await runner.run()

    assert result.success is False
    assert "not found" in result.output.lower() or "does not exist" in result.output.lower()

    # Verify state was recorded as failure
    state = state_manager.get_monitor_state(runner.state_key)
    assert state.last_results["last_success"] is False


@pytest.mark.asyncio
async def test_awl_start_workflow_runs_via_runtime(mock_agent, state_manager, tmp_path):
    """AWL scripts with @start (not @goal) should use AWLRuntime, not GoalRunner"""
    awl_file = tmp_path / "workflow.awl"
    awl_file.write_text("@start\n@task t1\nGoal: Do something.\nExpose: result\n@end\n@end\n")

    task = TaskDefinition(
        name="workflow-task",
        prompt=str(awl_file),
        interval="15:00 on weekdays",
    )

    mock_agent.query = AsyncMock(return_value='{"result": "done"}')

    runner = TaskRunner(task, mock_agent, state_manager)
    result = await runner.run()

    assert result.success is True
    mock_agent.query.assert_called_once()


# Event context tests


def _make_event_context(**kwargs) -> EventContext:
    defaults = {
        "source_type": "mqtt",
        "event_type": "message",
        "payload": "CPU usage 95%",
        "metadata": {"topic": "alerts/cpu", "qos": 0},
        "timestamp": datetime(2026, 5, 5, 12, 0, 0),
    }
    defaults.update(kwargs)
    return EventContext(**defaults)


@pytest.mark.asyncio
async def test_run_with_event_context_substitutes_vars(mock_agent, state_manager):
    """Event context variables should be substituted in prompt"""
    task_def = TaskDefinition(
        name="MQTT Handler",
        prompt="Analyze alert: ${event.payload} from ${event.topic}",
        trigger={"type": "mqtt", "topic": "alerts/#"},
        notify=False,
    )

    mock_agent.query.return_value = "Alert analyzed"
    event = _make_event_context()

    runner = TaskRunner(task_def, mock_agent, state_manager)
    result = await runner.run(event_context=event)

    assert result.success is True
    called_prompt = mock_agent.query.call_args[0][0]
    assert "CPU usage 95%" in called_prompt
    assert "alerts/cpu" in called_prompt
    assert "${event.payload}" not in called_prompt


@pytest.mark.asyncio
async def test_run_without_event_context_backward_compat(mock_agent, state_manager):
    """run() without event_context should work unchanged"""
    task_def = TaskDefinition(
        name="Timer Task",
        prompt="Check DCI failures",
        interval="5m",
    )

    mock_agent.query.return_value = "No failures"

    runner = TaskRunner(task_def, mock_agent, state_manager)
    result = await runner.run()

    assert result.success is True
    assert result.output == "No failures"


@pytest.mark.asyncio
async def test_run_with_event_context_records_source_in_history(mock_agent, state_manager):
    """Event source info should be recorded in state history"""
    task_def = TaskDefinition(
        name="MQTT Task",
        prompt="Handle: ${event.payload}",
        trigger={"type": "mqtt", "topic": "t"},
        notify=False,
    )

    mock_agent.query.return_value = "Done"
    event = _make_event_context()

    runner = TaskRunner(task_def, mock_agent, state_manager)
    await runner.run(event_context=event)

    history = runner.get_history(limit=1)
    assert len(history) == 1
    result_data = history[0].get("result", history[0])
    assert result_data.get("event_source") == "mqtt"


@pytest.mark.asyncio
async def test_substitute_event_vars_with_source_type(mock_agent, state_manager):
    """${event.source_type} should be substituted"""
    task_def = TaskDefinition(
        name="Test",
        prompt="Source: ${event.source_type}",
        trigger={"type": "dbus"},
        notify=False,
    )

    mock_agent.query.return_value = "ok"
    event = _make_event_context(source_type="dbus")

    runner = TaskRunner(task_def, mock_agent, state_manager)
    await runner.run(event_context=event)

    called_prompt = mock_agent.query.call_args[0][0]
    assert "Source: dbus" in called_prompt
