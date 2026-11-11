"""Tests for background task spawning in interactive mode"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_assist.background_task_tools import BackgroundTaskTools
from ai_assist.background_tasks import BackgroundTask, BackgroundTaskManager


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.query = AsyncMock(return_value="Task completed successfully")
    agent._background_task_count = 0
    return agent


@pytest.fixture
def mock_console():
    return MagicMock()


@pytest.fixture
def manager(mock_agent, mock_console, tmp_path):
    return BackgroundTaskManager(
        agent=mock_agent,
        console=mock_console,
        notifications_file=tmp_path / "notifications.jsonl",
    )


@pytest.fixture
def tools(manager):
    return BackgroundTaskTools(manager)


class TestBackgroundTask:
    def test_create_defaults(self):
        task = BackgroundTask(
            description="Test task",
            prompt="Do something",
        )
        assert task.status == "pending"
        assert task.id
        assert task.result is None
        assert task.error is None

    def test_status_values(self):
        task = BackgroundTask(description="Test", prompt="p")
        assert task.status == "pending"
        task.status = "running"
        task.status = "completed"
        task.status = "failed"


class TestBackgroundTaskManager:
    @pytest.mark.asyncio
    async def test_spawn_task_creates_and_tracks(self, manager, mock_agent):
        task = await manager.spawn_task(
            prompt="Analyze failures",
            description="Check DCI",
        )
        assert task.id in manager._tasks
        assert task.status == "pending"
        assert task.prompt == "Analyze failures"
        assert task.description == "Check DCI"

    @pytest.mark.asyncio
    async def test_spawn_task_runs_in_background(self, manager, mock_agent):
        mock_agent.query = AsyncMock(return_value="Done")
        task = await manager.spawn_task(prompt="Test query", description="Test")

        # Let the background task run
        await asyncio.sleep(0.1)

        updated = manager.get_task(task.id)
        assert updated is not None
        assert updated.status == "completed"
        assert updated.result == "Done"

    @pytest.mark.asyncio
    async def test_spawn_task_prefixes_no_kg_no_history(self, manager, mock_agent):
        mock_agent.query = AsyncMock(return_value="Done")
        await manager.spawn_task(prompt="Test query", description="Test")
        await asyncio.sleep(0.1)

        called_prompt = mock_agent.query.call_args[0][0]
        assert called_prompt.startswith("@no-kg @no-history ")

    @pytest.mark.asyncio
    async def test_background_agent_is_told_it_is_the_background_task(self, manager, mock_agent):
        """Otherwise it reports "I did it directly, not in the background" and confuses the main agent."""
        mock_agent.query = AsyncMock(return_value="Done")
        task = await manager.spawn_task(prompt="Test the background pipeline", description="Pipeline test")
        await asyncio.sleep(0.1)

        called_prompt = mock_agent.query.call_args[0][0]
        assert called_prompt.startswith("@no-kg @no-history ")
        assert called_prompt.endswith("Test the background pipeline")
        assert task.id in called_prompt
        assert "Pipeline test" in called_prompt
        assert "background" in called_prompt.lower()
        assert "main conversation" in called_prompt

    @pytest.mark.asyncio
    async def test_spawn_task_increments_background_count(self, manager, mock_agent):
        count_during_query = None

        async def capture_count(prompt, **kwargs):
            nonlocal count_during_query
            count_during_query = mock_agent._background_task_count
            return "Done"

        mock_agent.query = capture_count
        await manager.spawn_task(prompt="Test", description="Test")
        await asyncio.sleep(0.1)

        assert count_during_query == 1
        assert mock_agent._background_task_count == 0

    @pytest.mark.asyncio
    async def test_spawn_task_failure(self, manager, mock_agent):
        mock_agent.query = AsyncMock(side_effect=RuntimeError("API error"))
        task = await manager.spawn_task(prompt="Failing query", description="Fail")
        await asyncio.sleep(0.1)

        updated = manager.get_task(task.id)
        assert updated is not None
        assert updated.status == "failed"
        assert "API error" in updated.error

    @pytest.mark.asyncio
    async def test_spawn_task_writes_notification(self, manager, mock_agent, tmp_path):
        mock_agent.query = AsyncMock(return_value="Analysis complete")
        await manager.spawn_task(prompt="Analyze", description="Test notify")
        await asyncio.sleep(0.1)

        notification_file = tmp_path / "notifications.jsonl"
        assert notification_file.exists()
        content = notification_file.read_text()
        assert "Test notify" in content

    @pytest.mark.asyncio
    async def test_spawn_task_with_save_to_report(self, manager, mock_agent):
        mock_agent.query = AsyncMock(return_value="Report content")
        task = await manager.spawn_task(
            prompt="Generate report",
            description="Report",
            save_to_report="bg-report",
        )
        await asyncio.sleep(0.1)

        updated = manager.get_task(task.id)
        assert updated is not None
        assert updated.result_report == "bg-report"

    def test_list_tasks_empty(self, manager):
        assert manager.list_tasks() == []

    @pytest.mark.asyncio
    async def test_list_tasks_with_filter(self, manager, mock_agent):
        mock_agent.query = AsyncMock(return_value="Done")
        await manager.spawn_task(prompt="Task 1", description="First")
        await manager.spawn_task(prompt="Task 2", description="Second")
        await asyncio.sleep(0.1)

        completed = manager.list_tasks(status_filter="completed")
        assert len(completed) == 2

        pending = manager.list_tasks(status_filter="pending")
        assert len(pending) == 0

    def test_get_task_not_found(self, manager):
        assert manager.get_task("nonexistent") is None

    @pytest.mark.asyncio
    async def test_cancel_running_task(self, manager, mock_agent):
        async def slow_query(prompt, **kwargs):
            await asyncio.sleep(10)
            return "Done"

        mock_agent.query = slow_query
        task = await manager.spawn_task(prompt="Slow task", description="Slow")
        await asyncio.sleep(0.05)

        cancelled = await manager.cancel_task(task.id)
        assert cancelled is True

        updated = manager.get_task(task.id)
        assert updated is not None
        assert updated.status == "failed"
        assert "cancelled" in updated.error.lower()

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task(self, manager):
        cancelled = await manager.cancel_task("no-such-id")
        assert cancelled is False

    @pytest.mark.asyncio
    async def test_cancel_all(self, manager, mock_agent):
        async def slow_query(prompt, **kwargs):
            await asyncio.sleep(10)
            return "Done"

        mock_agent.query = slow_query
        await manager.spawn_task(prompt="Slow 1", description="S1")
        await manager.spawn_task(prompt="Slow 2", description="S2")
        await asyncio.sleep(0.05)

        cancelled = await manager.cancel_all()
        assert cancelled == 2

    @pytest.mark.asyncio
    async def test_cleanup_old_tasks(self, manager, mock_agent):
        mock_agent.query = AsyncMock(return_value="Done")
        task = await manager.spawn_task(prompt="Old task", description="Old")
        await asyncio.sleep(0.1)

        # Backdate the task
        manager._tasks[task.id].created_at = datetime.now() - timedelta(hours=25)

        manager.cleanup_old_tasks(max_age_hours=24)
        assert task.id not in manager._tasks


class TestCompletionDelivery:
    """A finished task is handed to the agent once, and can wake it up."""

    async def _finish(self, manager, **spawn):
        task = await manager.spawn_task(prompt="research", description="Source scan", **spawn)
        running = manager._asyncio_tasks.get(task.id)
        if running is not None:
            await running
        return task

    @pytest.mark.asyncio
    async def test_finished_task_note_is_delivered_once(self, manager, mock_agent):
        mock_agent.query = AsyncMock(return_value="Found 12 sources.")
        task = await self._finish(manager)

        assert manager.has_finished()
        note = manager.take_finished_note()
        assert task.id in note
        assert "Source scan" in note
        assert "completed" in note
        assert "Found 12 sources." in note
        assert not manager.has_finished()
        assert manager.take_finished_note() is None

    @pytest.mark.asyncio
    async def test_failed_task_note_has_the_error(self, manager, mock_agent):
        mock_agent.query = AsyncMock(side_effect=RuntimeError("API down"))
        await self._finish(manager)

        note = manager.take_finished_note()
        assert "failed" in note
        assert "API down" in note

    @pytest.mark.asyncio
    async def test_note_points_to_the_full_result(self, manager, mock_agent):
        mock_agent.query = AsyncMock(return_value="x" * 10_000)
        task = await self._finish(manager)

        note = manager.take_finished_note()
        assert len(note) < 5_000
        assert f'internal__get_background_task(task_id="{task.id}")' in note

    @pytest.mark.asyncio
    async def test_finishing_calls_the_wake_hook(self, manager):
        woken = []
        manager.on_finished = lambda: woken.append(manager.has_finished())
        await self._finish(manager)
        assert woken == [True]

    @pytest.mark.asyncio
    async def test_tasks_not_for_the_agent_do_not_wake_it(self, manager):
        """E.g. MCP channel messages: they reply through their own tool."""
        woken = []
        manager.on_finished = lambda: woken.append(True)
        await self._finish(manager, notify_agent=False)
        assert woken == []
        assert not manager.has_finished()

    @pytest.mark.asyncio
    async def test_cancelled_task_is_not_reported_back(self, manager, mock_agent):
        """The agent cancelled it itself: no need to tell it."""

        async def slow(*args, **kwargs):
            await asyncio.sleep(10)

        mock_agent.query = AsyncMock(side_effect=slow)
        task = await manager.spawn_task(prompt="p", description="d")
        await asyncio.sleep(0)
        await manager.cancel_task(task.id)
        assert not manager.has_finished()

    @pytest.mark.asyncio
    async def test_only_the_background_task_runs_in_background_context(self, manager, mock_agent):
        from ai_assist.background_tasks import in_background_task

        seen = []

        async def query(*args, **kwargs):
            seen.append(in_background_task())
            return "ok"

        mock_agent.query = AsyncMock(side_effect=query)
        await self._finish(manager)
        assert seen == [True]
        assert in_background_task() is False


class TestBackgroundTaskTools:
    def test_get_tool_definitions(self, tools):
        defs = tools.get_tool_definitions()
        assert len(defs) == 4
        names = {d["name"] for d in defs}
        assert names == {
            "internal__run_background",
            "internal__list_background_tasks",
            "internal__get_background_task",
            "internal__cancel_background_task",
        }

    @pytest.mark.asyncio
    async def test_run_background(self, tools, mock_agent):
        result = await tools.execute_tool(
            "run_background",
            {"prompt": "Test query", "description": "Test"},
        )
        assert "spawned" in result.lower() or "started" in result.lower()

    @pytest.mark.asyncio
    async def test_list_background_tasks(self, tools, mock_agent):
        await tools.execute_tool(
            "run_background",
            {"prompt": "Test", "description": "Test"},
        )
        await asyncio.sleep(0.1)

        result = await tools.execute_tool("list_background_tasks", {})
        assert "Test" in result

    @pytest.mark.asyncio
    async def test_get_background_task(self, tools, mock_agent):
        await tools.execute_tool(
            "run_background",
            {"prompt": "Test", "description": "Get test"},
        )
        await asyncio.sleep(0.1)

        # Extract task ID from spawn result
        tasks = tools.manager.list_tasks()
        task_id = tasks[0].id

        result = await tools.execute_tool(
            "get_background_task",
            {"task_id": task_id},
        )
        assert "Get test" in result

    @pytest.mark.asyncio
    async def test_cancel_background_task(self, tools, mock_agent):
        async def slow_query(prompt, **kwargs):
            await asyncio.sleep(10)
            return "Done"

        mock_agent.query = slow_query
        await tools.execute_tool(
            "run_background",
            {"prompt": "Slow", "description": "Cancel test"},
        )
        await asyncio.sleep(0.05)

        tasks = tools.manager.list_tasks()
        task_id = tasks[0].id

        result = await tools.execute_tool(
            "cancel_background_task",
            {"task_id": task_id},
        )
        assert "cancelled" in result.lower()

    @pytest.mark.asyncio
    async def test_background_task_cannot_spawn_another(self, tools, manager, mock_agent):
        seen = []

        async def query(*args, **kwargs):
            seen.append(await tools.execute_tool("run_background", {"prompt": "nested", "description": "nested"}))
            return "ok"

        mock_agent.query = AsyncMock(side_effect=query)
        await tools.execute_tool("run_background", {"prompt": "outer", "description": "outer"})
        await asyncio.sleep(0.1)

        assert seen and seen[0].startswith("Error")
        assert len(manager.list_tasks()) == 1

    @pytest.mark.asyncio
    async def test_get_nonexistent_task(self, tools):
        result = await tools.execute_tool(
            "get_background_task",
            {"task_id": "no-such-id"},
        )
        assert "not found" in result.lower()


def test_monitor_notifications_section_leaves_out_background_tasks(isolate_config_dir):
    """The session's background tasks reach the agent directly, not as /monitor reports."""
    import json
    from datetime import datetime

    from ai_assist.system_prompt_builder import SystemPromptBuilder

    now = datetime.now().isoformat()
    entries = [
        {"timestamp": now, "action_name": "background:abc123", "title": "Background: scan", "message": "bg result"},
        {"timestamp": now, "action_name": "nightly", "title": "Nightly report", "message": "monitor result"},
    ]
    (isolate_config_dir / "notifications.jsonl").write_text("\n".join(map(json.dumps, entries)) + "\n")

    section = SystemPromptBuilder(MagicMock())._get_recent_notifications_context()

    assert "Nightly report" in section
    assert "bg result" not in section
