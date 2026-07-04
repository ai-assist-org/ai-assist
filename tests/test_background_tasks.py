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
    async def test_get_nonexistent_task(self, tools):
        result = await tools.execute_tool(
            "get_background_task",
            {"task_id": "no-such-id"},
        )
        assert "not found" in result.lower()
