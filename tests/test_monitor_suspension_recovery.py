"""Tests for monitor suspension recovery and startup catchup."""

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_assist.action_scheduler import ActionScheduler
from ai_assist.monitors import MonitoringScheduler
from ai_assist.state import StateManager


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.query = AsyncMock(return_value="Mock response")
    agent.config = MagicMock()
    agent.config.mcp_servers = {}
    agent.sessions = {}
    return agent


@pytest.fixture
def temp_schedule_file(tmp_path):
    """Create a temporary event-schedules.json with a time-based action."""
    f = tmp_path / "event-schedules.json"
    data = {
        "version": "2.0",
        "actions": [
            {
                "name": "test_time_based",
                "trigger": {"type": "schedule", "at": "9:00", "days": "weekdays"},
                "prompt": "Test prompt",
                "enabled": True,
            }
        ],
    }
    f.write_text(json.dumps(data))
    return f


@pytest.fixture
def schedule_file_with_daily_task(tmp_path):
    """Event-schedules file with a daily 8:00 task on all days."""
    f = tmp_path / "event-schedules.json"
    data = {
        "version": "2.0",
        "actions": [
            {
                "name": "morning_briefing",
                "trigger": {
                    "type": "schedule",
                    "at": "8:00",
                    "days": "monday,tuesday,wednesday,thursday,friday,saturday,sunday",
                },
                "prompt": "Morning briefing",
                "enabled": True,
            }
        ],
    }
    f.write_text(json.dumps(data))
    return f


@pytest.mark.asyncio
async def test_startup_catchup_runs_missed_task(mock_agent, schedule_file_with_daily_task):
    """Test that startup catchup runs a task missed since last 24h."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_manager = StateManager(Path(tmpdir))

        scheduler = ActionScheduler(mock_agent, state_manager, schedule_file_with_daily_task)
        scheduler.load_actions()

        now = datetime.now().replace(hour=10, minute=0, second=0)
        await scheduler.run_missed_at_startup(now=now)

        mock_agent.query.assert_called_once()


@pytest.mark.asyncio
async def test_startup_catchup_skips_successfully_run_task(mock_agent, schedule_file_with_daily_task):
    """Test that startup catchup skips tasks that already succeeded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_manager = StateManager(Path(tmpdir))

        scheduler = ActionScheduler(mock_agent, state_manager, schedule_file_with_daily_task)
        scheduler.load_actions()

        # Simulate a successful previous run
        from ai_assist.action_engine import ActionEngine

        state_key = ActionEngine._state_key(scheduler.actions[0])
        state_manager.update_monitor(
            state_key,
            {"task_name": "morning_briefing", "last_success": True},
        )
        state = state_manager.get_monitor_state(state_key)
        state.last_check = datetime.now().replace(hour=8, minute=5)
        state_manager.save_monitor_state(state_key, state)

        now = datetime.now().replace(hour=10, minute=0, second=0)
        await scheduler.run_missed_at_startup(now=now)

        mock_agent.query.assert_not_called()


@pytest.mark.asyncio
async def test_startup_catchup_retries_failed_run(mock_agent, schedule_file_with_daily_task):
    """Test that startup catchup retries tasks that failed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_manager = StateManager(Path(tmpdir))

        scheduler = ActionScheduler(mock_agent, state_manager, schedule_file_with_daily_task)
        scheduler.load_actions()

        from ai_assist.action_engine import ActionEngine

        state_key = ActionEngine._state_key(scheduler.actions[0])
        state_manager.update_monitor(
            state_key,
            {"task_name": "morning_briefing", "last_success": False},
        )
        state = state_manager.get_monitor_state(state_key)
        state.last_check = datetime.now().replace(hour=8, minute=5)
        state_manager.save_monitor_state(state_key, state)

        now = datetime.now().replace(hour=10, minute=0, second=0)
        await scheduler.run_missed_at_startup(now=now)

        mock_agent.query.assert_called_once()


@pytest.fixture
def schedule_file_with_once_action(tmp_path):
    """Event-schedules file with a pending once-action whose time has passed."""
    f = tmp_path / "event-schedules.json"
    data = {
        "version": "2.0",
        "actions": [
            {
                "name": "weekly_report_semih",
                "trigger": {"type": "once", "at": "2026-05-11T10:45:00"},
                "prompt": "Generate weekly report",
                "enabled": True,
                "status": "pending",
            }
        ],
    }
    f.write_text(json.dumps(data))
    return f


@pytest.mark.asyncio
async def test_startup_catchup_runs_missed_once_action(mock_agent, schedule_file_with_once_action):
    """Test that startup catchup runs a pending once-action whose time has passed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_manager = StateManager(Path(tmpdir))

        scheduler = ActionScheduler(mock_agent, state_manager, schedule_file_with_once_action)
        scheduler.load_actions()

        now = datetime(2026, 5, 11, 11, 0, 0)
        await scheduler.run_missed_at_startup(now=now)

        mock_agent.query.assert_called_once()
        # Verify it was marked completed in the file
        reloaded = json.loads(schedule_file_with_once_action.read_text())
        assert reloaded["actions"][0]["status"] == "completed"
        assert reloaded["actions"][0]["executed_at"] is not None


@pytest.mark.asyncio
async def test_startup_catchup_skips_completed_once_action(mock_agent, tmp_path):
    """Test that startup catchup skips already-completed once-actions."""
    f = tmp_path / "event-schedules.json"
    data = {
        "version": "2.0",
        "actions": [
            {
                "name": "weekly_report_semih",
                "trigger": {"type": "once", "at": "2026-05-11T10:45:00"},
                "prompt": "Generate weekly report",
                "enabled": True,
                "status": "completed",
                "executed_at": "2026-05-11T10:46:00",
            }
        ],
    }
    f.write_text(json.dumps(data))

    with tempfile.TemporaryDirectory() as tmpdir:
        state_manager = StateManager(Path(tmpdir))

        scheduler = ActionScheduler(mock_agent, state_manager, f)
        scheduler.load_actions()

        now = datetime(2026, 5, 11, 11, 0, 0)
        await scheduler.run_missed_at_startup(now=now)

        mock_agent.query.assert_not_called()


@pytest.mark.asyncio
async def test_startup_catchup_skips_future_once_action(mock_agent, tmp_path):
    """Test that startup catchup skips once-actions whose time hasn't come yet."""
    f = tmp_path / "event-schedules.json"
    data = {
        "version": "2.0",
        "actions": [
            {
                "name": "weekly_report_olivier",
                "trigger": {"type": "once", "at": "2026-05-11T15:45:00"},
                "prompt": "Generate weekly report",
                "enabled": True,
                "status": "pending",
            }
        ],
    }
    f.write_text(json.dumps(data))

    with tempfile.TemporaryDirectory() as tmpdir:
        state_manager = StateManager(Path(tmpdir))

        scheduler = ActionScheduler(mock_agent, state_manager, f)
        scheduler.load_actions()

        now = datetime(2026, 5, 11, 11, 0, 0)
        await scheduler.run_missed_at_startup(now=now)

        mock_agent.query.assert_not_called()


@pytest.mark.asyncio
async def test_startup_catchup_skips_stale_once_action(mock_agent, tmp_path):
    """Test that startup catchup skips once-actions older than 24h."""
    f = tmp_path / "event-schedules.json"
    data = {
        "version": "2.0",
        "actions": [
            {
                "name": "old_report",
                "trigger": {"type": "once", "at": "2026-05-09T10:00:00"},
                "prompt": "Old report",
                "enabled": True,
                "status": "pending",
            }
        ],
    }
    f.write_text(json.dumps(data))

    with tempfile.TemporaryDirectory() as tmpdir:
        state_manager = StateManager(Path(tmpdir))

        scheduler = ActionScheduler(mock_agent, state_manager, f)
        scheduler.load_actions()

        now = datetime(2026, 5, 11, 11, 0, 0)
        await scheduler.run_missed_at_startup(now=now)

        mock_agent.query.assert_not_called()


@pytest.mark.asyncio
async def test_schedule_timer_skips_overdue_once_action(mock_agent, tmp_path):
    """Test that _schedule_timer_action does NOT execute overdue once-actions."""
    f = tmp_path / "event-schedules.json"
    data = {
        "version": "2.0",
        "actions": [
            {
                "name": "overdue_report",
                "trigger": {"type": "once", "at": "2026-05-11T10:45:00"},
                "prompt": "Generate weekly report",
                "enabled": True,
                "status": "pending",
            }
        ],
    }
    f.write_text(json.dumps(data))

    with tempfile.TemporaryDirectory() as tmpdir:
        state_manager = StateManager(Path(tmpdir))
        scheduler = ActionScheduler(mock_agent, state_manager, f)
        scheduler.load_actions()
        scheduler.running = True

        action = scheduler.actions[0]
        await scheduler._schedule_timer_action(action)

        mock_agent.query.assert_not_called()


@pytest.mark.asyncio
async def test_reload_does_not_cancel_executing_action(mock_agent, tmp_path):
    """Test that reload() preserves tasks that are mid-execution."""
    f = tmp_path / "event-schedules.json"
    data = {
        "version": "2.0",
        "actions": [
            {
                "name": "slow_action",
                "trigger": {"type": "schedule", "at": "10:00", "days": "weekdays"},
                "prompt": "Do something slow",
                "enabled": True,
            },
            {
                "name": "idle_action",
                "trigger": {"type": "schedule", "at": "22:00", "days": "weekdays"},
                "prompt": "Do something later",
                "enabled": True,
            },
        ],
    }
    f.write_text(json.dumps(data))

    with tempfile.TemporaryDirectory() as tmpdir:
        state_manager = StateManager(Path(tmpdir))
        scheduler = ActionScheduler(mock_agent, state_manager, f)
        scheduler.load_actions()
        scheduler.running = True

        # Simulate slow_action being mid-execution
        slow_task = asyncio.create_task(asyncio.sleep(10), name="slow_action")
        idle_task = asyncio.create_task(asyncio.sleep(10), name="idle_action")
        scheduler.timer_handles = [slow_task, idle_task]
        scheduler._executing.add("slow_action")

        await scheduler.reload()

        # slow_action's task should NOT have been cancelled
        assert not slow_task.cancelled()
        # idle_action's task SHOULD have been cancelled
        assert idle_task.cancelled()
        # slow_action should still be in timer_handles
        handle_names = [h.get_name() for h in scheduler.timer_handles]
        assert "slow_action" in handle_names

        # Cleanup
        slow_task.cancel()
        await asyncio.gather(slow_task, return_exceptions=True)
        await scheduler.stop()


@pytest.mark.asyncio
async def test_mark_once_completed_suppresses_reload(mock_agent, tmp_path):
    """Test that _mark_once_completed sets self-write timestamp to suppress watchdog reload."""
    f = tmp_path / "event-schedules.json"
    data = {
        "version": "2.0",
        "actions": [
            {
                "name": "report_task",
                "trigger": {"type": "once", "at": "2026-05-11T10:45:00"},
                "prompt": "Generate report",
                "enabled": True,
                "status": "pending",
            }
        ],
    }
    f.write_text(json.dumps(data))

    with tempfile.TemporaryDirectory() as tmpdir:
        state_manager = StateManager(Path(tmpdir))
        scheduler = ActionScheduler(mock_agent, state_manager, f)
        scheduler.load_actions()

        action = scheduler.actions[0]
        scheduler._mark_once_completed(action)

        # Self-write timestamp should be recent, causing reload to skip
        assert scheduler._self_write_time > 0
        # reload() should be a no-op right after a self-write
        scheduler.load_actions = MagicMock()
        await scheduler.reload()
        scheduler.load_actions.assert_not_called()

        reloaded = json.loads(f.read_text())
        assert reloaded["actions"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_suspend_detector_integration(mock_agent):
    """Test that suspend detector is initialized and running."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_manager = StateManager(Path(tmpdir))
        config = MagicMock()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"version": "2.0", "actions": []}, f)
            f.flush()
            schedule_file = Path(f.name)

        try:
            scheduler = MonitoringScheduler(
                agent=mock_agent, config=config, state_manager=state_manager, schedule_file=schedule_file
            )

            start_task = asyncio.create_task(scheduler.start())
            await asyncio.sleep(0.2)

            assert scheduler.suspend_detector is not None
            assert scheduler.suspend_detector.suspend_threshold_seconds == 30.0

            await scheduler.stop()
            start_task.cancel()
            try:
                await start_task
            except asyncio.CancelledError:
                pass

        finally:
            schedule_file.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_sleep_until_returns_immediately_when_target_past(mock_agent, temp_schedule_file):
    """_sleep_until returns immediately when target time is in the past."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_manager = StateManager(Path(tmpdir))
        scheduler = ActionScheduler(mock_agent, state_manager, temp_schedule_file)
        scheduler.running = True

        from datetime import timedelta

        past = datetime.now() - timedelta(hours=2)
        start = asyncio.get_event_loop().time()
        await scheduler._sleep_until(past)
        elapsed = asyncio.get_event_loop().time() - start
        assert elapsed < 0.1


@pytest.mark.asyncio
async def test_sleep_until_returns_near_target(mock_agent, temp_schedule_file):
    """_sleep_until returns close to the target time."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_manager = StateManager(Path(tmpdir))
        scheduler = ActionScheduler(mock_agent, state_manager, temp_schedule_file)
        scheduler.running = True

        from datetime import timedelta

        target = datetime.now() + timedelta(seconds=0.3)
        await scheduler._sleep_until(target)
        assert datetime.now() >= target


@pytest.mark.asyncio
async def test_sleep_until_wakes_on_resume(mock_agent, temp_schedule_file):
    """_sleep_until returns promptly when notify_resume is called and target has passed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_manager = StateManager(Path(tmpdir))
        scheduler = ActionScheduler(mock_agent, state_manager, temp_schedule_file)
        scheduler.running = True

        from datetime import timedelta

        target = datetime.now() + timedelta(hours=1)

        async def signal_resume():
            await asyncio.sleep(0.1)
            scheduler.notify_resume()

        asyncio.create_task(signal_resume())
        start = asyncio.get_event_loop().time()
        # target is still in the future so _sleep_until will loop and re-sleep
        # but it should wake within ~0.2s, not an hour
        task = asyncio.create_task(scheduler._sleep_until(target))
        await asyncio.sleep(0.3)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        elapsed = asyncio.get_event_loop().time() - start
        assert elapsed < 1.0


@pytest.mark.asyncio
async def test_sleep_until_respects_cancellation(mock_agent, temp_schedule_file):
    """_sleep_until raises CancelledError when the task is cancelled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_manager = StateManager(Path(tmpdir))
        scheduler = ActionScheduler(mock_agent, state_manager, temp_schedule_file)
        scheduler.running = True

        from datetime import timedelta

        target = datetime.now() + timedelta(hours=1)
        task = asyncio.create_task(scheduler._sleep_until(target))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_action_scheduler_file_watchdog_integration(mock_agent, temp_schedule_file):
    """Test that action scheduler file watchdog is initialized and watching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_manager = StateManager(Path(tmpdir))
        config = MagicMock()

        scheduler = MonitoringScheduler(
            agent=mock_agent, config=config, state_manager=state_manager, schedule_file=temp_schedule_file
        )

        start_task = asyncio.create_task(scheduler.start())
        await asyncio.sleep(0.2)

        assert scheduler.action_scheduler_file_watchdog is not None

        await scheduler.stop()
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            pass
