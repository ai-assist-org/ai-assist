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
