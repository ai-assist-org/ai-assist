"""Tests for hot reload functionality"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_assist.action_scheduler import ActionScheduler
from ai_assist.state import StateManager


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.query = AsyncMock(return_value="Test result")
    agent.config.mcp_servers = {}
    agent.sessions = {}
    return agent


@pytest.fixture
def state_manager(tmp_path):
    return StateManager(state_dir=tmp_path / "state")


class TestHotReload:
    @pytest.mark.asyncio
    async def test_timer_action_cancellation_handled_gracefully(self, mock_agent, state_manager, tmp_path):
        """Test that timer action cancellation during reload doesn't crash"""
        event_schedules = tmp_path / "event-schedules.json"
        event_schedules.write_text(
            json.dumps(
                {
                    "version": "2.0",
                    "actions": [
                        {"name": "Test", "trigger": {"type": "interval", "every": "1h"}, "prompt": "Test"},
                    ],
                }
            )
        )

        scheduler = ActionScheduler(mock_agent, state_manager, event_schedules)
        await scheduler.start()

        assert len(scheduler.timer_handles) >= 1
        handle = scheduler.timer_handles[0]

        # Cancel should complete without hanging
        handle.cancel()
        try:
            await handle
        except asyncio.CancelledError:
            pass

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_reload_replaces_actions(self, mock_agent, state_manager, tmp_path):
        """Test that reload properly replaces actions"""
        event_schedules = tmp_path / "event-schedules.json"
        event_schedules.write_text(
            json.dumps(
                {
                    "version": "2.0",
                    "actions": [
                        {"name": "Action 1", "trigger": {"type": "interval", "every": "1h"}, "prompt": "Test 1"},
                    ],
                }
            )
        )

        scheduler = ActionScheduler(mock_agent, state_manager, event_schedules)
        await scheduler.start()

        action_names = {a.name for a in scheduler.actions}
        assert "Action 1" in action_names

        # Update file and reload
        event_schedules.write_text(
            json.dumps(
                {
                    "version": "2.0",
                    "actions": [
                        {"name": "Action 2", "trigger": {"type": "interval", "every": "2h"}, "prompt": "Test 2"},
                    ],
                }
            )
        )

        await scheduler.reload()

        action_names = {a.name for a in scheduler.actions}
        assert "Action 2" in action_names
        assert "Action 1" not in action_names

        await scheduler.stop()
