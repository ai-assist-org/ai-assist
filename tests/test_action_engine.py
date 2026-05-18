"""Tests for unified action execution engine"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_assist.action_engine import ActionEngine
from ai_assist.action_model import ActionDefinition
from ai_assist.event_sources import EventContext
from ai_assist.state import StateManager


@pytest.fixture
def temp_state_dir(tmp_path):
    return tmp_path / "state"


@pytest.fixture
def state_manager(temp_state_dir):
    return StateManager(state_dir=temp_state_dir)


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.query = AsyncMock()
    agent.execute_mcp_prompt = AsyncMock()
    agent._run_synthesis_from_kg = AsyncMock()
    return agent


@pytest.fixture
def engine(mock_agent, state_manager):
    return ActionEngine(mock_agent, state_manager)


def _make_action(**kwargs) -> ActionDefinition:
    defaults = {
        "name": "test-action",
        "trigger": {"type": "interval", "every": "5m"},
        "prompt": "Test prompt",
    }
    defaults.update(kwargs)
    return ActionDefinition(**defaults)


def _make_event(**kwargs) -> EventContext:
    defaults = {
        "source_type": "mqtt",
        "event_type": "message",
        "payload": "CPU usage 95%",
        "metadata": {"topic": "alerts/cpu", "qos": 0},
        "timestamp": datetime(2026, 5, 7, 12, 0, 0),
    }
    defaults.update(kwargs)
    return EventContext(**defaults)


class TestActionEngine:
    @pytest.mark.asyncio
    async def test_execute_natural_language(self, engine, mock_agent):
        mock_agent.query.return_value = "Found 3 failures"
        action = _make_action(prompt="Check for failures")

        result = await engine.execute_action(action)

        assert result.success is True
        assert result.output == "Found 3 failures"
        mock_agent.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_mcp_prompt(self, engine, mock_agent):
        mock_agent.execute_mcp_prompt.return_value = "Report generated"
        action = _make_action(
            prompt="mcp://dci/rca",
            prompt_arguments={"days": "1"},
        )

        result = await engine.execute_action(action)

        assert result.success is True
        mock_agent.execute_mcp_prompt.assert_called_once_with("dci", "rca", {"days": "1"}, max_turns=100)

    @pytest.mark.asyncio
    async def test_execute_builtin_kg_synthesis(self, engine, mock_agent):
        mock_agent._run_synthesis_from_kg.return_value = "Synthesized 5 insights"
        action = _make_action(prompt="__builtin__:kg_synthesis")

        result = await engine.execute_action(action)

        assert result.success is True
        mock_agent._run_synthesis_from_kg.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_event_context(self, engine, mock_agent):
        mock_agent.query.return_value = "Alert analyzed"
        action = _make_action(
            prompt="Analyze this alert",
            trigger={"type": "mqtt", "topic": "alerts/#"},
        )
        event = _make_event()

        result = await engine.execute_action(action, event_context=event)

        assert result.success is True
        called_prompt = mock_agent.query.call_args[0][0]
        assert "mqtt" in called_prompt
        assert "CPU usage 95%" in called_prompt
        assert "Analyze this alert" in called_prompt

    @pytest.mark.asyncio
    async def test_execute_failure(self, engine, mock_agent):
        mock_agent.query.side_effect = RuntimeError("Connection failed")
        action = _make_action()

        with patch("ai_assist.execution_helpers.NotificationDispatcher") as mock_nd:
            mock_nd.return_value.dispatch = AsyncMock()
            result = await engine.execute_action(action)

        assert result.success is False
        assert "Connection failed" in result.output

    @pytest.mark.asyncio
    async def test_execute_records_state(self, engine, mock_agent, state_manager):
        mock_agent.query.return_value = "Done"
        action = _make_action(name="stateful")

        await engine.execute_action(action)

        state = state_manager.get_monitor_state("action_stateful")
        assert state.last_results.get("last_success") is True

    @pytest.mark.asyncio
    async def test_execute_records_history(self, engine, mock_agent, state_manager):
        mock_agent.query.return_value = "Done"
        action = _make_action(name="histaction")

        await engine.execute_action(action)

        history = state_manager.get_history("action_histaction", limit=1)
        assert len(history) == 1

    @pytest.mark.asyncio
    async def test_execute_records_event_source_in_history(self, engine, mock_agent, state_manager):
        mock_agent.query.return_value = "Done"
        action = _make_action(name="evtaction", trigger={"type": "mqtt", "topic": "t"})
        event = _make_event()

        await engine.execute_action(action, event_context=event)

        history = state_manager.get_history("action_evtaction", limit=1)
        assert len(history) == 1
        result_data = history[0].get("result", history[0])
        assert result_data.get("event_source") == "mqtt"

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_execute_sends_notification(self, engine, mock_agent):
        mock_agent.query.return_value = "Alert handled"
        action = _make_action(notify=True)

        with patch("ai_assist.action_engine.NotificationDispatcher") as mock_nd:
            mock_nd.return_value.dispatch = AsyncMock(return_value={})
            result = await engine.execute_action(action)

        assert result.success is True
        mock_nd.return_value.dispatch.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_failure_always_notifies(self, engine, mock_agent):
        mock_agent.query.side_effect = Exception("Boom")
        action = _make_action(notify=False)

        with patch("ai_assist.execution_helpers.NotificationDispatcher") as mock_nd:
            mock_nd.return_value.dispatch = AsyncMock()
            await engine.execute_action(action)

        mock_nd.return_value.dispatch.assert_called_once()
        notification = mock_nd.return_value.dispatch.call_args[0][0]
        assert notification.level == "error"

    @pytest.mark.asyncio
    async def test_execute_conditions(self, engine, mock_agent):
        mock_agent.query.return_value = "Found 10 failures in the system"
        action = _make_action(
            conditions=[{"if": "failures >= 5", "then": {"type": "notify", "message": "High failure rate"}}]
        )

        result = await engine.execute_action(action)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_event_context_formatting(self, engine, mock_agent):
        mock_agent.query.return_value = "ok"
        action = _make_action(
            prompt="Handle the event",
            trigger={"type": "dbus", "interface": "org.fd.UDisks2", "signal": "Added"},
        )
        event = _make_event(
            source_type="dbus",
            event_type="signal",
            payload="['/dev/sdb1']",
            metadata={"interface": "org.fd.UDisks2", "signal": "Added", "path": "/org/fd"},
        )

        await engine.execute_action(action, event_context=event)

        called_prompt = mock_agent.query.call_args[0][0]
        assert "dbus" in called_prompt
        assert "org.fd.UDisks2" in called_prompt
        assert "['/dev/sdb1']" in called_prompt
        assert "Handle the event" in called_prompt

    @pytest.mark.asyncio
    async def test_backward_compat_no_event(self, engine, mock_agent):
        mock_agent.query.return_value = "No issues"
        action = _make_action()

        result = await engine.execute_action(action)

        assert result.success is True
        called_prompt = mock_agent.query.call_args[0][0]
        assert called_prompt == "Test prompt"
