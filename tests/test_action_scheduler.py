"""Tests for ActionScheduler covering event, timer and startup-catchup paths."""

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_assist.action_model import ActionDefinition
from ai_assist.action_scheduler import ActionScheduler
from ai_assist.event_sources import EventContext
from ai_assist.state import StateManager


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.query = AsyncMock(return_value="Result")
    agent.config = MagicMock()
    agent.config.mcp_servers = {}
    agent.sessions = {}
    return agent


@pytest.fixture
def state_manager(tmp_path):
    return StateManager(state_dir=tmp_path / "state")


@pytest.fixture
def schedule_file(tmp_path):
    f = tmp_path / "event-schedules.json"
    f.write_text(json.dumps({"version": "2.0", "actions": []}))
    return f


@pytest.fixture
def scheduler(mock_agent, state_manager, schedule_file):
    s = ActionScheduler(mock_agent, state_manager, schedule_file)
    s.engine.execute_action = AsyncMock()
    return s


def _result(success=True, output="ok"):
    r = MagicMock()
    r.success = success
    r.output = output
    return r


def _event(topic="home/sensor", payload="hello"):
    return EventContext(
        source_type="mqtt",
        event_type="message",
        payload=payload,
        metadata={"topic": topic},
    )


def _mqtt_action(name="sensor_action", topic="home/sensor", enabled=True):
    return ActionDefinition(
        name=name,
        trigger={"type": "mqtt", "topic": topic},
        prompt="React to sensor",
        enabled=enabled,
    )


# --- load_actions error path (50-52) ---


def test_load_actions_handles_loader_error(scheduler):
    scheduler.loader.load_actions = MagicMock(side_effect=RuntimeError("boom"))
    assert scheduler.load_actions() == []
    assert scheduler.actions == []


# --- start() skips disabled actions (63-64) ---


async def test_start_skips_disabled_action(mock_agent, state_manager, tmp_path):
    f = tmp_path / "event-schedules.json"
    f.write_text(
        json.dumps(
            {
                "version": "2.0",
                "actions": [
                    {
                        "name": "disabled_timer",
                        "trigger": {"type": "interval", "every": "1h"},
                        "prompt": "Nope",
                        "enabled": False,
                    }
                ],
            }
        )
    )
    s = ActionScheduler(mock_agent, state_manager, f)
    tasks = await s.start()
    assert "disabled_timer" not in {t.get_name() for t in tasks}
    assert "disabled_timer" not in {h.get_name() for h in s.timer_handles}
    await s.stop()


# --- reload() skips disabled actions (109) ---


async def test_reload_skips_disabled_action(mock_agent, state_manager, tmp_path):
    f = tmp_path / "event-schedules.json"
    f.write_text(
        json.dumps(
            {
                "version": "2.0",
                "actions": [
                    {
                        "name": "disabled_timer",
                        "trigger": {"type": "interval", "every": "1h"},
                        "prompt": "Nope",
                        "enabled": False,
                    }
                ],
            }
        )
    )
    s = ActionScheduler(mock_agent, state_manager, f)
    await s.reload()
    assert "disabled_timer" not in {h.get_name() for h in s.timer_handles}
    await s.stop()


# --- _start_event_sources (128-155) ---


async def test_start_event_sources_warns_without_configs(scheduler):
    scheduler.actions = [_mqtt_action()]
    scheduler.loader.load_event_source_configs = MagicMock(return_value={})
    await scheduler._start_event_sources()
    assert scheduler.event_source_manager is None


async def test_start_event_sources_subscribes_actions(scheduler):
    scheduler.actions = [_mqtt_action()]
    scheduler.loader.load_event_source_configs = MagicMock(return_value={"mqtt": {"broker": "localhost"}})

    manager = MagicMock()
    manager.get_source.return_value = MagicMock()
    manager.start = AsyncMock()
    manager._sources = {"mqtt": object()}

    with patch("ai_assist.action_scheduler.EventSourceManager", return_value=manager):
        await scheduler._start_event_sources()

    manager.register_available_sources.assert_called_once()
    manager.get_source.return_value.subscribe.assert_called_once_with("sensor_action", scheduler.actions[0].trigger)
    manager.start.assert_awaited_once()


async def test_start_event_sources_no_event_actions_returns(scheduler):
    scheduler.actions = []
    scheduler.loader.load_event_source_configs = MagicMock(return_value={"mqtt": {}})
    await scheduler._start_event_sources()
    assert scheduler.event_source_manager is None


# --- _handle_event (162-184) ---


async def test_handle_event_debounces_and_recancels(scheduler):
    action = _mqtt_action()
    scheduler.actions = [action]
    scheduler._debounced_execute = AsyncMock()

    await scheduler._handle_event(_event())
    assert action.name in scheduler._debounce_tasks
    assert scheduler._debounce_events[action.name]
    first_task = scheduler._debounce_tasks[action.name]

    await scheduler._handle_event(_event())
    assert scheduler._debounce_tasks[action.name] is not first_task
    assert len(scheduler._debounce_events[action.name]) == 2

    for task in list(scheduler._debounce_tasks.values()):
        task.cancel()
    await asyncio.gather(*scheduler._debounce_tasks.values(), return_exceptions=True)


async def test_handle_event_skips_disabled_executing_and_cooldown(scheduler):
    import time

    disabled = _mqtt_action(name="disabled", enabled=False)
    executing = _mqtt_action(name="executing")
    cooling = _mqtt_action(name="cooling")
    scheduler.actions = [disabled, executing, cooling]
    scheduler._executing.add("executing")
    scheduler._last_completed["cooling"] = time.monotonic()
    scheduler._debounced_execute = AsyncMock()

    await scheduler._handle_event(_event())

    assert scheduler._debounce_tasks == {}


async def test_handle_event_ignores_non_matching(scheduler):
    action = _mqtt_action(topic="home/other")
    scheduler.actions = [action]
    scheduler._debounced_execute = AsyncMock()

    await scheduler._handle_event(_event(topic="home/sensor"))

    assert scheduler._debounce_tasks == {}


# --- _debounced_execute (186-218) ---


async def test_debounced_execute_single_event(scheduler):
    action = _mqtt_action()
    event = _event()
    scheduler._debounce_events[action.name] = [event]
    scheduler.engine.execute_action.return_value = _result(success=True)

    with patch("ai_assist.action_scheduler.asyncio.sleep", new_callable=AsyncMock):
        await scheduler._debounced_execute(action, 3.0)

    scheduler.engine.execute_action.assert_awaited_once_with(action, event_context=event)
    assert action.name not in scheduler._debounce_events
    assert action.name not in scheduler._debounce_tasks
    assert action.name not in scheduler._executing
    assert action.name in scheduler._last_completed


async def test_debounced_execute_combines_multiple_events_and_reports_failure(scheduler):
    action = _mqtt_action()
    scheduler._debounce_events[action.name] = [_event(payload="one"), _event(payload="two")]
    scheduler.engine.execute_action.return_value = _result(success=False, output="err")

    with patch("ai_assist.action_scheduler.asyncio.sleep", new_callable=AsyncMock):
        await scheduler._debounced_execute(action, 3.0)

    combined = scheduler.engine.execute_action.await_args.kwargs["event_context"]
    assert "2 events received" in combined.payload
    assert combined.metadata["event_count"] == 2


async def test_debounced_execute_no_events_returns_early(scheduler):
    action = _mqtt_action()
    with patch("ai_assist.action_scheduler.asyncio.sleep", new_callable=AsyncMock):
        await scheduler._debounced_execute(action, 3.0)
    scheduler.engine.execute_action.assert_not_awaited()


async def test_debounced_execute_handles_exception(scheduler):
    action = _mqtt_action()
    scheduler._debounce_events[action.name] = [_event()]
    scheduler.engine.execute_action.side_effect = RuntimeError("boom")

    with patch("ai_assist.action_scheduler.asyncio.sleep", new_callable=AsyncMock):
        await scheduler._debounced_execute(action, 3.0)

    assert action.name not in scheduler._executing
    assert action.name in scheduler._last_completed


# --- _schedule_timer_action (252-302) ---


async def test_schedule_timer_once_already_completed_breaks(scheduler):
    action = ActionDefinition(
        name="done_once",
        trigger={"type": "once", "at": "2030-01-01T10:00:00"},
        prompt="Once",
        status="completed",
    )
    scheduler.running = True
    await scheduler._schedule_timer_action(action)
    scheduler.engine.execute_action.assert_not_awaited()


async def test_schedule_timer_once_future_runs_and_marks(mock_agent, state_manager, tmp_path):
    at = (datetime.now() + timedelta(hours=1)).isoformat()
    f = tmp_path / "event-schedules.json"
    f.write_text(
        json.dumps(
            {
                "version": "2.0",
                "actions": [
                    {
                        "name": "future_once",
                        "trigger": {"type": "once", "at": at},
                        "prompt": "Once",
                        "enabled": True,
                        "status": "pending",
                    }
                ],
            }
        )
    )
    s = ActionScheduler(mock_agent, state_manager, f)
    s.load_actions()
    s.engine.execute_action = AsyncMock(return_value=_result(True))
    s._sleep_until = AsyncMock()
    s.running = True

    await s._schedule_timer_action(s.actions[0])

    s.engine.execute_action.assert_awaited_once()
    saved = json.loads(f.read_text())
    assert saved["actions"][0]["status"] == "completed"


async def test_schedule_timer_schedule_sleeps_then_executes(scheduler):
    action = ActionDefinition(
        name="daily",
        trigger={"type": "schedule", "at": "09:00", "days": "monday,tuesday,wednesday,thursday,friday,saturday,sunday"},
        prompt="Daily",
    )
    scheduler.running = True
    scheduler._sleep_until = AsyncMock()

    async def _stop(_action):
        scheduler.running = False

    scheduler._execute_timer_action = AsyncMock(side_effect=_stop)

    await scheduler._schedule_timer_action(action)

    scheduler._execute_timer_action.assert_awaited_once()
    scheduler._sleep_until.assert_awaited()


async def test_schedule_timer_interval_range_sleeps_then_executes(scheduler):
    action = ActionDefinition(
        name="ranged",
        trigger={"type": "interval_range", "every": "30m", "between": "09:00", "and": "17:00", "days": "weekdays"},
        prompt="Ranged",
    )
    scheduler.running = True
    scheduler._sleep_until = AsyncMock()

    async def _stop(_action):
        scheduler.running = False

    scheduler._execute_timer_action = AsyncMock(side_effect=_stop)

    await scheduler._schedule_timer_action(action)

    scheduler._execute_timer_action.assert_awaited_once()


async def test_schedule_timer_interval_executes_then_sleeps(scheduler):
    action = ActionDefinition(
        name="interval",
        trigger={"type": "interval", "every": "1h"},
        prompt="Interval",
    )
    scheduler.running = True
    scheduler._execute_timer_action = AsyncMock()

    with patch("ai_assist.action_scheduler.asyncio.sleep", new_callable=AsyncMock) as sleep:
        sleep.side_effect = asyncio.CancelledError
        await scheduler._schedule_timer_action(action)

    scheduler._execute_timer_action.assert_awaited_once()


async def test_schedule_timer_interval_recovers_from_error(scheduler):
    action = ActionDefinition(
        name="interval_err",
        trigger={"type": "interval", "every": "1h"},
        prompt="Interval",
    )
    scheduler.running = True
    scheduler._execute_timer_action = AsyncMock(side_effect=RuntimeError("boom"))

    with patch("ai_assist.action_scheduler.asyncio.sleep", new_callable=AsyncMock) as sleep:
        sleep.side_effect = asyncio.CancelledError
        await scheduler._schedule_timer_action(action)

    scheduler._execute_timer_action.assert_awaited_once()


# --- _execute_timer_action (304-316) ---


async def test_execute_timer_action_success_with_output(scheduler):
    action = _mqtt_action()
    scheduler.engine.execute_action.return_value = _result(success=True, output="details")
    await scheduler._execute_timer_action(action)
    assert action.name not in scheduler._executing


async def test_execute_timer_action_failure(scheduler):
    action = _mqtt_action()
    scheduler.engine.execute_action.return_value = _result(success=False, output="failed reason")
    await scheduler._execute_timer_action(action)
    assert action.name not in scheduler._executing


# --- _mark_once_completed error path (330-331) ---


def test_mark_once_completed_handles_error(scheduler):
    action = _mqtt_action()
    scheduler.loader.load_actions = MagicMock(side_effect=RuntimeError("boom"))
    scheduler._mark_once_completed(action)  # should not raise


# --- run_missed_at_startup edge cases (341, 348-349, 360-361, 365, 370-371, 388-389) ---


async def test_run_missed_skips_disabled(scheduler):
    scheduler.actions = [
        ActionDefinition(
            name="disabled_once",
            trigger={"type": "once", "at": "2026-06-15T11:00:00"},
            prompt="Once",
            enabled=False,
            status="pending",
        )
    ]
    await scheduler.run_missed_at_startup(now=datetime(2026, 6, 15, 12, 0, 0))
    scheduler.engine.execute_action.assert_not_awaited()


async def test_run_missed_skips_once_with_invalid_time(scheduler):
    scheduler.actions = [
        ActionDefinition(
            name="bad_once",
            trigger={"type": "once", "at": "not-a-timestamp"},
            prompt="Once",
            status="pending",
        )
    ]
    await scheduler.run_missed_at_startup(now=datetime(2026, 6, 15, 12, 0, 0))
    scheduler.engine.execute_action.assert_not_awaited()


async def test_run_missed_once_execution_error(scheduler):
    scheduler.actions = [
        ActionDefinition(
            name="err_once",
            trigger={"type": "once", "at": "2026-06-15T11:00:00"},
            prompt="Once",
            status="pending",
        )
    ]
    scheduler.engine.execute_action.side_effect = RuntimeError("boom")
    await scheduler.run_missed_at_startup(now=datetime(2026, 6, 15, 12, 0, 0))
    scheduler.engine.execute_action.assert_awaited_once()


async def test_run_missed_skips_non_schedule_trigger(scheduler):
    scheduler.actions = [
        ActionDefinition(
            name="interval_action",
            trigger={"type": "interval", "every": "1h"},
            prompt="Interval",
        )
    ]
    await scheduler.run_missed_at_startup(now=datetime(2026, 6, 15, 12, 0, 0))
    scheduler.engine.execute_action.assert_not_awaited()


async def test_run_missed_skips_schedule_with_invalid_trigger(scheduler):
    scheduler.actions = [
        ActionDefinition(
            name="bad_schedule",
            trigger={"type": "schedule", "at": "09:00"},  # missing days -> KeyError
            prompt="Daily",
        )
    ]
    await scheduler.run_missed_at_startup(now=datetime(2026, 6, 15, 12, 0, 0))
    scheduler.engine.execute_action.assert_not_awaited()


async def test_run_missed_schedule_execution_error(scheduler):
    scheduler.actions = [
        ActionDefinition(
            name="daily_err",
            trigger={
                "type": "schedule",
                "at": "09:00",
                "days": "monday,tuesday,wednesday,thursday,friday,saturday,sunday",
            },
            prompt="Daily",
        )
    ]
    scheduler.engine.execute_action.side_effect = RuntimeError("boom")
    await scheduler.run_missed_at_startup(now=datetime(2026, 6, 15, 12, 0, 0))
    scheduler.engine.execute_action.assert_awaited_once()
