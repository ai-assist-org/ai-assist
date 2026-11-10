"""Tests for ActionScheduler covering event, timer and startup-catchup paths."""

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_assist.action_loader import ActionLoader
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
    assert _once_actions(f) == []
    [archived] = _archive(f)
    assert archived["name"] == "future_once"
    assert archived["status"] == "completed"


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


# --- _finish_once error path ---


def test_finish_once_handles_error(scheduler):
    action = _mqtt_action()
    scheduler.loader.load_actions = MagicMock(side_effect=RuntimeError("boom"))
    scheduler._finish_once(action, success=True)  # should not raise


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


# --- Finished one-time actions leave event-schedules.json for the archive ---


def _once_actions(schedule_file):
    return [a for a in json.loads(schedule_file.read_text())["actions"] if a["trigger"]["type"] == "once"]


def _archive(schedule_file):
    archive = schedule_file.with_name("event-schedules-archive.jsonl")
    if not archive.exists():
        return []
    return [json.loads(line) for line in archive.read_text().splitlines()]


def _once(name, at, status="pending"):
    return {"name": name, "trigger": {"type": "once", "at": at}, "prompt": name, "enabled": True, "status": status}


def _write(schedule_file, actions):
    schedule_file.write_text(json.dumps({"version": "2.0", "actions": actions}))


async def _run_once(mock_agent, state_manager, schedule_file, success):
    s = ActionScheduler(mock_agent, state_manager, schedule_file)
    s.load_actions()
    s.engine.execute_action = AsyncMock(return_value=_result(success))
    s._sleep_until = AsyncMock()
    s.running = True
    await s._schedule_timer_action(s.actions[0])
    return s


async def test_failed_once_action_archived_as_failed(mock_agent, state_manager, schedule_file):
    _write(schedule_file, [_once("report", (datetime.now() + timedelta(hours=1)).isoformat())])

    await _run_once(mock_agent, state_manager, schedule_file, success=False)

    assert _once_actions(schedule_file) == []
    [archived] = _archive(schedule_file)
    assert archived["status"] == "failed"
    assert archived["archive_reason"] == "failed"
    assert "archived_at" in archived


async def test_finishing_one_of_two_same_named_actions_keeps_the_other(mock_agent, state_manager, schedule_file):
    soon = (datetime.now() + timedelta(hours=1)).isoformat()
    later = (datetime.now() + timedelta(days=7)).isoformat()
    _write(schedule_file, [_once("Weekly report", soon), _once("Weekly report", later)])

    await _run_once(mock_agent, state_manager, schedule_file, success=True)

    [remaining] = _once_actions(schedule_file)
    assert remaining["trigger"]["at"] == later
    assert remaining["status"] == "pending"
    assert [a["trigger"]["at"] for a in _archive(schedule_file)] == [soon]


def test_load_archives_finished_and_expired_once_actions(mock_agent, state_manager, schedule_file):
    now = datetime.now()
    recurring = {"name": "daily", "trigger": {"type": "schedule", "at": "8:00", "days": "weekdays"}, "prompt": "x"}
    _write(
        schedule_file,
        [
            _once("done", (now - timedelta(days=2)).isoformat(), status="completed"),
            _once("broke", (now - timedelta(days=2)).isoformat(), status="failed"),
            _once("never ran", (now - timedelta(days=3)).isoformat()),
            _once("due an hour ago", (now - timedelta(hours=1)).isoformat()),
            _once("tomorrow", (now + timedelta(days=1)).isoformat()),
            recurring,
        ],
    )
    s = ActionScheduler(mock_agent, state_manager, schedule_file)

    s.load_actions()

    defaults = {d["name"] for d in ActionLoader.DEFAULT_ACTIONS}
    kept = [a["name"] for a in json.loads(schedule_file.read_text())["actions"] if a["name"] not in defaults]
    assert kept == ["due an hour ago", "tomorrow", "daily"]
    assert [a.name for a in s.actions if a.name not in defaults] == kept
    reasons = {a["name"]: a["archive_reason"] for a in _archive(schedule_file)}
    assert reasons == {"done": "completed", "broke": "failed", "never ran": "expired"}


def test_load_without_stale_actions_writes_nothing(mock_agent, state_manager, schedule_file):
    _write(schedule_file, [_once("tomorrow", (datetime.now() + timedelta(days=1)).isoformat())])
    scheduler = ActionScheduler(mock_agent, state_manager, schedule_file)
    scheduler.load_actions()  # adds the default actions once
    before = schedule_file.read_text()

    scheduler.load_actions()

    assert schedule_file.read_text() == before
    assert _archive(schedule_file) == []


async def test_missed_once_action_archived_after_catch_up(scheduler, schedule_file):
    now = datetime(2026, 6, 15, 12, 0, 0)
    _write(schedule_file, [_once("missed", "2026-06-15T11:00:00")])
    scheduler.actions = scheduler.loader.load_actions()
    scheduler.engine.execute_action = AsyncMock(return_value=_result(True))

    await scheduler.run_missed_at_startup(now=now)

    scheduler.engine.execute_action.assert_awaited_once()
    assert _once_actions(schedule_file) == []
    assert _archive(schedule_file)[0]["status"] == "completed"


# --- Concurrent reloads must not orphan timers ---


async def test_concurrent_reloads_leave_one_timer_per_action(mock_agent, state_manager, schedule_file):
    _write(
        schedule_file,
        [{"name": "daily", "trigger": {"type": "schedule", "at": "8:00", "days": "weekdays"}, "prompt": "x"}],
    )
    s = ActionScheduler(mock_agent, state_manager, schedule_file)

    async def sleep_forever(target):
        await asyncio.Event().wait()

    async def yield_to_others():
        await asyncio.sleep(0.01)

    s._sleep_until = sleep_forever
    s._start_event_sources = yield_to_others
    s._stop_event_sources = yield_to_others
    await s.start()

    try:
        await asyncio.gather(*(s.reload() for _ in range(10)))
        await asyncio.sleep(0)

        live = [t for t in asyncio.all_tasks() if t.get_name() == "daily" and not t.done()]
        assert len(live) == 1
        assert [h.get_name() for h in s.timer_handles].count("daily") == 1
    finally:
        await s.stop()


# --- Event sources: a restart must never leave the previous ones running ---


async def test_starting_event_sources_stops_the_previous_manager(scheduler):
    scheduler.actions = [_mqtt_action()]
    scheduler.loader.load_event_source_configs = MagicMock(return_value={"mqtt": {"broker": "localhost"}})
    previous = MagicMock()
    previous.stop = AsyncMock()
    scheduler.event_source_manager = previous

    manager = MagicMock()
    manager.start = AsyncMock()
    manager._sources = {"mqtt": object()}
    with patch("ai_assist.action_scheduler.EventSourceManager", return_value=manager):
        await scheduler._start_event_sources()

    previous.stop.assert_awaited_once()
    assert scheduler.event_source_manager is manager


async def test_start_and_reload_never_start_event_sources_concurrently(mock_agent, state_manager, schedule_file):
    """A reload during start-up would create a second manager and orphan the first one's listeners."""
    s = ActionScheduler(mock_agent, state_manager, schedule_file)
    active, overlaps = [], []

    async def slow_start_sources():
        if active:
            overlaps.append(True)
        active.append(True)
        await asyncio.sleep(0.01)
        active.pop()

    s._start_event_sources = slow_start_sources
    s._stop_event_sources = AsyncMock()
    try:
        await asyncio.gather(s.start(), s.reload(), s.reload())
        assert not overlaps
    finally:
        await s.stop()
