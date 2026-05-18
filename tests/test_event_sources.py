"""Tests for event source core abstractions"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_assist.event_sources import EventContext, EventSource, EventSourceManager


class FakeEventSource(EventSource):
    """Test implementation of EventSource"""

    def __init__(self):
        self.started = False
        self.stopped = False
        self._dispatch = None
        self._subscriptions: list[tuple[str, dict]] = []

    @property
    def name(self) -> str:
        return "fake"

    def subscribe(self, task_name: str, trigger_config: dict) -> None:
        self._subscriptions.append((task_name, trigger_config))

    def unsubscribe_all(self) -> None:
        self._subscriptions.clear()

    async def start(self, dispatch) -> None:
        self.started = True
        self._dispatch = dispatch

    async def stop(self) -> None:
        self.stopped = True

    async def simulate_event(self, task_name: str, event: EventContext) -> None:
        await self._dispatch(task_name, event)


def _make_event(**kwargs) -> EventContext:
    defaults = {
        "source_type": "fake",
        "event_type": "test",
        "payload": "hello",
        "metadata": {},
        "timestamp": datetime(2026, 5, 5, 12, 0, 0),
    }
    defaults.update(kwargs)
    return EventContext(**defaults)


class TestEventContext:
    def test_construction(self):
        event = _make_event(payload="test data", metadata={"key": "val"})
        assert event.source_type == "fake"
        assert event.event_type == "test"
        assert event.payload == "test data"
        assert event.metadata == {"key": "val"}
        assert event.timestamp == datetime(2026, 5, 5, 12, 0, 0)

    def test_default_timestamp(self):
        event = EventContext(source_type="mqtt", event_type="message", payload="x", metadata={})
        assert isinstance(event.timestamp, datetime)

    def test_metadata_access(self):
        event = _make_event(metadata={"topic": "alerts/cpu", "qos": 1})
        assert event.metadata["topic"] == "alerts/cpu"
        assert event.metadata["qos"] == 1


class TestEventSourceABC:
    def test_fake_source_implements_interface(self):
        source = FakeEventSource()
        assert isinstance(source, EventSource)
        assert source.name == "fake"

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        source = FakeEventSource()
        assert not source.started
        assert not source.stopped

        await source.start(AsyncMock())
        assert source.started

        await source.stop()
        assert source.stopped

    def test_subscribe(self):
        source = FakeEventSource()
        source.subscribe("task1", {"topic": "alerts/#"})
        source.subscribe("task2", {"topic": "status/#"})
        assert len(source._subscriptions) == 2
        assert source._subscriptions[0] == ("task1", {"topic": "alerts/#"})

    def test_unsubscribe_all(self):
        source = FakeEventSource()
        source.subscribe("task1", {"topic": "a"})
        source.subscribe("task2", {"topic": "b"})
        source.unsubscribe_all()
        assert len(source._subscriptions) == 0


class TestEventSourceManager:
    def test_register_source(self):
        manager = EventSourceManager()
        source = FakeEventSource()
        manager.register_source("fake", source)
        assert manager.get_source("fake") is source

    def test_get_unknown_source(self):
        manager = EventSourceManager()
        assert manager.get_source("unknown") is None

    @pytest.mark.asyncio
    async def test_configure_and_start(self):
        manager = EventSourceManager()
        source = FakeEventSource()
        manager.register_source("fake", source)

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock()
        triggered_tasks = [("my-task", {"type": "fake", "key": "val"}, mock_runner)]
        manager.configure(triggered_tasks)

        await manager.start()
        assert source.started
        assert len(source._subscriptions) == 1
        assert source._subscriptions[0] == ("my-task", {"type": "fake", "key": "val"})

        await manager.stop()
        assert source.stopped

    @pytest.mark.asyncio
    async def test_dispatch_routes_to_runner(self):
        manager = EventSourceManager()
        source = FakeEventSource()
        manager.register_source("fake", source)

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock()
        triggered_tasks = [("my-task", {"type": "fake"}, mock_runner)]
        manager.configure(triggered_tasks)
        await manager.start()

        event = _make_event(payload="alert data")
        await source.simulate_event("my-task", event)

        mock_runner.run.assert_called_once_with(event_context=event)

        await manager.stop()

    @pytest.mark.asyncio
    async def test_dispatch_unknown_task_logs_warning(self, caplog):
        manager = EventSourceManager()
        source = FakeEventSource()
        manager.register_source("fake", source)
        manager.configure([])
        await manager.start()

        event = _make_event()
        await source.simulate_event("nonexistent-task", event)

        assert "Unknown task" in caplog.text

        await manager.stop()

    @pytest.mark.asyncio
    async def test_configure_skips_unknown_source_type(self, caplog):
        manager = EventSourceManager()
        mock_runner = MagicMock()
        triggered_tasks = [("my-task", {"type": "nonexistent_source"}, mock_runner)]
        manager.configure(triggered_tasks)

        assert "No event source registered" in caplog.text

    @pytest.mark.asyncio
    async def test_start_stop_with_no_sources(self):
        manager = EventSourceManager()
        manager.configure([])
        await manager.start()
        await manager.stop()

    @pytest.mark.asyncio
    async def test_multiple_tasks_same_source(self):
        manager = EventSourceManager()
        source = FakeEventSource()
        manager.register_source("fake", source)

        runner1 = MagicMock()
        runner1.run = AsyncMock()
        runner2 = MagicMock()
        runner2.run = AsyncMock()

        triggered_tasks = [
            ("task1", {"type": "fake", "topic": "a"}, runner1),
            ("task2", {"type": "fake", "topic": "b"}, runner2),
        ]
        manager.configure(triggered_tasks)
        await manager.start()

        assert len(source._subscriptions) == 2

        event = _make_event()
        await source.simulate_event("task1", event)
        runner1.run.assert_called_once()
        runner2.run.assert_not_called()

        await manager.stop()

    @pytest.mark.asyncio
    async def test_dispatch_handles_runner_exception(self, caplog):
        manager = EventSourceManager()
        source = FakeEventSource()
        manager.register_source("fake", source)

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(side_effect=RuntimeError("task failed"))
        mock_runner.task_def = MagicMock()
        mock_runner.task_def.name = "broken-task"

        triggered_tasks = [("broken-task", {"type": "fake"}, mock_runner)]
        manager.configure(triggered_tasks)
        await manager.start()

        event = _make_event()
        await source.simulate_event("broken-task", event)

        assert "Error dispatching event" in caplog.text

        await manager.stop()
