"""Tests for event bridge (monitor→interactive IPC via events.jsonl)"""

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.panel import Panel

from ai_assist.event_bridge import BridgeEvent, BridgePublisher, BridgeWatcher


def _make_bridge_event(**kwargs) -> BridgeEvent:
    defaults = {
        "id": "evt-001",
        "timestamp": datetime(2026, 5, 5, 12, 0, 0),
        "type": "notify",
        "source_task": "MQTT Alert Handler",
        "source_type": "mqtt",
        "title": "Alert received",
        "body": "CPU usage is at 95%",
        "event_data": {"topic": "alerts/cpu"},
        "level": "warning",
    }
    defaults.update(kwargs)
    return BridgeEvent(**defaults)


class TestBridgeEvent:
    def test_construction(self):
        event = _make_bridge_event()
        assert event.id == "evt-001"
        assert event.type == "notify"
        assert event.source_task == "MQTT Alert Handler"
        assert event.source_type == "mqtt"
        assert event.level == "warning"

    def test_default_values(self):
        event = BridgeEvent(
            id="evt-002",
            timestamp=datetime.now(),
            type="notify",
            source_task="task1",
            source_type="timer",
            title="Test",
            body="body",
        )
        assert event.event_data == {}
        assert event.level == "info"

    def test_serialization_roundtrip(self):
        event = _make_bridge_event()
        json_str = event.model_dump_json()
        parsed = BridgeEvent.model_validate_json(json_str)
        assert parsed.id == event.id
        assert parsed.source_task == event.source_task
        assert parsed.event_data == event.event_data


class TestBridgePublisher:
    @pytest.mark.asyncio
    async def test_publish_creates_file(self, tmp_path):
        events_file = tmp_path / "events.jsonl"
        publisher = BridgePublisher(events_file)
        event = _make_bridge_event()

        await publisher.publish(event)

        assert events_file.exists()
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["id"] == "evt-001"
        assert parsed["source_task"] == "MQTT Alert Handler"

    @pytest.mark.asyncio
    async def test_publish_appends(self, tmp_path):
        events_file = tmp_path / "events.jsonl"
        publisher = BridgePublisher(events_file)

        await publisher.publish(_make_bridge_event(id="evt-001"))
        await publisher.publish(_make_bridge_event(id="evt-002"))

        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["id"] == "evt-001"
        assert json.loads(lines[1])["id"] == "evt-002"

    @pytest.mark.asyncio
    async def test_publish_creates_parent_dir(self, tmp_path):
        events_file = tmp_path / "subdir" / "events.jsonl"
        publisher = BridgePublisher(events_file)

        await publisher.publish(_make_bridge_event())
        assert events_file.exists()

    @pytest.mark.asyncio
    async def test_cleanup_removes_old_entries(self, tmp_path):
        events_file = tmp_path / "events.jsonl"
        publisher = BridgePublisher(events_file)

        old_event = _make_bridge_event(id="old", timestamp=datetime.now() - timedelta(days=10))
        recent_event = _make_bridge_event(id="recent", timestamp=datetime.now() - timedelta(hours=1))

        await publisher.publish(old_event)
        await publisher.publish(recent_event)

        removed = await publisher.cleanup(max_age_days=7)
        assert removed == 1

        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["id"] == "recent"

    @pytest.mark.asyncio
    async def test_cleanup_on_empty_file(self, tmp_path):
        events_file = tmp_path / "events.jsonl"
        publisher = BridgePublisher(events_file)

        removed = await publisher.cleanup()
        assert removed == 0

    @pytest.mark.asyncio
    async def test_cleanup_on_nonexistent_file(self, tmp_path):
        events_file = tmp_path / "nonexistent.jsonl"
        publisher = BridgePublisher(events_file)

        removed = await publisher.cleanup()
        assert removed == 0


class TestBridgeWatcher:
    @pytest.mark.asyncio
    async def test_reads_new_lines_only(self, tmp_path):
        events_file = tmp_path / "events.jsonl"
        event1 = _make_bridge_event(id="evt-001")
        events_file.write_text(event1.model_dump_json() + "\n")

        console = MagicMock()
        console.print = MagicMock()
        watcher = BridgeWatcher(console, events_file)

        event2 = _make_bridge_event(id="evt-002", title="New alert")
        with open(events_file, "a") as f:
            f.write(event2.model_dump_json() + "\n")

        await watcher.on_file_change()

        assert console.print.called
        panels = [
            call.args[0] for call in console.print.call_args_list if call.args and isinstance(call.args[0], Panel)
        ]
        assert len(panels) == 1
        assert "New alert" in panels[0].title

    @pytest.mark.asyncio
    async def test_handles_nonexistent_file(self, tmp_path):
        events_file = tmp_path / "nonexistent.jsonl"
        console = MagicMock()
        watcher = BridgeWatcher(console, events_file)

        await watcher.on_file_change()
        # Should not raise

    @pytest.mark.asyncio
    async def test_skips_malformed_lines(self, tmp_path):
        events_file = tmp_path / "events.jsonl"
        console = MagicMock()
        watcher = BridgeWatcher(console, events_file)

        with open(events_file, "a") as f:
            f.write("not valid json\n")
            f.write(_make_bridge_event(id="good").model_dump_json() + "\n")

        await watcher.on_file_change()
        # Should not raise, should process the good line

    @pytest.mark.asyncio
    async def test_start_stop(self, tmp_path):
        events_file = tmp_path / "events.jsonl"
        events_file.touch()
        console = MagicMock()
        watcher = BridgeWatcher(console, events_file)

        with patch("ai_assist.event_bridge.FileWatchdog") as MockWatchdog:
            mock_wd = MagicMock()
            mock_wd.start = AsyncMock()
            mock_wd.stop = AsyncMock()
            MockWatchdog.return_value = mock_wd

            await watcher.start()
            mock_wd.start.assert_called_once()

            await watcher.stop()
            mock_wd.stop.assert_called_once()
