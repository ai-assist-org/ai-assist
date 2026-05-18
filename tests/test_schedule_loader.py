"""Tests for schedule loader"""

import json
import tempfile
from pathlib import Path

import pytest

from ai_assist.schedule_loader import ScheduleLoader
from ai_assist.tasks import TaskDefinition


@pytest.fixture
def temp_json_file():
    """Create a temporary JSON file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "schedules.json"


@pytest.fixture
def sample_schedules():
    """Sample schedule data"""
    return {
        "version": "1.0",
        "monitors": [
            {
                "name": "Test Monitor",
                "prompt": "Check for failures",
                "interval": "5m",
                "description": "Test monitor",
                "enabled": True,
                "conditions": [],
                "knowledge_graph": {"enabled": True},
            }
        ],
        "tasks": [
            {
                "name": "Daily Report",
                "prompt": "Generate report",
                "interval": "morning on weekdays",
                "description": "Daily summary",
                "enabled": True,
                "conditions": [],
            }
        ],
    }


class TestScheduleLoader:
    """Tests for ScheduleLoader class"""

    def test_initialization(self, temp_json_file):
        """Test that ScheduleLoader initializes correctly"""
        loader = ScheduleLoader(temp_json_file)
        assert loader.json_file == temp_json_file

    def test_load_monitors_from_json(self, temp_json_file, sample_schedules):
        """Test loading monitors from JSON file (now returns TaskDefinition)"""
        temp_json_file.write_text(json.dumps(sample_schedules))

        loader = ScheduleLoader(temp_json_file)
        monitors = loader.load_monitors()

        assert len(monitors) == 1
        assert isinstance(monitors[0], TaskDefinition)
        assert monitors[0].name == "Test Monitor"
        assert monitors[0].interval == "5m"
        # knowledge_graph is not part of TaskDefinition anymore

    def test_load_tasks_from_json(self, temp_json_file, sample_schedules):
        """Test loading tasks from JSON file"""
        temp_json_file.write_text(json.dumps(sample_schedules))

        loader = ScheduleLoader(temp_json_file)
        tasks = loader.load_tasks()

        assert len(tasks) == 1
        assert isinstance(tasks[0], TaskDefinition)
        assert tasks[0].name == "Daily Report"
        assert tasks[0].interval == "morning on weekdays"

    def test_load_from_empty_json(self, temp_json_file):
        """Test loading from empty JSON file"""
        temp_json_file.write_text(json.dumps({"version": "1.0"}))

        loader = ScheduleLoader(temp_json_file)
        monitors = loader.load_monitors()
        tasks = loader.load_tasks()

        assert len(monitors) == 0
        assert len(tasks) == 0

    def test_load_from_missing_file(self, temp_json_file):
        """Test loading from non-existent file"""
        loader = ScheduleLoader(temp_json_file)
        monitors = loader.load_monitors()
        tasks = loader.load_tasks()

        assert len(monitors) == 0
        assert len(tasks) == 0

    def test_load_with_invalid_monitor(self, temp_json_file, caplog):
        """Test that invalid monitors are skipped with warning"""
        data = {
            "version": "1.0",
            "monitors": [
                {"name": "Valid Monitor", "prompt": "Test", "interval": "5m"},
                {"name": "Invalid Monitor", "prompt": "Test", "interval": "invalid"},  # Invalid interval
            ],
            "tasks": [],
        }
        temp_json_file.write_text(json.dumps(data))

        with caplog.at_level("WARNING"):
            loader = ScheduleLoader(temp_json_file)
            monitors = loader.load_monitors()

        # Should load only the valid monitor
        assert len(monitors) == 1
        assert monitors[0].name == "Valid Monitor"

        # Should log warning about invalid monitor
        assert "Invalid Monitor" in caplog.text or "Skipping invalid monitor" in caplog.text

    def test_load_with_missing_required_fields(self, temp_json_file, caplog):
        """Test that monitors with missing required fields are skipped"""
        data = {
            "version": "1.0",
            "monitors": [
                {
                    "name": "Incomplete Monitor",
                    # Missing prompt and interval
                }
            ],
            "tasks": [],
        }
        temp_json_file.write_text(json.dumps(data))

        with caplog.at_level("WARNING"):
            loader = ScheduleLoader(temp_json_file)
            monitors = loader.load_monitors()

        assert len(monitors) == 0
        assert "Incomplete Monitor" in caplog.text or "Skipping invalid monitor" in caplog.text

    def test_load_with_corrupted_json(self, temp_json_file, caplog):
        """Test handling of corrupted JSON file"""
        temp_json_file.write_text("{invalid json")

        with caplog.at_level("ERROR"):
            loader = ScheduleLoader(temp_json_file)
            monitors = loader.load_monitors()
            tasks = loader.load_tasks()

        assert len(monitors) == 0
        assert len(tasks) == 0
        assert "Failed to parse" in caplog.text or "returning empty schedules" in caplog.text

    def test_enabled_and_disabled_schedules(self, temp_json_file):
        """Test loading both enabled and disabled schedules"""
        data = {
            "version": "1.0",
            "monitors": [
                {"name": "Enabled Monitor", "prompt": "Test", "interval": "5m", "enabled": True},
                {"name": "Disabled Monitor", "prompt": "Test", "interval": "10m", "enabled": False},
            ],
            "tasks": [],
        }
        temp_json_file.write_text(json.dumps(data))

        loader = ScheduleLoader(temp_json_file)
        monitors = loader.load_monitors()

        # Both should be loaded (filtering happens in scheduler)
        assert len(monitors) == 2
        assert monitors[0].enabled is True
        assert monitors[1].enabled is False

    def test_time_based_schedules(self, temp_json_file):
        """Test loading time-based schedules"""
        data = {
            "version": "1.0",
            "monitors": [],
            "tasks": [
                {"name": "Morning Task", "prompt": "Morning report", "interval": "morning on weekdays"},
                {"name": "Specific Time Task", "prompt": "Run at 9am", "interval": "9:00 on monday,friday"},
            ],
        }
        temp_json_file.write_text(json.dumps(data))

        loader = ScheduleLoader(temp_json_file)
        tasks = loader.load_tasks()

        assert len(tasks) == 2
        assert tasks[0].is_time_based is True
        assert tasks[1].is_time_based is True

    def test_conditions_and_knowledge_graph(self, temp_json_file):
        """Test loading schedules with conditions (KG storage now automatic)"""
        data = {
            "version": "1.0",
            "monitors": [
                {
                    "name": "KG Monitor",
                    "prompt": "Test",
                    "interval": "5m",
                    "conditions": [{"type": "cache", "key": "test", "max_age": "1h"}],
                    "knowledge_graph": {"enabled": True, "save_results": True},  # Ignored now
                }
            ],
            "tasks": [],
        }
        temp_json_file.write_text(json.dumps(data))

        loader = ScheduleLoader(temp_json_file)
        monitors = loader.load_monitors()

        assert len(monitors) == 1
        assert len(monitors[0].conditions) == 1
        # knowledge_graph is no longer part of TaskDefinition - agent decides storage

    def test_multiple_monitors_and_tasks(self, temp_json_file):
        """Test loading multiple monitors and tasks"""
        data = {
            "version": "1.0",
            "monitors": [{"name": f"Monitor {i}", "prompt": "Test", "interval": f"{i}m"} for i in range(1, 6)],
            "tasks": [{"name": f"Task {i}", "prompt": "Test", "interval": f"{i}h"} for i in range(1, 4)],
        }
        temp_json_file.write_text(json.dumps(data))

        loader = ScheduleLoader(temp_json_file)
        monitors = loader.load_monitors()
        tasks = loader.load_tasks()

        assert len(monitors) == 5
        assert len(tasks) == 3

    def test_ensure_default_tasks_adds_synthesis(self, temp_json_file):
        """ensure_default_tasks should add kg-synthesis to schedules.json"""
        # Start with empty tasks
        temp_json_file.write_text(json.dumps({"version": "1.0", "monitors": [], "tasks": []}))

        loader = ScheduleLoader(temp_json_file)
        loader.ensure_default_tasks()

        # Reload and verify it was added to the file
        tasks = loader.load_tasks()
        names = [t.name for t in tasks]
        assert "kg-synthesis" in names

        synthesis_task = next(t for t in tasks if t.name == "kg-synthesis")
        assert synthesis_task.prompt == "__builtin__:kg_synthesis"
        assert synthesis_task.is_time_based is True
        assert synthesis_task.enabled is True

    def test_ensure_default_tasks_preserves_user_override(self, temp_json_file):
        """ensure_default_tasks should not overwrite existing user entry"""
        data = {
            "version": "1.0",
            "monitors": [],
            "tasks": [
                {
                    "name": "kg-synthesis",
                    "prompt": "__builtin__:kg_synthesis",
                    "interval": "1h",
                    "description": "Run hourly instead",
                }
            ],
        }
        temp_json_file.write_text(json.dumps(data))

        loader = ScheduleLoader(temp_json_file)
        loader.ensure_default_tasks()

        # User's interval should be preserved
        tasks = loader.load_tasks()
        synthesis_task = next(t for t in tasks if t.name == "kg-synthesis")
        assert synthesis_task.interval == "1h"

    def test_ensure_default_tasks_preserves_renamed_task(self, temp_json_file):
        """ensure_default_tasks should not add duplicate when user renamed the task"""
        data = {
            "version": "1.0",
            "monitors": [],
            "tasks": [
                {
                    "name": "my-custom-synthesis",
                    "prompt": "__builtin__:kg_synthesis",
                    "interval": "2h",
                    "description": "Renamed by user",
                }
            ],
        }
        temp_json_file.write_text(json.dumps(data))

        loader = ScheduleLoader(temp_json_file)
        loader.ensure_default_tasks()

        tasks = loader.load_tasks()
        synthesis_tasks = [t for t in tasks if t.prompt == "__builtin__:kg_synthesis"]
        assert len(synthesis_tasks) == 1
        assert synthesis_tasks[0].name == "my-custom-synthesis"
        assert synthesis_tasks[0].interval == "2h"

    def test_ensure_default_tasks_creates_file(self, temp_json_file):
        """ensure_default_tasks should create schedules.json if missing"""
        assert not temp_json_file.exists()

        loader = ScheduleLoader(temp_json_file)
        loader.ensure_default_tasks()

        assert temp_json_file.exists()
        tasks = loader.load_tasks()
        names = [t.name for t in tasks]
        assert "kg-synthesis" in names

    def test_load_event_source_configs(self, temp_json_file):
        """Test loading event_sources section from JSON"""
        data = {
            "version": "1.0",
            "event_sources": {
                "mqtt": {"broker": "localhost", "port": 1883},
                "dbus": {"bus": "session"},
            },
            "monitors": [],
            "tasks": [],
        }
        temp_json_file.write_text(json.dumps(data))

        loader = ScheduleLoader(temp_json_file)
        configs = loader.load_event_source_configs()

        assert "mqtt" in configs
        assert configs["mqtt"]["broker"] == "localhost"
        assert configs["dbus"]["bus"] == "session"

    def test_load_event_source_configs_missing(self, temp_json_file):
        """Test event_sources defaults to empty dict when missing"""
        data = {"version": "1.0", "monitors": [], "tasks": []}
        temp_json_file.write_text(json.dumps(data))

        loader = ScheduleLoader(temp_json_file)
        configs = loader.load_event_source_configs()

        assert configs == {}

    def test_load_monitors_with_trigger(self, temp_json_file):
        """Test loading monitors with trigger instead of interval"""
        data = {
            "version": "1.0",
            "monitors": [
                {
                    "name": "MQTT Monitor",
                    "prompt": "Handle: ${event.payload}",
                    "trigger": {"type": "mqtt", "topic": "alerts/#"},
                }
            ],
            "tasks": [],
        }
        temp_json_file.write_text(json.dumps(data))

        loader = ScheduleLoader(temp_json_file)
        monitors = loader.load_monitors()

        assert len(monitors) == 1
        assert monitors[0].is_event_triggered is True
        assert monitors[0].trigger == {"type": "mqtt", "topic": "alerts/#"}
        assert monitors[0].interval is None

    def test_load_mixed_timer_and_triggered(self, temp_json_file):
        """Test loading mix of timer and event-triggered monitors"""
        data = {
            "version": "1.0",
            "monitors": [
                {"name": "Timer", "prompt": "Check stuff", "interval": "5m"},
                {
                    "name": "MQTT",
                    "prompt": "Handle: ${event.payload}",
                    "trigger": {"type": "mqtt", "topic": "t"},
                },
            ],
            "tasks": [],
        }
        temp_json_file.write_text(json.dumps(data))

        loader = ScheduleLoader(temp_json_file)
        monitors = loader.load_monitors()

        assert len(monitors) == 2
        assert monitors[0].is_event_triggered is False
        assert monitors[1].is_event_triggered is True
