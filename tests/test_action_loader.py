"""Tests for action loader (event-schedules.json)"""

import json
import tempfile
from pathlib import Path

import pytest

from ai_assist.action_loader import ActionLoader
from ai_assist.action_model import ActionDefinition


@pytest.fixture
def temp_json_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "event-schedules.json"


@pytest.fixture
def sample_actions_data():
    return {
        "version": "2.0",
        "event_sources": {
            "mqtt": {"broker": "localhost", "port": 1883},
        },
        "actions": [
            {
                "name": "DCI Monitor",
                "trigger": {"type": "interval", "every": "5m"},
                "prompt": "Check DCI failures",
                "enabled": True,
            },
            {
                "name": "Morning Report",
                "trigger": {"type": "schedule", "at": "9:00", "days": "weekdays"},
                "prompt": "mcp://dci/rca",
                "prompt_arguments": {"days": "1"},
            },
            {
                "name": "Alert Handler",
                "trigger": {"type": "mqtt", "topic": "alerts/#"},
                "prompt": "Analyze alert",
                "notify": True,
            },
        ],
    }


class TestActionLoader:
    def test_initialization(self, temp_json_file):
        loader = ActionLoader(temp_json_file)
        assert loader.json_file == temp_json_file

    def test_load_actions(self, temp_json_file, sample_actions_data):
        temp_json_file.write_text(json.dumps(sample_actions_data))
        loader = ActionLoader(temp_json_file)
        actions = loader.load_actions()

        assert len(actions) == 3
        assert all(isinstance(a, ActionDefinition) for a in actions)
        assert actions[0].name == "DCI Monitor"
        assert actions[0].trigger_type == "interval"
        assert actions[1].name == "Morning Report"
        assert actions[1].trigger_type == "schedule"
        assert actions[2].name == "Alert Handler"
        assert actions[2].trigger_type == "mqtt"

    def test_load_event_source_configs(self, temp_json_file, sample_actions_data):
        temp_json_file.write_text(json.dumps(sample_actions_data))
        loader = ActionLoader(temp_json_file)
        configs = loader.load_event_source_configs()

        assert "mqtt" in configs
        assert configs["mqtt"]["broker"] == "localhost"

    def test_load_from_missing_file(self, temp_json_file):
        loader = ActionLoader(temp_json_file)
        actions = loader.load_actions()
        assert actions == []

    def test_load_from_empty_json(self, temp_json_file):
        temp_json_file.write_text(json.dumps({"version": "2.0"}))
        loader = ActionLoader(temp_json_file)
        actions = loader.load_actions()
        assert actions == []

    def test_load_skips_invalid_action(self, temp_json_file, caplog):
        data = {
            "version": "2.0",
            "actions": [
                {"name": "Valid", "trigger": {"type": "interval", "every": "5m"}, "prompt": "Test"},
                {"name": "Invalid", "trigger": {"type": "interval"}, "prompt": "Test"},
            ],
        }
        temp_json_file.write_text(json.dumps(data))

        with caplog.at_level("WARNING"):
            loader = ActionLoader(temp_json_file)
            actions = loader.load_actions()

        assert len(actions) == 1
        assert actions[0].name == "Valid"
        assert "Invalid" in caplog.text

    def test_load_corrupted_json(self, temp_json_file, caplog):
        temp_json_file.write_text("{bad json")

        with caplog.at_level("ERROR"):
            loader = ActionLoader(temp_json_file)
            actions = loader.load_actions()

        assert actions == []

    def test_save_actions(self, temp_json_file):
        loader = ActionLoader(temp_json_file)
        actions = [
            ActionDefinition(
                name="Test",
                trigger={"type": "interval", "every": "5m"},
                prompt="Check stuff",
            ),
        ]
        loader.save_actions(actions)

        assert temp_json_file.exists()
        data = json.loads(temp_json_file.read_text())
        assert len(data["actions"]) == 1
        assert data["actions"][0]["name"] == "Test"

    def test_save_preserves_event_sources(self, temp_json_file, sample_actions_data):
        temp_json_file.write_text(json.dumps(sample_actions_data))
        loader = ActionLoader(temp_json_file)

        actions = loader.load_actions()
        actions.append(ActionDefinition(name="New", trigger={"type": "interval", "every": "1h"}, prompt="New action"))
        loader.save_actions(actions)

        data = json.loads(temp_json_file.read_text())
        assert "mqtt" in data["event_sources"]
        assert len(data["actions"]) == 4

    def test_save_creates_parent_dir(self, tmp_path):
        json_file = tmp_path / "subdir" / "event-schedules.json"
        loader = ActionLoader(json_file)
        loader.save_actions([])
        assert json_file.exists()

    def test_ensure_defaults_adds_kg_synthesis(self, temp_json_file):
        temp_json_file.write_text(json.dumps({"version": "2.0", "actions": []}))
        loader = ActionLoader(temp_json_file)
        loader.ensure_defaults()

        actions = loader.load_actions()
        names = [a.name for a in actions]
        assert "kg-synthesis" in names
        synthesis = next(a for a in actions if a.name == "kg-synthesis")
        assert synthesis.prompt == "__builtin__:kg_synthesis"
        assert synthesis.trigger_type == "schedule"

    def test_ensure_defaults_preserves_existing(self, temp_json_file):
        data = {
            "version": "2.0",
            "actions": [
                {
                    "name": "custom-synthesis",
                    "trigger": {"type": "schedule", "at": "22:00", "days": "weekdays"},
                    "prompt": "__builtin__:kg_synthesis",
                }
            ],
        }
        temp_json_file.write_text(json.dumps(data))
        loader = ActionLoader(temp_json_file)
        loader.ensure_defaults()

        actions = loader.load_actions()
        synthesis_actions = [a for a in actions if a.prompt == "__builtin__:kg_synthesis"]
        assert len(synthesis_actions) == 1
        assert synthesis_actions[0].name == "custom-synthesis"

    def test_ensure_defaults_creates_file(self, temp_json_file):
        assert not temp_json_file.exists()
        loader = ActionLoader(temp_json_file)
        loader.ensure_defaults()
        assert temp_json_file.exists()

    def test_load_event_source_configs_missing(self, temp_json_file):
        temp_json_file.write_text(json.dumps({"version": "2.0", "actions": []}))
        loader = ActionLoader(temp_json_file)
        assert loader.load_event_source_configs() == {}

    def test_load_enabled_and_disabled(self, temp_json_file):
        data = {
            "version": "2.0",
            "actions": [
                {"name": "Enabled", "trigger": {"type": "interval", "every": "5m"}, "prompt": "T", "enabled": True},
                {"name": "Disabled", "trigger": {"type": "interval", "every": "5m"}, "prompt": "T", "enabled": False},
            ],
        }
        temp_json_file.write_text(json.dumps(data))
        loader = ActionLoader(temp_json_file)
        actions = loader.load_actions()

        assert len(actions) == 2
        assert actions[0].enabled is True
        assert actions[1].enabled is False


class TestMigration:
    def test_migrate_from_schedules_json(self, tmp_path):
        old_schedules = tmp_path / "schedules.json"
        new_file = tmp_path / "event-schedules.json"

        old_data = {
            "version": "1.0",
            "event_sources": {"mqtt": {"broker": "localhost"}},
            "monitors": [
                {"name": "Mon1", "prompt": "Check stuff", "interval": "5m", "notify": True},
            ],
            "tasks": [
                {"name": "Task1", "prompt": "mcp://dci/rca", "interval": "morning on weekdays"},
            ],
        }
        old_schedules.write_text(json.dumps(old_data))

        loader = ActionLoader(new_file)
        loader.migrate_from_old_format(old_schedules)

        assert new_file.exists()
        actions = loader.load_actions()
        assert len(actions) == 2

        mon = next(a for a in actions if a.name == "Mon1")
        assert mon.trigger_type == "interval"
        assert mon.trigger["every"] == "5m"
        assert mon.notify is True

        task = next(a for a in actions if a.name == "Task1")
        assert task.trigger_type == "schedule"
        assert task.trigger["at"] == "morning"
        assert task.trigger["days"] == "weekdays"

    def test_migrate_preserves_event_sources(self, tmp_path):
        old_schedules = tmp_path / "schedules.json"
        new_file = tmp_path / "event-schedules.json"

        old_data = {
            "version": "1.0",
            "event_sources": {"mqtt": {"broker": "mqtt.local", "port": 8883}},
            "monitors": [{"name": "Mon", "prompt": "T", "interval": "5m"}],
            "tasks": [],
        }
        old_schedules.write_text(json.dumps(old_data))

        loader = ActionLoader(new_file)
        loader.migrate_from_old_format(old_schedules)

        configs = loader.load_event_source_configs()
        assert configs["mqtt"]["broker"] == "mqtt.local"

    def test_migrate_interval_with_range(self, tmp_path):
        old_schedules = tmp_path / "schedules.json"
        new_file = tmp_path / "event-schedules.json"

        old_data = {
            "version": "1.0",
            "monitors": [
                {"name": "BizHours", "prompt": "Check", "interval": "1h between 9:00 and 18:00 on weekdays"},
            ],
            "tasks": [],
        }
        old_schedules.write_text(json.dumps(old_data))

        loader = ActionLoader(new_file)
        loader.migrate_from_old_format(old_schedules)

        actions = loader.load_actions()
        assert len(actions) == 1
        assert actions[0].trigger_type == "interval_range"
        assert actions[0].trigger["every"] == "1h"
        assert actions[0].trigger["between"] == "9:00"
        assert actions[0].trigger["and"] == "18:00"
        assert actions[0].trigger["days"] == "weekdays"

    def test_migrate_scheduled_actions(self, tmp_path):
        old_actions_file = tmp_path / "scheduled-actions.json"
        new_file = tmp_path / "event-schedules.json"

        old_data = {
            "version": "1.0",
            "actions": [
                {
                    "id": "act-001",
                    "prompt": "Check deploy",
                    "scheduled_at": "2026-05-07T14:00:00",
                    "created_at": "2026-05-07T10:00:00",
                    "created_by": "agent",
                    "status": "pending",
                    "execute_query": True,
                    "notify": True,
                },
            ],
        }
        old_actions_file.write_text(json.dumps(old_data))

        loader = ActionLoader(new_file)
        loader.migrate_scheduled_actions(old_actions_file)

        actions = loader.load_actions()
        assert len(actions) == 1
        assert actions[0].trigger_type == "once"
        assert actions[0].trigger["at"] == "2026-05-07T14:00:00"
        assert actions[0].created_by == "agent"
        assert actions[0].status == "pending"

    def test_migration_skips_already_present(self, tmp_path):
        old_schedules = tmp_path / "schedules.json"
        new_file = tmp_path / "event-schedules.json"

        old_schedules.write_text(
            json.dumps({"monitors": [{"name": "Mon1", "prompt": "T", "interval": "5m"}], "tasks": []})
        )
        new_file.write_text(
            json.dumps(
                {
                    "version": "2.0",
                    "actions": [{"name": "Mon1", "trigger": {"type": "interval", "every": "5m"}, "prompt": "T"}],
                }
            )
        )

        loader = ActionLoader(new_file)
        loader.migrate_from_old_format(old_schedules)

        actions = loader.load_actions()
        assert len(actions) == 1
