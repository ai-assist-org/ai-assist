"""Tests for unified action model and trigger matcher"""

from datetime import datetime

import pytest

from ai_assist.action_model import ActionDefinition, TriggerMatcher
from ai_assist.event_sources import EventContext


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


class TestActionDefinition:
    def test_construction_interval(self):
        action = _make_action()
        assert action.name == "test-action"
        assert action.trigger_type == "interval"
        assert action.is_time_based is True
        assert action.is_event_based is False

    def test_construction_schedule(self):
        action = _make_action(trigger={"type": "schedule", "at": "9:00", "days": "weekdays"})
        assert action.trigger_type == "schedule"
        assert action.is_time_based is True

    def test_construction_mqtt(self):
        action = _make_action(trigger={"type": "mqtt", "topic": "alerts/#"})
        assert action.trigger_type == "mqtt"
        assert action.is_event_based is True
        assert action.is_time_based is False

    def test_construction_dbus(self):
        action = _make_action(
            trigger={"type": "dbus", "interface": "org.freedesktop.UDisks2", "signal": "InterfacesAdded"}
        )
        assert action.trigger_type == "dbus"
        assert action.is_event_based is True

    def test_construction_file(self):
        action = _make_action(trigger={"type": "file", "path": "/tmp/watch.txt"})
        assert action.trigger_type == "file"
        assert action.is_event_based is True

    def test_construction_file_glob(self):
        action = _make_action(trigger={"type": "file", "path": "~/Downloads/*.pdf"})
        assert action.trigger_type == "file"
        assert action.is_event_based is True

    def test_construction_once(self):
        action = _make_action(trigger={"type": "once", "at": "2026-05-07T14:00:00"})
        assert action.trigger_type == "once"
        assert action.is_time_based is True

    def test_construction_interval_range(self):
        action = _make_action(
            trigger={"type": "interval_range", "every": "1h", "between": "9:00", "and": "18:00", "days": "weekdays"}
        )
        assert action.trigger_type == "interval_range"
        assert action.is_time_based is True

    def test_default_values(self):
        action = _make_action()
        assert action.enabled is True
        assert action.prompt_arguments is None
        assert action.max_turns == 100
        assert action.conditions == []
        assert action.notify is False
        assert action.notification_channels == ["console"]
        assert action.created_by is None
        assert action.status is None

    def test_mcp_prompt_detection(self):
        action = _make_action(prompt="mcp://dci/rca")
        assert action.is_mcp_prompt is True

        action2 = _make_action(prompt="Check failures")
        assert action2.is_mcp_prompt is False

    def test_mcp_prompt_parsing(self):
        action = _make_action(prompt="mcp://dci/rca")
        server, prompt = action.parse_mcp_prompt()
        assert server == "dci"
        assert prompt == "rca"

    def test_validate_interval(self):
        action = _make_action(trigger={"type": "interval", "every": "5m"})
        action.validate_definition()

    def test_validate_schedule(self):
        action = _make_action(trigger={"type": "schedule", "at": "9:00", "days": "weekdays"})
        action.validate_definition()

    def test_validate_mqtt(self):
        action = _make_action(trigger={"type": "mqtt", "topic": "alerts/#"})
        action.validate_definition()

    def test_validate_dbus(self):
        action = _make_action(
            trigger={"type": "dbus", "interface": "org.freedesktop.UDisks2", "signal": "InterfacesAdded"}
        )
        action.validate_definition()

    def test_validate_file(self):
        action = _make_action(trigger={"type": "file", "path": "/tmp/watch.txt"})
        action.validate_definition()

    def test_validate_file_missing_path(self):
        with pytest.raises(ValueError, match="path"):
            action = _make_action(trigger={"type": "file"})
            action.validate_definition()

    def test_validate_once(self):
        action = _make_action(trigger={"type": "once", "at": "2026-05-07T14:00:00"})
        action.validate_definition()

    def test_validate_missing_name(self):
        with pytest.raises(ValueError, match="name"):
            action = _make_action(name="")
            action.validate_definition()

    def test_validate_missing_prompt(self):
        with pytest.raises(ValueError, match="prompt"):
            action = _make_action(prompt="")
            action.validate_definition()

    def test_validate_missing_trigger_type(self):
        with pytest.raises(ValueError, match="type"):
            action = _make_action(trigger={"topic": "alerts/#"})
            action.validate_definition()

    def test_validate_unknown_trigger_type(self):
        with pytest.raises(ValueError, match="Unknown trigger type"):
            action = _make_action(trigger={"type": "unknown"})
            action.validate_definition()

    def test_validate_interval_missing_every(self):
        with pytest.raises(ValueError, match="every"):
            action = _make_action(trigger={"type": "interval"})
            action.validate_definition()

    def test_validate_invalid_interval_format(self):
        with pytest.raises(ValueError, match="Invalid interval"):
            action = _make_action(trigger={"type": "interval", "every": "invalid"})
            action.validate_definition()

    def test_validate_schedule_missing_at(self):
        with pytest.raises(ValueError, match="at"):
            action = _make_action(trigger={"type": "schedule", "days": "weekdays"})
            action.validate_definition()

    def test_validate_schedule_missing_days(self):
        with pytest.raises(ValueError, match="days"):
            action = _make_action(trigger={"type": "schedule", "at": "9:00"})
            action.validate_definition()

    def test_validate_mqtt_missing_topic(self):
        with pytest.raises(ValueError, match="topic"):
            action = _make_action(trigger={"type": "mqtt"})
            action.validate_definition()

    def test_validate_dbus_missing_interface(self):
        with pytest.raises(ValueError, match="interface"):
            action = _make_action(trigger={"type": "dbus", "signal": "InterfacesAdded"})
            action.validate_definition()

    def test_validate_dbus_missing_signal(self):
        with pytest.raises(ValueError, match="signal"):
            action = _make_action(trigger={"type": "dbus", "interface": "org.freedesktop.UDisks2"})
            action.validate_definition()

    def test_validate_once_missing_at(self):
        with pytest.raises(ValueError, match="at"):
            action = _make_action(trigger={"type": "once"})
            action.validate_definition()

    def test_validate_mcp_prompt_format(self):
        with pytest.raises(ValueError, match="Invalid MCP"):
            action = _make_action(prompt="mcp://invalid", trigger={"type": "interval", "every": "5m"})
            action.validate_definition()

    def test_from_dict(self):
        data = {
            "name": "Alert Handler",
            "trigger": {"type": "mqtt", "topic": "alerts/#"},
            "prompt": "Handle alert",
            "enabled": True,
            "notify": True,
        }
        action = ActionDefinition.from_dict(data)
        assert action.name == "Alert Handler"
        assert action.trigger_type == "mqtt"
        assert action.notify is True

    def test_from_dict_defaults(self):
        data = {
            "name": "Minimal",
            "trigger": {"type": "interval", "every": "5m"},
            "prompt": "Test",
        }
        action = ActionDefinition.from_dict(data)
        assert action.enabled is True
        assert action.notify is False
        assert action.conditions == []

    def test_to_dict(self):
        action = _make_action(name="test", notify=True)
        d = action.to_dict()
        assert d["name"] == "test"
        assert d["trigger"] == {"type": "interval", "every": "5m"}
        assert d["notify"] is True


class TestTriggerMatcher:
    def test_mqtt_source_match(self):
        matcher = TriggerMatcher()
        event = _make_event()
        trigger = {"type": "mqtt", "topic": "alerts/cpu"}
        assert matcher.matches(event, trigger) is True

    def test_mqtt_source_mismatch(self):
        matcher = TriggerMatcher()
        event = _make_event(source_type="dbus")
        trigger = {"type": "mqtt", "topic": "alerts/#"}
        assert matcher.matches(event, trigger) is False

    def test_mqtt_topic_exact_match(self):
        matcher = TriggerMatcher()
        event = _make_event(metadata={"topic": "alerts/cpu"})
        trigger = {"type": "mqtt", "topic": "alerts/cpu"}
        assert matcher.matches(event, trigger) is True

    def test_mqtt_topic_no_match(self):
        matcher = TriggerMatcher()
        event = _make_event(metadata={"topic": "status/ok"})
        trigger = {"type": "mqtt", "topic": "alerts/cpu"}
        assert matcher.matches(event, trigger) is False

    def test_mqtt_topic_wildcard_hash(self):
        matcher = TriggerMatcher()
        event = _make_event(metadata={"topic": "alerts/cpu/server42"})
        trigger = {"type": "mqtt", "topic": "alerts/#"}
        assert matcher.matches(event, trigger) is True

    def test_mqtt_topic_wildcard_plus(self):
        matcher = TriggerMatcher()
        event = _make_event(metadata={"topic": "alerts/cpu"})
        trigger = {"type": "mqtt", "topic": "alerts/+"}
        assert matcher.matches(event, trigger) is True

    def test_mqtt_topic_wildcard_plus_no_match_deeper(self):
        matcher = TriggerMatcher()
        event = _make_event(metadata={"topic": "alerts/cpu/high"})
        trigger = {"type": "mqtt", "topic": "alerts/+"}
        assert matcher.matches(event, trigger) is False

    def test_dbus_match(self):
        matcher = TriggerMatcher()
        event = _make_event(
            source_type="dbus",
            metadata={"interface": "org.freedesktop.UDisks2", "signal": "InterfacesAdded", "path": "/org/fd/UDisks2"},
        )
        trigger = {"type": "dbus", "interface": "org.freedesktop.UDisks2", "signal": "InterfacesAdded"}
        assert matcher.matches(event, trigger) is True

    def test_dbus_mismatch_signal(self):
        matcher = TriggerMatcher()
        event = _make_event(
            source_type="dbus",
            metadata={"interface": "org.freedesktop.UDisks2", "signal": "InterfacesRemoved"},
        )
        trigger = {"type": "dbus", "interface": "org.freedesktop.UDisks2", "signal": "InterfacesAdded"}
        assert matcher.matches(event, trigger) is False

    def test_dbus_path_filter_match(self):
        matcher = TriggerMatcher()
        event = _make_event(
            source_type="dbus",
            metadata={"interface": "org.fd.UDisks2", "signal": "Added", "path": "/org/fd/UDisks2"},
        )
        trigger = {"type": "dbus", "interface": "org.fd.UDisks2", "signal": "Added", "path": "/org/fd/UDisks2"}
        assert matcher.matches(event, trigger) is True

    def test_dbus_path_filter_mismatch(self):
        matcher = TriggerMatcher()
        event = _make_event(
            source_type="dbus",
            metadata={"interface": "org.fd.UDisks2", "signal": "Added", "path": "/wrong"},
        )
        trigger = {"type": "dbus", "interface": "org.fd.UDisks2", "signal": "Added", "path": "/org/fd/UDisks2"}
        assert matcher.matches(event, trigger) is False

    def test_payload_contains(self):
        matcher = TriggerMatcher()
        event = _make_event(payload="CRITICAL: CPU usage 95%")
        trigger = {"type": "mqtt", "topic": "alerts/cpu", "payload_contains": "CRITICAL"}
        assert matcher.matches(event, trigger) is True

    def test_payload_contains_no_match(self):
        matcher = TriggerMatcher()
        event = _make_event(payload="INFO: CPU usage 50%")
        trigger = {"type": "mqtt", "topic": "alerts/cpu", "payload_contains": "CRITICAL"}
        assert matcher.matches(event, trigger) is False

    def test_payload_regex(self):
        matcher = TriggerMatcher()
        event = _make_event(payload="ERROR: disk usage 95%")
        trigger = {"type": "mqtt", "topic": "alerts/cpu", "payload_regex": r".*\d{2}%$"}
        assert matcher.matches(event, trigger) is True

    def test_payload_regex_no_match(self):
        matcher = TriggerMatcher()
        event = _make_event(payload="all clear")
        trigger = {"type": "mqtt", "topic": "alerts/cpu", "payload_regex": r"ERROR.*"}
        assert matcher.matches(event, trigger) is False

    def test_file_match_exact(self):
        matcher = TriggerMatcher()
        event = _make_event(source_type="file", metadata={"path": "/tmp/config.json"})
        trigger = {"type": "file", "path": "/tmp/config.json"}
        assert matcher.matches(event, trigger) is True

    def test_file_no_match(self):
        matcher = TriggerMatcher()
        event = _make_event(source_type="file", metadata={"path": "/tmp/other.json"})
        trigger = {"type": "file", "path": "/tmp/config.json"}
        assert matcher.matches(event, trigger) is False

    def test_file_glob_match(self):
        matcher = TriggerMatcher()
        event = _make_event(source_type="file", metadata={"path": "/tmp/report.pdf"})
        trigger = {"type": "file", "path": "/tmp/*.pdf"}
        assert matcher.matches(event, trigger) is True

    def test_file_glob_no_match(self):
        matcher = TriggerMatcher()
        event = _make_event(source_type="file", metadata={"path": "/tmp/report.txt"})
        trigger = {"type": "file", "path": "/tmp/*.pdf"}
        assert matcher.matches(event, trigger) is False

    def test_file_source_mismatch(self):
        matcher = TriggerMatcher()
        event = _make_event(source_type="mqtt", metadata={"topic": "alerts/cpu"})
        trigger = {"type": "file", "path": "/tmp/config.json"}
        assert matcher.matches(event, trigger) is False

    def test_time_trigger_never_matches_events(self):
        matcher = TriggerMatcher()
        event = _make_event()
        trigger = {"type": "interval", "every": "5m"}
        assert matcher.matches(event, trigger) is False
