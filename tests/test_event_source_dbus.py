"""Tests for D-Bus event source"""

from unittest.mock import MagicMock

import pytest

dbus_next = pytest.importorskip("dbus_next")

from ai_assist.event_source_dbus import DbusEventSource  # noqa: E402


@pytest.fixture
def dbus_source():
    return DbusEventSource({"bus": "session"})


class TestDbusEventSource:
    def test_name(self, dbus_source):
        assert dbus_source.name == "dbus"

    def test_subscribe(self, dbus_source):
        dbus_source.subscribe(
            "task1",
            {
                "type": "dbus",
                "interface": "org.freedesktop.UDisks2",
                "signal": "InterfacesAdded",
            },
        )
        assert len(dbus_source._subscriptions) == 1
        assert dbus_source._subscriptions[0][0] == "task1"

    def test_unsubscribe_all(self, dbus_source):
        dbus_source.subscribe("task1", {"type": "dbus", "interface": "a", "signal": "b"})
        dbus_source.unsubscribe_all()
        assert len(dbus_source._subscriptions) == 0

    def test_config_defaults(self):
        source = DbusEventSource({})
        assert source.default_bus_type == "session"

    def test_config_system_bus(self):
        source = DbusEventSource({"bus": "system"})
        assert source.default_bus_type == "system"

    def test_build_match_rule(self):
        rule = DbusEventSource._build_match_rule(
            {
                "interface": "org.freedesktop.UDisks2",
                "signal": "InterfacesAdded",
                "path": "/org/freedesktop/UDisks2",
            }
        )
        assert "type='signal'" in rule
        assert "interface='org.freedesktop.UDisks2'" in rule
        assert "member='InterfacesAdded'" in rule
        assert "path='/org/freedesktop/UDisks2'" in rule

    def test_build_match_rule_without_path(self):
        rule = DbusEventSource._build_match_rule(
            {
                "interface": "org.freedesktop.login1.Manager",
                "signal": "PrepareForSleep",
            }
        )
        assert "path=" not in rule

    def test_signal_matches(self):
        msg = MagicMock()
        msg.interface = "org.freedesktop.UDisks2"
        msg.member = "InterfacesAdded"
        msg.path = "/org/freedesktop/UDisks2"

        trigger = {
            "interface": "org.freedesktop.UDisks2",
            "signal": "InterfacesAdded",
        }
        assert DbusEventSource._signal_matches(msg, trigger) is True

    def test_signal_no_match_wrong_interface(self):
        msg = MagicMock()
        msg.interface = "org.freedesktop.Other"
        msg.member = "InterfacesAdded"

        trigger = {
            "interface": "org.freedesktop.UDisks2",
            "signal": "InterfacesAdded",
        }
        assert DbusEventSource._signal_matches(msg, trigger) is False

    def test_signal_no_match_wrong_signal(self):
        msg = MagicMock()
        msg.interface = "org.freedesktop.UDisks2"
        msg.member = "InterfacesRemoved"

        trigger = {
            "interface": "org.freedesktop.UDisks2",
            "signal": "InterfacesAdded",
        }
        assert DbusEventSource._signal_matches(msg, trigger) is False

    def test_signal_match_with_path_filter(self):
        msg = MagicMock()
        msg.interface = "org.freedesktop.UDisks2"
        msg.member = "InterfacesAdded"
        msg.path = "/org/freedesktop/UDisks2"

        trigger = {
            "interface": "org.freedesktop.UDisks2",
            "signal": "InterfacesAdded",
            "path": "/org/freedesktop/UDisks2",
        }
        assert DbusEventSource._signal_matches(msg, trigger) is True

    def test_signal_no_match_wrong_path(self):
        msg = MagicMock()
        msg.interface = "org.freedesktop.UDisks2"
        msg.member = "InterfacesAdded"
        msg.path = "/wrong/path"

        trigger = {
            "interface": "org.freedesktop.UDisks2",
            "signal": "InterfacesAdded",
            "path": "/org/freedesktop/UDisks2",
        }
        assert DbusEventSource._signal_matches(msg, trigger) is False

    @pytest.mark.asyncio
    async def test_stop_without_start(self, dbus_source):
        await dbus_source.stop()  # should not raise
