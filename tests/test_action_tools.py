"""Tests for action tools (agent CRUD for event-schedules.json)"""

import json
import tempfile
from pathlib import Path

import pytest

from ai_assist.action_tools import ActionTools


@pytest.fixture
def temp_json_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "event-schedules.json"


@pytest.fixture
def tools(temp_json_file):
    return ActionTools(schedules_file=temp_json_file)


@pytest.fixture
def seeded_tools(temp_json_file):
    data = {
        "version": "2.0",
        "actions": [
            {"name": "Monitor", "trigger": {"type": "interval", "every": "5m"}, "prompt": "Check stuff"},
            {"name": "Alert", "trigger": {"type": "mqtt", "topic": "alerts/#"}, "prompt": "Handle alert"},
        ],
    }
    temp_json_file.write_text(json.dumps(data))
    return ActionTools(schedules_file=temp_json_file)


class TestCreateAction:
    @pytest.mark.asyncio
    async def test_create_interval_action(self, tools, temp_json_file):
        result = await tools.execute_tool(
            "internal__create_action",
            {"name": "DCI Check", "trigger": {"type": "interval", "every": "10m"}, "prompt": "Check DCI"},
        )
        assert "Created" in result
        data = json.loads(temp_json_file.read_text())
        assert len(data["actions"]) == 1
        assert data["actions"][0]["name"] == "DCI Check"

    @pytest.mark.asyncio
    async def test_create_mqtt_action(self, tools):
        result = await tools.execute_tool(
            "internal__create_action",
            {"name": "Alert Handler", "trigger": {"type": "mqtt", "topic": "alerts/#"}, "prompt": "Handle alert"},
        )
        assert "Created" in result
        assert "mqtt" in result

    @pytest.mark.asyncio
    async def test_create_duplicate_name(self, seeded_tools):
        result = await seeded_tools.execute_tool(
            "internal__create_action",
            {"name": "Monitor", "trigger": {"type": "interval", "every": "1h"}, "prompt": "New"},
        )
        assert "already exists" in result

    @pytest.mark.asyncio
    async def test_create_invalid_trigger(self, tools):
        result = await tools.execute_tool(
            "internal__create_action",
            {"name": "Bad", "trigger": {"type": "interval"}, "prompt": "Test"},
        )
        assert "Invalid" in result

    @pytest.mark.asyncio
    async def test_create_once_action(self, tools):
        result = await tools.execute_tool(
            "internal__create_action",
            {
                "name": "Reminder",
                "trigger": {"type": "once", "at": "2026-05-07T14:00:00"},
                "prompt": "Check deploy",
                "notify": True,
            },
        )
        assert "Created" in result


class TestListActions:
    @pytest.mark.asyncio
    async def test_list_all(self, seeded_tools):
        result = await seeded_tools.execute_tool("internal__list_actions", {})
        assert "Monitor" in result
        assert "Alert" in result

    @pytest.mark.asyncio
    async def test_list_filtered(self, seeded_tools):
        result = await seeded_tools.execute_tool("internal__list_actions", {"trigger_type": "mqtt"})
        assert "Alert" in result
        assert "Monitor" not in result

    @pytest.mark.asyncio
    async def test_list_empty(self, tools):
        result = await tools.execute_tool("internal__list_actions", {})
        assert "No actions" in result


class TestUpdateAction:
    @pytest.mark.asyncio
    async def test_update_prompt(self, seeded_tools):
        result = await seeded_tools.execute_tool(
            "internal__update_action",
            {"name": "Monitor", "prompt": "New prompt"},
        )
        assert "Updated" in result

    @pytest.mark.asyncio
    async def test_update_trigger(self, seeded_tools):
        result = await seeded_tools.execute_tool(
            "internal__update_action",
            {"name": "Monitor", "trigger": {"type": "interval", "every": "10m"}},
        )
        assert "Updated" in result

    @pytest.mark.asyncio
    async def test_update_not_found(self, seeded_tools):
        result = await seeded_tools.execute_tool(
            "internal__update_action",
            {"name": "Nonexistent", "prompt": "New"},
        )
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_update_invalid(self, seeded_tools):
        result = await seeded_tools.execute_tool(
            "internal__update_action",
            {"name": "Monitor", "trigger": {"type": "unknown"}},
        )
        assert "Invalid" in result


class TestDeleteAction:
    @pytest.mark.asyncio
    async def test_delete(self, seeded_tools, temp_json_file):
        result = await seeded_tools.execute_tool("internal__delete_action", {"name": "Monitor"})
        assert "Deleted" in result
        data = json.loads(temp_json_file.read_text())
        assert len(data["actions"]) == 1

    @pytest.mark.asyncio
    async def test_delete_not_found(self, seeded_tools):
        result = await seeded_tools.execute_tool("internal__delete_action", {"name": "Ghost"})
        assert "not found" in result


class TestEnableAction:
    @pytest.mark.asyncio
    async def test_disable(self, seeded_tools):
        result = await seeded_tools.execute_tool("internal__enable_action", {"name": "Monitor", "enabled": False})
        assert "disabled" in result

    @pytest.mark.asyncio
    async def test_enable(self, seeded_tools):
        result = await seeded_tools.execute_tool("internal__enable_action", {"name": "Monitor", "enabled": True})
        assert "enabled" in result

    @pytest.mark.asyncio
    async def test_enable_not_found(self, seeded_tools):
        result = await seeded_tools.execute_tool("internal__enable_action", {"name": "Ghost", "enabled": True})
        assert "not found" in result


class TestGetActionStatus:
    @pytest.mark.asyncio
    async def test_status(self, seeded_tools):
        result = await seeded_tools.execute_tool("internal__get_action_status", {"name": "Monitor"})
        assert "Monitor" in result
        assert "interval" in result

    @pytest.mark.asyncio
    async def test_status_not_found(self, seeded_tools):
        result = await seeded_tools.execute_tool("internal__get_action_status", {"name": "Ghost"})
        assert "not found" in result
