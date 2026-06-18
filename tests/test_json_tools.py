"""Tests for json_tools — internal__json_query wrapping jq."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ai_assist.json_tools import JsonTools


@pytest.fixture
def filesystem_tools():
    """Mock filesystem tools with path validation."""
    ft = AsyncMock()
    ft._validate_path = AsyncMock(return_value=None)
    return ft


@pytest.fixture
def json_tools(filesystem_tools):
    return JsonTools(filesystem_tools=filesystem_tools)


@pytest.fixture
def sample_json_file():
    data = {"name": "test", "count": 42, "items": [{"id": 1, "status": "ok"}, {"id": 2, "status": "failed"}]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


requires_jq = pytest.mark.skipif(not shutil.which("jq"), reason="jq not installed")


class TestToolDefinitions:
    @requires_jq
    def test_tool_definitions_with_jq(self, json_tools):
        defs = json_tools.get_tool_definitions()
        assert len(defs) == 1
        assert defs[0]["name"] == "internal__json_query"
        assert defs[0]["_server"] == "internal"
        assert defs[0]["_original_name"] == "json_query"
        schema = defs[0]["input_schema"]
        assert "file" in schema["properties"]
        assert "filter" in schema["properties"]
        assert schema["required"] == ["file", "filter"]

    def test_tool_definitions_without_jq(self, filesystem_tools):
        with patch("ai_assist.json_tools.shutil.which", return_value=None):
            tools = JsonTools(filesystem_tools=filesystem_tools)
        assert tools.get_tool_definitions() == []


@requires_jq
class TestJsonQuery:
    @pytest.mark.asyncio
    async def test_basic_key_extraction(self, json_tools, sample_json_file):
        result = await json_tools.execute_tool("json_query", {"file": sample_json_file, "filter": ".name"})
        assert json.loads(result.strip()) == "test"

    @pytest.mark.asyncio
    async def test_nested_field(self, json_tools, sample_json_file):
        result = await json_tools.execute_tool("json_query", {"file": sample_json_file, "filter": ".count"})
        assert json.loads(result.strip()) == 42

    @pytest.mark.asyncio
    async def test_array_filter(self, json_tools, sample_json_file):
        result = await json_tools.execute_tool(
            "json_query", {"file": sample_json_file, "filter": '[.items[] | select(.status == "failed")]'}
        )
        parsed = json.loads(result.strip())
        assert len(parsed) == 1
        assert parsed[0]["id"] == 2

    @pytest.mark.asyncio
    async def test_field_extraction(self, json_tools, sample_json_file):
        result = await json_tools.execute_tool(
            "json_query", {"file": sample_json_file, "filter": "[.items[] | {id, status}]"}
        )
        parsed = json.loads(result.strip())
        assert len(parsed) == 2
        assert parsed[0] == {"id": 1, "status": "ok"}

    @pytest.mark.asyncio
    async def test_raw_output(self, json_tools, sample_json_file):
        result = await json_tools.execute_tool(
            "json_query", {"file": sample_json_file, "filter": ".name", "raw_output": True}
        )
        assert result.strip() == "test"

    @pytest.mark.asyncio
    async def test_invalid_filter(self, json_tools, sample_json_file):
        result = await json_tools.execute_tool("json_query", {"file": sample_json_file, "filter": ".["})
        assert result.startswith("jq error:")

    @pytest.mark.asyncio
    async def test_file_not_found(self, json_tools):
        result = await json_tools.execute_tool("json_query", {"file": "/nonexistent/file.json", "filter": "."})
        assert "error" in result.lower() or "Error" in result

    @pytest.mark.asyncio
    async def test_path_validation_rejected(self, json_tools, sample_json_file):
        json_tools.filesystem_tools._validate_path = AsyncMock(return_value="Error: Path not allowed")
        result = await json_tools.execute_tool("json_query", {"file": sample_json_file, "filter": "."})
        assert "Error: Path not allowed" in result

    @pytest.mark.asyncio
    async def test_unknown_tool_name(self, json_tools):
        result = await json_tools.execute_tool("unknown_tool", {"file": "x", "filter": "."})
        assert "Error" in result


@requires_jq
class TestFilterString:
    def test_basic_key_extraction(self, json_tools):
        data = json.dumps({"name": "test", "count": 42})
        result = json_tools.filter_string(data, ".name")
        assert json.loads(result.strip()) == "test"

    def test_array_select(self, json_tools):
        data = json.dumps([{"id": 1, "status": "ok"}, {"id": 2, "status": "failed"}])
        result = json_tools.filter_string(data, '[.[] | select(.status == "failed")]')
        parsed = json.loads(result.strip())
        assert len(parsed) == 1
        assert parsed[0]["id"] == 2

    def test_invalid_filter(self, json_tools):
        result = json_tools.filter_string('{"a": 1}', ".[")
        assert result.startswith("jq error:")

    def test_non_json_input(self, json_tools):
        result = json_tools.filter_string("not json", ".name")
        assert result.startswith("jq error:")

    def test_jq_not_installed(self, filesystem_tools):
        tools = JsonTools(filesystem_tools=filesystem_tools)
        tools.jq_path = None
        result = tools.filter_string('{"a": 1}', ".")
        assert "jq is not installed" in result

    def test_timeout(self, json_tools):
        with patch("ai_assist.json_tools.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="jq", timeout=30)):
            result = json_tools.filter_string('{"a": 1}', ".")
        assert "timed out" in result
