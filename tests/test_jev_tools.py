"""Tests for the jev decision tool (ad-hoc interactive jev access)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from ai_assist.jev_client import JevError
from ai_assist.jev_tools import JevTools


def _config():
    return SimpleNamespace(
        jev_enabled=True,
        jev_api_key="test-key",
        jev_api_url="https://example.test/v1/systemone",
        jev_model="jev-latest",
    )


class TestJevToolDefinitions:
    """Test tool definition structure."""

    def test_get_tool_definitions_returns_three_tools(self):
        defs = JevTools(_config()).get_tool_definitions()
        assert len(defs) == 3

    def test_tool_names(self):
        names = {d["name"] for d in JevTools(_config()).get_tool_definitions()}
        assert names == {"internal__jev_decide", "internal__jev_choose", "internal__jev_score"}

    def test_decide_schema_requires_state_and_question(self):
        schema = _schema("internal__jev_decide")
        assert schema["type"] == "object"
        assert set(schema["required"]) == {"state", "question"}

    def test_choose_schema_requires_options(self):
        schema = _schema("internal__jev_choose")
        assert set(schema["required"]) == {"state", "question", "options"}
        assert schema["properties"]["options"]["type"] == "object"

    def test_score_schema_requires_levels(self):
        schema = _schema("internal__jev_score")
        assert set(schema["required"]) == {"state", "question", "levels"}
        assert schema["properties"]["levels"]["type"] == "array"

    def test_all_tools_internal_and_readonly(self):
        for d in JevTools(_config()).get_tool_definitions():
            assert d["_server"] == "internal"
            assert d["_readonly"] is True
            assert d["_original_name"] == d["name"].removeprefix("internal__")


def _schema(name: str) -> dict:
    for d in JevTools(_config()).get_tool_definitions():
        if d["name"] == name:
            return d["input_schema"]
    raise AssertionError(f"tool {name} not found")


class TestJevToolExecution:
    """Test tool execution against a mocked jev client."""

    @pytest.mark.asyncio
    async def test_yes_verdict(self):
        tools = JevTools(_config())
        response = {"answers": {"decision": {"type": "noul", "noul": 0.91}}}
        with patch("ai_assist.jev_tools.jev_decide", new=AsyncMock(return_value=response)):
            result = await tools.execute_tool(
                "jev_decide", {"state": "All 42 tests passed.", "question": "Did the tests pass?"}
            )
        assert "yes" in result
        assert "0.91" in result

    @pytest.mark.asyncio
    async def test_no_verdict_below_threshold(self):
        tools = JevTools(_config())
        response = {"answers": {"decision": {"type": "noul", "noul": 0.20}}}
        with patch("ai_assist.jev_tools.jev_decide", new=AsyncMock(return_value=response)):
            result = await tools.execute_tool(
                "jev_decide", {"state": "3 tests failed.", "question": "Did the tests pass?"}
            )
        assert "no" in result
        assert "0.20" in result

    @pytest.mark.asyncio
    async def test_passes_state_and_question_to_client(self):
        tools = JevTools(_config())
        response = {"answers": {"decision": {"type": "noul", "noul": 0.5}}}
        mock = AsyncMock(return_value=response)
        with patch("ai_assist.jev_tools.jev_decide", new=mock):
            await tools.execute_tool("jev_decide", {"state": "the state", "question": "ok?"})
        _, kwargs = mock.call_args
        assert kwargs["state"] == "the state"
        assert kwargs["questions"]["decision"]["instructions"] == "ok?"

    @pytest.mark.asyncio
    async def test_missing_arguments(self):
        tools = JevTools(_config())
        result = await tools.execute_tool("jev_decide", {"state": "", "question": "ok?"})
        assert result.startswith("Error")

    @pytest.mark.asyncio
    async def test_unknown_tool_name(self):
        tools = JevTools(_config())
        result = await tools.execute_tool("not_jev", {"state": "s", "question": "q"})
        assert result.startswith("Error")

    @pytest.mark.asyncio
    async def test_client_error_surfaces_cleanly(self):
        tools = JevTools(_config())
        with patch("ai_assist.jev_tools.jev_decide", new=AsyncMock(side_effect=JevError("boom"))):
            result = await tools.execute_tool("jev_decide", {"state": "s", "question": "q"})
        assert result.startswith("Error")
        assert "boom" in result

    @pytest.mark.asyncio
    async def test_no_probability_returns_error(self):
        tools = JevTools(_config())
        with patch("ai_assist.jev_tools.jev_decide", new=AsyncMock(return_value={"answers": {}})):
            result = await tools.execute_tool("jev_decide", {"state": "s", "question": "q"})
        assert result.startswith("Error")


class TestJevChoose:
    """Test the Choice primitive."""

    @pytest.mark.asyncio
    async def test_returns_winning_option_and_distribution(self):
        tools = JevTools(_config())
        response = {
            "answers": {
                "decision": {
                    "type": "choice",
                    "choice": "negative",
                    "probabilities": {"positive": 0.0, "neutral": 0.0, "negative": 1.0},
                    "confidence": 1.0,
                }
            }
        }
        with patch("ai_assist.jev_tools.jev_decide", new=AsyncMock(return_value=response)):
            result = await tools.execute_tool(
                "jev_choose",
                {
                    "state": "an angry email",
                    "question": "sentiment?",
                    "options": {"positive": "p", "negative": "n", "neutral": "x"},
                },
            )
        assert "negative" in result
        assert "negative=1.00" in result

    @pytest.mark.asyncio
    async def test_accepts_options_as_list(self):
        tools = JevTools(_config())
        mock = AsyncMock(
            return_value={
                "answers": {"decision": {"type": "choice", "choice": "a", "probabilities": {}, "confidence": 1.0}}
            }
        )
        with patch("ai_assist.jev_tools.jev_decide", new=mock):
            await tools.execute_tool("jev_choose", {"state": "s", "question": "q", "options": ["a", "b"]})
        _, kwargs = mock.call_args
        assert kwargs["questions"]["decision"]["criteria"] == {"a": "a", "b": "b"}

    @pytest.mark.asyncio
    async def test_requires_at_least_two_options(self):
        tools = JevTools(_config())
        result = await tools.execute_tool("jev_choose", {"state": "s", "question": "q", "options": {"a": "only"}})
        assert result.startswith("Error")


class TestJevScore:
    """Test the Score primitive."""

    @pytest.mark.asyncio
    async def test_returns_score_and_nearest_level(self):
        tools = JevTools(_config())
        response = {
            "answers": {
                "decision": {
                    "type": "score",
                    "score": 3.22,
                    "legend": {"0": "very low", "1": "low", "2": "medium", "3": "high", "4": "critical"},
                    "probabilities": {"3": 0.8, "4": 0.2},
                    "confidence": 0.81,
                }
            }
        }
        with patch("ai_assist.jev_tools.jev_decide", new=AsyncMock(return_value=response)):
            result = await tools.execute_tool(
                "jev_score",
                {
                    "state": "an angry email",
                    "question": "urgency?",
                    "levels": ["very low", "low", "medium", "high", "critical"],
                },
            )
        assert "3.22" in result
        assert "high" in result

    @pytest.mark.asyncio
    async def test_passes_levels_as_criteria(self):
        tools = JevTools(_config())
        mock = AsyncMock(
            return_value={
                "answers": {
                    "decision": {"type": "score", "score": 1.0, "legend": {}, "probabilities": {}, "confidence": 0.5}
                }
            }
        )
        with patch("ai_assist.jev_tools.jev_decide", new=mock):
            await tools.execute_tool("jev_score", {"state": "s", "question": "q", "levels": ["low", "high"]})
        _, kwargs = mock.call_args
        assert kwargs["questions"]["decision"]["criteria"] == ["low", "high"]

    @pytest.mark.asyncio
    async def test_requires_at_least_two_levels(self):
        tools = JevTools(_config())
        result = await tools.execute_tool("jev_score", {"state": "s", "question": "q", "levels": ["only"]})
        assert result.startswith("Error")
