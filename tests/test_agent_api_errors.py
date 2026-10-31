"""Tests for agent API error handling."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import APIConnectionError, BadRequestError, RateLimitError
from httpx import Response

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig


@pytest.mark.asyncio
async def test_context_limit_error_returned_to_agent():
    """Test that context limit errors are returned as text for agent to handle."""
    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"])
    agent = AiAssistAgent(config=config)

    # Mock Anthropic client to raise BadRequestError
    mock_response = MagicMock(spec=Response)
    mock_response.status_code = 400
    mock_response.headers = {"request-id": "test-request-id"}

    error = BadRequestError(
        message="prompt is too long: 250000 tokens > 200000 maximum",
        response=mock_response,
        body={"error": {"message": "prompt is too long: 250000 tokens > 200000 maximum"}},
    )

    with patch.object(agent.anthropic.messages, "create", side_effect=error):
        with patch.object(agent.anthropic.messages, "stream") as mock_stream:
            mock_stream.side_effect = error

            result = await agent.query("test query", max_turns=1)

    # Agent should receive error message, not crash
    assert "too long" in result.lower() or "context" in result.lower()
    assert "save_to_file" in result.lower() or "__save_to_file" in result
    assert "batch" in result.lower() or "chunk" in result.lower()


@pytest.mark.asyncio
async def test_context_limit_error_with_streaming():
    """Test context limit error handling with streaming API."""
    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"], model="claude-opus-4-6")
    agent = AiAssistAgent(config=config)

    mock_response = MagicMock(spec=Response)
    mock_response.status_code = 400
    mock_response.headers = {"request-id": "test-request-id"}

    error = BadRequestError(
        message="messages: text content is too long",
        response=mock_response,
        body={"error": {"message": "messages: text content is too long"}},
    )

    with patch.object(agent.anthropic.messages, "stream") as mock_stream:
        mock_stream.return_value.__enter__.side_effect = error

        result = await agent.query("test query", max_turns=1)

    assert "too long" in result.lower() or "content" in result.lower()
    assert "__save_to_file" in result


@pytest.mark.asyncio
async def test_rate_limit_error_returned_to_agent():
    """Test that rate limit errors are returned as text."""
    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"])
    agent = AiAssistAgent(config=config)

    mock_response = MagicMock(spec=Response)
    mock_response.status_code = 429
    mock_response.headers = {"request-id": "test-request-id"}

    error = RateLimitError(
        message="rate limit exceeded",
        response=mock_response,
        body={"error": {"message": "rate limit exceeded"}},
    )

    with (
        patch.object(agent.anthropic.messages, "create", side_effect=error),
        patch.object(agent.anthropic.messages, "stream", side_effect=error),
    ):
        result = await agent.query("test query", max_turns=1)

    assert "rate limit" in result.lower()
    assert "retry" in result.lower() or "wait" in result.lower()


@pytest.mark.asyncio
async def test_connection_error_returned_to_agent():
    """Test that connection errors are returned as text."""
    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"])
    agent = AiAssistAgent(config=config)

    error = APIConnectionError(message="Connection timeout", request=MagicMock())

    with (
        patch.object(agent.anthropic.messages, "create", side_effect=error),
        patch.object(agent.anthropic.messages, "stream", side_effect=error),
    ):
        result = await agent.query("test query", max_turns=1)

    assert "connection" in result.lower() or "timeout" in result.lower()
    assert "network" in result.lower() or "api" in result.lower()


@pytest.mark.asyncio
async def test_generic_api_error_returned_to_agent():
    """Test that generic API errors are returned as text."""
    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"])
    agent = AiAssistAgent(config=config)

    mock_response = MagicMock(spec=Response)
    mock_response.status_code = 500
    mock_response.headers = {"request-id": "test-request-id"}

    from anthropic import InternalServerError

    error = InternalServerError(
        message="Internal server error",
        response=mock_response,
        body={"error": {"message": "Internal server error"}},
    )

    with (
        patch.object(agent.anthropic.messages, "create", side_effect=error),
        patch.object(agent.anthropic.messages, "stream", side_effect=error),
    ):
        result = await agent.query("test query", max_turns=1)

    assert "error" in result.lower()
    assert "Internal server error" in result or "500" in result


@pytest.mark.asyncio
async def test_successful_query_not_affected():
    """Test that successful queries still work normally."""
    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"])
    agent = AiAssistAgent(config=config)

    # Mock successful response
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="Hello, I'm working fine!")]
    mock_response.stop_reason = "end_turn"
    mock_response.usage = MagicMock(input_tokens=100, output_tokens=20)

    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__enter__ = MagicMock(
        return_value=MagicMock(get_final_message=MagicMock(return_value=mock_response))
    )
    mock_stream_ctx.__exit__ = MagicMock(return_value=False)

    with (
        patch.object(agent.anthropic.messages, "create", return_value=mock_response),
        patch.object(agent.anthropic.messages, "stream", return_value=mock_stream_ctx),
    ):
        result = await agent.query("test query", max_turns=1)

    assert "Hello, I'm working fine!" in result
    assert "error" not in result.lower()


def _rate_limit_error() -> RateLimitError:
    mock_response = MagicMock(spec=Response)
    mock_response.status_code = 429
    mock_response.headers = {"request-id": "test-request-id"}
    return RateLimitError(
        message="usage limit reached",
        response=mock_response,
        body={"error": {"message": "usage limit reached"}},
    )


@pytest.mark.asyncio
async def test_api_error_raised_when_requested():
    """Callers that must not mistake an API failure for an answer can ask for an exception."""
    from ai_assist.agent import QueryAPIError

    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"])
    agent = AiAssistAgent(config=config)
    error = _rate_limit_error()

    with (
        patch.object(agent.anthropic.messages, "create", side_effect=error),
        patch.object(agent.anthropic.messages, "stream", side_effect=error),
    ):
        with pytest.raises(QueryAPIError, match="Rate Limit"):
            await agent.query("test query", max_turns=1, raise_on_api_error=True)


@pytest.mark.asyncio
async def test_awl_task_fails_when_model_api_fails():
    """An AWL task whose model calls all fail must not be recorded as a success."""
    from ai_assist.awl_ast import TaskNode, WorkflowNode
    from ai_assist.awl_runtime import AWLRuntime

    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"])
    agent = AiAssistAgent(config=config)
    error = _rate_limit_error()
    workflow = WorkflowNode(
        body=[
            TaskNode(task_id="collect", goal="Collect the data.", expose=["data_path"]),
            TaskNode(task_id="report", goal="Write the report."),
        ]
    )

    with (
        patch.object(agent.anthropic.messages, "create", side_effect=error),
        patch.object(agent.anthropic.messages, "stream", side_effect=error),
    ):
        result = await AWLRuntime(agent).execute(workflow)

    assert result.success is False
    assert [o.status for o in result.task_outcomes] == ["failed"]
    assert "Rate Limit" in result.task_outcomes[0].summary


@pytest.mark.asyncio
async def test_non_api_error_still_returned_as_text_when_raising_requested():
    """Only model API failures raise; task-limit outcomes such as max turns stay text."""
    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"])
    agent = AiAssistAgent(config=config)

    async def fake_stream(**kwargs):
        yield {"type": "error", "message": "Maximum turns reached without final answer"}

    with patch.object(agent, "query_streaming", side_effect=fake_stream):
        result = await agent.query("test query", max_turns=1, raise_on_api_error=True)

    assert result == "Maximum turns reached without final answer"


@pytest.mark.asyncio
async def test_action_fails_when_model_api_fails(tmp_path):
    """A scheduled action whose model calls all fail must be recorded as failed."""
    from ai_assist.action_engine import ActionEngine
    from ai_assist.action_model import ActionDefinition
    from ai_assist.state import StateManager

    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"])
    agent = AiAssistAgent(config=config)
    state_manager = StateManager(state_dir=tmp_path / "state")
    engine = ActionEngine(agent, state_manager)
    action = ActionDefinition(name="daily-check", trigger={"type": "interval", "every": "1h"}, prompt="Check things")
    error = _rate_limit_error()

    with (
        patch.object(agent.anthropic.messages, "create", side_effect=error),
        patch.object(agent.anthropic.messages, "stream", side_effect=error),
        patch("ai_assist.execution_helpers.NotificationDispatcher") as mock_nd,
    ):
        mock_nd.return_value.dispatch = AsyncMock()
        result = await engine.execute_action(action)

    assert result.success is False
    assert "Rate Limit" in result.output
    assert state_manager.get_monitor_state(engine._state_key(action)).last_results["last_success"] is False


@pytest.mark.asyncio
async def test_mcp_prompt_api_error_raised_when_requested():
    from ai_assist.agent import QueryAPIError

    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"])
    agent = AiAssistAgent(config=config)
    message = MagicMock(role="user")
    message.content.text = "Write the daily report."
    session = MagicMock()
    session.get_prompt = AsyncMock(return_value=MagicMock(messages=[message]))
    agent.sessions = {"srv": session}
    agent.available_prompts = {"srv": {"report": MagicMock(arguments=None)}}
    error = _rate_limit_error()

    with (
        patch.object(agent.anthropic.messages, "create", side_effect=error),
        patch.object(agent.anthropic.messages, "stream", side_effect=error),
    ):
        with pytest.raises(QueryAPIError, match="Rate Limit"):
            await agent.execute_mcp_prompt("srv", "report", max_turns=1, raise_on_api_error=True)
        # Without the flag, the partial (here empty) response is still returned.
        assert await agent.execute_mcp_prompt("srv", "report", max_turns=1) == ""
