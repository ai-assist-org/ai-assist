"""Tests for the optional MLflow tracing integration"""

import contextvars
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_assist import mlflow_tracing as mt
from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig, get_config_dir
from ai_assist.eval import QueryTrace


@pytest.fixture(autouse=True)
def _reset_tracing_state():
    """Isolate the module-level tracing state between tests."""
    saved_enabled, saved_mlflow = mt._enabled, mt._mlflow
    mt._enabled, mt._mlflow = False, None
    yield
    mt._enabled, mt._mlflow = saved_enabled, saved_mlflow


def _fake_mlflow():
    """A MagicMock mlflow whose start_span_no_context returns a span."""
    fake = MagicMock()
    span = MagicMock()
    fake.start_span_no_context.return_value = span
    return fake, span


def _config(**overrides):
    base = {"mlflow_tracking_uri": None, "mlflow_experiment": "ai-assist"}
    base.update(overrides)
    return AiAssistConfig(**base)


# --- config -----------------------------------------------------------------


def test_mlflow_config_from_env():
    with patch.dict(
        os.environ,
        {
            "AI_ASSIST_ENABLE_MLFLOW": "true",
            "MLFLOW_TRACKING_URI": "http://localhost:5000",
            "AI_ASSIST_MLFLOW_EXPERIMENT": "custom-exp",
        },
        clear=True,
    ):
        config = AiAssistConfig.from_env()

        assert config.enable_mlflow is True
        assert config.mlflow_tracking_uri == "http://localhost:5000"
        assert config.mlflow_experiment == "custom-exp"


def test_mlflow_config_defaults():
    with patch.dict(os.environ, {}, clear=True):
        config = AiAssistConfig.from_env()

        assert config.enable_mlflow is False
        assert config.mlflow_tracking_uri is None
        assert config.mlflow_experiment == "ai-assist"


# --- setup ------------------------------------------------------------------


def test_setup_mlflow_missing_dependency_returns_false():
    with patch.dict(sys.modules, {"mlflow": None}):
        assert mt.setup_mlflow(_config(mlflow_tracking_uri="http://x")) is False
    assert mt.is_enabled() is False


def test_setup_mlflow_enabled_configures_backend():
    fake, _span = _fake_mlflow()
    with patch.dict(sys.modules, {"mlflow": fake}):
        assert mt.setup_mlflow(_config(mlflow_tracking_uri="http://server:5000")) is True

    fake.set_tracking_uri.assert_called_once_with("http://server:5000")
    fake.set_experiment.assert_called_once_with("ai-assist")
    fake.anthropic.autolog.assert_called_once()
    assert mt.is_enabled() is True


def test_setup_mlflow_falls_back_to_local_store():
    fake, _span = _fake_mlflow()
    with patch.dict(sys.modules, {"mlflow": fake}):
        mt.setup_mlflow(_config(mlflow_tracking_uri=None))

    fake.set_tracking_uri.assert_called_once_with(f"sqlite:///{get_config_dir() / 'mlflow.db'}")


def test_setup_mlflow_backend_error_returns_false():
    fake, _span = _fake_mlflow()
    fake.set_experiment.side_effect = RuntimeError("boom")
    with patch.dict(sys.modules, {"mlflow": fake}):
        assert mt.setup_mlflow(_config()) is False
    assert mt.is_enabled() is False


# --- helpers are no-ops when disabled --------------------------------------


def test_helpers_noop_when_disabled():
    assert mt.start_query_span("hi", "model") is None
    assert mt.start_tool_span("srv__tool", {"a": 1}) is None
    # Must not raise on None spans
    mt.end_span(None, outputs="out")
    mt.record_query_trace(None, None)


# --- spans emit when enabled -----------------------------------------------


def test_start_query_span_sets_inputs_when_enabled():
    fake, span = _fake_mlflow()
    mt._enabled, mt._mlflow = True, fake
    got = mt.start_query_span("what is up", "claude-sonnet-4-6")
    assert got is span
    fake.start_span_no_context.assert_called_once_with(
        name="agent.query",
        inputs={"query": "what is up", "model": "claude-sonnet-4-6"},
    )


def test_start_tool_span_redacts_secret_arguments():
    fake, span = _fake_mlflow()
    mt._enabled, mt._mlflow = True, fake
    parent = MagicMock()
    got = mt.start_tool_span("srv__do", {"api_key": "shh", "path": "/tmp", "n": 3}, parent)
    assert got is span
    fake.start_span_no_context.assert_called_once_with(
        name="tool:srv__do",
        parent_span=parent,
        inputs={"api_key": "[REDACTED]", "path": "/tmp", "n": 3},
    )


def test_end_span_none_is_noop():
    mt.end_span(None)  # must not raise


def test_end_span_forwards_outputs():
    span = MagicMock()
    mt.end_span(span, outputs="result-text")
    span.end.assert_called_once_with(outputs="result-text")


def test_end_span_swallows_errors():
    span = MagicMock()
    span.end.side_effect = RuntimeError("boom")
    mt.end_span(span)  # must not propagate


def test_start_query_span_ends_across_contexts_without_recording_exception(tmp_path):
    """The root span must open in one context and end in another cleanly.

    Regression for the streaming UI crash: the old fluent span attached an
    OpenTelemetry context token, so ending it after the async generator was closed
    from a different context made ``detach()`` raise ``ValueError`` — which MLflow
    recorded as an ``exception`` event on the trace (visible in the UI). The
    context-free span carries no token, so ending it from any context neither
    raises nor records an exception. Using real mlflow (not a mock) is what
    exercises the OTel context machinery.
    """
    mlflow = pytest.importorskip("mlflow")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test-mismatch")
    mt._enabled, mt._mlflow = True, mlflow

    # Open in a *different* context (as the streaming generator does), end here.
    span = contextvars.copy_context().run(mt.start_query_span, "q", "m")
    assert span is not None
    mt.end_span(span, outputs="answer")  # must not raise across contexts

    assert not any(event.name == "exception" for event in span.events)


def test_record_query_trace_maps_fields():
    span = MagicMock()
    trace = QueryTrace(
        query_text="q",
        timestamp="2026-08-20T00:00:00",
        tool_calls=[{"tool_name": "a", "arguments": {}}, {"tool_name": "b", "arguments": {}}],
        turn_count=3,
        response_text="answer",
        total_input_tokens=100,
        total_output_tokens=50,
        total_thinking_tokens=10,
        total_cost_usd=0.25,
        duration_seconds=1.5,
        model="claude-sonnet-4-6",
    )

    mt.record_query_trace(span, trace)

    span.set_outputs.assert_called_once_with("answer")
    span.set_attributes.assert_called_once_with(
        {
            "model": "claude-sonnet-4-6",
            "total_cost_usd": 0.25,
            "total_input_tokens": 100,
            "total_output_tokens": 50,
            "total_thinking_tokens": 10,
            "duration_seconds": 1.5,
            "turn_count": 3,
            "tool_call_count": 2,
        }
    )


# --- streaming integration: span teardown must not corrupt agent state -------


@pytest.mark.asyncio
async def test_query_streaming_closed_early_leaves_clean_state(tmp_path):
    """Closing the stream generator early must not crash or leak span state.

    Regression for the streaming UI crash: the root span used to be a fluent,
    OTel-context span held open across the generator's yields. Closing the
    generator early (``aclose``) tore the span down from a different context,
    raising ``ValueError`` — recorded on the trace and, before that fix, leaking
    ``_query_depth``. With a context-free root span this must finish cleanly:
    depth back to 0, root span reference cleared, and no exception recorded on
    the trace. Uses real mlflow so the OTel context machinery is exercised.
    """
    mlflow = pytest.importorskip("mlflow")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test-streaming")
    mt._enabled, mt._mlflow = True, mlflow

    agent = AiAssistAgent(config=AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"]))

    # A single assistant turn with plain text and no tool calls => stream ends.
    text_block = MagicMock(type="text", text="hi")
    final_message = MagicMock(content=[text_block], stop_reason="end_turn")
    final_message.usage = MagicMock(input_tokens=10, output_tokens=5)

    stream_ctx = MagicMock()
    stream_ctx.__enter__.return_value = stream_ctx
    stream_ctx.__exit__.return_value = False
    stream_ctx.__iter__.return_value = iter([])  # no streamed deltas
    stream_ctx.get_final_message.return_value = final_message

    assert agent._query_depth == 0
    captured_span = None
    with (
        patch.object(agent.anthropic.messages, "stream", return_value=stream_ctx),
        patch.object(agent, "_execute_tools_concurrently", new=AsyncMock(return_value=([], False))),
    ):
        agen = agent.query_streaming("hello", max_turns=3)
        await agen.__anext__()  # start the generator; root span is now open
        captured_span = agent._mlflow_root_span
        await agen.aclose()  # close early, from this context => used to raise

    assert captured_span is not None
    assert agent._query_depth == 0  # depth restored despite early close
    assert agent._mlflow_root_span is None  # root span reference cleared
    assert not any(event.name == "exception" for event in captured_span.events)
