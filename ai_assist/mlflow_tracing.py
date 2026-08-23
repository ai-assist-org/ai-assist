"""Optional MLflow tracing integration (GenAI observability).

All MLflow knowledge lives here so the rest of the codebase stays MLflow-agnostic
and the dependency (mlflow-skinny) stays optional. Every helper is a cheap no-op
when tracing is disabled or when mlflow is not installed, so call sites need no
guards.

Enable with ``AI_ASSIST_ENABLE_MLFLOW=true``. The backend is chosen the standard
MLflow way via ``MLFLOW_TRACKING_URI``; unset falls back to a local SQLite store
under the config dir so ``mlflow ui`` (with the same URI) works with no server.
"""

from __future__ import annotations

import logging
from typing import Any

from .audit import SECRET_KEY_PATTERNS, SECRET_PATTERNS
from .config import get_config_dir

logger = logging.getLogger(__name__)

_enabled = False
_mlflow: Any = None  # bound to the mlflow module once setup succeeds


def setup_mlflow(config) -> bool:
    """Initialize MLflow tracing once, from the agent constructor.

    Returns True when tracing is active. On any failure (mlflow missing, backend
    unreachable) it logs and returns False so the caller keeps running without
    tracing.
    """
    global _enabled, _mlflow

    try:
        import mlflow
    except ImportError:
        logger.warning(
            "MLflow tracing enabled but mlflow is not installed; " "install with: pip install 'ai-assist[mlflow]'"
        )
        return False

    # Local default: a SQLite tracking store under the config dir. A SQL backend
    # (not a file store) is what the Traces UI metrics endpoint needs, and the
    # [mlflow] extra bundles SQLAlchemy for it. Four slashes => absolute path.
    tracking_uri = config.mlflow_tracking_uri or f"sqlite:///{get_config_dir() / 'mlflow.db'}"
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment)
        mlflow.anthropic.autolog()
    except Exception:
        logger.exception("Failed to initialize MLflow tracing")
        return False

    _mlflow = mlflow
    _enabled = True
    logger.info(
        "MLflow tracing enabled (uri=%s, experiment=%s)",
        tracking_uri,
        config.mlflow_experiment,
    )
    return True


def is_enabled() -> bool:
    """Whether MLflow tracing is currently active."""
    return _enabled


def _redact(arguments: dict) -> dict:
    """Redact secrets from tool arguments, reusing audit.py's patterns."""
    redacted: dict = {}
    for key, value in arguments.items():
        if any(pattern in key.lower() for pattern in SECRET_KEY_PATTERNS):
            redacted[key] = "[REDACTED]"
        elif isinstance(value, str):
            redacted[key] = SECRET_PATTERNS.sub("[REDACTED]", value)
        else:
            redacted[key] = value
    return redacted


def start_query_span(query_text: str, model: str):
    """Open the root span for one query; returns a LiveSpan (or None when off).

    Uses ``start_span_no_context`` so the span carries **no** OpenTelemetry
    context token. The streaming query holds this span open across the async
    generator's yields; a fluent (context-attached) span would fail its
    ``detach()`` when the generator is closed from a different context, recording
    a spurious ``ValueError`` on the span. Children are linked explicitly via
    ``parent_span`` instead of the ambient OTel context.
    """
    if not _enabled:
        return None
    try:
        return _mlflow.start_span_no_context(
            name="agent.query",
            inputs={"query": query_text, "model": model},
        )
    except Exception:
        logger.debug("MLflow start_query_span failed", exc_info=True)
        return None


def start_tool_span(tool_name: str, arguments: dict, parent=None):
    """Open a child span for a single tool call, with redacted inputs.

    Explicitly parented to the query's root span (``parent``) rather than the
    ambient OTel context, so nesting survives async-generator streaming.
    """
    if not _enabled:
        return None
    try:
        return _mlflow.start_span_no_context(
            name=f"tool:{tool_name}",
            parent_span=parent,
            inputs=_redact(arguments),
        )
    except Exception:
        logger.debug("MLflow start_tool_span failed", exc_info=True)
        return None


def end_span(span, outputs=None) -> None:
    """End a span opened with ``start_query_span``/``start_tool_span``.

    Safe to call with ``None`` and safe to call from a generator's ``finally``:
    ending a context-free span never touches an OTel token, so it cannot raise
    the cross-context ``ValueError`` the fluent API does.
    """
    if span is None:
        return
    try:
        span.end(outputs=outputs)
    except Exception:
        logger.debug("MLflow end_span failed", exc_info=True)


def record_query_trace(span, trace) -> None:
    """Mirror QueryTrace metrics onto the root span as outputs + attributes."""
    if span is None or trace is None:
        return
    try:
        span.set_outputs(trace.response_text)
        span.set_attributes(
            {
                "model": trace.model,
                "total_cost_usd": trace.total_cost_usd,
                "total_input_tokens": trace.total_input_tokens,
                "total_output_tokens": trace.total_output_tokens,
                "total_thinking_tokens": trace.total_thinking_tokens,
                "duration_seconds": trace.duration_seconds,
                "turn_count": trace.turn_count,
                "tool_call_count": len(trace.tool_calls),
            }
        )
    except Exception:
        logger.debug("MLflow record_query_trace failed", exc_info=True)
