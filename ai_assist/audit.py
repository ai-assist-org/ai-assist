"""Audit logging for tool calls"""

import json
import logging
import os
import re
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from .audit_writer import append_line as _append_line
from .config import get_config_dir
from .redaction import SECRET_PATTERN_SOURCES

logger = logging.getLogger(__name__)

SECRET_PATTERNS = re.compile(
    r"(sk-ant-[a-zA-Z0-9]+|ghp_[a-zA-Z0-9]+|gho_[a-zA-Z0-9]+"
    r"|xoxb-[a-zA-Z0-9-]+|xoxp-[a-zA-Z0-9-]+"
    r"|eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+"  # JWTs
    r"|(?=[a-zA-Z0-9+]*[A-Z+])(?=[a-zA-Z0-9+]*[0-9+])[a-zA-Z0-9+]{40,}={0,2})"  # Base64 keys (must contain uppercase AND digit/+), excludes hex SHAs and CamelCase identifiers
    + "".join(f"|(?:{p})" for p in SECRET_PATTERN_SOURCES)  # Same formats as LLM-bound redaction
)

SECRET_KEY_PATTERNS = {"api_key", "token", "secret", "password", "credential", "auth"}

MAX_RESULT_SUMMARY_LENGTH = 4000

# The live log is rotated (renamed, never rewritten) once it reaches this size;
# rotated files are deleted by cleanup() once older than its max_age_days.
MAX_LOG_BYTES = 10 * 1024 * 1024
LOG_FILE_NAME = "tool_calls.jsonl"
ROTATED_GLOB = "tool_calls-*.jsonl"
# Optional second, append-only copy of every record (a path outside the config dir)
AUDIT_SINK_ENV = "AI_ASSIST_AUDIT_SINK"

DECISIONS = frozenset({"autonomous", "approved", "refused"})
_REFUSED_ANSWERS = frozenset({"denied", "auto-denied (non-interactive)", "refused"})


@dataclass
class AuditedCall:
    """One tool call being executed: its correlation id and the authorization events it raised."""

    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    events: list[dict] = field(default_factory=list)
    logged: bool = False


_current_call: ContextVar[AuditedCall | None] = ContextVar("audited_call", default=None)


@contextmanager
def audited_call() -> Iterator[AuditedCall]:
    """Scope one tool call: records written inside share its correlation id (one record per call)."""
    call = AuditedCall()
    token = _current_call.set(call)
    try:
        yield call
    finally:
        _current_call.reset(token)


def record_authorization_event(kind: str, question: str, decision: str) -> None:
    """Record a security prompt / policy verdict for the tool call in progress (no-op outside one)."""
    call = _current_call.get()
    if call is not None:
        call.events.append({"kind": kind, "question": question[:500], "decision": decision})


def authorization_from(events: list[dict], rule: str) -> dict:
    """The authorization verdict of a call from its events; ``rule`` names the policy when nothing asked."""
    refused = [e for e in events if e.get("decision") in _REFUSED_ANSWERS]
    if refused:
        event = refused[0]
        if event["decision"] == "auto-denied (non-interactive)":
            rule = f"{event['kind']}-confirmation: no approver (non-interactive)"
        elif event["decision"] == "denied":
            rule = f"{event['kind']}-confirmation: denied by user"
        else:
            rule = event["kind"]
        return {"decision": "refused", "rule": rule, "reason": event.get("question", "")}
    if events:
        return {
            "decision": "approved",
            "rule": f"{events[0]['kind']}-confirmation",
            "approver": "user",
            "questions": [e.get("question", "") for e in events],
        }
    return {"decision": "autonomous", "rule": rule}


class AuditLogger:
    """Log tool calls to JSONL for audit trail"""

    def __init__(self, audit_dir: Path | None = None, sink_path: Path | None = None, max_bytes: int = MAX_LOG_BYTES):
        if audit_dir is None:
            audit_dir = get_config_dir() / "audit"

        audit_dir.mkdir(parents=True, exist_ok=True)
        self.audit_dir = audit_dir
        self.log_file = audit_dir / LOG_FILE_NAME
        self.max_bytes = max_bytes
        if sink_path is None and os.getenv(AUDIT_SINK_ENV):
            sink_path = Path(os.environ[AUDIT_SINK_ENV]).expanduser()
        self.sink_path = sink_path
        # (script, task) of the AWL tasks running, innermost last (see task_context)
        self._task_stack: list[tuple[str, str]] = []
        # Keep the audit trail readable only by the owner (it holds tool args/results)
        try:
            os.chmod(audit_dir, 0o700)
            if self.log_file.exists():
                os.chmod(self.log_file, 0o600)
        except OSError:
            logger.debug("Could not tighten audit permissions on %s", audit_dir)

    def _sanitize_value(self, value: str) -> str:
        """Remove secrets from a string value"""
        return SECRET_PATTERNS.sub("[REDACTED]", value)

    def _sanitize_arguments(self, arguments: dict) -> dict:
        """Remove secrets from argument values"""
        sanitized = {}
        for key, value in arguments.items():
            if any(pattern in key.lower() for pattern in SECRET_KEY_PATTERNS):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str):
                sanitized[key] = self._sanitize_value(value)
            else:
                sanitized[key] = value
        return sanitized

    @contextmanager
    def task_context(self, script: str, task: str) -> Iterator[None]:
        """Attribute the tool calls logged inside the block to an AWL script's task."""
        self._task_stack.append((script, task))
        try:
            yield
        finally:
            self._task_stack.pop()

    def log_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        result: str,
        success: bool,
        confirmations: list[dict] | None = None,
        rule: str = "no approval gate",
    ) -> None:
        """Append one JSON line to the audit log.

        Inside an ``audited_call()`` the record carries that call's correlation id and
        authorization events, and a second record for the same call is refused.

        Args:
            tool_name: Name of the tool called
            arguments: Tool arguments (will be sanitized)
            result: Tool result text (will be truncated and sanitized)
            success: Whether the tool call succeeded
            confirmations: Authorization events (defaults to those of the current call)
            rule: The policy that allowed the call when nothing asked or refused
        """
        call = _current_call.get()
        if call is not None:
            if call.logged:
                logger.warning("Audit: second record for call %s (%s) dropped", call.correlation_id, tool_name)
                return
            call.logged = True
            if confirmations is None:
                confirmations = call.events
        correlation_id = call.correlation_id if call is not None else uuid.uuid4().hex
        confirmations = [{**c, "question": self._sanitize_value(c.get("question", ""))} for c in confirmations or []]
        authorization = authorization_from(confirmations, rule)
        if authorization["decision"] not in DECISIONS:
            raise ValueError(f"Invalid authorization decision: {authorization['decision']}")

        result_summary = result[:MAX_RESULT_SUMMARY_LENGTH]
        if len(result) > MAX_RESULT_SUMMARY_LENGTH:
            result_summary += f"... [truncated, {len(result)} chars total]"

        result_summary = self._sanitize_value(result_summary)
        sanitized_args = self._sanitize_arguments(arguments)

        entry = {
            "correlation_id": correlation_id,
            "tool_name": tool_name,
            "arguments": sanitized_args,
            "result_summary": result_summary,
            "success": success,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pid": os.getpid(),
        }
        if self._task_stack:
            entry["script"], entry["task"] = self._task_stack[-1]
        entry["authorization"] = authorization
        if confirmations:
            entry["confirmations"] = confirmations

        # Log to standard logger for log file visibility
        args_full = json.dumps(sanitized_args)
        logger.info("Tool call: %s(%s) -> %s", tool_name, args_full, "ok" if success else "FAILED")

        if self.sink_path is not None:
            try:
                _append_line(self.sink_path, json.dumps(entry))
            except OSError:
                logger.exception("Audit sink %s unavailable", self.sink_path)
                entry["sink_unavailable"] = True

        self._rotate_if_needed()
        _append_line(self.log_file, json.dumps(entry))

    def _rotate_if_needed(self) -> None:
        """Rename the live log aside once it reaches max_bytes (records are never rewritten)."""
        try:
            if self.log_file.stat().st_size < self.max_bytes:
                return
        except FileNotFoundError:
            return
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        rotated = self.audit_dir / f"tool_calls-{stamp}.jsonl"
        os.replace(self.log_file, rotated)
        os.chmod(rotated, 0o600)

    def rotated_files(self) -> list[Path]:
        """Rotated logs, oldest first."""
        return sorted(self.audit_dir.glob(ROTATED_GLOB))

    def cleanup(self, max_age_days: int = 7) -> int:
        """Delete rotated logs last written more than max_age_days ago.

        The live log is never touched: records are only ever appended.

        Returns:
            Number of rotated files removed
        """
        cutoff = (datetime.now() - timedelta(days=max_age_days)).timestamp()
        removed = 0
        for path in self.rotated_files():
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                logger.exception("Could not remove rotated audit log %s", path)
        return removed
