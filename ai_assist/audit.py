"""Audit logging for tool calls"""

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

from .config import get_config_dir

logger = logging.getLogger(__name__)

SECRET_PATTERNS = re.compile(
    r"(sk-ant-[a-zA-Z0-9]+|ghp_[a-zA-Z0-9]+|gho_[a-zA-Z0-9]+"
    r"|xoxb-[a-zA-Z0-9-]+|xoxp-[a-zA-Z0-9-]+"
    r"|eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+"  # JWTs
    r"|(?=[a-zA-Z0-9+]*[A-Z+])(?=[a-zA-Z0-9+]*[0-9+])[a-zA-Z0-9+]{40,}={0,2})"  # Base64 keys (must contain uppercase AND digit/+), excludes hex SHAs and CamelCase identifiers
)

SECRET_KEY_PATTERNS = {"api_key", "token", "secret", "password", "credential", "auth"}

MAX_RESULT_SUMMARY_LENGTH = 4000


class AuditLogger:
    """Log tool calls to JSONL for audit trail"""

    def __init__(self, audit_dir: Path | None = None):
        if audit_dir is None:
            audit_dir = get_config_dir() / "audit"

        audit_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = audit_dir / "tool_calls.jsonl"

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

    def log_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        result: str,
        success: bool,
    ) -> None:
        """Append one JSON line to the audit log.

        Args:
            tool_name: Name of the tool called
            arguments: Tool arguments (will be sanitized)
            result: Tool result text (will be truncated and sanitized)
            success: Whether the tool call succeeded
        """
        result_summary = result[:MAX_RESULT_SUMMARY_LENGTH]
        if len(result) > MAX_RESULT_SUMMARY_LENGTH:
            result_summary += f"... [truncated, {len(result)} chars total]"

        result_summary = self._sanitize_value(result_summary)
        sanitized_args = self._sanitize_arguments(arguments)

        entry = {
            "tool_name": tool_name,
            "arguments": sanitized_args,
            "result_summary": result_summary,
            "success": success,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pid": os.getpid(),
        }

        # Log to standard logger for log file visibility
        args_full = json.dumps(sanitized_args)
        logger.info("Tool call: %s(%s) -> %s", tool_name, args_full, "ok" if success else "FAILED")

        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def cleanup(self, max_age_days: int = 7) -> int:
        """Remove entries older than max_age_days.

        Args:
            max_age_days: Maximum age of entries to keep

        Returns:
            Number of entries removed
        """
        if not self.log_file.exists():
            return 0

        cutoff = datetime.now() - timedelta(days=max_age_days)
        kept: list[str] = []
        removed = 0

        with open(self.log_file) as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                    ts = datetime.fromisoformat(entry["timestamp"])
                    if ts >= cutoff:
                        kept.append(stripped)
                    else:
                        removed += 1
                except json.JSONDecodeError, KeyError, ValueError:
                    kept.append(stripped)

        with open(self.log_file, "w") as f:
            for line in kept:
                f.write(line + "\n")

        return removed
