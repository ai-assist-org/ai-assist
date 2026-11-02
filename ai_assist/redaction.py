"""Secret redaction for content sent to the LLM and for subprocess environments.

Three sources of secrets are redacted from tool results:
- values of environment variables whose names look like secrets (exact match),
- well-known secret formats (API key prefixes, JWTs, bearer tokens, ...),
- user-supplied strings and regexes from ``redact.json`` in the config dir.

Redaction stops accidental leaks; it is not a boundary against a model that
deliberately transforms a secret (``rev``, ``base64``) before printing it.
That is why subprocess environments are also filtered (filter_sensitive_env).
"""

import json
import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path

from .config import get_config_dir

logger = logging.getLogger(__name__)

REDACT_CONFIG_FILE = "redact.json"

# Env var names stripped from subprocess environments. Broad on purpose: it
# also drops non-secret config (JIRA_URL, AWS_REGION), which can be re-allowed.
SENSITIVE_ENV_PATTERNS = (
    "API_KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIALS",
    "ANTHROPIC_",
    "GOOGLE_",
    "AWS_",
    "AZURE_",
    "GITHUB_TOKEN",
    "JIRA_",
)

# Env var names whose values are redacted from LLM content. Narrower than
# SENSITIVE_ENV_PATTERNS so that URLs, regions and project ids stay visible.
_SECRET_VALUE_NAME = re.compile(r"TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|_KEY$|_PAT$", re.IGNORECASE)

# Shorter values are too likely to be common words ("true", "dummy").
MIN_SECRET_LENGTH = 8

SECRET_PATTERN_SOURCES = (
    r"\bsk-[A-Za-z0-9_-]{20,}",  # Anthropic / OpenAI
    r"\bgh[pousr]_[A-Za-z0-9]{20,}",  # GitHub tokens
    r"\bgithub_pat_[A-Za-z0-9_]{20,}",
    r"\bglpat-[A-Za-z0-9_-]{20,}",  # GitLab
    r"\bx(?:ox[abposr]|app)-[A-Za-z0-9-]{10,}",  # Slack
    r"\bAIza[0-9A-Za-z_-]{35}",  # Google API key
    r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",  # AWS access key id
    r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]*)?",  # JWT
    r"\b(?i:bearer)\s+[A-Za-z0-9._~+/=-]{16,}",
    r"(?<=://)[^/\s:@]+:[^/\s@]+(?=@)",  # user:password@ in URLs
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
)

SECRET_PATTERNS = re.compile("|".join(f"(?:{p})" for p in SECRET_PATTERN_SOURCES))


def filter_sensitive_env(environ: Mapping[str, str], allowed: set[str] | None = None) -> dict[str, str]:
    """Return environ without sensitive variables, except names in allowed."""
    allowed = allowed or set()
    return {
        key: value
        for key, value in environ.items()
        if key in allowed or not any(pattern in key.upper() for pattern in SENSITIVE_ENV_PATTERNS)
    }


class Redactor:
    """Replace secrets in text with [REDACTED] markers."""

    def __init__(self, values: dict[str, str], patterns: list[re.Pattern[str]]):
        """
        Args:
            values: literal secret -> label shown in the marker ("" for none)
            patterns: regexes whose matches are replaced with [REDACTED]
        """
        self._values = values
        # Longest first so a secret containing another one is replaced whole
        self._values_re = (
            re.compile("|".join(re.escape(v) for v in sorted(values, key=len, reverse=True))) if values else None
        )
        self._patterns = patterns

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None, config_path: Path | None = None) -> Redactor:
        """Build a redactor from environment secrets, built-in patterns and redact.json."""
        if environ is None:
            environ = os.environ
        if config_path is None:
            config_path = get_config_dir() / REDACT_CONFIG_FILE

        values = {
            value: name
            for name, value in environ.items()
            if _SECRET_VALUE_NAME.search(name) and len(value) >= MIN_SECRET_LENGTH
        }
        patterns = [SECRET_PATTERNS]

        user_config = _load_config(config_path)
        for string in user_config.get("strings", []):
            if isinstance(string, str) and string:
                values.setdefault(string, "")
        for source in user_config.get("patterns", []):
            try:
                patterns.append(re.compile(source))
            except re.error:
                logger.exception("Ignoring invalid redaction pattern %r in %s", source, config_path)

        return cls(values, patterns)

    def redact(self, text: str) -> str:
        if self._values_re is not None:
            text = self._values_re.sub(self._value_marker, text)
        for pattern in self._patterns:
            text = pattern.sub("[REDACTED]", text)
        return text

    def _value_marker(self, match: re.Match[str]) -> str:
        label = self._values[match.group(0)]
        return f"[REDACTED:{label}]" if label else "[REDACTED]"


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError, OSError:
        logger.exception("Ignoring unreadable redaction config %s", path)
        return {}
    return data if isinstance(data, dict) else {}
