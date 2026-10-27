"""Security utilities for AI agent protection.

Provides three defense layers against MCP-related attacks:
1. Tool result sanitization — detects prompt injection in tool outputs
2. Tool description validation — detects tool poisoning in descriptions
3. Rug-pull detection — detects tool definition changes after connection
"""

import difflib
import hashlib
import json
import logging
import re
import secrets
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Tool Result Sanitization
# ---------------------------------------------------------------------------

INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "ignore_instructions",
        re.compile(
            r"(?i)(ignore|disregard|forget|override)\s+((all|the|your|any|of|these)\s+)*"
            r"(previous|prior|above|earlier|system|preceding)\s+"
            r"(instructions?|prompts?|rules?|guidelines?|constraints?|directives?|commands?|messages?)"
        ),
    ),
    (
        "new_instructions",
        re.compile(
            r"(?i)(you\s+are\s+now|from\s+now\s+on|new\s+instructions?|"
            r"your\s+new\s+(role|task|purpose)|act\s+as\s+if)"
        ),
    ),
    (
        "system_prompt_extraction",
        re.compile(
            r"(?i)(reveal|show|print|output|display|repeat)\s+(your\s+)?"
            r"(system\s+prompt|instructions?|initial\s+prompt|hidden\s+prompt)"
        ),
    ),
    (
        "role_hijack",
        re.compile(
            r"(?i)(you\s+are\s+a\s+|pretend\s+(to\s+be|you\s+are)|"
            r"roleplay\s+as|switch\s+to\s+role|assume\s+the\s+role)"
        ),
    ),
    (
        "output_manipulation",
        re.compile(
            r"(?i)(do\s+not\s+(mention|reveal|tell|say)|"
            r"never\s+(mention|reveal|tell|say)|"
            r"hide\s+(this|the\s+fact)|"
            r"respond\s+only\s+with)"
        ),
    ),
    (
        "delimiter_injection",
        re.compile(r"(?i)(</?(system|user|assistant|human|ai)>|" r"\[SYSTEM\]|\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>)"),
    ),
]

# Markers around tool output: "[UNTRUSTED_TOOL_OUTPUT_START id=<random>]" ... END.
# The id is random per call, so content cannot close the block with a fake end marker.
UNTRUSTED_START = "[UNTRUSTED_TOOL_OUTPUT_START"
UNTRUSTED_END = "[UNTRUSTED_TOOL_OUTPUT_END"
_MARKER_TEXT = re.compile(r"UNTRUSTED_TOOL_OUTPUT", re.IGNORECASE)
# Zero-width and bidi control characters used to split trigger words
_INVISIBLE_CHARS = re.compile("[\u00ad\u180e\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")


def detect_injection(text: str) -> list[str]:
    """Names of INJECTION_PATTERNS found in text, after Unicode normalization.

    A heuristic: it catches common phrasings, not homoglyphs or encodings.
    """
    normalized = _INVISIBLE_CHARS.sub("", unicodedata.normalize("NFKC", text))
    return [name for name, pattern in INJECTION_PATTERNS if pattern.search(normalized)]


def wrap_untrusted(text: str, matched: list[str] | None = None) -> str:
    """Wrap text in per-call markers, defusing any marker text inside it."""
    marker_id = secrets.token_hex(4)
    body = _MARKER_TEXT.sub("UNTRUSTED-TOOL-OUTPUT", text)
    warning = f"Warning: possible prompt injection detected ({', '.join(matched)}).\n" if matched else ""
    return f"{UNTRUSTED_START} id={marker_id}]\n{warning}{body}\n{UNTRUSTED_END} id={marker_id}]"


def sanitize_tool_result(result: str, tool_name: str = "") -> tuple[str, list[str]]:
    """Scan tool output for prompt injection and wrap it in untrusted markers.

    Args:
        result: Raw tool result text
        tool_name: Name of the tool (for logging)

    Returns:
        Tuple of (wrapped result, list of matched pattern names). Output is
        always wrapped; a warning line is added when patterns matched.
    """
    if not result:
        return result, []

    matched = detect_injection(result)
    for name in matched:
        logger.warning("Prompt injection pattern '%s' detected in tool '%s' result", name, tool_name)
    return wrap_untrusted(result, matched), matched


# ---------------------------------------------------------------------------
# 2. Tool Description Validation
# ---------------------------------------------------------------------------

DESCRIPTION_SUSPICIOUS_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "imperative_to_model",
        re.compile(
            r"(?i)(you\s+must|you\s+should\s+always|always\s+include|"
            r"make\s+sure\s+(to|you)|remember\s+to|be\s+sure\s+to)"
        ),
    ),
    (
        "references_system_prompt",
        re.compile(r"(?i)(system\s+prompt|system\s+message|initial\s+instructions?|hidden\s+instructions?)"),
    ),
    (
        "behavioral_override",
        re.compile(r"(?i)(ignore\s+|disregard\s+|override\s+|forget\s+)(all\s+)?(other|previous|prior|any)"),
    ),
    (
        "data_exfiltration",
        re.compile(
            r"(?i)(exfiltrate|"
            r"(?:send|post)\s+(?:the\s+)?(?:all\s+)?"
            r"(?:user(?:'s)?\s+|private\s+|sensitive\s+|collected\s+|secret\s+|"
            r"credential\s+|api\s*key\s+|token\s+|system\s+prompt\s+)?data\s+to|"
            r"(?:send|post)\s+(?:the\s+)?(?:user(?:'s)?\s+)?"
            r"(?:api\s*key|secret|credential|token|system\s+prompt)\s+to|"
            r"include\s+in\s+(every|all)\s+(response|output|answer))"
        ),
    ),
    (
        "hidden_instructions",
        re.compile(
            r"(?i)(when\s+called,?\s+(also|always)|before\s+responding,?\s+(also|always)|"
            r"after\s+execution,?\s+(also|always)|in\s+addition\s+to\s+the\s+result)"
        ),
    ),
]


def validate_tool_description(tool_name: str, description: str, max_length: int = 5000) -> list[str]:
    """Validate a tool description for suspicious content (tool poisoning).

    Args:
        tool_name: Tool name (for logging context)
        description: Tool description text
        max_length: Maximum acceptable description length

    Returns:
        List of warning messages (empty if clean)
    """
    if not description:
        return []

    warnings = []

    if len(description) > max_length:
        warnings.append(f"Description length ({len(description)} chars) exceeds maximum ({max_length})")

    for pattern_name, pattern in DESCRIPTION_SUSPICIOUS_PATTERNS:
        if pattern.search(description):
            warnings.append(f"Suspicious pattern '{pattern_name}' found in description")

    return warnings


# ---------------------------------------------------------------------------
# 3. Rug-Pull Detection
# ---------------------------------------------------------------------------


def compute_tool_fingerprint(tool_def: dict) -> str:
    """Compute a stable hash of a tool definition.

    Hashes: name, description, and input_schema.
    Ignores internal fields (_server, _original_name, _full_description).

    Args:
        tool_def: Tool definition dict

    Returns:
        SHA-256 hex digest string
    """
    canonical = {
        "name": tool_def.get("name", ""),
        "description": tool_def.get("description", ""),
        "input_schema": tool_def.get("input_schema", {}),
    }
    serialized = json.dumps(canonical, sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()


# Approved MCP tool definitions, in the config dir
MCP_TOOL_DEFINITIONS_FILE = "mcp-tool-definitions.json"


class ToolDefinitionRegistry:
    """Approved MCP tool definitions per server, for rug-pull detection.

    The first time a server connects, its tools are trusted and recorded
    (trust on first use). Afterwards, tools whose name, description, or input
    schema differ from the recorded ones, and new tools, are held back until
    approved. Definitions are persisted so changes between runs (e.g. an MCP
    server update) are caught too; without a path the registry is in-memory.
    """

    def __init__(self, path: Path | None = None):
        self._path = path
        self._servers: dict[str, dict[str, dict]] = self._load()

    def _load(self) -> dict[str, dict[str, dict]]:
        if self._path is None or not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text())
        except OSError, json.JSONDecodeError:
            logger.exception("Ignoring unreadable MCP tool definitions file %s", self._path)
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._servers, indent=2, sort_keys=True))
        tmp.replace(self._path)

    @staticmethod
    def _definition(tool: dict) -> dict:
        return {"description": tool.get("description", ""), "input_schema": tool.get("input_schema", {})}

    def review(self, server: str, tools: list[dict]) -> tuple[list[dict], list[dict]]:
        """Split a server's tools into (approved tools, pending changes).

        Each pending change is {"tool": tool, "change": "added"|"modified",
        "previous": recorded definition or None}. Removed tools need no review.
        """
        known = self._servers.get(server)
        if known is None:
            self.approve(server, tools)
            return list(tools), []

        approved: list[dict] = []
        pending: list[dict] = []
        for tool in tools:
            previous = known.get(tool.get("name", ""))
            if previous is None:
                pending.append({"tool": tool, "change": "added", "previous": None})
            elif compute_tool_fingerprint({"name": tool.get("name", ""), **previous}) != compute_tool_fingerprint(tool):
                pending.append({"tool": tool, "change": "modified", "previous": previous})
            else:
                approved.append(tool)
        return approved, pending

    def approve(self, server: str, tools: list[dict]) -> None:
        """Record tool definitions as approved for a server."""
        known = self._servers.setdefault(server, {})
        for tool in tools:
            known[tool.get("name", "")] = self._definition(tool)
        self._save()


def describe_tool_change(change: dict) -> str:
    """Human-readable description of a pending tool change, with a diff."""
    tool = change["tool"]
    current = ToolDefinitionRegistry._definition(tool)
    if change["change"] == "added":
        return (
            f"New tool {tool['name']}:\n{current['description']}\n"
            f"Input schema: {json.dumps(current['input_schema'], sort_keys=True)}"
        )
    lines = [f"Changed tool {tool['name']}:"]
    for field in ("description", "input_schema"):
        old, new = change["previous"].get(field, ""), current[field]
        if old == new:
            continue
        if field == "input_schema":
            old, new = json.dumps(old, indent=2, sort_keys=True), json.dumps(new, indent=2, sort_keys=True)
        lines.extend(difflib.unified_diff(old.splitlines(), new.splitlines(), field, field, lineterm=""))
    return "\n".join(lines)
