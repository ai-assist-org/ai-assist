"""Security utilities for AI agent protection.

Provides three defense layers against MCP-related attacks:
1. Tool result sanitization — detects prompt injection in tool outputs
2. Tool description validation — detects tool poisoning in descriptions
3. Rug-pull detection — detects tool definition changes after connection
"""

import hashlib
import json
import logging
import re
import secrets
import unicodedata

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


class ToolDefinitionRegistry:
    """Registry that tracks tool definition fingerprints for rug-pull detection.

    Stores the hash of each tool's definition at first registration.
    On subsequent checks, compares against stored hashes to detect changes.
    """

    def __init__(self):
        self._fingerprints: dict[str, str] = {}

    def register_tools(self, tools: list[dict]) -> None:
        """Register tool definitions (stores fingerprints).

        Args:
            tools: List of tool definition dicts
        """
        for tool in tools:
            name = tool.get("name", "")
            self._fingerprints[name] = compute_tool_fingerprint(tool)

    def check_for_changes(self, tools: list[dict], scope: set[str] | None = None) -> list[dict]:
        """Check tools against stored fingerprints.

        Args:
            tools: Current list of tool definition dicts
            scope: If provided, only check for removals among these tool names.
                   Use this when reconnecting a single server to avoid flagging
                   tools from other servers as removed.

        Returns:
            List of change dicts: [{"tool_name": str, "change_type": "modified"|"added"|"removed"}]
        """
        changes = []
        current_names = set()

        for tool in tools:
            name = tool.get("name", "")
            current_names.add(name)
            fingerprint = compute_tool_fingerprint(tool)

            if name not in self._fingerprints:
                changes.append({"tool_name": name, "change_type": "added"})
            elif self._fingerprints[name] != fingerprint:
                changes.append({"tool_name": name, "change_type": "modified"})

        check_names = scope if scope is not None else set(self._fingerprints.keys())
        for name in check_names:
            if name in self._fingerprints and name not in current_names:
                changes.append({"tool_name": name, "change_type": "removed"})

        return changes

    def get_fingerprint(self, tool_name: str) -> str | None:
        """Get stored fingerprint for a tool.

        Args:
            tool_name: Tool name

        Returns:
            Hex digest or None if not registered
        """
        return self._fingerprints.get(tool_name)
