"""Claude model pricing and token cost computation."""

import logging
import re

logger = logging.getLogger(__name__)

# Per-million-token pricing: (input, output, cache_write, cache_read)
# Cache write = 1.25× input, cache read = 0.1× input
# Source: https://docs.anthropic.com/en/docs/about-claude/models (2026-06-24)
MODEL_PRICING: dict[str, tuple[float, float, float, float]] = {
    "claude-fable-5": (10.00, 50.00, 12.50, 1.00),
    "claude-mythos-5": (10.00, 50.00, 12.50, 1.00),
    "claude-opus-4-8": (5.00, 25.00, 6.25, 0.50),
    "claude-opus-4-7": (5.00, 25.00, 6.25, 0.50),
    "claude-opus-4-6": (5.00, 25.00, 6.25, 0.50),
    "claude-opus-4-5": (5.00, 25.00, 6.25, 0.50),
    "claude-sonnet-5": (3.00, 15.00, 3.75, 0.30),
    "claude-sonnet-4-6": (3.00, 15.00, 3.75, 0.30),
    "claude-sonnet-4-5": (3.00, 15.00, 3.75, 0.30),
    "claude-haiku-4-5": (1.00, 5.00, 1.25, 0.10),
    "claude-3-5-haiku": (0.80, 4.00, 1.00, 0.08),
    "claude-3-5-sonnet": (3.00, 15.00, 3.75, 0.30),
    "claude-3-opus": (15.00, 75.00, 18.75, 1.50),
    "claude-3-haiku": (0.25, 1.25, 0.3125, 0.025),
}

_SUFFIX_RE = re.compile(r"[-@](20\d{6}|default|latest)$")
_DEFAULT_FAMILY = "claude-sonnet-4-6"


def get_pricing(model: str) -> tuple[float, float, float, float]:
    """Get per-million-token pricing for a model.

    Strips date suffixes (-YYYYMMDD, @YYYYMMDD, @default) to find the family.
    Falls back to Sonnet 4.6 pricing with a warning.
    """
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]

    family = _SUFFIX_RE.sub("", model)
    if family in MODEL_PRICING:
        return MODEL_PRICING[family]

    logger.warning("Unknown model %r for pricing, using %s rates", model, _DEFAULT_FAMILY)
    return MODEL_PRICING[_DEFAULT_FAMILY]


def compute_turn_cost(model: str, token_entry: dict) -> float:
    """Compute USD cost for a single turn's token usage.

    Args:
        model: Model identifier (e.g. "claude-opus-4-6-20260205")
        token_entry: Dict with input_tokens, output_tokens, and optional cache fields

    Returns:
        Cost in USD
    """
    input_rate, output_rate, cache_write_rate, cache_read_rate = get_pricing(model)

    def _safe_int(val):
        try:
            return int(val) if val else 0
        except TypeError, ValueError:
            return 0

    input_tokens = _safe_int(token_entry.get("input_tokens", 0))
    output_tokens = _safe_int(token_entry.get("output_tokens", 0))
    cache_write_tokens = _safe_int(token_entry.get("cache_creation_input_tokens", 0))
    cache_read_tokens = _safe_int(token_entry.get("cache_read_input_tokens", 0))

    # Non-cache input tokens = total input - cache reads - cache writes
    plain_input = max(0, input_tokens - cache_read_tokens - cache_write_tokens)

    cost = (
        plain_input * input_rate
        + output_tokens * output_rate
        + cache_write_tokens * cache_write_rate
        + cache_read_tokens * cache_read_rate
    ) / 1_000_000

    return cost
