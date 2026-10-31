"""Utility functions for message handling and truncation."""


def truncate_large_messages(messages: list[dict], max_chars: int = 50000) -> None:
    """Truncate individual large messages to prevent context overflow.

    Tool results can be huge (e.g., search_in_file with max_results=200).
    This function modifies messages in-place.

    Note: The agent now calls this with adaptive limits from get_truncation_limits()
    that scale with the context window (200K standard, 1M extended).

    Args:
        messages: List of message dicts with 'content' field
        max_chars: Maximum characters per message (default: 50K for backward compatibility,
                   but agent passes adaptive limits)
    """
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > max_chars:
            truncated = content[:max_chars]
            msg["content"] = (
                truncated + f"\n\n[... truncated {len(content) - max_chars:,} characters to fit context ...]"
            )


def is_api_error(response: str) -> bool:
    """Check if a response is an API error message.

    Args:
        response: The response text to check

    Returns:
        True if the response is an API error message
    """
    return response.startswith(
        (
            "API Error:",
            "API Rate Limit Error:",
            "API Connection Error:",
            "The API is currently overloaded",
            "The context is too large",
            "Model stream timed out",
        )
    )


def is_context_overflow_error(response: str) -> bool:
    """Check if a response is a context overflow error.

    Args:
        response: The response text to check

    Returns:
        True if the response indicates context overflow
    """
    return is_api_error(response) and ("too long" in response.lower() or "context" in response.lower())
