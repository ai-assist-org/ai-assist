"""Exceptions a caller can ask AiAssistAgent.query to raise instead of returning error text."""


class QueryAPIError(Exception):
    """The model API failed (or its stream stalled), so the query has no answer."""


class QueryIncompleteError(Exception):
    """The query stopped before a final answer (time or turn limit, loop, refusal)."""
