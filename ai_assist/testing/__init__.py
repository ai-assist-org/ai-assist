"""Testing helpers for ai-assist.

Provides a cassette-driven fake Anthropic client so the agent loop can be driven
offline and deterministically (no API key, no network), plus a recording client
that captures real runs into cassettes for later replay.
"""

from .mock_llm import (
    Cassette,
    CassetteExhausted,
    FakeAnthropicClient,
    load_cassette,
    make_client,
)
from .record import RecordingClient

__all__ = [
    "Cassette",
    "CassetteExhausted",
    "FakeAnthropicClient",
    "RecordingClient",
    "load_cassette",
    "make_client",
]
