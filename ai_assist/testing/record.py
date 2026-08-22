"""Recording client that captures real Anthropic runs into a cassette.

Wraps a real ``Anthropic``/``AnthropicVertex`` client, passes ``create``/``stream``
calls through unchanged, and captures each final message as a cassette turn.  Call
``save(path)`` afterwards to write a JSON cassette replayable by
``FakeAnthropicClient``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .mock_llm import Cassette


def _turn_from_message(message: Any) -> dict[str, Any]:
    """Convert a real SDK message into a cassette turn dict."""
    content: list[dict[str, Any]] = []
    for block in getattr(message, "content", []):
        btype = getattr(block, "type", None)
        if btype == "tool_use":
            content.append(
                {
                    "type": "tool_use",
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "input": getattr(block, "input", {}),
                }
            )
        elif btype == "text":
            content.append({"type": "text", "text": getattr(block, "text", "")})
    usage = getattr(message, "usage", None)
    return {
        "content": content,
        "stop_reason": getattr(message, "stop_reason", "end_turn"),
        "usage": {
            "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
            "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
        },
    }


class _RecordingStream:
    """Wraps a real stream context manager and captures its final message."""

    def __init__(self, real_stream: Any, cassette: Cassette):
        self._real_stream = real_stream
        self._cassette = cassette

    def __enter__(self) -> _RecordingStream:
        self._entered = self._real_stream.__enter__()
        return self

    def __exit__(self, *exc: object) -> Any:
        return self._real_stream.__exit__(*exc)

    def __iter__(self):
        return iter(self._entered)

    def get_final_message(self) -> Any:
        message = self._entered.get_final_message()
        self._cassette.turns.append(_turn_from_message(message))
        return message


class _RecordingMessages:
    def __init__(self, real_messages: Any, cassette: Cassette):
        self._real_messages = real_messages
        self._cassette = cassette

    def create(self, **kwargs: Any) -> Any:
        message = self._real_messages.create(**kwargs)
        self._cassette.turns.append(_turn_from_message(message))
        return message

    def stream(self, **kwargs: Any) -> _RecordingStream:
        return _RecordingStream(self._real_messages.stream(**kwargs), self._cassette)


class RecordingClient:
    """Drop-in replacement for ``AiAssistAgent.anthropic`` that records turns."""

    def __init__(self, real_client: Any):
        self._real_client = real_client
        self.cassette = Cassette()
        self.messages = _RecordingMessages(real_client.messages, self.cassette)

    def save(self, path: str | Path) -> None:
        """Write the captured cassette to a JSON file."""
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.cassette.to_dict(), indent=2, default=str))
