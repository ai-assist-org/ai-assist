"""Cassette-driven fake Anthropic client for deterministic, offline agent tests.

A *cassette* is an ordered list of assistant turns for a single query.  The Nth
call to ``messages.create`` / ``messages.stream`` returns the Nth recorded turn
(turn-index matching), so tests exercise the real agent loop without depending on
prompt-string equality or a live API.

The fake mirrors only the surface the agent consumes (verified against
``ai_assist/agent.py``): content blocks with ``.type``/``.text`` or
``.id``/``.name``/``.input`` plus ``model_dump()``; a message with ``.content``,
``.stop_reason`` and ``.usage``; and a stream that is an iterable context manager
exposing ``get_final_message()``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class CassetteExhausted(RuntimeError):
    """Raised when the agent requests more turns than the cassette recorded."""


@dataclass
class Cassette:
    """An ordered list of recorded assistant turns for one query."""

    turns: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"turns": self.turns}


def load_cassette(path: str | Path) -> Cassette:
    """Load a cassette from a JSON file."""
    data = json.loads(Path(path).expanduser().read_text())
    return Cassette(turns=data.get("turns", []))


class FakeBlock:
    """A content block mimicking an SDK block (text or tool_use)."""

    def __init__(
        self,
        block_type: str,
        *,
        text: str | None = None,
        block_id: str | None = None,
        name: str | None = None,
        block_input: dict[str, Any] | None = None,
    ):
        self.type = block_type
        if text is not None:
            self.text = text
        if block_id is not None:
            self.id = block_id
        if name is not None:
            self.name = name
        if block_input is not None:
            self.input = block_input

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        """Return the plain-dict form, as ``_serialize_content`` expects."""
        if self.type == "tool_use":
            return {
                "type": "tool_use",
                "id": getattr(self, "id", ""),
                "name": getattr(self, "name", ""),
                "input": getattr(self, "input", {}),
            }
        return {"type": self.type, "text": getattr(self, "text", "")}


class FakeUsage:
    """Token usage mirroring the ``.usage`` attribute the agent reads."""

    def __init__(self, input_tokens: int = 0, output_tokens: int = 0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeMessage:
    """A final message with ``.content``, ``.stop_reason`` and ``.usage``."""

    def __init__(self, content: list[FakeBlock], stop_reason: str, usage: FakeUsage):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage


def _block_from_dict(d: dict[str, Any]) -> FakeBlock:
    if d.get("type") == "tool_use":
        return FakeBlock(
            "tool_use",
            block_id=d.get("id", ""),
            name=d.get("name", ""),
            block_input=d.get("input", {}),
        )
    return FakeBlock("text", text=d.get("text", ""))


def _message_from_turn(turn: dict[str, Any]) -> FakeMessage:
    content = [_block_from_dict(b) for b in turn.get("content", [])]
    usage_data = turn.get("usage", {})
    usage = FakeUsage(
        input_tokens=usage_data.get("input_tokens", 0),
        output_tokens=usage_data.get("output_tokens", 0),
    )
    stop_reason = turn.get("stop_reason", "end_turn")
    return FakeMessage(content, stop_reason, usage)


class _Delta:
    def __init__(self, text: str):
        self.text = text


class _Event:
    def __init__(self, event_type: str, *, delta: _Delta | None = None, content_block: FakeBlock | None = None):
        self.type = event_type
        if delta is not None:
            self.delta = delta
        if content_block is not None:
            self.content_block = content_block


class FakeStream:
    """Iterable context manager mirroring ``messages.stream(...)``."""

    def __init__(self, message: FakeMessage):
        self._message = message

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def __iter__(self):
        for block in self._message.content:
            if block.type == "tool_use":
                yield _Event("content_block_start", content_block=block)
            elif getattr(block, "text", ""):
                yield _Event("content_block_delta", delta=_Delta(block.text))

    def get_final_message(self) -> FakeMessage:
        return self._message


class FakeMessages:
    """The ``.messages`` attribute: ``create`` and ``stream`` walk the cassette."""

    def __init__(self, cassette: Cassette):
        self._cassette = cassette
        self._index = 0

    def _next_message(self) -> FakeMessage:
        if self._index >= len(self._cassette.turns):
            raise CassetteExhausted(
                f"agent requested turn {self._index + 1} but cassette has " f"{len(self._cassette.turns)} turn(s)"
            )
        turn = self._cassette.turns[self._index]
        self._index += 1
        return _message_from_turn(turn)

    def create(self, **_kwargs: Any) -> FakeMessage:
        return self._next_message()

    def stream(self, **_kwargs: Any) -> FakeStream:
        return FakeStream(self._next_message())


class FakeAnthropicClient:
    """Drop-in replacement for ``AiAssistAgent.anthropic`` driven by a cassette."""

    def __init__(self, cassette: Cassette):
        self.messages = FakeMessages(cassette)


def make_client(turns: list[dict[str, Any]]) -> FakeAnthropicClient:
    """Build a fake client from an inline list of turn dicts (for tests)."""
    return FakeAnthropicClient(Cassette(turns=turns))
