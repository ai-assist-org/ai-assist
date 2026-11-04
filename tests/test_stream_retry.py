"""A model turn whose stream drops mid-reply is retried; limits fail AWL tasks."""

import ssl

import httpx2
import pytest

from ai_assist.query_errors import QueryAPIError, QueryIncompleteError


def _text(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}


class _DroppedStream:
    """A stream that opens fine, then breaks while the reply is being read."""

    def __init__(self, error: Exception):
        self._error = error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def __iter__(self):
        raise self._error


def _drop_first_streams(agent, errors: list[Exception]) -> dict:
    """Make the first len(errors) stream calls drop with those errors."""
    real_stream = agent.anthropic.messages.stream
    calls = {"count": 0}

    def stream(**kwargs):
        calls["count"] += 1
        if calls["count"] <= len(errors):
            return _DroppedStream(errors[calls["count"] - 1])
        return real_stream(**kwargs)

    agent.anthropic.messages.stream = stream
    return calls


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr("ai_assist.agent._STREAM_RETRY_BACKOFF_SECONDS", (0, 0))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        ssl.SSLError(1, "[SSL: RECORD_LAYER_FAILURE] record layer failure"),
        httpx2.ReadError("connection dropped"),
        httpx2.RemoteProtocolError("peer closed connection"),
        ConnectionResetError(104, "Connection reset by peer"),
    ],
    ids=["ssl", "read_error", "protocol_error", "connection_reset"],
)
async def test_dropped_stream_is_retried(make_replay_agent, error):
    agent = await make_replay_agent([_text("All good.")])
    calls = _drop_first_streams(agent, [error])

    assert await agent.query("Hello") == "All good."
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_retry_does_not_consume_the_turn_budget(make_replay_agent):
    agent = await make_replay_agent([_text("All good.")])
    _drop_first_streams(agent, [httpx2.ReadError("drop"), httpx2.ReadError("drop")])

    assert await agent.query("Hello", max_turns=1) == "All good."


@pytest.mark.asyncio
async def test_gives_up_after_two_retries(make_replay_agent):
    agent = await make_replay_agent([_text("never reached")])
    calls = _drop_first_streams(agent, [httpx2.ReadError("drop")] * 3)

    with pytest.raises(QueryAPIError, match="drop"):
        await agent.query("Hello", raise_on_api_error=True)
    assert calls["count"] == 3


@pytest.mark.asyncio
async def test_non_transient_error_is_not_retried(make_replay_agent):
    agent = await make_replay_agent([_text("never reached")])
    calls = _drop_first_streams(agent, [ValueError("bad request shape")])

    with pytest.raises(QueryAPIError):
        await agent.query("Hello", raise_on_api_error=True)
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_incomplete_query_raises_when_requested(make_replay_agent):
    agent = await make_replay_agent([{"content": [], "stop_reason": "refusal"}])

    with pytest.raises(QueryIncompleteError, match="declined"):
        await agent.query("Hello", raise_on_incomplete=True)


@pytest.mark.asyncio
async def test_incomplete_query_still_text_by_default(make_replay_agent):
    agent = await make_replay_agent([{"content": [], "stop_reason": "refusal"}])

    assert await agent.query("Hello") == "The model declined this request for safety reasons."
