"""Tests for the deterministic snapshot/replay harness (ai_assist.testing).

These run fully offline: the agent's Anthropic client is swapped for a
cassette-driven fake, and internal tools run against an isolated temp KG.
"""

from pathlib import Path

import pytest

from ai_assist.testing import (
    Cassette,
    CassetteExhausted,
    FakeAnthropicClient,
    RecordingClient,
    load_cassette,
    make_client,
)

SNAPSHOTS = Path(__file__).parent / "snapshots"


# --- Unit tests for the fake client itself -------------------------------------


def test_create_returns_turns_in_order():
    client = make_client(
        [
            {"content": [{"type": "text", "text": "first"}], "stop_reason": "end_turn"},
            {"content": [{"type": "text", "text": "second"}], "stop_reason": "end_turn"},
        ]
    )
    m1 = client.messages.create(model="x", messages=[])
    m2 = client.messages.create(model="x", messages=[])
    assert m1.content[0].text == "first"
    assert m2.content[0].text == "second"
    assert m1.usage.input_tokens == 0


def test_tool_use_block_model_dump():
    client = make_client(
        [
            {
                "content": [{"type": "tool_use", "id": "t1", "name": "internal__think", "input": {"thought": "x"}}],
                "stop_reason": "tool_use",
            }
        ]
    )
    block = client.messages.create().content[0]
    assert block.type == "tool_use"
    assert block.model_dump() == {
        "type": "tool_use",
        "id": "t1",
        "name": "internal__think",
        "input": {"thought": "x"},
    }


def test_stream_iterates_text_then_final_message():
    client = make_client(
        [{"content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn", "usage": {"output_tokens": 5}}]
    )
    with client.messages.stream(model="x", messages=[]) as stream:
        chunks = [event.delta.text for event in stream if event.type == "content_block_delta"]
        final = stream.get_final_message()
    assert chunks == ["hello"]
    assert final.content[0].text == "hello"
    assert final.usage.output_tokens == 5


def test_overrun_raises_cassette_exhausted():
    client = make_client([{"content": [{"type": "text", "text": "only"}], "stop_reason": "end_turn"}])
    client.messages.create()
    with pytest.raises(CassetteExhausted):
        client.messages.create()


def test_load_cassette_from_file():
    cassette = load_cassette(SNAPSHOTS / "think_then_answer.json")
    assert len(cassette.turns) == 2
    assert cassette.turns[0]["content"][1]["name"] == "internal__think"


# --- Full agent-loop replay ----------------------------------------------------


async def test_text_only_query(make_replay_agent):
    agent = await make_replay_agent([{"content": [{"type": "text", "text": "Hello there"}], "stop_reason": "end_turn"}])
    result = await agent.query(prompt="hi")
    assert result == "Hello there"
    assert agent.last_tool_calls == []


async def test_tool_call_then_answer_stream_path(make_replay_agent):
    agent = await make_replay_agent(
        [
            {
                "content": [{"type": "tool_use", "id": "t1", "name": "internal__think", "input": {"thought": "plan"}}],
                "stop_reason": "tool_use",
            },
            {"content": [{"type": "text", "text": "Done"}], "stop_reason": "end_turn"},
        ]
    )
    result = await agent.query(prompt="do it")
    assert result == "Done"
    assert [tc["tool_name"] for tc in agent.last_tool_calls] == ["internal__think"]


async def test_tool_call_then_answer_create_path(make_replay_agent):
    # max_output_tokens <= 8192 routes query() through messages.create()
    agent = await make_replay_agent(
        [
            {
                "content": [{"type": "tool_use", "id": "t1", "name": "internal__think", "input": {"thought": "plan"}}],
                "stop_reason": "tool_use",
            },
            {"content": [{"type": "text", "text": "Finished"}], "stop_reason": "end_turn"},
        ],
        max_output_tokens=4096,
    )
    result = await agent.query(prompt="do it")
    assert result == "Finished"
    assert [tc["tool_name"] for tc in agent.last_tool_calls] == ["internal__think"]


async def test_query_streaming_yields_chunks_and_tool_use(make_replay_agent):
    agent = await make_replay_agent(
        [
            {
                "content": [
                    {"type": "text", "text": "thinking..."},
                    {"type": "tool_use", "id": "t1", "name": "internal__think", "input": {"thought": "plan"}},
                ],
                "stop_reason": "tool_use",
            },
            {"content": [{"type": "text", "text": "All set"}], "stop_reason": "end_turn"},
        ]
    )
    text_chunks = []
    tool_uses = []
    saw_done = False
    async for item in agent.query_streaming(prompt="go"):
        if isinstance(item, str):
            text_chunks.append(item)
        elif item.get("type") == "tool_use":
            tool_uses.append(item["name"])
        elif item.get("type") == "done":
            saw_done = True
    assert "All set" in "".join(text_chunks)
    assert tool_uses == ["internal__think"]
    assert saw_done


async def test_full_replay_from_snapshot_file(make_replay_agent):
    cassette = load_cassette(SNAPSHOTS / "think_then_answer.json")
    agent = await make_replay_agent(cassette.turns)
    result = await agent.query(prompt="what is the answer?")
    assert result == "The answer is 42."
    assert [tc["tool_name"] for tc in agent.last_tool_calls] == ["internal__think"]


# --- Recording client round-trip ----------------------------------------------


def test_recording_client_round_trips_through_replay(tmp_path):
    # A stand-in "real" client whose captured turns must replay identically.
    real = make_client(
        [
            {
                "content": [{"type": "tool_use", "id": "t1", "name": "internal__think", "input": {"thought": "p"}}],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 4},
            },
            {"content": [{"type": "text", "text": "answer"}], "stop_reason": "end_turn"},
        ]
    )
    recorder = RecordingClient(real)

    # Drive it the way the agent would: a stream turn, then a create turn.
    with recorder.messages.stream(model="x", messages=[]) as stream:
        list(stream)
        stream.get_final_message()
    recorder.messages.create(model="x", messages=[])

    out = tmp_path / "cassette.json"
    recorder.save(out)

    cassette = load_cassette(out)
    assert cassette.turns[0]["content"][0]["name"] == "internal__think"
    assert cassette.turns[0]["usage"] == {"input_tokens": 10, "output_tokens": 4}
    assert cassette.turns[1]["content"][0]["text"] == "answer"

    # And the recorded cassette replays through the fake client.
    replay = FakeAnthropicClient(Cassette(turns=cassette.turns))
    assert replay.messages.stream().get_final_message().content[0].name == "internal__think"
    assert replay.messages.create().content[0].text == "answer"
