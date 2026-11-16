"""AI_ASSIST_CAPTURE_MODEL_REQUESTS: save each model request and response, to debug a model's answer."""

import json
import stat

import pytest

TURNS = [
    {
        "content": [{"type": "tool_use", "id": "t1", "name": "internal__think", "input": {"thought": "check"}}],
        "stop_reason": "tool_use",
    },
    {"content": [{"type": "text", "text": "Done."}], "stop_reason": "end_turn"},
]


def _captures(config_dir) -> list[dict]:
    files = sorted((config_dir / "traces" / "requests").glob("*.json"))
    return [json.loads(f.read_text()) | {"_file": f} for f in files]


@pytest.mark.asyncio
async def test_requests_are_not_captured_by_default(make_replay_agent, isolate_config_dir, monkeypatch):
    monkeypatch.delenv("AI_ASSIST_CAPTURE_MODEL_REQUESTS", raising=False)
    agent = await make_replay_agent(TURNS)
    await agent.query("Hello")
    assert not (isolate_config_dir / "traces" / "requests").exists()


@pytest.mark.asyncio
async def test_each_turn_request_and_response_are_captured(make_replay_agent, isolate_config_dir, monkeypatch):
    monkeypatch.setenv("AI_ASSIST_CAPTURE_MODEL_REQUESTS", "true")
    agent = await make_replay_agent(TURNS)
    assert await agent.query("Is the new sensor paired?") == "Done."

    first, second = _captures(isolate_config_dir)
    assert (first["turn"], second["turn"]) == (1, 2)
    assert first["query_id"] == second["query_id"]
    events = (isolate_config_dir / "traces" / "query_events.jsonl").read_text()
    assert f'"query_id": "{first["query_id"]}"' in events  # to line the capture up with the timeline

    request = first["request"]
    assert request["model"] == agent.config.model and request["max_tokens"] == agent.get_max_tokens()
    assert request["system"] and any(tool["name"] == "internal__think" for tool in request["tools"])
    assert "Is the new sensor paired?" in json.dumps(request["messages"])
    assert first["response"]["stop_reason"] == "tool_use"

    # the second request holds the whole conversation as sent: question, tool call, tool result
    sent = second["request"]["messages"]
    assert "Is the new sensor paired?" in json.dumps(sent[0])
    assert [m["role"] for m in sent[-2:]] == ["assistant", "user"]
    assert sent[-1]["content"][0]["type"] == "tool_result"
    assert second["response"]["content"] == [{"type": "text", "text": "Done."}]


@pytest.mark.asyncio
async def test_captures_are_private(make_replay_agent, isolate_config_dir, monkeypatch):
    """They hold whole conversations."""
    monkeypatch.setenv("AI_ASSIST_CAPTURE_MODEL_REQUESTS", "true")
    agent = await make_replay_agent(TURNS)
    await agent.query("Hello")
    captures = _captures(isolate_config_dir)
    assert captures
    for capture in captures:
        assert stat.S_IMODE(capture["_file"].stat().st_mode) == 0o600
