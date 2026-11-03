"""Work limits wind a query down instead of cutting it off.

The turn budget is soft (a warning, then up to 50% more turns); when it runs
out, or an explicit time budget passes, the model gets one final turn to
answer -- or, in an AWL task, to report through internal__finish_task.
"""

import time

import pytest

from ai_assist.query_errors import QueryIncompleteError


def _text(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}


def _think(n: int) -> dict:
    return {
        "content": [{"type": "tool_use", "id": f"t{n}", "name": "internal__think", "input": {"thought": f"step {n}"}}],
        "stop_reason": "tool_use",
    }


def _finish(arguments: dict) -> dict:
    return {
        "content": [{"type": "tool_use", "id": "toolu_f", "name": "internal__finish_task", "input": arguments}],
        "stop_reason": "tool_use",
    }


def _record_calls(agent) -> list[dict]:
    """Record the tools and messages sent with every model call."""
    calls: list[dict] = []
    real_stream = agent.anthropic.messages.stream

    def stream(**kwargs):
        calls.append({"tools": [t["name"] for t in kwargs["tools"]], "messages": list(kwargs["messages"])})
        return real_stream(**kwargs)

    agent.anthropic.messages.stream = stream
    return calls


def _last_user_text(call: dict) -> str:
    message = call["messages"][-1]
    content = message["content"]
    return content if isinstance(content, str) else str(content)


@pytest.mark.asyncio
async def test_turn_budget_is_soft_then_final_turn_answers(make_replay_agent):
    # Budget 2 -> 1 extra turn after the warning -> a final turn without tools.
    agent = await make_replay_agent([_think(1), _think(2), _think(3), _text("Here is what I found.")])
    calls = _record_calls(agent)

    reply = await agent.query("Investigate", max_turns=2)

    assert reply == "Here is what I found."
    assert len(calls) == 4
    assert "wrap up" in _last_user_text(calls[2]).lower()
    assert calls[3]["tools"] == []
    assert "final answer" in _last_user_text(calls[3]).lower()


@pytest.mark.asyncio
async def test_finishing_within_the_extra_turns_needs_no_final_turn(make_replay_agent):
    agent = await make_replay_agent([_think(1), _think(2), _text("Done.")])
    calls = _record_calls(agent)

    assert await agent.query("Investigate", max_turns=2) == "Done."
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_awl_task_final_turn_offers_only_finish_task(make_replay_agent):
    agent = await make_replay_agent(
        [_think(1), _think(2), _think(3), _finish({"status": "success", "values": {"count": 3}})]
    )
    calls = _record_calls(agent)
    prev = agent.task_result_tool.begin()

    await agent.query("You are executing task 'collect'.", max_turns=2, raise_on_incomplete=True)

    assert calls[3]["tools"] == ["internal__finish_task"]
    assert agent.task_result_tool.end(prev) == {"status": "success", "reason": "", "values": {"count": 3}}


@pytest.mark.asyncio
async def test_awl_task_still_not_reporting_after_final_turn_is_incomplete(make_replay_agent):
    agent = await make_replay_agent([_think(1), _think(2), _think(3), _text("I could not finish.")])
    prev = agent.task_result_tool.begin()

    with pytest.raises(QueryIncompleteError):
        await agent.query("You are executing task 'collect'.", max_turns=2, raise_on_incomplete=True)
    agent.task_result_tool.end(prev)


@pytest.mark.asyncio
async def test_explicit_time_budget_triggers_final_turn(make_replay_agent, monkeypatch):
    agent = await make_replay_agent([_think(1), _text("Partial answer, time is up.")])
    calls = _record_calls(agent)
    clock = {"now": time.time()}

    def fake_time():
        clock["now"] += 40  # every reading advances 40 s of "work"
        return clock["now"]

    monkeypatch.setattr("ai_assist.agent.time.time", fake_time)

    reply = await agent.query("Investigate", max_turns=50, max_time_seconds=60)

    assert reply == "Partial answer, time is up."
    assert calls[1]["tools"] == []
    assert "final answer" in _last_user_text(calls[1]).lower()


@pytest.mark.asyncio
async def test_no_time_limit_lets_a_slow_query_finish(make_replay_agent):
    from ai_assist.agent import NO_TIME_LIMIT

    agent = await make_replay_agent([_text("Slow but complete.")])
    agent.config.default_query_timeout = 0.05
    real_stream = agent.anthropic.messages.stream

    def slow_stream(**kwargs):
        time.sleep(0.3)
        return real_stream(**kwargs)

    agent.anthropic.messages.stream = slow_stream

    assert await agent.query("Investigate", max_time_seconds=NO_TIME_LIMIT) == "Slow but complete."


@pytest.mark.asyncio
async def test_default_time_limit_still_applies_without_opting_out(make_replay_agent):
    agent = await make_replay_agent([_text("Too slow.")])
    agent.config.default_query_timeout = 0.05
    real_stream = agent.anthropic.messages.stream

    def slow_stream(**kwargs):
        time.sleep(0.3)
        return real_stream(**kwargs)

    agent.anthropic.messages.stream = slow_stream

    assert (await agent.query("Investigate")).startswith("Task timeout")
