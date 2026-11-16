"""Trimming an over-budget history: whole exchanges only, never the current question."""

import copy
import logging

import pytest

from ai_assist.agent import AiAssistAgent, _meta_param


def _exchange(i: int, size: int) -> list[dict]:
    """One old exchange: question, tool call, tool result, answer."""
    return [
        {"role": "user", "content": f"old question {i} " + "q" * size},
        {"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "x__y", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "r" * size}]},
        {"role": "assistant", "content": [{"type": "text", "text": f"old answer {i}"}]},
    ]


def test_under_budget_nothing_is_dropped():
    messages = _exchange(0, 10) + [{"role": "user", "content": "now"}]
    assert AiAssistAgent._trim_oldest_messages(messages, 10_000, len(messages) - 1) == 0
    assert len(messages) == 5


def test_drops_whole_oldest_exchanges_and_keeps_a_clean_start():
    messages = _exchange(0, 1000) + _exchange(1, 1000) + _exchange(2, 1000) + [{"role": "user", "content": "now"}]

    removed = AiAssistAgent._trim_oldest_messages(messages, 2500, len(messages) - 1)

    assert removed == 8
    assert messages[0]["content"].startswith("old question 2")
    assert messages[-1]["content"] == "now"


def test_never_drops_the_current_question_or_its_turns(caplog):
    history = _exchange(0, 100)
    current = [{"role": "user", "content": "summarize video X"}] + _exchange(1, 5000)[1:]
    messages = history + current

    with caplog.at_level(logging.WARNING):
        removed = AiAssistAgent._trim_oldest_messages(messages, 1000, len(history))

    assert removed == len(history)
    assert messages == current
    assert "Current query alone exceeds" in caplog.text


def test_dropping_is_logged(caplog):
    messages = _exchange(0, 1000) + [{"role": "user", "content": "now"}]
    with caplog.at_level(logging.INFO):
        AiAssistAgent._trim_oldest_messages(messages, 100, len(messages) - 1)
    assert "dropped the 4 oldest messages" in caplog.text


def test_tool_inputs_count_towards_the_budget():
    big_input = {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "t", "name": "x", "input": {"a": "z" * 500}}],
    }
    assert AiAssistAgent._message_chars(big_input) > 500


@pytest.mark.asyncio
async def test_query_keeps_the_question_when_history_overflows(make_replay_agent):
    agent = await make_replay_agent(
        [
            {
                "content": [{"type": "tool_use", "id": "t1", "name": "internal__think", "input": {"thought": "x"}}],
                "stop_reason": "tool_use",
            },
            {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"},
        ]
    )
    agent.config.model_context_window = 10_000  # 24k chars for all messages
    sent: list[list[dict]] = []
    original_stream = agent.anthropic.messages.stream

    def capturing_stream(**kwargs):
        sent.append(copy.deepcopy(kwargs["messages"]))
        return original_stream(**kwargs)

    agent.anthropic.messages.stream = capturing_stream
    history = [m for i in range(6) for m in _exchange(i, 3000)]

    result = await agent.query(messages=history + [{"role": "user", "content": "summarize video X"}])

    assert result == "done"
    assert len(sent) == 2
    for request in sent:
        assert AiAssistAgent._is_user_text_turn(request[0])
        assert any(
            m["role"] == "user" and "summarize video X" in str(m["content"]) for m in request
        ), "the current question was dropped"
        assert "old question 0" not in str(request)


@pytest.mark.parametrize("value", [None, False, "", "false", "False", "null", "none", " NULL "])
def test_meta_param_unset_values(value):
    arguments = {"__jq_filter": value, "q": 1}
    assert _meta_param(arguments, "__jq_filter") is None
    assert arguments == {"q": 1}


@pytest.mark.parametrize("value", ["report", ".items[]", "out.json"])
def test_meta_param_real_values(value):
    assert _meta_param({"__jq_filter": value}, "__jq_filter") == value


@pytest.mark.asyncio
async def test_false_collect_to_report_runs_the_tool_normally(tmp_path):
    from unittest.mock import AsyncMock, MagicMock

    from ai_assist.config import AiAssistConfig

    agent = AiAssistAgent(config=AiAssistConfig(anthropic_api_key="test-key"))
    session = AsyncMock()
    result = MagicMock()
    result.content = [MagicMock(text='{"ok": true}')]
    result.is_error = False
    session.call_tool.return_value = result
    agent.sessions = {"test": session}
    agent.available_tools = [{"name": "test__list", "description": "", "input_schema": {"type": "object"}}]

    out = await agent._execute_tool("test__list", {"__collect_to_report": "false", "__jq_filter": "false"})

    assert out == '{"ok": true}'
    session.call_tool.assert_called_once_with("list", {})
