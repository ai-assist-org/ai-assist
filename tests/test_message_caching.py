"""Tests for rolling prompt-cache breakpoints on conversation messages."""

import copy
from unittest.mock import MagicMock, patch

import pytest

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig

EPHEMERAL = {"type": "ephemeral"}


def _markers(messages: list) -> list[tuple[int, int]]:
    """Return (message index, block index) of every cache_control marker."""
    found = []
    for i, msg in enumerate(messages):
        content = msg.get("content")
        if isinstance(content, list):
            for j, block in enumerate(content):
                if isinstance(block, dict) and "cache_control" in block:
                    found.append((i, j))
    return found


def _tool_loop_messages() -> list[dict]:
    return [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me look."},
                {"type": "tool_use", "id": "t1", "name": "srv__a", "input": {}},
                {"type": "tool_use", "id": "t2", "name": "srv__b", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "result one"},
                {"type": "tool_result", "tool_use_id": "t2", "content": "result two"},
            ],
        },
    ]


def test_marks_last_block_of_last_two_user_messages():
    messages = _tool_loop_messages()

    result = AiAssistAgent._with_message_cache_breakpoints(messages)

    assert _markers(result) == [(2, 0), (4, 1)]
    assert result[2]["content"] == [{"type": "text", "text": "second question", "cache_control": EPHEMERAL}]
    assert result[4]["content"][1]["cache_control"] == EPHEMERAL
    assert result[4]["content"][1]["content"] == "result two"


def test_does_not_mutate_caller_messages():
    messages = _tool_loop_messages()
    original = copy.deepcopy(messages)

    AiAssistAgent._with_message_cache_breakpoints(messages)

    assert messages == original


def test_single_user_message_gets_one_marker():
    result = AiAssistAgent._with_message_cache_breakpoints([{"role": "user", "content": "hello"}])

    assert result == [{"role": "user", "content": [{"type": "text", "text": "hello", "cache_control": EPHEMERAL}]}]


def test_empty_user_content_is_left_alone():
    messages = [{"role": "user", "content": ""}, {"role": "user", "content": []}]

    assert _markers(AiAssistAgent._with_message_cache_breakpoints(messages)) == []


def _successful_stream():
    response = MagicMock()
    response.content = [MagicMock(type="text", text="done")]
    response.stop_reason = "end_turn"
    response.usage = MagicMock(input_tokens=100, output_tokens=20)
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=MagicMock(get_final_message=MagicMock(return_value=response)))
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


@pytest.mark.asyncio
@pytest.mark.parametrize("caching", [True, False])
async def test_request_carries_markers_only_when_caching_enabled(caching):
    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"], enable_prompt_caching=caching)
    agent = AiAssistAgent(config=config)
    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "new question"},
    ]

    with patch.object(agent.anthropic.messages, "stream", return_value=_successful_stream()) as mock_stream:
        await agent.query(messages=history, max_turns=1)

    sent = mock_stream.call_args.kwargs["messages"]
    assert _markers(sent) == ([(0, 0), (2, 0)] if caching else [])
    assert _markers(history) == []
