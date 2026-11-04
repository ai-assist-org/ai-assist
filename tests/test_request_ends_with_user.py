"""Every model request must end with a user message.

Newer Claude models reject assistant "prefill": a conversation ending with an
assistant message fails with a 400.
"""

import logging

import pytest


def _text(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}


def _record_roles(agent) -> list[list[str]]:
    sent: list[list[str]] = []
    real_stream = agent.anthropic.messages.stream

    def stream(**kwargs):
        sent.append([m["role"] for m in kwargs["messages"]])
        return real_stream(**kwargs)

    agent.anthropic.messages.stream = stream
    return sent


@pytest.mark.asyncio
async def test_empty_reply_is_not_left_as_the_last_message(make_replay_agent):
    agent = await make_replay_agent([{"content": [], "stop_reason": "end_turn"}, _text("Here is the answer.")])
    sent = _record_roles(agent)

    assert await agent.query("Question") == "Here is the answer."
    assert len(sent) == 2
    assert all(roles[-1] == "user" for roles in sent)


@pytest.mark.asyncio
async def test_history_ending_with_assistant_gets_a_user_turn(make_replay_agent, caplog):
    agent = await make_replay_agent([_text("Continuing.")])
    sent = _record_roles(agent)
    history = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Partial answer"},
    ]

    with caplog.at_level(logging.WARNING, logger="ai_assist.agent"):
        assert await agent.query(messages=history) == "Continuing."

    assert sent[0][-1] == "user"
    assert "ends with an assistant message" in caplog.text
    assert "user, assistant" in caplog.text  # the role sequence, to find the culprit
