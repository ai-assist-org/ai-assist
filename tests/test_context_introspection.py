"""Tests for context management introspection tools."""

import pytest

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig


@pytest.mark.asyncio
async def test_get_context_usage():
    """Test that agents can check their context usage."""
    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"])
    agent = AiAssistAgent(config=config)

    # Simulate some token usage
    agent._turn_token_usage = [
        {"turn": 1, "input_tokens": 5000, "output_tokens": 500},
        {"turn": 2, "input_tokens": 15000, "output_tokens": 800},
        {"turn": 3, "input_tokens": 25000, "output_tokens": 1200},
    ]

    result = await agent.introspection_tools.execute_tool("get_context_usage", {})

    # Should return JSON with context stats
    import json

    data = json.loads(result)

    assert "input_tokens" in data
    assert "context_window" in data
    assert "utilization" in data
    assert "turns_in_conversation" in data

    assert data["input_tokens"] == 25000
    assert data["context_window"] == 1000000  # Native 1M for 4.6+
    assert "%" in data["utilization"]
    assert data["turns_in_conversation"] == 3


@pytest.mark.asyncio
async def test_compact_conversation():
    """Test that agents can manually compact conversation."""
    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"])
    agent = AiAssistAgent(config=config)

    # Create messages with tool results
    messages = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "response 1"},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "1", "content": "old result 1"},
                {"type": "tool_result", "tool_use_id": "2", "content": "old result 2"},
            ],
        },
        {"role": "assistant", "content": "response 2"},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "3", "content": "recent result 1"},
            ],
        },
        {"role": "assistant", "content": "response 3"},
    ]

    # Store messages in agent's conversation (simulate)
    agent._conversation_messages = messages.copy()

    result = await agent.introspection_tools.execute_tool("compact_conversation", {"keep_recent_turns": 1})

    # Should return summary
    assert "compacted" in result.lower() or "masked" in result.lower()
    assert "1" in result  # keep_recent_turns=1

    # Verify old tool results were masked
    assert messages[2]["content"][0]["content"] == "[Result already retrieved]"
    assert messages[2]["content"][1]["content"] == "[Result already retrieved]"

    # Recent result should NOT be masked
    assert messages[4]["content"][0]["content"] == "recent result 1"


@pytest.mark.asyncio
async def test_compact_conversation_default_keep_recent():
    """Test compact with default keep_recent parameter."""
    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"])
    agent = AiAssistAgent(config=config)

    messages = []
    # Create 20 turns with tool results
    for i in range(20):
        messages.extend(
            [
                {"role": "user", "content": f"turn {i}"},
                {"role": "assistant", "content": f"response {i}"},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": str(i), "content": f"result {i}"},
                    ],
                },
            ]
        )

    agent._conversation_messages = messages.copy()

    result = await agent.introspection_tools.execute_tool("compact_conversation", {})

    # Default should be 10 (from _mask_old_observations)
    assert "10" in result or "recent" in result.lower()


@pytest.mark.asyncio
async def test_compact_conversation_no_tool_results():
    """Test compact when there are no tool results."""
    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"])
    agent = AiAssistAgent(config=config)

    messages = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "response 1"},
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "response 2"},
    ]

    agent._conversation_messages = messages.copy()

    result = await agent.introspection_tools.execute_tool("compact_conversation", {"keep_recent_turns": 1})

    # Should report nothing to compact
    assert "no tool results" in result.lower() or "nothing to compact" in result.lower()


@pytest.mark.asyncio
async def test_context_usage_with_no_turns():
    """Test context usage when no turns have occurred yet."""
    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"])
    agent = AiAssistAgent(config=config)

    # Empty token usage
    agent._turn_token_usage = []

    result = await agent.introspection_tools.execute_tool("get_context_usage", {})

    import json

    data = json.loads(result)

    assert data["input_tokens"] == 0
    assert data["turns_in_conversation"] == 0
    assert "0%" in data["utilization"] or "0.0%" in data["utilization"]


@pytest.mark.asyncio
async def test_compact_conversation_validates_keep_recent():
    """Test that keep_recent_turns must be positive."""
    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"])
    agent = AiAssistAgent(config=config)

    agent._conversation_messages = []

    # Negative keep_recent should be rejected
    result = await agent.introspection_tools.execute_tool("compact_conversation", {"keep_recent_turns": -5})

    assert "error" in result.lower() or "invalid" in result.lower() or "positive" in result.lower()
