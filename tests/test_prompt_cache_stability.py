"""The cacheable request prefix (tools + system) must not change between queries.

Anything volatile (clock, notifications, KG context) goes into the current user
turn instead, so the system prompt and earlier exchanges stay cache hits.
"""

from datetime import datetime
from unittest.mock import patch

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig
from ai_assist.knowledge_graph import KnowledgeGraph


def _make_agent(kg=None):
    config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={})
    return AiAssistAgent(config, knowledge_graph=kg)


def _kg_with_preference():
    kg = KnowledgeGraph(":memory:")
    kg.insert_knowledge(
        entity_type="user_preference",
        key="style",
        content="Prefers detailed explanations",
        confidence=1.0,
    )
    return kg


def test_system_prompt_is_one_cached_block_without_volatile_content():
    agent = _make_agent(kg=_kg_with_preference())
    agent._current_query_text = "anything"
    blocks = agent._build_system_prompt()
    assert len(blocks) == 1
    assert blocks[0].get("cache_control") == {"type": "ephemeral"}
    assert "Today is" not in blocks[0]["text"]
    assert "What You Know" not in blocks[0]["text"]


def test_system_prompt_is_identical_across_minutes():
    agent = _make_agent()
    with patch("ai_assist.identity.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 4, 6, 10, 30)
        first = agent._build_system_prompt()
        mock_dt.now.return_value = datetime(2026, 4, 6, 10, 31)
        second = agent._build_system_prompt()
    assert first == second


def test_query_context_carries_clock_and_kg():
    agent = _make_agent(kg=_kg_with_preference())
    agent._current_query_text = "anything"
    context = agent._build_query_context()
    assert "Today is" in context
    assert "Prefers detailed explanations" in context


def test_query_context_respects_no_kg():
    agent = _make_agent(kg=_kg_with_preference())
    agent._no_kg = True
    context = agent._build_query_context()
    assert "Today is" in context
    assert "What You Know" not in context


def test_with_query_context_splits_string_user_turn_into_blocks_without_mutating_caller():
    history = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    current = {"role": "user", "content": "q2"}
    messages = [*history, current]

    result = AiAssistAgent._with_query_context(messages, "CTX")

    assert result[:2] == history
    assert result[-1]["content"][0] == {"type": "text", "text": "q2"}
    assert result[-1]["content"][-1]["text"].endswith("CTX")
    assert current["content"] == "q2"


def test_with_query_context_appends_block_to_list_user_turn():
    messages = [{"role": "user", "content": [{"type": "text", "text": "q"}]}]
    result = AiAssistAgent._with_query_context(messages, "CTX")
    assert result[-1]["content"][0] == {"type": "text", "text": "q"}
    assert result[-1]["content"][-1]["text"].endswith("CTX")
    assert len(messages[0]["content"]) == 1


def test_with_query_context_skips_when_last_message_is_not_user():
    messages = [{"role": "assistant", "content": "a"}]
    assert AiAssistAgent._with_query_context(messages, "CTX") == messages


def test_breakpoint_skips_query_context_block_so_next_query_matches():
    """The marker sits on the user's own text: the next query's history has exactly that prefix"""
    current = AiAssistAgent._with_query_context([{"role": "user", "content": "q2"}], "CTX")
    marked = AiAssistAgent._with_message_cache_breakpoints(current)[-1]["content"]
    assert marked[0] == {"type": "text", "text": "q2", "cache_control": {"type": "ephemeral"}}
    assert "cache_control" not in marked[-1]

    next_query = AiAssistAgent._with_message_cache_breakpoints(
        [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}, {"role": "user", "content": "q3"}]
    )
    assert next_query[0]["content"] == [marked[0]]


def test_notifications_ask_the_model_to_name_their_source(tmp_path):
    """Sent with the user turn, notifications were stated as fact unless the block asks for the source"""
    import json

    (tmp_path / "notifications.jsonl").write_text(
        json.dumps(
            {
                "level": "error",
                "title": "Backup failed",
                "message": "disk full",
                "timestamp": datetime.now().isoformat(),
            }
        )
        + "\n"
    )
    agent = _make_agent()
    agent.interactive_mode = True
    with patch("ai_assist.config.get_config_dir", return_value=tmp_path):
        context = agent._build_query_context()
    assert "Backup failed" in context
    assert "say it comes from a /monitor notification" in context
