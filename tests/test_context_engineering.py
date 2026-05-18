"""Tests for context engineering improvements"""

from unittest.mock import MagicMock, patch

import pytest

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig
from ai_assist.context import ConversationMemory


class TestTokenBudgetMonitoring:
    """Tests for token usage tracking"""

    def test_model_context_window_known_model(self):
        """Known models return correct context window size"""
        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={})
        agent = AiAssistAgent(config)
        assert agent.get_context_window_size() == 1000000

    def test_model_context_window_unknown_model(self):
        """Unknown models return default context window"""
        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={})
        agent = AiAssistAgent(config)
        agent.config.model = "unknown-model-v1"
        assert agent.get_context_window_size() == 200000

    def test_track_token_usage_basic(self):
        """Token usage is recorded from API response"""
        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={})
        agent = AiAssistAgent(config)

        mock_response = MagicMock()
        mock_response.usage.input_tokens = 5000
        mock_response.usage.output_tokens = 1000
        # No cache fields
        del mock_response.usage.cache_creation_input_tokens
        del mock_response.usage.cache_read_input_tokens

        agent._track_token_usage(mock_response, turn=0)

        usage = agent.get_token_usage()
        assert len(usage) == 1
        assert usage[0]["turn"] == 0
        assert usage[0]["input_tokens"] == 5000
        assert usage[0]["output_tokens"] == 1000

    def test_track_token_usage_with_cache(self):
        """Cache metrics are recorded when available"""
        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={})
        agent = AiAssistAgent(config)

        mock_response = MagicMock()
        mock_response.usage.input_tokens = 5000
        mock_response.usage.output_tokens = 1000
        mock_response.usage.cache_creation_input_tokens = 2000
        mock_response.usage.cache_read_input_tokens = 3000

        agent._track_token_usage(mock_response, turn=0)

        usage = agent.get_token_usage()
        assert usage[0]["cache_creation_input_tokens"] == 2000
        assert usage[0]["cache_read_input_tokens"] == 3000

    def test_track_token_usage_warns_at_threshold(self):
        """Warning is logged when token usage exceeds threshold"""
        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={})
        agent = AiAssistAgent(config)

        mock_response = MagicMock()
        mock_response.usage.input_tokens = 850000  # 85% of 1M
        mock_response.usage.output_tokens = 1000
        del mock_response.usage.cache_creation_input_tokens
        del mock_response.usage.cache_read_input_tokens

        with patch("logging.warning") as mock_warn:
            agent._track_token_usage(mock_response, turn=0)
            mock_warn.assert_called_once()
            assert "Context budget warning" in mock_warn.call_args[0][0]

    def test_track_token_usage_no_warn_below_threshold(self):
        """No warning when token usage is below threshold"""
        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={})
        agent = AiAssistAgent(config)

        mock_response = MagicMock()
        mock_response.usage.input_tokens = 100000  # 50% of 200K
        mock_response.usage.output_tokens = 1000
        del mock_response.usage.cache_creation_input_tokens
        del mock_response.usage.cache_read_input_tokens

        with patch("logging.warning") as mock_warn:
            agent._track_token_usage(mock_response, turn=0)
            mock_warn.assert_not_called()

    def test_get_token_usage_returns_copy(self):
        """get_token_usage returns a copy, not the original list"""
        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={})
        agent = AiAssistAgent(config)

        mock_response = MagicMock()
        mock_response.usage.input_tokens = 5000
        mock_response.usage.output_tokens = 1000
        del mock_response.usage.cache_creation_input_tokens
        del mock_response.usage.cache_read_input_tokens

        agent._track_token_usage(mock_response, turn=0)

        usage1 = agent.get_token_usage()
        usage2 = agent.get_token_usage()
        assert usage1 is not usage2

    def test_track_token_usage_no_usage_field(self):
        """Handles response without usage field gracefully"""
        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={})
        agent = AiAssistAgent(config)

        mock_response = MagicMock(spec=[])  # No usage attribute

        agent._track_token_usage(mock_response, turn=0)
        assert len(agent.get_token_usage()) == 0


class TestObservationMasking:
    """Tests for _mask_old_observations"""

    def test_no_tool_results(self):
        """Messages without tool results are untouched"""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        original = [m.copy() for m in messages]
        AiAssistAgent._mask_old_observations(messages)
        assert messages == original

    def test_recent_results_preserved(self):
        """Most recent tool results are kept intact"""
        messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "t1", "input": {}}]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "1", "content": "result1"}],
            },
            {"role": "assistant", "content": [{"type": "tool_use", "id": "2", "name": "t2", "input": {}}]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "2", "content": "result2"}],
            },
        ]
        AiAssistAgent._mask_old_observations(messages, keep_recent=2)
        # Both should be preserved (only 2 tool result rounds, keep_recent=2)
        assert messages[2]["content"][0]["content"] == "result1"
        assert messages[4]["content"][0]["content"] == "result2"

    def test_old_results_masked(self):
        """Old tool results beyond keep_recent are replaced with placeholder"""
        messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "t1", "input": {}}]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "1", "content": "old result data"}],
            },
            {"role": "assistant", "content": [{"type": "tool_use", "id": "2", "name": "t2", "input": {}}]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "2", "content": "mid result"}],
            },
            {"role": "assistant", "content": [{"type": "tool_use", "id": "3", "name": "t3", "input": {}}]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "3", "content": "recent result"}],
            },
        ]
        AiAssistAgent._mask_old_observations(messages, keep_recent=2)

        # First tool result should be masked
        assert "Result already retrieved" in messages[2]["content"][0]["content"]
        # Last two should be preserved
        assert messages[4]["content"][0]["content"] == "mid result"
        assert messages[6]["content"][0]["content"] == "recent result"

    def test_tool_use_id_preserved(self):
        """Tool use IDs are preserved during masking"""
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "abc", "name": "t", "input": {}}]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "abc", "content": "big result"}],
            },
            {"role": "assistant", "content": [{"type": "tool_use", "id": "def", "name": "t", "input": {}}]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "def", "content": "newer"}],
            },
            {"role": "assistant", "content": [{"type": "tool_use", "id": "ghi", "name": "t", "input": {}}]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "ghi", "content": "newest"}],
            },
        ]
        AiAssistAgent._mask_old_observations(messages, keep_recent=2)
        assert messages[2]["content"][0]["tool_use_id"] == "abc"

    def test_multiple_tool_results_in_one_message(self):
        """Multiple tool results in a single message are all masked when old"""
        messages = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "1", "name": "t1", "input": {}},
                    {"type": "tool_use", "id": "2", "name": "t2", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "1", "content": "r1"},
                    {"type": "tool_result", "tool_use_id": "2", "content": "r2"},
                ],
            },
            {"role": "assistant", "content": [{"type": "tool_use", "id": "3", "name": "t3", "input": {}}]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "3", "content": "recent"}],
            },
            {"role": "assistant", "content": [{"type": "tool_use", "id": "4", "name": "t4", "input": {}}]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "4", "content": "latest"}],
            },
        ]
        AiAssistAgent._mask_old_observations(messages, keep_recent=2)
        # Both results in the first tool results message should be masked
        assert "Result already retrieved" in messages[2]["content"][0]["content"]
        assert "Result already retrieved" in messages[2]["content"][1]["content"]

    def test_empty_messages(self):
        """Empty messages list is handled gracefully"""
        messages = []
        AiAssistAgent._mask_old_observations(messages)
        assert messages == []


class TestConversationCompaction:
    """Tests for ConversationMemory.compact()"""

    def test_needs_compaction_false_initially(self):
        """Fresh memory does not need compaction"""
        mem = ConversationMemory(max_exchanges=10, compaction_threshold=8)
        assert mem.needs_compaction() is False

    def test_needs_compaction_at_threshold(self):
        """Compaction is needed when threshold is reached"""
        mem = ConversationMemory(max_exchanges=10, compaction_threshold=3)
        mem.add_exchange("q1", "a1")
        mem.add_exchange("q2", "a2")
        mem.add_exchange("q3", "a3")
        assert mem.needs_compaction() is True

    def test_compact_below_keep_recent(self):
        """Compaction is skipped when exchanges <= keep_recent"""
        mem = ConversationMemory(max_exchanges=10)
        mem.add_exchange("q1", "a1")
        mem.add_exchange("q2", "a2")
        mock_client = MagicMock()
        assert mem.compact(mock_client, "test-model", keep_recent=4) is False
        mock_client.messages.create.assert_not_called()

    def test_compact_success(self):
        """Successful compaction replaces old exchanges with summary"""
        mem = ConversationMemory(max_exchanges=10)
        for i in range(8):
            mem.add_exchange(f"question {i}", f"answer {i}")

        # Mock the Claude API response
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "Summary: 8 exchanges about various questions."
        mock_response.content = [mock_block]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        result = mem.compact(mock_client, "test-model", keep_recent=4)

        assert result is True
        # 1 summary + 4 recent = 5 exchanges
        assert len(mem.exchanges) == 5
        assert mem.exchanges[0]["user"] == "[Conversation summary]"
        assert "Summary" in mem.exchanges[0]["assistant"]
        # Recent exchanges preserved
        assert mem.exchanges[1]["user"] == "question 4"
        assert mem.exchanges[4]["user"] == "question 7"

    def test_compact_api_failure(self):
        """Compaction handles API failure gracefully"""
        mem = ConversationMemory(max_exchanges=10)
        for i in range(8):
            mem.add_exchange(f"q{i}", f"a{i}")

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")

        result = mem.compact(mock_client, "test-model", keep_recent=4)

        assert result is False
        # Exchanges unchanged
        assert len(mem.exchanges) == 8

    def test_compact_empty_summary(self):
        """Compaction handles empty summary response gracefully"""
        mem = ConversationMemory(max_exchanges=10)
        for i in range(8):
            mem.add_exchange(f"q{i}", f"a{i}")

        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "   "  # Whitespace-only
        mock_response.content = [mock_block]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        result = mem.compact(mock_client, "test-model", keep_recent=4)

        assert result is False
        assert len(mem.exchanges) == 8

    def test_compaction_threshold_default(self):
        """Default compaction threshold is 8"""
        mem = ConversationMemory()
        assert mem.compaction_threshold == 8

    def test_compaction_threshold_custom(self):
        """Custom compaction threshold is respected"""
        mem = ConversationMemory(compaction_threshold=5)
        assert mem.compaction_threshold == 5


class TestNativeContextWindow:
    """Tests for native 1M context window on Claude 4.6+ models"""

    def test_context_window_1m_for_opus_46(self):
        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={}, model="claude-opus-4-6")
        agent = AiAssistAgent(config)
        assert agent.get_context_window_size() == 1000000

    def test_context_window_1m_for_sonnet_46(self):
        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={}, model="claude-sonnet-4-6")
        agent = AiAssistAgent(config)
        assert agent.get_context_window_size() == 1000000

    def test_context_window_1m_for_opus_47(self):
        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={}, model="claude-opus-4-7")
        agent = AiAssistAgent(config)
        assert agent.get_context_window_size() == 1000000

    def test_context_window_200k_for_older_models(self):
        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={}, model="claude-opus-4-5")
        agent = AiAssistAgent(config)
        assert agent.get_context_window_size() == 200000

    def test_context_window_default_for_unknown_model(self):
        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={}, model="claude-unknown")
        agent = AiAssistAgent(config)
        assert agent.get_context_window_size() == 200000


class TestAdaptiveTruncationLimits:
    """Tests for adaptive truncation limit calculation"""

    def test_standard_context_limits(self):
        """Standard 200K context returns default percentages"""
        config = AiAssistConfig(
            anthropic_api_key="test-key",
            mcp_servers={},
            model="claude-opus-4-5",
            message_limit_pct=5.0,
            total_messages_pct=60.0,
            reserve_pct=25.0,
        )
        agent = AiAssistAgent(config)

        limits = agent.get_truncation_limits()

        assert limits["max_message_chars"] == 40000
        assert limits["max_total_chars"] == 480000
        assert limits["context_window_tokens"] == 200000

    def test_1m_context_limits(self):
        """Native 1M context scales limits proportionally"""
        config = AiAssistConfig(
            anthropic_api_key="test-key",
            mcp_servers={},
            model="claude-opus-4-6",
            message_limit_pct=5.0,
            total_messages_pct=60.0,
        )
        agent = AiAssistAgent(config)

        limits = agent.get_truncation_limits()

        assert limits["max_message_chars"] == 200000
        assert limits["max_total_chars"] == 2400000
        assert limits["context_window_tokens"] == 1000000

    def test_custom_percentage_configuration(self):
        """Custom percentages are respected"""
        config = AiAssistConfig(
            anthropic_api_key="test-key",
            mcp_servers={},
            model="claude-opus-4-5",
            message_limit_pct=10.0,
            total_messages_pct=70.0,
            reserve_pct=20.0,
        )
        agent = AiAssistAgent(config)

        limits = agent.get_truncation_limits()

        assert limits["max_message_chars"] == 80000
        assert limits["max_total_chars"] == 560000


class TestToolResultCache:
    """Tests for per-query tool result dedup cache"""

    @pytest.mark.asyncio
    async def test_duplicate_tool_call_returns_cached_result(self):
        """Second identical tool call in same query returns cached result without re-execution"""
        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={})
        agent = AiAssistAgent(config)

        agent.available_tools.append(
            {
                "name": "dci__today",
                "description": "Get today's date.",
                "input_schema": {"type": "object", "properties": {}},
                "_server": "dci",
                "_original_name": "today",
            }
        )

        def make_usage_mock():
            m = MagicMock()
            m.input_tokens = 1000
            m.output_tokens = 200
            del m.cache_creation_input_tokens
            del m.cache_read_input_tokens
            return m

        # Response 1: two identical tool calls in one turn
        mock_tool_block_1 = MagicMock()
        mock_tool_block_1.type = "tool_use"
        mock_tool_block_1.name = "dci__today"
        mock_tool_block_1.input = {}
        mock_tool_block_1.id = "call_1"

        mock_tool_block_2 = MagicMock()
        mock_tool_block_2.type = "tool_use"
        mock_tool_block_2.name = "dci__today"
        mock_tool_block_2.input = {}
        mock_tool_block_2.id = "call_2"

        mock_response_1 = MagicMock()
        mock_response_1.content = [mock_tool_block_1, mock_tool_block_2]
        mock_response_1.stop_reason = "tool_use"
        mock_response_1.usage = make_usage_mock()

        # Response 2: text answer (no grounding nudge since tools were already called)
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Today is 2026-02-24."

        mock_response_2 = MagicMock()
        mock_response_2.content = [mock_text_block]
        mock_response_2.stop_reason = "end_turn"
        mock_response_2.usage = make_usage_mock()

        agent.anthropic = MagicMock()
        agent.anthropic.messages.create.side_effect = [mock_response_1, mock_response_2]

        execute_call_count = 0

        async def counting_execute(name, args):
            nonlocal execute_call_count
            execute_call_count += 1
            return '{"date": "2026-02-24"}'

        with (
            patch.object(agent, "get_max_tokens", return_value=8192),
            patch.object(agent, "_execute_tool", side_effect=counting_execute),
        ):
            result = await agent.query("What is today?")

        # _execute_tool should have been called only once (second call was cached)
        assert execute_call_count == 1
        # Duplicate count should be 1
        assert agent._duplicate_tool_call_count == 1
        assert "2026-02-24" in result
