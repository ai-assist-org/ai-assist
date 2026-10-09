"""Tests for parallel tool execution."""

import asyncio

import pytest

from ai_assist.agent import AiAssistAgent


class FakeBlock:
    """Fake content block for testing."""

    def __init__(self, block_type, name=None, block_id=None, block_input=None, text=None):
        self.type = block_type
        self.name = name
        self.id = block_id
        self.input = block_input or {}
        if text is not None:
            self.text = text


class TestExecuteToolsConcurrently:
    """Test _execute_tools_concurrently method."""

    @pytest.fixture
    def agent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from ai_assist.config import AiAssistConfig

        config = AiAssistConfig(
            anthropic_api_key="test-key",
            config_dir=str(tmp_path),
        )
        agent = AiAssistAgent(config)
        # Initialize per-query state normally set in the streaming loop
        agent._tool_result_cache = {}
        agent._duplicate_tool_call_count = 0
        agent._recent_tool_calls_for_loop = []
        return agent

    @pytest.mark.asyncio
    async def test_single_tool_call(self, agent):
        """Single tool call should work."""
        call_log = []

        async def mock_execute(name, args):
            call_log.append(name)
            return f"result-{name}"

        agent._execute_tool = mock_execute

        blocks = [FakeBlock("tool_use", name="internal__think", block_id="1", block_input={"thought": "plan"})]
        results, loop_detected = await agent._execute_tools_concurrently(blocks)

        assert len(results) == 1
        assert results[0]["tool_use_id"] == "1"
        assert results[0]["content"] == "result-internal__think"
        assert not loop_detected

    @pytest.mark.asyncio
    async def test_multiple_tools_executed_concurrently(self, agent):
        """Multiple tool calls should run concurrently."""
        execution_order = []

        async def mock_execute(name, args):
            execution_order.append(f"start-{name}")
            # Simulate I/O delay
            await asyncio.sleep(0.05)
            execution_order.append(f"end-{name}")
            return f"result-{name}"

        agent._execute_tool = mock_execute

        blocks = [
            FakeBlock("tool_use", name="mcp__dci__search_jobs", block_id="1", block_input={"query": "a"}),
            FakeBlock("tool_use", name="mcp__dci__get_jira_ticket", block_id="2", block_input={"key": "b"}),
        ]
        results, loop_detected = await agent._execute_tools_concurrently(blocks)

        assert len(results) == 2
        # Results should be in original order
        assert results[0]["tool_use_id"] == "1"
        assert results[1]["tool_use_id"] == "2"
        # Both should have started before either finished (concurrent)
        assert execution_order[0].startswith("start-")
        assert execution_order[1].startswith("start-")
        assert not loop_detected

    @pytest.mark.asyncio
    async def test_cached_results_reused(self, agent):
        """Duplicate tool calls should use cache, not re-execute."""
        call_count = 0

        async def mock_execute(name, args):
            nonlocal call_count
            call_count += 1
            return f"result-{call_count}"

        agent._execute_tool = mock_execute
        agent._tool_result_cache = {}

        blocks = [
            FakeBlock("tool_use", name="internal__list_directory", block_id="1", block_input={}),
            FakeBlock("tool_use", name="internal__list_directory", block_id="2", block_input={}),
        ]
        results, _ = await agent._execute_tools_concurrently(blocks)

        assert len(results) == 2
        # Both should have the same result (cached)
        assert results[0]["content"] == results[1]["content"]
        # Only one actual execution
        assert call_count == 1
        assert agent._duplicate_tool_call_count == 1

    @pytest.mark.asyncio
    async def test_error_results_marked(self, agent):
        """Error results should have is_error flag."""

        async def mock_execute(name, args):
            return "Error: tool failed"

        agent._execute_tool = mock_execute

        blocks = [FakeBlock("tool_use", name="internal__read_file", block_id="1", block_input={"path": "/bad"})]
        results, _ = await agent._execute_tools_concurrently(blocks)

        assert results[0]["is_error"] is True

    @pytest.mark.asyncio
    async def test_results_preserve_order(self, agent):
        """Results should match the order of input blocks."""

        async def mock_execute(name, args):
            # Second tool finishes faster
            if "fast" in name:
                await asyncio.sleep(0.01)
            else:
                await asyncio.sleep(0.05)
            return f"result-{name}"

        agent._execute_tool = mock_execute

        blocks = [
            FakeBlock("tool_use", name="slow_tool", block_id="1", block_input={}),
            FakeBlock("tool_use", name="fast_tool", block_id="2", block_input={}),
        ]
        results, _ = await agent._execute_tools_concurrently(blocks)

        assert results[0]["tool_use_id"] == "1"
        assert results[0]["content"] == "result-slow_tool"
        assert results[1]["tool_use_id"] == "2"
        assert results[1]["content"] == "result-fast_tool"

    @pytest.mark.asyncio
    async def test_loop_detection(self, agent):
        """Loop detection should trigger when same call repeats."""
        import hashlib
        import json

        async def mock_execute(name, args):
            return "result"

        agent._execute_tool = mock_execute
        agent._tool_result_cache = {}

        # Compute the real signature
        sig = f"internal__think:{hashlib.md5(json.dumps({}, sort_keys=True).encode()).hexdigest()[:8]}"
        # Pre-fill recent calls to trigger loop detection (threshold is 3)
        agent._recent_tool_calls_for_loop = [sig, sig]

        blocks = [FakeBlock("tool_use", name="internal__think", block_id="1", block_input={})]
        results, loop_detected = await agent._execute_tools_concurrently(blocks)

        assert loop_detected

    @pytest.mark.asyncio
    async def test_cancel_event_stops_execution(self, agent):
        """Cancel event should prevent tool execution."""
        call_count = 0

        async def mock_execute(name, args):
            nonlocal call_count
            call_count += 1
            return "result"

        agent._execute_tool = mock_execute

        cancel = asyncio.Event()
        cancel.set()  # Already cancelled

        blocks = [FakeBlock("tool_use", name="internal__think", block_id="1", block_input={})]
        results, _ = await agent._execute_tools_concurrently(blocks, cancel_event=cancel)

        assert len(results) == 0
        assert call_count == 0

    @pytest.mark.asyncio
    async def test_non_tool_use_blocks_ignored(self, agent):
        """Non-tool_use blocks should be skipped."""

        async def mock_execute(name, args):
            return "result"

        agent._execute_tool = mock_execute

        blocks = [
            FakeBlock("text", text="Hello"),
            FakeBlock("tool_use", name="internal__think", block_id="1", block_input={"thought": "test"}),
        ]
        results, _ = await agent._execute_tools_concurrently(blocks)

        assert len(results) == 1
        assert results[0]["tool_use_id"] == "1"


class TestLoopDetectionRecovery:
    """A detected loop should nudge the agent to wrap up once, not immediately
    discard the query with a top-level error — only a *second* loop within the
    same query (i.e. the nudge didn't help) falls back to the hard error."""

    @staticmethod
    def _repeat_think_turn():
        return {
            "content": [{"type": "tool_use", "id": "t1", "name": "internal__think", "input": {"thought": "x"}}],
            "stop_reason": "tool_use",
        }

    @pytest.mark.asyncio
    async def test_first_loop_nudges_and_recovers(self, make_replay_agent):
        agent = await make_replay_agent(
            [
                self._repeat_think_turn(),
                self._repeat_think_turn(),
                self._repeat_think_turn(),  # 3rd identical call trips the detector
                {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"},
            ]
        )
        result = await agent.query("test")
        assert result == "done"
        assert agent._loop_nudge_fired is True

    @pytest.mark.asyncio
    async def test_second_loop_after_nudge_aborts(self, make_replay_agent):
        agent = await make_replay_agent(
            [
                self._repeat_think_turn(),
                self._repeat_think_turn(),
                self._repeat_think_turn(),  # trips + nudges
                self._repeat_think_turn(),  # still repeating after the nudge
            ]
        )
        result = await agent.query("test")
        assert "loop detected" in result.lower()
