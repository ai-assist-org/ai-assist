"""Tests for the tool-use narration consistency check (_check_narration_consistency).

Catches the agent claiming to have used a tool/filter (e.g. __jq_filter) that
never actually appears in its tool calls this query. Uses jev when configured
(general — catches any tool/claim mismatch); otherwise falls back to a narrower
keyword check covering the one confirmed pattern (claimed jq usage).
"""

from unittest.mock import patch

import pytest

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig
from ai_assist.jev_client import JevError


def _agent(jev_enabled=False, jev_api_key=None):
    config = AiAssistConfig(
        anthropic_api_key="test-key",
        mcp_servers={},
        jev_enabled=jev_enabled,
        jev_api_key=jev_api_key,
    )
    return AiAssistAgent(config)


class TestNarrationConsistencyJev:
    @pytest.mark.asyncio
    async def test_jev_flags_inconsistent_narration(self):
        agent = _agent(jev_enabled=True, jev_api_key="k")
        agent.last_tool_calls = [
            {"tool_name": "internal__execute_command", "arguments": {"command": "echo x"}, "meta_params": {}}
        ]

        async def fake_decide(config, state, questions):
            return {"answers": {"narration_honest": {"type": "noul", "noul": 0.1}}}

        with patch("ai_assist.agent.jev_decide", side_effect=fake_decide):
            nudge = await agent._check_narration_consistency("I applied the jq filter map(.name).")

        assert nudge is not None

    @pytest.mark.asyncio
    async def test_jev_approves_consistent_narration(self):
        agent = _agent(jev_enabled=True, jev_api_key="k")
        agent.last_tool_calls = []

        async def fake_decide(config, state, questions):
            return {"answers": {"narration_honest": {"type": "noul", "noul": 0.95}}}

        with patch("ai_assist.agent.jev_decide", side_effect=fake_decide):
            nudge = await agent._check_narration_consistency("Here's the answer: 42.")

        assert nudge is None

    @pytest.mark.asyncio
    async def test_jev_error_falls_back_to_keyword_check(self):
        agent = _agent(jev_enabled=True, jev_api_key="k")
        agent.last_tool_calls = [
            {"tool_name": "internal__execute_command", "arguments": {"command": "echo x"}, "meta_params": {}}
        ]

        async def fake_decide(config, state, questions):
            raise JevError("boom")

        with patch("ai_assist.agent.jev_decide", side_effect=fake_decide):
            nudge = await agent._check_narration_consistency("I applied the jq filter map(.name).")

        assert nudge is not None  # keyword fallback still catches it

    @pytest.mark.asyncio
    async def test_jev_inconclusive_falls_back_to_keyword_check(self):
        agent = _agent(jev_enabled=True, jev_api_key="k")
        agent.last_tool_calls = []

        async def fake_decide(config, state, questions):
            return {"answers": {}}  # no probability for our question

        with patch("ai_assist.agent.jev_decide", side_effect=fake_decide):
            nudge = await agent._check_narration_consistency("The answer is 42.")

        assert nudge is None  # keyword fallback finds nothing suspicious either


class TestNarrationConsistencyKeywordFallback:
    """jev not configured — exercises the deterministic fallback alone."""

    @pytest.mark.asyncio
    async def test_claims_jq_without_using_it(self):
        agent = _agent()
        agent.last_tool_calls = [
            {"tool_name": "internal__execute_command", "arguments": {"command": "echo x"}, "meta_params": {}}
        ]
        nudge = await agent._check_narration_consistency("I ran the command and applied the jq filter map(.name).")
        assert nudge is not None

    @pytest.mark.asyncio
    async def test_claims_jq_and_actually_used_filter_param(self):
        agent = _agent()
        agent.last_tool_calls = [
            {
                "tool_name": "internal__execute_command",
                "arguments": {"command": "echo x"},
                "meta_params": {"__jq_filter": "map(.name)"},
            }
        ]
        nudge = await agent._check_narration_consistency("I applied the jq filter map(.name) to the output.")
        assert nudge is None

    @pytest.mark.asyncio
    async def test_claims_jq_and_actually_piped_to_jq(self):
        """Piping straight through `| jq` in the command also counts as genuine usage."""
        agent = _agent()
        agent.last_tool_calls = [
            {
                "tool_name": "internal__execute_command",
                "arguments": {"command": "echo x | jq '.[].name'"},
                "meta_params": {},
            }
        ]
        nudge = await agent._check_narration_consistency("I ran it through jq to get the names.")
        assert nudge is None

    @pytest.mark.asyncio
    async def test_no_jq_mention_is_fine(self):
        agent = _agent()
        agent.last_tool_calls = []
        nudge = await agent._check_narration_consistency("The answer is 42.")
        assert nudge is None


class TestNarrationNudgeEndToEnd:
    """Full turn-loop behavior via the public query() API."""

    @pytest.mark.asyncio
    async def test_nudges_once_then_accepts_revised_answer(self, make_replay_agent):
        agent = await make_replay_agent(
            [
                {
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "internal__think", "input": {"thought": "plan"}}
                    ],
                    "stop_reason": "tool_use",
                },
                {
                    "content": [{"type": "text", "text": "I applied the jq filter map(.name)."}],
                    "stop_reason": "end_turn",
                },
                {
                    "content": [{"type": "text", "text": "I read the output manually: alpha, beta, gamma."}],
                    "stop_reason": "end_turn",
                },
            ]
        )
        agent.config.jev_verify_narration = True
        result = await agent.query("extract names")
        assert result == "I read the output manually: alpha, beta, gamma."
        assert agent._narration_nudge_fired is True

    @pytest.mark.asyncio
    async def test_disabled_by_default(self, make_replay_agent):
        agent = await make_replay_agent(
            [
                {
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "internal__think", "input": {"thought": "plan"}}
                    ],
                    "stop_reason": "tool_use",
                },
                {
                    "content": [{"type": "text", "text": "I applied the jq filter map(.name)."}],
                    "stop_reason": "end_turn",
                },
            ]
        )
        # jev_verify_narration defaults to False — hallucinated answer passes through untouched.
        result = await agent.query("extract names")
        assert result == "I applied the jq filter map(.name)."
