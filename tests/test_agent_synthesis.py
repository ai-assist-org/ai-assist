"""Tests for agent synthesis functionality"""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig
from ai_assist.context import ConversationMemory
from ai_assist.knowledge_graph import KnowledgeGraph
from ai_assist.report_tools import ReportTools


def _mock_stream_message(response):
    """Mimic ``client.messages.stream(...)`` as a context manager whose
    ``get_final_message()`` returns ``response`` (matches the streaming call
    now used in production)."""
    cm = MagicMock()
    cm.__enter__.return_value.get_final_message.return_value = response
    return cm


@pytest.fixture
def kg():
    """Create in-memory knowledge graph"""
    return KnowledgeGraph(":memory:")


@pytest.fixture
def config():
    """Create test configuration"""
    return AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-3-5-sonnet-20241022",
        mcp_servers={},
    )


@pytest.fixture
def agent(config, kg, tmp_path):
    """Create agent with knowledge graph and isolated reports directory"""
    agent = AiAssistAgent(config, knowledge_graph=kg)
    agent.report_tools = ReportTools(reports_dir=tmp_path / "reports")
    return agent


@pytest.fixture
def conversation():
    """Create conversation memory with sample exchanges"""
    conv = ConversationMemory()
    conv.add_exchange(
        "I prefer pytest over unittest for Python testing",
        "Got it! I'll use pytest for Python tests.",
    )
    conv.add_exchange(
        "Also, DCI jobs tend to fail more on Fridays due to upstream CI",
        "Interesting pattern. I'll keep that in mind.",
    )
    return conv


class TestSynthesisEngine:
    """Test synthesis of conversation learnings"""

    @pytest.mark.asyncio
    async def test_synthesis_extracts_preferences(self, agent, conversation):
        """Synthesis should extract user preferences from conversation"""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "insights": [
                            {
                                "category": "user_preference",
                                "key": "python_test_framework",
                                "content": "User prefers pytest over unittest",
                                "confidence": 1.0,
                                "tags": ["python", "testing"],
                            }
                        ]
                    }
                )
            )
        ]

        with patch.object(agent.anthropic.messages, "stream", return_value=_mock_stream_message(mock_response)):
            await agent._run_synthesis(conversation, focus="preferences")

        results = agent.knowledge_graph.search_knowledge(entity_type="user_preference")
        assert len(results) >= 1
        assert any("pytest" in r["content"].lower() for r in results)

    @pytest.mark.asyncio
    async def test_synthesis_extracts_lessons(self, agent, conversation):
        """Synthesis should extract lessons learned"""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "insights": [
                            {
                                "category": "lesson_learned",
                                "key": "dci_friday_pattern",
                                "content": "DCI jobs fail more on Fridays due to upstream CI",
                                "confidence": 0.8,
                                "tags": ["dci", "patterns"],
                            }
                        ]
                    }
                )
            )
        ]

        with patch.object(agent.anthropic.messages, "stream", return_value=_mock_stream_message(mock_response)):
            await agent._run_synthesis(conversation, focus="lessons")

        results = agent.knowledge_graph.search_knowledge(entity_type="lesson_learned")
        assert len(results) >= 1
        assert any("friday" in r["content"].lower() for r in results)

    @pytest.mark.asyncio
    async def test_synthesis_handles_no_insights(self, agent, conversation):
        """Synthesis handles case with no new insights"""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({"insights": []}))]

        with patch.object(agent.anthropic.messages, "stream", return_value=_mock_stream_message(mock_response)):
            await agent._run_synthesis(conversation, focus="all")

        results = agent.knowledge_graph.search_knowledge()
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_synthesis_handles_json_error(self, agent, conversation):
        """Synthesis handles invalid JSON gracefully"""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Not valid JSON")]

        with patch.object(agent.anthropic.messages, "stream", return_value=_mock_stream_message(mock_response)):
            await agent._run_synthesis(conversation, focus="all")

        results = agent.knowledge_graph.search_knowledge()
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_synthesis_handles_markdown_json(self, agent, conversation):
        """Synthesis handles JSON wrapped in markdown code blocks"""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text='```json\n{"insights": [{"category": "user_preference", "key": "test", "content": "test", "confidence": 1.0, "tags": []}]}\n```'
            )
        ]

        with patch.object(agent.anthropic.messages, "stream", return_value=_mock_stream_message(mock_response)):
            await agent._run_synthesis(conversation, focus="all")

        results = agent.knowledge_graph.search_knowledge()
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_pending_synthesis_flag(self, agent):
        """Check pending synthesis flag is set correctly"""
        assert agent._pending_synthesis is None

        agent._pending_synthesis = {"focus": "all", "triggered_at": "2024-01-01"}
        assert agent._pending_synthesis is not None
        assert agent._pending_synthesis["focus"] == "all"

        agent._pending_synthesis = None
        assert agent._pending_synthesis is None


class TestSynthesisIntegration:
    """Test integration of synthesis with agent query"""

    @pytest.mark.asyncio
    async def test_synthesis_after_trigger(self, agent, conversation):
        """Synthesis runs after trigger_synthesis tool is called"""
        agent._pending_synthesis = {"focus": "all"}

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "insights": [
                            {
                                "category": "user_preference",
                                "key": "test_pref",
                                "content": "Test preference",
                                "confidence": 1.0,
                                "tags": [],
                            }
                        ]
                    }
                )
            )
        ]

        with patch.object(agent.anthropic.messages, "stream", return_value=_mock_stream_message(mock_response)):
            await agent.check_and_run_synthesis(conversation)

        assert agent._pending_synthesis is None
        results = agent.knowledge_graph.search_knowledge()
        assert len(results) >= 1


class TestSynthesisFromKG:
    """Test synthesis that reads conversation entities from the KG"""

    @pytest.mark.asyncio
    async def test_synthesis_from_kg_extracts_insights(self, agent, kg):
        """Synthesis from KG should extract insights from conversation entities"""
        now = datetime.now()
        kg.insert_entity(
            entity_type="conversation",
            data={"user": "I prefer pytest over unittest", "assistant": "Noted, using pytest."},
            valid_from=now - timedelta(hours=2),
            tx_from=now - timedelta(hours=2),
        )

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "insights": [
                            {
                                "category": "user_preference",
                                "key": "test_framework",
                                "content": "User prefers pytest over unittest",
                                "confidence": 1.0,
                                "tags": ["testing"],
                            }
                        ]
                    }
                )
            )
        ]

        with patch.object(agent.anthropic.messages, "stream", return_value=_mock_stream_message(mock_response)):
            await agent._run_synthesis_from_kg()

        results = kg.search_knowledge(entity_type="user_preference")
        assert len(results) >= 1
        assert any("pytest" in r["content"].lower() for r in results)

    @pytest.mark.asyncio
    async def test_synthesis_from_kg_only_recent(self, agent, kg):
        """Synthesis should only process conversations from the specified time window"""
        now = datetime.now()

        # Old conversation (48 hours ago)
        kg.insert_entity(
            entity_type="conversation",
            data={"user": "Old question about Jenkins", "assistant": "Jenkins info..."},
            valid_from=now - timedelta(hours=48),
            tx_from=now - timedelta(hours=48),
        )

        # Recent conversation (1 hour ago)
        kg.insert_entity(
            entity_type="conversation",
            data={"user": "I like dark mode", "assistant": "Dark mode enabled."},
            valid_from=now - timedelta(hours=1),
            tx_from=now - timedelta(hours=1),
        )

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "insights": [
                            {
                                "category": "user_preference",
                                "key": "dark_mode",
                                "content": "User likes dark mode",
                                "confidence": 0.9,
                                "tags": ["ui"],
                            }
                        ]
                    }
                )
            )
        ]

        with patch.object(
            agent.anthropic.messages, "stream", return_value=_mock_stream_message(mock_response)
        ) as mock_stream:
            await agent._run_synthesis_from_kg(hours=24)

        # The LLM should only have been called with the recent conversation
        call_args = mock_stream.call_args
        prompt_text = call_args[1]["messages"][0]["content"]
        assert "dark mode" in prompt_text
        assert "Jenkins" not in prompt_text

    @pytest.mark.asyncio
    async def test_synthesis_from_kg_no_conversations(self, agent, kg):
        """Synthesis with no conversations and no reports should return early"""
        with patch.object(agent, "_get_report_snapshots", return_value={}):
            result = await agent._run_synthesis_from_kg()

        assert "No new conversations" in result

        # No synthesis_marker should be created (nothing was processed)
        now = datetime.now()
        markers = kg.query_as_of(now, entity_type="synthesis_marker")
        assert len(markers) == 0

        # No knowledge insights should be saved
        for etype in ["user_preference", "lesson_learned", "project_context", "decision_rationale"]:
            results = kg.search_knowledge(entity_type=etype)
            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_synthesis_from_kg_no_conversations_but_new_reports(self, agent, kg):
        """When no conversations exist but reports changed, connection discovery should run"""
        now = datetime.now()

        # Insert an entity so connection discovery has something to work with
        kg.insert_knowledge(
            entity_type="lesson_learned",
            key="test_lesson",
            content="Test lesson content",
            metadata={"source": "test"},
            confidence=1.0,
        )

        # Mock _get_report_snapshots to simulate a new report
        with patch.object(agent, "_get_report_snapshots", return_value={"my_report.md": "2026-02-25T10:00:00"}):
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text=json.dumps({"connections": []}))]

            with patch.object(
                agent.anthropic.messages, "stream", return_value=_mock_stream_message(mock_response)
            ) as mock_stream:
                with patch.object(agent, "_gather_recent_reports", return_value="Report content here"):
                    await agent._run_synthesis_from_kg()

            # LLM should have been called for connection discovery
            assert mock_stream.called

        # A synthesis_marker should have been created
        markers = kg.query_as_of(now + timedelta(seconds=10), entity_type="synthesis_marker")
        assert len(markers) >= 1

    @pytest.mark.asyncio
    async def test_synthesis_from_kg_no_llm_calls_when_nothing_new(self, agent, kg):
        """No LLM calls should be made when there are no new conversations or reports"""
        now = datetime.now()

        # Create a previous synthesis marker with report snapshots
        kg.insert_entity(
            entity_type="synthesis_marker",
            data={
                "synthesized_conversations": 0,
                "reports_processed": {"existing_report.md": "2026-02-25T08:00:00"},
            },
            valid_from=now - timedelta(hours=1),
        )

        # Mock _get_report_snapshots to return same snapshots (no change)
        with patch.object(agent, "_get_report_snapshots", return_value={"existing_report.md": "2026-02-25T08:00:00"}):
            with patch.object(agent.anthropic.messages, "stream") as mock_stream:
                result = await agent._run_synthesis_from_kg()

            # No LLM calls should have been made
            mock_stream.assert_not_called()

        assert "No new conversations" in result

    @pytest.mark.asyncio
    async def test_synthesis_from_kg_skips_already_synthesized(self, agent, kg):
        """Synthesis should not re-process already synthesized conversations"""
        now = datetime.now()

        # First conversation (recent, within 24h)
        kg.insert_entity(
            entity_type="conversation",
            data={"user": "First question", "assistant": "First answer"},
            valid_from=now - timedelta(hours=3),
            tx_from=now - timedelta(hours=3),
        )

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({"insights": []}))]

        # First synthesis processes the first conversation
        with patch.object(agent.anthropic.messages, "stream", return_value=_mock_stream_message(mock_response)):
            await agent._run_synthesis_from_kg()

        # Verify synthesis_marker was created
        markers = kg.query_as_of(now + timedelta(hours=1), entity_type="synthesis_marker")
        assert len(markers) >= 1

        # Second conversation added after first synthesis marker
        # valid_from must be after the marker but tx_from must be <= "now" when synthesis runs
        import time

        time.sleep(0.01)  # Ensure marker's valid_from is in the past
        second_time = datetime.now()
        kg.insert_entity(
            entity_type="conversation",
            data={"user": "Second question", "assistant": "Second answer"},
            valid_from=second_time,
            tx_from=second_time,
        )

        # Run synthesis again — marker cutoff should exclude the first conversation
        with patch.object(
            agent.anthropic.messages, "stream", return_value=_mock_stream_message(mock_response)
        ) as mock_stream:
            await agent._run_synthesis_from_kg()

        # Should only include the second conversation (first already synthesized)
        call_args = mock_stream.call_args
        prompt_text = call_args[1]["messages"][0]["content"]
        assert "Second question" in prompt_text
        assert "First question" not in prompt_text
