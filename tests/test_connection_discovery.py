"""Tests for connection discovery in KG synthesis"""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig
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
def agent(config, kg):
    """Create agent with knowledge graph"""
    return AiAssistAgent(config, knowledge_graph=kg)


class TestConnectionDiscovery:
    """Test connection discovery between KG entities"""

    @pytest.mark.asyncio
    async def test_connection_discovery_creates_relationships(self, agent, kg):
        """Connection discovery should create relationships between existing entities"""
        kg.insert_knowledge(
            entity_type="lesson_learned",
            key="pytest_fixture_scope",
            content="Use session scope for expensive fixtures",
        )
        kg.insert_knowledge(
            entity_type="project_context",
            key="testing_strategy",
            content="Project uses pytest with fixtures",
        )

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "connections": [
                            {
                                "source_id": "lesson_learned:pytest_fixture_scope",
                                "target_id": "project_context:testing_strategy",
                                "rel_type": "relates_to",
                                "description": "Fixture scope lesson applies to testing strategy",
                            }
                        ]
                    }
                )
            )
        ]

        with patch.object(agent.anthropic.messages, "stream", return_value=_mock_stream_message(mock_response)):
            result = await agent._run_connection_discovery()

        assert "Created 1" in result
        rels = kg.get_all_current_relationships()
        assert len(rels) == 1
        assert rels[0].rel_type == "relates_to"
        assert rels[0].source_id == "lesson_learned:pytest_fixture_scope"
        assert rels[0].target_id == "project_context:testing_strategy"

    @pytest.mark.asyncio
    async def test_connection_discovery_skips_duplicates(self, agent, kg):
        """Should not create duplicate relationships"""
        kg.insert_knowledge(entity_type="lesson_learned", key="key_a", content="Lesson A")
        kg.insert_knowledge(entity_type="lesson_learned", key="key_b", content="Lesson B")

        kg.insert_relationship(
            rel_type="relates_to",
            source_id="lesson_learned:key_a",
            target_id="lesson_learned:key_b",
            valid_from=datetime.now(),
        )

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "connections": [
                            {
                                "source_id": "lesson_learned:key_a",
                                "target_id": "lesson_learned:key_b",
                                "rel_type": "relates_to",
                                "description": "These are related",
                            }
                        ]
                    }
                )
            )
        ]

        with patch.object(agent.anthropic.messages, "stream", return_value=_mock_stream_message(mock_response)):
            result = await agent._run_connection_discovery()

        assert "Created 0" in result
        rels = kg.get_all_current_relationships()
        assert len(rels) == 1

    @pytest.mark.asyncio
    async def test_connection_discovery_rejects_invalid_ids(self, agent, kg):
        """Should skip connections with non-existent entity IDs"""
        kg.insert_knowledge(entity_type="lesson_learned", key="real", content="Real lesson")

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "connections": [
                            {
                                "source_id": "lesson_learned:real",
                                "target_id": "lesson_learned:fake_nonexistent",
                                "rel_type": "relates_to",
                                "description": "Invalid connection",
                            }
                        ]
                    }
                )
            )
        ]

        with patch.object(agent.anthropic.messages, "stream", return_value=_mock_stream_message(mock_response)):
            result = await agent._run_connection_discovery()

        assert "Created 0" in result
        rels = kg.get_all_current_relationships()
        assert len(rels) == 0

    @pytest.mark.asyncio
    async def test_connection_discovery_includes_reports(self, agent, kg, tmp_path):
        """Reports modified since cutoff should be included in the prompt"""
        agent.report_tools = ReportTools(reports_dir=tmp_path)

        report_file = tmp_path / "test_report.md"
        report_file.write_text("# Test Report\nSome content about DCI failures")

        kg.insert_knowledge(
            entity_type="lesson_learned",
            key="dci_failures",
            content="DCI jobs fail on Fridays",
        )

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({"connections": []}))]

        with patch.object(
            agent.anthropic.messages, "stream", return_value=_mock_stream_message(mock_response)
        ) as mock_stream:
            await agent._run_connection_discovery()

        call_args = mock_stream.call_args
        prompt_text = call_args[1]["messages"][0]["content"]
        assert "DCI failures" in prompt_text

    @pytest.mark.asyncio
    async def test_already_processed_reports_are_skipped(self, agent, kg, tmp_path):
        """Unchanged reports should not appear in the prompt"""
        agent.report_tools = ReportTools(reports_dir=tmp_path)

        report_file = tmp_path / "old_report.md"
        report_file.write_text("# Old Report\nAlready processed content")
        mod_time = datetime.fromtimestamp(report_file.stat().st_mtime).isoformat()

        kg.insert_knowledge(
            entity_type="lesson_learned",
            key="some_lesson",
            content="Some lesson",
        )

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({"connections": []}))]

        # Pass the exact modification time so the report is considered unchanged
        with patch.object(
            agent.anthropic.messages, "stream", return_value=_mock_stream_message(mock_response)
        ) as mock_stream:
            await agent._run_connection_discovery(previous_reports_processed={"old_report.md": mod_time})

        call_args = mock_stream.call_args
        prompt_text = call_args[1]["messages"][0]["content"]
        assert "Already processed content" not in prompt_text
        assert "No new reports" in prompt_text

    @pytest.mark.asyncio
    async def test_updated_report_is_reprocessed(self, agent, kg, tmp_path):
        """A report modified after previous processing should be included again"""
        agent.report_tools = ReportTools(reports_dir=tmp_path)

        report_file = tmp_path / "evolving_report.md"
        report_file.write_text("# Version 1\nOriginal content")
        old_mod_time = "2020-01-01T00:00:00"  # Stale timestamp from previous run

        kg.insert_knowledge(
            entity_type="lesson_learned",
            key="some_lesson",
            content="Some lesson",
        )

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({"connections": []}))]

        # The file's current mod_time won't match old_mod_time, so it should be included
        with patch.object(
            agent.anthropic.messages, "stream", return_value=_mock_stream_message(mock_response)
        ) as mock_stream:
            await agent._run_connection_discovery(previous_reports_processed={"evolving_report.md": old_mod_time})

        call_args = mock_stream.call_args
        prompt_text = call_args[1]["messages"][0]["content"]
        assert "Original content" in prompt_text

    @pytest.mark.asyncio
    async def test_connection_discovery_no_entities(self, agent, kg):
        """Should return early when no entities available"""
        result = await agent._run_connection_discovery()
        assert "No entities" in result

    @pytest.mark.asyncio
    async def test_connection_discovery_handles_invalid_response(self, agent, kg):
        """Should handle non-JSON from Claude gracefully without crashing"""
        kg.insert_knowledge(entity_type="lesson_learned", key="test", content="Test")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Not valid JSON")]

        with patch.object(agent.anthropic.messages, "stream", return_value=_mock_stream_message(mock_response)):
            result = await agent._run_connection_discovery()

        # Falls through to partial extraction, finds nothing, returns gracefully
        assert "no new connections" in result.lower() or "Created 0" in result

    @pytest.mark.asyncio
    async def test_synthesis_from_kg_runs_connection_discovery(self, agent, kg):
        """_run_synthesis_from_kg should call _run_connection_discovery"""
        now = datetime.now()
        kg.insert_entity(
            entity_type="conversation",
            data={"user": "Test question", "assistant": "Test answer"},
            valid_from=now - timedelta(hours=1),
            tx_from=now - timedelta(hours=1),
        )

        synthesis_response = MagicMock()
        synthesis_response.content = [MagicMock(text=json.dumps({"insights": []}))]

        with patch.object(agent.anthropic.messages, "stream", return_value=_mock_stream_message(synthesis_response)):
            with patch.object(
                agent,
                "_run_connection_discovery",
                return_value="Created 0 new connections",
            ) as mock_cd:
                await agent._run_synthesis_from_kg()

        mock_cd.assert_called_once()


class TestEntitySummarization:
    """Test entity summarization for prompts"""

    def test_summarize_knowledge_entities(self):
        """Knowledge entities show key and content"""
        from ai_assist.knowledge_graph import Entity

        entities = [
            Entity(
                id="lesson_learned:test_key",
                entity_type="lesson_learned",
                valid_from=datetime.now(),
                valid_to=None,
                tx_from=datetime.now(),
                tx_to=None,
                data={"key": "test_key", "content": "Test lesson content"},
            ),
        ]

        result = AiAssistAgent._summarize_entities_for_prompt(entities)
        assert "lesson_learned:test_key" in result
        assert "test_key" in result
        assert "Test lesson content" in result

    def test_summarize_tool_result_entities(self):
        """Tool result entities show tool name and args"""
        from ai_assist.knowledge_graph import Entity

        entities = [
            Entity(
                id="search_dci_jobs:abc12345",
                entity_type="tool_result",
                valid_from=datetime.now(),
                valid_to=None,
                tx_from=datetime.now(),
                tx_to=None,
                data={
                    "tool_name": "search_dci_jobs",
                    "arguments": {"query": "status=failure"},
                },
            ),
        ]

        result = AiAssistAgent._summarize_entities_for_prompt(entities)
        assert "search_dci_jobs" in result
        assert "status=failure" in result


class TestPartialJsonExtraction:
    """Test extraction of connections from truncated JSON"""

    def test_extracts_complete_objects_from_truncated_json(self):
        """Should extract complete connection objects even if JSON is truncated"""
        truncated = """{
  "connections": [
    {
      "source_id": "lesson_learned:a",
      "target_id": "project_context:b",
      "rel_type": "relates_to",
      "description": "They are related"
    },
    {
      "source_id": "lesson_learned:c",
      "target_id": "lesson_learned:d",
      "rel_type": "supports",
      "description": "C supports D"
    },
    {
      "source_id": "lesson_learned:e",
      "target_id": "project_con"""

        result = AiAssistAgent._extract_connections_from_partial_json(truncated)
        assert len(result["connections"]) == 2
        assert result["connections"][0]["source_id"] == "lesson_learned:a"
        assert result["connections"][1]["rel_type"] == "supports"

    def test_returns_empty_for_no_matches(self):
        """Should return empty connections list for gibberish input"""
        result = AiAssistAgent._extract_connections_from_partial_json("not json at all")
        assert result == {"connections": []}

    @pytest.mark.asyncio
    async def test_truncated_response_still_creates_relationships(self, agent, kg):
        """Truncated JSON should still create relationships from complete objects"""
        kg.insert_knowledge(entity_type="lesson_learned", key="a", content="Lesson A")
        kg.insert_knowledge(entity_type="project_context", key="b", content="Context B")

        truncated_json = """{
  "connections": [
    {
      "source_id": "lesson_learned:a",
      "target_id": "project_context:b",
      "rel_type": "relates_to",
      "description": "Related"
    },
    {
      "source_id": "lesson_learned:a",
      "target_id": "project_context:trun"""

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=truncated_json)]

        with patch.object(agent.anthropic.messages, "stream", return_value=_mock_stream_message(mock_response)):
            result = await agent._run_connection_discovery()

        assert "Created 1" in result
        rels = kg.get_all_current_relationships()
        assert len(rels) == 1
