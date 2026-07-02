"""Tests for agent knowledge management tools"""

import json
from datetime import datetime, timedelta

import pytest

from ai_assist.knowledge_graph import KnowledgeGraph
from ai_assist.knowledge_tools import KnowledgeTools


@pytest.fixture
def kg():
    """Create in-memory knowledge graph for testing"""
    return KnowledgeGraph(":memory:")


@pytest.fixture
def knowledge_tools(kg):
    """Create knowledge tools instance"""
    return KnowledgeTools(kg)


class TestSaveKnowledge:
    """Test saving knowledge to graph"""

    async def test_save_user_preference(self, knowledge_tools, kg):
        """Save a user preference"""
        result = await knowledge_tools.save_knowledge(
            entity_type="user_preference",
            key="python_test_framework",
            content="User prefers pytest over unittest",
            tags=["python", "testing"],
            confidence=1.0,
        )

        assert "Saved" in result
        assert "user_preference" in result

        entities = kg.search_knowledge(entity_type="user_preference")
        assert len(entities) == 1
        assert entities[0]["key"] == "python_test_framework"
        assert "pytest" in entities[0]["content"]

    async def test_save_lesson_learned(self, knowledge_tools, kg):
        """Save a lesson learned"""
        result = await knowledge_tools.save_knowledge(
            entity_type="lesson_learned",
            key="dci_friday_failures",
            content="DCI job failures spike on Fridays due to upstream CI runs",
            tags=["dci", "patterns"],
            confidence=0.8,
        )

        assert "Saved" in result

        entities = kg.search_knowledge(entity_type="lesson_learned")
        assert len(entities) == 1
        assert entities[0]["metadata"]["confidence"] == 0.8

    async def test_save_project_context(self, knowledge_tools, kg):
        """Save project context"""
        result = await knowledge_tools.save_knowledge(
            entity_type="project_context",
            key="telco_ci_goals",
            content="Reduce failure notification noise by 50%",
            tags=["telco-partner-ci"],
        )

        assert "Saved" in result
        entities = kg.search_knowledge(entity_type="project_context")
        assert len(entities) == 1

    async def test_save_decision_rationale(self, knowledge_tools, kg):
        """Save decision rationale"""
        result = await knowledge_tools.save_knowledge(
            entity_type="decision_rationale",
            key="use_async_io",
            content="Using async/await for concurrent MCP server connections",
            tags=["architecture"],
        )

        assert "Saved" in result
        entities = kg.search_knowledge(entity_type="decision_rationale")
        assert len(entities) == 1

    async def test_save_with_future_valid_from(self, knowledge_tools, kg):
        """Save knowledge with a future valid_from date"""
        future = (datetime.now() + timedelta(days=10)).isoformat()
        result = await knowledge_tools.save_knowledge(
            entity_type="project_context",
            key="conference_berlin",
            content="User will be at conference in Berlin",
            valid_from=future,
        )

        assert "Saved" in result
        assert "valid from" in result
        entity = kg.get_entity("project_context:conference_berlin")
        assert entity is not None
        assert entity.valid_from > datetime.now()


class TestSearchKnowledge:
    """Test searching knowledge"""

    async def test_search_by_type(self, knowledge_tools, kg):
        """Search knowledge by entity type"""
        await knowledge_tools.save_knowledge("user_preference", "key1", "content1", tags=["tag1"])
        await knowledge_tools.save_knowledge("lesson_learned", "key2", "content2", tags=["tag2"])

        result = await knowledge_tools.search_knowledge(entity_type="user_preference")
        data = json.loads(result)

        assert data["count"] == 1
        assert data["results"][0]["type"] == "user_preference"

    async def test_search_by_query(self, knowledge_tools, kg):
        """Search knowledge by query pattern"""
        await knowledge_tools.save_knowledge("user_preference", "python_style", "Use black formatting")
        await knowledge_tools.save_knowledge("user_preference", "javascript_style", "Use prettier")

        result = await knowledge_tools.search_knowledge(query="%python%")
        data = json.loads(result)

        assert data["count"] == 1
        assert "python" in data["results"][0]["key"]

    async def test_search_by_tags(self, knowledge_tools, kg):
        """Search knowledge by tags"""
        await knowledge_tools.save_knowledge("lesson_learned", "key1", "content1", tags=["python", "testing"])
        await knowledge_tools.save_knowledge("lesson_learned", "key2", "content2", tags=["javascript"])

        result = await knowledge_tools.search_knowledge(tags=["python"])
        data = json.loads(result)

        assert data["count"] == 1
        assert "python" in data["results"][0]["tags"]

    async def test_search_all(self, knowledge_tools, kg):
        """Search all knowledge"""
        await knowledge_tools.save_knowledge("user_preference", "key1", "content1")
        await knowledge_tools.save_knowledge("lesson_learned", "key2", "content2")

        result = await knowledge_tools.search_knowledge(entity_type="all")
        data = json.loads(result)

        assert data["count"] == 2

    async def test_search_empty(self, knowledge_tools, kg):
        """Search returns empty when no matches"""
        result = await knowledge_tools.search_knowledge(query="%nonexistent%")
        data = json.loads(result)

        assert data["count"] == 0
        assert data["results"] == []

    async def test_search_with_since(self, knowledge_tools, kg):
        """Search with since only returns knowledge learned after that time"""
        old_time = datetime.now() - timedelta(hours=2)
        kg.insert_entity(
            entity_type="lesson_learned",
            entity_id="lesson_learned:old_lesson",
            data={"key": "old_lesson", "content": "An old lesson", "metadata": {"tags": []}},
            valid_from=old_time,
            tx_from=old_time,
        )
        await knowledge_tools.save_knowledge("lesson_learned", "new_lesson", "A new lesson")

        since = (datetime.now() - timedelta(hours=1)).isoformat()
        result = await knowledge_tools.search_knowledge(entity_type="lesson_learned", since=since)
        data = json.loads(result)

        assert data["count"] == 1
        assert data["results"][0]["key"] == "new_lesson"

    async def test_search_excludes_future_knowledge(self, knowledge_tools, kg):
        """Future-dated knowledge should not appear in current searches"""
        future = (datetime.now() + timedelta(days=10)).isoformat()
        await knowledge_tools.save_knowledge("project_context", "future_event", "Future thing", valid_from=future)
        await knowledge_tools.save_knowledge("project_context", "current_event", "Current thing")

        result = await knowledge_tools.search_knowledge(entity_type="project_context")
        data = json.loads(result)

        keys = [r["key"] for r in data["results"]]
        assert "current_event" in keys
        assert "future_event" not in keys

    async def test_search_include_future(self, knowledge_tools, kg):
        """include_future=True surfaces future-dated knowledge"""
        future = (datetime.now() + timedelta(days=10)).isoformat()
        await knowledge_tools.save_knowledge("project_context", "future_event", "Future thing", valid_from=future)
        await knowledge_tools.save_knowledge("project_context", "current_event", "Current thing")

        result = await knowledge_tools.search_knowledge(entity_type="project_context", include_future=True)
        data = json.loads(result)

        keys = [r["key"] for r in data["results"]]
        assert "current_event" in keys
        assert "future_event" in keys

    async def test_search_with_since_no_matches(self, knowledge_tools, kg):
        """Search with future since returns nothing"""
        await knowledge_tools.save_knowledge("user_preference", "key1", "content1")

        future = (datetime.now() + timedelta(hours=1)).isoformat()
        result = await knowledge_tools.search_knowledge(since=future)
        data = json.loads(result)

        assert data["count"] == 0

    async def test_search_records_access(self, knowledge_tools, kg):
        """Search results are recorded in knowledge_access table"""
        await knowledge_tools.save_knowledge("user_preference", "test_pref", "test content")

        await knowledge_tools.search_knowledge(entity_type="user_preference")

        cursor = kg.conn.execute(
            "SELECT entity_id, query_context FROM knowledge_access WHERE query_context = 'agent_search'"
        )
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "user_preference:test_pref"


class TestTriggerSynthesis:
    """Test synthesis trigger mechanism"""

    async def test_trigger_sets_flag(self, knowledge_tools):
        """Triggering synthesis sets flag on agent"""
        from unittest.mock import MagicMock

        mock_agent = MagicMock()
        knowledge_tools.agent = mock_agent

        result = await knowledge_tools.trigger_synthesis(focus="preferences")

        assert "Synthesis scheduled" in result
        assert mock_agent._pending_synthesis is not None
        assert mock_agent._pending_synthesis["focus"] == "preferences"

    async def test_trigger_all_focus(self, knowledge_tools):
        """Trigger synthesis with all focus"""
        from unittest.mock import MagicMock

        mock_agent = MagicMock()
        knowledge_tools.agent = mock_agent

        result = await knowledge_tools.trigger_synthesis(focus="all")

        assert "all" in result.lower()
        assert mock_agent._pending_synthesis["focus"] == "all"


class TestKnowledgeUpsert:
    """Test that insert_knowledge updates existing entries instead of failing"""

    async def test_insert_knowledge_twice_same_key(self, knowledge_tools, kg):
        """Inserting knowledge with the same key updates instead of raising"""
        await knowledge_tools.save_knowledge(
            entity_type="user_preference",
            key="editor",
            content="I use vim",
        )
        # Second insert with same key should update, not raise
        await knowledge_tools.save_knowledge(
            entity_type="user_preference",
            key="editor",
            content="I switched to neovim",
        )

        results = kg.search_knowledge(entity_type="user_preference")
        assert len(results) == 1
        assert "neovim" in results[0]["content"]

    async def test_upsert_preserves_entity_id(self, knowledge_tools, kg):
        """Upserting preserves the same entity ID"""
        await knowledge_tools.save_knowledge(
            entity_type="lesson_learned",
            key="testing_tip",
            content="Always mock external APIs",
        )
        await knowledge_tools.save_knowledge(
            entity_type="lesson_learned",
            key="testing_tip",
            content="Use fixtures for test data",
        )

        entity = kg.get_entity("lesson_learned:testing_tip")
        assert entity is not None
        data = entity.data
        assert "fixtures" in data["content"]


class TestExpireKnowledge:
    """Test expiring knowledge entries"""

    async def test_expire_no_longer_valid(self, knowledge_tools, kg):
        """Expire a fact that stopped being true"""
        await knowledge_tools.save_knowledge("user_preference", "editor", "Uses vim")

        result = await knowledge_tools.expire_knowledge("user_preference:editor", "no_longer_valid")

        assert "Expired" in result
        assert "no_longer_valid" in result
        entity = kg.get_entity("user_preference:editor")
        assert entity is not None
        assert entity.valid_to is not None

    async def test_expire_retract(self, knowledge_tools, kg):
        """Retract an incorrect belief"""
        await knowledge_tools.save_knowledge("lesson_learned", "wrong", "Incorrect assumption")

        result = await knowledge_tools.expire_knowledge("lesson_learned:wrong", "retract")

        assert "Expired" in result
        assert "retract" in result
        entity = kg.get_entity("lesson_learned:wrong")
        assert entity is not None
        assert entity.tx_to is not None

    async def test_expire_not_found(self, knowledge_tools):
        """Expiring a non-existent entity returns error"""
        result = await knowledge_tools.expire_knowledge("no_such:entity", "retract")
        data = json.loads(result)

        assert data["error"] == "not_found"
