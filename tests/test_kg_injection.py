"""Tests for KG auto-context injection and learning reinforcement in system prompt"""

from datetime import datetime

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig
from ai_assist.knowledge_graph import KnowledgeGraph


def _make_agent(kg=None):
    config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={})
    return AiAssistAgent(config, knowledge_graph=kg)


def _prompt_text(blocks: list[dict]) -> str:
    return "\n\n".join(b["text"] for b in blocks)


class TestGetKgLearningsSection:
    def test_no_kg_returns_empty(self):
        agent = _make_agent(kg=None)
        assert agent._get_kg_learnings_section() == ""

    def test_empty_kg_returns_empty(self):
        kg = KnowledgeGraph(":memory:")
        agent = _make_agent(kg=kg)
        assert agent._get_kg_learnings_section() == ""

    def test_preferences_always_injected(self):
        kg = KnowledgeGraph(":memory:")
        kg.insert_knowledge(
            entity_type="user_preference",
            key="report_style",
            content="User prefers concise reports with bullet points",
            confidence=0.9,
        )
        agent = _make_agent(kg=kg)
        section = agent._get_kg_learnings_section()
        assert "User Preferences" in section
        assert "report_style" in section
        assert "concise reports" in section

    def test_preferences_injected_without_query_text(self):
        """Preferences are injected even when there's no current query"""
        kg = KnowledgeGraph(":memory:")
        kg.insert_knowledge(
            entity_type="user_preference",
            key="code_style",
            content="Prefers minimal comments",
            confidence=0.8,
        )
        agent = _make_agent(kg=kg)
        agent._current_query_text = ""
        section = agent._get_kg_learnings_section()
        assert "code_style" in section

    def test_low_confidence_preferences_filtered(self):
        kg = KnowledgeGraph(":memory:")
        kg.insert_knowledge(
            entity_type="user_preference",
            key="uncertain_pref",
            content="Maybe prefers tabs",
            confidence=0.3,
        )
        agent = _make_agent(kg=kg)
        section = agent._get_kg_learnings_section()
        assert "uncertain_pref" not in section

    def test_lessons_injected_when_query_matches(self):
        kg = KnowledgeGraph(":memory:")
        kg.insert_knowledge(
            entity_type="lesson_learned",
            key="pipeline_timeout",
            content="Pipeline failures on Fridays are usually infrastructure timeouts",
            confidence=0.9,
        )
        agent = _make_agent(kg=kg)
        agent._current_query_text = "Why did the pipeline fail?"
        section = agent._get_kg_learnings_section()
        assert "pipeline" in section.lower()
        assert "infrastructure timeouts" in section

    def test_lessons_excluded_when_query_doesnt_match(self):
        kg = KnowledgeGraph(":memory:")
        kg.insert_knowledge(
            entity_type="lesson_learned",
            key="pipeline_timeout",
            content="Pipeline failures on Fridays are usually infrastructure timeouts",
            confidence=0.9,
        )
        agent = _make_agent(kg=kg)
        agent._current_query_text = "What is the weather today?"
        section = agent._get_kg_learnings_section()
        assert "infrastructure timeouts" not in section

    def test_no_query_text_skips_lessons(self):
        kg = KnowledgeGraph(":memory:")
        kg.insert_knowledge(
            entity_type="lesson_learned",
            key="some_lesson",
            content="An important lesson about testing",
            confidence=0.9,
        )
        agent = _make_agent(kg=kg)
        agent._current_query_text = ""
        section = agent._get_kg_learnings_section()
        assert "important lesson" not in section

    def test_project_context_injected_when_relevant(self):
        kg = KnowledgeGraph(":memory:")
        kg.insert_knowledge(
            entity_type="project_context",
            key="deployment_process",
            content="Deployments require approval from team lead",
            confidence=0.8,
        )
        agent = _make_agent(kg=kg)
        agent._current_query_text = "How does the deployment process work?"
        section = agent._get_kg_learnings_section()
        assert "approval from team lead" in section

    def test_decision_rationale_injected_when_relevant(self):
        kg = KnowledgeGraph(":memory:")
        kg.insert_knowledge(
            entity_type="decision_rationale",
            key="chose_pytest",
            content="Chose pytest over unittest for better fixture support",
            confidence=0.85,
        )
        agent = _make_agent(kg=kg)
        agent._current_query_text = "Which testing framework should we use for pytest?"
        section = agent._get_kg_learnings_section()
        assert "pytest" in section

    def test_low_confidence_lessons_filtered(self):
        kg = KnowledgeGraph(":memory:")
        kg.insert_knowledge(
            entity_type="lesson_learned",
            key="vague_pattern",
            content="Maybe something about network issues",
            confidence=0.4,
        )
        agent = _make_agent(kg=kg)
        agent._current_query_text = "Tell me about network issues"
        section = agent._get_kg_learnings_section()
        assert "vague_pattern" not in section

    def test_hard_cap_truncation(self):
        kg = KnowledgeGraph(":memory:")
        for i in range(50):
            kg.insert_knowledge(
                entity_type="user_preference",
                key=f"preference_{i}",
                content=f"A verbose preference description number {i} with lots of detail " * 3,
                confidence=1.0,
            )
        agent = _make_agent(kg=kg)
        section = agent._get_kg_learnings_section()
        assert len(section) <= 2000  # 1500 cap + header

    def test_learnings_in_system_prompt(self):
        kg = KnowledgeGraph(":memory:")
        kg.insert_knowledge(
            entity_type="user_preference",
            key="style",
            content="Prefers detailed explanations",
            confidence=1.0,
        )
        agent = _make_agent(kg=kg)
        prompt = _prompt_text(agent._build_system_prompt())
        assert "What You Know" in prompt
        assert "detailed explanations" in prompt

    def test_query_aware_learnings_max_five(self):
        kg = KnowledgeGraph(":memory:")
        for i in range(20):
            kg.insert_knowledge(
                entity_type="lesson_learned",
                key=f"testing_lesson_{i}",
                content=f"Testing lesson number {i} about testing patterns",
                confidence=0.9,
            )
        agent = _make_agent(kg=kg)
        agent._current_query_text = "Tell me about testing patterns"
        section = agent._get_kg_learnings_section()
        # Count lesson entries (lines starting with "- [")
        lesson_lines = [line for line in section.split("\n") if line.strip().startswith("- [")]
        assert len(lesson_lines) <= 5

    def test_learnings_ranked_by_semantic_similarity(self):
        """Results are ranked by semantic relevance to the query"""
        kg = KnowledgeGraph(":memory:")
        kg.insert_knowledge(
            entity_type="lesson_learned",
            key="deploy_lesson",
            content="Deployment failures are usually infrastructure timeouts",
            confidence=0.9,
        )
        kg.insert_knowledge(
            entity_type="lesson_learned",
            key="cake_recipe",
            content="The best chocolate cake recipe uses Dutch cocoa",
            confidence=0.9,
        )
        agent = _make_agent(kg=kg)
        agent._current_query_text = "Why do deployments fail?"
        section = agent._get_kg_learnings_section()
        # Deploy lesson should appear, cake recipe should not (or appear after)
        assert "infrastructure timeouts" in section


class TestGetKgAutoContextSection:
    def test_no_kg_returns_empty(self):
        agent = _make_agent(kg=None)
        agent._current_query_text = "some query"
        assert agent._get_kg_auto_context_section() == ""

    def test_no_query_text_returns_empty(self):
        kg = KnowledgeGraph(":memory:")
        agent = _make_agent(kg=kg)
        agent._current_query_text = ""
        assert agent._get_kg_auto_context_section() == ""

    def test_matching_entity_injected(self):
        kg = KnowledgeGraph(":memory:")
        kg.insert_entity(
            entity_type="component",
            entity_id="openshift-4.16",
            valid_from=datetime.now(),
            data={"name": "OpenShift", "version": "4.16", "status": "active"},
        )

        agent = _make_agent(kg=kg)
        agent._current_query_text = "What happened with OpenShift recently?"
        section = agent._get_kg_auto_context_section()
        assert "openshift-4.16" in section

    def test_non_matching_entity_excluded(self):
        kg = KnowledgeGraph(":memory:")
        kg.insert_entity(
            entity_type="component",
            entity_id="openshift-4.16",
            valid_from=datetime.now(),
            data={"name": "OpenShift", "version": "4.16"},
        )

        agent = _make_agent(kg=kg)
        agent._current_query_text = "How is the weather?"
        section = agent._get_kg_auto_context_section()
        assert section == ""

    def test_knowledge_types_excluded(self):
        """Knowledge entities are handled by learnings section, not auto context"""
        kg = KnowledgeGraph(":memory:")
        kg.insert_knowledge(
            entity_type="user_preference",
            key="testing_pref",
            content="Prefers testing with pytest",
            confidence=1.0,
        )
        agent = _make_agent(kg=kg)
        agent._current_query_text = "Tell me about testing preferences"
        section = agent._get_kg_auto_context_section()
        # user_preference should NOT appear in auto context
        assert "testing_pref" not in section

    def test_max_five_results(self):
        kg = KnowledgeGraph(":memory:")
        for i in range(20):
            kg.insert_entity(
                entity_type="job",
                entity_id=f"build-job-{i}",
                valid_from=datetime.now(),
                data={"name": f"build job {i}", "status": "failure"},
            )

        agent = _make_agent(kg=kg)
        agent._current_query_text = "Show me the build jobs"
        section = agent._get_kg_auto_context_section()
        entity_lines = [line for line in section.split("\n") if line.strip().startswith("- [")]
        assert len(entity_lines) <= 5

    def test_deduplication(self):
        kg = KnowledgeGraph(":memory:")
        kg.insert_entity(
            entity_type="job",
            entity_id="build-deploy-job",
            valid_from=datetime.now(),
            data={"name": "build and deploy job", "status": "success"},
        )

        agent = _make_agent(kg=kg)
        # Both "build" and "deploy" match the same entity
        agent._current_query_text = "Check the build deploy status"
        section = agent._get_kg_auto_context_section()
        assert section.count("build-deploy-job") == 1

    def test_auto_context_in_system_prompt(self):
        kg = KnowledgeGraph(":memory:")
        kg.insert_entity(
            entity_type="ticket",
            entity_id="PROJECT-999",
            valid_from=datetime.now(),
            data={"name": "PROJECT-999", "summary": "Critical production issue", "status": "Open"},
        )

        agent = _make_agent(kg=kg)
        agent._current_query_text = "What is the critical production issue?"
        prompt = _prompt_text(agent._build_system_prompt())
        assert "Relevant Context" in prompt


class TestNoKgPrefix:
    def test_no_kg_prefix_strips_from_query_text(self):
        """@no-kg prefix is removed from _current_query_text"""
        agent = _make_agent()
        agent._apply_no_kg_prefix("@no-kg what is X")
        assert agent._current_query_text == "what is X"
        assert agent._no_kg is True

    def test_no_prefix_keeps_query_unchanged(self):
        agent = _make_agent()
        agent._apply_no_kg_prefix("what is X")
        assert agent._current_query_text == "what is X"

    def test_no_prefix_preserves_existing_no_kg_flag(self):
        """Nested calls (depth > 1) without @no-kg should not reset an outer _no_kg=True"""
        agent = _make_agent()
        agent._no_kg = True
        agent._query_depth = 2  # Simulate nested call
        agent._apply_no_kg_prefix("inner prompt without prefix")
        assert agent._no_kg is True

    def test_top_level_call_resets_no_kg(self):
        """Top-level calls (depth <= 1) reset _no_kg so periodic tasks start clean"""
        agent = _make_agent()
        agent._no_kg = True
        agent._query_depth = 1  # Top-level call
        agent._apply_no_kg_prefix("normal prompt")
        assert agent._no_kg is False

    def test_no_kg_suppresses_learnings_section(self):
        kg = KnowledgeGraph(":memory:")
        kg.insert_knowledge(
            entity_type="user_preference",
            key="style",
            content="Prefers detailed explanations",
            confidence=1.0,
        )
        agent = _make_agent(kg=kg)
        agent._no_kg = True
        assert agent._get_kg_learnings_section() == ""

    def test_no_kg_suppresses_auto_context_section(self):
        kg = KnowledgeGraph(":memory:")
        kg.insert_entity(
            entity_type="component",
            entity_id="openshift-4.16",
            valid_from=datetime.now(),
            data={"name": "OpenShift", "version": "4.16"},
        )
        agent = _make_agent(kg=kg)
        agent._current_query_text = "openshift"
        agent._no_kg = True
        assert agent._get_kg_auto_context_section() == ""

    def test_no_kg_suppresses_kg_in_system_prompt(self):
        kg = KnowledgeGraph(":memory:")
        kg.insert_knowledge(
            entity_type="user_preference",
            key="style",
            content="Prefers detailed explanations",
            confidence=1.0,
        )
        agent = _make_agent(kg=kg)
        agent._no_kg = True
        prompt = _prompt_text(agent._build_system_prompt())
        assert "Knowledge Graph" not in prompt
        assert "What You Know" not in prompt

    def test_without_no_kg_prompt_still_has_kg(self):
        """Sanity: without @no-kg, KG is present"""
        kg = KnowledgeGraph(":memory:")
        kg.insert_knowledge(
            entity_type="user_preference",
            key="style",
            content="Prefers detailed explanations",
            confidence=1.0,
        )
        agent = _make_agent(kg=kg)
        agent._no_kg = False
        prompt = _prompt_text(agent._build_system_prompt())
        assert "Knowledge Graph" in prompt
