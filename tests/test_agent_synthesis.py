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
    """Test report snapshotting and connection discovery in scheduled synthesis.

    Conversation fact-extraction is deliberately NOT part of this task — it is
    handled at compaction, ``/clear`` and on exit in the interactive TUI.
    """

    @pytest.mark.asyncio
    async def test_synthesis_from_kg_ignores_conversations(self, agent, kg):
        """Conversation entities must not be mined by the scheduled task."""
        now = datetime.now()
        kg.insert_entity(
            entity_type="conversation",
            data={"user": "I prefer pytest over unittest", "assistant": "Noted, using pytest."},
            valid_from=now - timedelta(hours=2),
            tx_from=now - timedelta(hours=2),
        )

        # No report change → early exit, no LLM call, no insights extracted from
        # the conversation above.
        with patch.object(agent, "_get_report_snapshots", return_value={}):
            with patch.object(agent.anthropic.messages, "stream") as mock_stream:
                result = await agent._run_synthesis_from_kg()

            mock_stream.assert_not_called()

        assert "No new reports" in result
        for etype in ["user_preference", "lesson_learned", "project_context", "decision_rationale"]:
            assert len(kg.search_knowledge(entity_type=etype)) == 0

    @pytest.mark.asyncio
    async def test_synthesis_from_kg_no_new_reports(self, agent, kg):
        """Synthesis with no new reports should return early without a marker."""
        with patch.object(agent, "_get_report_snapshots", return_value={}):
            result = await agent._run_synthesis_from_kg()

        assert "No new reports" in result

        # No synthesis_marker should be created (nothing was processed)
        now = datetime.now()
        markers = kg.query_as_of(now, entity_type="synthesis_marker")
        assert len(markers) == 0

    @pytest.mark.asyncio
    async def test_synthesis_from_kg_processes_new_reports(self, agent, kg):
        """When reports change, connection discovery should run and mark progress."""
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
    async def test_synthesis_from_kg_no_llm_calls_when_reports_unchanged(self, agent, kg):
        """No LLM calls should be made when reports have not changed."""
        now = datetime.now()

        # Create a previous synthesis marker with report snapshots
        kg.insert_entity(
            entity_type="synthesis_marker",
            data={"reports_processed": {"existing_report.md": "2026-02-25T08:00:00"}},
            valid_from=now - timedelta(hours=1),
        )

        # Mock _get_report_snapshots to return same snapshots (no change)
        with patch.object(agent, "_get_report_snapshots", return_value={"existing_report.md": "2026-02-25T08:00:00"}):
            with patch.object(agent.anthropic.messages, "stream") as mock_stream:
                result = await agent._run_synthesis_from_kg()

            # No LLM calls should have been made
            mock_stream.assert_not_called()

        assert "No new reports" in result


class TestSaveInsights:
    """Tests for the shared _save_insights helper used by synthesis and compaction."""

    def test_save_insights_writes_retrievable_entity(self, agent, kg):
        n = agent._save_insights(
            [
                {
                    "category": "project_context",
                    "key": "code-loc",
                    "content": "the tests live at ~/external/eco-gotests",
                    "confidence": 0.9,
                    "tags": ["path"],
                }
            ],
            "compaction_extraction",
        )

        assert n == 1
        results = kg.search_knowledge(entity_type="project_context")
        saved = next(r for r in results if r["key"] == "code-loc")
        assert "eco-gotests" in saved["content"]

    def test_save_insights_skips_bad_items(self, agent):
        n = agent._save_insights(
            [
                {"category": "bogus_type", "key": "k", "content": "c"},
                {"category": "project_context", "key": "ok", "content": "c"},
            ],
            "compaction_extraction",
        )

        assert n == 1

    def test_save_insights_no_kg_returns_zero(self, config):
        agent_no_kg = AiAssistAgent(config)
        n = agent_no_kg._save_insights(
            [{"category": "project_context", "key": "k", "content": "c"}],
            "compaction_extraction",
        )
        assert n == 0


# Labeled fixture: clear re-statements of the SAME fact (a file/code location
# repeated with light rewording). These reliably clear DEDUP_SIM_THRESHOLD in
# both directions, so extraction under a fresh slug must reuse the existing key.
_DUP_PAIRS = [
    (
        "The main config file is at /etc/ai-assist/config.yaml",
        "The configuration file lives at /etc/ai-assist/config.yaml",
    ),
    (
        "The AWL script at /home/fred/ai-assist/awl/eda-reports.awl generates EDA reports",
        "The EDA reports AWL script is located at /home/fred/ai-assist/awl/eda-reports.awl",
    ),
    (
        "Service logs are written to /var/log/ai-assist/service.log",
        "Service logs are written to /var/log/ai-assist/service.log",
    ),
]

# Labeled fixture: distinct-but-similar facts. These MUST NOT merge — a false
# merge overwrites a real, different fact irrecoverably. Includes the hardest
# case (same "X's jira user is Y" template, different person) which embeddings
# rank deceptively high (~0.74) yet still below the conservative threshold.
_DISTINCT_PAIRS = [
    (
        "The AWL script at /home/fred/ai-assist/awl/eda-reports.awl generates EDA reports",
        "The AWL script at /home/fred/ai-assist/awl/quarterly-review.awl generates quarterly reviews",
    ),
    (
        "Jennifer Chen's jira user is jenchen@redhat.com",
        "Semih Kisa's jira user is skisa@redhat.com",
    ),
    (
        "The eco-gotests repo is at ~/work/eco-gotests",
        "The dci-mcp-server repo is at ~/work/dci-mcp-server",
    ),
]


class TestDedupKeyReuse:
    """Evaluation of conservative semantic key-reuse in _save_insights.

    Two properties, measured on labeled dup/distinct pairs with real embeddings:
    true-merges (restatements collapse onto one key) and — critically — zero
    false-merges (distinct facts stay separate, since a merge is destructive).
    """

    def _save(self, agent, key, content):
        return agent._save_insights(
            [{"category": "project_context", "key": key, "content": content}],
            "compaction_extraction",
        )

    def _live_keys(self, kg):
        return {r["key"] for r in kg.search_knowledge(entity_type="project_context")}

    def test_restatement_reuses_existing_key(self, agent, kg):
        """A re-stated fact under a new slug supersedes the original (one live row)."""
        for i, (existing, restated) in enumerate(_DUP_PAIRS):
            self._save(agent, f"orig_{i}", existing)
            before = self._live_keys(kg)
            self._save(agent, f"restated_{i}", restated)
            after = self._live_keys(kg)
            # No new key introduced: the restatement folded onto orig_{i}.
            assert after == before, f"pair {i}: restatement forked a new key {after - before}"
            assert f"restated_{i}" not in after
            saved = next(r for r in kg.search_knowledge(entity_type="project_context") if r["key"] == f"orig_{i}")
            assert saved["content"] == restated  # last write wins

    def test_distinct_facts_do_not_merge(self, agent, kg):
        """Distinct-but-similar facts each keep their own key (no data loss)."""
        for i, (first, second) in enumerate(_DISTINCT_PAIRS):
            self._save(agent, f"a_{i}", first)
            self._save(agent, f"b_{i}", second)
            keys = self._live_keys(kg)
            assert f"a_{i}" in keys and f"b_{i}" in keys, f"pair {i}: distinct facts wrongly merged"

    def test_loose_paraphrase_stays_separate(self, agent, kg):
        """Documents the conservative tradeoff: a loosely-reworded restatement
        that falls below threshold is NOT merged (kept as a distinct key, i.e.
        today's behaviour). We accept missed merges to guarantee no false ones."""
        self._save(agent, "loc_a", "The eco-gotests tests are located at ~/work/eco-gotests")
        self._save(agent, "loc_b", "eco-gotests is located at ~/work/eco-gotests on my machine")
        keys = self._live_keys(kg)
        assert "loc_a" in keys and "loc_b" in keys

    def test_non_project_context_is_not_deduped(self, agent, kg):
        """Dedup is scoped to project_context; other knowledge types keep their key."""
        self._save(agent, "cfg", "The main config file is at /etc/ai-assist/config.yaml")
        agent._save_insights(
            [
                {
                    "category": "lesson_learned",
                    "key": "cfg_lesson",
                    "content": "The configuration file lives at /etc/ai-assist/config.yaml",
                }
            ],
            "compaction_extraction",
        )
        assert "cfg" in self._live_keys(kg)
        lessons = {r["key"] for r in kg.search_knowledge(entity_type="lesson_learned")}
        assert "cfg_lesson" in lessons

    def test_similarity_separation_holds(self, kg):
        """Guards the threshold: every duplicate scores above it, every distinct below.

        If the embedding model or DEDUP_SIM_THRESHOLD drifts so this gap closes,
        this fails loudly rather than silently enabling false merges.
        """

        def score(existing, probe):
            g = KnowledgeGraph(":memory:")
            g.insert_knowledge("project_context", "x", existing)
            res = g.semantic_search(probe, limit=1, entity_types=["project_context"], include_future=True)
            g.close()
            return res[0]["score"] if res else 0.0

        dup_scores = [score(a, b) for a, b in _DUP_PAIRS]
        distinct_scores = [score(a, b) for a, b in _DISTINCT_PAIRS]
        thr = AiAssistAgent.DEDUP_SIM_THRESHOLD
        assert min(dup_scores) >= thr, f"a duplicate scored below threshold: {dup_scores}"
        assert max(distinct_scores) < thr, f"a distinct pair scored at/above threshold: {distinct_scores}"
