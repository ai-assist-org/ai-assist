"""Tests for jev reranking of knowledge-graph semantic search.

Covers the pure rerank helpers in ``jev_client`` (ordering, graceful fallback,
the sync bridge) and the gating in ``KnowledgeGraph.semantic_search`` (only
reranks when jev is configured and ``jev_rerank`` is on, over-fetches a pool to
reorder, and never breaks retrieval when jev fails).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from ai_assist.jev_client import JevError, rerank_candidates, rerank_candidates_sync
from ai_assist.knowledge_graph import KnowledgeGraph


def _jev_config(rerank=True, key="k"):
    return SimpleNamespace(
        jev_enabled=True,
        jev_api_key=key,
        jev_api_url="https://example/systemone",
        jev_model="jev-latest",
        jev_rerank=rerank,
    )


def _score_response(scores):
    """Build a jev response assigning each c{i} the given score."""
    return {"answers": {f"c{i}": {"type": "score", "score": s, "legend": {}} for i, s in enumerate(scores)}}


class TestRerankCandidates:
    async def test_reorders_by_jev_score(self):
        candidates = [{"key": "a", "content": "a"}, {"key": "b", "content": "b"}, {"key": "c", "content": "c"}]
        # jev rates b highest, then c, then a.
        resp = _score_response([1.0, 3.0, 2.0])
        with patch("ai_assist.jev_client.jev_decide", AsyncMock(return_value=resp)):
            out = await rerank_candidates(_jev_config(), "q", candidates)
        assert [c["key"] for c in out] == ["b", "c", "a"]

    async def test_ties_keep_cosine_order(self):
        candidates = [{"key": "a", "content": "a"}, {"key": "b", "content": "b"}]
        resp = _score_response([2.0, 2.0])
        with patch("ai_assist.jev_client.jev_decide", AsyncMock(return_value=resp)):
            out = await rerank_candidates(_jev_config(), "q", candidates)
        assert [c["key"] for c in out] == ["a", "b"]

    async def test_fallback_to_input_order_on_jev_error(self):
        candidates = [{"key": "a", "content": "a"}, {"key": "b", "content": "b"}]
        with patch("ai_assist.jev_client.jev_decide", AsyncMock(side_effect=JevError("boom"))):
            out = await rerank_candidates(_jev_config(), "q", candidates)
        assert [c["key"] for c in out] == ["a", "b"]

    async def test_single_candidate_skips_jev(self):
        decide = AsyncMock()
        with patch("ai_assist.jev_client.jev_decide", decide):
            out = await rerank_candidates(_jev_config(), "q", [{"key": "a", "content": "a"}])
        assert [c["key"] for c in out] == ["a"]
        decide.assert_not_awaited()


class TestRerankSync:
    def test_sync_bridge_reorders(self):
        candidates = [{"key": "a", "content": "a"}, {"key": "b", "content": "b"}]
        resp = _score_response([1.0, 3.0])
        with patch("ai_assist.jev_client.jev_decide", AsyncMock(return_value=resp)):
            out = rerank_candidates_sync(_jev_config(), "q", candidates)
        assert [c["key"] for c in out] == ["b", "a"]

    async def test_sync_bridge_safe_inside_running_loop(self):
        # Called from within an active event loop; the thread bridge must not deadlock.
        candidates = [{"key": "a", "content": "a"}, {"key": "b", "content": "b"}]
        resp = _score_response([3.0, 1.0])
        with patch("ai_assist.jev_client.jev_decide", AsyncMock(return_value=resp)):
            out = rerank_candidates_sync(_jev_config(), "q", candidates)
        assert [c["key"] for c in out] == ["a", "b"]


@pytest.fixture
def kg():
    graph = KnowledgeGraph(db_path=":memory:")
    for key, content in [
        ("deploy_issue", "Deployment failures on Fridays are infrastructure timeouts"),
        ("test_flaky", "Flaky tests in CI come from shared database state"),
        ("cake_recipe", "The best chocolate cake uses Dutch cocoa"),
        ("db_backup", "Nightly database backups run at 2am"),
        ("auth_flow", "Users authenticate with OAuth2 tokens"),
        ("cache_redis", "Redis is used for caching responses"),
    ]:
        graph.insert_knowledge("lesson_learned", key, content)
    yield graph
    graph.close()


class TestSemanticSearchRerankGating:
    def test_no_rerank_when_jev_config_absent(self, kg):
        with patch("ai_assist.knowledge_graph.rerank_candidates_sync") as m:
            kg.semantic_search("deploy error", limit=2)
            m.assert_not_called()

    def test_no_rerank_when_flag_off(self, kg):
        kg.set_jev_config(_jev_config(rerank=False))
        with patch("ai_assist.knowledge_graph.rerank_candidates_sync") as m:
            kg.semantic_search("deploy error", limit=2)
            m.assert_not_called()

    def test_no_rerank_when_jev_not_configured(self, kg):
        kg.set_jev_config(_jev_config(rerank=True, key=None))  # no api key => not configured
        with patch("ai_assist.knowledge_graph.rerank_candidates_sync") as m:
            kg.semantic_search("deploy error", limit=2)
            m.assert_not_called()

    def test_rerank_applied_and_overfetches_pool(self, kg):
        kg.set_jev_config(_jev_config(rerank=True))
        captured = {}

        def fake_rerank(config, query, candidates, **kwargs):
            captured["candidates"] = candidates
            return list(reversed(candidates))

        with patch("ai_assist.knowledge_graph.rerank_candidates_sync", side_effect=fake_rerank):
            results = kg.semantic_search("deploy error", limit=2)

        # Over-fetched more than `limit` so jev had a real pool to reorder.
        assert len(captured["candidates"]) > 2
        # The reranked order (reversed) is applied, then trimmed to `limit`.
        assert results == list(reversed(captured["candidates"]))[:2]

    def test_rerank_failure_falls_back_to_cosine(self, kg):
        kg.set_jev_config(_jev_config(rerank=True))
        with patch("ai_assist.knowledge_graph.rerank_candidates_sync", side_effect=RuntimeError("boom")):
            results = kg.semantic_search("deploy error", limit=2)
        # Retrieval still returns cosine results despite the rerank blowing up.
        assert len(results) >= 1
        assert all("key" in r for r in results)
