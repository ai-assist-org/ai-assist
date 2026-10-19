"""Tests for outcome-based skill confidence tracking."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_assist.introspection_tools import IntrospectionTools
from ai_assist.knowledge_graph import KnowledgeGraph
from ai_assist.skill_performance import (
    INITIAL_CONFIDENCE,
    MAX_NEGATIVE_EXAMPLES,
    REVIEW_THRESHOLD,
    SkillPerformanceStore,
    update_confidence,
)


@pytest.fixture
def kg():
    graph = KnowledgeGraph(db_path=":memory:")
    yield graph
    graph.close()


@pytest.fixture
def store(kg):
    return SkillPerformanceStore(kg)


# --- pure confidence math -------------------------------------------------


def test_success_moves_a_tenth_toward_one():
    assert update_confidence(0.0, "success") == pytest.approx(0.1)
    assert update_confidence(0.5, "success") == pytest.approx(0.55)


def test_success_is_asymptotic_to_one():
    c = 0.0
    for _ in range(200):
        c = update_confidence(c, "success")
    assert c <= 1.0
    assert c > 0.99


def test_failure_penalty_is_double_a_success_gain():
    assert update_confidence(0.5, "failure") == pytest.approx(0.3)


def test_partial_is_mild_penalty():
    assert update_confidence(0.5, "partial") == pytest.approx(0.45)


def test_confidence_is_clamped():
    assert update_confidence(0.1, "failure") == 0.0
    assert update_confidence(1.0, "success") == 1.0


def test_unknown_outcome_raises():
    with pytest.raises(ValueError):
        update_confidence(0.5, "bogus")  # type: ignore[arg-type]


# --- store behaviour ------------------------------------------------------


def test_unseen_skill_returns_none(store):
    assert store.get("never-used") is None


def test_record_outcome_creates_record_at_initial_confidence(store):
    perf = store.record_outcome("web-research", "success")
    # Started at INITIAL_CONFIDENCE, then one success applied.
    assert perf.confidence == pytest.approx(update_confidence(INITIAL_CONFIDENCE, "success"))
    assert perf.success_count == 1
    assert perf.failure_count == 0


def test_outcomes_accumulate_and_persist(store):
    store.record_outcome("flaky", "failure")
    store.record_outcome("flaky", "failure")
    reloaded = store.get("flaky")
    assert reloaded is not None
    assert reloaded.failure_count == 2
    assert reloaded.confidence == pytest.approx(0.6)  # 1.0 -0.2 -0.2


def test_record_invocation_counts_uses_without_touching_confidence(store):
    perf = store.record_invocation("web-research")
    assert perf.usage_count == 1
    assert perf.confidence == INITIAL_CONFIDENCE
    assert perf.last_used is not None

    store.record_invocation("web-research")
    reloaded = store.get("web-research")
    assert reloaded is not None
    assert reloaded.usage_count == 2
    assert reloaded.confidence == INITIAL_CONFIDENCE


def test_usage_count_is_owned_by_invocation_not_outcome(store):
    # Outcomes record results of a use but must not double-count usage.
    store.record_invocation("web-research")
    store.record_outcome("web-research", "success")
    reloaded = store.get("web-research")
    assert reloaded is not None
    assert reloaded.usage_count == 1
    assert reloaded.success_count == 1


def test_failure_reason_stored_as_negative_example(store):
    perf = store.record_outcome("flaky", "failure", reason="timed out on step 2")
    assert perf.negative_examples == ["timed out on step 2"]


def test_success_does_not_store_negative_example(store):
    perf = store.record_outcome("solid", "success", reason="ignored")
    assert perf.negative_examples == []


def test_negative_examples_are_capped(store):
    for i in range(MAX_NEGATIVE_EXAMPLES + 5):
        store.record_outcome("flaky", "failure", reason=f"reason {i}")
    perf = store.get("flaky")
    assert perf is not None
    assert len(perf.negative_examples) == MAX_NEGATIVE_EXAMPLES
    # Oldest reasons dropped, newest kept.
    assert perf.negative_examples[-1] == f"reason {MAX_NEGATIVE_EXAMPLES + 4}"


def test_flagged_for_review_lists_low_confidence_sorted(store):
    store.record_outcome("good", "success")  # stays high
    for _ in range(4):
        store.record_outcome("bad", "failure")  # 1.0 - 4*0.2 = 0.2
    for _ in range(3):
        store.record_outcome("worse", "failure")  # 1.0 - 3*0.2 = 0.4 -> not < 0.4

    flagged = store.flagged_for_review()
    names = [p.name for p in flagged]
    assert "good" not in names
    assert "bad" in names
    # 0.4 is not strictly below the threshold.
    assert "worse" not in names
    assert all(p.confidence < REVIEW_THRESHOLD for p in flagged)
    assert names == sorted(names, key=lambda n: next(p.confidence for p in flagged if p.name == n))


# --- isolation from semantic recall (the point of the KG changes) ---------


def test_skill_performance_never_leaks_into_semantic_search(store, kg):
    store.record_outcome("web-research", "failure", reason="search returned nothing useful")

    # A semantic query strongly overlapping the record text must not surface it.
    results = kg.semantic_search("web research search returned nothing", limit=10)
    assert all(r.get("entity_type") != "skill_performance" for r in results)


def test_usage_stats_lists_skills_most_used_first(store):
    for _ in range(3):
        store.record_invocation("popular")
    store.record_invocation("rare")

    stats = store.usage_stats()
    names = [p.name for p in stats]
    assert names == ["popular", "rare"]
    assert stats[0].usage_count == 3
    assert stats[1].usage_count == 1


def test_get_skill_help_records_invocation(kg):
    """The introspection tool hook logs a skill invocation into the KG."""
    content = SimpleNamespace(
        metadata=SimpleNamespace(description="Research the web", skill_path="/tmp/web-research"),
        body="do the research",
    )
    agent = SimpleNamespace(
        knowledge_graph=kg,
        skills_manager=SimpleNamespace(loaded_skills={"web-research": content}),
    )
    tools = IntrospectionTools(knowledge_graph=kg, agent=agent)

    tools._get_skill_help({"skill_name": "web-research"})

    perf = SkillPerformanceStore(kg).get("web-research")
    assert perf is not None
    assert perf.usage_count == 1
    assert perf.confidence == INITIAL_CONFIDENCE


def test_get_skill_help_unknown_skill_records_nothing(kg):
    agent = SimpleNamespace(
        knowledge_graph=kg,
        skills_manager=SimpleNamespace(loaded_skills={}),
    )
    tools = IntrospectionTools(knowledge_graph=kg, agent=agent)

    tools._get_skill_help({"skill_name": "does-not-exist"})

    assert SkillPerformanceStore(kg).get("does-not-exist") is None


async def test_manual_skill_invocation_records_usage(kg):
    """The /<skill-name> manual path logs an invocation too."""
    from ai_assist.tui_interactive import handle_skill_invocation

    content = SimpleNamespace(
        metadata=SimpleNamespace(user_invocable=True, argument_hint=None),
        body="do the research",
    )
    agent = SimpleNamespace(
        knowledge_graph=kg,
        skills_manager=SimpleNamespace(loaded_skills={"web-research": content}),
    )

    with patch("ai_assist.tui_interactive.query_with_feedback", new=AsyncMock()) as run:
        handled = await handle_skill_invocation("/web-research some query", agent, MagicMock(), MagicMock(), None)

    assert handled is True
    run.assert_awaited_once()
    perf = SkillPerformanceStore(kg).get("web-research")
    assert perf is not None
    assert perf.usage_count == 1


async def test_skill_stats_command_renders_usage(kg):
    from ai_assist.tui_interactive import handle_skill_management_command

    SkillPerformanceStore(kg).record_invocation("web-research")
    agent = SimpleNamespace(knowledge_graph=kg)
    console = MagicMock()

    handled = await handle_skill_management_command("/skill/stats", agent, console)

    assert handled is True
    console.print.assert_called()  # rendered a table rather than erroring


async def test_skill_stats_command_handles_no_usage(kg):
    from ai_assist.tui_interactive import handle_skill_management_command

    agent = SimpleNamespace(knowledge_graph=kg)
    console = MagicMock()

    handled = await handle_skill_management_command("/skill/stats", agent, console)

    assert handled is True
    console.print.assert_any_call("No skill usage recorded yet.")


def test_skill_performance_excluded_from_backfill(store, kg):
    store.record_outcome("web-research", "failure")
    # Backfill embeds entities that lack vectors; skill_performance must be skipped.
    kg.backfill_embeddings()
    count = kg.conn.execute(
        "SELECT COUNT(*) FROM vec_embeddings v JOIN entities e ON v.entity_id = e.id "
        "WHERE e.entity_type = 'skill_performance'"
    ).fetchone()[0]
    assert count == 0
