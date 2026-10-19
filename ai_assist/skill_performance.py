"""Outcome-based confidence tracking for skills.

A prototype of a self-improvement loop inspired by gitagent's skill
crystallization: each skill accumulates a confidence score that is nudged up on
success and down (harder) on failure, so unreliable skills surface for review.

Performance records are stored in the knowledge graph as ``skill_performance``
entities keyed by skill name. That type is listed in
``KnowledgeGraph.NON_EMBEDDED_TYPES`` so these bookkeeping records never leak
into semantic recall or system-prompt context injection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from .knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

ENTITY_TYPE = "skill_performance"

# Asymmetric reward shaping (ported from gitagent's reinforcement loop): a win
# moves confidence a tenth of the way to 1.0, a failure costs twice as much as a
# success would gain, and a partial result is a mild penalty.
SUCCESS_REWARD = 0.1
FAILURE_PENALTY = 0.2
PARTIAL_PENALTY = 0.05
INITIAL_CONFIDENCE = 1.0

# Skills below this confidence are flagged for human review.
REVIEW_THRESHOLD = 0.4

# Keep only the most recent failure reasons so a record cannot grow unbounded.
MAX_NEGATIVE_EXAMPLES = 10

Outcome = Literal["success", "failure", "partial"]


def update_confidence(confidence: float, outcome: Outcome) -> float:
    """Return the new confidence after one outcome, clamped to [0.0, 1.0].

    success -> confidence + 0.1 * (1 - confidence)  (asymptotic to 1.0)
    failure -> confidence - 0.2
    partial -> confidence - 0.05
    """
    if outcome == "success":
        confidence += SUCCESS_REWARD * (1.0 - confidence)
    elif outcome == "failure":
        confidence -= FAILURE_PENALTY
    elif outcome == "partial":
        confidence -= PARTIAL_PENALTY
    else:  # pragma: no cover - guarded by the Outcome type
        raise ValueError(f"Unknown outcome: {outcome}")
    return max(0.0, min(1.0, confidence))


@dataclass
class SkillPerformance:
    """In-memory view of a skill's accumulated performance."""

    name: str
    confidence: float = INITIAL_CONFIDENCE
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    partial_count: int = 0
    negative_examples: list[str] = field(default_factory=list)
    updated_at: str | None = None
    last_used: str | None = None

    @property
    def needs_review(self) -> bool:
        return self.confidence < REVIEW_THRESHOLD

    @classmethod
    def from_data(cls, name: str, data: dict) -> SkillPerformance:
        return cls(
            name=name,
            confidence=data.get("confidence", INITIAL_CONFIDENCE),
            usage_count=data.get("usage_count", 0),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            partial_count=data.get("partial_count", 0),
            negative_examples=list(data.get("negative_examples", [])),
            updated_at=data.get("updated_at"),
            last_used=data.get("last_used"),
        )

    def to_data(self) -> dict:
        return {
            "name": self.name,
            "confidence": self.confidence,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "partial_count": self.partial_count,
            "negative_examples": self.negative_examples,
            "updated_at": self.updated_at,
            "last_used": self.last_used,
        }


class SkillPerformanceStore:
    """Persist and update skill confidence in the knowledge graph."""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def _entity_id(self, name: str) -> str:
        return f"{ENTITY_TYPE}:{name}"

    def get(self, name: str) -> SkillPerformance | None:
        """Return the stored performance for a skill, or None if unseen."""
        entity = self.kg.get_entity(self._entity_id(name))
        if entity is None:
            return None
        return SkillPerformance.from_data(name, entity.data)

    def _persist(self, perf: SkillPerformance) -> None:
        self.kg.upsert_entity(
            entity_type=ENTITY_TYPE,
            entity_id=self._entity_id(perf.name),
            data=perf.to_data(),
            valid_from=datetime.now(),
        )

    def record_invocation(self, name: str) -> SkillPerformance:
        """Record that a skill was invoked (used), without an outcome label.

        This is the invocation-only signal: it bumps usage_count and last_used
        but leaves confidence untouched. ``usage_count`` is owned solely by this
        method so an outcome recorded later for the same use is not double-counted.
        """
        perf = self.get(name) or SkillPerformance(name=name)
        perf.usage_count += 1
        now = datetime.now().isoformat()
        perf.last_used = now
        perf.updated_at = now
        self._persist(perf)
        return perf

    def record_outcome(
        self,
        name: str,
        outcome: Outcome,
        reason: str | None = None,
    ) -> SkillPerformance:
        """Record one skill outcome and persist the updated confidence.

        A new skill starts at INITIAL_CONFIDENCE. ``reason`` is kept as a capped
        negative example on failure/partial so failed approaches aren't repeated.
        """
        perf = self.get(name) or SkillPerformance(name=name)

        perf.confidence = update_confidence(perf.confidence, outcome)
        if outcome == "success":
            perf.success_count += 1
        elif outcome == "failure":
            perf.failure_count += 1
        else:
            perf.partial_count += 1

        if outcome in ("failure", "partial") and reason:
            perf.negative_examples.append(reason)
            if len(perf.negative_examples) > MAX_NEGATIVE_EXAMPLES:
                perf.negative_examples = perf.negative_examples[-MAX_NEGATIVE_EXAMPLES:]

        perf.updated_at = datetime.now().isoformat()

        self._persist(perf)
        if perf.needs_review:
            logger.info("Skill %r dropped to confidence %.2f (flagged for review)", name, perf.confidence)
        return perf

    def usage_stats(self) -> list[SkillPerformance]:
        """Return all tracked skills, most-used first.

        This is the read side of invocation logging: which skills are actually
        used, how often, and when last. Answers whether there is any usage
        variance worth scoring before investing in outcome labeling.
        """
        stats = [
            SkillPerformance.from_data(e.data.get("name", e.id), e.data)
            for e in self.kg.query_as_of(datetime.now(), entity_type=ENTITY_TYPE)
        ]
        stats.sort(key=lambda p: p.usage_count, reverse=True)
        return stats

    def flagged_for_review(self) -> list[SkillPerformance]:
        """Return skills whose confidence has fallen below REVIEW_THRESHOLD."""
        flagged: list[SkillPerformance] = []
        for entity in self.kg.query_as_of(datetime.now(), entity_type=ENTITY_TYPE):
            perf = SkillPerformance.from_data(entity.data.get("name", entity.id), entity.data)
            if perf.needs_review:
                flagged.append(perf)
        flagged.sort(key=lambda p: p.confidence)
        return flagged
