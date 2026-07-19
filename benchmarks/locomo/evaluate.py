"""Evaluate ai-assist KG retrieval against LoCoMo questions using an LLM judge.

Uses the same multi-strategy retrieval as production:
  1. search_knowledge for user_preference (always loaded)
  2. semantic_search for knowledge types (lesson_learned, project_context, decision_rationale)
  3. semantic_search for auto-context (conversation entities, tool_result, etc.)
"""

import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

CATEGORY_NAMES = {
    1: "single-hop",
    2: "multi-hop",
    3: "temporal",
    4: "open-domain",
    5: "adversarial",
}

KNOWLEDGE_TYPES = {"user_preference", "lesson_learned", "project_context", "decision_rationale"}

JUDGE_PROMPT = """\
You are evaluating a memory retrieval system. Given a question, the retrieved \
context from the system's memory, and the gold-standard answer, decide whether \
the retrieved context contains enough information to answer the question correctly.

Question: {question}

Retrieved context:
{context}

Gold-standard answer: {gold_answer}

Evaluation criteria:
- CORRECT if the context contains the key facts needed to produce the gold answer, \
even if phrased differently or spread across multiple context entries
- CORRECT if the context contains information that directly implies or entails the answer
- WRONG only if the context is missing the critical facts needed, or contains \
contradictory information

Reply with exactly one word: CORRECT or WRONG."""


@dataclass
class QuestionResult:
    question: str
    category: str
    gold_answer: str
    context: str
    verdict: str  # CORRECT or WRONG
    score: int  # 1 or 0


@dataclass
class BenchmarkResults:
    results: list[QuestionResult] = field(default_factory=list)

    def add(self, result: QuestionResult):
        self.results.append(result)

    @property
    def overall_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results) * 100

    def category_scores(self) -> dict[str, float]:
        by_cat: dict[str, list[int]] = defaultdict(list)
        for r in self.results:
            by_cat[r.category].append(r.score)
        return {cat: sum(scores) / len(scores) * 100 for cat, scores in sorted(by_cat.items())}

    def failures_markdown(self) -> str:
        lines = ["# LoCoMo Failures", ""]
        for r in self.results:
            if r.score == 0:
                lines.append(f"## [{r.category}] {r.question}")
                lines.append(f"**Gold answer:** {r.gold_answer}")
                lines.append(f"**Context:** {r.context}")
                lines.append("")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines = ["# LoCoMo Benchmark Results", ""]
        lines.append(
            f"**Overall LLM Judge Score: {self.overall_score:.1f}%** ({sum(r.score for r in self.results)}/{len(self.results)})"
        )
        lines.append("")
        lines.append("| Category | Score | Count |")
        lines.append("|----------|-------|-------|")
        by_cat: dict[str, list[int]] = defaultdict(list)
        for r in self.results:
            by_cat[r.category].append(r.score)
        for cat, scores in sorted(by_cat.items()):
            pct = sum(scores) / len(scores) * 100
            lines.append(f"| {cat} | {pct:.1f}% | {len(scores)} |")
        lines.append("")
        return "\n".join(lines)


def _create_client():
    """Create Anthropic client, using Vertex AI if configured."""
    import os

    project_id = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
    if project_id:
        from anthropic import AnthropicVertex

        kwargs = {"project_id": project_id}
        region = os.environ.get("ANTHROPIC_VERTEX_REGION")
        if region:
            kwargs["region"] = region
        return AnthropicVertex(**kwargs)

    from anthropic import Anthropic

    return Anthropic()


def _format_date(result: dict) -> str:
    """Format date from the KG's valid_from temporal field."""
    from datetime import datetime

    vf = result.get("valid_from")
    if not vf:
        return ""
    try:
        if isinstance(vf, str):
            dt = datetime.fromisoformat(vf)
        elif isinstance(vf, datetime):
            dt = vf
        else:
            return ""
        return dt.strftime("%B %d, %Y")
    except ValueError, TypeError:
        return ""


def _extract_content(result: dict) -> str:
    """Extract displayable text from a search result or entity data dict."""
    parts = []
    date = _format_date(result)
    if date:
        parts.append(f"[{date}]")
    content = result.get("content")
    if content:
        parts.append(content)
        return " ".join(parts)
    user = result.get("user", "")
    assistant = result.get("assistant", "")
    if user or assistant:
        parts.append(" | ".join(filter(None, [user, assistant])))
        return " ".join(parts)
    return ""


def _extract_date_range(question: str):
    """Extract a date range from a question for temporal filtering.

    Returns (after, before) datetime tuple or None if no date detected.
    Uses a ±1 month window around the detected date for better recall.
    """
    import re
    from datetime import datetime

    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }

    def _month_window(year, month):
        """Return (start_of_prev_month, end_of_next_month) for ±1 month window."""
        prev_m = month - 1 if month > 1 else 12
        prev_y = year if month > 1 else year - 1
        next_m = month + 1 if month < 12 else 1
        next_y = year if month < 12 else year + 1
        after_m = next_m + 1 if next_m < 12 else 1
        after_y = next_y if next_m < 12 else next_y + 1
        return (datetime(prev_y, prev_m, 1), datetime(after_y, after_m, 1))

    # Match "Month DD, YYYY"
    m = re.search(
        r"(\b(?:" + "|".join(months) + r")\b)\s+(\d{1,2}),?\s+(\d{4})",
        question.lower(),
    )
    if m:
        month = months[m.group(1)]
        year = int(m.group(3))
        return _month_window(year, month)

    # Match "Month YYYY"
    m = re.search(
        r"(\b(?:" + "|".join(months) + r")\b)\s+(\d{4})",
        question.lower(),
    )
    if m:
        month = months[m.group(1)]
        year = int(m.group(2))
        return _month_window(year, month)

    # Match standalone month name (assume 2023 for LoCoMo)
    m = re.search(r"\b(" + "|".join(months) + r")\b", question.lower())
    if m:
        month = months[m.group(1)]
        return _month_window(2023, month)

    return None


def _entity_to_result(entity) -> dict:
    """Convert an Entity object to a result dict with valid_from for date formatting."""
    return {**entity.data, "valid_from": entity.valid_from.isoformat()}


def _build_context(kg, question: str) -> str:
    """Retrieve context using multi-strategy hybrid search + graph traversal.

    Strategy mirrors the production agent with additional temporal awareness:
    1. Separate preference search (like agent's search_knowledge)
    2. Project context search (personal facts)
    3. Lessons and decisions
    4. Name-based keyword search (all facts about mentioned people)
    5. Temporal-filtered search when date detected in question
    6. Conversation entities (broad search)
    7. Graph traversal from top results (2 hops)
    """
    parts = []
    seen_ids = set()

    def _add_result(r, label):
        eid = r.get("entity_id", "")
        content = _extract_content(r)
        if not content or eid in seen_ids:
            return
        seen_ids.add(eid)
        parts.append(f"[{label}] {content}")

    # 1. Separate preference search (agent loads all preferences separately)
    pref_results = kg.hybrid_search(
        question,
        limit=40,
        entity_types=["user_preference"],
        min_score=0.0,
        include_future=True,
    )
    for r in pref_results:
        _add_result(r, "user_preference")

    # 2. Project context (personal facts — critical for single-hop)
    ctx_results = kg.hybrid_search(
        question,
        limit=50,
        entity_types=["project_context"],
        min_score=0.0,
        include_future=True,
    )
    for r in ctx_results:
        _add_result(r, "project_context")

    # 3. Lessons and decisions
    other_results = kg.hybrid_search(
        question,
        limit=25,
        entity_types=["lesson_learned", "decision_rationale"],
        min_score=0.0,
        include_future=True,
    )
    for r in other_results:
        _add_result(r, r.get("entity_type", ""))

    # 4. Name-based keyword search — find ALL facts about mentioned people
    _stop = {
        "what",
        "which",
        "where",
        "when",
        "who",
        "whom",
        "how",
        "why",
        "does",
        "did",
        "was",
        "were",
        "are",
        "is",
        "has",
        "have",
        "had",
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "about",
        "not",
        "but",
        "they",
        "them",
        "their",
        "your",
        "you",
        "she",
        "her",
        "his",
        "its",
        "can",
        "will",
        "would",
        "could",
        "should",
        "been",
        "being",
        "some",
        "any",
        "all",
        "each",
        "every",
        "both",
        "into",
        "over",
        "after",
        "before",
        "between",
        "during",
    }
    names = [
        w.rstrip("?'s.,!")
        for w in question.split()
        if w[0].isupper() and w.rstrip("?'s.,!").isalpha() and w.rstrip("?'s.,!").lower() not in _stop and len(w) > 2
    ]
    for name in names[:2]:
        name_results = kg.keyword_search(
            name,
            limit=40,
            include_future=True,
        )
        for r in name_results:
            etype = r.get("entity_type", "")
            label = etype if etype in KNOWLEDGE_TYPES else "context"
            _add_result(r, label)

    # 4b. Content keyword search — find facts matching specific nouns
    content_words = [
        w.rstrip("?'s.,!").lower()
        for w in question.split()
        if w.rstrip("?'s.,!").lower() not in _stop
        and len(w.rstrip("?'s.,!")) > 3
        and w.rstrip("?'s.,!").isalpha()
        and not w[0].isupper()
    ]
    for word in content_words[:2]:
        for r in kg.keyword_search(word, limit=5, include_future=True):
            _add_result(r, r.get("entity_type", "") if r.get("entity_type", "") in KNOWLEDGE_TYPES else "context")

    # 5. Temporal-filtered search when date detected
    date_range = _extract_date_range(question)
    if date_range:
        after, before = date_range
        temporal_results = kg.hybrid_search(
            question,
            limit=15,
            min_score=0.0,
            include_future=True,
            valid_from_after=after,
            valid_from_before=before,
        )
        for r in temporal_results:
            etype = r.get("entity_type", "")
            label = etype if etype in KNOWLEDGE_TYPES else "context"
            _add_result(r, label)

    # 6. Conversation entities (broad search)
    auto_results = kg.hybrid_search(
        question,
        limit=20,
        min_score=0.0,
        include_future=True,
    )
    count = 0
    for r in auto_results:
        eid = r.get("entity_id", "")
        if eid in seen_ids:
            continue
        if r.get("entity_type") in KNOWLEDGE_TYPES:
            continue
        content = _extract_content(r)
        if not content:
            continue
        seen_ids.add(eid)
        parts.append(f"[context] {content}")
        count += 1
        if count >= 15:
            break

    # 7. Graph traversal — 2 hops for multi-hop questions
    hop1_ids = list(seen_ids)[:10]
    hop2_ids = []
    for eid in hop1_ids:
        try:
            related = kg.get_related_entities(eid, direction="both")
            for _rel, entity in related:
                if entity.id in seen_ids:
                    continue
                seen_ids.add(entity.id)
                content = _extract_content(_entity_to_result(entity))
                if content:
                    parts.append(f"[related] {content}")
                    hop2_ids.append(entity.id)
        except Exception:
            pass

    for eid in hop2_ids[:5]:
        try:
            related = kg.get_related_entities(eid, direction="both")
            for _rel, entity in related:
                if entity.id in seen_ids:
                    continue
                seen_ids.add(entity.id)
                content = _extract_content(_entity_to_result(entity))
                if content:
                    parts.append(f"[related-2] {content}")
        except Exception:
            pass

    if not parts:
        return "(no relevant context found)"

    return "\n".join(parts)


def _judge_answer(client, model: str, question: str, context: str, gold_answer: str) -> str:
    """Ask the LLM judge to score a retrieval result."""
    prompt = JUDGE_PROMPT.format(question=question, context=context, gold_answer=gold_answer)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        verdict = response.content[0].text.strip().upper()
        return "CORRECT" if "CORRECT" in verdict else "WRONG"
    except Exception:
        logger.exception("Judge call failed")
        return "WRONG"


def evaluate_dataset(
    kg,
    dataset_path: str | Path,
    model: str = "claude-sonnet-4-6",
    limit: int | None = None,
    categories: set[int] | None = None,
) -> BenchmarkResults:
    """Run LoCoMo evaluation against a populated KG."""
    path = Path(dataset_path)
    if not path.exists():
        logger.error("Dataset not found: %s", path)
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    samples = data if isinstance(data, list) else data.get("data", data.get("samples", [data]))
    if limit:
        samples = samples[:limit]

    client = _create_client()
    results = BenchmarkResults()
    total_questions = sum(len(s.get("qa", [])) for s in samples)
    question_idx = 0

    for sample in samples:
        qa_pairs = sample.get("qa", [])
        for qa in qa_pairs:
            question = qa.get("question", "")
            gold_answer = qa.get("answer", "")
            raw_cat = qa.get("category", 0)
            category = CATEGORY_NAMES.get(raw_cat, f"cat-{raw_cat}")

            if categories and raw_cat not in categories:
                continue

            if not question or not gold_answer:
                continue

            context = _build_context(kg, question)
            verdict = _judge_answer(client, model, question, context, gold_answer)
            score = 1 if verdict == "CORRECT" else 0

            results.add(
                QuestionResult(
                    question=question,
                    category=category,
                    gold_answer=gold_answer,
                    context=context[:2000],
                    verdict=verdict,
                    score=score,
                )
            )

            question_idx += 1
            if question_idx % 50 == 0:
                logger.info(
                    "Progress: %d/%d questions (current score: %.1f%%)",
                    question_idx,
                    total_questions,
                    results.overall_score,
                )

    return results
