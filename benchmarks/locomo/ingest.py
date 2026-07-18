"""Ingest LoCoMo conversations into an ai-assist KnowledgeGraph.

Three-phase pipeline matching production:
  A) Store raw conversation turns as 'conversation' entities
  B) Run LLM synthesis to extract structured knowledge
  C) Run LLM connection discovery to link entities
"""

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

KNOWLEDGE_TYPES = {"user_preference", "lesson_learned", "project_context", "decision_rationale"}


def _strip_code_fence(text: str) -> str:
    """Remove markdown code fences from LLM response."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1 :]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


BATCH_SIZE = 15
BATCH_STRIDE = 10

LOCOMO_SYNTHESIS_PROMPT = """\
Extract ALL factual information from this conversation. Be thorough — every \
specific detail matters for later recall.

Extract:
- **User Preferences**: Likes, dislikes, favorites (food, music, activities, \
places, brands, styles)
- **Personal Facts**: Name, age, occupation, relationships, where they live, \
health conditions, pets
- **Events**: What happened, when (exact dates/times), where, with whom, outcomes
- **Plans & Decisions**: Future plans, travel plans, commitments, reasons for decisions
- **Opinions & Experiences**: Views expressed, experiences shared, recommendations

For each fact:
- Write a specific, detailed summary. Include names, dates, places, numbers.
- Use a descriptive unique key
- Assign confidence (0.0-1.0)
- Add relevant tags including any dates mentioned

Conversation:
{history_text}

Output valid JSON only (no markdown):
{{
  "insights": [
    {{
      "category": "user_preference|lesson_learned|project_context|decision_rationale",
      "key": "unique_identifier",
      "content": "Specific factual summary with names, dates, places",
      "confidence": 0.9,
      "tags": ["tag1", "tag2"]
    }}
  ]
}}

Map to categories: personal facts/events -> project_context, \
preferences -> user_preference, opinions/experiences -> lesson_learned, \
plans/decisions -> decision_rationale.

Extract as many facts as possible. If no facts, return {{"insights": []}}
"""


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


def _parse_datetime(dt_str: str) -> datetime:
    """Parse LoCoMo date_time strings like '1:56 pm on 8 May, 2023'."""
    dt_str = dt_str.strip()
    m = re.match(r"(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+),?\s+(\d{4})", dt_str, re.IGNORECASE)
    if m:
        hour, minute, ampm, day, month_name, year = m.groups()
        hour = int(hour)
        if ampm.lower() == "pm" and hour != 12:
            hour += 12
        elif ampm.lower() == "am" and hour == 12:
            hour = 0
        try:
            dt = datetime.strptime(f"{year} {month_name} {day} {hour}:{minute}", "%Y %B %d %H:%M")
            return dt
        except ValueError:
            pass
    for fmt in ("%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d %B, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    return datetime(2023, 1, 1)


def _store_conversations(kg, sample: dict, sample_idx: int) -> tuple[int, list[dict]]:
    """Phase A: Store raw conversation turns as 'conversation' entities.

    Returns (count, turns_for_synthesis) where turns_for_synthesis is a list
    of dicts with speaker, text, and datetime for batching into synthesis.
    """
    sample_id = sample.get("sample_id", f"conv-{sample_idx}")
    conv = sample.get("conversation", {})
    count = 0
    all_turns = []

    session_keys = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("_date_time") and isinstance(conv[k], list)],
        key=lambda k: int(k.split("_")[1]),
    )

    for session_key in session_keys:
        session_idx = int(session_key.split("_")[1])
        date_key = f"{session_key}_date_time"
        session_dt = _parse_datetime(conv.get(date_key, "1 January, 2023"))
        turns = conv[session_key]

        # Pair adjacent turns into user/assistant exchanges (like production)
        i = 0
        while i < len(turns):
            turn_a = turns[i]
            turn_b = turns[i + 1] if i + 1 < len(turns) else None

            user_text = turn_a.get("text", "").strip()
            assistant_text = turn_b.get("text", "").strip() if turn_b else ""

            if user_text:
                entity_id = f"locomo_{sample_id}_s{session_idx}_t{i}"
                kg.insert_entity(
                    entity_type="conversation",
                    data={
                        "user": f"{turn_a.get('speaker', 'A')}: {user_text}",
                        "assistant": f"{turn_b.get('speaker', 'B')}: {assistant_text}" if assistant_text else "",
                        "session": session_idx,
                    },
                    valid_from=session_dt,
                    entity_id=entity_id,
                )
                count += 1

            all_turns.append(
                {
                    "speaker": turn_a.get("speaker", "unknown"),
                    "text": user_text,
                    "session": session_idx,
                    "datetime": session_dt,
                }
            )
            if turn_b:
                all_turns.append(
                    {
                        "speaker": turn_b.get("speaker", "unknown"),
                        "text": assistant_text,
                        "session": session_idx,
                        "datetime": session_dt,
                    }
                )

            i += 2 if turn_b else 1

    return count, all_turns


def _run_synthesis(kg, client, model: str, turns: list[dict], sample_id: str) -> int:
    """Phase B: Run LLM synthesis on batches of turns to extract structured knowledge."""
    count = 0
    for batch_start in range(0, len(turns), BATCH_STRIDE):
        batch = turns[batch_start : batch_start + BATCH_SIZE]
        date_info = ""
        if batch and batch[0].get("datetime"):
            date_info = f" [Date: {batch[0]['datetime'].strftime('%B %d, %Y')}]"
        history_text = "\n".join(f"{t['speaker']}: {t['text']}" for t in batch if t["text"])
        if not history_text.strip():
            continue

        prompt = LOCOMO_SYNTHESIS_PROMPT.format(history_text=date_info + "\n" + history_text)

        try:
            response = client.messages.create(
                model=model,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = _strip_code_fence(response.content[0].text.strip())

            result = json.loads(response_text)
            insights = result.get("insights", [])

            for insight in insights:
                category = insight.get("category", "lesson_learned")
                if category not in KNOWLEDGE_TYPES:
                    category = "lesson_learned"
                key = insight.get("key", f"locomo_{sample_id}_{count}")
                content = insight.get("content", "")
                if not content:
                    continue

                batch_dt = batch[0]["datetime"] if batch else datetime(2023, 1, 1)
                kg.insert_knowledge(
                    entity_type=category,
                    key=key,
                    content=content,
                    metadata={
                        "tags": insight.get("tags", []),
                        "source": "locomo_synthesis",
                        "sample_id": sample_id,
                    },
                    confidence=insight.get("confidence", 1.0),
                    valid_from=batch_dt,
                )
                count += 1

            logger.debug(
                "Synthesis batch %d-%d: %d insights",
                batch_start,
                batch_start + len(batch),
                len(insights),
            )

        except Exception:
            logger.exception("Synthesis failed for batch %d-%d", batch_start, batch_start + len(batch))

    return count


def _run_connection_discovery(kg, client, model: str) -> int:
    """Phase C: Discover relationships between knowledge entities."""
    from ai_assist.agent import CONNECTION_DISCOVERY_PROMPT_TEMPLATE

    all_entities = []
    for et in KNOWLEDGE_TYPES:
        all_entities.extend(kg.search_knowledge(entity_type=et, limit=500))

    if not all_entities:
        logger.info("No knowledge entities for connection discovery")
        return 0

    entity_ids = {e.get("entity_id") for e in all_entities}
    total_count = 0

    # Process in batches of 40 to avoid prompt truncation
    batch_size = 40
    for batch_start in range(0, len(all_entities), batch_size):
        batch = all_entities[batch_start : batch_start + batch_size]

        entities_text = "\n".join(
            f"- ID: {e.get('entity_id', '?')} | Type: {e.get('entity_type', '?')} | "
            f"Key: {e.get('key', '?')}, Content: {str(e.get('content', ''))[:200]}"
            for e in batch
        )

        prompt = CONNECTION_DISCOVERY_PROMPT_TEMPLATE.format(
            entities_text=entities_text,
            reports_text="(no reports available — benchmark context)",
        )

        try:
            response = client.messages.create(
                model=model,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = _strip_code_fence(response.content[0].text.strip())

            result = json.loads(response_text)
            connections = result.get("connections", [])
            batch_count = 0

            for conn in connections:
                source_id = conn.get("source_id", "")
                target_id = conn.get("target_id", "")
                rel_type = conn.get("rel_type", "relates_to")

                if source_id not in entity_ids or target_id not in entity_ids:
                    continue
                if source_id == target_id:
                    continue
                if kg.relationship_exists(rel_type, source_id, target_id):
                    continue

                kg.insert_relationship(
                    rel_type=rel_type,
                    source_id=source_id,
                    target_id=target_id,
                    valid_from=datetime.now(),
                    properties={"description": conn.get("description", ""), "source": "locomo_benchmark"},
                )
                batch_count += 1

            total_count += batch_count
            logger.debug(
                "Connection discovery batch %d-%d: %d relationships", batch_start, batch_start + len(batch), batch_count
            )

        except Exception:
            logger.exception("Connection discovery batch %d-%d failed", batch_start, batch_start + len(batch))

    logger.info("Connection discovery: %d relationships created", total_count)
    return total_count


def ingest_conversation(kg, client, model: str, sample: dict, sample_idx: int) -> dict:
    """Ingest one LoCoMo conversation through the full pipeline.

    Returns dict with counts: conversations, insights, relationships.
    """
    sample_id = sample.get("sample_id", f"conv-{sample_idx}")

    n_conv, turns = _store_conversations(kg, sample, sample_idx)
    logger.info("Conversation %s: stored %d conversation entities", sample_id, n_conv)

    n_insights = _run_synthesis(kg, client, model, turns, sample_id)
    logger.info("Conversation %s: extracted %d knowledge insights", sample_id, n_insights)

    return {"conversations": n_conv, "insights": n_insights}


def ingest_dataset(
    kg,
    dataset_path: str | Path,
    model: str = "claude-sonnet-4-6",
    limit: int | None = None,
    skip_synthesis: bool = False,
) -> dict:
    """Ingest the full LoCoMo dataset into a KG.

    Returns dict with total counts.
    """
    path = Path(dataset_path)
    if not path.exists():
        logger.error("Dataset not found: %s", path)
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    samples = data if isinstance(data, list) else [data]
    if limit:
        samples = samples[:limit]

    client = _create_client() if not skip_synthesis else None
    totals = {"conversations": 0, "insights": 0, "relationships": 0}

    for idx, sample in enumerate(samples):
        if skip_synthesis:
            n_conv, _ = _store_conversations(kg, sample, idx)
            totals["conversations"] += n_conv
            logger.info("Conversation %d/%d: stored %d turns (synthesis skipped)", idx + 1, len(samples), n_conv)
        else:
            counts = ingest_conversation(kg, client, model, sample, idx)
            totals["conversations"] += counts["conversations"]
            totals["insights"] += counts["insights"]

    if client and not skip_synthesis:
        n_rels = _run_connection_discovery(kg, client, model)
        totals["relationships"] = n_rels

    kg.backfill_embeddings()
    logger.info(
        "Ingestion complete: %d conversations, %d insights, %d relationships",
        totals["conversations"],
        totals["insights"],
        totals["relationships"],
    )
    return totals
