"""Conversation synthesis and knowledge-graph connection discovery.

Extracted from ``agent.py``. ``SynthesisEngine`` holds an agent back-reference so
it can read collaborators (``knowledge_graph``, ``knowledge_tools``,
``report_tools``, ``anthropic``) live — the agent may reassign them after
construction — and reuse the agent's ``_model_for`` / ``_track_token_usage``
helpers and the ``_pending_synthesis`` flag.
"""

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from .output import console_print

if TYPE_CHECKING:
    from .context import ConversationMemory

logger = logging.getLogger(__name__)


SYNTHESIS_PROMPT_TEMPLATE = """Review this conversation and identify learnings to save.

Extract:
- **User Preferences**: Stated preferences about code style, workflows, tools
- **Lessons Learned**: Insights about bugs, patterns, best practices, gotchas
- **Project Context**: Background about projects, goals, teams, constraints
- **Decision Rationale**: Why certain implementation choices were made

For each learning:
- Write 1-2 sentence summary
- Suggest unique key (e.g., "python_test_framework", "dci_friday_failures")
- Assign confidence (0.0-1.0 based on how explicit/clear it was)
- Add relevant tags

Focus: {focus}

Conversation:
{history_text}

Output valid JSON only (no markdown):
{{
  "insights": [
    {{
      "category": "user_preference|lesson_learned|project_context|decision_rationale",
      "key": "unique_identifier",
      "content": "1-2 sentence summary",
      "confidence": 0.9,
      "tags": ["tag1", "tag2"]
    }}
  ]
}}

If no learnings, return {{"insights": []}}
"""

CONNECTION_DISCOVERY_PROMPT_TEMPLATE = """Analyze the following knowledge graph entities and reports to identify connections between them.

## Existing Entities

{entities_text}

## Recent Reports

{reports_text}

## Instructions

Identify meaningful relationships between the entities listed above. A relationship connects two entities that are related in a meaningful way. Focus on:
- Entities that reference the same project, tool, component, or concept
- Lessons learned that support or contradict each other
- Decisions that were influenced by specific project contexts
- Preferences that relate to specific project workflows
- Tool results that corroborate or conflict with insights

For each relationship, provide:
- source_id: The ID of the source entity (must be from the entities listed above)
- target_id: The ID of the target entity (must be from the entities listed above)
- rel_type: One of: relates_to, caused_by, references, contradicts, supports, part_of
- description: Brief explanation of why these entities are connected

Output valid JSON only (no markdown):
{{
  "connections": [
    {{
      "source_id": "entity_id_1",
      "target_id": "entity_id_2",
      "rel_type": "relates_to",
      "description": "Brief explanation"
    }}
  ]
}}

If no meaningful connections are found, return {{"connections": []}}
"""


class SynthesisEngine:
    """Reflect on conversations and discover connections between KG entities."""

    def __init__(self, agent):
        self._agent = agent

    def _resolve_dedup_key(self, entity_type: str, key: str, content: str) -> str:
        """Return the key to persist a fact under, reusing an existing one on a
        near-identical match so the write *supersedes* it instead of forking a
        new slug.

        Extraction LLMs invent a fresh slug for a re-stated fact, which creates
        a parallel live entity rather than letting the KG's last-write-wins
        upsert engage. This looks up the single best same-type match by semantic
        similarity; if it clears ``DEDUP_SIM_THRESHOLD`` the existing key is
        returned so ``insert_knowledge`` overwrites that record.

        The threshold is deliberately conservative: a false merge overwrites a
        distinct fact irrecoverably (knowledge upsert keeps no history), and
        measurement showed distinct-but-similar facts top out ~0.66 while true
        restatements sit ~0.80+. Below threshold we keep the new key (status quo).

        Args:
            entity_type: Knowledge entity type (only project_context is deduped).
            key: The slug the extractor proposed.
            content: The fact text, used as the similarity probe.

        Returns:
            The existing key to reuse, or the proposed key unchanged.
        """
        if not self._agent.knowledge_graph:
            return key
        try:
            matches = self._agent.knowledge_graph.semantic_search(
                content,
                limit=1,
                entity_types=[entity_type],
                min_score=self._agent.DEDUP_SIM_THRESHOLD,
                include_future=True,
            )
        except Exception:
            logger.exception("Dedup lookup failed for %s:%s", entity_type, key)
            return key
        if matches and matches[0].get("key") and matches[0]["key"] != key:
            existing = matches[0]["key"]
            logger.info(
                "Dedup: reusing key %s (score=%.3f) for extracted fact %s",
                existing,
                matches[0]["score"],
                key,
            )
            return existing
        return key

    def _save_insights(self, insights: list[dict], source: str, *, verbose: bool = False) -> int:
        """Persist extracted knowledge insights to the KG (upsert by key).

        Shared by conversation synthesis and compaction-time fact extraction.
        Each insight is a dict shaped like {category, key, content, confidence, tags}.
        Per-insight failures are logged and skipped so one bad item can't abort
        the rest. Safe to call from a worker thread (sqlite conn is thread-shared).

        Args:
            insights: List of insight dicts to save.
            source: Provenance tag stored in metadata (e.g. "auto_synthesis").
            verbose: When True, print a per-insight "Learned" line (interactive use).

        Returns:
            Number of insights successfully saved.
        """
        if not self._agent.knowledge_graph:
            return 0

        saved_count = 0
        for insight in insights:
            try:
                key = insight["key"]
                if insight["category"] == "project_context":
                    key = self._resolve_dedup_key("project_context", key, insight["content"])
                self._agent.knowledge_graph.insert_knowledge(
                    entity_type=insight["category"],
                    key=key,
                    content=insight["content"],
                    metadata={
                        "tags": insight.get("tags", []),
                        "source": source,
                        "synthesized_at": datetime.now().isoformat(),
                    },
                    confidence=insight.get("confidence", 1.0),
                )
                saved_count += 1
                if verbose:
                    console_print(f"💡 Learned: {insight['category']}:{key}")
            except Exception:
                logger.warning("Failed to save insight %s", insight.get("key"), exc_info=True)

        return saved_count

    async def _run_synthesis(self, conversation_memory: ConversationMemory, focus: str = "all"):
        """Agent reflects on conversation and extracts learnings

        Args:
            conversation_memory: Conversation to analyze
            focus: What to focus on (all, preferences, lessons, context)
        """
        if not self._agent.knowledge_graph or not self._agent.knowledge_tools:
            return

        history_parts = []
        for ex in conversation_memory.exchanges:
            history_parts.append(f"User: {ex['user']}")
            history_parts.append(f"Assistant: {ex['assistant']}")
        history_text = "\n\n".join(history_parts)

        synthesis_prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            focus=focus if focus != "all" else "everything",
            history_text=history_text,
        )

        try:
            with self._agent.anthropic.messages.stream(
                model=self._agent._model_for("synthesis"),
                max_tokens=2000,
                messages=[{"role": "user", "content": synthesis_prompt}],
            ) as stream:
                response = stream.get_final_message()
            self._agent._track_token_usage(response, turn=-1)

            first_block = response.content[0]
            response_text = first_block.text.strip() if hasattr(first_block, "text") else ""

            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            insights_data = json.loads(response_text)
            insights = insights_data.get("insights", [])

            if not insights:
                logger.info("Synthesis complete - no new learnings to save")
                return

            saved_count = self._save_insights(insights, "auto_synthesis", verbose=True)
            logger.info("Saved %d new learnings to knowledge base", saved_count)

        except json.JSONDecodeError:
            logger.exception("Synthesis failed - invalid JSON")
        except Exception as e:
            logger.exception("Synthesis failed: %s", e)

    async def _run_synthesis_from_kg(self, hours: int = 24) -> str:
        """Snapshot new/changed reports and discover connections between entities.

        Conversation fact-extraction happens at compaction, ``/clear`` and on
        exit in the interactive TUI, so this scheduled task focuses solely on
        reports and cross-entity connection discovery.

        Args:
            hours: Retained for API/back-compat; no longer used.

        Returns:
            Summary of synthesis results
        """
        if not self._agent.knowledge_graph or not self._agent.knowledge_tools:
            return "Knowledge graph not available"

        now = datetime.now()

        # Recall which reports the previous synthesis run processed
        markers = self._agent.knowledge_graph.query_as_of(now, entity_type="synthesis_marker", limit=1)
        previous_reports_processed: dict[str, str] = {}
        if markers:
            previous_reports_processed = markers[0].data.get("reports_processed", {})

        # Detect new/modified reports
        reports_processed = self._agent._get_report_snapshots()
        has_new_reports = reports_processed != previous_reports_processed

        # Early exit if no reports changed since last run
        if not has_new_reports:
            logger.info("No new reports to synthesize")
            return "No new reports to synthesize"

        # Discover connections between entities using the new reports as context
        try:
            connection_result = await self._agent._run_connection_discovery(previous_reports_processed)
        except Exception as e:
            connection_result = f"Connection discovery error: {e}"
            logger.exception("%s", connection_result)

        # Record synthesis marker so unchanged reports are skipped next run
        self._agent.knowledge_graph.insert_entity(
            entity_type="synthesis_marker",
            data={"reports_processed": reports_processed},
            valid_from=now,
        )

        return connection_result

    @staticmethod
    def _summarize_entities_for_prompt(entities: list) -> str:
        """Build compact entity summaries for the connection discovery prompt"""
        lines = []
        for entity in entities:
            data = entity.data
            if entity.entity_type == "tool_result":
                tool_name = data.get("tool_name", "unknown")
                args_summary = json.dumps(data.get("arguments", {}))[:100]
                summary = f"Tool: {tool_name}, Args: {args_summary}"
            else:
                key = data.get("key", "")
                content = data.get("content", "")[:200]
                summary = f"Key: {key}, Content: {content}"
            lines.append(f"- ID: {entity.id} | Type: {entity.entity_type} | {summary}")
        return "\n".join(lines)

    def _gather_recent_reports(self, already_processed: dict[str, str]) -> str:
        """Read reports that are new or modified since last processing.

        Args:
            already_processed: Dict mapping report name to its modification time
                when it was last processed. A report is skipped only if its current
                modification time matches the recorded one.

        Returns:
            Concatenated report content for new/modified reports
        """
        try:
            reports_json = self._agent.report_tools._list_reports()
            reports = json.loads(reports_json)
        except Exception:
            return ""

        parts = []
        for report in reports:
            name = report.get("name", "")
            modified = report.get("modified", "")
            fmt = report.get("format", "md")

            report_key = f"{name}.{fmt}"
            if report_key in already_processed and already_processed[report_key] == modified:
                continue

            content = self._agent.report_tools._read_report(name, fmt=fmt)
            if content and not content.startswith("Report '"):
                if len(content) > 5000:
                    content = content[:5000] + "\n... [truncated]"
                parts.append(f"### Report: {name} ({fmt})\n{content}")

        return "\n\n".join(parts)

    @staticmethod
    def _extract_connections_from_partial_json(text: str) -> dict:
        """Extract connection objects from truncated JSON output.

        When the LLM hits max_tokens, the JSON may be incomplete. This
        extracts all complete connection objects using regex.
        """
        import re

        pattern = re.compile(
            r'\{\s*"source_id"\s*:\s*"([^"]+)"\s*,'
            r'\s*"target_id"\s*:\s*"([^"]+)"\s*,'
            r'\s*"rel_type"\s*:\s*"([^"]+)"\s*,'
            r'\s*"description"\s*:\s*"([^"]*?)"\s*\}',
        )
        connections = [
            {
                "source_id": m.group(1),
                "target_id": m.group(2),
                "rel_type": m.group(3),
                "description": m.group(4),
            }
            for m in pattern.finditer(text)
        ]
        return {"connections": connections}

    def _get_report_snapshots(self) -> dict[str, str]:
        """Get name->modified mapping for all current reports.

        Used to record in the synthesis marker which reports (and at what
        modification time) have been processed.
        """
        try:
            reports_json = self._agent.report_tools._list_reports()
            reports = json.loads(reports_json)
            return {f"{r.get('name', '')}.{r.get('format', 'md')}": r.get("modified", "") for r in reports}
        except Exception:
            return {}

    async def _run_connection_discovery(self, previous_reports_processed: dict[str, str] | None = None) -> str:
        """Discover and create relationships between KG entities using reports as context

        Args:
            previous_reports_processed: Dict mapping report name to modification time
                from the previous synthesis run. Reports unchanged since then are skipped.

        Returns:
            Summary of connections created
        """
        if not self._agent.knowledge_graph:
            return "Knowledge graph not available"

        now = datetime.now()

        last_reports_processed = previous_reports_processed or {}

        entity_types_to_include = [
            "user_preference",
            "lesson_learned",
            "project_context",
            "decision_rationale",
            "tool_result",
        ]
        entities = []
        for etype in entity_types_to_include:
            entities.extend(self._agent.knowledge_graph.query_as_of(now, entity_type=etype, limit=50))

        if not entities:
            logger.info("No entities available for connection discovery")
            return "No entities for connection discovery"

        entities_text = self._summarize_entities_for_prompt(entities)
        reports_text = self._agent._gather_recent_reports(last_reports_processed)

        prompt = CONNECTION_DISCOVERY_PROMPT_TEMPLATE.format(
            entities_text=entities_text,
            reports_text=reports_text if reports_text else "No new reports.",
        )

        try:
            with self._agent.anthropic.messages.stream(
                model=self._agent._model_for("synthesis"),
                max_tokens=8000,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                response = stream.get_final_message()
            self._agent._track_token_usage(response, turn=-1)

            first_block = response.content[0]
            response_text = first_block.text.strip() if hasattr(first_block, "text") else ""

            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            # Handle truncated JSON from hitting max_tokens: extract
            # individual connection objects even if the array is incomplete
            try:
                connections_data = json.loads(response_text)
            except json.JSONDecodeError:
                connections_data = self._extract_connections_from_partial_json(response_text)

            connections = connections_data.get("connections", [])

            if not connections:
                logger.info("No new connections discovered")
                return "No new connections discovered"

            created_count = 0
            entity_ids = {e.id for e in entities}

            for conn in connections:
                source_id = conn.get("source_id", "")
                target_id = conn.get("target_id", "")
                rel_type = conn.get("rel_type", "relates_to")
                description = conn.get("description", "")

                if source_id not in entity_ids or target_id not in entity_ids:
                    continue

                if source_id == target_id:
                    continue

                if self._agent.knowledge_graph.relationship_exists(rel_type, source_id, target_id):
                    continue

                self._agent.knowledge_graph.insert_relationship(
                    rel_type=rel_type,
                    source_id=source_id,
                    target_id=target_id,
                    valid_from=now,
                    properties={"description": description, "source": "connection_discovery"},
                )
                created_count += 1
                logger.info("Connected: %s --[%s]--> %s", source_id, rel_type, target_id)

            summary = f"Created {created_count} new connections"
            logger.info("%s", summary)
            return summary

        except json.JSONDecodeError as e:
            logger.exception("Connection discovery failed - invalid JSON")
            return f"Connection discovery failed - invalid JSON: {e}"
        except Exception as e:
            logger.exception("Connection discovery failed: %s", e)
            return f"Connection discovery failed: {e}"

    async def check_and_run_synthesis(self, conversation_memory: ConversationMemory):
        """Check if synthesis was triggered and run it

        Args:
            conversation_memory: Current conversation
        """
        if self._agent._pending_synthesis:
            focus = self._agent._pending_synthesis.get("focus", "all")
            self._agent._pending_synthesis = None

            await self._run_synthesis(conversation_memory, focus)
