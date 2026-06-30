"""Internal tools for agent knowledge management"""

import json
from datetime import datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .agent import AiAssistAgent
    from .knowledge_graph import KnowledgeGraph


class KnowledgeTools:
    """Tools for agent to manage its own knowledge"""

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.agent: AiAssistAgent | None = None

    def get_tool_definitions(self) -> list[dict]:
        """Get MCP-style tool definitions for knowledge management"""
        return [
            {
                "name": "internal__save_knowledge",
                "description": """AGENT-ONLY: Directly save a specific piece of knowledge.

Use this for immediate, explicit saves (not synthesis).
For broader reflection, use internal__trigger_synthesis instead.

SAVE when you notice:
- User stated a preference about workflows/tools/style
- User corrected you (save the correction as a lesson)
- You learned a pattern or best practice from discussion
- User provided project context, goals, or team info
- A decision was made with clear rationale

DO NOT SAVE:
- Transient task details (current debugging steps, temporary state)
- Information already in the codebase (code patterns, file structure)
- Raw data from tool results (these are auto-captured separately)
- Anything you're not confident about (use confidence < 0.5 if uncertain)

PREFER updating existing entries over creating duplicates — use the same key
to upsert. Check with internal__search_knowledge first if unsure.

Args:
    entity_type: Type of knowledge (user_preference, lesson_learned, project_context, decision_rationale)
    key: Unique identifier (e.g., "python_test_framework")
    content: The knowledge (1-2 sentences)
    tags: Optional categorization tags (default: [])
    confidence: How confident 0.0-1.0 (default: 1.0)

Returns:
    Confirmation with entity ID
""",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "entity_type": {
                            "type": "string",
                            "enum": [
                                "user_preference",
                                "lesson_learned",
                                "project_context",
                                "decision_rationale",
                            ],
                            "description": "Type of knowledge being stored",
                        },
                        "key": {
                            "type": "string",
                            "description": "Unique identifier for this knowledge",
                        },
                        "content": {
                            "type": "string",
                            "description": "The actual knowledge (1-2 sentences)",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags for categorization",
                            "default": [],
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence level (0.0-1.0)",
                            "default": 1.0,
                        },
                        "valid_from": {
                            "type": "string",
                            "description": "ISO datetime when this becomes true (e.g. '2026-07-10'). Defaults to now. Use a future date for planned events.",
                        },
                    },
                    "required": ["entity_type", "key", "content"],
                },
                "_server": "internal",
                "_original_name": "save_knowledge",
            },
            {
                "name": "internal__search_knowledge",
                "description": """Search stored knowledge before making decisions.

Use this to check:
- "Do I know user's preference for X?"
- "Have I learned about Y before?"
- "What's the context for project Z?"

Args:
    entity_type: Filter by type or "all" (default: "all")
    semantic_query: Natural language search (finds related concepts, not just exact matches)
    query: Search in keys (SQL LIKE pattern, e.g., "%test%"). Use semantic_query instead when possible.
    tags: Must have these tags
    limit: Max results (default: 10)

Returns:
    JSON list of matching knowledge entries

Example:
    # Before suggesting a test framework:
    search_knowledge(entity_type="user_preference", semantic_query="testing framework preference")
""",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "entity_type": {
                            "type": "string",
                            "enum": [
                                "user_preference",
                                "lesson_learned",
                                "project_context",
                                "decision_rationale",
                                "all",
                            ],
                            "description": "Filter by entity type",
                            "default": "all",
                        },
                        "semantic_query": {
                            "type": "string",
                            "description": "Natural language search text (finds related concepts)",
                        },
                        "query": {
                            "type": "string",
                            "description": "Search pattern for keys (SQL LIKE, e.g., '%python%')",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by tags (must have all)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results to return",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 100,
                        },
                        "since": {
                            "type": "string",
                            "description": "ISO datetime (e.g. '2026-06-27T00:00:00'). Only return knowledge learned after this time.",
                        },
                        "include_future": {
                            "type": "boolean",
                            "description": "Include knowledge with future valid_from dates (default: false). Use to find planned events.",
                            "default": False,
                        },
                    },
                    "required": [],
                },
                "_server": "internal",
                "_original_name": "search_knowledge",
            },
            {
                "name": "internal__trigger_synthesis",
                "description": """AGENT-ONLY: Trigger synthesis of conversation learnings.

Call this when you notice:
- User stated a preference about workflows/tools/style
- You learned a pattern or best practice from discussion
- User provided project context or goals
- A decision was made with clear rationale

The synthesis happens in background after your response.
User won't be interrupted.

Args:
    focus: What to synthesize (default: "all")
        - "all": everything
        - "preferences": only user preferences
        - "lessons": only lessons learned
        - "context": only project context

Returns:
    Confirmation that synthesis will run

Example:
    If user says "I prefer pytest", call this before responding.
""",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "focus": {
                            "type": "string",
                            "enum": ["all", "preferences", "lessons", "context"],
                            "description": "What to synthesize",
                            "default": "all",
                        }
                    },
                    "required": [],
                },
                "_server": "internal",
                "_original_name": "trigger_synthesis",
            },
            {
                "name": "internal__run_kg_synthesis",
                "description": """Run knowledge graph synthesis now.

Processes recent conversations stored in the knowledge graph and extracts
learnings (preferences, lessons, context). Also discovers connections
between entities. This is the same synthesis that runs as a nightly task.

Use this when the user explicitly asks to run KG synthesis manually.

Args:
    hours: How many hours back to look for conversations (default: 24)

Returns:
    Summary of synthesis results
""",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "hours": {
                            "type": "integer",
                            "description": "How many hours back to look for conversations",
                            "default": 24,
                        }
                    },
                    "required": [],
                },
                "_server": "internal",
                "_original_name": "run_kg_synthesis",
            },
            {
                "name": "internal__expire_knowledge",
                "description": (
                    "AGENT-ONLY: Mark a knowledge entry as no longer valid or retract an incorrect belief. "
                    "Use 'no_longer_valid' when the fact stopped being true (e.g. user changed preference). "
                    "Use 'retract' when the knowledge was wrong (e.g. incorrect assumption)."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "Entity ID to expire (e.g. 'user_preference:python_test_framework')",
                        },
                        "reason": {
                            "type": "string",
                            "enum": ["no_longer_valid", "retract"],
                            "description": "no_longer_valid: fact stopped being true. retract: belief was incorrect.",
                        },
                    },
                    "required": ["entity_id", "reason"],
                },
                "_server": "internal",
                "_original_name": "expire_knowledge",
            },
        ]

    async def execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a knowledge management tool"""
        if tool_name == "save_knowledge":
            return await self.save_knowledge(
                entity_type=arguments["entity_type"],
                key=arguments["key"],
                content=arguments["content"],
                tags=arguments.get("tags", []),
                confidence=arguments.get("confidence", 1.0),
                valid_from=arguments.get("valid_from"),
            )
        elif tool_name == "search_knowledge":
            return await self.search_knowledge(
                entity_type=arguments.get("entity_type", "all"),
                semantic_query=arguments.get("semantic_query"),
                query=arguments.get("query"),
                tags=arguments.get("tags"),
                limit=arguments.get("limit", 10),
                since=arguments.get("since"),
                include_future=arguments.get("include_future", False),
            )
        elif tool_name == "trigger_synthesis":
            return await self.trigger_synthesis(focus=arguments.get("focus", "all"))
        elif tool_name == "run_kg_synthesis":
            return await self.run_kg_synthesis(hours=arguments.get("hours", 24))
        elif tool_name == "expire_knowledge":
            return await self.expire_knowledge(
                entity_id=arguments["entity_id"],
                reason=arguments["reason"],
            )
        else:
            raise ValueError(f"Unknown knowledge tool: {tool_name}")

    async def save_knowledge(
        self,
        entity_type: Literal["user_preference", "lesson_learned", "project_context", "decision_rationale"],
        key: str,
        content: str,
        tags: list[str] | None = None,
        confidence: float = 1.0,
        valid_from: str | None = None,
    ) -> str:
        """Save a piece of knowledge to the graph"""
        valid_from_dt = datetime.fromisoformat(valid_from) if valid_from else None

        metadata = {
            "tags": tags or [],
            "source": "agent_direct_save",
            "saved_at": datetime.now().isoformat(),
        }

        entity_id = self.kg.insert_knowledge(
            entity_type=entity_type,
            key=key,
            content=content,
            metadata=metadata,
            confidence=confidence,
            valid_from=valid_from_dt,
        )

        suffix = f" (valid from {valid_from})" if valid_from else ""
        return f"✓ Saved {entity_type}: {key} (ID: {entity_id}){suffix}"

    async def search_knowledge(
        self,
        entity_type: Literal[
            "user_preference",
            "lesson_learned",
            "project_context",
            "decision_rationale",
            "all",
        ] = "all",
        semantic_query: str | None = None,
        query: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
        since: str | None = None,
        include_future: bool = False,
    ) -> str:
        """Search stored knowledge"""
        type_filter = None if entity_type == "all" else entity_type
        since_dt = datetime.fromisoformat(since) if since else None

        if semantic_query:
            entity_types: list[str] | None = [type_filter] if type_filter else None
            fetch_limit = limit * 5 if since_dt else limit
            results = self.kg.semantic_search(
                semantic_query, limit=fetch_limit, entity_types=entity_types, include_future=include_future
            )
            if since_dt:
                since_iso = since_dt.isoformat()
                results = [r for r in results if r.get("learned_at") and r["learned_at"] >= since_iso][:limit]
        else:
            results = self.kg.search_knowledge(
                entity_type=type_filter,
                key_pattern=query,
                tags=tags,
                since=since_dt,
                limit=limit,
                include_future=include_future,
            )

        if not results:
            return json.dumps({"results": [], "count": 0})

        self.kg.record_access([r["entity_id"] for r in results], "agent_search")

        return json.dumps(
            {
                "results": [
                    {
                        "type": r["entity_type"],
                        "key": r["key"],
                        "content": r["content"],
                        "tags": r["metadata"].get("tags", []),
                        "confidence": r["metadata"].get("confidence", 1.0),
                        "learned_at": r["learned_at"],
                    }
                    for r in results
                ],
                "count": len(results),
            },
            indent=2,
        )

    async def trigger_synthesis(self, focus: Literal["all", "preferences", "lessons", "context"] = "all") -> str:
        """Trigger synthesis of conversation learnings"""
        if self.agent is None:
            return "Error: Agent not set on KnowledgeTools"

        self.agent._pending_synthesis = {"focus": focus, "triggered_at": datetime.now()}

        return f"✓ Synthesis scheduled (focus: {focus}). Will run after this response completes."

    async def run_kg_synthesis(self, hours: int = 24) -> str:
        """Run KG synthesis immediately, processing conversations stored in the knowledge graph"""
        if self.agent is None:
            return "Error: Agent not set on KnowledgeTools"

        return await self.agent._run_synthesis_from_kg(hours=hours)

    async def expire_knowledge(
        self,
        entity_id: str,
        reason: Literal["no_longer_valid", "retract"],
    ) -> str:
        """Mark a knowledge entry as expired or retracted"""
        now = datetime.now()
        if reason == "no_longer_valid":
            updated = self.kg.update_entity(entity_id, valid_to=now)
        else:
            updated = self.kg.update_entity(entity_id, tx_to=now)

        if updated is None:
            return json.dumps({"error": "not_found", "entity_id": entity_id})

        return f"Expired {updated.entity_type}: {entity_id} (reason: {reason})"
