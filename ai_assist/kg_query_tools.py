"""Agent tools for querying the knowledge graph"""

import json
from datetime import datetime
from typing import Any

from .kg_queries import KnowledgeGraphQueries
from .knowledge_graph import KnowledgeGraph


class KGQueryTools:
    """Tools exposing KG query capabilities to the agent"""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.queries = KnowledgeGraphQueries(kg)

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get MCP-style tool definitions for KG query tools"""
        return [
            {
                "name": "internal__kg_recent_changes",
                "description": (
                    "AGENT-ONLY: Get recent changes in the knowledge graph. "
                    "Shows new entities discovered and corrected beliefs within a time window."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "hours": {
                            "type": "integer",
                            "description": "Number of hours to look back (default: 1)",
                            "default": 1,
                            "minimum": 1,
                        },
                        "entity_type": {
                            "type": "string",
                            "description": "Optional filter by entity type",
                        },
                    },
                    "required": [],
                },
                "_server": "internal",
                "_original_name": "kg_recent_changes",
            },
            {
                "name": "internal__kg_late_discoveries",
                "description": (
                    "AGENT-ONLY: Find entities discovered significantly after they became valid. "
                    "Identifies monitoring lag — entities that changed but weren't noticed promptly."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "min_delay_minutes": {
                            "type": "integer",
                            "description": "Minimum delay in minutes to consider 'late' (default: 30)",
                            "default": 30,
                            "minimum": 1,
                        },
                        "entity_type": {
                            "type": "string",
                            "description": "Optional filter by entity type",
                        },
                    },
                    "required": [],
                },
                "_server": "internal",
                "_original_name": "kg_late_discoveries",
            },
            {
                "name": "internal__kg_discovery_lag_stats",
                "description": (
                    "AGENT-ONLY: Analyze discovery lag statistics for an entity type. "
                    "Returns avg/min/max/percentile lag times."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "entity_type": {
                            "type": "string",
                            "description": "Entity type to analyze",
                        },
                        "days": {
                            "type": "integer",
                            "description": "Number of days to look back (default: 7)",
                            "default": 7,
                            "minimum": 1,
                        },
                    },
                    "required": ["entity_type"],
                },
                "_server": "internal",
                "_original_name": "kg_discovery_lag_stats",
            },
            {
                "name": "internal__kg_entity_context",
                "description": (
                    "AGENT-ONLY: Get an entity with all related entities, grouped by type. "
                    "Provides full context for investigating any entity in the knowledge graph."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "The entity ID in the knowledge graph",
                        },
                    },
                    "required": ["entity_id"],
                },
                "_server": "internal",
                "_original_name": "kg_entity_context",
            },
            {
                "name": "internal__kg_stats",
                "description": (
                    "AGENT-ONLY: Get knowledge graph statistics — entity counts by type, "
                    "relationship counts, and database info."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                "_server": "internal",
                "_original_name": "kg_stats",
            },
            {
                "name": "internal__kg_snapshot",
                "description": (
                    "AGENT-ONLY: Query the knowledge graph at a point in time. "
                    "Mode 'known_at': what the agent believed at that moment (transaction time). "
                    "Mode 'valid_at': what was true in reality at that moment (valid time)."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "time": {
                            "type": "string",
                            "description": "ISO datetime for the snapshot (e.g. '2026-06-28T15:00:00')",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["known_at", "valid_at"],
                            "description": "known_at: what the agent believed. valid_at: what was true in reality.",
                            "default": "known_at",
                        },
                        "entity_type": {
                            "type": "string",
                            "description": "Optional filter by entity type",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default: 50)",
                            "default": 50,
                            "minimum": 1,
                            "maximum": 200,
                        },
                    },
                    "required": ["time"],
                },
                "_server": "internal",
                "_original_name": "kg_snapshot",
            },
            {
                "name": "internal__kg_knowledge_health",
                "description": (
                    "AGENT-ONLY: Report knowledge graph health metrics. "
                    "Shows most-accessed entries, never-accessed entries, "
                    "and stale entries not accessed recently. "
                    "Helps identify valuable knowledge and what can be pruned."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "top_n": {
                            "type": "integer",
                            "description": "Number of top-accessed entries to show (default: 10)",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50,
                        },
                        "stale_days": {
                            "type": "integer",
                            "description": "Days without access to consider stale (default: 7)",
                            "default": 7,
                            "minimum": 1,
                        },
                    },
                    "required": [],
                },
                "_server": "internal",
                "_original_name": "kg_knowledge_health",
            },
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a KG query tool and return JSON result"""
        if tool_name == "kg_recent_changes":
            return self._recent_changes(arguments)
        elif tool_name == "kg_late_discoveries":
            return self._late_discoveries(arguments)
        elif tool_name == "kg_discovery_lag_stats":
            return self._discovery_lag_stats(arguments)
        elif tool_name == "kg_entity_context":
            return self._entity_context(arguments)
        elif tool_name == "kg_stats":
            return self._stats()
        elif tool_name == "kg_snapshot":
            return self._snapshot(arguments)
        elif tool_name == "kg_knowledge_health":
            return self._knowledge_health(arguments)
        else:
            raise ValueError(f"Unknown KG query tool: {tool_name}")

    def _recent_changes(self, arguments: dict[str, Any]) -> str:
        hours = arguments.get("hours", 1)
        entity_type = arguments.get("entity_type")
        result = self.queries.what_changed_recently(hours=hours, entity_type=entity_type)
        return json.dumps(result, indent=2, default=str)

    def _late_discoveries(self, arguments: dict[str, Any]) -> str:
        min_delay = arguments.get("min_delay_minutes", 30)
        entity_type = arguments.get("entity_type")
        discoveries = self.queries.find_late_discoveries(min_delay_minutes=min_delay, entity_type=entity_type)
        return json.dumps(
            {"count": len(discoveries), "discoveries": discoveries},
            indent=2,
            default=str,
        )

    def _discovery_lag_stats(self, arguments: dict[str, Any]) -> str:
        entity_type = arguments["entity_type"]
        days = arguments.get("days", 7)
        result = self.queries.analyze_discovery_lag(entity_type=entity_type, days=days)
        return json.dumps(result, indent=2, default=str)

    def _entity_context(self, arguments: dict[str, Any]) -> str:
        entity_id = arguments["entity_id"]
        result = self.queries.get_entity_with_context(entity_id)
        if result is None:
            return json.dumps({"error": "not_found", "entity_id": entity_id})
        return json.dumps(result, indent=2, default=str)

    def _stats(self) -> str:
        result = self.kg.get_stats()
        return json.dumps(result, indent=2, default=str)

    def _snapshot(self, arguments: dict[str, Any]) -> str:
        time = datetime.fromisoformat(arguments["time"])
        mode = arguments.get("mode", "known_at")
        entity_type = arguments.get("entity_type")
        limit = arguments.get("limit", 50)

        if mode == "known_at":
            entities = self.queries.what_did_we_know_at(time, entity_type=entity_type)
            return json.dumps(
                {
                    "mode": "known_at",
                    "time": time.isoformat(),
                    "count": len(entities[:limit]),
                    "entities": entities[:limit],
                },
                indent=2,
                default=str,
            )
        else:
            raw_entities = self.kg.query_valid_at(time, entity_type=entity_type, limit=limit)
            entities = [
                {
                    "id": e.id,
                    "type": e.entity_type,
                    "data": e.data,
                    "valid_from": e.valid_from.isoformat(),
                    "valid_to": e.valid_to.isoformat() if e.valid_to else None,
                    "known_since": e.tx_from.isoformat(),
                }
                for e in raw_entities
            ]
            return json.dumps(
                {"mode": "valid_at", "time": time.isoformat(), "count": len(entities), "entities": entities},
                indent=2,
                default=str,
            )

    def _knowledge_health(self, arguments: dict[str, Any]) -> str:
        top_n = arguments.get("top_n", 10)
        stale_days = arguments.get("stale_days", 7)
        result = self.kg.get_access_stats(top_n=top_n, stale_days=stale_days)
        return json.dumps(result, indent=2, default=str)
