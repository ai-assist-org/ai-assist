"""Bi-temporal knowledge graph for tracking facts and discoveries over time"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal
from uuid import uuid4

from .config import get_config_dir


@dataclass
class Entity:
    """An entity with bi-temporal tracking

    Attributes:
        id: Unique identifier for the entity
        entity_type: Type of entity (e.g., 'dci_job', 'jira_ticket', 'component')
        valid_from: When the fact became true in reality
        valid_to: When the fact stopped being true (None if still valid)
        tx_from: When ai-assist learned about the fact
        tx_to: When ai-assist stopped believing the fact (None if current belief)
        data: Flexible JSON data for the entity
    """

    id: str
    entity_type: str
    valid_from: datetime
    valid_to: datetime | None
    tx_from: datetime
    tx_to: datetime | None
    data: dict[str, Any]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "entity_type": self.entity_type,
            "valid_from": self.valid_from.isoformat(),
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "tx_from": self.tx_from.isoformat(),
            "tx_to": self.tx_to.isoformat() if self.tx_to else None,
            "data": self.data,
        }

    @classmethod
    def from_row(cls, row: tuple) -> Entity:
        """Create Entity from database row"""
        return cls(
            id=row[0],
            entity_type=row[1],
            valid_from=datetime.fromisoformat(row[2]),
            valid_to=datetime.fromisoformat(row[3]) if row[3] else None,
            tx_from=datetime.fromisoformat(row[4]),
            tx_to=datetime.fromisoformat(row[5]) if row[5] else None,
            data=json.loads(row[6]),
        )


@dataclass
class Relationship:
    """A relationship between entities with bi-temporal tracking

    Attributes:
        id: Unique identifier for the relationship
        rel_type: Type of relationship (e.g., 'job_uses_component', 'ticket_references_job')
        source_id: ID of the source entity
        target_id: ID of the target entity
        valid_from: When the relationship became true
        valid_to: When the relationship stopped being true (None if still valid)
        tx_from: When ai-assist learned about the relationship
        tx_to: When ai-assist stopped believing it (None if current belief)
        properties: Additional properties for the relationship
    """

    id: str
    rel_type: str
    source_id: str
    target_id: str
    valid_from: datetime
    valid_to: datetime | None
    tx_from: datetime
    tx_to: datetime | None
    properties: dict[str, Any]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "rel_type": self.rel_type,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "valid_from": self.valid_from.isoformat(),
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "tx_from": self.tx_from.isoformat(),
            "tx_to": self.tx_to.isoformat() if self.tx_to else None,
            "properties": self.properties,
        }

    @classmethod
    def from_row(cls, row: tuple) -> Relationship:
        """Create Relationship from database row"""
        return cls(
            id=row[0],
            rel_type=row[1],
            source_id=row[2],
            target_id=row[3],
            valid_from=datetime.fromisoformat(row[4]),
            valid_to=datetime.fromisoformat(row[5]) if row[5] else None,
            tx_from=datetime.fromisoformat(row[6]),
            tx_to=datetime.fromisoformat(row[7]) if row[7] else None,
            properties=json.loads(row[8]) if row[8] else {},
        )


class KnowledgeGraph:
    """Bi-temporal knowledge graph storage engine"""

    def __init__(self, db_path: str | None = None):
        """Initialize the knowledge graph

        Args:
            db_path: Path to SQLite database file. If None, uses <config_dir>/knowledge_graph.db
                    If ":memory:", uses in-memory database for testing
        """
        if db_path is None:
            db_path = str(get_config_dir() / "knowledge_graph.db")

        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
        self._batch_mode = False
        self._backfill_done = False
        try:
            self.conn.enable_load_extension(True)
        except AttributeError:
            raise RuntimeError(
                "Your Python's sqlite3 module was compiled without extension loading support.\n"
                "This is common with the official python.org macOS installer.\n"
                "Please use a Python that includes it, for example:\n"
                "  - uv python install 3.12\n"
                "  - brew install python@3.12\n"
                "  - pyenv install 3.12"
            ) from None
        import sqlite_vec

        sqlite_vec.load(self.conn)
        logging.info("sqlite-vec extension loaded for vector search")
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            pass  # WAL mode is best-effort; falls back to default journal mode
        self._create_schema()

    def _create_schema(self):
        """Create database schema if it doesn't exist"""
        cursor = self.conn.cursor()

        # Entities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                valid_from TIMESTAMP NOT NULL,
                valid_to TIMESTAMP,
                tx_from TIMESTAMP NOT NULL,
                tx_to TIMESTAMP,
                data JSON NOT NULL
            )
        """)

        # Indexes for temporal queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_type
            ON entities(entity_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_valid_time
            ON entities(valid_from, valid_to)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_tx_time
            ON entities(tx_from, tx_to)
        """)

        # Relationships table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                rel_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                valid_from TIMESTAMP NOT NULL,
                valid_to TIMESTAMP,
                tx_from TIMESTAMP NOT NULL,
                tx_to TIMESTAMP,
                properties JSON,
                FOREIGN KEY (source_id) REFERENCES entities(id),
                FOREIGN KEY (target_id) REFERENCES entities(id)
            )
        """)

        # Indexes for relationship queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_relationships_type
            ON relationships(rel_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_relationships_source
            ON relationships(source_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_relationships_target
            ON relationships(target_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_relationships_valid_time
            ON relationships(valid_from, valid_to)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_relationships_tx_time
            ON relationships(tx_from, tx_to)
        """)

        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
                entity_id TEXT PRIMARY KEY,
                embedding float[384]
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_access (
                entity_id TEXT NOT NULL,
                accessed_at TIMESTAMP NOT NULL,
                query_context TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_knowledge_access_entity
            ON knowledge_access(entity_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_knowledge_access_time
            ON knowledge_access(accessed_at)
        """)

        self.conn.commit()

    def _maybe_commit(self):
        """Commit unless we are in batch mode."""
        if not self._batch_mode:
            self.conn.commit()

    @contextmanager
    def batch(self):
        """Context manager for batching multiple writes into a single commit.

        Defers conn.commit() until the batch exits, which is dramatically
        faster for multiple sequential inserts.
        """
        self._batch_mode = True
        try:
            yield self
        finally:
            self._batch_mode = False
            self.conn.commit()

    def insert_entity(
        self,
        entity_type: str,
        data: dict[str, Any],
        valid_from: datetime,
        tx_from: datetime | None = None,
        entity_id: str | None = None,
        valid_to: datetime | None = None,
        tx_to: datetime | None = None,
    ) -> Entity:
        """Insert a new entity into the knowledge graph

        Args:
            entity_type: Type of entity (e.g., 'dci_job', 'jira_ticket')
            data: Entity data as a dictionary
            valid_from: When the fact became true in reality
            tx_from: When ai-assist learned about it (defaults to now)
            entity_id: Optional entity ID (generated if not provided)
            valid_to: When the fact stopped being true (None if still valid)
            tx_to: When ai-assist stopped believing it (None if current belief)

        Returns:
            The created Entity
        """
        if tx_from is None:
            tx_from = datetime.now()

        if entity_id is None:
            entity_id = f"{entity_type}-{uuid4().hex[:8]}"

        entity = Entity(
            id=entity_id,
            entity_type=entity_type,
            valid_from=valid_from,
            valid_to=valid_to,
            tx_from=tx_from,
            tx_to=tx_to,
            data=data,
        )

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO entities (id, entity_type, valid_from, valid_to, tx_from, tx_to, data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                entity.id,
                entity.entity_type,
                entity.valid_from.isoformat(),
                entity.valid_to.isoformat() if entity.valid_to else None,
                entity.tx_from.isoformat(),
                entity.tx_to.isoformat() if entity.tx_to else None,
                json.dumps(entity.data),
            ),
        )
        if entity.entity_type != "tool_result":
            text = self._entity_text_repr(entity.entity_type, entity.data)
            self._embed_and_store(entity.id, text)
        self._maybe_commit()

        return entity

    def upsert_entity(
        self,
        entity_type: str,
        data: dict[str, Any],
        valid_from: datetime,
        tx_from: datetime | None = None,
        entity_id: str | None = None,
        valid_to: datetime | None = None,
        tx_to: datetime | None = None,
    ) -> Entity:
        """Insert or update an entity in the knowledge graph.

        If an entity with the same id already exists, its data and timestamps
        are replaced. This is useful for tool results where the same tool+args
        should refresh the stored data rather than silently fail.

        Args:
            entity_type: Type of entity (e.g., 'tool_result')
            data: Entity data as a dictionary
            valid_from: When the fact became true in reality
            tx_from: When ai-assist learned about it (defaults to now)
            entity_id: Optional entity ID (generated if not provided)
            valid_to: When the fact stopped being true (None if still valid)
            tx_to: When ai-assist stopped believing it (None if current belief)

        Returns:
            The created or updated Entity
        """
        if tx_from is None:
            tx_from = datetime.now()

        if entity_id is None:
            entity_id = f"{entity_type}-{uuid4().hex[:8]}"

        entity = Entity(
            id=entity_id,
            entity_type=entity_type,
            valid_from=valid_from,
            valid_to=valid_to,
            tx_from=tx_from,
            tx_to=tx_to,
            data=data,
        )

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO entities (id, entity_type, valid_from, valid_to, tx_from, tx_to, data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                entity.id,
                entity.entity_type,
                entity.valid_from.isoformat(),
                entity.valid_to.isoformat() if entity.valid_to else None,
                entity.tx_from.isoformat(),
                entity.tx_to.isoformat() if entity.tx_to else None,
                json.dumps(entity.data),
            ),
        )
        if entity.entity_type != "tool_result":
            text = self._entity_text_repr(entity.entity_type, entity.data)
            self._embed_and_store(entity.id, text)
        self._maybe_commit()

        return entity

    def update_entity(
        self, entity_id: str, valid_to: datetime | None = None, tx_to: datetime | None = None
    ) -> Entity | None:
        """Update an entity's temporal bounds

        This is used to close the temporal window (set valid_to or tx_to)
        For updating data, insert a new entity version instead.

        Args:
            entity_id: ID of the entity to update
            valid_to: Set when the fact stopped being true
            tx_to: Set when ai-assist stopped believing it

        Returns:
            The updated Entity or None if not found
        """
        cursor = self.conn.cursor()

        # First get the current entity
        cursor.execute("SELECT * FROM entities WHERE id = ?", (entity_id,))
        row = cursor.fetchone()
        if not row:
            return None

        entity = Entity.from_row(row)

        # Update temporal bounds
        if valid_to is not None:
            entity.valid_to = valid_to
        if tx_to is not None:
            entity.tx_to = tx_to

        cursor.execute(
            """
            UPDATE entities
            SET valid_to = ?, tx_to = ?
            WHERE id = ?
        """,
            (
                entity.valid_to.isoformat() if entity.valid_to else None,
                entity.tx_to.isoformat() if entity.tx_to else None,
                entity_id,
            ),
        )
        self._maybe_commit()

        return entity

    def insert_relationship(
        self,
        rel_type: str,
        source_id: str,
        target_id: str,
        valid_from: datetime,
        tx_from: datetime | None = None,
        properties: dict[str, Any] | None = None,
        rel_id: str | None = None,
        valid_to: datetime | None = None,
        tx_to: datetime | None = None,
    ) -> Relationship:
        """Insert a new relationship between entities

        Args:
            rel_type: Type of relationship
            source_id: Source entity ID
            target_id: Target entity ID
            valid_from: When the relationship became true
            tx_from: When ai-assist learned about it (defaults to now)
            properties: Additional properties
            rel_id: Optional relationship ID (generated if not provided)
            valid_to: When the relationship stopped being true
            tx_to: When ai-assist stopped believing it

        Returns:
            The created Relationship
        """
        if tx_from is None:
            tx_from = datetime.now()

        if rel_id is None:
            rel_id = f"rel-{uuid4().hex[:8]}"

        if properties is None:
            properties = {}

        relationship = Relationship(
            id=rel_id,
            rel_type=rel_type,
            source_id=source_id,
            target_id=target_id,
            valid_from=valid_from,
            valid_to=valid_to,
            tx_from=tx_from,
            tx_to=tx_to,
            properties=properties,
        )

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO relationships
            (id, rel_type, source_id, target_id, valid_from, valid_to, tx_from, tx_to, properties)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                relationship.id,
                relationship.rel_type,
                relationship.source_id,
                relationship.target_id,
                relationship.valid_from.isoformat(),
                relationship.valid_to.isoformat() if relationship.valid_to else None,
                relationship.tx_from.isoformat(),
                relationship.tx_to.isoformat() if relationship.tx_to else None,
                json.dumps(relationship.properties),
            ),
        )
        self._maybe_commit()

        return relationship

    def relationship_exists(self, rel_type: str, source_id: str, target_id: str) -> bool:
        """Check if a current relationship already exists between two entities

        Args:
            rel_type: Type of relationship
            source_id: Source entity ID
            target_id: Target entity ID

        Returns:
            True if such a relationship currently exists (tx_to IS NULL)
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) FROM relationships
            WHERE rel_type = ? AND source_id = ? AND target_id = ?
            AND tx_to IS NULL
        """,
            (rel_type, source_id, target_id),
        )
        return cursor.fetchone()[0] > 0

    def query_as_of(
        self,
        tx_time: datetime,
        entity_type: str | None = None,
        limit: int | None = None,
        search_text: str | None = None,
        valid_from_after: datetime | None = None,
    ) -> list[Entity]:
        """Query entities as they were known at a specific transaction time

        This answers: "What did ai-assist know at time X?"

        Args:
            tx_time: The transaction time to query
            entity_type: Optional filter by entity type
            limit: Optional limit on number of results
            search_text: Optional case-insensitive text search within entity data JSON
            valid_from_after: Optional minimum valid_from time filter

        Returns:
            List of entities that ai-assist believed at tx_time
        """
        cursor = self.conn.cursor()

        query = """
            SELECT * FROM entities
            WHERE tx_from <= ? AND (tx_to IS NULL OR tx_to > ?)
        """
        params: list[str | int | float] = [tx_time.isoformat(), tx_time.isoformat()]

        if entity_type:
            query += " AND entity_type = ?"
            params.append(entity_type)

        if search_text:
            query += " AND data LIKE ?"
            params.append(f"%{search_text}%")

        if valid_from_after:
            query += " AND valid_from >= ?"
            params.append(valid_from_after.isoformat())

        query += " ORDER BY tx_from DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        return [Entity.from_row(row) for row in cursor.fetchall()]

    def query_valid_at(
        self, valid_time: datetime, entity_type: str | None = None, limit: int | None = None
    ) -> list[Entity]:
        """Query entities that were valid at a specific time in reality

        This answers: "What was true at time X?"

        Args:
            valid_time: The valid time to query
            entity_type: Optional filter by entity type
            limit: Optional limit on number of results

        Returns:
            List of entities that were valid at valid_time
        """
        cursor = self.conn.cursor()

        query = """
            SELECT * FROM entities
            WHERE valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)
            AND tx_to IS NULL
        """
        params: list[str | int | float] = [valid_time.isoformat(), valid_time.isoformat()]

        if entity_type:
            query += " AND entity_type = ?"
            params.append(entity_type)

        query += " ORDER BY valid_from DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        return [Entity.from_row(row) for row in cursor.fetchall()]

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID

        Args:
            entity_id: The entity ID

        Returns:
            The entity or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM entities WHERE id = ?", (entity_id,))
        row = cursor.fetchone()
        return Entity.from_row(row) if row else None

    def get_related_entities(
        self,
        entity_id: str,
        rel_type: str | None = None,
        direction: Literal["outgoing", "incoming", "both"] = "outgoing",
    ) -> list[tuple[Relationship, Entity]]:
        """Get entities related to this entity

        Args:
            entity_id: The entity ID
            rel_type: Optional filter by relationship type
            direction: Direction of traversal (outgoing, incoming, or both)

        Returns:
            List of (relationship, related_entity) tuples
        """
        cursor = self.conn.cursor()
        results = []

        if direction in ["outgoing", "both"]:
            query = """
                SELECT r.*, e.*
                FROM relationships r
                JOIN entities e ON r.target_id = e.id
                WHERE r.source_id = ? AND r.tx_to IS NULL AND e.tx_to IS NULL
            """
            params = [entity_id]

            if rel_type:
                query += " AND r.rel_type = ?"
                params.append(rel_type)

            cursor.execute(query, params)
            for row in cursor.fetchall():
                rel = Relationship.from_row(row[:9])
                entity = Entity.from_row(row[9:])
                results.append((rel, entity))

        if direction in ["incoming", "both"]:
            query = """
                SELECT r.*, e.*
                FROM relationships r
                JOIN entities e ON r.source_id = e.id
                WHERE r.target_id = ? AND r.tx_to IS NULL AND e.tx_to IS NULL
            """
            params = [entity_id]

            if rel_type:
                query += " AND r.rel_type = ?"
                params.append(rel_type)

            cursor.execute(query, params)
            for row in cursor.fetchall():
                rel = Relationship.from_row(row[:9])
                entity = Entity.from_row(row[9:])
                results.append((rel, entity))

        return results

    def insert_knowledge(
        self,
        entity_type: str,
        key: str,
        content: str,
        metadata: dict | None = None,
        valid_from: datetime | None = None,
        confidence: float = 1.0,
    ) -> str:
        """Insert a knowledge entity (preference, lesson, context, rationale)

        Args:
            entity_type: One of user_preference, lesson_learned, project_context, decision_rationale
            key: Unique identifier (e.g., "python_testing_framework")
            content: The actual knowledge (text)
            metadata: Additional context (tags, source, etc.)
            valid_from: When this became true (defaults to now)
            confidence: Agent's confidence (0.0-1.0)

        Returns:
            Entity ID
        """
        valid_types = ["user_preference", "lesson_learned", "project_context", "decision_rationale"]
        if entity_type not in valid_types:
            raise ValueError(f"Invalid knowledge entity type: {entity_type}")

        if valid_from is None:
            valid_from = datetime.now()

        if metadata is None:
            metadata = {}
        metadata["confidence"] = confidence

        entity_id = f"{entity_type}:{key}"

        data = {"key": key, "content": content, "metadata": metadata}

        # Upsert: update existing entity or insert new one
        existing = self.get_entity(entity_id)
        if existing:
            now = datetime.now()
            cursor = self.conn.cursor()
            cursor.execute(
                """
                UPDATE entities
                SET data = ?, valid_from = ?, tx_from = ?, tx_to = NULL
                WHERE id = ?
            """,
                (json.dumps(data), valid_from.isoformat(), now.isoformat(), entity_id),
            )
            text = self._entity_text_repr(entity_type, data)
            self._embed_and_store(entity_id, text)
            self._maybe_commit()
        else:
            # insert_entity already handles embedding for non-tool_result types
            self.insert_entity(entity_id=entity_id, entity_type=entity_type, data=data, valid_from=valid_from)

        return entity_id

    def search_knowledge(
        self,
        entity_type: str | None = None,
        key_pattern: str | None = None,
        tags: list[str] | None = None,
        since: datetime | None = None,
        min_confidence: float = 0.0,
        limit: int = 20,
        include_future: bool = False,
    ) -> list[dict]:
        """Search knowledge entities

        Args:
            entity_type: Filter by entity type
            key_pattern: Search keys with LIKE pattern
            tags: Filter by tags (must have all)
            since: Only return knowledge learned since this time
            min_confidence: Minimum confidence threshold
            limit: Max results

        Returns:
            List of dicts with entity_id, key, content, metadata, timestamps
        """
        query = """
            SELECT id, entity_type, data, valid_from, tx_from
            FROM entities
            WHERE tx_to IS NULL
        """

        params: list[str | int | float] = []

        if not include_future:
            query += " AND valid_from <= ?"
            params.append(datetime.now().isoformat())

        if entity_type:
            query += " AND entity_type = ?"
            params.append(entity_type)

        if key_pattern:
            query += " AND json_extract(data, '$.key') LIKE ?"
            params.append(key_pattern)

        if since:
            query += " AND tx_from >= ?"
            params.append(since.isoformat())

        if min_confidence > 0.0:
            query += " AND CAST(json_extract(data, '$.metadata.confidence') AS REAL) >= ?"
            params.append(min_confidence)

        if tags:
            for tag in tags:
                query += " AND json_extract(data, '$.metadata.tags') LIKE ?"
                params.append(f"%{tag}%")

        query += " ORDER BY tx_from DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.execute(query, params)

        results = []
        for row in cursor.fetchall():
            entity_id, entity_type, data_json, valid_from, tx_from = row
            data = json.loads(data_json)

            results.append(
                {
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "key": data.get("key"),
                    "content": data.get("content"),
                    "metadata": data.get("metadata", {}),
                    "valid_from": valid_from,
                    "learned_at": tx_from,
                }
            )

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the knowledge graph

        Returns:
            Dictionary with counts and other statistics
        """
        cursor = self.conn.cursor()

        # Count entities by type
        cursor.execute("""
            SELECT entity_type, COUNT(*)
            FROM entities
            WHERE tx_to IS NULL
            GROUP BY entity_type
        """)
        entity_counts = dict(cursor.fetchall())

        # Total entities
        cursor.execute("SELECT COUNT(*) FROM entities WHERE tx_to IS NULL")
        total_entities = cursor.fetchone()[0]

        # Count relationships by type
        cursor.execute("""
            SELECT rel_type, COUNT(*)
            FROM relationships
            WHERE tx_to IS NULL
            GROUP BY rel_type
        """)
        relationship_counts = dict(cursor.fetchall())

        # Total relationships
        cursor.execute("SELECT COUNT(*) FROM relationships WHERE tx_to IS NULL")
        total_relationships = cursor.fetchone()[0]

        return {
            "db_path": self.db_path,
            "total_entities": total_entities,
            "entities_by_type": entity_counts,
            "total_relationships": total_relationships,
            "relationships_by_type": relationship_counts,
        }

    def get_all_current_entities(self) -> list[Entity]:
        """Get all current entities (tx_to IS NULL)

        Returns:
            List of all entities that ai-assist currently believes
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM entities WHERE tx_to IS NULL ORDER BY entity_type, tx_from DESC")
        return [Entity.from_row(row) for row in cursor.fetchall()]

    def get_all_current_relationships(self) -> list[Relationship]:
        """Get all current relationships (tx_to IS NULL)

        Returns:
            List of all relationships that ai-assist currently believes
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM relationships WHERE tx_to IS NULL")
        return [Relationship.from_row(row) for row in cursor.fetchall()]

    def record_access(self, entity_ids: list[str], query_context: str) -> None:
        """Record that entities were accessed/recalled."""
        if not entity_ids:
            return
        now = datetime.now().isoformat()
        self.conn.executemany(
            "INSERT INTO knowledge_access (entity_id, accessed_at, query_context) VALUES (?, ?, ?)",
            [(eid, now, query_context) for eid in entity_ids],
        )
        self._maybe_commit()

    def get_access_stats(self, top_n: int = 10, stale_days: int = 7) -> dict[str, Any]:
        """Get knowledge access statistics for health reporting."""
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM knowledge_access")
        total_accesses = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT ka.entity_id, e.entity_type, COUNT(*) as cnt, MAX(ka.accessed_at) as last_access
            FROM knowledge_access ka
            JOIN entities e ON e.id = ka.entity_id AND e.tx_to IS NULL
            GROUP BY ka.entity_id
            ORDER BY cnt DESC
            LIMIT ?
            """,
            (top_n,),
        )
        most_accessed = [
            {"entity_id": r[0], "entity_type": r[1], "access_count": r[2], "last_accessed": r[3]}
            for r in cursor.fetchall()
        ]

        knowledge_types = ("user_preference", "lesson_learned", "project_context", "decision_rationale")
        cursor.execute(
            """
            SELECT e.id, e.entity_type
            FROM entities e
            WHERE e.tx_to IS NULL AND e.entity_type IN (?,?,?,?)
            AND e.id NOT IN (SELECT DISTINCT entity_id FROM knowledge_access)
            LIMIT 20
            """,
            knowledge_types,
        )
        never_accessed = [{"entity_id": r[0], "entity_type": r[1]} for r in cursor.fetchall()]

        cutoff = (datetime.now() - timedelta(days=stale_days)).isoformat()
        cursor.execute(
            """
            SELECT e.id, e.entity_type, MAX(ka.accessed_at) as last_access
            FROM entities e
            JOIN knowledge_access ka ON ka.entity_id = e.id
            WHERE e.tx_to IS NULL AND e.entity_type IN (?,?,?,?)
            GROUP BY e.id
            HAVING MAX(ka.accessed_at) < ?
            LIMIT 20
            """,
            (*knowledge_types, cutoff),
        )
        stale_entries = [{"entity_id": r[0], "entity_type": r[1], "last_accessed": r[2]} for r in cursor.fetchall()]

        return {
            "total_accesses": total_accesses,
            "most_accessed": most_accessed,
            "never_accessed": never_accessed,
            "never_accessed_count": len(never_accessed),
            "stale_entries": stale_entries,
            "stale_entries_count": len(stale_entries),
            "stale_threshold_days": stale_days,
        }

    @staticmethod
    def _entity_text_repr(entity_type: str, data: dict[str, Any]) -> str:
        """Build text representation of an entity for embedding."""
        if "key" in data and "content" in data:
            return f"{data['key']}: {data['content']}"
        summary = data.get("summary") or data.get("name") or data.get("content")
        if summary:
            return f"{entity_type}: {summary}"
        return f"{entity_type}: {json.dumps(data)[:200]}"

    def _embed_and_store(self, entity_id: str, text: str) -> None:
        """Embed text and store the vector alongside the entity.

        Does not commit -- the caller is responsible for calling _maybe_commit().
        Failures are logged and silently ignored so entity insertion is never
        blocked by embedding issues (e.g. model not yet downloaded).
        """
        try:
            from .embedding import EmbeddingModel

            vec_bytes = EmbeddingModel.get().encode_one(text)
            self.conn.execute("DELETE FROM vec_embeddings WHERE entity_id = ?", (entity_id,))
            self.conn.execute(
                "INSERT INTO vec_embeddings(entity_id, embedding) VALUES (?, ?)",
                (entity_id, vec_bytes),
            )
        except Exception as e:
            logging.debug("Embedding failed for %s, will be backfilled later: %s", entity_id, e)

    def semantic_search(
        self,
        query_text: str,
        limit: int = 5,
        entity_types: list[str] | None = None,
        min_confidence: float = 0.0,
        min_score: float = 0.0,
        include_future: bool = False,
    ) -> list[dict]:
        """Search entities by semantic similarity to query text."""
        from .embedding import EmbeddingModel

        if not self._backfill_done:
            self._backfill_done = True
            self.backfill_embeddings()

        # Check if there are any embeddings
        count = self.conn.execute("SELECT COUNT(*) FROM vec_embeddings").fetchone()[0]
        if count == 0:
            return []

        try:
            query_vec = EmbeddingModel.get().encode_one(query_text)
        except Exception as e:
            logging.debug("Embedding model unavailable for search: %s", e)
            return []

        over_fetch = limit * 3
        if include_future:
            rows = self.conn.execute(
                """
                SELECT v.entity_id, v.distance, e.entity_type, e.data, e.valid_from, e.tx_from
                FROM vec_embeddings v
                JOIN entities e ON v.entity_id = e.id
                WHERE v.embedding MATCH ? AND k = ?
                AND e.tx_to IS NULL
                ORDER BY v.distance
                """,
                (query_vec, over_fetch),
            ).fetchall()
        else:
            now = datetime.now().isoformat()
            rows = self.conn.execute(
                """
                SELECT v.entity_id, v.distance, e.entity_type, e.data, e.valid_from, e.tx_from
                FROM vec_embeddings v
                JOIN entities e ON v.entity_id = e.id
                WHERE v.embedding MATCH ? AND k = ?
                AND e.tx_to IS NULL
                AND e.valid_from <= ?
                ORDER BY v.distance
                """,
                (query_vec, over_fetch, now),
            ).fetchall()

        results: list[dict] = []
        skipped_type = 0
        skipped_conf = 0
        skipped_score = 0
        for row in rows:
            entity_id, distance, entity_type, data_json, valid_from, tx_from = row
            # Convert L2 distance to similarity score (0-1) for normalized vectors
            score = max(0.0, 1.0 - distance * distance / 2.0)
            if entity_types and entity_type not in entity_types:
                skipped_type += 1
                continue
            data = json.loads(data_json)
            conf = data.get("metadata", {}).get("confidence", 1.0)
            if isinstance(conf, str):
                try:
                    conf = float(conf)
                except ValueError, TypeError:
                    conf = 1.0
            if conf < min_confidence:
                skipped_conf += 1
                logging.debug(
                    "semantic_search: skip %s (conf=%.2f < %.2f, score=%.3f)",
                    entity_id,
                    conf,
                    min_confidence,
                    score,
                )
                continue
            if score < min_score:
                skipped_score += 1
                logging.debug(
                    "semantic_search: skip %s (score=%.3f < %.2f)",
                    entity_id,
                    score,
                    min_score,
                )
                continue
            logging.debug(
                "semantic_search: match %s [%s] score=%.3f conf=%.2f",
                entity_id,
                entity_type,
                score,
                conf,
            )
            results.append(
                {
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "key": data.get("key"),
                    "content": data.get("content"),
                    "metadata": data.get("metadata", {}),
                    "valid_from": valid_from,
                    "learned_at": tx_from,
                    "score": score,
                }
            )
            if len(results) >= limit:
                break
        logging.debug(
            "semantic_search: query=%r candidates=%d matched=%d " "skipped(type=%d conf=%d score=%d)",
            query_text[:80],
            len(rows),
            len(results),
            skipped_type,
            skipped_conf,
            skipped_score,
        )
        return results

    def backfill_embeddings(self) -> int:
        """Populate vector embeddings for entities that don't have them yet.

        Skips tool_result entities (high-frequency JSON blobs not useful for
        text embedding).  Called automatically at startup to migrate
        pre-existing KG databases.

        Returns:
            Number of entities backfilled.
        """
        cursor = self.conn.execute("""
            SELECT e.id, e.entity_type, e.data
            FROM entities e
            LEFT JOIN vec_embeddings v ON e.id = v.entity_id
            WHERE e.tx_to IS NULL AND v.entity_id IS NULL
            AND e.entity_type != 'tool_result'
            """)
        count = 0
        for row in cursor.fetchall():
            entity_id, entity_type, data_json = row
            data = json.loads(data_json)
            text = self._entity_text_repr(entity_type, data)
            self._embed_and_store(entity_id, text)
            count += 1
        if count:
            self.conn.commit()
            logging.info("Backfilled embeddings for %d entities", count)
        return count

    def close(self):
        """Close the database connection"""
        self.conn.close()

    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
