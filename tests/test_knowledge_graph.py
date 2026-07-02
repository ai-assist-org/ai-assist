"""Tests for bi-temporal knowledge graph"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from ai_assist.knowledge_graph import KnowledgeGraph


@pytest.fixture
def kg():
    """Create in-memory knowledge graph for testing"""
    graph = KnowledgeGraph(db_path=":memory:")
    yield graph
    graph.close()


def test_insert_entity_with_bitemporal(kg):
    """Test inserting an entity with bi-temporal tracking"""
    valid_time = datetime(2026, 2, 4, 10, 0)
    tx_time = datetime(2026, 2, 4, 11, 0)

    entity = kg.insert_entity(
        entity_type="dci_job",
        entity_id="job-test",
        valid_from=valid_time,
        tx_from=tx_time,
        data={"status": "failure", "remoteci": "test-lab"},
    )

    assert entity.id == "job-test"
    assert entity.entity_type == "dci_job"
    assert entity.valid_from == valid_time
    assert entity.tx_from == tx_time
    assert entity.valid_to is None
    assert entity.tx_to is None
    assert entity.data["status"] == "failure"

    # Verify we learned about it later than it happened
    assert entity.valid_from < entity.tx_from


def test_insert_entity_auto_tx_time(kg):
    """Test that tx_from defaults to now if not provided"""
    before = datetime.now()
    entity = kg.insert_entity(
        entity_type="jira_ticket", data={"key": "CILAB-1234"}, valid_from=datetime(2026, 2, 4, 9, 0)
    )
    after = datetime.now()

    assert before <= entity.tx_from <= after


def test_get_entity(kg):
    """Test retrieving an entity by ID"""
    _entity = kg.insert_entity(
        entity_type="component",
        entity_id="comp-123",
        valid_from=datetime(2026, 2, 4, 10, 0),
        data={"type": "ocp", "version": "4.19.0"},
    )

    retrieved = kg.get_entity("comp-123")
    assert retrieved is not None
    assert retrieved.id == "comp-123"
    assert retrieved.data["version"] == "4.19.0"

    # Non-existent entity
    assert kg.get_entity("non-existent") is None


def test_update_entity_temporal_bounds(kg):
    """Test updating entity temporal bounds"""
    _entity = kg.insert_entity(
        entity_type="dci_job", entity_id="job-456", valid_from=datetime(2026, 2, 4, 10, 0), data={"status": "running"}
    )

    # Job finished at 11:00
    valid_to = datetime(2026, 2, 4, 11, 0)
    updated = kg.update_entity("job-456", valid_to=valid_to)

    assert updated is not None
    assert updated.valid_to == valid_to

    # Verify in database
    retrieved = kg.get_entity("job-456")
    assert retrieved.valid_to == valid_to


def test_query_as_of(kg):
    """Test querying what ai-assist knew at a specific transaction time"""
    # Insert entity at tx_time = 11:00
    kg.insert_entity(
        entity_type="dci_job",
        entity_id="job-early",
        valid_from=datetime(2026, 2, 4, 10, 0),
        tx_from=datetime(2026, 2, 4, 11, 0),
        data={"status": "failure"},
    )

    # Insert another at tx_time = 12:00
    kg.insert_entity(
        entity_type="dci_job",
        entity_id="job-late",
        valid_from=datetime(2026, 2, 4, 10, 30),
        tx_from=datetime(2026, 2, 4, 12, 0),
        data={"status": "error"},
    )

    # Query as_of 10:30 -> should find nothing (we didn't know yet)
    results = kg.query_as_of(datetime(2026, 2, 4, 10, 30))
    assert len(results) == 0

    # Query as_of 11:30 -> should find only job-early
    results = kg.query_as_of(datetime(2026, 2, 4, 11, 30))
    assert len(results) == 1
    assert results[0].id == "job-early"

    # Query as_of 13:00 -> should find both
    results = kg.query_as_of(datetime(2026, 2, 4, 13, 0))
    assert len(results) == 2


def test_query_as_of_with_corrections(kg):
    """Test that corrected beliefs don't show up in as-of queries"""
    # Insert initial belief at 11:00
    _entity = kg.insert_entity(
        entity_type="dci_job",
        entity_id="job-corrected",
        valid_from=datetime(2026, 2, 4, 10, 0),
        tx_from=datetime(2026, 2, 4, 11, 0),
        data={"status": "failure"},
    )

    # At 11:30, we realize our belief was wrong and correct it
    kg.update_entity("job-corrected", tx_to=datetime(2026, 2, 4, 11, 30))

    # Insert new version with corrected data
    kg.insert_entity(
        entity_type="dci_job",
        entity_id="job-corrected-v2",
        valid_from=datetime(2026, 2, 4, 10, 0),
        tx_from=datetime(2026, 2, 4, 11, 30),
        data={"status": "success"},  # Actually succeeded
    )

    # Query as_of 11:15 -> should see original (wrong) belief
    results = kg.query_as_of(datetime(2026, 2, 4, 11, 15))
    assert len(results) == 1
    assert results[0].id == "job-corrected"
    assert results[0].data["status"] == "failure"

    # Query as_of 12:00 -> should see corrected belief
    results = kg.query_as_of(datetime(2026, 2, 4, 12, 0))
    assert len(results) == 1
    assert results[0].id == "job-corrected-v2"
    assert results[0].data["status"] == "success"


def test_query_valid_at(kg):
    """Test querying what was true at a specific valid time"""
    # Job that ran from 10:00 to 11:00
    kg.insert_entity(
        entity_type="dci_job",
        entity_id="job-morning",
        valid_from=datetime(2026, 2, 4, 10, 0),
        valid_to=datetime(2026, 2, 4, 11, 0),
        tx_from=datetime(2026, 2, 4, 11, 5),
        data={"status": "failure"},
    )

    # Job that ran from 11:00 to 12:00
    kg.insert_entity(
        entity_type="dci_job",
        entity_id="job-midday",
        valid_from=datetime(2026, 2, 4, 11, 0),
        valid_to=datetime(2026, 2, 4, 12, 0),
        tx_from=datetime(2026, 2, 4, 12, 5),
        data={"status": "success"},
    )

    # Query valid_at 10:30 -> should find only morning job
    results = kg.query_valid_at(datetime(2026, 2, 4, 10, 30))
    assert len(results) == 1
    assert results[0].id == "job-morning"

    # Query valid_at 11:30 -> should find only midday job
    results = kg.query_valid_at(datetime(2026, 2, 4, 11, 30))
    assert len(results) == 1
    assert results[0].id == "job-midday"

    # Query valid_at 9:00 -> should find nothing
    results = kg.query_valid_at(datetime(2026, 2, 4, 9, 0))
    assert len(results) == 0


def test_insert_relationship(kg):
    """Test inserting relationships between entities"""
    # Create job and component
    job = kg.insert_entity(
        entity_type="dci_job", entity_id="job-789", valid_from=datetime(2026, 2, 4, 10, 0), data={"status": "failure"}
    )

    component = kg.insert_entity(
        entity_type="component",
        entity_id="comp-ocp-419",
        valid_from=datetime(2026, 2, 4, 0, 0),
        data={"type": "ocp", "version": "4.19.0"},
    )

    # Create relationship
    rel = kg.insert_relationship(
        rel_type="job_uses_component",
        source_id=job.id,
        target_id=component.id,
        valid_from=datetime(2026, 2, 4, 10, 0),
        properties={"build": "ga"},
    )

    assert rel.rel_type == "job_uses_component"
    assert rel.source_id == job.id
    assert rel.target_id == component.id
    assert rel.properties["build"] == "ga"


def test_get_related_entities_outgoing(kg):
    """Test getting entities via outgoing relationships"""
    # Create job
    job = kg.insert_entity(
        entity_type="dci_job", entity_id="job-abc", valid_from=datetime(2026, 2, 4, 10, 0), data={"status": "failure"}
    )

    # Create components
    comp1 = kg.insert_entity(
        entity_type="component",
        entity_id="comp-1",
        valid_from=datetime(2026, 2, 4, 0, 0),
        data={"type": "ocp", "version": "4.19.0"},
    )

    comp2 = kg.insert_entity(
        entity_type="component",
        entity_id="comp-2",
        valid_from=datetime(2026, 2, 4, 0, 0),
        data={"type": "storage", "name": "ceph"},
    )

    # Create relationships
    kg.insert_relationship(
        rel_type="job_uses_component", source_id=job.id, target_id=comp1.id, valid_from=datetime(2026, 2, 4, 10, 0)
    )

    kg.insert_relationship(
        rel_type="job_uses_component", source_id=job.id, target_id=comp2.id, valid_from=datetime(2026, 2, 4, 10, 0)
    )

    # Get related entities
    related = kg.get_related_entities(job.id, direction="outgoing")
    assert len(related) == 2

    # Extract entity IDs
    entity_ids = {entity.id for _, entity in related}
    assert entity_ids == {"comp-1", "comp-2"}


def test_get_related_entities_incoming(kg):
    """Test getting entities via incoming relationships"""
    # Create ticket
    ticket = kg.insert_entity(
        entity_type="jira_ticket",
        entity_id="ticket-123",
        valid_from=datetime(2026, 2, 4, 12, 0),
        data={"key": "CILAB-1234"},
    )

    # Create jobs that reference this ticket
    job1 = kg.insert_entity(
        entity_type="dci_job", entity_id="job-1", valid_from=datetime(2026, 2, 4, 10, 0), data={"status": "failure"}
    )

    job2 = kg.insert_entity(
        entity_type="dci_job", entity_id="job-2", valid_from=datetime(2026, 2, 4, 10, 30), data={"status": "failure"}
    )

    # Create relationships from jobs to ticket
    kg.insert_relationship(
        rel_type="job_references_ticket", source_id=job1.id, target_id=ticket.id, valid_from=datetime(2026, 2, 4, 12, 0)
    )

    kg.insert_relationship(
        rel_type="job_references_ticket", source_id=job2.id, target_id=ticket.id, valid_from=datetime(2026, 2, 4, 12, 0)
    )

    # Get jobs that reference this ticket (incoming)
    related = kg.get_related_entities(ticket.id, direction="incoming")
    assert len(related) == 2

    entity_ids = {entity.id for _, entity in related}
    assert entity_ids == {"job-1", "job-2"}


def test_get_related_entities_filtered_by_type(kg):
    """Test filtering related entities by relationship type"""
    job = kg.insert_entity(
        entity_type="dci_job", entity_id="job-xyz", valid_from=datetime(2026, 2, 4, 10, 0), data={"status": "failure"}
    )

    comp = kg.insert_entity(
        entity_type="component", entity_id="comp-x", valid_from=datetime(2026, 2, 4, 0, 0), data={"type": "ocp"}
    )

    ticket = kg.insert_entity(
        entity_type="jira_ticket",
        entity_id="ticket-x",
        valid_from=datetime(2026, 2, 4, 12, 0),
        data={"key": "CILAB-5678"},
    )

    # Create different relationship types
    kg.insert_relationship(
        rel_type="job_uses_component", source_id=job.id, target_id=comp.id, valid_from=datetime(2026, 2, 4, 10, 0)
    )

    kg.insert_relationship(
        rel_type="job_references_ticket", source_id=job.id, target_id=ticket.id, valid_from=datetime(2026, 2, 4, 12, 0)
    )

    # Get only component relationships
    components = kg.get_related_entities(job.id, rel_type="job_uses_component", direction="outgoing")
    assert len(components) == 1
    assert components[0][1].id == "comp-x"

    # Get only ticket relationships
    tickets = kg.get_related_entities(job.id, rel_type="job_references_ticket", direction="outgoing")
    assert len(tickets) == 1
    assert tickets[0][1].id == "ticket-x"


def test_get_stats(kg):
    """Test getting knowledge graph statistics"""
    # Initially empty
    stats = kg.get_stats()
    assert stats["total_entities"] == 0
    assert stats["total_relationships"] == 0

    # Add some entities
    job = kg.insert_entity(
        entity_type="dci_job", entity_id="job-1", valid_from=datetime(2026, 2, 4, 10, 0), data={"status": "failure"}
    )

    kg.insert_entity(
        entity_type="dci_job", entity_id="job-2", valid_from=datetime(2026, 2, 4, 11, 0), data={"status": "success"}
    )

    comp = kg.insert_entity(
        entity_type="component", entity_id="comp-1", valid_from=datetime(2026, 2, 4, 0, 0), data={"type": "ocp"}
    )

    # Add relationship
    kg.insert_relationship(
        rel_type="job_uses_component", source_id=job.id, target_id=comp.id, valid_from=datetime(2026, 2, 4, 10, 0)
    )

    stats = kg.get_stats()
    assert stats["total_entities"] == 3
    assert stats["entities_by_type"]["dci_job"] == 2
    assert stats["entities_by_type"]["component"] == 1
    assert stats["total_relationships"] == 1
    assert stats["relationships_by_type"]["job_uses_component"] == 1


def test_entity_to_dict(kg):
    """Test entity serialization to dictionary"""
    entity = kg.insert_entity(
        entity_type="dci_job",
        entity_id="job-dict",
        valid_from=datetime(2026, 2, 4, 10, 0),
        tx_from=datetime(2026, 2, 4, 11, 0),
        data={"status": "failure"},
    )

    entity_dict = entity.to_dict()
    assert entity_dict["id"] == "job-dict"
    assert entity_dict["entity_type"] == "dci_job"
    assert entity_dict["valid_from"] == "2026-02-04T10:00:00"
    assert entity_dict["tx_from"] == "2026-02-04T11:00:00"
    assert entity_dict["data"]["status"] == "failure"


def test_relationship_to_dict(kg):
    """Test relationship serialization to dictionary"""
    job = kg.insert_entity(
        entity_type="dci_job", entity_id="job-rel", valid_from=datetime(2026, 2, 4, 10, 0), data={"status": "failure"}
    )

    comp = kg.insert_entity(
        entity_type="component", entity_id="comp-rel", valid_from=datetime(2026, 2, 4, 0, 0), data={"type": "ocp"}
    )

    rel = kg.insert_relationship(
        rel_type="job_uses_component",
        source_id=job.id,
        target_id=comp.id,
        valid_from=datetime(2026, 2, 4, 10, 0),
        properties={"build": "ga"},
    )

    rel_dict = rel.to_dict()
    assert rel_dict["rel_type"] == "job_uses_component"
    assert rel_dict["source_id"] == "job-rel"
    assert rel_dict["target_id"] == "comp-rel"
    assert rel_dict["properties"]["build"] == "ga"


def test_context_manager(kg):
    """Test that knowledge graph can be used as context manager"""
    # Note: kg fixture already provides a graph, so we create a new one
    with KnowledgeGraph(db_path=":memory:") as graph:
        entity = graph.insert_entity(entity_type="test", valid_from=datetime.now(), data={"test": "data"})
        assert entity is not None

    # Connection should be closed after context exit
    # (We can't directly test this without accessing internals)


def test_get_all_current_entities(kg):
    """get_all_current_entities returns only tx_to IS NULL entities"""
    # Insert a current entity
    kg.insert_entity(
        entity_type="dci_job",
        entity_id="job-current",
        valid_from=datetime(2026, 2, 4, 10, 0),
        data={"status": "failure"},
    )
    # Insert an expired entity (tx_to is set)
    kg.insert_entity(
        entity_type="dci_job",
        entity_id="job-expired",
        valid_from=datetime(2026, 2, 4, 10, 0),
        tx_to=datetime(2026, 2, 4, 11, 0),
        data={"status": "failure"},
    )
    # Insert another current entity
    kg.insert_entity(
        entity_type="component",
        entity_id="comp-current",
        valid_from=datetime(2026, 2, 4, 0, 0),
        data={"type": "ocp", "version": "4.19.0"},
    )

    entities = kg.get_all_current_entities()
    entity_ids = {e.id for e in entities}
    assert "job-current" in entity_ids
    assert "comp-current" in entity_ids
    assert "job-expired" not in entity_ids
    assert len(entities) == 2


def test_get_all_current_relationships(kg):
    """get_all_current_relationships returns only tx_to IS NULL relationships"""
    job = kg.insert_entity(
        entity_type="dci_job",
        entity_id="job-1",
        valid_from=datetime(2026, 2, 4, 10, 0),
        data={"status": "failure"},
    )
    comp = kg.insert_entity(
        entity_type="component",
        entity_id="comp-1",
        valid_from=datetime(2026, 2, 4, 0, 0),
        data={"type": "ocp"},
    )
    ticket = kg.insert_entity(
        entity_type="jira_ticket",
        entity_id="ticket-1",
        valid_from=datetime(2026, 2, 4, 12, 0),
        data={"key": "CILAB-1234"},
    )

    # Insert a current relationship
    kg.insert_relationship(
        rel_type="job_uses_component",
        source_id=job.id,
        target_id=comp.id,
        valid_from=datetime(2026, 2, 4, 10, 0),
    )
    # Insert an expired relationship (tx_to is set)
    kg.insert_relationship(
        rel_type="job_references_ticket",
        source_id=job.id,
        target_id=ticket.id,
        valid_from=datetime(2026, 2, 4, 12, 0),
        tx_to=datetime(2026, 2, 4, 13, 0),
    )

    relationships = kg.get_all_current_relationships()
    assert len(relationships) == 1
    assert relationships[0].rel_type == "job_uses_component"


def test_discovery_lag_scenario(kg):
    """Test scenario: identify jobs discovered late"""
    # Job failed at 10:00, but we discovered it at 10:45 (45 min lag)
    _job_late = kg.insert_entity(
        entity_type="dci_job",
        entity_id="job-late-discovery",
        valid_from=datetime(2026, 2, 4, 10, 0),
        valid_to=datetime(2026, 2, 4, 10, 15),  # 15 min duration
        tx_from=datetime(2026, 2, 4, 10, 45),  # Discovered 45 min after start
        data={"status": "failure"},
    )

    # Job failed at 11:00, discovered at 11:05 (5 min lag)
    _job_quick = kg.insert_entity(
        entity_type="dci_job",
        entity_id="job-quick-discovery",
        valid_from=datetime(2026, 2, 4, 11, 0),
        valid_to=datetime(2026, 2, 4, 11, 10),
        tx_from=datetime(2026, 2, 4, 11, 5),
        data={"status": "failure"},
    )

    # Query all jobs
    all_jobs = kg.query_as_of(datetime(2026, 2, 4, 12, 0), entity_type="dci_job")

    # Calculate discovery lag for each
    for job in all_jobs:
        lag = (job.tx_from - job.valid_from).total_seconds() / 60
        if job.id == "job-late-discovery":
            assert lag == 45  # 45 minutes
        elif job.id == "job-quick-discovery":
            assert lag == 5  # 5 minutes


def test_conversation_entity_insert_and_query(kg):
    """Test inserting and querying conversation entities"""
    now = datetime.now()
    kg.insert_entity(
        entity_type="conversation",
        data={"user": "What are the failing DCI jobs?", "assistant": "Here are 3 failing jobs..."},
        valid_from=now,
        tx_from=now,
    )
    kg.insert_entity(
        entity_type="conversation",
        data={"user": "Show me CILAB tickets", "assistant": "Found 5 open tickets..."},
        valid_from=now,
        tx_from=now,
    )

    results = kg.query_as_of(now, entity_type="conversation")
    assert len(results) == 2


def test_query_as_of_with_search_text(kg):
    """Test text search within entity data"""
    now = datetime.now()
    kg.insert_entity(
        entity_type="conversation",
        data={"user": "What are the failing DCI jobs?", "assistant": "Here are 3 failing jobs..."},
        valid_from=now,
        tx_from=now,
    )
    kg.insert_entity(
        entity_type="conversation",
        data={"user": "Show me CILAB tickets", "assistant": "Found 5 open tickets..."},
        valid_from=now,
        tx_from=now,
    )
    kg.insert_entity(
        entity_type="dci_job",
        data={"status": "failure", "name": "test-job"},
        valid_from=now,
        tx_from=now,
    )

    # Search for "DCI" in conversations only
    results = kg.query_as_of(now, entity_type="conversation", search_text="DCI")
    assert len(results) == 1
    assert "DCI" in results[0].data["user"]

    # Search for "CILAB" across all types
    results = kg.query_as_of(now, search_text="CILAB")
    assert len(results) == 1

    # Case-insensitive search
    results = kg.query_as_of(now, entity_type="conversation", search_text="dci")
    assert len(results) == 1


def test_query_as_of_with_valid_from_after(kg):
    """Test filtering entities by valid_from minimum time"""
    old_time = datetime(2026, 2, 1, 10, 0)
    recent_time = datetime(2026, 2, 17, 10, 0)
    query_time = datetime(2026, 2, 17, 12, 0)

    # Insert old conversation
    kg.insert_entity(
        entity_type="conversation",
        data={"user": "Old question", "assistant": "Old answer"},
        valid_from=old_time,
        tx_from=old_time,
    )

    # Insert recent conversation
    kg.insert_entity(
        entity_type="conversation",
        data={"user": "Recent question", "assistant": "Recent answer"},
        valid_from=recent_time,
        tx_from=recent_time,
    )

    # Query all conversations (no valid_from_after filter)
    results = kg.query_as_of(query_time, entity_type="conversation")
    assert len(results) == 2

    # Query only recent conversations (valid_from_after = Feb 16)
    cutoff = datetime(2026, 2, 16, 0, 0)
    results = kg.query_as_of(query_time, entity_type="conversation", valid_from_after=cutoff)
    assert len(results) == 1
    assert results[0].data["user"] == "Recent question"


def test_batch_mode_defers_commit(kg):
    """Batch mode defers commits until batch exits"""
    commit_count = 0
    original_maybe = kg._maybe_commit

    def counting_maybe():
        nonlocal commit_count
        commit_count += 1
        original_maybe()

    kg._maybe_commit = counting_maybe

    with kg.batch():
        kg.insert_entity(
            entity_type="test",
            entity_id="batch-1",
            valid_from=datetime(2026, 2, 18, 10, 0),
            data={"x": 1},
        )
        kg.insert_entity(
            entity_type="test",
            entity_id="batch-2",
            valid_from=datetime(2026, 2, 18, 10, 0),
            data={"x": 2},
        )
        # _maybe_commit was called but should not have committed (batch mode)
        assert commit_count == 2

    # Both entities should be readable after batch exits
    assert kg.get_entity("batch-1") is not None
    assert kg.get_entity("batch-2") is not None


def test_batch_mode_commits_on_exception(kg):
    """Batch mode commits even when exception occurs inside the block"""
    try:
        with kg.batch():
            kg.insert_entity(
                entity_type="test",
                entity_id="batch-ex",
                valid_from=datetime(2026, 2, 18, 10, 0),
                data={"x": 1},
            )
            raise ValueError("test error")
    except ValueError:
        pass

    # Entity should still be committed
    assert kg.get_entity("batch-ex") is not None


def test_non_batch_mode_commits_immediately(kg):
    """Without batch mode, each insert commits immediately (backward compat)"""
    # Insert first entity
    kg.insert_entity(
        entity_type="test",
        entity_id="immediate-1",
        valid_from=datetime(2026, 2, 18, 10, 0),
        data={"x": 1},
    )
    assert kg.get_entity("immediate-1") is not None

    # Insert second entity
    kg.insert_entity(
        entity_type="test",
        entity_id="immediate-2",
        valid_from=datetime(2026, 2, 18, 10, 0),
        data={"x": 2},
    )
    assert kg.get_entity("immediate-2") is not None

    # Verify _batch_mode is False by default
    assert kg._batch_mode is False


def test_relationship_exists(kg):
    """relationship_exists returns True for existing current relationships"""
    job = kg.insert_entity(
        entity_type="dci_job",
        entity_id="job-re",
        valid_from=datetime(2026, 2, 4, 10, 0),
        data={"status": "failure"},
    )
    comp = kg.insert_entity(
        entity_type="component",
        entity_id="comp-re",
        valid_from=datetime(2026, 2, 4, 0, 0),
        data={"type": "ocp"},
    )

    assert kg.relationship_exists("job_uses_component", job.id, comp.id) is False

    kg.insert_relationship(
        rel_type="job_uses_component",
        source_id=job.id,
        target_id=comp.id,
        valid_from=datetime(2026, 2, 4, 10, 0),
    )

    assert kg.relationship_exists("job_uses_component", job.id, comp.id) is True
    assert kg.relationship_exists("other_type", job.id, comp.id) is False
    assert kg.relationship_exists("job_uses_component", comp.id, job.id) is False


def test_init_raises_on_missing_extension_support():
    """KnowledgeGraph raises RuntimeError when sqlite3 lacks enable_load_extension."""
    mock_conn = MagicMock(spec=["execute", "commit", "close"])
    with patch("sqlite3.connect", return_value=mock_conn):
        with pytest.raises(RuntimeError, match="extension loading support"):
            KnowledgeGraph(db_path=":memory:")


# === Access Tracking ===


def test_record_access_single(kg):
    """Record a single entity access"""
    kg.insert_entity(entity_type="task", entity_id="t1", valid_from=datetime.now(), data={"x": 1})
    kg.record_access(["t1"], "test_context")

    cursor = kg.conn.execute("SELECT entity_id, query_context FROM knowledge_access")
    rows = cursor.fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "t1"
    assert rows[0][1] == "test_context"


def test_record_access_batch(kg):
    """Record access for multiple entities at once"""
    for i in range(3):
        kg.insert_entity(entity_type="task", entity_id=f"t{i}", valid_from=datetime.now(), data={"x": i})
    kg.record_access(["t0", "t1", "t2"], "batch_test")

    cursor = kg.conn.execute("SELECT COUNT(*) FROM knowledge_access")
    assert cursor.fetchone()[0] == 3


def test_record_access_empty_list(kg):
    """Empty entity list is a no-op"""
    kg.record_access([], "should_not_insert")
    cursor = kg.conn.execute("SELECT COUNT(*) FROM knowledge_access")
    assert cursor.fetchone()[0] == 0


def test_get_access_stats_empty(kg):
    """Access stats on empty KG returns zeroes"""
    stats = kg.get_access_stats()
    assert stats["total_accesses"] == 0
    assert stats["most_accessed"] == []
    assert stats["never_accessed"] == []


def test_get_access_stats_most_accessed(kg):
    """Most accessed entities appear in order"""
    kg.insert_knowledge("user_preference", "pref1", "content1", metadata={"tags": []})
    kg.insert_knowledge("user_preference", "pref2", "content2", metadata={"tags": []})

    kg.record_access(["user_preference:pref1"], "test")
    kg.record_access(["user_preference:pref1"], "test")
    kg.record_access(["user_preference:pref2"], "test")

    stats = kg.get_access_stats()
    assert stats["total_accesses"] == 3
    assert len(stats["most_accessed"]) == 2
    assert stats["most_accessed"][0]["entity_id"] == "user_preference:pref1"
    assert stats["most_accessed"][0]["access_count"] == 2


def test_get_access_stats_never_accessed(kg):
    """Knowledge entries with no access appear in never_accessed"""
    kg.insert_knowledge("lesson_learned", "lesson1", "content", metadata={"tags": []})
    kg.insert_knowledge("lesson_learned", "lesson2", "content", metadata={"tags": []})
    kg.record_access(["lesson_learned:lesson1"], "test")

    stats = kg.get_access_stats()
    never_ids = [e["entity_id"] for e in stats["never_accessed"]]
    assert "lesson_learned:lesson2" in never_ids
    assert "lesson_learned:lesson1" not in never_ids


def test_get_access_stats_stale(kg):
    """Entries accessed long ago appear as stale"""
    kg.insert_knowledge("user_preference", "old_pref", "old content", metadata={"tags": []})
    old_time = (datetime.now() - timedelta(days=30)).isoformat()
    kg.conn.execute(
        "INSERT INTO knowledge_access (entity_id, accessed_at, query_context) VALUES (?, ?, ?)",
        ("user_preference:old_pref", old_time, "old_test"),
    )
    kg.conn.commit()

    stats = kg.get_access_stats(stale_days=7)
    stale_ids = [e["entity_id"] for e in stats["stale_entries"]]
    assert "user_preference:old_pref" in stale_ids
