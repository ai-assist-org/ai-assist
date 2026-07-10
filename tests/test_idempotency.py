"""Tests for AWL idempotency store"""

import sqlite3
from datetime import datetime, timedelta

import pytest

from ai_assist.idempotency import (
    IdempotencyStore,
    compute_idempotency_key,
    compute_workflow_run_id,
    is_write_tool,
)


@pytest.fixture
def store(tmp_path):
    return IdempotencyStore(tmp_path / "idempotency.db")


class TestIsWriteTool:
    def test_create_tools_are_write(self):
        assert is_write_tool("dci__create_jira_ticket") is True
        assert is_write_tool("mcp__create_issue") is True

    def test_add_tools_are_write(self):
        assert is_write_tool("dci__add_jira_comment") is True
        assert is_write_tool("dci__add_jira_weblink") is True

    def test_update_tools_are_write(self):
        assert is_write_tool("dci__update_jira_ticket") is True

    def test_delete_tools_are_write(self):
        assert is_write_tool("some__delete_resource") is True

    def test_send_tools_are_write(self):
        assert is_write_tool("email__send_message") is True

    def test_post_tools_are_write(self):
        assert is_write_tool("api__post_data") is True

    def test_search_tools_are_read(self):
        assert is_write_tool("dci__search_jira_tickets") is False
        assert is_write_tool("dci__search_dci_jobs") is False

    def test_get_tools_are_read(self):
        assert is_write_tool("dci__get_jira_ticket") is False
        assert is_write_tool("dci__get_support_case") is False

    def test_list_tools_are_read(self):
        assert is_write_tool("dci__list_jira_boards") is False

    def test_count_tools_are_read(self):
        assert is_write_tool("dci__count_jira_tickets") is False

    def test_query_tools_are_read(self):
        assert is_write_tool("dci__query_dci_components") is False

    def test_find_tools_are_read(self):
        assert is_write_tool("dci__find_folder_by_name") is False

    def test_unknown_tools_are_read_by_default(self):
        assert is_write_tool("some__do_something") is False

    def test_internal_tools_are_never_write(self):
        assert is_write_tool("internal__write_report") is False
        assert is_write_tool("internal__write_file") is False
        assert is_write_tool("internal__create_action") is False

    def test_introspection_tools_are_never_write(self):
        assert is_write_tool("introspection__update_something") is False

    def test_goal_tools_are_never_write(self):
        assert is_write_tool("goal__update_goal") is False


class TestComputeIdempotencyKey:
    def test_deterministic(self):
        key1 = compute_idempotency_key("run1", "task1", "create_ticket", {"summary": "test"})
        key2 = compute_idempotency_key("run1", "task1", "create_ticket", {"summary": "test"})
        assert key1 == key2

    def test_different_run_id_different_key(self):
        key1 = compute_idempotency_key("run1", "task1", "tool", {"a": 1})
        key2 = compute_idempotency_key("run2", "task1", "tool", {"a": 1})
        assert key1 != key2

    def test_different_task_id_different_key(self):
        key1 = compute_idempotency_key("run1", "task_a", "tool", {"a": 1})
        key2 = compute_idempotency_key("run1", "task_b", "tool", {"a": 1})
        assert key1 != key2

    def test_different_args_different_key(self):
        key1 = compute_idempotency_key("run1", "task1", "tool", {"a": 1})
        key2 = compute_idempotency_key("run1", "task1", "tool", {"a": 2})
        assert key1 != key2

    def test_arg_order_does_not_matter(self):
        key1 = compute_idempotency_key("run1", "task1", "tool", {"b": 2, "a": 1})
        key2 = compute_idempotency_key("run1", "task1", "tool", {"a": 1, "b": 2})
        assert key1 == key2


class TestComputeWorkflowRunId:
    def test_deterministic(self):
        rid1 = compute_workflow_run_id("/path/to/script.awl", {"x": "1"})
        rid2 = compute_workflow_run_id("/path/to/script.awl", {"x": "1"})
        assert rid1 == rid2

    def test_different_path_different_id(self):
        rid1 = compute_workflow_run_id("/a.awl", {})
        rid2 = compute_workflow_run_id("/b.awl", {})
        assert rid1 != rid2

    def test_different_vars_different_id(self):
        rid1 = compute_workflow_run_id("/a.awl", {"x": "1"})
        rid2 = compute_workflow_run_id("/a.awl", {"x": "2"})
        assert rid1 != rid2

    def test_none_vars_handled(self):
        rid = compute_workflow_run_id("/a.awl", None)
        assert isinstance(rid, str) and len(rid) > 0


class TestIdempotencyStore:
    def test_get_unknown_key_returns_none(self, store):
        assert store.get("nonexistent") is None

    def test_put_and_get_roundtrip(self, store):
        store.put(
            key="k1",
            tool_name="create_ticket",
            arguments={"summary": "test"},
            result="PROJ-123",
            is_error=False,
            workflow_run_id="run1",
            task_id="task1",
        )
        assert store.get("k1") == "PROJ-123"

    def test_get_skips_error_results(self, store):
        store.put(
            key="k_err",
            tool_name="create_ticket",
            arguments={},
            result="Error: connection refused",
            is_error=True,
            workflow_run_id="run1",
            task_id="task1",
        )
        assert store.get("k_err") is None

    def test_put_overwrites_existing(self, store):
        store.put("k1", "tool", {}, "result1", False, "run1", "t1")
        store.put("k1", "tool", {}, "result2", False, "run1", "t1")
        assert store.get("k1") == "result2"

    def test_clear_workflow_removes_matching(self, store):
        store.put("k1", "tool", {}, "r1", False, "run_a", "t1")
        store.put("k2", "tool", {}, "r2", False, "run_b", "t1")
        removed = store.clear_workflow("run_a")
        assert removed == 1
        assert store.get("k1") is None
        assert store.get("k2") == "r2"

    def test_cleanup_removes_old_entries(self, store):
        store.put("k_old", "tool", {}, "old", False, "run1", "t1")
        # Manually set created_at to 10 days ago
        conn = sqlite3.connect(store._db_path)
        old_date = (datetime.now() - timedelta(days=10)).isoformat()
        conn.execute("UPDATE idempotent_results SET created_at = ? WHERE key = ?", (old_date, "k_old"))
        conn.commit()
        conn.close()

        store.put("k_new", "tool", {}, "new", False, "run1", "t1")
        removed = store.cleanup(max_age_days=7)
        assert removed == 1
        assert store.get("k_old") is None
        assert store.get("k_new") == "new"

    def test_db_created_on_first_use(self, tmp_path):
        db_path = tmp_path / "sub" / "idempotency.db"
        store = IdempotencyStore(db_path)
        store.put("k1", "tool", {}, "result", False, "run1", "t1")
        assert db_path.exists()
        assert store.get("k1") == "result"
