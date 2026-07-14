"""Idempotency store for AWL tool calls — prevents duplicate side effects on re-run."""

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_WRITE_PREFIXES = ("create_", "add_", "update_", "delete_", "send_", "post_")
_LOCAL_SERVERS = ("internal", "introspection", "goal")


def is_write_tool(tool_name: str) -> bool:
    """Classify a tool as a remote write whose result is safe to cache.

    Local tools (internal__, introspection__, goal__) are excluded because
    their output depends on runtime context, not just their arguments.
    """
    parts = tool_name.split("__", 1)
    if len(parts) == 2 and parts[0] in _LOCAL_SERVERS:
        return False
    base = parts[-1]
    return any(base.startswith(p) for p in _WRITE_PREFIXES)


def compute_idempotency_key(workflow_run_id: str, task_id: str, tool_name: str, arguments: dict[str, Any]) -> str:
    args_hash = hashlib.md5(
        json.dumps(arguments, sort_keys=True, default=str).encode(),
        usedforsecurity=False,
    ).hexdigest()[:12]
    return f"awl:{workflow_run_id}:{task_id}:{tool_name}:{args_hash}"


def compute_workflow_run_id(script_path: str, variables: dict[str, Any] | None) -> str:
    data = f"{script_path}:{json.dumps(variables or {}, sort_keys=True, default=str)}"
    return hashlib.md5(data.encode(), usedforsecurity=False).hexdigest()[:12]


class IdempotencyStore:
    """SQLite-backed cache for write-oriented tool call results."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS idempotent_results (
                key TEXT PRIMARY KEY,
                tool_name TEXT NOT NULL,
                arguments_json TEXT NOT NULL,
                result_text TEXT NOT NULL,
                is_error INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                workflow_run_id TEXT,
                task_id TEXT
            )""")
        conn.commit()
        conn.close()

    def get(self, key: str) -> str | None:
        """Return cached result, or None if not found or was an error."""
        conn = sqlite3.connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT result_text FROM idempotent_results WHERE key = ? AND is_error = 0",
                (key,),
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def put(
        self,
        key: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: str,
        is_error: bool,
        workflow_run_id: str,
        task_id: str,
    ) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                """INSERT OR REPLACE INTO idempotent_results
                   (key, tool_name, arguments_json, result_text, is_error, created_at, workflow_run_id, task_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    key,
                    tool_name,
                    json.dumps(arguments, sort_keys=True, default=str),
                    result,
                    int(is_error),
                    datetime.now().isoformat(),
                    workflow_run_id,
                    task_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def clear_workflow(self, workflow_run_id: str) -> int:
        """Remove all cached results for a workflow run. Returns count removed."""
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute(
                "DELETE FROM idempotent_results WHERE workflow_run_id = ?",
                (workflow_run_id,),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def cleanup(self, max_age_days: int = 7) -> int:
        """Remove entries older than max_age_days. Returns count removed."""
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute(
                "DELETE FROM idempotent_results WHERE created_at < ?",
                (cutoff,),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()
