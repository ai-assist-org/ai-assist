"""Runtime identity and single-owner leases for MCP server processes."""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MCPRuntimeIdentity:
    """The invocation context used to select MCP server scopes."""

    mode: str
    session_name: str | None = None

    def describe(self) -> str:
        if self.session_name:
            return f"{self.mode} session '{self.session_name}'"
        return self.mode


class MCPServerLease:
    """A non-blocking, config-directory-local exclusive MCP ownership lease."""

    def __init__(self, config_dir: Path, server_name: str, runtime: MCPRuntimeIdentity):
        self.path = config_dir / "state" / "mcp-leases" / f"{server_name}.json"
        self.server_name = server_name
        self.runtime = runtime
        self._handle: Any = None

    def acquire(self) -> dict[str, Any] | None:
        """Acquire this lease, or return the current owner's safe metadata."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return self.read_owner(self.path)

        metadata = {
            "server": self.server_name,
            "pid": os.getpid(),
            "started_at": datetime.now(UTC).isoformat(),
            "mode": self.runtime.mode,
            "session_name": self.runtime.session_name,
            "config_dir": str(self.path.parent.parent.parent),
        }
        handle.seek(0)
        handle.truncate()
        json.dump(metadata, handle)
        handle.flush()
        os.fsync(handle.fileno())
        self._handle = handle
        return None

    def release(self) -> None:
        if self._handle is None:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None

    @staticmethod
    def read_owner(path: Path) -> dict[str, Any]:
        """Return best-effort metadata for the holder of a lease file."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except FileNotFoundError, OSError, json.JSONDecodeError:
            return {}

    @classmethod
    def current_owner(cls, path: Path) -> dict[str, Any]:
        """Return metadata only when a process currently holds the lease."""
        if not path.exists():
            return {}
        try:
            handle = path.open("r+", encoding="utf-8")
        except OSError:
            return {}
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return cls.read_owner(path)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        return {}


def format_lease_owner(owner: dict[str, Any]) -> str:
    """Format safe lease metadata for a human-facing diagnostic."""
    if not owner:
        return "another process"
    description = str(owner.get("mode", "another process"))
    if owner.get("session_name"):
        description += f" session '{owner['session_name']}'"
    if owner.get("pid"):
        description += f" (PID {owner['pid']})"
    return description
