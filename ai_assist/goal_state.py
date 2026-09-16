"""Goal state persistence for AWL @goal directives"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GoalState:
    """Persisted state for a single goal between cycles"""

    status: str = "active"  # active, paused, completed, cancelled
    variables: dict[str, Any] = field(default_factory=dict)
    last_run: str | None = None
    cycle_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    success_reason: str | None = None


class GoalStateManager:
    """Read/write goal sidecar state files"""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def state_path(self, goal_id: str) -> Path:
        return self.state_dir / f"goal_{goal_id}.json"

    def load(self, goal_id: str) -> GoalState:
        """Load goal state from sidecar file, or return default"""
        path = self.state_path(goal_id)
        if not path.exists():
            return GoalState()

        try:
            with open(path) as f:
                data = json.load(f)
            return GoalState(
                status=data.get("status", "active"),
                variables=data.get("variables", {}),
                last_run=data.get("last_run"),
                cycle_count=data.get("cycle_count", 0),
                created_at=data.get("created_at", datetime.now().isoformat()),
                success_reason=data.get("success_reason"),
            )
        except json.JSONDecodeError, ValueError:
            logger.exception("Error loading goal state for %s", goal_id)
            return GoalState()

    def save(self, goal_id: str, state: GoalState):
        """Save goal state to sidecar file"""
        path = self.state_path(goal_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "status": state.status,
            "variables": state.variables,
            "last_run": state.last_run,
            "cycle_count": state.cycle_count,
            "created_at": state.created_at,
            "success_reason": state.success_reason,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def list_all(self) -> list[tuple[str, GoalState]]:
        """List all goal states"""
        results = []
        for path in sorted(self.state_dir.glob("goal_*.json")):
            goal_id = path.stem.removeprefix("goal_")
            state = self.load(goal_id)
            results.append((goal_id, state))
        return results
