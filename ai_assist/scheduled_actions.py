"""Scheduled actions system for one-shot future executions"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ScheduledAction(BaseModel):
    """A one-time scheduled action to execute in the future"""

    model_config = ConfigDict(
        ser_json_timedelta="iso8601",
    )

    id: str
    prompt: str  # Action to execute (natural language or mcp://)
    scheduled_at: datetime  # When to execute
    created_at: datetime  # When scheduled
    created_by: str  # "user" or "agent"
    description: str | None = None  # Human-readable description

    # Execution configuration
    execute_query: bool = False  # Execute prompt via agent (False = simple reminder)

    # Output configuration (independent flags)
    notify: bool = True  # Send notification when complete
    create_report: bool = False  # Create report/save results
    notification_channels: list[str] = Field(default_factory=lambda: ["console", "desktop"])

    status: str = "pending"  # "pending", "executing", "completed", "failed"
    result: str | None = None  # Execution result
    executed_at: datetime | None = None
    pid: int = Field(default_factory=os.getpid)


class ScheduledActionManager:
    """Manages scheduled actions - loading, saving, and execution"""

    def __init__(self, action_file: Path, agent):
        self.action_file = action_file
        self.agent = agent
        self.actions: list[ScheduledAction] = []
        self._executor_event: asyncio.Event | None = None

    async def load_actions(self) -> list[ScheduledAction]:
        """Load scheduled actions from JSON file"""
        if not self.action_file.exists():
            return []

        try:
            with open(self.action_file) as f:
                data = json.load(f)

            self.actions = []
            for item in data.get("actions", []):
                # Parse datetime strings
                if isinstance(item.get("scheduled_at"), str):
                    item["scheduled_at"] = datetime.fromisoformat(item["scheduled_at"])
                if isinstance(item.get("created_at"), str):
                    item["created_at"] = datetime.fromisoformat(item["created_at"])
                if isinstance(item.get("executed_at"), str):
                    item["executed_at"] = datetime.fromisoformat(item["executed_at"])

                self.actions.append(ScheduledAction(**item))

            return self.actions

        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Error loading scheduled actions: %s", e)
            return []

    async def save_action(self, action: ScheduledAction):
        """Save a new scheduled action"""
        # Load existing actions first
        await self.load_actions()

        # Add new action
        self.actions.append(action)

        # Persist
        await self._persist()

    async def get_action(self, action_id: str) -> ScheduledAction | None:
        """Get a specific action by ID"""
        await self.load_actions()

        for action in self.actions:
            if action.id == action_id:
                return action

        return None

    async def _persist(self):
        """Write actions to JSON file"""
        # Ensure directory exists
        self.action_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "actions": [
                {
                    **a.model_dump(mode="json"),
                    "scheduled_at": a.scheduled_at.isoformat(),
                    "created_at": a.created_at.isoformat(),
                    "executed_at": a.executed_at.isoformat() if a.executed_at else None,
                }
                for a in self.actions
            ],
        }

        with open(self.action_file, "w") as f:
            json.dump(data, f, indent=2)

    async def execute_due_actions(self):
        """Execute all actions that are due"""
        await self.load_actions()
        now = datetime.now()

        for action in self.actions:
            if action.status != "pending":
                continue

            if action.scheduled_at <= now:
                await self._execute_action(action)

    async def _execute_action(self, action: ScheduledAction):
        """Execute a single scheduled action"""
        print(f"🔔 Executing scheduled action: {action.description or action.prompt[:50]}")

        action.status = "executing"
        await self._persist()

        try:
            # Check if we need to execute a query via the agent
            if action.execute_query:
                # Execute via agent to get results
                result = await self.agent.query(action.prompt)

                action.status = "completed"
                action.result = result
                action.executed_at = datetime.now()

                # Dispatch notification if requested
                if action.notify:
                    await self._notify_completion(action)
            else:
                # Simple reminder - just send notification without querying agent
                action.status = "completed"
                action.result = action.prompt  # Use prompt as the reminder message
                action.executed_at = datetime.now()

                # Dispatch notification
                if action.notify:
                    await self._notify_completion(action)

        except Exception as e:
            action.status = "failed"
            action.result = str(e)
            action.executed_at = datetime.now()

            # Notify about failure
            if action.notify:
                await self._notify_completion(action)

        await self._persist()

        # Run cleanup after every execution
        await self.cleanup_old_actions(max_age_days=7)

    async def cleanup_old_actions(self, max_age_days: int = 7):
        """Archive completed/failed actions older than max_age_days

        Archives to scheduled-actions-archive.jsonl (append-only JSONL)
        Keeps pending actions regardless of age
        """
        archive_file = self.action_file.parent / "scheduled-actions-archive.jsonl"
        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        # Find actions to archive
        to_archive = []
        to_keep = []

        for action in self.actions:
            # Never archive pending actions
            if action.status == "pending":
                to_keep.append(action)
                continue

            # Archive completed/failed actions older than cutoff
            if action.executed_at and action.executed_at < cutoff_date:
                to_archive.append(action)
            else:
                to_keep.append(action)

        # If nothing to archive, skip
        if not to_archive:
            return 0

        # Append to archive (JSONL format - one JSON object per line)
        archive_file.parent.mkdir(parents=True, exist_ok=True)
        with open(archive_file, "a") as f:
            for action in to_archive:
                json_line = action.model_dump_json()
                f.write(json_line + "\n")

        # Update in-memory list
        self.actions = to_keep

        # Persist cleaned JSON
        await self._persist()

        print(f"Archived {len(to_archive)} old actions to {archive_file.name}")
        return len(to_archive)

    async def _notify_completion(self, action: ScheduledAction):
        """Send notification when action completes"""
        from ai_assist.notification_dispatcher import Notification, NotificationDispatcher

        # Determine notification level
        level = "success" if action.status == "completed" else "error"

        # Format notification based on whether we executed a query
        if not action.execute_query and action.status == "completed":
            # Simple reminder - show the prompt directly
            title = "⏰ Reminder"
            message = action.result if action.result else action.prompt
        else:
            # Query result - show full result (channels will handle their own truncation)
            title = f"Scheduled Action {'Completed' if action.status == 'completed' else 'Failed'}"
            message = (
                f"{action.description or action.prompt}\n\nResult: {action.result if action.result else 'No result'}"
            )

        # Create notification
        notification = Notification(
            id=f"notif-{action.id}",
            action_id=action.id,
            title=title,
            message=message,
            level=level,
            timestamp=datetime.now(),
            channels=action.notification_channels,
            delivered={},
        )

        # Dispatch
        dispatcher = NotificationDispatcher()
        await dispatcher.dispatch(notification)

    def _calculate_next_execution_time(self) -> datetime | None:
        """Calculate when the next pending action should execute"""
        pending_actions = [a for a in self.actions if a.status == "pending"]

        if not pending_actions:
            print("No pending actions")
            return None

        # Return earliest scheduled time
        next_time = min(a.scheduled_at for a in pending_actions)
        time_until = (next_time - datetime.now()).total_seconds()
        print(f"Next action in {time_until:.1f}s at {next_time.strftime('%H:%M:%S')}")
        return next_time

    async def on_file_change(self):
        """Called when scheduled-actions.json changes (FileWatchdog callback)"""
        print("Scheduled actions file changed, reloading...")
        await self.load_actions()
        print(
            f"Loaded {len(self.actions)} total actions, {len([a for a in self.actions if a.status == 'pending'])} pending"
        )

        # Wake up executor to check new actions
        if self._executor_event is not None:
            print("Waking up executor...")
            self._executor_event.set()
        else:
            logger.warning("Executor event not initialized yet")

    async def start_executor(self):
        """Start event-driven executor (no polling)"""
        print("Starting scheduled action executor (event-driven)")

        # Event to wake up executor on file changes
        self._executor_event = asyncio.Event()

        while True:
            try:
                # Execute any due actions now
                await self.execute_due_actions()

                # Calculate when to wake up next
                next_time = self._calculate_next_execution_time()

                if next_time is None:
                    # No pending actions - wait for file changes only
                    print("Waiting for file changes...")
                    await self._executor_event.wait()
                    self._executor_event.clear()
                    print("Woke up from file change")
                else:
                    # Sleep until next action is due OR file changes (whichever comes first)
                    sleep_seconds = (next_time - datetime.now()).total_seconds()

                    if sleep_seconds > 0:
                        print(f"Sleeping for {sleep_seconds:.1f}s until next action...")
                        try:
                            await asyncio.wait_for(self._executor_event.wait(), timeout=sleep_seconds)
                            self._executor_event.clear()
                            print("Woke up from file change")
                        except TimeoutError:
                            # Timeout means we reached scheduled time
                            print("Woke up from timeout (scheduled time reached)")

            except asyncio.CancelledError:
                print("Scheduled action executor stopped")
                break
            except Exception as e:
                logger.exception("Error in scheduled action executor: %s", e)
                await asyncio.sleep(60)  # Wait a minute before retrying
