"""Background task spawning for interactive sessions"""

import asyncio
import json
import logging
import os
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# True inside a background task's own asyncio task (each task runs in a copy of
# the context): tells its security questions apart from the foreground's.
_in_background_task: ContextVar[bool] = ContextVar("in_background_task", default=False)

# Characters of a finished task's result handed to the agent; the rest is one
# internal__get_background_task call away.
_NOTE_RESULT_CHARS = 3000


def in_background_task() -> bool:
    """Whether the code runs inside a background task (nobody can answer its questions)."""
    return _in_background_task.get()


class BackgroundTask(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str
    prompt: str
    status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: str | None = None
    result_report: str | None = None
    error: str | None = None
    # Hand the result to the agent when it finishes (see take_finished_note)
    notify_agent: bool = True


class BackgroundTaskManager:
    def __init__(self, agent, console, notifications_file: Path) -> None:
        self.agent = agent
        self.console = console
        self.notifications_file = notifications_file
        self._tasks: dict[str, BackgroundTask] = {}
        self._asyncio_tasks: dict[str, asyncio.Task] = {}
        # Finished tasks not yet handed to the agent, and the hook waking it up
        self._finished: list[BackgroundTask] = []
        self.on_finished: Callable[[], None] | None = None

    async def spawn_task(
        self,
        prompt: str,
        description: str,
        save_to_report: str | None = None,
        max_turns: int = 50,
        max_time_seconds: int = 300,
        notify_agent: bool = True,
    ) -> BackgroundTask:
        """Run a query in the background.

        notify_agent: hand the result to the agent when it finishes, waking it
        if idle. False for tasks that deliver their result themselves (e.g. an
        MCP channel reply).
        """
        task = BackgroundTask(description=description, prompt=prompt, notify_agent=notify_agent)
        if save_to_report:
            task.result_report = save_to_report
        self._tasks[task.id] = task

        asyncio_task = asyncio.create_task(self._run_task(task, save_to_report, max_turns, max_time_seconds))
        self._asyncio_tasks[task.id] = asyncio_task
        return task

    async def _run_task(
        self,
        task: BackgroundTask,
        save_to_report: str | None,
        max_turns: int,
        max_time_seconds: int,
    ) -> None:
        task.status = "running"
        task.started_at = datetime.now()
        _in_background_task.set(True)  # this asyncio task's own context

        self.agent._background_task_count += 1
        try:
            # The agent must know it is the background task: otherwise it reports
            # having "done it directly instead of in the background".
            context = (
                f'[You are background task {task.id} ("{task.description}"), started by the main '
                "conversation. Do the work yourself, directly: your final answer is handed back to the "
                "main conversation. Nobody can answer questions or approve commands while you run.]"
            )
            prefixed_prompt = f"@no-kg @no-history {context}\n\n{task.prompt}"
            result = await self.agent.query(
                prefixed_prompt,
                max_turns=max_turns,
                max_time_seconds=max_time_seconds,
                raise_on_api_error=True,
            )
            task.status = "completed"
            task.result = result
            task.completed_at = datetime.now()

            if save_to_report and hasattr(self.agent, "report_tools"):
                try:
                    self.agent.report_tools._write_report(save_to_report, result)
                except Exception:
                    logger.exception("Failed to save background task result to report")

            await self._write_notification(task, "success")
            self._hand_to_agent(task)
        except asyncio.CancelledError:
            task.status = "failed"
            task.error = "Task was cancelled"
            task.completed_at = datetime.now()
            await self._write_notification(task, "warning")
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.completed_at = datetime.now()
            await self._write_notification(task, "error")
            self._hand_to_agent(task)
        finally:
            self.agent._background_task_count -= 1
            self._asyncio_tasks.pop(task.id, None)

    async def _write_notification(self, task: BackgroundTask, level: str) -> None:
        self.notifications_file.parent.mkdir(parents=True, exist_ok=True)
        if task.status == "completed":
            body = task.result[:500] if task.result else "No output"
            if task.result_report and hasattr(self.agent, "report_tools"):
                report_file = self.agent.report_tools.reports_dir / f"{task.result_report}.md"
                body += f"\n\nFull result saved to report: file://{report_file}"
        else:
            body = task.error or "Unknown error"

        notification = {
            "timestamp": datetime.now().isoformat(),
            "action_name": f"background:{task.id}",
            "title": f"Background: {task.description}",
            "message": body,
            "level": level,
            "channels": ["file"],
            "pid": os.getpid(),
        }
        with open(self.notifications_file, "a") as f:
            f.write(json.dumps(notification) + "\n")

    def _hand_to_agent(self, task: BackgroundTask) -> None:
        """Queue a finished task for the agent and wake it (cancelled tasks: it cancelled them)."""
        if not task.notify_agent:
            return
        self._finished.append(task)
        if self.on_finished is not None:
            try:
                self.on_finished()
            except Exception:
                logger.exception("Background task wake-up hook failed")

    def has_finished(self) -> bool:
        """Whether finished tasks are waiting to be handed to the agent."""
        return bool(self._finished)

    def take_finished_note(self) -> str | None:
        """The results of the tasks finished since the last call, for the agent; None if none."""
        if not self._finished:
            return None
        tasks, self._finished = self._finished, []
        lines = ["[Background task results, delivered automatically: no need to poll internal__list_background_tasks]"]
        for task in tasks:
            lines.append(f'\n## Task {task.id} "{task.description}": {task.status}')
            if task.status == "completed":
                result = task.result or "No output"
                lines.append(result[:_NOTE_RESULT_CHARS])
                if len(result) > _NOTE_RESULT_CHARS:
                    lines.append(
                        f"[{len(result) - _NOTE_RESULT_CHARS} more characters: "
                        f'internal__get_background_task(task_id="{task.id}")]'
                    )
                if task.result_report and hasattr(self.agent, "report_tools"):
                    lines.append(f"Full result saved to report: {task.result_report}")
            else:
                lines.append(f"Error: {task.error or 'Unknown error'}")
        return "\n".join(lines)

    def list_tasks(self, status_filter: str | None = None) -> list[BackgroundTask]:
        tasks = list(self._tasks.values())
        if status_filter:
            tasks = [t for t in tasks if t.status == status_filter]
        return tasks

    def get_task(self, task_id: str) -> BackgroundTask | None:
        return self._tasks.get(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        asyncio_task = self._asyncio_tasks.get(task_id)
        if asyncio_task is None:
            return False
        asyncio_task.cancel()
        try:
            await asyncio_task
        except asyncio.CancelledError:
            pass
        return True

    async def cancel_all(self) -> int:
        task_ids = list(self._asyncio_tasks.keys())
        for task_id in task_ids:
            await self.cancel_task(task_id)
        return len(task_ids)

    def cleanup_old_tasks(self, max_age_hours: int = 24) -> None:
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = [tid for tid, t in self._tasks.items() if t.created_at < cutoff and tid not in self._asyncio_tasks]
        for tid in to_remove:
            del self._tasks[tid]
