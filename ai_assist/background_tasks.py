"""Background task spawning for interactive sessions"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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


class BackgroundTaskManager:
    def __init__(self, agent, console, notifications_file: Path) -> None:
        self.agent = agent
        self.console = console
        self.notifications_file = notifications_file
        self._tasks: dict[str, BackgroundTask] = {}
        self._asyncio_tasks: dict[str, asyncio.Task] = {}

    async def spawn_task(
        self,
        prompt: str,
        description: str,
        save_to_report: str | None = None,
        max_turns: int = 50,
        max_time_seconds: int = 300,
    ) -> BackgroundTask:
        task = BackgroundTask(description=description, prompt=prompt)
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

        self.agent._background_task_count += 1
        try:
            prefixed_prompt = f"@no-kg @no-history {task.prompt}"
            result = await self.agent.query(
                prefixed_prompt,
                max_turns=max_turns,
                max_time_seconds=max_time_seconds,
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
        }
        with open(self.notifications_file, "a") as f:
            f.write(json.dumps(notification) + "\n")

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
