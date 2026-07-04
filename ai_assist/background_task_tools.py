"""Agent tools for spawning and managing background tasks"""

import json
import logging

from .background_tasks import BackgroundTaskManager

logger = logging.getLogger(__name__)


class BackgroundTaskTools:
    def __init__(self, manager: BackgroundTaskManager) -> None:
        self.manager = manager

    def get_tool_definitions(self) -> list[dict]:
        return [
            {
                "name": "internal__run_background",
                "description": (
                    "Spawn a long-running query as a background task. "
                    "The user can continue chatting while it runs. "
                    "Results are delivered via notification when complete."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The query/prompt to execute in the background",
                        },
                        "description": {
                            "type": "string",
                            "description": "Short human-readable description of the task",
                        },
                        "save_to_report": {
                            "type": "string",
                            "description": "Report name to save full results to",
                        },
                        "max_turns": {
                            "type": "integer",
                            "description": "Maximum agentic turns (default: 50)",
                        },
                        "max_time_seconds": {
                            "type": "integer",
                            "description": "Maximum wall-clock time in seconds (default: 300)",
                        },
                    },
                    "required": ["prompt", "description"],
                },
            },
            {
                "name": "internal__list_background_tasks",
                "description": "List background tasks with their status",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "description": "Filter by status: pending, running, completed, failed",
                        },
                    },
                },
            },
            {
                "name": "internal__get_background_task",
                "description": "Get details and results of a specific background task",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "The task ID",
                        },
                    },
                    "required": ["task_id"],
                },
            },
            {
                "name": "internal__cancel_background_task",
                "description": "Cancel a running background task",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "The task ID to cancel",
                        },
                    },
                    "required": ["task_id"],
                },
            },
        ]

    async def execute_tool(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "run_background":
            task = await self.manager.spawn_task(
                prompt=arguments["prompt"],
                description=arguments["description"],
                save_to_report=arguments.get("save_to_report"),
                max_turns=arguments.get("max_turns", 50),
                max_time_seconds=arguments.get("max_time_seconds", 300),
            )
            return (
                f"Background task spawned (id: {task.id}). "
                f"Description: {task.description}. "
                f"You will be notified when it completes."
            )

        if tool_name == "list_background_tasks":
            tasks = self.manager.list_tasks(status_filter=arguments.get("status"))
            if not tasks:
                return "No background tasks found."
            lines = []
            for t in tasks:
                line = f"- [{t.status}] {t.id}: {t.description}"
                if t.completed_at:
                    line += f" (completed: {t.completed_at.isoformat()})"
                lines.append(line)
            return "\n".join(lines)

        if tool_name == "get_background_task":
            found = self.manager.get_task(arguments["task_id"])
            if found is None:
                return f"Task not found: {arguments['task_id']}"
            info = {
                "id": found.id,
                "description": found.description,
                "status": found.status,
                "prompt": found.prompt,
                "created_at": found.created_at.isoformat(),
                "started_at": found.started_at.isoformat() if found.started_at else None,
                "completed_at": found.completed_at.isoformat() if found.completed_at else None,
                "result": found.result,
                "result_report": found.result_report,
                "error": found.error,
            }
            return json.dumps(info, indent=2)

        if tool_name == "cancel_background_task":
            cancelled = await self.manager.cancel_task(arguments["task_id"])
            if cancelled:
                return f"Task {arguments['task_id']} cancelled."
            return f"Task not found or already completed: {arguments['task_id']}"

        return f"Unknown background task tool: {tool_name}"
