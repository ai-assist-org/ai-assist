"""Execute user-defined tasks and track state"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .agent import AiAssistAgent
from .conditions import ActionExecutor, ConditionEvaluator
from .event_sources import EventContext
from .notification_dispatcher import Notification, NotificationDispatcher
from .state import StateManager
from .tasks import TaskDefinition

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result from executing a task"""

    task_name: str
    success: bool
    output: str
    timestamp: datetime
    metadata: dict[str, Any]  # Extracted values for conditions


class TaskRunner:
    """Execute a user-defined task and track its state"""

    def __init__(self, task_def: TaskDefinition, agent: AiAssistAgent, state_manager: StateManager):
        self.task_def = task_def
        self.agent = agent
        self.state_manager = state_manager
        self.state_key = self._get_state_key()

    def _get_state_key(self) -> str:
        """Generate state key for this task"""
        # Sanitize task name for use in filenames
        sanitized = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.task_def.name)
        return f"task_{sanitized}"

    def _substitute_event_vars(self, prompt: str, event: EventContext) -> str:
        prompt = prompt.replace("${event.payload}", event.payload)
        prompt = prompt.replace("${event.source_type}", event.source_type)
        prompt = prompt.replace("${event.timestamp}", event.timestamp.isoformat())
        for key, value in event.metadata.items():
            prompt = prompt.replace(f"${{event.{key}}}", str(value))
        return prompt

    async def run(self, event_context: EventContext | None = None) -> TaskResult:
        """Execute the task and return results"""
        timestamp = datetime.now()

        try:
            prompt = self.task_def.prompt
            if event_context is not None:
                prompt = self._substitute_event_vars(prompt, event_context)

            # Detect and execute built-in, AWL goals, MCP prompts, or natural language
            if prompt == "__builtin__:kg_synthesis":
                output = await self.agent._run_synthesis_from_kg()
            elif prompt.endswith(".awl"):
                output = await self._run_awl_script()
            elif self.task_def.is_mcp_prompt:
                server_name, prompt_name = self.task_def.parse_mcp_prompt()
                output = await self.agent.execute_mcp_prompt(
                    server_name, prompt_name, self.task_def.prompt_arguments, max_turns=self.task_def.max_turns
                )
            else:
                output = await self.agent.query(prompt, max_turns=self.task_def.max_turns)

            evaluator = ConditionEvaluator()
            metadata = evaluator.extract_metadata(output)

            if self.task_def.conditions:
                executor = ActionExecutor(self.agent, self.state_manager)

                for condition in self.task_def.conditions:
                    if "if" in condition and "then" in condition:
                        if evaluator.evaluate(condition["if"], metadata):
                            context = {
                                "result": output,
                                "metadata": metadata,
                                "task_name": self.task_def.name,
                            }
                            await executor.execute(condition["then"], context)

            self.state_manager.update_monitor(
                self.state_key,
                {
                    "task_name": self.task_def.name,
                    "last_success": True,
                    "last_output_length": len(output),
                    "last_metadata": metadata,
                },
            )

            from .execution_helpers import build_history_entry

            history_entry = build_history_entry(self.task_def.name, True, timestamp, metadata, event_context)
            self.state_manager.append_history(self.state_key, history_entry)

            result = TaskResult(
                task_name=self.task_def.name, success=True, output=output, timestamp=timestamp, metadata=metadata
            )

            # Dispatch notification if configured
            if self.task_def.notify:
                await self._send_notification(result)

            return result

        except Exception as e:
            logger.exception("Task '%s' failed", self.task_def.name)
            error_msg = str(e)
            self.state_manager.update_monitor(
                self.state_key,
                {
                    "task_name": self.task_def.name,
                    "last_success": False,
                    "last_error": error_msg,
                },
            )

            self.state_manager.append_history(
                self.state_key,
                {
                    "task_name": self.task_def.name,
                    "success": False,
                    "error": error_msg,
                    "timestamp": timestamp.isoformat(),
                },
            )

            result = TaskResult(
                task_name=self.task_def.name, success=False, output=error_msg, timestamp=timestamp, metadata={}
            )

            # Always notify on failure so errors are never silently swallowed
            await self._send_failure_notification(result)

            # Also dispatch to configured channels if notify is enabled
            if self.task_def.notify:
                await self._send_notification(result)

            return result

    async def _run_awl_script(self) -> str:
        from .awl_executor import run_awl_script

        return await run_awl_script(self.task_def.prompt, self.agent, variables=self.task_def.prompt_arguments)

    def get_last_run(self) -> datetime | None:
        """Get timestamp of last successful run"""
        state = self.state_manager.get_monitor_state(self.state_key)
        return state.last_check

    def get_history(self, limit: int = 10) -> list[dict]:
        """Get historical execution results"""
        return self.state_manager.get_history(self.state_key, limit=limit)

    async def _send_failure_notification(self, result: TaskResult):
        """Always send a notification on task failure, regardless of notify setting.

        Uses desktop and console channels so failures are never silently swallowed.
        """
        from .execution_helpers import send_failure_notification

        await send_failure_notification(self.task_def.name, result.output, result.timestamp)

    async def _send_notification(self, result: TaskResult):
        """Send notification for task completion"""
        # Determine notification level
        level = "success" if result.success else "error"

        # Truncate output for notification (max 500 chars)
        message = result.output[:500] if result.output else "No output"

        # Create notification
        notification = Notification(
            id=f"task-{self.task_def.name}-{int(result.timestamp.timestamp() * 1000)}",
            action_id=self.task_def.name,
            title=f"Task: {self.task_def.name}",
            message=message,
            level=level,
            timestamp=result.timestamp,
            channels=self.task_def.notification_channels,
            delivered={},
        )

        # Dispatch
        dispatcher = NotificationDispatcher()
        await dispatcher.dispatch(notification)
