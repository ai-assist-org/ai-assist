"""Unified action execution engine"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .action_model import ActionDefinition
from .agent import AiAssistAgent
from .conditions import ActionExecutor, ConditionEvaluator
from .event_sources import EventContext
from .notification_dispatcher import Notification, NotificationDispatcher
from .state import StateManager

logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    """Result from executing an action"""

    action_name: str
    success: bool
    output: str
    timestamp: datetime
    metadata: dict[str, Any]


class ActionEngine:
    """Execute actions and track state — replaces TaskRunner"""

    def __init__(self, agent: AiAssistAgent, state_manager: StateManager) -> None:
        self.agent = agent
        self.state_manager = state_manager

    async def execute_action(self, action: ActionDefinition, event_context: EventContext | None = None) -> ActionResult:
        timestamp = datetime.now()
        state_key = self._state_key(action)

        try:
            prompt = action.prompt
            if event_context is not None:
                prompt = self._build_prompt_with_event(prompt, event_context)

            if prompt == "__builtin__:kg_synthesis":
                output = await self.agent._run_synthesis_from_kg()
            elif prompt.endswith(".awl"):
                output = await self._run_awl_script(action, prompt)
            elif action.is_mcp_prompt:
                server, prompt_name = action.parse_mcp_prompt()
                output = await self.agent.execute_mcp_prompt(
                    server, prompt_name, action.prompt_arguments, max_turns=action.max_turns
                )
            else:
                output = await self.agent.query(prompt, max_turns=action.max_turns)

            evaluator = ConditionEvaluator()
            metadata = evaluator.extract_metadata(output)

            if action.conditions:
                executor = ActionExecutor(self.agent, self.state_manager)
                for condition in action.conditions:
                    if "if" in condition and "then" in condition:
                        if evaluator.evaluate(condition["if"], metadata):
                            context = {"result": output, "metadata": metadata, "task_name": action.name}
                            await executor.execute(condition["then"], context)

            self.state_manager.update_monitor(
                state_key,
                {
                    "task_name": action.name,
                    "last_success": True,
                    "last_output_length": len(output),
                    "last_metadata": metadata,
                },
            )

            from .execution_helpers import build_history_entry

            history_entry = build_history_entry(action.name, True, timestamp, metadata, event_context)
            self.state_manager.append_history(state_key, history_entry)

            result = ActionResult(
                action_name=action.name, success=True, output=output, timestamp=timestamp, metadata=metadata
            )

            if action.notify:
                await self._send_notification(action, result)

            return result

        except Exception:
            logger.exception("Action '%s' failed", action.name)
            error_msg = str(Exception.__context__) if Exception.__context__ else "Unknown error"

            import sys

            exc_info = sys.exc_info()
            error_msg = str(exc_info[1]) if exc_info[1] else "Unknown error"

            self.state_manager.update_monitor(
                state_key, {"task_name": action.name, "last_success": False, "last_error": error_msg}
            )
            self.state_manager.append_history(
                state_key,
                {"task_name": action.name, "success": False, "error": error_msg, "timestamp": timestamp.isoformat()},
            )

            result = ActionResult(
                action_name=action.name, success=False, output=error_msg, timestamp=timestamp, metadata={}
            )

            await self._send_failure_notification(action, result)
            if action.notify:
                await self._send_notification(action, result)

            return result

    async def _run_awl_script(self, action: ActionDefinition, prompt: str) -> str:
        from .awl_executor import run_awl_script

        return await run_awl_script(prompt, self.agent, variables=action.prompt_arguments)

    @staticmethod
    def _build_prompt_with_event(prompt: str, event: EventContext) -> str:
        lines = ["[Event]", f"Source: {event.source_type}", f"Type: {event.event_type}"]
        for key, value in event.metadata.items():
            lines.append(f"{key.capitalize()}: {value}")
        lines.append(f"Payload: {event.payload}")
        lines.append(f"Timestamp: {event.timestamp.isoformat()}")
        lines.append("")
        lines.append("[Prompt]")
        lines.append(prompt)
        return "\n".join(lines)

    @staticmethod
    def _state_key(action: ActionDefinition) -> str:
        sanitized = "".join(c if c.isalnum() or c in "-_" else "_" for c in action.name)
        return f"action_{sanitized}"

    async def _send_notification(self, action: ActionDefinition, result: ActionResult) -> None:
        level = "success" if result.success else "error"
        message = result.output[:500] if result.output else "No output"

        notification = Notification(
            id=f"action-{action.name}-{int(result.timestamp.timestamp() * 1000)}",
            action_id=action.name,
            title=f"Action: {action.name}",
            message=message,
            level=level,
            timestamp=result.timestamp,
            channels=action.notification_channels,
            delivered={},
        )
        dispatcher = NotificationDispatcher()
        await dispatcher.dispatch(notification)

    async def _send_failure_notification(self, action: ActionDefinition, result: ActionResult) -> None:
        from .execution_helpers import send_failure_notification

        await send_failure_notification(action.name, result.output, result.timestamp)
