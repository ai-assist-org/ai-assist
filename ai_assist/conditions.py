"""Condition evaluation and action execution for user-defined tasks"""

import logging
import re
from datetime import datetime
from typing import Any

from .agent import AiAssistAgent
from .config import get_config_dir
from .state import StateManager

logger = logging.getLogger(__name__)


class ConditionEvaluator:
    """Evaluate conditions on task results"""

    def extract_metadata(self, output: str) -> dict[str, Any]:
        """Extract metadata from agent output for condition evaluation

        Looks for patterns like:
        - "Found X items" -> {items: X}
        - "Success rate: X%" -> {success_rate: X}
        - "X failures detected" -> {failures: X}
        - "Status: X" -> {status: "X"}
        """
        metadata: dict[str, Any] = {}

        # Pattern: "Found X <something>"
        found_pattern = r"[Ff]ound\s+(\d+)\s+(\w+)"
        for match in re.finditer(found_pattern, output):
            count = int(match.group(1))
            item_type = match.group(2).lower()
            # Store both specific and generic counts
            metadata[item_type] = count
            if "count" not in metadata:
                metadata["count"] = count

        # Pattern: "X failures"
        failures_pattern = r"(\d+)\s+failures?"
        failures_match = re.search(failures_pattern, output, re.IGNORECASE)
        if failures_match:
            metadata["failures"] = int(failures_match.group(1))

        # Pattern: "Success rate: X%"
        success_rate_pattern = r"[Ss]uccess\s+rate[:\s]+(\d+(?:\.\d+)?)%?"
        success_rate_match = re.search(success_rate_pattern, output)
        if success_rate_match:
            metadata["success_rate"] = float(success_rate_match.group(1))

        # Pattern: "Status: X"
        status_pattern = r"[Ss]tatus[:\s]+(\w+)"
        status_match = re.search(status_pattern, output)
        if status_match:
            metadata["status"] = status_match.group(1).lower()

        # Pattern: "X updated" or "X new"
        updated_pattern = r"(\d+)\s+(updated|new)"
        for match in re.finditer(updated_pattern, output, re.IGNORECASE):
            count = int(match.group(1))
            update_type = match.group(2).lower()
            metadata[f"{update_type}_count"] = count

        return metadata

    def evaluate(self, condition_str: str, metadata: dict[str, Any]) -> bool:
        """Evaluate a condition string against metadata

        Supports:
        - Comparison operators: >, <, >=, <=, ==, !=
        - String matching: contains, not_contains

        Examples:
        - "count > 5"
        - "failures >= 10"
        - "status == 'failed'"
        - "message contains 'critical'"
        """
        condition_str = condition_str.strip()

        # Handle "contains" operator
        if " contains " in condition_str:
            parts = condition_str.split(" contains ", 1)
            field = parts[0].strip()
            value = parts[1].strip().strip("'\"")

            str_field_value = str(metadata.get(field, ""))
            return value in str_field_value

        # Handle "not_contains" operator
        if " not_contains " in condition_str:
            parts = condition_str.split(" not_contains ", 1)
            field = parts[0].strip()
            value = parts[1].strip().strip("'\"")

            str_field_value = str(metadata.get(field, ""))
            return value not in str_field_value

        # Handle comparison operators
        for op in [">=", "<=", "==", "!=", ">", "<"]:
            if op in condition_str:
                parts = condition_str.split(op, 1)
                field = parts[0].strip()
                value_str = parts[1].strip().strip("'\"")

                # Get field value from metadata
                field_value: Any = metadata.get(field)
                if field_value is None:
                    return False

                # Try to convert value to appropriate type
                compare_value: Any
                try:
                    if isinstance(field_value, int | float):
                        compare_value = float(value_str)
                    else:
                        compare_value = value_str
                except ValueError, TypeError:
                    compare_value = value_str

                # Evaluate comparison
                if op == ">":
                    return field_value > compare_value
                elif op == "<":
                    return field_value < compare_value
                elif op == ">=":
                    return field_value >= compare_value
                elif op == "<=":
                    return field_value <= compare_value
                elif op == "==":
                    return field_value == compare_value
                elif op == "!=":
                    return field_value != compare_value

        return False


class ActionExecutor:
    """Execute actions based on condition evaluation"""

    def __init__(self, agent: AiAssistAgent, state_manager: StateManager):
        self.agent = agent
        self.state_manager = state_manager

    async def execute(self, action: dict, context: dict):
        """Execute an action with given context

        Available actions:
        - notify: Print notification to console
        - log: Write to log file
        - create_google_doc: Create Google Doc with results
        - store_kg: Store in knowledge graph

        Context should include:
        - result: The task result output
        - metadata: Extracted metadata
        - task_name: Name of the task
        """
        action_type = action.get("action")

        if action_type == "notify":
            await self._notify(action, context)
        elif action_type == "log":
            await self._log(action, context)
        elif action_type == "create_google_doc":
            await self._create_doc(action, context)
        elif action_type == "store_kg":
            await self._store_kg(action, context)
        else:
            logger.warning("Unknown action type: %s", action_type)

    async def _notify(self, action: dict, context: dict):
        """Send notification (console for now, extensible to Slack/email)"""
        message = action.get("message", "Notification")
        level = action.get("level", "info")

        message = self._replace_placeholders(message, context)

        if level == "error":
            logger.error("%s", message)
        elif level == "warning":
            logger.warning("%s", message)
        else:
            logger.info("%s", message)

    async def _log(self, action: dict, context: dict):
        """Write to log file"""
        message = action.get("message", context.get("result", ""))
        log_file_name = action.get("file", f"{context['task_name']}.log")

        message = self._replace_placeholders(message, context)

        log_dir = get_config_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / log_file_name
        timestamp = datetime.now().isoformat()

        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")

        logger.info("Logged to %s", log_file)

    async def _create_doc(self, action: dict, context: dict):
        """Create Google Doc with results"""
        title = action.get("title", f"Report - {datetime.now().strftime('%Y-%m-%d')}")
        folder = action.get("folder")

        title = self._replace_placeholders(title, context)

        prompt = f"""
        Create a Google Doc with the following content:

        Title: {title}
        {f"Folder: {folder}" if folder else ""}

        Content:
        {context.get('result', '')}

        Please use the create_google_doc_from_markdown MCP tool.
        """

        try:
            await self.agent.query(prompt)
            logger.info("Created Google Doc: %s", title)
        except Exception as e:
            logger.exception("Error creating Google Doc: %s", e)

    async def _store_kg(self, action: dict, context: dict):
        """Store result in knowledge graph"""
        # This would integrate with the knowledge graph
        # For now, just print a message
        entity_type = action.get("entity_type", "task_result")
        logger.info("Would store in KG as %s", entity_type)
        # TODO: Implement KG storage

    def _replace_placeholders(self, text: str, context: dict) -> str:
        """Replace placeholders like {count}, {date}, etc."""
        metadata = context.get("metadata", {})

        text = text.replace("{date}", datetime.now().strftime("%Y-%m-%d"))

        for key, value in metadata.items():
            placeholder = f"{{{key}}}"
            if placeholder in text:
                text = text.replace(placeholder, str(value))

        return text
