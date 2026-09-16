"""Routing of tool results: jq filtering, redirection, and paginated collection."""

import json
import logging
from pathlib import Path
from typing import Any

from .report_tools import SUPPORTED_FORMATS

logger = logging.getLogger(__name__)


def _resolve_dotpath(data: dict, path: str) -> Any:
    """Navigate a dot-separated path in a dict.

    Handles Elasticsearch-style totals where the value is {value: N, relation: ...}.
    """
    current: Any = data
    for key in path.split("."):
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return None
    if isinstance(current, dict) and "value" in current:
        return current["value"]
    return current


def _extract_data_items(data: Any, data_field: str) -> list:
    """Extract the data array from a paginated response.

    With data_field="auto", finds the first top-level list key not starting with '_'.
    """
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    if data_field == "auto":
        for key, val in data.items():
            if isinstance(val, list) and not key.startswith("_"):
                return val
        return []
    return data.get(data_field, [])


class ToolResultRouter:
    """Handle special tool-result redirection and pagination parameters."""

    def __init__(self, agent, config, report_tools, json_tools):
        self._agent = agent
        self.config = config
        self.report_tools = report_tools
        self.json_tools = json_tools

    @property
    def sessions(self):
        """Live MCP sessions dict (read from the agent, which may reassign it)."""
        return self._agent.sessions

    @staticmethod
    def parse_report_param(value: str) -> tuple[str, str]:
        """Parse 'name' or 'name:format' into (name, format). Default format is 'md'."""
        if ":" in value:
            name, fmt = value.rsplit(":", 1)
        else:
            name, fmt = value, "md"

        name, fmt = name.strip(), fmt.strip().lower()
        if not name:
            raise ValueError("Report name cannot be empty")
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format '{fmt}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}")
        return name, fmt

    @staticmethod
    def parse_collect_param(value: str) -> tuple[str, str, int | None]:
        """Parse 'name', 'name:format', or 'name:format:N' for __collect_to_report.

        Default format is 'jsonl'. Returns (name, format, max_items).
        """
        parts = value.split(":")
        if len(parts) == 3:
            name, fmt, limit_str = parts[0].strip(), parts[1].strip().lower(), parts[2].strip()
            max_items = int(limit_str)
            if max_items <= 0:
                raise ValueError("Max items must be positive")
        elif len(parts) == 2:
            name, fmt, max_items = parts[0].strip(), parts[1].strip().lower(), None
        else:
            name, fmt, max_items = value.strip(), "jsonl", None

        if not name:
            raise ValueError("Report name cannot be empty")
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format '{fmt}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}")
        return name, fmt, max_items

    def apply_jq_filter(self, result_text: str, jq_filter: str) -> str:
        """Apply a jq filter expression to a result string."""
        return self.json_tools.filter_string(result_text, jq_filter)

    def handle_result_redirection(
        self,
        result_text: str,
        save_to_file: str | None,
        write_to_report: str | None,
        append_to_report: str | None,
    ) -> str | None:
        """Handle __save_to_file, __write_to_report, __append_to_report.

        Returns a summary string if any redirection was performed, None otherwise.
        """
        summaries: list[str] = []

        if save_to_file:
            try:
                output_path = Path(save_to_file).expanduser()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(result_text)
                summary = f"Result saved to {save_to_file} ({len(result_text):,} bytes, {len(result_text.splitlines())} lines)"
                logger.info("Saved tool result to file: %s (%d bytes)", save_to_file, len(result_text))
                summaries.append(summary)
            except Exception:
                logger.exception("Error saving result to %s", save_to_file)
                summaries.append(f"Error saving result to {save_to_file}")

        if write_to_report:
            try:
                name, fmt = self.parse_report_param(write_to_report)
                report_result = self.report_tools._write_report(name, result_text, fmt=fmt)
                logger.info("Wrote tool result to report: %s (%s)", name, fmt)
                summaries.append(report_result)
            except Exception:
                logger.exception("Error writing result to report '%s'", write_to_report)
                summaries.append(f"Error writing result to report '{write_to_report}'")

        if append_to_report:
            try:
                name, fmt = self.parse_report_param(append_to_report)
                report_result = self.report_tools._append_to_report(name, result_text, fmt=fmt)
                logger.info("Appended tool result to report: %s (%s)", name, fmt)
                summaries.append(report_result)
            except Exception:
                logger.exception("Error appending result to report '%s'", append_to_report)
                summaries.append(f"Error appending result to report '{append_to_report}'")

        return "\n".join(summaries) if summaries else None

    async def collect_paginated_to_report(
        self,
        server_name: str,
        original_tool_name: str,
        arguments: dict,
        collect_param: str,
    ) -> str:
        """Auto-paginate an MCP tool and collect all results into a report."""
        name, fmt, max_items = self.parse_collect_param(collect_param)

        server_config = self.config.mcp_servers.get(server_name)
        pagination = server_config.pagination if server_config else None

        if not pagination:
            session = self.sessions[server_name]
            result = await session.call_tool(original_tool_name, arguments)
            result_text = (
                "\n".join(item.text if hasattr(item, "text") else str(item) for item in (result.content or [])) or ""
            )
            self.report_tools._write_report(name, result_text, fmt=fmt)
            return f"Collected results to report '{name}' (1 page, no pagination config for server '{server_name}')"

        # Apply per-tool overrides
        overrides = pagination.tool_overrides.get(original_tool_name, {})
        offset_param = overrides.get("offset_param", pagination.offset_param)
        limit_param = overrides.get("limit_param", pagination.limit_param)
        total_field = overrides.get("total_field", pagination.total_field)
        data_field = overrides.get("data_field", pagination.data_field)

        page_size = int(arguments.get(limit_param, pagination.default_page_size))
        arguments[limit_param] = page_size
        current_offset = int(arguments.get(offset_param, 0))
        arguments[offset_param] = current_offset

        session = self.sessions[server_name]
        total_collected = 0
        page_count = 0
        is_first_page = True
        total_available: int | float = float("inf")
        max_pages = 50

        while page_count < max_pages:
            result = await session.call_tool(original_tool_name, dict(arguments))
            result_text = (
                "\n".join(item.text if hasattr(item, "text") else str(item) for item in (result.content or [])) or ""
            )

            try:
                data = json.loads(result_text)
            except json.JSONDecodeError:
                if is_first_page:
                    self.report_tools._write_report(name, result_text, fmt=fmt)
                else:
                    self.report_tools._append_to_report(name, result_text, fmt=fmt)
                page_count += 1
                break

            if is_first_page:
                resolved_total = _resolve_dotpath(data, total_field)
                if resolved_total is not None:
                    total_available = int(resolved_total)

            items = _extract_data_items(data, data_field)
            if not items:
                if is_first_page:
                    self.report_tools._write_report(name, "", fmt=fmt)
                break

            if max_items is not None:
                remaining = max_items - total_collected
                if remaining <= 0:
                    break
                items = items[:remaining]

            if fmt == "jsonl":
                content = "\n".join(json.dumps(item) for item in items)
            else:
                content = json.dumps(items, indent=2)

            if is_first_page:
                self.report_tools._write_report(name, content, fmt=fmt)
            else:
                self.report_tools._append_to_report(name, content, fmt=fmt)

            total_collected += len(items)
            page_count += 1
            is_first_page = False

            current_offset += page_size
            arguments[offset_param] = current_offset

            effective_limit = total_available
            if max_items is not None:
                effective_limit = min(effective_limit, max_items)
            if total_collected >= effective_limit:
                break
            if len(items) < page_size:
                break

        page_label = "page" if page_count == 1 else "pages"
        return f"Collected {total_collected} items to report '{name}' ({page_count} {page_label})"
