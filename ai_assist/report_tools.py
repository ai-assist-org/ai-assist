"""Internal report management tools for ai-assist"""

import json
from datetime import datetime
from pathlib import Path

SUPPORTED_FORMATS: dict[str, str] = {
    "md": ".md",
    "jsonl": ".jsonl",
    "csv": ".csv",
    "tsv": ".tsv",
}

FORMAT_ENUM = sorted(SUPPORTED_FORMATS.keys())


class ReportTools:
    """Internal tools for managing reports in multiple formats (md, jsonl, csv, tsv)"""

    def __init__(self, reports_dir: Path | None = None):
        from .config import get_reports_dir

        self.reports_dir = Path(reports_dir) if reports_dir else get_reports_dir()
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def get_tool_definitions(self) -> list[dict]:
        format_property = {
            "type": "string",
            "enum": FORMAT_ENUM,
            "description": "Report format (default: md)",
        }
        return [
            {
                "name": "internal__write_report",
                "_readonly": False,
                "description": (
                    "Create or completely replace a report file. "
                    "Supports md (markdown), jsonl, csv, and tsv formats."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Report name (without file extension)",
                        },
                        "content": {
                            "type": "string",
                            "description": (
                                "Content to write. For md: markdown text. "
                                "For jsonl: one JSON object per line. "
                                "For csv/tsv: header row + data rows."
                            ),
                        },
                        "format": format_property,
                    },
                    "required": ["name", "content"],
                },
                "_server": "internal",
            },
            {
                "name": "internal__append_to_report",
                "_readonly": False,
                "description": (
                    "Add content to the end of a report (creates file if needed). "
                    "For jsonl: content must be valid JSON (one object per line). "
                    "For csv/tsv: raw rows to append."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Report name (without file extension)",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to append",
                        },
                        "section": {
                            "type": "string",
                            "description": "Optional section header (md format only, without ##)",
                        },
                        "format": format_property,
                    },
                    "required": ["name", "content"],
                },
                "_server": "internal",
            },
            {
                "name": "internal__read_report",
                "_readonly": True,
                "description": "Read a report's current content. Auto-detects format if not specified.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Report name (without file extension)",
                        },
                        "format": format_property,
                    },
                    "required": ["name"],
                },
                "_server": "internal",
            },
            {
                "name": "internal__list_reports",
                "_readonly": True,
                "description": "List all available reports with metadata",
                "input_schema": {"type": "object", "properties": {}},
                "_server": "internal",
            },
            {
                "name": "internal__delete_report",
                "_readonly": False,
                "description": "Delete a report file. Auto-detects format if not specified.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Report name (without file extension)",
                        },
                        "format": format_property,
                    },
                    "required": ["name"],
                },
                "_server": "internal",
            },
        ]

    async def execute_tool(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "write_report":
            return self._write_report(
                arguments["name"],
                arguments["content"],
                fmt=arguments.get("format", "md"),
            )
        elif tool_name == "append_to_report":
            return self._append_to_report(
                arguments["name"],
                arguments["content"],
                section=arguments.get("section"),
                fmt=arguments.get("format", "md"),
            )
        elif tool_name == "read_report":
            return self._read_report(arguments["name"], fmt=arguments.get("format"))
        elif tool_name == "list_reports":
            return self._list_reports()
        elif tool_name == "delete_report":
            return self._delete_report(arguments["name"], fmt=arguments.get("format"))
        else:
            raise ValueError(f"Unknown report tool: {tool_name}")

    def _validate_report_path(self, report_file: Path) -> str | None:
        """Validate that a report path is within the reports directory.

        Returns:
            Error message if path escapes reports_dir, None if valid.
        """
        try:
            report_file.resolve().relative_to(self.reports_dir.resolve())
            return None
        except ValueError:
            return "Error: Invalid report name (path traversal blocked)"

    def _resolve_report_file(self, name: str, fmt: str | None = None) -> Path | str:
        if fmt:
            if fmt not in SUPPORTED_FORMATS:
                return f"Unsupported format '{fmt}'. Supported: {', '.join(FORMAT_ENUM)}"
            path = self.reports_dir / f"{name}{SUPPORTED_FORMATS[fmt]}"
            error = self._validate_report_path(path)
            if error:
                return error
            return path

        # Validate path before scanning (catches traversal in auto-detect mode)
        sample_path = self.reports_dir / f"{name}.md"
        error = self._validate_report_path(sample_path)
        if error:
            return error

        matches = []
        for fmt_key, ext in SUPPORTED_FORMATS.items():
            candidate = self.reports_dir / f"{name}{ext}"
            if candidate.exists():
                matches.append((fmt_key, candidate))

        if len(matches) == 0:
            return f"Report '{name}' not found"
        elif len(matches) == 1:
            return matches[0][1]
        else:
            formats = [m[0] for m in matches]
            return (
                f"Ambiguous: report '{name}' exists in multiple formats: "
                f"{', '.join(formats)}. Specify the 'format' parameter."
            )

    @staticmethod
    def _validate_jsonl(content: str) -> str | None:
        for i, raw_line in enumerate(content.strip().splitlines(), 1):
            stripped = raw_line.strip()
            if stripped:
                try:
                    json.loads(stripped)
                except json.JSONDecodeError as e:
                    return f"Error: Invalid JSON on line {i}: {e}"
        return None

    def _write_report(self, name: str, content: str, fmt: str = "md") -> str:
        if fmt not in SUPPORTED_FORMATS:
            return f"Error: Unsupported format '{fmt}'. Supported: {', '.join(FORMAT_ENUM)}"

        ext = SUPPORTED_FORMATS[fmt]
        report_file = self.reports_dir / f"{name}{ext}"
        error = self._validate_report_path(report_file)
        if error:
            return error

        if fmt == "md":
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = f"<!-- Generated by AI Assistant on {timestamp} -->\n\n"
            report_file.write_text(header + content)
        elif fmt == "jsonl":
            error = self._validate_jsonl(content)
            if error:
                return error
            report_file.write_text(content.rstrip("\n") + "\n")
        elif fmt in ("csv", "tsv"):
            report_file.write_text(content.rstrip("\n") + "\n")

        return f"Report '{name}' ({fmt}) written to {report_file}"

    def _append_to_report(self, name: str, content: str, section: str | None = None, fmt: str = "md") -> str:
        if fmt not in SUPPORTED_FORMATS:
            return f"Error: Unsupported format '{fmt}'. Supported: {', '.join(FORMAT_ENUM)}"

        ext = SUPPORTED_FORMATS[fmt]
        report_file = self.reports_dir / f"{name}{ext}"
        error = self._validate_report_path(report_file)
        if error:
            return error

        if fmt == "md":
            if not report_file.exists():
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                header = f"<!-- Generated by AI Assistant on {timestamp} -->\n\n"
                report_file.write_text(header)

            if section:
                content = f"\n## {section}\n\n{content}\n"
            else:
                content = f"\n{content}\n"

            with open(report_file, "a") as f:
                f.write(content)
        elif fmt == "jsonl":
            error = self._validate_jsonl(content)
            if error:
                return error
            with open(report_file, "a") as f:
                f.write(content.rstrip("\n") + "\n")
        elif fmt in ("csv", "tsv"):
            with open(report_file, "a") as f:
                f.write(content.rstrip("\n") + "\n")

        return f"Content appended to '{name}' ({fmt}) at {report_file}"

    def _read_report(self, name: str, fmt: str | None = None) -> str:
        resolved = self._resolve_report_file(name, fmt)
        if isinstance(resolved, str):
            return resolved

        if resolved.exists():
            content = resolved.read_text()
            return f"[Report file: {resolved}]\n{content}"
        else:
            return f"Report '{name}' not found"

    def _list_reports(self) -> str:
        reports = []
        for fmt_key, ext in SUPPORTED_FORMATS.items():
            for report_file in sorted(self.reports_dir.glob(f"*{ext}")):
                stat = report_file.stat()
                reports.append(
                    {
                        "name": report_file.stem,
                        "format": fmt_key,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    }
                )

        reports.sort(key=lambda r: (r["name"], r["format"]))
        return json.dumps(reports, indent=2)

    def _delete_report(self, name: str, fmt: str | None = None) -> str:
        resolved = self._resolve_report_file(name, fmt)
        if isinstance(resolved, str):
            return resolved

        if resolved.exists():
            resolved.unlink()
            return f"Report '{name}' deleted"
        else:
            return f"Report '{name}' not found"
