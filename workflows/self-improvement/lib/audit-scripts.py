"""List the inline scripts agents ran, with what they were for, from ai-assist's audit log.

Usage: audit-scripts.py <tool_calls.jsonl> <days>

An inline script (`python3 -c ...`, `python3 - <<EOF`, an awk program, `node
-e`, `perl -e`) is code the agent wrote to do something no tool did for it:
a recurring purpose is a candidate for a tool, or a habit when a tool exists.
One TSV row per inline program in an execute_command call of the last <days>
days, in extract-awl-signals.awk's format (date, time, pid, script, task, event
"inline_script", detail = language \\037 purpose \\037 the command's start).
The purpose is a "+"-joined, sorted set of tags read from the code (json, yaml,
regex, csv, date, table, xml, sqlite, http, subprocess, file-write for python;
fields, aggregate, filter, text for awk), "other" when none applies. The full
commands are in the audit log only: the regular log truncates them.

Lines that are not JSON objects are skipped. Does not import ai-assist.
"""

import json
import os
import re
import sys
from datetime import datetime, timedelta

_INLINE = re.compile(
    r"\b(?P<lang>python3?|node|perl|ruby)\b(?:\s+-[a-zA-Z]+)*?\s+(?:-[a-z]*[ce]|-)(?:\s|$)",
)
# awk, its options (-F, / -F ',' / -v x=1: only -F and -v take a separate
# value), then the program in single quotes
_AWK = re.compile(r"(?:^|[\s|;&(])(?:g|m)?awk\b(?:\s+-[Fv]\s+(?:'[^']*'|\S+)|\s+-\S+)*?\s+'([^']*)'")
_PYTHON_TAGS = (
    ("json", r"\bjson\b"),
    ("yaml", r"\byaml\b"),
    ("regex", r"\bre\.(search|match|findall|sub|finditer|compile|split)\b"),
    ("csv", r"\bcsv\b"),
    ("date", r"\b(datetime|zoneinfo|dateutil)\b"),
    ("table", r"\b(pandas|pyarrow|parquet|duckdb|polars)\b"),
    ("xml", r"\b(xml|ElementTree|lxml|bs4|BeautifulSoup|html\.parser)\b"),
    ("sqlite", r"\bsqlite3\b"),
    ("http", r"\b(requests|urllib|httpx|http\.client)\b"),
    ("subprocess", r"\b(subprocess|os\.system)\b"),
    ("file-write", r"\.write(_text)?\(|open\([^)]*['\"][wa]b?['\"]"),
)
_AWK_TAGS = (
    ("fields", r"\$([0-9]+|NF)"),
    ("aggregate", r"\+=|\bEND\b|\bcount\b|\bsum\b"),
    ("filter", r"(^|[\s{;])/[^/]+/|~"),
    ("text", r"\b(g?sub|split|substr|index|match|tolower|toupper)\("),
)
_SECRET = re.compile(r"(password|token|secret|api[_-]?key)[=:]\S*", re.IGNORECASE)


def clean(value: object, limit: int = 200) -> str:
    text = re.sub(r"[\t\r\n]+", " ", str(value))
    return _SECRET.sub(r"\1=<redacted>", text)[:limit]


def purpose(code: str, tags: tuple) -> str:
    found = sorted({name for name, pattern in tags if re.search(pattern, code)})
    return "+".join(found) or "other"


def inline_scripts(command: str) -> list[tuple[str, str]]:
    """(language, purpose) of each inline program in a command."""
    found = []
    match = _INLINE.search(command)
    if match:
        lang = match.group("lang").rstrip("3")
        found.append((lang, purpose(command, _PYTHON_TAGS) if lang == "python" else "other"))
    found.extend(("awk", purpose(program, _AWK_TAGS)) for program in _AWK.findall(command))
    return found


def main(path: str, days: int) -> None:
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    with open(path, errors="replace") as f:
        for line in f:
            try:
                entry = json.loads(line, strict=False)
            except ValueError:
                continue
            if not isinstance(entry, dict) or entry.get("tool_name") != "internal__execute_command":
                continue
            if str(entry.get("timestamp", "")) < cutoff:
                continue
            command = str((entry.get("arguments") or {}).get("command", ""))
            date, _, time = str(entry["timestamp"]).partition("T")
            script = os.path.basename(str(entry.get("script") or "")) or "unknown"
            row = [date, time + ",000", entry.get("pid", ""), script, entry.get("task") or "-", "inline_script"]
            for lang, why in inline_scripts(command):
                print("\t".join(clean(x) for x in row) + "\t" + clean(f"{lang}\037{why}\037{command[:150]}"))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
