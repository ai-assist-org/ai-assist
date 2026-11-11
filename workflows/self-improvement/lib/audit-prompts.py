"""List the security questions of ai-assist's audit log as TSV.

Usage: audit-prompts.py <tool_calls.jsonl> <days>

One row per question asked in the last <days> days, for extract-prompts.awk:
date, time, pid, script (base name, "unknown" outside AWL), task ("-"),
tool, kind, decision, question — newlines in the question replaced by \\036.
Lines that are not JSON objects are skipped. Does not import ai-assist.
"""

import json
import os
import sys
from datetime import datetime, timedelta


def clean(value: object) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", "\036")


def main(path: str, days: int) -> None:
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    with open(path, errors="replace") as f:
        for line in f:
            try:
                entry = json.loads(line, strict=False)
            except ValueError:
                continue
            if not isinstance(entry, dict) or str(entry.get("timestamp", "")) < cutoff:
                continue
            date, _, time = str(entry["timestamp"]).partition("T")
            script = os.path.basename(str(entry.get("script") or "")) or "unknown"
            for c in entry.get("confirmations") or []:
                row = [
                    date,
                    time + ",000",
                    entry.get("pid", ""),
                    script,
                    entry.get("task") or "-",
                    entry.get("tool_name", ""),
                    c.get("kind", ""),
                    c.get("decision", ""),
                    c.get("question", ""),
                ]
                print("\t".join(clean(x) for x in row))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
