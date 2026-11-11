"""List pipelines whose exit code 0 may hide a failure, from ai-assist's audit log.

Usage: audit-pipes.py <tool_calls.jsonl> <days>

A shell pipeline reports the exit code of its last command: `make 2>&1 | tail`
exits 0 even when make failed. One TSV row per suspicious execute_command call
of the last <days> days, in extract-awl-signals.awk's format (date, time, pid,
script, task, event "pipe_masked", detail):

- certain: ai-assist's own warning that an earlier pipeline stage failed;
- candidate: exit code 0 but an error line as programs print them (`fatal:`,
  `error:`, a traceback, `No such file or directory`...) on stderr, or in
  stdout when the command merged stderr (2>&1), and the command itself does not
  mention it (so `grep 'error:' log | head` is a probe, not a failure).
  Candidates need judgement: never count them as failures.

Lines that are not JSON objects are skipped. Does not import ai-assist.
"""

import json
import os
import re
import shlex
import sys
from datetime import datetime, timedelta

_WARNING = "Warning: the exit code is that of the last command in the pipeline"
# Error lines as programs print them (anchored: a file or an email that merely
# mentions "failed" is not one)
_ERROR_LINE = re.compile(
    r"^((fatal|error|Error|ERROR)(\[[^\]]*\])?: |\S+: (fatal|error): |Traceback \(most recent call last\):|"
    r"(/bin/sh|bash|sh): line \d+: )|: (command not found|No such file or directory|Permission denied)$",
    re.MULTILINE,
)
_SECRET = re.compile(r"(password|token|secret|api[_-]?key)[=:]\S*", re.IGNORECASE)


def has_pipe(command: str) -> bool:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        return "|" in list(lexer)
    except ValueError:
        return "|" in command


def clean(value: object, limit: int = 200) -> str:
    text = re.sub(r"[\t\r\n]+", " ", str(value))
    return _SECRET.sub(r"\1=<redacted>", text)[:limit]


def suspicion(command: str, result: str) -> str | None:
    """Why this pipeline may hide a failure, or None."""
    if not has_pipe(command) or not re.search(r"^Exit code: 0$", result, re.MULTILINE):
        return None
    if _WARNING in result:
        statuses = re.search(r"exit statuses: ([0-9 ]+)", result)
        return f"[exit statuses {statuses.group(1) if statuses else '?'}]"
    output = result.split("Exit code: 0", 1)[1]
    # an earlier stage's error lands on stderr, or in stdout when merged (2>&1)
    stdout, _, stderr = output.partition("\nSTDERR:\n")
    searched = stderr + ("\n" + stdout if "2>&1" in command else "")
    for match in _ERROR_LINE.finditer(searched):
        if _keyword(match.group(0)) in command.lower():  # `grep 'error:' log | head` is a probe
            continue
        line = searched[match.start() :].split("\n", 1)[0]
        return f"[output: {line.strip()[:60]}]"
    return None


def _keyword(error_text: str) -> str:
    """The word a command searching for this error would contain."""
    lowered = error_text.lower()
    for keyword in ("fatal", "error", "traceback", "command not found", "no such file", "permission denied"):
        if keyword in lowered:
            return keyword
    return lowered.strip()


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
            why = suspicion(command, str(entry.get("result_summary", "")))
            if why is None:
                continue
            date, _, time = str(entry["timestamp"]).partition("T")
            script = os.path.basename(str(entry.get("script") or "")) or "unknown"
            row = [date, time + ",000", entry.get("pid", ""), script, entry.get("task") or "-", "pipe_masked"]
            print("\t".join(clean(x) for x in row) + "\t" + clean(f"{command[:120]} {why}"))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
