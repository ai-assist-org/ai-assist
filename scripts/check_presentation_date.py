#!/usr/bin/env python3
"""A modified presentation must carry the current month in its \\date{...} (see AGENTS.md).

Usage: check_presentation_date.py FILE...

Only files that differ from HEAD (staged or not) are checked: an untouched
presentation keeps the month of its last change, so running every hook on all
files (as CI does) never fails because the month changed.
"""

import re
import subprocess
import sys
from datetime import date
from pathlib import Path

# English names whatever the locale (the slides are in English)
_MONTHS = ("January February March April May June July August September October November December").split()
_DATE = re.compile(r"^\\date\{([^}]*)\}", re.MULTILINE)


def date_problem(text: str, today: date) -> str | None:
    """Why the \\date line is not the current month, or None when it is."""
    expected = f"{_MONTHS[today.month - 1]} {today.year}"
    match = _DATE.search(text)
    if match is None:
        return "no \\date{...} line"
    if match.group(1).strip() != expected:
        return f"\\date{{{match.group(1)}}} should be \\date{{{expected}}}"
    return None


def _modified(path: str, cwd: Path | None) -> bool:
    """Whether the file differs from HEAD (staged or in the working tree)."""
    result = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", path], cwd=cwd, check=False)
    return result.returncode != 0


def main(paths: list[str], today: date | None = None, cwd: Path | None = None) -> int:
    today = today or date.today()
    failed = False
    for path in paths:
        if not _modified(path, cwd):
            continue
        problem = date_problem(Path(path).read_text(), today)
        if problem:
            print(f"{path}: the presentation changed: {problem}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
