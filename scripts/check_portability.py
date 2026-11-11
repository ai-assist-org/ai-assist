#!/usr/bin/env python3
"""Check that shell and awk code runs on Linux and macOS (see "Portability" in AGENTS.md).

Usage: check_portability.py FILE...

Shell files (*.sh, *.bash, or a sh/bash shebang) must run on bash 3.2 (macOS
/bin/bash) with BSD tools; awk files (*.awk) on any POSIX awk (BSD awk, mawk).
Prints "path:line: problem" for each finding and exits 1 if there is any.
Other files are ignored. Comment lines are not checked.
"""

import re
import sys
from pathlib import Path

_SHELL_RULES = [
    (re.compile(r"\b(mapfile|readarray|coproc)\b"), "bash 4 only (macOS has bash 3.2): use a while-read loop"),
    (re.compile(r"\b(declare|local|typeset) -[a-zA-Z]*A"), "associative arrays are bash 4 only"),
    (re.compile(r"\$\{\w+(,,?|\^\^?)\}"), "case conversion ${var,,} / ${var^^} is bash 4 only"),
    (re.compile(r"&>>"), "&>> is bash 4 only: use >> file 2>&1"),
    (re.compile(r"\|&"), "|& is bash 4 only: use 2>&1 |"),
    (re.compile(r"\bdate (-I|--iso-8601)"), "BSD date has no -I: give an explicit +FORMAT"),
    (re.compile(r"\bstat (-c|--format)\b"), "BSD stat has no -c (macOS: stat -f)"),
    (re.compile(r"\bsed -i\b"), "sed -i differs on BSD (needs an argument): write to a new file"),
    (re.compile(r"\breadlink -f\b"), "BSD readlink has no -f: cd + pwd -P, or realpath in Python"),
    (re.compile(r"\bgrep (-[a-zA-Z]*P|--perl-regexp)\b"), "BSD grep has no -P: use -E"),
    (re.compile(r"\bfind\b.*\s-printf\b"), "BSD find has no -printf"),
]
_SHA1SUM = re.compile(r"\bsha1sum\b")

_AWK_RULES = [
    (
        re.compile(r"\b(mktime|strftime|systime|gensub|asorti?|patsplit|isarray|typeof)\s*\("),
        "gawk-only function (BSD awk, mawk lack it)",
    ),
    (re.compile(r"\b(PROCINFO|IGNORECASE|FPAT|BINMODE)\b|\b(BEGINFILE|ENDFILE)\b|@include\b"), "gawk-only feature"),
    (re.compile(r'[>|]\s*"/dev/(stdout|stderr)"'), '"/dev/stdout" is a real file outside gawk: print to stdout'),
    (
        re.compile(r'\bg?sub\s*\([^,]+,\s*"[^"]*\\\\'),
        "backslashes in a sub()/gsub() replacement differ between awks: split and join instead",
    ),
]


def _kind(path: Path) -> str | None:
    if path.suffix == ".awk":
        return "awk"
    if path.suffix in (".sh", ".bash"):
        return "shell"
    try:
        with open(path, "rb") as f:
            first = f.readline(200)
    except OSError:
        return None
    if re.match(rb"#!\s*\S*/(env\s+)?(ba)?sh\b", first):
        return "shell"
    return None


def check_file(path: Path) -> list[str]:
    """The portability problems of one file, as "path:line: problem"."""
    kind = _kind(path)
    if kind is None:
        return []
    rules = _AWK_RULES if kind == "awk" else _SHELL_RULES
    found = []
    for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        for pattern, why in rules:
            if pattern.search(line):
                found.append(f"{path}:{number}: {why}: {line.strip()}")
        if kind == "shell" and _SHA1SUM.search(line) and "shasum" not in line:
            found.append(f"{path}:{number}: macOS has shasum, not sha1sum: add a 'shasum -a 1' fallback")
    return found


def main(paths: list[str]) -> int:
    found = [problem for p in paths for problem in check_file(Path(p))]
    for problem in found:
        print(problem)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
