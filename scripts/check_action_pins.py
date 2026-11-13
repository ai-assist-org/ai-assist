#!/usr/bin/env python3
"""Check that GitHub workflows reference actions by immutable commit SHA.

Usage: check_action_pins.py FILE...

A tag or branch (`actions/checkout@v7`) can be repointed upstream, changing what
runs with nothing in the diff to review. Every `uses:` must be a local path
(`./...`), a full 40-hex commit SHA (`owner/repo@<sha>  # vX.Y.Z`), or a docker
image pinned by digest (`docker://image@sha256:...`).
Prints "path:line: problem" for each finding and exits 1 if there is any.
"""

import re
import sys
from pathlib import Path

_USES = re.compile(r"""^\s*(?:-\s+)?uses:\s*["']?([^"'\s#]+)""")
_PINNED = re.compile(r"^[^@\s]+@[0-9a-f]{40}$|^docker://[^@\s]+@sha256:[0-9a-f]{64}$")


def check_file(path: Path) -> list[str]:
    found = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        match = _USES.match(line)
        if not match:
            continue
        ref = match.group(1)
        if ref.startswith("./") or _PINNED.match(ref):
            continue
        found.append(f"{path}:{lineno}: {ref} is not pinned by commit SHA (use owner/repo@<40-hex sha>  # vX.Y.Z)")
    return found


def main(argv: list[str]) -> int:
    found = [p for arg in argv for p in check_file(Path(arg))]
    for problem in found:
        print(problem)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
