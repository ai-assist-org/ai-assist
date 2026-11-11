"""The "Project Stats" box of the presentation must match the repository (see AGENTS.md).

A figure may not claim more than the reality, nor fall behind it by more than
20%: refresh the slide when this test fails.
"""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SLIDES = (ROOT / "presentation" / "ai-assist-presentation.tex").read_text()
STALE = 0.8


def claimed(pattern: str) -> list[int]:
    match = re.search(pattern, SLIDES)
    assert match, f"Project Stats line not found in the presentation: {pattern}"
    return [int(g.replace(",", "")) for g in match.groups()]


def tracked(*patterns: str) -> list[Path]:
    out = subprocess.run(["git", "ls-files", *patterns], cwd=ROOT, capture_output=True, text=True, check=True)
    return [ROOT / f for f in out.stdout.split()]


def check(name: str, claim: int, actual: int) -> None:
    assert claim <= actual, f"the slide claims {claim:,} {name}, the repository has {actual:,}"
    assert claim >= STALE * actual, f"the slide claims {claim:,} {name}, now {actual:,}: refresh it"


def test_lines_of_production_code():
    [claim] = claimed(r"([\d,]+)\+ lines of production code")
    actual = sum(len(p.read_text().splitlines()) for p in tracked("ai_assist/*.py", "ai_assist/**/*.py"))
    check("lines of production code", claim, actual)


def test_tests_and_test_files():
    tests, files = claimed(r"([\d,]+)\+ tests across ([\d,]+) test files")
    check("test files", files, len(tracked("tests/test_*.py")))
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "-n0"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    collected = re.search(r"(\d+)(?:/\d+)? tests? collected", out)
    assert collected, out[-500:]
    check("tests", tests, int(collected.group(1)))


def test_core_dependencies():
    [claim] = claimed(r"(\d+) core dependencies")
    actual = len(tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"])
    assert claim == actual, f"the slide claims {claim} core dependencies, pyproject.toml has {actual}"


@pytest.mark.parametrize("claim, actual, ok", [(100, 100, True), (101, 100, False), (80, 100, True), (79, 100, False)])
def test_staleness_rule(claim, actual, ok):
    if ok:
        check("things", claim, actual)
    else:
        with pytest.raises(AssertionError):
            check("things", claim, actual)
