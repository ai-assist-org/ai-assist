"""scripts/check_action_pins.py: workflow actions must be pinned by commit SHA."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("check_action_pins", ROOT / "scripts" / "check_action_pins.py")
assert _spec and _spec.loader
check_action_pins = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_action_pins)

SHA = "3d3c42e5aac5ba805825da76410c181273ba90b1"


def problems(tmp_path, text):
    path = tmp_path / "wf.yml"
    path.write_text(text)
    return check_action_pins.check_file(path)


@pytest.mark.parametrize(
    "line",
    [
        "      - uses: actions/checkout@v7",
        "        uses: astral-sh/setup-uv@v7.6.0",
        "      - uses: actions/checkout@main",
        "      - uses: actions/checkout",
        '      - uses: "actions/checkout@v7"',
        "      - uses: actions/checkout@3d3c42e  # short sha",
        "    uses: org/repo/.github/workflows/build.yml@v1",
        "      - uses: docker://alpine:3.20",
    ],
)
def test_mutable_reference_is_flagged(tmp_path, line):
    found = problems(tmp_path, f"jobs:\n  a:\n    steps:\n{line}\n")
    assert len(found) == 1
    assert found[0].startswith(f"{tmp_path / 'wf.yml'}:4:")


@pytest.mark.parametrize(
    "line",
    [
        f"      - uses: actions/checkout@{SHA}  # v7.0.1",
        f"        uses: 'astral-sh/setup-uv@{SHA}'",
        f"    uses: org/repo/.github/workflows/build.yml@{SHA}",
        "      - uses: ./.github/actions/local",
        "      - uses: docker://alpine@sha256:" + "a" * 64,
        "      # - uses: actions/checkout@v7",
        "      - run: echo uses: actions/checkout@v7",
    ],
)
def test_immutable_or_unrelated_line_is_accepted(tmp_path, line):
    assert problems(tmp_path, f"jobs:\n  a:\n    steps:\n{line}\n") == []


def test_repository_workflows_are_pinned():
    files = sorted((ROOT / ".github" / "workflows").glob("*.y*ml"))
    assert files
    assert [p for f in files for p in check_action_pins.check_file(f)] == []


def test_cli_exit_status(tmp_path):
    bad = tmp_path / "bad.yml"
    bad.write_text("steps:\n  - uses: actions/checkout@v7\n")
    good = tmp_path / "good.yml"
    good.write_text(f"steps:\n  - uses: actions/checkout@{SHA}\n")
    script = str(ROOT / "scripts" / "check_action_pins.py")

    ok = subprocess.run([sys.executable, script, str(good)], capture_output=True, text=True, check=False)
    assert ok.returncode == 0 and ok.stdout == ""

    ko = subprocess.run([sys.executable, script, str(good), str(bad)], capture_output=True, text=True, check=False)
    assert ko.returncode == 1
    assert f"{bad}:2:" in ko.stdout
