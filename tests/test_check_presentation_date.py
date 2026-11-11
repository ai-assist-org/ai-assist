"""scripts/check_presentation_date.py: a modified presentation must carry the current month in \\date."""

import importlib.util
import subprocess
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "check_presentation_date", ROOT / "scripts" / "check_presentation_date.py"
)
assert _spec and _spec.loader
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

TODAY = date(2026, 11, 11)


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("\\title{x}\n\\date{November 2026}\n", None),
        ("\\date{October 2026}\n", "\\date{October 2026} should be \\date{November 2026}"),
        ("\\date{\\today}\n", "\\date{\\today} should be \\date{November 2026}"),
        ("\\title{x}\n", "no \\date{...} line"),
    ],
)
def test_date_problem(text, problem):
    assert mod.date_problem(text, TODAY) == problem


def test_month_names_do_not_depend_on_the_locale():
    assert mod.date_problem("\\date{January 2027}\n", date(2027, 1, 3)) is None


def git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@example.com")
    git(tmp_path, "config", "user.name", "t")
    git(tmp_path, "config", "commit.gpgsign", "false")
    slides = tmp_path / "slides.tex"
    slides.write_text("\\date{January 2000}\n")
    git(tmp_path, "add", "slides.tex")
    git(tmp_path, "commit", "-q", "-m", "init", "--no-verify")
    return tmp_path, slides


def test_unmodified_presentation_is_not_checked(repo):
    """CI runs every hook on all files: an untouched presentation must not fail."""
    root, slides = repo
    assert mod.main([str(slides)], today=TODAY, cwd=root) == 0


@pytest.mark.parametrize("stage", [True, False])
def test_modified_presentation_needs_the_current_month(repo, stage, capsys):
    root, slides = repo
    slides.write_text("\\date{January 2000}\n\\title{changed}\n")
    if stage:
        git(root, "add", "slides.tex")
    assert mod.main([str(slides)], today=TODAY, cwd=root) == 1
    assert "should be \\date{November 2026}" in capsys.readouterr().out

    slides.write_text("\\date{November 2026}\n\\title{changed}\n")
    assert mod.main([str(slides)], today=TODAY, cwd=root) == 0
