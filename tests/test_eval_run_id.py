"""The eval runner's default run ID: one workspace and results directory per run."""

import importlib.util
import re
from pathlib import Path

import pytest

pytest.importorskip("agent_eval")


def _run_eval():
    path = Path(__file__).resolve().parent.parent / "eval" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("run_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_concurrent_runs_get_their_own_run_id():
    """Runs on the same day (e.g. the pre-commit replay hook during a live run) shared one and mixed results."""
    run_eval = _run_eval()
    first, second = run_eval.default_run_id("claude-sonnet-4-6"), run_eval.default_run_id("claude-sonnet-4-6")
    assert first != second
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}-\d{6}-claude-sonnet-4-6-[0-9a-f]{6}", first)  # sorts by start time
