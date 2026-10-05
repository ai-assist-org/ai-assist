"""Tests for the jev goal-success measurement helpers (eval/jev/measure_goal_success.py)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "eval" / "jev" / "measure_goal_success.py"
_spec = importlib.util.spec_from_file_location("measure_goal_success", _MODULE_PATH)
mgs = importlib.util.module_from_spec(_spec)
sys.modules["measure_goal_success"] = mgs
_spec.loader.exec_module(mgs)


def test_load_cases_reads_dataset():
    cases = mgs.load_cases(_MODULE_PATH.parent / "cases.yaml")
    assert cases
    first = cases[0]
    assert first.name
    assert isinstance(first.expected, bool)
    assert isinstance(first.variables, dict)


def test_probability_from_reason_parses_jev_string():
    assert mgs.probability_from_reason("jev Noul p(yes)=0.91") == pytest.approx(0.91)
    assert mgs.probability_from_reason("jev Noul p(yes)=1") == pytest.approx(1.0)
    assert mgs.probability_from_reason("") is None
    assert mgs.probability_from_reason("done") is None


def _result(name, predicted, expected, probability=None, latency=0.1, parse_failed=False):
    return mgs.CaseResult(
        name=name,
        predicted=predicted,
        expected=expected,
        probability=probability,
        latency_s=latency,
        parse_failed=parse_failed,
    )


def test_summarize_accuracy_and_brier():
    results = [
        _result("a", True, True, probability=0.9),
        _result("b", False, True, probability=0.4),  # wrong
        _result("c", False, False, probability=0.1),
    ]
    s = mgs.summarize("jev", results)
    assert s.total == 3
    assert s.correct == 2
    assert s.undecided == 0
    assert s.accuracy == pytest.approx(2 / 3)
    # Brier = mean((p - outcome)^2) = ((0.9-1)^2 + (0.4-1)^2 + (0.1-0)^2)/3
    assert s.brier == pytest.approx((0.01 + 0.36 + 0.01) / 3)


def test_summarize_counts_undecided_and_parse_failures():
    results = [
        _result("a", None, True, parse_failed=True),
        _result("b", True, True),
    ]
    s = mgs.summarize("baseline", results)
    assert s.undecided == 1
    assert s.parse_failures == 1
    # Only decided cases count toward accuracy.
    assert s.accuracy == pytest.approx(1.0)
    assert s.brier is None


def test_p95_picks_high_value():
    assert mgs._p95([]) is None
    assert mgs._p95([0.5]) == 0.5
    assert mgs._p95([0.1, 0.2, 0.3, 0.4, 100.0]) == 100.0


def test_format_table_includes_metrics_and_arms():
    s = mgs.summarize("jev", [_result("a", True, True, probability=0.9)])
    table = mgs.format_table([s])
    assert "accuracy" in table
    assert "brier" in table
    assert "jev" in table


@pytest.mark.asyncio
async def test_run_case_jev_arm_uses_probability():
    config = SimpleNamespace(jev_enabled=True, jev_api_key="k", jev_model="m", jev_api_url="u")
    agent = mgs._ShimAgent(config)
    case = mgs.Case(name="c", goal_id="g", success_criteria="done?", variables={"x": 1}, expected=True)

    async def fake_decide(config, state, questions):
        return {"answers": {"success_met": {"type": "noul", "noul": 0.8}}}

    with patch("ai_assist.awl_runtime.jev_decide", side_effect=fake_decide):
        res = await mgs._run_case(agent, case)

    assert res.predicted is True
    assert res.probability == pytest.approx(0.8)
    assert res.parse_failed is False


@pytest.mark.asyncio
async def test_run_case_baseline_arm_uses_llm():
    config = SimpleNamespace(jev_enabled=False, jev_api_key=None, model="m", model_tiers={})
    agent = SimpleNamespace(
        config=config,
        query=AsyncMock(return_value='```json\n{"success_met": false, "reason": "not yet"}\n```'),
    )
    case = mgs.Case(name="c", goal_id="g", success_criteria="done?", variables={"x": 1}, expected=False)
    res = await mgs._run_case(agent, case)
    assert res.predicted is False
    assert res.probability is None
    assert res.parse_failed is False
    agent.query.assert_awaited_once()
