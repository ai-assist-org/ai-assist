#!/usr/bin/env python3
"""Measure jev goal-success accuracy against the LLM baseline.

This is a controlled A/B over the AWL runtime's one self-made judgment,
``AWLRuntime._evaluate_goal_success``. Each labeled case in ``cases.yaml`` is
run through the *real* runtime code path twice:

* **jev arm**   — jev configured, so the yes/no is decided by jev's Noul.
* **baseline**  — jev disabled, so the same decision is made by an LLM turn.

Both arms set the same ``_goal_success_met``; we compare it to the case's
ground-truth label and report accuracy, latency, parse reliability, and (jev
only) a Brier calibration score. There is no bespoke scoring logic here: the
script drives production methods, so what it measures is what ships.

Requirements to produce real numbers:
* jev arm needs ``AI_ASSIST_JEV_API_KEY`` (and a reachable ``AI_ASSIST_JEV_URL``).
* baseline arm needs Anthropic API access (same as any agent run).

Usage::

    uv run --extra eval python eval/jev/measure_goal_success.py            # both arms
    uv run --extra eval python eval/jev/measure_goal_success.py --arm jev
    uv run --extra eval python eval/jev/measure_goal_success.py --arm baseline
"""

from __future__ import annotations

import argparse
import asyncio
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_REASON_PROB = re.compile(r"p\(yes\)=([0-9]*\.?[0-9]+)")


@dataclass
class Case:
    name: str
    goal_id: str
    success_criteria: str
    variables: dict[str, Any]
    expected: bool


@dataclass
class CaseResult:
    name: str
    predicted: bool | None  # None => the arm could not produce a decision
    expected: bool
    probability: float | None  # jev arm only
    latency_s: float
    parse_failed: bool  # baseline arm: LLM output could not be parsed


@dataclass
class ArmSummary:
    arm: str
    total: int = 0
    correct: int = 0
    undecided: int = 0
    parse_failures: int = 0
    latencies: list[float] = field(default_factory=list)
    brier_terms: list[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float | None:
        decided = self.total - self.undecided
        return self.correct / decided if decided else None

    @property
    def brier(self) -> float | None:
        return statistics.fmean(self.brier_terms) if self.brier_terms else None


def load_cases(path: Path) -> list[Case]:
    data = yaml.safe_load(path.read_text())
    return [
        Case(
            name=c["name"],
            goal_id=c["goal_id"],
            success_criteria=c["success_criteria"],
            variables=c.get("variables", {}),
            expected=bool(c["expected"]),
        )
        for c in data["cases"]
    ]


def probability_from_reason(reason: str) -> float | None:
    """Pull the yes-probability jev writes into ``_goal_success_reason``."""
    match = _REASON_PROB.search(reason or "")
    return float(match.group(1)) if match else None


def summarize(arm: str, results: list[CaseResult]) -> ArmSummary:
    s = ArmSummary(arm=arm)
    for r in results:
        s.total += 1
        if r.parse_failed:
            s.parse_failures += 1
        if r.predicted is None:
            s.undecided += 1
        else:
            if r.predicted == r.expected:
                s.correct += 1
            s.latencies.append(r.latency_s)
        if r.probability is not None:
            outcome = 1.0 if r.expected else 0.0
            s.brier_terms.append((r.probability - outcome) ** 2)
    return s


def format_table(summaries: list[ArmSummary]) -> str:
    def cell(v: float | None, fmt: str) -> str:
        return format(v, fmt) if v is not None else "n/a"

    rows = [("metric", *[s.arm for s in summaries])]
    rows.append(("cases", *[str(s.total) for s in summaries]))
    rows.append(("accuracy", *[cell(s.accuracy, ".3f") for s in summaries]))
    rows.append(("undecided", *[str(s.undecided) for s in summaries]))
    rows.append(("parse_failures", *[str(s.parse_failures) for s in summaries]))
    rows.append(
        ("latency_median_s", *[cell(statistics.median(s.latencies) if s.latencies else None, ".3f") for s in summaries])
    )
    rows.append(("latency_p95_s", *[cell(_p95(s.latencies), ".3f") for s in summaries]))
    rows.append(("brier", *[cell(s.brier, ".3f") for s in summaries]))

    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for ri, row in enumerate(rows):
        lines.append("  ".join(col.ljust(widths[i]) for i, col in enumerate(row)))
        if ri == 0:
            lines.append("  ".join("-" * widths[i] for i in range(len(row))))
    return "\n".join(lines)


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return ordered[idx]


class _ShimAgent:
    """Minimal agent for the jev arm: only ``.config`` is ever read."""

    def __init__(self, config: Any):
        self.config = config


async def _run_case(agent: Any, case: Case) -> CaseResult:
    # Imported lazily so the pure helpers above stay importable without pulling
    # in the whole runtime (keeps the unit tests light).
    from ai_assist.awl_ast import GoalNode
    from ai_assist.awl_runtime import AWLRuntime

    runtime = AWLRuntime(agent)
    runtime._variables = dict(case.variables)
    goal = GoalNode(goal_id=case.goal_id, success_criteria=case.success_criteria)

    start = time.perf_counter()
    await runtime._evaluate_goal_success(goal)
    latency = time.perf_counter() - start

    predicted = runtime._variables.get("_goal_success_met")
    reason = runtime._variables.get("_goal_success_reason", "")
    probability = probability_from_reason(reason)
    # Baseline parse failure: the LLM path leaves reason empty and defaults
    # success to False when it cannot parse a response.
    parse_failed = probability is None and reason == ""
    return CaseResult(
        name=case.name,
        predicted=bool(predicted) if predicted is not None else None,
        expected=case.expected,
        probability=probability,
        latency_s=latency,
        parse_failed=parse_failed,
    )


async def _run_jev_arm(cases: list[Case]) -> ArmSummary:
    from ai_assist.config import AiAssistConfig
    from ai_assist.jev_client import jev_configured

    config = AiAssistConfig.from_env()
    config.jev_enabled = True
    if not jev_configured(config):
        raise SystemExit("jev arm needs AI_ASSIST_JEV_API_KEY (and jev enabled).")
    agent = _ShimAgent(config)
    results = [await _run_case(agent, c) for c in cases]
    return summarize("jev", results)


async def _run_baseline_arm(cases: list[Case]) -> ArmSummary:
    from ai_assist.agent import AiAssistAgent
    from ai_assist.config import AiAssistConfig

    config = AiAssistConfig.from_env()
    config.jev_enabled = False  # force the LLM path even if a key is present
    agent = AiAssistAgent(config)
    # The goal-success judgment is a single max_turns=1 LLM call with no tool
    # use, so we deliberately skip connect_to_servers(): booting the MCP servers
    # adds minutes of startup (and can hang) while contributing nothing to the
    # decision, which would pollute the latency measurement.
    results = [await _run_case(agent, c) for c in cases]
    return summarize("baseline", results)


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).resolve().parent / "cases.yaml",
        help="Path to the labeled cases YAML.",
    )
    parser.add_argument(
        "--arm",
        choices=["jev", "baseline", "both"],
        default="both",
        help="Which arm(s) to run.",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)
    summaries: list[ArmSummary] = []
    if args.arm in ("jev", "both"):
        summaries.append(await _run_jev_arm(cases))
    if args.arm in ("baseline", "both"):
        summaries.append(await _run_baseline_arm(cases))

    print(f"\njev goal-success A/B over {len(cases)} labeled cases\n")
    print(format_table(summaries))
    print(
        "\nCost is not auto-measured: read jev input-token spend from your provider "
        "dashboard and baseline token spend from the model provider."
    )


if __name__ == "__main__":
    asyncio.run(_main())
