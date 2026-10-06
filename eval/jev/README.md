# Measuring jev vs the LLM baseline for AWL goal-success

jev is integrated transparently: when it is configured, the AWL runtime's one
self-made judgment — "has this goal's success criterion been met?" in
`AWLRuntime._evaluate_goal_success` — is decided by jev's Noul (yes/no) instead
of an LLM turn. Everything else in a workflow is unchanged. This directory lets
you measure whether that swap helps.

## What is measured

`measure_goal_success.py` runs every labeled case in `cases.yaml` through the
*real* runtime method twice:

* **jev arm** — `AI_ASSIST_JEV_ENABLED=true` with a key, so jev decides.
* **baseline arm** — jev disabled, so an LLM turn decides (the original behavior).

Both arms set the same `_goal_success_met`, which is compared to the case's
ground-truth label. The script reports:

| metric            | meaning                                               |
|-------------------|-------------------------------------------------------|
| accuracy          | fraction of decided cases matching the label          |
| undecided         | cases where the arm produced no decision              |
| parse_failures    | baseline cases whose LLM output could not be parsed   |
| latency_median_s  | median wall-clock per decision                         |
| latency_p95_s     | 95th-percentile wall-clock per decision               |
| brier             | jev calibration: mean (p_yes − outcome)², lower better |

Cost is not auto-measured. Read jev input-token spend from your provider
dashboard (jev charges input tokens only) and baseline spend from the model
provider.

## Running it

```bash
# jev arm needs a key; baseline arm needs Anthropic API access.
export AI_ASSIST_JEV_API_KEY=...        # OpenRouter or TypeSafe key
uv run --extra eval python eval/jev/measure_goal_success.py            # both arms
uv run --extra eval python eval/jev/measure_goal_success.py --arm jev
uv run --extra eval python eval/jev/measure_goal_success.py --arm baseline
```

## Quick smoke test (one AWL script)

`jev-demo.awl` is the smallest script that exercises jev. Its `@goal` body only
sets a variable (no `@task`, so no LLM call), then the runtime evaluates the
success criterion. The printed `Success:` line tells you which engine decided:

```bash
export AI_ASSIST_JEV_API_KEY=...        # OpenRouter or TypeSafe key
uv run ai-assist /run eval/jev/jev-demo.awl
# Success: jev Noul p(yes)=0.93         <- jev decided

AI_ASSIST_JEV_ENABLED=false uv run ai-assist /run eval/jev/jev-demo.awl
# Success: <an LLM-written sentence>    <- LLM decided (jev off)
```

## Two datasets

* `cases.yaml` — 8 clean-cut, deterministic cases (e.g. `test_failures: 0` →
  pass). A sanity check: both arms should score ~100%. It does **not**
  discriminate decision quality.
* `cases_hard.yaml` — 12 judgment-heavy cases that need interpretation (weighing
  several signals, reading the criterion precisely, resisting a distractor).
  This is the set that actually tells the arms apart.

Run the hard set with `--cases eval/jev/cases_hard.yaml`.

## Measured results (2026-10-05, model claude-opus-4-6)

Easy set (`cases.yaml`, both arms 100% — as expected, no signal):

| metric           | jev   | baseline |
|------------------|-------|----------|
| accuracy         | 1.000 | 1.000    |
| latency_median_s | 0.30  | 2.05     |

Hard set (`cases_hard.yaml`) — the discriminating run:

| metric           | jev   | baseline |
|------------------|-------|----------|
| accuracy         | 0.833 (10/12) | 1.000 (12/12) |
| latency_median_s | 0.29  | 2.50     |
| latency_p95_s    | 0.32  | 3.92     |
| brier            | 0.103 | n/a      |

**Interpretation.** jev is ~8–9× faster and (per provider pricing) far cheaper,
and it reads precise criteria well — it correctly handled "no *critical* vulns
remain" (p=0.97), conflicting deploy-health signals, a missing API-doc
distractor, a non-reversible migration, and "*exactly* 3 reviewers" (p=0.95).
Its two misses were both on genuinely subjective judgments: a soft latency
threshold (said no at p=0.31 — low confidence) and a lukewarm 58+/49− feedback
split (said positive at p=0.76 — confidently wrong). The LLM baseline got all 12
but paid ~9× latency. Takeaway: for crisp, variable-grounded criteria jev is a
sound speed/cost swap; for fuzzy criteria it loses accuracy, and one of its two
errors was low-confidence — so confidence-threshold routing (defer p near 0.5 to
the LLM) is the obvious next experiment.

> Caveats: n=12, single model, single run; labels on the two subjective cases
> are themselves arguable. Re-run before trusting the exact numbers.

## Interactive jev tools — no system-prompt nudge (and why)

When jev is configured the interactive agent also gets three decision tools
(`internal__jev_decide`/`choose`/`score`) for ad-hoc calibrated judgments. There is
**deliberately no system-prompt guidance nudging the agent to use them** — the agent
reaches for them on its own when a decision fits.

We measured a nudge before removing it. A controlled A/B (claude-opus-4-6, 15
labeled decision prompts, guidance on vs off, jev tools available in both arms)
showed the nudge made **no measurable difference**: adoption 0.93/0.93 when jev was
the only decision tool, and 0.80/0.80 when it competed with ~50 other internal tools
— guided and unguided identical in both settings, accuracy 0.93 throughout.
Accuracy was in fact slightly *higher* when the agent answered itself (1.00, n=3)
than when it used jev (0.92), the persistent miss being an all-green release the jev
path called "don't ship". So nudging toward *more* jev adoption had no upside and a
small accuracy-downside risk. We dropped the nudge (and its `AI_ASSIST_JEV_TOOL_
GUIDANCE` flag and the A/B harness) rather than keep unjustified complexity.

## Extending the dataset

Add cases to `cases.yaml` or `cases_hard.yaml` (goal-success A/B). Each case is the
variable state after a cycle, a goal with a success criterion, and the ground-truth
`expected` yes/no. Keep cases generic (English, no personal data) so both arms see
identical neutral inputs.
