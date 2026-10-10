# RCA Model Comparison Benchmark

Compares models on a real-world task: the `/redhat/rca <job_id>` prompt from
[dci-mcp-server](https://github.com/redhat-community-ai-tools/dci-mcp-server), which performs root-cause analysis
of a failing DCI CI/CD job (Evidence Gathering → 5-Whys → Adversarial
Challenge → Report).

Unlike `eval/` (ai-assist's internal-tools-only eval suite, replayed from
cassettes), this benchmark drives a real dci-mcp-server MCP connection
against live DCI/Jira/GitHub data, so it can only be run interactively with
real credentials — it's not part of CI.

## What it measures

For each candidate model, runs the full RCA prompt via
`AiAssistAgent.execute_mcp_prompt(server_name, "rca", {"dci_job_id": ...})`
and scores the resulting report with an LLM judge, using the same
7-criterion rubric as dci-mcp-server's own `tests/test_rca_eval.py`
(causal_depth, evidence_quality, adversarial_challenge,
confidence_calibration, report_structure, must_gather_usage,
actionable_recommendations — each 0-2). Also records wall-clock duration,
tool-call count, total tokens (input + output + cache write + cache read,
summed across turns — a useful cross-check when cost looks out of line with
duration/tool-call count, since cache-hit rate can vary a lot between
providers), and cost (via `ai_assist.pricing.compute_turn_cost`). The full
per-turn token breakdown is kept in each model's `meta.json`.

### Consensus check (correctness, not just per-report quality)

The 7-criterion judge scores each report in isolation, so it can't catch a
report that is confident, well-evidenced, and well-structured but still
reaches the *wrong* conclusion — it has no other report to compare against
and no ground truth for the job. When 2+ models produce a report,
`run_consensus_check()` makes one extra judge call with all reports together,
asking it to group models that agree on the same underlying root cause and
flag any that reach a genuinely different one (not just a shallower stop on
the same causal chain). Written to `consensus.json` and rendered as a
"Consensus check" section in `comparison.md`. This surfaces disagreement; it
doesn't prove which explanation is correct, but a report that diverges from
everyone else — especially when the majority cites harder evidence — is
worth a manual look before trusting it.

## Prerequisites

- Your `mcp_servers.yaml` (default: `get_config_dir()/mcp_servers.yaml`,
  i.e. `$AI_ASSIST_CONFIG_DIR` or `~/.ai-assist`) must have a working
  dci-mcp-server entry. The server's *name* in that file varies by instance
  config — it's `dci` in `~/.ai-assist/mcp_servers.yaml` but `redhat` in
  `~/.aut2/mcp_servers.yaml` on this machine — pass `--server-name` (default
  `dci`) and `--mcp-servers-file` to match whichever config you're using.
- Credentials for whichever models you select in `models.yaml`:
  - `ENMAAS_API_KEY` in your shell for the `enmaas-*` models.
  - A running [`litellm-proxy/`](../../litellm-proxy/) instance and
    `LITELLM_MASTER_KEY` for `openai-gpt-5`.
- A working default Anthropic credential (`ANTHROPIC_API_KEY` or Vertex ADC)
  in your ambient shell env — used only to judge reports, independent of the
  models under test (avoids self-bias and avoids pulling the judge through a
  proxy).

## Quick start

```bash
# All models in models.yaml, against the default pinned job
uv run python run_comparison.py -v

# Specific models and job
uv run python run_comparison.py \
  --job-id d47dd2ec-253e-4fe6-838c-952966b2bdff \
  --models enmaas-opus-5-5,enmaas-sonnet-5-5,openai-gpt-5 -v
```

Results land in `results/<job_id>/<timestamp>/`: one subdirectory per model
(`report.md`, `response.md`, `meta.json` with duration/cost/token usage/judge
scores) plus a top-level `comparison.md` summary table.

## Options

```
python run_comparison.py \
  --job-id <id> \            # DCI job to analyze (default: a previously-investigated job)
  --models-file models.yaml \
  --models name1,name2 \     # default: all entries in models.yaml
  --mcp-servers-file path \  # default: ~/.aut2/mcp_servers.yaml
  --max-time 1800 \          # per-model wall-clock budget in seconds
  --judge-model claude-haiku-4-5 \
  --output-dir results \
  -v
```

## Important: sequential execution only

The rca prompt always writes its report to a fixed path,
`/tmp/dci/rca-<job_id>.md`, regardless of which model is running it. The
script runs models one at a time and copies each report out immediately
after the run completes, before starting the next model. Do not run multiple
instances of this script concurrently against the same job ID — their
writes to that path would collide.

## Reproducibility caveats

DCI job records/files are effectively immutable once archived, but the
rendered prompt text can drift slightly between runs due to live PR/Jira
lookups embedded in it.

dci-mcp-server's file pre-download/staging under `/tmp/dci/<job_id>/` is
idempotent and keyed only on job ID (`prompts.py:_autodownload_triage_files`)
— it does **not** isolate per model run. `run_model()` therefore removes that
directory before every run, so each model downloads fresh and the tool-call
count/duration numbers reflect that model's own work, not a free ride on an
earlier model's cache.
