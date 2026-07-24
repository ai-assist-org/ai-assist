# ai-assist Evaluation Suite

Evaluates whether the ai-assist agent selects the correct internal tools given natural language queries. Uses [agent-eval-harness](https://github.com/opendatahub-io/agent-eval-harness) as an optional dependency for config parsing and CLI runner infrastructure.

## Quick Start

```bash
# Install with eval dependency
uv sync --extra eval

# Run all 15 test cases
uv run --extra eval python eval/run_eval.py --config eval/eval.yaml --model claude-sonnet-4-6

# Run specific cases
uv run --extra eval python eval/run_eval.py --config eval/eval.yaml --model claude-sonnet-4-6 --cases kg-stats report-list

# Run cases in parallel
uv run --extra eval python eval/run_eval.py --config eval/eval.yaml --model claude-sonnet-4-6 --parallelism 4

# Skip LLM judges (faster, only check judges)
uv run --extra eval python eval/run_eval.py --config eval/eval.yaml --model claude-sonnet-4-6 --no-llm-judges
```

The process exits with code 0 on all-pass, 1 on any failure.

## What It Tests

The eval suite tests **tool selection from natural language** — the gap not covered by unit tests (which mock the LLM). Each case provides a prompt and checks whether the agent calls the expected internal tool.

### Tool Categories (15 cases)

| Category | Cases | Tools Tested |
|----------|-------|-------------|
| Knowledge Graph | `kg-stats`, `kg-save`, `kg-search`, `kg-recent` | `kg_stats`, `save_knowledge`, `search_knowledge`, `kg_recent_changes` |
| Reports | `report-write`, `report-list`, `report-read` | `write_report`, `list_reports`, `read_report` |
| Schedules | `schedule-list`, `schedule-create`, `action-schedule` | `list_schedules`, `create_monitor`/`create_task`, `schedule_action` |
| Filesystem | `fs-read`, `fs-list`, `fs-search` | `read_file`, `list_directory`, `search_in_file` |
| Misc | `think`, `tool-help` | `think`, `get_tool_help` |

### Judges

| Judge | Type | What It Checks |
|-------|------|---------------|
| `tool_selection` | check | Agent called at least one expected tool (substring match on tool name) |
| `no_wrong_category` | check | Agent did not call tools from a forbidden category |
| `response_quality` | LLM | Response is coherent and addresses the prompt (requires Claude Code CLI; skipped with `--no-llm-judges` or if `claude` is not installed) |

## How It Works

```
eval.yaml          ─── config (runner, judges, dataset path)
eval_wrapper.py    ─── bridges eval harness → ai-assist agent API
run_eval.py        ─── self-contained pipeline (workspace → execute → collect → score)
dataset/cases/     ─── test cases (input.yaml + annotations.yaml per case)
```

1. **`run_eval.py`** parses `eval.yaml`, discovers cases, and creates temporary workspaces
2. For each case, it invokes **`eval_wrapper.py`** via the CLI runner as a subprocess
3. **`eval_wrapper.py`** creates a minimal ai-assist agent (internal tools only, isolated KG), runs the query, and writes `response.txt`, `tool_calls.json`, and `metrics.json` to the output directory
4. **`run_eval.py`** runs check judges against the outputs and prints a pass/fail table

## Adding Test Cases

Create a new directory under `eval/dataset/cases/<case-name>/` with:

**`input.yaml`** — the prompt:
```yaml
prompt: "Your natural language query here"
```

**`annotations.yaml`** — expected behavior:
```yaml
expected_tools:
  - "tool_name_substring"     # at least one must be called
forbidden_tools:
  - "wrong_category_prefix"   # none of these should be called
```

Tool matching is substring-based: `"list_reports"` matches `"internal__list_reports"`.

**`fixtures/`** (optional) — files to pre-seed in the workspace before the agent runs:
```
fixtures/
  reports/status.md    # pre-seeded report for read_report tests
```

The `fixtures/` directory is copied into the workspace root, so `fixtures/reports/status.md` becomes accessible at the `AI_ASSIST_REPORTS_DIR` path.

## Output

Results are saved to `eval/runs/<skill>/<run-id>/`:

```
eval/runs/ai-assist-internal-tools/2026-06-02-claude-sonnet-4-6/
  summary.yaml              # aggregated judge results
  cases/
    report-list/
      output/
        response.txt        # agent response
        tool_calls.json     # tools called [{name, input}]
        metrics.json        # token usage, turns, model, cost_usd
      run_result.json       # exit code, duration, cost
      stdout.log
      stderr.log
```

## Prerequisites

- `ANTHROPIC_API_KEY` or Vertex AI credentials configured in `~/.ai-assist/.env`
- [Claude Code](https://claude.com/claude-code) installed for the `response_quality` LLM judge (optional — use `--no-llm-judges` to skip)
- For filesystem test cases (`fs-read`, `fs-search`): create `/tmp/test-eval-file.txt` with test content before running

## Cost

Each case makes 1-3 LLM API calls (agent query). With `claude-sonnet-4-6`, a full 15-case run costs approximately $0.10-0.30 depending on tool call complexity.

Token costs are automatically tracked in query traces (`~/.ai-assist/traces/query_traces.jsonl`) with per-turn `cost_usd` and per-query `total_cost_usd` fields. Use `/cost` or `/cost 30d` to view cost summaries, or `/eval-stats` for aggregate metrics including cost.
