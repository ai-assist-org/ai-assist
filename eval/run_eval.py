#!/usr/bin/env python3
"""Self-contained eval runner for ai-assist.

Uses EvalConfig and CliRunner from agent-eval-harness (optional dep)
to run evaluation cases and score them with check judges.

Usage:
    uv run --extra eval python eval/run_eval.py --config eval/eval.yaml --model claude-sonnet-4-6
    uv run --extra eval python eval/run_eval.py --config eval/eval.yaml --model claude-sonnet-4-6 --cases kg-stats report-list
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml
from agent_eval.agent import RUNNERS
from agent_eval.config import EvalConfig


def discover_cases(config):
    dataset_dir = config.resolve_path(config.dataset_path)
    if not dataset_dir.is_dir():
        print(f"ERROR: dataset directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)
    return sorted(d.name for d in dataset_dir.iterdir() if d.is_dir() and (d / "input.yaml").exists())


def create_workspace(case_id, config, run_id):
    dataset_dir = config.resolve_path(config.dataset_path)
    workspace = Path(f"/tmp/agent-eval/{run_id}/cases/{case_id}")  # noqa: S108
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset_dir / case_id / "input.yaml", workspace / "input.yaml")
    answers = dataset_dir / case_id / "answers.yaml"
    if answers.exists():
        shutil.copy2(answers, workspace / "answers.yaml")
    # Copy fixtures into workspace (e.g., pre-seeded reports, files)
    fixtures = dataset_dir / case_id / "fixtures"
    if fixtures.is_dir():
        shutil.copytree(fixtures, workspace, dirs_exist_ok=True)
    return workspace


def collect_outputs(workspace, case_output_dir, config):
    case_output_dir.mkdir(parents=True, exist_ok=True)
    for out in config.outputs:
        if not out.path:
            continue
        src_dir = workspace / out.path
        if not src_dir.is_dir():
            continue
        dst_dir = case_output_dir / out.path
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in src_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, dst_dir / f.name)
    for log in ("stdout.log", "stderr.log", "run_result.json"):
        src = case_output_dir / log
        if not src.exists():
            ws_log = workspace / log
            if ws_log.exists():
                shutil.copy2(ws_log, src)


def load_case_record(case_dir, config):
    case_dir = Path(case_dir)
    record = {"files": {}, "case_dir": str(case_dir)}
    case_id = case_dir.name
    dataset_dir = config.resolve_path(config.dataset_path)

    ann_path = dataset_dir / case_id / "annotations.yaml"
    if ann_path.is_file():
        with open(ann_path) as f:
            record["annotations"] = yaml.safe_load(f) or {}
    else:
        record["annotations"] = {}

    for out in config.outputs:
        if not out.path:
            continue
        out_dir = case_dir / out.path
        if not out_dir.is_dir():
            continue
        for f in sorted(out_dir.iterdir()):
            if f.is_file():
                try:
                    record["files"][f"{out.path}/{f.name}"] = f.read_text()
                except UnicodeDecodeError:
                    pass
        dirname = out.path.replace("/", "_")
        files_in_dir = [f for f in sorted(out_dir.iterdir()) if f.is_file()]
        if files_in_dir:
            try:
                record[f"{dirname}_content"] = files_in_dir[0].read_text()
                record[f"{dirname}_file"] = str(files_in_dir[0])
            except UnicodeDecodeError:
                pass

    rr_path = case_dir / "run_result.json"
    if rr_path.is_file():
        rr = json.loads(rr_path.read_text())
        for key in ("exit_code", "duration_s", "token_usage", "cost_usd", "num_turns"):
            if key in rr:
                record[key] = rr[key]

    for log_name in ("stdout", "stderr"):
        log_path = case_dir / f"{log_name}.log"
        if log_path.is_file():
            record[log_name] = log_path.read_text()

    dataset_input = dataset_dir / case_id / "input.yaml"
    if dataset_input.is_file():
        record["input_path"] = str(dataset_input)

    return record


# ── Judges ────────────────────────────────────────────────────────


def make_check_judge(name, check_source, arguments):
    wrapped = f"def _check(outputs, arguments):\n{textwrap.indent(check_source, '    ')}"
    code = compile(wrapped, f"<check:{name}>", "exec")
    ns = {"__builtins__": __builtins__}
    exec(code, ns)  # noqa: S102
    check_fn = ns["_check"]

    def scorer(outputs=None):
        return check_fn(outputs or {}, arguments)

    return scorer


def _parse_llm_response(response):
    try:
        text = response
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text.strip())
        return parsed.get("passed", False), parsed.get("rationale", "")
    except json.JSONDecodeError, ValueError:
        return "passed" in response.lower() or "true" in response.lower(), response[:200]


def make_llm_judge(name, jc, config):
    prompt_template = jc.prompt
    if not prompt_template and jc.prompt_file:
        prompt_template = config.resolve_path(jc.prompt_file).read_text()

    def scorer(outputs=None):
        from jinja2 import Environment

        out = outputs or {}
        inputs = {}
        input_path = out.get("input_path")
        if input_path and Path(input_path).is_file():
            inputs = yaml.safe_load(Path(input_path).read_text()) or {}

        rendered = Environment().from_string(prompt_template).render(inputs=inputs, outputs=out)

        try:
            result = subprocess.run(
                ["claude", "--print", "--bare", "--model", "sonnet"],
                input=rendered,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "LLM judge timed out"
        if result.returncode != 0:
            return False, f"claude exited {result.returncode}: {result.stderr.strip()[:200]}"
        if not result.stdout.strip():
            return False, "LLM judge returned empty response"
        return _parse_llm_response(result.stdout.strip())

    return scorer


def _has_claude_cli():
    return shutil.which("claude") is not None


def load_judges(config, no_llm_judges=False):
    judges = []
    for jc in config.judges:
        if jc.check:
            scorer = make_check_judge(jc.name, jc.check, jc.arguments)
            judges.append((jc.name, scorer, jc.condition, "check"))
        elif jc.prompt or jc.prompt_file:
            if no_llm_judges:
                judges.append((jc.name, None, jc.condition, "llm"))
            elif _has_claude_cli():
                judges.append((jc.name, make_llm_judge(jc.name, jc, config), jc.condition, "llm"))
            else:
                print(f"  Warning: skipping LLM judge '{jc.name}' (claude CLI not found)", file=sys.stderr)
                judges.append((jc.name, None, jc.condition, "llm"))
        elif jc.builtin:
            judges.append((jc.name, None, jc.condition, "builtin"))
    return judges


def score_case(case_dir, judges, config):
    record = load_case_record(case_dir, config)
    results = {}
    for name, scorer, condition, judge_type in judges:
        if scorer is None:
            results[name] = {"value": None, "rationale": f"Skipped ({judge_type} judge)", "judge_type": judge_type}
            continue
        if condition:
            try:
                annotations = record.get("annotations", {})
                if not eval(
                    condition, {"__builtins__": {}}, {"annotations": annotations, "outputs": record}
                ):  # noqa: S307
                    results[name] = {"value": None, "rationale": "Skipped: condition false", "judge_type": judge_type}
                    continue
            except Exception as e:
                results[name] = {"value": None, "rationale": f"Condition error: {e}", "judge_type": judge_type}
                continue
        try:
            result = scorer(outputs=record)
            if isinstance(result, tuple) and len(result) == 2:
                results[name] = {"value": result[0], "rationale": result[1], "judge_type": judge_type}
            else:
                results[name] = {"value": result, "rationale": "", "judge_type": judge_type}
        except Exception as e:
            results[name] = {"value": None, "error": str(e), "judge_type": judge_type}
    return results


# ── Pipeline ──────────────────────────────────────────────────────


def execute_cases(all_cases, config, runner, model, run_output, run_id, parallelism):
    timeout = config.execution.timeout or 300
    max_budget = config.execution.max_budget_usd or 5.0
    args_template = config.execution.arguments or ""
    case_results = {}
    total_start = time.monotonic()

    def run_case(case_id):
        workspace = create_workspace(case_id, config, run_id)
        input_data = yaml.safe_load((workspace / "input.yaml").read_text()) or {}
        args = args_template
        for k, v in input_data.items():
            args = args.replace(f"{{{k}}}", str(v) if v is not None else "")

        result = runner.run_skill(
            skill_name=config.skill,
            args=args,
            workspace=workspace,
            model=model,
            timeout_s=timeout,
            max_budget_usd=max_budget,
        )

        case_out = run_output / "cases" / case_id
        case_out.mkdir(parents=True, exist_ok=True)
        if result.stdout:
            (case_out / "stdout.log").write_text(result.stdout)
        if result.stderr:
            (case_out / "stderr.log").write_text(result.stderr)
        rr = {
            "exit_code": result.exit_code,
            "duration_s": result.duration_s,
            "token_usage": result.token_usage,
            "cost_usd": result.cost_usd,
            "num_turns": result.num_turns,
            "model": result.resolved_model or model,
        }
        (case_out / "run_result.json").write_text(json.dumps(rr, indent=2))
        collect_outputs(workspace, case_out, config)
        status = "OK" if result.exit_code == 0 else f"FAIL({result.exit_code})"
        print(f"  [{case_id}] {status} ({result.duration_s:.0f}s)")
        return case_id, result

    print("Executing cases...")
    with ThreadPoolExecutor(max_workers=max(1, parallelism)) as pool:
        futures = {pool.submit(run_case, c): c for c in all_cases}
        for future in as_completed(futures):
            try:
                cid, result = future.result()
                case_results[cid] = result
            except Exception as e:
                print(f"  [{futures[future]}] ERROR: {e}", file=sys.stderr)

    wall_clock = time.monotonic() - total_start
    print(f"\nExecution complete: {len(case_results)}/{len(all_cases)} cases in {wall_clock:.0f}s")
    return case_results


def score_and_report(run_output, config, judges):
    case_dirs = sorted((run_output / "cases").iterdir()) if (run_output / "cases").is_dir() else []
    per_case = {}
    aggregated = {name: {"values": []} for name, *_ in judges}

    for i, case_dir in enumerate(case_dirs, 1):
        case_id = case_dir.name
        print(f"  [{i}/{len(case_dirs)}] {case_id}...", end="", flush=True)
        results = score_case(case_dir, judges, config)
        per_case[case_id] = results
        failed = [n for n, r in results.items() if r.get("value") is False]
        print(f" FAIL ({', '.join(failed)})" if failed else " ok")
        for name, r in results.items():
            if name in aggregated and r.get("value") is not None:
                aggregated[name]["values"].append(r["value"])

    for _name, agg in aggregated.items():
        values = agg["values"]
        if not values:
            agg["mean"] = None
            agg["pass_rate"] = None
        elif all(isinstance(v, bool) for v in values):
            agg["pass_rate"] = sum(values) / len(values)
            agg["mean"] = agg["pass_rate"]
        elif all(isinstance(v, (int, float)) for v in values):
            agg["mean"] = sum(values) / len(values)
            agg["pass_rate"] = None

    summary = {"judges": {}, "per_case": {}}
    for name, agg in aggregated.items():
        summary["judges"][name] = {k: v for k, v in agg.items() if k != "values"}
    for case_id, results in per_case.items():
        summary["per_case"][case_id] = {
            name: {"value": r["value"], "rationale": r.get("rationale", "")} for name, r in results.items()
        }
    (run_output / "summary.yaml").write_text(yaml.dump(summary, default_flow_style=False))
    return per_case, aggregated


def print_results(config, run_id, per_case, aggregated, judges, run_output):
    print("\n" + "=" * 70)
    print(f"Results: {config.skill} / {run_id}")
    print("=" * 70)
    for name, agg in aggregated.items():
        if agg.get("pass_rate") is not None:
            print(f"  {name}: {agg['pass_rate'] * 100:.0f}% pass rate")
        elif agg.get("mean") is not None:
            print(f"  {name}: {agg['mean']:.2f} mean")
        else:
            print(f"  {name}: (no data)")

    judge_names = [n for n, *_ in judges]
    has_failures = False
    print()
    for case_id in sorted(per_case):
        verdicts = []
        for jn in judge_names:
            r = per_case[case_id].get(jn, {})
            val = r.get("value")
            if val is True:
                verdicts.append(("PASS", ""))
            elif val is False:
                has_failures = True
                verdicts.append(("FAIL", r.get("rationale", "")))
            elif val is None:
                verdicts.append(("SKIP", r.get("rationale", "")))
            else:
                verdicts.append((str(val), r.get("rationale", "")))

        if all(v in {"PASS", "SKIP"} for v, _ in verdicts):
            print(f"  PASS  {case_id}")
        else:
            print(f"  FAIL  {case_id}")
            for jn, (verdict, rationale) in zip(judge_names, verdicts, strict=False):
                if verdict == "FAIL":
                    print(f"        {jn}: {rationale}")

    print(f"\nOutput: {run_output}")
    return has_failures


def run_pipeline(config_path, model, run_id=None, cases=None, parallelism=1, no_llm_judges=False):
    config = EvalConfig.from_yaml(config_path)

    # Resolve {project_root} in runner command to the config file's parent directory
    project_root = str(Path(config_path).resolve().parent.parent)
    config.runner.command = [c.replace("{project_root}", project_root) for c in config.runner.command]

    if not run_id:
        run_id = f"{datetime.now().strftime('%Y-%m-%d')}-{model}"

    all_cases = discover_cases(config)
    if cases:
        all_cases = [c for c in all_cases if c in cases]
    if not all_cases:
        print("No cases found.", file=sys.stderr)
        sys.exit(1)

    runs_dir = Path(os.environ.get("AGENT_EVAL_RUNS_DIR", "eval/runs"))
    run_output = runs_dir / config.skill / run_id

    print(f"Eval: {config.skill}")
    print(f"Run ID: {run_id}")
    print(f"Model: {model}")
    print(f"Cases: {len(all_cases)}")
    print()

    runner_type = config.runner.type
    if runner_type not in RUNNERS:
        print(f"ERROR: unknown runner type '{runner_type}'. Available: {list(RUNNERS.keys())}", file=sys.stderr)
        sys.exit(1)
    runner = RUNNERS[runner_type].from_config(config)

    execute_cases(all_cases, config, runner, model, run_output, run_id, parallelism)

    print("\nScoring...")
    judges = load_judges(config, no_llm_judges=no_llm_judges)
    per_case, aggregated = score_and_report(run_output, config, judges)
    has_failures = print_results(config, run_id, per_case, aggregated, judges, run_output)

    if has_failures:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run ai-assist evaluation suite")
    parser.add_argument("--config", required=True, help="Path to eval.yaml")
    parser.add_argument("--model", required=True, help="Model identifier")
    parser.add_argument("--run-id", help="Custom run ID (default: auto-generated)")
    parser.add_argument("--cases", nargs="*", help="Run specific cases only")
    parser.add_argument("--parallelism", type=int, default=1, help="Concurrent cases")
    parser.add_argument("--no-llm-judges", action="store_true", help="Skip LLM-based judges")
    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        model=args.model,
        run_id=args.run_id,
        cases=args.cases,
        parallelism=args.parallelism,
        no_llm_judges=args.no_llm_judges,
    )


if __name__ == "__main__":
    main()
