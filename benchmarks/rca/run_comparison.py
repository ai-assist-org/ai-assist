#!/usr/bin/env python3
"""Compare models on the DCI /redhat/rca root-cause-analysis task.

Runs the same real-world RCA prompt sequentially against each candidate model
(EnMaaS-hosted Claude/GLM models, OpenAI via litellm-proxy, etc.), then scores
each resulting report with an LLM judge using the rubric from dci-mcp-server's
own tests/test_rca_eval.py.

Usage:
    python run_comparison.py --job-id <dci_job_id> \
        --models enmaas-opus-5-5,enmaas-sonnet-5-5,openai-gpt-5

Prerequisites:
    - ~/.aut2/mcp_servers.yaml configured with working `redhat`/`tpci` servers
      (real DCI/Jira/GitHub credentials), same as interactive ai-assist use.
    - Credentials for whichever models.yaml entries you select (e.g.
      ENMAAS_API_KEY in your shell, litellm-proxy running locally for
      openai-gpt-5 — see litellm-proxy/README.md).
    - A working default Anthropic credential (ANTHROPIC_API_KEY or Vertex ADC)
      in your ambient shell env, used only for judging reports — independent
      of the models under test.

Runs are sequential, not parallel: the rca prompt always writes its report to
a fixed /tmp/dci/rca-<job_id>.md, so each model's report is copied out before
the next model's run starts.
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import string
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from judge_prompt import CONSENSUS_PROMPT, CRITERIA, JUDGE_PROMPT  # noqa: E402

from ai_assist.agent import AiAssistAgent  # noqa: E402
from ai_assist.config import AiAssistConfig  # noqa: E402
from ai_assist.pricing import compute_turn_cost  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_JOB_ID = "d47dd2ec-253e-4fe6-838c-952966b2bdff"

# Env vars that select the LLM provider/model; reset and reapplied per model run
# so each candidate gets a clean, isolated AiAssistConfig.from_env().
PROVIDER_ENV_KEYS = [
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "AI_ASSIST_API_KEY",
    "AI_ASSIST_MODEL",
    "AI_ASSIST_MODEL_MAX_TOKENS",
    "AI_ASSIST_MODEL_CONTEXT_WINDOW",
]


def _apply_model_env(model_env: dict) -> dict:
    """Clear provider env vars, apply model_env (${VAR} expanded), return prior values."""
    previous = {k: os.environ.get(k) for k in PROVIDER_ENV_KEYS}
    for k in PROVIDER_ENV_KEYS:
        os.environ.pop(k, None)
    for key, raw_value in model_env.items():
        value = string.Template(raw_value).safe_substitute(os.environ)
        if "${" in value:
            raise SystemExit(f"Missing required env var to resolve {key}={raw_value!r}")
        os.environ[key] = value
    return previous


def _restore_env(previous: dict) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


async def run_model(
    model_name: str,
    model_env: dict,
    job_id: str,
    *,
    mcp_servers_file: Path | None,
    server_name: str,
    max_time: int,
    out_dir: Path,
) -> tuple[dict, str | None]:
    """Run the RCA prompt against one model and capture its report + metrics."""
    report_path = Path(f"/tmp/dci/rca-{job_id}.md")  # nosec B108 - path fixed by dci-mcp-server
    report_path.unlink(missing_ok=True)
    # dci-mcp-server's file pre-download/staging is idempotent and keyed only on
    # job_id (prompts.py:_autodownload_triage_files), so without this it would be
    # shared across models run sequentially in this loop, giving later models a
    # free ride on an earlier model's downloads (fewer tool calls, shorter
    # duration) instead of a fair, independent run.
    shutil.rmtree(Path(f"/tmp/dci/{job_id}"), ignore_errors=True)  # nosec B108 - path fixed by dci-mcp-server

    model_dir = out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    previous_env = _apply_model_env(model_env)
    report_text: str | None = None
    try:
        config = AiAssistConfig.from_env(mcp_servers_file=mcp_servers_file)
        agent = AiAssistAgent(config)
        await agent.connect_to_servers()
        try:
            start = asyncio.get_event_loop().time()
            response_text = await agent.execute_mcp_prompt(
                server_name, "rca", {"dci_job_id": job_id}, max_time_seconds=max_time
            )
            duration = asyncio.get_event_loop().time() - start
            token_usage = agent.get_token_usage()
            tool_call_count = len(agent.last_tool_calls)
        finally:
            await agent.close()

        if not report_path.exists():
            raise RuntimeError(f"RCA report not found at {report_path}")
        report_text = report_path.read_text()
        if len(report_text) < 500:
            raise RuntimeError(f"RCA report is suspiciously short ({len(report_text)} chars)")

        cost = sum(compute_turn_cost(config.model, t, zero_if_unknown=True) for t in token_usage)
        total_tokens = sum(
            t["input_tokens"]
            + t["output_tokens"]
            + t.get("cache_creation_input_tokens", 0)
            + t.get("cache_read_input_tokens", 0)
            for t in token_usage
        )
        shutil.copy(report_path, model_dir / "report.md")
        (model_dir / "response.md").write_text(response_text)
        meta = {
            "model_name": model_name,
            "model_id": config.model,
            "duration_s": round(duration, 1),
            "cost_usd": round(cost, 4),
            "tool_call_count": tool_call_count,
            "total_tokens": total_tokens,
            "token_usage": token_usage,
        }
    except Exception as exc:
        logger.exception("Run failed for model %s", model_name)
        meta = {"model_name": model_name, "error": str(exc)}
        report_text = None
    finally:
        _restore_env(previous_env)

    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta, report_text


def run_judge(client, judge_model: str, report_text: str) -> dict:
    import re

    response = client.messages.create(
        model=judge_model,
        max_tokens=1024,
        messages=[{"role": "user", "content": JUDGE_PROMPT + report_text}],
    )
    answer = "".join(block.text for block in response.content if hasattr(block, "text"))
    match = re.search(r"\{[\s\S]*\}", answer)
    if not match:
        raise RuntimeError(f"Judge did not return valid JSON: {answer[:500]}")
    return json.loads(match.group())


def run_consensus_check(client, judge_model: str, reports: dict[str, str]) -> dict:
    """Compare independently-generated reports and flag disagreement on root cause.

    The per-report rubric judge (run_judge) scores each report in isolation, so
    it cannot catch a confident, well-evidenced report that simply reached the
    wrong conclusion because the others agree with each other instead. This
    makes one extra call with all reports together so that kind of outlier is
    visible without needing a pre-established ground truth.
    """
    import re

    body = "\n\n".join(f"### Report from `{name}`\n\n{text}" for name, text in reports.items())
    response = client.messages.create(
        model=judge_model,
        max_tokens=2048,
        messages=[{"role": "user", "content": CONSENSUS_PROMPT + body}],
    )
    answer = "".join(block.text for block in response.content if hasattr(block, "text"))
    match = re.search(r"\{[\s\S]*\}", answer)
    if not match:
        raise RuntimeError(f"Consensus check did not return valid JSON: {answer[:500]}")
    return json.loads(match.group())


def build_comparison_markdown(job_id: str, results: list[dict], consensus: dict | None = None) -> str:
    lines = [
        f"# RCA Model Comparison — job `{job_id}`",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    header = ["Model", *CRITERIA, "avg", "duration_s", "tool_calls", "total_tokens", "cost_usd"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for r in results:
        if "error" in r:
            lines.append(f"| {r['model_name']} | ERROR: {r['error']} |" + " |" * (len(header) - 2))
            continue
        scores = r.get("judge_scores", {}).get("scores", {})
        row_scores = [str(scores.get(c, "?")) for c in CRITERIA]
        avg = sum(scores.get(c, 0) for c in CRITERIA) / len(CRITERIA) if scores else 0
        row = [
            r["model_name"],
            *row_scores,
            f"{avg:.2f}",
            str(r["duration_s"]),
            str(r["tool_call_count"]),
            str(r["total_tokens"]),
            str(r["cost_usd"]),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    for r in results:
        if "error" in r:
            continue
        lines.append(f"## {r['model_name']}")
        lines.append(r.get("judge_scores", {}).get("overall_assessment", "(no judge assessment)"))
        lines.append("")

    if consensus:
        lines.append("## Consensus check")
        lines.append(consensus.get("summary", ""))
        lines.append("")
        for group in consensus.get("agreement_groups", []):
            models = ", ".join(f"`{m}`" for m in group.get("models", []))
            lines.append(f"- **Agree** ({models}): {group.get('shared_root_cause', '')}")
            if group.get("depth_note"):
                lines.append(f"  - {group['depth_note']}")
        for outlier in consensus.get("outliers", []):
            lines.append(f"- **Outlier** (`{outlier.get('model', '')}`): {outlier.get('divergent_root_cause', '')}")
            if outlier.get("reason"):
                lines.append(f"  - {outlier['reason']}")
        if consensus.get("strongest_evidence"):
            lines.append(f"\n**Strongest evidence:** {consensus['strongest_evidence']}")
        lines.append("")
    return "\n".join(lines)


async def main_async(args) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env", override=False)

    models_data = yaml.safe_load(Path(args.models_file).read_text())
    all_models = {m["name"]: m["env"] for m in models_data["models"]}
    selected = args.models.split(",") if args.models else list(all_models.keys())
    unknown = set(selected) - set(all_models)
    if unknown:
        raise SystemExit(f"Unknown model(s): {', '.join(sorted(unknown))}. Available: {', '.join(all_models)}")

    # Capture the judge's client from the ambient env, before any per-model env
    # swapping, so judging stays independent of the models under test. Reuses
    # AiAssistAgent's own client construction (custom endpoint/Vertex/direct
    # key branches) instead of duplicating it; never connects to any MCP
    # server, so .close() below is a no-op.
    judge_config = AiAssistConfig.from_env()
    judge_agent = AiAssistAgent(judge_config)
    judge_client = judge_agent.anthropic

    mcp_servers_file = Path(args.mcp_servers_file) if args.mcp_servers_file else None

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) / args.job_id / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    report_texts = {}
    for name in selected:
        logger.info("=== Running %s ===", name)
        meta, report_text = await run_model(
            name,
            all_models[name],
            args.job_id,
            mcp_servers_file=mcp_servers_file,
            server_name=args.server_name,
            max_time=args.max_time,
            out_dir=out_dir,
        )
        if report_text:
            report_texts[name] = report_text
            logger.info("Judging %s ...", name)
            try:
                meta["judge_scores"] = run_judge(judge_client, args.judge_model, report_text)
            except Exception:
                logger.exception("Judging failed for %s", name)
                meta["judge_error"] = "judge failed, see logs"
            (out_dir / name / "meta.json").write_text(json.dumps(meta, indent=2))
        results.append(meta)

    consensus = None
    if len(report_texts) >= 2:
        logger.info("Running consensus check across %d reports ...", len(report_texts))
        try:
            consensus = run_consensus_check(judge_client, args.judge_model, report_texts)
            (out_dir / "consensus.json").write_text(json.dumps(consensus, indent=2))
        except Exception:
            logger.exception("Consensus check failed")

    comparison_md = build_comparison_markdown(args.job_id, results, consensus)
    (out_dir / "comparison.md").write_text(comparison_md)
    print(comparison_md)
    logger.info("Full results in %s", out_dir)
    await judge_agent.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare models on the DCI /redhat/rca task")
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID, help="DCI job ID to analyze")
    parser.add_argument("--models-file", default=str(Path(__file__).parent / "models.yaml"))
    parser.add_argument("--models", default=None, help="Comma-separated model names (default: all in models.yaml)")
    parser.add_argument(
        "--mcp-servers-file",
        default=None,
        help="Path to mcp_servers.yaml (default: get_config_dir()/mcp_servers.yaml, "
        "i.e. $AI_ASSIST_CONFIG_DIR or ~/.ai-assist)",
    )
    parser.add_argument(
        "--server-name",
        default="dci",
        help="MCP server name for the dci-mcp-server instance in mcp_servers.yaml "
        "(default: dci; it's named 'redhat' in some instance configs, e.g. ~/.aut2)",
    )
    parser.add_argument("--max-time", type=int, default=1800, help="Max wall-clock seconds per model run")
    parser.add_argument("--judge-model", default="claude-haiku-4-5", help="Model used to judge each report")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "results"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
