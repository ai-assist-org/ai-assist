"""Tests for the self-improvement workflow and its deterministic helpers."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_assist.awl_executor import get_missing_variables, load_awl_workflow

WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "workflows" / "self-improvement"
LIB = WORKFLOW_DIR / "lib"
SOURCE_DIR = str(Path(__file__).resolve().parent.parent)

FLOW = """@start

@task gather
Goal: Gather the items.
Success: The items are listed.
Expose: items
@end

@task summarize
Goal: Summarize the items.
Expose: summary
@end

@end
"""


def run(script: str, *args: str, home: Path | None = None) -> str:
    env = dict(os.environ, AI_ASSIST_PYTHON=sys.executable)
    if home is not None:
        env["HOME"] = str(home)
    result = subprocess.run(["bash", str(LIB / script), *args], capture_output=True, text=True, env=env, check=True)
    return result.stdout


def counters(output: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line and " " not in line)


def log_line(time: str, pid: int, logger: str, level: str, msg: str) -> str:
    return f"2026-11-05 {time},000 - {pid} - {logger} - {level} - {msg}"


def shell_call(time: str, command: str) -> str:
    call = json.dumps({"command": command})
    return log_line(time, 100, "ai_assist.audit", "INFO", f"Tool call: internal__execute_command({call}) -> ok")


@pytest.fixture
def flow(tmp_path):
    path = tmp_path / "flows" / "demo.awl"
    path.parent.mkdir()
    path.write_text(FLOW)
    return path


@pytest.fixture
def signals(tmp_path, flow):
    logs = tmp_path / "logs"
    logs.mkdir()
    lock = json.dumps({"script": str(flow), "resolved_path": str(flow)})
    released = json.dumps({"duration_s": 12.5, "script": str(flow), "resolved_path": str(flow)})
    lines = [
        log_line("06:00:00", 100, "ai_assist.awl_executor", "INFO", f"AWL script lock acquired: {lock}"),
        log_line("06:00:01", 100, "ai_assist.awl_runtime", "INFO", "AWL task 'gather' starting"),
        log_line(
            "06:00:02",
            100,
            "ai_assist.awl_runtime",
            "WARNING",
            "AWL task 'gather': unresolved variable(s) in prompt: item_id",
        ),
        log_line("06:00:03", 100, "ai_assist.awl_runtime", "WARNING", "AWL task 'gather': 2 security rejection(s):"),
        f"    [!] internal__execute_command: Error: Command '{tmp_path}/flows/tool.sh' is not in the allowed"
        " commands list: grep, ls",
        "    [!] internal__execute_skill_script: Error: Skill 'demo-skill' not allowed to execute scripts"
        " (missing 'internal__execute_skill_script' in allowed-tools)",
        shell_call("06:00:03", "sed -n 10,20p /var/log/app.log | grep error"),
        shell_call("06:00:03", "sed -n 30,40p /var/log/app.log | grep error"),
        shell_call("06:00:03", "cd /srv && grep -rl TODO ."),
        shell_call("06:00:03", "git status"),
        log_line("06:00:04", 100, "ai_assist.awl_runtime", "ERROR", "AWL task 'gather' failed"),
        log_line("06:00:05", 100, "ai_assist.awl_executor", "INFO", f"AWL script lock released: {released}"),
        log_line("07:00:00", 100, "ai_assist.action_scheduler", "INFO", "Running nightly..."),
        log_line("07:00:00", 101, "ai_assist.action_scheduler", "INFO", "Running nightly..."),
        log_line("07:00:09", 100, "ai_assist.action_scheduler", "ERROR", "nightly: failed - boom"),
    ]
    (logs / "ai-assist.log").write_text("\n".join(lines) + "\n")
    out = tmp_path / "signals.jsonl"
    run(
        "awl-signals-aggregate.sh",
        "7",
        str(out),
        str(logs),
        str(tmp_path / "signals.tsv"),
        SOURCE_DIR,
        str(tmp_path / "cfg"),
    )
    return out


def records(path: Path, kind: str) -> list[dict]:
    return [r for r in map(json.loads, path.read_text().splitlines()) if r["kind"] == kind]


def test_workflow_needs_no_input():
    path = WORKFLOW_DIR / "self-improvement.awl"
    workflow, _ = load_awl_workflow(str(path))
    assert get_missing_variables(workflow, {"days": "7"}, path) == set()


def test_signals_attribute_events_to_the_workflow(signals, flow):
    found = {(r["script"], r["task"], r["event"]) for r in records(signals, "signal")}
    assert ("demo.awl", "gather", "unresolved_var") in found
    assert ("demo.awl", "gather", "command_refused") in found
    assert ("demo.awl", "gather", "skill_refused") in found
    assert ("demo.awl", "gather", "task_fail") in found
    assert ("scheduler", "nightly", "action_duplicate") in found
    assert ("scheduler", "nightly", "action_fail") in found
    [script] = records(signals, "script")
    assert script["script"] == "demo.awl"
    assert script["path"] == str(flow)
    [action] = records(signals, "action")
    assert (action["runs"], action["duplicate"], action["fail"]) == (2, 1, 1)


def test_shell_commands_an_internal_tool_could_run_are_rolled_up(signals):
    shell = {(r["tool"], r["form"]): r for r in records(signals, "shell")}
    assert set(shell) == {
        ("internal__read_file", "sed -n N,Mp | ..."),
        ("internal__search_in_file", "grep -rl"),
    }
    sed = shell[("internal__read_file", "sed -n N,Mp | ...")]
    assert (sed["count"], sed["top_task"], sed["top_task_count"]) == (2, "demo.awl::gather", 2)
    assert not [r for r in records(signals, "signal") if r["event"] == "shell_instead_of_tool"]


def test_ledger_reports_each_signal_once(signals, tmp_path):
    state = str(tmp_path / "state.jsonl")
    first = counters(run("awl-signals-diff.sh", str(signals), state, "--record-state"))
    second = counters(run("awl-signals-diff.sh", str(signals), state))
    assert int(first["new_signals"]) > 0
    assert second["new_signals"] == "0"


def test_ledger_reports_a_signal_again_when_it_comes_back(signals, tmp_path):
    state = str(tmp_path / "state.jsonl")
    run("awl-signals-diff.sh", str(signals), state, "--record-state")
    lines = [json.loads(line) for line in signals.read_text().splitlines()]
    for rec in lines:
        if rec.get("event") == "unresolved_var":
            rec["last"] = "2026-11-09T06:00:00,000"
    later = tmp_path / "later.jsonl"
    later.write_text("\n".join(json.dumps(rec) for rec in lines) + "\n")

    again = run("awl-signals-diff.sh", str(later), state, "--record-state")
    assert counters(again)["new_signals"] == "1"
    assert "unresolved_var" in again
    assert counters(run("awl-signals-diff.sh", str(later), state))["new_signals"] == "0"


@pytest.fixture
def config(tmp_path):
    cfg = tmp_path / "cfg"
    cfg.mkdir(exist_ok=True)
    (cfg / "allowed_commands.json").write_text(json.dumps(["ls", "\nfor f", "bash"]))
    (cfg / "allowed_paths.json").write_text(json.dumps([str(tmp_path), "/does/not/exist"]))
    skill = tmp_path / "skills" / "demo-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: demo-skill\nallowed-tools: internal__execute_skill_script, internal__execute_command\n---\n"
    )
    (cfg / "installed-skills.json").write_text(
        json.dumps({"skills": [{"name": "demo-skill", "cache_path": str(skill)}]})
    )
    return cfg


def refusal(cmd: str, script: str) -> str:
    return json.dumps(
        {
            "kind": "signal",
            "event": "command_refused",
            "script": script,
            "task": "t",
            "sample": cmd,
            "count": 3,
            "last": "2026-11-05T06:00:00",
        }
    )


def test_config_lint(tmp_path, config):
    tool = tmp_path / "bin" / "tool.sh"
    tool.parent.mkdir()
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o755)
    in_flow = tmp_path / "bin" / "flow-only.sh"
    in_flow.write_text("#!/bin/sh\n")
    in_flow.chmod(0o755)
    signals = tmp_path / "s.jsonl"
    signals.write_text(refusal(str(tool), "unknown") + "\n" + refusal(str(in_flow), "demo.awl") + "\n")

    out = run("awl-check-config.sh", "lint", str(config), str(signals), str(tmp_path / "p"), SOURCE_DIR, home=tmp_path)

    assert 'junk entry "\\nfor f"' in out
    assert 'junk entry "/does/not/exist"' in out
    assert '"bash" allows arbitrary code' in out
    assert f'CANDIDATE allowed_commands.json add "{tool}"' in out
    assert f'NOTE "{in_flow}"' in out and "name it in backticks" in out
    assert "LINT skill demo-skill: allowed-tools" in out
    assert counters(out)["config_issues"] == "4"
    assert (tmp_path / "p" / "live-installed-skills.json").exists()


def test_allowlist_gate_accepts_only_policy_changes(tmp_path, config):
    tool = tmp_path / "bin" / "tool.sh"
    tool.parent.mkdir()
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o755)
    signals = tmp_path / "s.jsonl"
    signals.write_text(refusal(str(tool), "unknown") + "\n")
    pdir = tmp_path / "p"
    pdir.mkdir()
    (pdir / "01-allowed_commands.json").write_text(json.dumps(["ls", "bash", str(tool)]))
    (pdir / "02-allowed_paths.json").write_text(json.dumps([str(tmp_path), "/etc"]))
    (pdir / "live-allowed_commands.json").write_text("[]")

    out = run("awl-check-config.sh", "gate", str(config), str(signals), str(pdir), SOURCE_DIR, home=tmp_path)

    assert "OK 01-allowed_commands.json" in out
    assert "REJECT 02-allowed_paths.json" in out
    assert "live-allowed_commands.json" not in out
    assert counters(out) == {"accepted_allowlist": "1", "rejected_allowlist": "1"}


def test_schedules_lint_and_gate(tmp_path, flow):
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    action = {
        "name": "nightly",
        "trigger": {"type": "schedule", "at": "8:00", "days": "weekdays"},
        "prompt": f"Run {flow} with days=1",
        "enabled": True,
    }
    (cfg / "event-schedules.json").write_text(json.dumps({"version": "2.0", "actions": [action]}))
    pdir = tmp_path / "p"

    lint = run("awl-check-schedules.sh", "lint", str(cfg), str(pdir), SOURCE_DIR)
    assert "LINT nightly: prompt names demo.awl but does not end in .awl" in lint
    assert "event_sources" not in (pdir / "live-event-schedules.json").read_text()

    fixed = dict(action, prompt=str(flow), prompt_arguments={"days": "1"})
    (pdir / "01-event-schedules.json").write_text(json.dumps({"version": "2.0", "actions": [fixed]}))
    added = dict(fixed, name="extra")
    (pdir / "02-event-schedules.json").write_text(json.dumps({"version": "2.0", "actions": [fixed, added]}))
    gate = run("awl-check-schedules.sh", "gate", str(cfg), str(pdir), SOURCE_DIR)
    assert "OK 01-event-schedules.json" in gate
    assert "REJECT 02-event-schedules.json" in gate
    assert counters(gate) == {"accepted_config": "1", "rejected_config": "1"}


def change(n: int, path: Path, diff: str) -> str:
    return f"## {n}. demo.awl :: gather\n**File**: {path}\n```diff\n{diff}\n```\n"


def test_excerpt_gate(tmp_path, signals, flow, config):
    other = tmp_path / "other.awl"
    other.write_text(FLOW)
    proposals = tmp_path / "proposals.md"
    proposals.write_text(
        change(1, flow, " Goal: Gather the items.\n-Success: The items are listed.\n+Success: Every item is listed.")
        + change(2, flow, "-Goal: Something else.\n+Goal: New.")
        + change(3, flow, " @end\n+# note")
        + change(4, other, " Goal: Gather the items.\n+Success: x")
        + change(5, flow, "  Goal: Summarize the items.\n+Success: The summary is written.")
        + change(6, flow, " Goal: Gather the items.\n-Success: The items are listed.\n+@task broken")
    )

    out = run("awl-check-excerpts.sh", "check", str(proposals), str(signals), str(config), SOURCE_DIR)

    assert "OK #1" in out
    assert "REJECT #2" in out and "not in the current file" in out
    assert "REJECT #3" in out and "appears 3 times" in out
    assert "REJECT #4" in out and "neither a workflow seen in the logs" in out
    assert "OK #5" in out and "indentation differs" in out
    assert "REJECT #6" in out and "no longer parses" in out
    assert counters(out) == {"accepted_changes": "2", "rejected_changes": "4"}


def test_excerpt_gate_accepts_installed_skill_files(tmp_path, signals, config):
    skill_md = tmp_path / "skills" / "demo-skill" / "SKILL.md"
    proposals = tmp_path / "proposals.md"
    proposals.write_text(
        change(
            1,
            skill_md,
            " name: demo-skill\n-allowed-tools: internal__execute_skill_script, internal__execute_command"
            "\n+allowed-tools: internal__execute_skill_script internal__execute_command",
        )
    )
    out = run("awl-check-excerpts.sh", "check", str(proposals), str(signals), str(config), SOURCE_DIR)
    assert counters(out) == {"accepted_changes": "1", "rejected_changes": "0"}


def test_report_assembly_appends_proposals(tmp_path):
    report = tmp_path / "report.md"
    report.write_text("# Report\n")
    proposals = tmp_path / "proposals.md"
    proposals.write_text("## 1. a proposal\n")
    run("awl-check-excerpts.sh", "append", str(proposals), str(report))
    assert report.read_text() == "# Report\n\n## 1. a proposal\n"
    assert "none" in run("awl-check-excerpts.sh", "append", str(tmp_path / "missing.md"), str(report))
