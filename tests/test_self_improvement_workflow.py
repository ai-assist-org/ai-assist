"""Tests for the self-improvement workflow and its deterministic helpers."""

import json
import os
import re
import shutil
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


def run(script: str, *args: str, home: Path | None = None, path_prefix: Path | None = None) -> str:
    env = dict(os.environ, AI_ASSIST_PYTHON=sys.executable)
    if home is not None:
        env["HOME"] = str(home)
    if path_prefix is not None:
        env["PATH"] = f"{path_prefix}{os.pathsep}{env['PATH']}"
    result = subprocess.run(["bash", str(LIB / script), *args], capture_output=True, text=True, env=env, check=False)
    assert result.returncode == 0, f"{script} exited {result.returncode}: {result.stderr}"
    return result.stdout


def counters(output: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line and " " not in line)


def log_line(time: str, pid: int, logger: str, level: str, msg: str) -> str:
    return f"2026-11-05 {time},000 - {pid} - {logger} - {level} - {msg}"


def shell_call(time: str, command: str) -> str:
    call = json.dumps({"command": command})
    return log_line(time, 100, "ai_assist.audit", "INFO", f"Tool call: internal__execute_command({call}) -> ok")


def audit_entry(tool: str, kind: str, decision: str, question: str, task: str | None = "gather", **extra) -> dict:
    from datetime import datetime, timedelta

    entry = {
        "tool_name": tool,
        "arguments": {},
        "result_summary": "",
        "success": True,
        "timestamp": (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S"),
        "pid": 100,
        "confirmations": [{"kind": kind, "question": question, "decision": decision}],
        **extra,
    }
    if task:
        entry.update(script="/flows/demo.awl", task=task)
    return entry


def command_entry(command: str, result: str) -> dict:
    entry = audit_entry("internal__execute_command", "command", "approved", "x")
    entry.pop("confirmations")
    entry.update(arguments={"command": command}, result_summary=result)
    return entry


@pytest.fixture
def flow(tmp_path):
    path = tmp_path / "flows" / "demo.awl"
    path.parent.mkdir()
    path.write_text(FLOW)
    return path


@pytest.fixture(params=["awk", "posix-awk"])
def awk_dir(request, tmp_path):
    """None for the system awk; or a directory whose `awk` accepts only POSIX awk.

    macOS (BSD awk) and Debian/Ubuntu (mawk) have no gawk extensions such as
    mktime(); gawk --posix rejects them the same way.
    """
    if request.param == "awk":
        return None
    gawk = shutil.which("gawk")
    if gawk is None:
        pytest.skip("gawk is needed to emulate a POSIX-only awk")
    bin_dir = tmp_path / "posix-bin"
    bin_dir.mkdir()
    (bin_dir / "awk").write_text(f'#!/bin/sh\nexec {gawk} --posix "$@"\n')
    (bin_dir / "awk").chmod(0o755)
    return bin_dir


@pytest.fixture
def signals(tmp_path, flow, awk_dir):
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
        shell_call("06:00:03", "curl -s https://example.com/api | head -20"),
        shell_call("06:00:03", "curl -s https://example.com/other | head -5"),
        shell_call("06:00:03", "make test 2>&1 | tail -30"),
        shell_call("06:00:03", "git log --oneline | cut -c1-80"),
        # backslashes and quotes must stay valid JSON with any awk
        shell_call("06:00:03", 'grep -n "a\\|b" "f\\"x"'),
        log_line("06:00:04", 100, "ai_assist.awl_runtime", "ERROR", "AWL task 'gather' failed"),
        log_line("06:00:05", 100, "ai_assist.awl_executor", "INFO", f"AWL script lock released: {released}"),
        log_line("07:00:00", 100, "ai_assist.action_scheduler", "INFO", "Running nightly..."),
        log_line("07:00:00", 101, "ai_assist.action_scheduler", "INFO", "Running nightly..."),
        log_line("07:00:09", 100, "ai_assist.action_scheduler", "ERROR", "nightly: failed - boom"),
    ]
    (logs / "ai-assist.log").write_text("\n".join(lines) + "\n")
    old = audit_entry("internal__execute_command", "command", "approved", "cat /old")
    old["timestamp"] = "2020-01-01T00:00:00"
    audit = [
        audit_entry(
            "internal__execute_command",
            "command",
            "approved",
            "awk runs arbitrary code/package: cd /srv && grep -c x f | awk '{print $1}'",
        ),
        audit_entry(
            "internal__execute_command", "path", "denied", "Access path: /help\nReason: cd /srv && uv run app /help"
        ),
        audit_entry("internal__execute_command", "command", "approved", "cat /etc/hosts"),
        audit_entry("internal__execute_command", "command", "auto-denied (non-interactive)", "cat /etc/passwd"),
        audit_entry("internal__create_action", "command", "approved", "Schedule action 'nightly' running", task=None),
        audit_entry(
            "internal__execute_command",
            "command",
            "approved",
            "python3 module:pytest execution: cd /srv && python3 -m pytest",
        ),
        audit_entry(
            "internal__execute_command",
            "path",
            "denied",
            "Access path: /home\nReason: cd ~ && (tool status 2>&1 | head) ; which tool",
        ),
        old,
        {"tool_name": "internal__read_file", "arguments": {}, "timestamp": "2026-11-05T06:00:00", "pid": 100},
        audit_entry("internal__execute_command", "command", "approved", "git worktree list"),
        audit_entry("internal__execute_command", "command", "approved", "cd /srv && git worktree list --porcelain"),
        audit_entry("internal__execute_command", "command", "approved", "whoami"),
        audit_entry("internal__execute_command", "command", "approved", "LC_ALL=C git branch -a | head -5"),
        audit_entry("internal__execute_command", "command", "approved", "2>&1 git log --oneline"),
        command_entry("python3 -c \"import json,sys; print(json.load(open('a.json'))['x'])\"", "Exit code: 0"),
        command_entry("python3 - <<'EOF'\nimport yaml\nprint(yaml.safe_load(open('c.yaml')))\nEOF", "Exit code: 0"),
        command_entry("python3 -c 'import json; print(json.dumps({}))'", "Exit code: 0"),
        command_entry("awk -F, '{s+=$3} END{print s}' data.csv", "Exit code: 0"),
        command_entry("cat x | awk '{print $2}'", "Exit code: 0"),
        command_entry("python3 tools/report.py --all", "Exit code: 0"),
        command_entry(
            "make build 2>&1 | tail -3",
            "Command: make build 2>&1 | tail -3\nExit code: 0\n\nSTDOUT:\nmake: *** [all] Error 2\nerror: build failed\n",
        ),
        command_entry(  # an intentional probe for errors, not a hidden failure
            "grep -n 'error:' build.log | head -3",
            "Command: grep -n 'error:' build.log | head -3\nExit code: 0\n\nSTDOUT:\n12: error: x\n",
        ),
        command_entry(  # the new execute_command warning: certain
            "curl -s https://example.com/x | jq .",
            "Command: curl -s https://example.com/x | jq .\nExit code: 0\n\nWarning: the exit code is that of the "
            "last command in the pipeline; an earlier command failed (exit statuses: 6 0).",
        ),
    ]
    audit_dir = tmp_path / "cfg" / "audit"
    audit_dir.mkdir(parents=True)
    (audit_dir / "tool_calls.jsonl").write_text("\n".join(map(json.dumps, audit)) + "\nnot json\n")
    out = tmp_path / "signals.jsonl"
    run(
        "awl-signals-aggregate.sh",
        "7",
        str(out),
        str(logs),
        str(tmp_path / "signals.tsv"),
        SOURCE_DIR,
        str(tmp_path / "cfg"),
        path_prefix=awk_dir,
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
        ("internal__read_file", "sed -n N,Mp"),
        ("internal__read_file", "sed |"),
        ("internal__search_in_file", "grep -r"),
        ("internal__search_in_file", "grep -l"),
        ("internal__search_in_file", "grep -n"),
    }
    sed = shell[("internal__read_file", "sed -n N,Mp")]
    assert (sed["count"], sed["top_task"], sed["top_task_count"]) == (2, "demo.awl::gather", 2)
    assert not [r for r in records(signals, "signal") if r["event"] == "shell_instead_of_tool"]


def test_security_prompts_are_rolled_up(signals):
    prompts = {(r["prompt"], r["form"], r["tool"]): r for r in records(signals, "prompt")}
    assert set(prompts) == {
        ("command", "awk (arbitrary code)", "-"),
        ("path", "path for uv", "-"),
        ("command", "cat", "internal__read_file"),
        ("command", "internal__create_action", "-"),
        ("command", "python3 module:pytest execution", "-"),
        ("path", "path for tool", "-"),
        ("command", "git worktree list", "-"),
        ("command", "whoami", "-"),
        ("command", "git branch", "-"),
        ("command", "git log", "-"),
    }
    cat = prompts[("command", "cat", "internal__read_file")]
    assert (cat["count"], cat["approved"], cat["denied"], cat["auto_denied"]) == (2, 1, 0, 1)  # not /old: too old
    assert (cat["top_task"], cat["top_task_count"]) == ("demo.awl::gather", 2)
    assert cat["sample"] == "cat /etc/hosts"
    path = prompts[("path", "path for uv", "-")]
    assert (path["denied"], path["sample"]) == (1, "Access path: /help Reason: cd /srv && uv run app /help")
    assert "grep -c x f | awk" in prompts[("command", "awk (arbitrary code)", "-")]["sample"]
    assert prompts[("command", "internal__create_action", "-")]["top_task"] == "unknown::-"
    assert not [r for r in records(signals, "signal") if r["event"] == "security_prompt"]


def test_pipe_reducers_are_rolled_up(signals):
    pipes = {r["reducer"]: r for r in records(signals, "pipe")}
    assert {"head -#", "tail -#", "cut -c#-#"} <= set(pipes)
    head = pipes["head -#"]
    assert (head["count"], head["top_producer"], head["top_task"]) == (2, "curl", "demo.awl::gather")
    assert pipes["tail -#"]["stderr_merged"] == 1
    assert re.fullmatch(r"[0-9a-f]{12}", head["signature"]) and head["event"] == "pipe_reducer"


def test_hidden_pipeline_failures_are_candidates_but_not_probes(signals):
    masked = [r for r in records(signals, "signal") if r["event"] == "pipe_masked"]
    samples = " ".join(r["sample"] for r in masked)
    assert "make build" in samples
    assert "curl -s https://example.com/x" in samples
    assert "grep -n" not in samples
    assert all(r["script"] == "demo.awl" for r in masked)


@pytest.mark.parametrize(
    ("question", "form"),
    [
        ("git -C /srv/repo worktree list", "git worktree list"),
        ("git -c core.pager=cat log -3", "git log"),
        ("timeout 30 whoami", "whoami"),
        ("env LC_ALL=C sort f", "sort"),
        ("for f in *.log; do wc -l $f; done", "(shell construct)"),
        ("mapfile -t x < f", "(shell construct)"),
    ],
)
def test_command_question_forms(tmp_path, question, form):
    row = "\t".join(
        [
            "2026-11-05",
            "06:00:00,000",
            "1",
            "demo.awl",
            "t",
            "internal__execute_command",
            "command",
            "approved",
            question,
        ]
    )
    out = subprocess.run(
        ["awk", "-f", str(LIB / "signals-common.awk"), "-f", str(LIB / "extract-prompts.awk")],
        input=row + "\n",
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out.split("\t")[6].split("\037")[2] == form


def test_command_questions_keep_the_subcommand(signals):
    prompts = {(r["prompt"], r["form"]): r for r in records(signals, "prompt")}
    worktree = prompts[("command", "git worktree list")]
    assert (worktree["count"], worktree["approved"], worktree["denied"]) == (2, 2, 0)


def test_inline_scripts_are_rolled_up_by_language_and_purpose(signals):
    scripts = {(r["language"], r["purpose"]): r for r in records(signals, "inline")}
    assert set(scripts) == {("python", "json"), ("python", "yaml"), ("awk", "aggregate+fields"), ("awk", "fields")}
    js = scripts[("python", "json")]
    assert (js["count"], js["top_task"], js["event"]) == (2, "demo.awl::gather", "inline_script")
    assert "json.load" in js["sample"]
    assert re.fullmatch(r"[0-9a-f]{12}", js["signature"])


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


# --- Review verdicts: what a reviewer concluded feeds the next runs ---


def run_failing(script: str, *args: str) -> subprocess.CompletedProcess:
    result = subprocess.run(["bash", str(LIB / script), *args], capture_output=True, text=True, check=False)
    assert result.returncode != 0, result.stdout
    return result


@pytest.fixture
def ledger(signals, tmp_path):
    """A ledger recorded from the signals fixture, a verdicts file, and a signature per event."""
    state = tmp_path / "state.jsonl"
    run("awl-signals-diff.sh", str(signals), str(state), "--record-state")
    sigs = {r["event"]: r["signature"] for r in records(signals, "signal")}
    return state, tmp_path / "verdicts.jsonl", sigs


def recurred(signals, tmp_path, events, when="2099-01-01T06:00:00,000"):
    """The signals fixture where the given events occurred again later."""
    lines = [json.loads(line) for line in signals.read_text().splitlines()]
    for rec in lines:
        if rec.get("event") in events:
            rec["last"] = when
    later = tmp_path / "later.jsonl"
    later.write_text("\n".join(json.dumps(rec) for rec in lines) + "\n")
    return later


def verdict(ledger, event, value, note="because"):
    state, verdicts, sigs = ledger
    return run("awl-verdict.sh", str(state), str(verdicts), sigs[event], value, note, "agent:test")


def diff(signals_file, ledger, *extra):
    state, verdicts, _ = ledger
    return run("awl-signals-diff.sh", str(signals_file), str(state), "--verdicts", str(verdicts), *extra)


def test_verdict_is_appended_with_reviewer_and_note(ledger):
    state, verdicts, sigs = ledger
    verdict(ledger, "unresolved_var", "false_positive", "variable comes from variables.yaml")
    [line] = [json.loads(x) for x in verdicts.read_text().splitlines()]
    assert line["signature"] == sigs["unresolved_var"]
    assert (line["verdict"], line["reviewer"]) == ("false_positive", "agent:test")
    assert line["note"] == "variable comes from variables.yaml"
    assert line["at"][:4].isdigit()


@pytest.mark.parametrize(
    ("signature", "value", "note", "message"),
    [
        ("000000000000", "real", "x", "not in the ledger"),
        (None, "maybe", "x", "verdict must be one of"),
        (None, "real", "", "note"),
    ],
)
def test_invalid_verdicts_are_refused(ledger, signature, value, note, message):
    state, verdicts, sigs = ledger
    result = run_failing(
        "awl-verdict.sh", str(state), str(verdicts), signature or sigs["task_fail"], value, note, "human:x"
    )
    assert message in result.stderr
    assert not verdicts.exists()


@pytest.mark.parametrize("value", ["false_positive", "wont_fix"])
def test_dismissed_signal_is_suppressed_when_it_recurs(signals, tmp_path, ledger, value):
    verdict(ledger, "unresolved_var", value)
    out = counters(diff(recurred(signals, tmp_path, {"unresolved_var"}), ledger))
    assert (out["new_signals"], out["suppressed"]) == ("0", "1")


def test_fixed_signal_that_recurs_is_a_regression_once(signals, tmp_path, ledger):
    verdict(ledger, "task_fail", "fixed")
    later = recurred(signals, tmp_path, {"task_fail"})
    review = tmp_path / "review.jsonl"

    out = counters(diff(later, ledger, "--review-out", str(review), "--record-state"))
    assert (out["new_signals"], out["regressions"]) == ("1", "1")
    [rec] = [r for r in map(json.loads, review.read_text().splitlines()) if r["kind"] == "signal"]
    assert (rec["event"], rec["regression"], rec["prior_verdict"]) == ("task_fail", True, "fixed")

    again = counters(diff(later, ledger))  # idempotent: reported once
    assert (again["new_signals"], again["regressions"]) == ("0", "0")


def test_fixed_signal_that_stays_quiet_is_not_reported(signals, ledger):
    verdict(ledger, "task_fail", "fixed")
    out = counters(diff(signals, ledger))
    assert (out["new_signals"], out["regressions"]) == ("0", "0")


def test_confirmed_real_signal_is_open_not_new(signals, tmp_path, ledger):
    verdict(ledger, "command_refused", "real", "tool.sh must be allowlisted")
    output = diff(recurred(signals, tmp_path, {"command_refused"}), ledger)
    out = counters(output)
    assert (out["new_signals"], out["open"]) == ("0", "1")
    assert "tool.sh must be allowlisted" in output.split("=== open")[1]


def test_review_file_has_only_new_signals_and_counter_examples(signals, tmp_path):
    state, verdicts = tmp_path / "s.jsonl", tmp_path / "v.jsonl"
    review = tmp_path / "review.jsonl"
    run("awl-signals-diff.sh", str(signals), str(state), "--record-state")
    sig = {r["event"]: r["signature"] for r in records(signals, "signal")}
    run("awl-verdict.sh", str(state), str(verdicts), sig["skill_refused"], "false_positive", "demo skill", "h:x")
    later = recurred(signals, tmp_path, {"unresolved_var", "skill_refused"})

    run("awl-signals-diff.sh", str(later), str(state), "--verdicts", str(verdicts), "--review-out", str(review))
    recs = [json.loads(x) for x in review.read_text().splitlines()]
    assert [r["event"] for r in recs if r["kind"] == "signal"] == ["unresolved_var"]
    [example] = [r for r in recs if r["kind"] == "counter_example"]
    assert (example["event"], example["verdict"], example["note"]) == ("skill_refused", "false_positive", "demo skill")


def test_precision_by_event(ledger, signals):
    verdict(ledger, "unresolved_var", "false_positive")
    verdict(ledger, "task_fail", "fixed")
    verdict(ledger, "command_refused", "real")
    output = diff(signals, ledger)
    lines = [x for x in output.splitlines() if x.startswith("PRECISION ")]
    assert "PRECISION unresolved_var reviewed=1 real=0 false_positive=1" in lines
    assert "PRECISION task_fail reviewed=1 real=1 false_positive=0" in lines


def test_latest_verdict_wins(signals, tmp_path, ledger):
    verdict(ledger, "unresolved_var", "false_positive")
    verdict(ledger, "unresolved_var", "real", "it was real after all")
    out = counters(diff(recurred(signals, tmp_path, {"unresolved_var"}), ledger))
    assert (out["suppressed"], out["open"]) == ("0", "1")


def test_diff_without_verdicts_keeps_working(signals, tmp_path):
    state = str(tmp_path / "state.jsonl")
    out = counters(run("awl-signals-diff.sh", str(signals), state, "--verdicts", str(tmp_path / "none.jsonl")))
    assert int(out["new_signals"]) > 0
    assert (out["regressions"], out["open"], out["suppressed"]) == ("0", "0", "0")


# --- Rolled-up records (shell forms, security questions) can be reviewed too ---


def rollup(signals, kind, **match):
    return next(r for r in records(signals, kind) if all(r[k] == v for k, v in match.items()))


def test_rollups_have_stable_signatures_and_a_last_occurrence(signals, tmp_path):
    shell = rollup(signals, "shell", tool="internal__search_in_file", form="grep -r")
    prompt = rollup(signals, "prompt", form="cat")
    for rec in (shell, prompt):
        assert re.fullmatch(r"[0-9a-f]{12}", rec["signature"])
        assert rec["last"].startswith(("2026-", "20"))
    sigs = [r["signature"] for r in records(signals, "shell") + records(signals, "prompt")]
    assert len(sigs) == len(set(sigs))
    assert (shell["event"], prompt["event"]) == ("shell_instead_of_tool", "security_prompt")


def test_record_state_adds_rollups_so_verdicts_accept_them(ledger, signals):
    state, verdicts, _ = ledger
    shell = rollup(signals, "shell", tool="internal__search_in_file", form="grep -r")
    run("awl-verdict.sh", str(state), str(verdicts), shell["signature"], "real", "needs recursive search", "h:x")
    assert json.loads(verdicts.read_text().splitlines()[-1])["signature"] == shell["signature"]


def review_records(signals_file, ledger, tmp_path):
    review = tmp_path / "review.jsonl"
    output = diff(signals_file, ledger, "--review-out", str(review))
    return counters(output), [json.loads(x) for x in review.read_text().splitlines()], output


def test_rollups_go_to_review_unless_dismissed_or_confirmed(ledger, signals, tmp_path):
    grep_r = rollup(signals, "shell", tool="internal__search_in_file", form="grep -r")
    grep_l = rollup(signals, "shell", tool="internal__search_in_file", form="grep -l")
    cat = rollup(signals, "prompt", form="cat")
    state, verdicts, _ = ledger
    run("awl-verdict.sh", str(state), str(verdicts), grep_r["signature"], "false_positive", "fine", "h:x")
    run("awl-verdict.sh", str(state), str(verdicts), cat["signature"], "real", "use read_file", "h:x")

    out, recs, output = review_records(signals, ledger, tmp_path)
    reviewed = {r["signature"] for r in recs if r["kind"] in ("shell", "prompt")}
    assert grep_l["signature"] in reviewed
    assert grep_r["signature"] not in reviewed and cat["signature"] not in reviewed
    assert int(out["suppressed"]) >= 1 and int(out["open"]) >= 1
    assert "use read_file" in output.split("=== open")[1]
    assert out["new_signals"] == "0"  # rollups never count as new signals


def test_fixed_rollup_seen_again_is_a_regression(ledger, signals, tmp_path):
    grep_l = rollup(signals, "shell", tool="internal__search_in_file", form="grep -l")
    state, verdicts, _ = ledger
    run("awl-verdict.sh", str(state), str(verdicts), grep_l["signature"], "fixed", "tool has output=files", "h:x")

    _, quiet, _ = review_records(signals, ledger, tmp_path)  # last occurrence before the fix
    assert grep_l["signature"] not in {r["signature"] for r in quiet}

    later = tmp_path / "later.jsonl"
    lines = [json.loads(x) for x in signals.read_text().splitlines()]
    for rec in lines:
        if rec.get("signature") == grep_l["signature"]:
            rec["last"] = "2099-01-01T00:00:00,000"
    later.write_text("\n".join(map(json.dumps, lines)) + "\n")
    out, recs, _ = review_records(later, ledger, tmp_path)
    [rec] = [r for r in recs if r["signature"] == grep_l["signature"]]
    assert (rec["regression"], rec["prior_verdict"]) == (True, "fixed")
    assert out["regressions"] == "1"


@pytest.fixture
def config(tmp_path):
    cfg = tmp_path / "cfg"
    cfg.mkdir(exist_ok=True)
    (cfg / "allowed_commands.json").write_text(json.dumps(["ls", "\nfor f", "bash", "-r .access_token", "wc $f"]))
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
    assert 'junk entry "-r .access_token"' in out and 'junk entry "wc $f"' in out
    assert 'junk entry "/does/not/exist"' in out
    assert '"bash" allows arbitrary code' in out
    assert f'CANDIDATE allowed_commands.json add "{tool}"' in out
    assert f'NOTE "{in_flow}"' in out and "name it in backticks" in out
    assert "LINT skill demo-skill: allowed-tools" in out
    assert counters(out)["config_issues"] == "6"
    assert (tmp_path / "p" / "live-installed-skills.json").exists()


def test_config_lint_notes_entries_never_used_in_the_logs(tmp_path, config):
    signals = tmp_path / "s.jsonl"
    signals.write_text("")
    lint = ["lint", str(config), str(signals), str(tmp_path / "p"), SOURCE_DIR]
    without_logs = run("awl-check-config.sh", *lint, home=tmp_path)
    logs = config / "logs"
    logs.mkdir()
    (logs / "ai-assist.log").write_text(shell_call("06:00:00", "ls -la /tmp") + "\n")

    out = run("awl-check-config.sh", *lint, home=tmp_path)

    [note] = [x for x in out.splitlines() if "never matched" in x]
    assert note.startswith("NOTE allowed_commands.json: 1 of 2 valid entries never matched")
    assert '"bash"' in note and '"ls"' not in note
    assert "rare use looks the same" in note
    assert counters(out)["config_issues"] == counters(without_logs)["config_issues"]  # a NOTE is no issue


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


def test_review_skill_loads_without_warnings():
    from ai_assist.security import validate_tool_description
    from ai_assist.skills_loader import SkillsLoader

    skill = SkillsLoader().load_skill_from_local(WORKFLOW_DIR.parent.parent / "skills" / "self-improvement-review")
    assert skill.metadata.name == "self-improvement-review"
    assert not validate_tool_description("skill", skill.metadata.description)
    assert not validate_tool_description("skill/body", skill.body)
    assert "awl-verdict.sh" in skill.body


def test_pipe_records_and_hidden_failures_reach_the_review_file(signals, tmp_path):
    review = tmp_path / "review.jsonl"
    run("awl-signals-diff.sh", str(signals), str(tmp_path / "fresh.jsonl"), "--review-out", str(review))
    recs = [json.loads(x) for x in review.read_text().splitlines()]
    assert "head -#" in {r.get("reducer") for r in recs if r["kind"] == "pipe"}
    assert any(r["kind"] == "signal" and r["event"] == "pipe_masked" for r in recs)
