"""Tests for audit logging of tool calls"""

import json
import os
import time

import pytest

from ai_assist.audit import AuditLogger, audited_call, record_authorization_event


@pytest.fixture
def audit_dir(tmp_path):
    """Create a temporary audit directory"""
    audit_path = tmp_path / "audit"
    audit_path.mkdir()
    return audit_path


@pytest.fixture
def logger(audit_dir):
    """Create an AuditLogger with temporary directory"""
    return AuditLogger(audit_dir=audit_dir)


def test_audit_log_written_on_tool_call(logger, audit_dir):
    """JSONL file created with expected fields on tool call"""
    logger.log_tool_call(
        tool_name="internal__read_file",
        arguments={"path": "/tmp/test.txt"},
        result="File contents (10 chars):\n\nhello world",
        success=True,
    )

    log_file = audit_dir / "tool_calls.jsonl"
    assert log_file.exists()

    with open(log_file) as f:
        lines = f.readlines()

    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["tool_name"] == "internal__read_file"
    assert entry["arguments"] == {"path": "/tmp/test.txt"}
    assert entry["success"] is True
    assert "timestamp" in entry
    assert "result_summary" in entry
    assert entry["pid"] == os.getpid()


def test_audit_log_sanitizes_secrets(logger, audit_dir):
    """API keys and tokens are not written to the log"""
    logger.log_tool_call(
        tool_name="test_tool",
        arguments={"api_key": "sk-ant-12345", "token": "ghp_secret123", "safe_param": "visible"},
        result="Result with sk-ant-99999 and ghp_token in it",
        success=True,
    )

    log_file = audit_dir / "tool_calls.jsonl"
    content = log_file.read_text()

    assert "sk-ant-12345" not in content
    assert "ghp_secret123" not in content
    assert "sk-ant-99999" not in content
    assert "ghp_token" not in content
    assert "visible" in content


def test_audit_log_truncates_large_results(logger, audit_dir):
    """result_summary is capped at a reasonable size"""
    large_result = "x" * 50000

    logger.log_tool_call(
        tool_name="test_tool",
        arguments={},
        result=large_result,
        success=True,
    )

    log_file = audit_dir / "tool_calls.jsonl"
    entry = json.loads(log_file.read_text().strip())

    # result_summary should be truncated
    assert len(entry["result_summary"]) <= 4100  # 4000 chars + truncation message


def test_cleanup_deletes_old_rotated_files_and_never_touches_the_live_log(audit_dir):
    live = audit_dir / "tool_calls.jsonl"
    live.write_text('{"timestamp": "2020-01-01T00:00:00"}\n')
    old = audit_dir / "tool_calls-20200101T000000000000.jsonl"
    recent = audit_dir / "tool_calls-20991231T000000000000.jsonl"
    old.write_text("{}\n")
    recent.write_text("{}\n")
    ancient = time.time() - 30 * 86400
    os.utime(old, (ancient, ancient))
    os.utime(live, (ancient, ancient))

    removed = AuditLogger(audit_dir=audit_dir).cleanup(max_age_days=7)

    assert removed == 1
    assert not old.exists()
    assert recent.exists()
    assert live.read_text() == '{"timestamp": "2020-01-01T00:00:00"}\n'


def test_rotation_renames_the_live_log_and_keeps_it_owner_only(audit_dir):
    logger = AuditLogger(audit_dir=audit_dir, max_bytes=200)
    for i in range(5):
        logger.log_tool_call("x__y", {"i": i}, "r" * 100, success=True)

    rotated = logger.rotated_files()
    assert rotated
    live = audit_dir / "tool_calls.jsonl"
    for path in [*rotated, live]:
        assert os.stat(path).st_mode & 0o777 == 0o600
    # Nothing lost or rewritten: every record is in exactly one file, in order
    lines = [json.loads(line) for path in [*rotated, live] for line in path.read_text().splitlines()]
    assert [entry["arguments"]["i"] for entry in lines] == list(range(5))


def test_new_log_file_is_created_owner_only(audit_dir):
    AuditLogger(audit_dir=audit_dir).log_tool_call("x__y", {}, "ok", success=True)
    assert os.stat(audit_dir / "tool_calls.jsonl").st_mode & 0o777 == 0o600


def test_audit_log_multiple_entries(logger, audit_dir):
    """Multiple tool calls append to the same file"""
    for i in range(3):
        logger.log_tool_call(
            tool_name=f"tool_{i}",
            arguments={"index": i},
            result=f"result_{i}",
            success=True,
        )

    log_file = audit_dir / "tool_calls.jsonl"
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 3

    for i, line in enumerate(lines):
        entry = json.loads(line)
        assert entry["tool_name"] == f"tool_{i}"


def test_audit_log_records_failures(logger, audit_dir):
    """Failed tool calls are logged with success=False"""
    logger.log_tool_call(
        tool_name="failing_tool",
        arguments={"path": "/nonexistent"},
        result="Error: File not found",
        success=False,
    )

    log_file = audit_dir / "tool_calls.jsonl"
    entry = json.loads(log_file.read_text().strip())

    assert entry["success"] is False
    assert "Error" in entry["result_summary"]


def test_log_tool_call_records_confirmations(tmp_path):
    from ai_assist.audit import AuditLogger

    logger = AuditLogger(audit_dir=tmp_path)
    logger.log_tool_call(
        "internal__execute_command",
        {"command": "python3 -c 'print(1)'"},
        "ran",
        success=True,
        confirmations=[{"kind": "command", "question": "python3 -c execution: ...", "decision": "approved"}],
    )
    entry = json.loads((tmp_path / "tool_calls.jsonl").read_text().strip())
    assert entry["confirmations"][0]["decision"] == "approved"
    assert "python3 -c" in entry["confirmations"][0]["question"]


def test_log_tool_call_omits_confirmations_when_none(tmp_path):
    from ai_assist.audit import AuditLogger

    logger = AuditLogger(audit_dir=tmp_path)
    logger.log_tool_call("x__y", {}, "ok", success=True)
    entry = json.loads((tmp_path / "tool_calls.jsonl").read_text().strip())
    assert "confirmations" not in entry


def _entries(audit_dir):
    return [json.loads(line) for line in (audit_dir / "tool_calls.jsonl").read_text().splitlines()]


def test_entries_outside_a_workflow_have_no_task(logger, audit_dir):
    logger.log_tool_call("internal__read_file", {"path": "/tmp/f"}, "ok", success=True)
    [entry] = _entries(audit_dir)
    assert "script" not in entry
    assert "task" not in entry


def test_task_context_attributes_entries_to_the_workflow_task(logger, audit_dir):
    with logger.task_context("/flows/outer.awl", "gather"):
        logger.log_tool_call("internal__read_file", {"path": "/tmp/f"}, "ok", success=True)
        with logger.task_context("/flows/inner.awl", "fetch"):
            logger.log_tool_call("internal__read_file", {"path": "/tmp/g"}, "ok", success=True)
        logger.log_tool_call("internal__read_file", {"path": "/tmp/h"}, "ok", success=True)
    logger.log_tool_call("internal__read_file", {"path": "/tmp/i"}, "ok", success=True)

    tasks = [(e.get("script"), e.get("task")) for e in _entries(audit_dir)]
    assert tasks == [
        ("/flows/outer.awl", "gather"),
        ("/flows/inner.awl", "fetch"),
        ("/flows/outer.awl", "gather"),
        (None, None),
    ]


def test_task_context_is_restored_on_error(logger, audit_dir):
    with pytest.raises(RuntimeError), logger.task_context("/flows/a.awl", "t"):
        raise RuntimeError("boom")
    logger.log_tool_call("internal__read_file", {"path": "/tmp/f"}, "ok", success=True)
    assert "task" not in _entries(audit_dir)[0]


def test_confirmation_questions_are_sanitized(logger, audit_dir):
    logger.log_tool_call(
        "internal__execute_command",
        {"command": "x"},
        "ran",
        success=True,
        confirmations=[
            {"kind": "command", "question": "curl -H 'Authorization: ghp_abc123secret'", "decision": "approved"}
        ],
    )
    [entry] = _entries(audit_dir)
    assert "ghp_abc123secret" not in entry["confirmations"][0]["question"]
    assert entry["confirmations"][0]["decision"] == "approved"


def test_unprompted_call_is_autonomous_with_its_rule(logger, audit_dir):
    logger.log_tool_call("srv__tool", {}, "ok", success=True, rule="MCP tool: no approval gate")
    entry = _entries(audit_dir)[0]
    assert entry["authorization"] == {"decision": "autonomous", "rule": "MCP tool: no approval gate"}


def test_approved_call_records_approver_and_question(logger, audit_dir):
    with audited_call():
        record_authorization_event("command", "python3 -c execution", "approved")
        logger.log_tool_call("internal__execute_command", {}, "ok", success=True)
    auth = _entries(audit_dir)[0]["authorization"]
    assert auth["decision"] == "approved"
    assert auth["approver"] == "user"
    assert auth["questions"] == ["python3 -c execution"]
    assert auth["rule"] == "command-confirmation"


@pytest.mark.parametrize(
    "decision,rule",
    [
        ("denied", "path-confirmation: denied by user"),
        ("auto-denied (non-interactive)", "path-confirmation: no approver (non-interactive)"),
    ],
)
def test_refused_prompt_is_recorded(logger, audit_dir, decision, rule):
    with audited_call():
        record_authorization_event("path", "Access path: /x", decision)
        logger.log_tool_call("internal__write_file", {}, "Error: rejected", success=False)
    auth = _entries(audit_dir)[0]["authorization"]
    assert auth == {"decision": "refused", "rule": rule, "reason": "Access path: /x"}


def test_policy_refusal_is_recorded(logger, audit_dir):
    with audited_call():
        record_authorization_event("command-policy", "Error: rm is not allowed", "refused")
        logger.log_tool_call("internal__execute_command", {}, "Error: rm is not allowed", success=False)
    auth = _entries(audit_dir)[0]["authorization"]
    assert auth["decision"] == "refused"
    assert auth["rule"] == "command-policy"


def test_every_record_has_a_known_decision(logger, audit_dir):
    for decision in ("approved", "denied", "auto-denied (non-interactive)", "refused"):
        with audited_call():
            record_authorization_event("command", "q", decision)
            logger.log_tool_call("x__y", {}, "r", success=True)
    logger.log_tool_call("x__y", {}, "r", success=True)
    assert {e["authorization"]["decision"] for e in _entries(audit_dir)} == {"approved", "refused", "autonomous"}


def test_one_record_per_call_with_a_unique_correlation_id(logger, audit_dir):
    with audited_call() as call:
        logger.log_tool_call("x__y", {}, "ok", success=True)
        logger.log_tool_call("x__y", {}, "Error: again", success=False)  # dropped
    with audited_call():
        logger.log_tool_call("x__y", {}, "ok", success=True)
    entries = _entries(audit_dir)
    assert len(entries) == 2
    assert entries[0]["correlation_id"] == call.correlation_id
    assert entries[0]["success"] is True
    assert entries[0]["correlation_id"] != entries[1]["correlation_id"]


def test_authorization_question_is_sanitized(logger, audit_dir):
    with audited_call():
        record_authorization_event("command", "curl -H 'Authorization: ghp_abc123secret'", "approved")
        logger.log_tool_call("internal__execute_command", {}, "ok", success=True)
    entry = _entries(audit_dir)[0]
    assert "ghp_abc123secret" not in json.dumps(entry)


def test_sink_receives_every_record(audit_dir, tmp_path):
    sink = tmp_path / "elsewhere" / "audit.jsonl"
    sink.parent.mkdir()
    logger = AuditLogger(audit_dir=audit_dir, sink_path=sink)
    logger.log_tool_call("x__y", {}, "ok", success=True)
    assert json.loads(sink.read_text()) == _entries(audit_dir)[0]
    assert "sink_unavailable" not in _entries(audit_dir)[0]


def test_sink_from_environment(audit_dir, tmp_path, monkeypatch):
    sink = tmp_path / "sink.jsonl"
    monkeypatch.setenv("AI_ASSIST_AUDIT_SINK", str(sink))
    AuditLogger(audit_dir=audit_dir).log_tool_call("x__y", {}, "ok", success=True)
    assert sink.exists()


def test_unreachable_sink_is_flagged_not_fatal(audit_dir, tmp_path):
    logger = AuditLogger(audit_dir=audit_dir, sink_path=tmp_path / "missing-dir" / "audit.jsonl")
    for _ in range(3):
        logger.log_tool_call("x__y", {}, "ok", success=True)
    entries = _entries(audit_dir)
    assert len(entries) == 3
    assert all(e["sink_unavailable"] is True for e in entries)
