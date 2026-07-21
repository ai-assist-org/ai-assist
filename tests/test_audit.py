"""Tests for audit logging of tool calls"""

import json
import os
import time

import pytest

from ai_assist.audit import AuditLogger


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


def test_audit_log_rotation(audit_dir):
    """Old entries are cleaned up by cleanup()"""
    log_file = audit_dir / "tool_calls.jsonl"

    # Write entries with old timestamps
    old_entries = []
    for i in range(5):
        entry = {
            "tool_name": f"old_tool_{i}",
            "arguments": {},
            "result_summary": "old result",
            "success": True,
            "timestamp": "2020-01-01T00:00:00",
        }
        old_entries.append(json.dumps(entry))

    # Write a recent entry
    recent_entry = {
        "tool_name": "recent_tool",
        "arguments": {},
        "result_summary": "recent result",
        "success": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    old_entries.append(json.dumps(recent_entry))

    log_file.write_text("\n".join(old_entries) + "\n")

    logger = AuditLogger(audit_dir=audit_dir)
    removed = logger.cleanup(max_age_days=7)

    assert removed == 5  # 5 old entries removed

    # Recent entry should remain
    remaining = log_file.read_text().strip().split("\n")
    assert len(remaining) == 1
    assert "recent_tool" in remaining[0]


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
