"""Tests for command validation in main.py"""

import json
import subprocess
import sys


def test_unknown_command_exits_with_error():
    """Test that unknown commands exit with error message"""
    result = subprocess.run(
        [sys.executable, "-m", "ai_assist.main", "/unknown-command"], capture_output=True, text=True, check=False
    )

    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert "Unknown command '/unknown-command'" in output
    assert "ai-assist /help" in output


def test_command_without_slash_gives_helpful_error():
    """Test that commands without / get a helpful error"""
    result = subprocess.run(
        [sys.executable, "-m", "ai_assist.main", "help"], capture_output=True, text=True, check=False
    )

    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert "Commands must start with /" in output
    assert "Did you mean: /help?" in output


def test_help_command_works():
    """Test that /help command works"""
    result = subprocess.run(
        [sys.executable, "-m", "ai_assist.main", "/help"], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0
    assert "Available commands:" in result.stdout
    assert "/monitor" in result.stdout
    assert "/query" in result.stdout


def test_unknown_command_does_not_initialize_agent():
    """Test that unknown commands don't initialize agent or connect to MCP servers"""
    result = subprocess.run(
        [sys.executable, "-m", "ai_assist.main", "/invalid"], capture_output=True, text=True, check=False
    )

    output = result.stdout + result.stderr

    # Should not see any agent initialization messages
    assert "Using Vertex AI" not in output
    assert "Connected to" not in output

    # Should see the error message immediately
    assert "Unknown command '/invalid'" in output
    assert result.returncode == 1


def _modules_after_running(argv):
    """Run ai_assist.main with the given argv and return the loaded module names."""
    code = (
        "import sys, json, runpy\n"
        f"sys.argv = {argv!r}\n"
        "try:\n"
        "    runpy.run_module('ai_assist.main', run_name='__main__')\n"
        "except SystemExit:\n"
        "    pass\n"
        "print(json.dumps([m for m in sys.modules if m == 'anthropic' or m.startswith('mcp')]))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_help_does_not_import_anthropic_stack():
    """The /help path must not import the heavy anthropic/mcp stack (keeps CLI cold-start fast)."""
    loaded = _modules_after_running(["ai-assist", "/help"])
    assert loaded == [], f"heavy modules imported on /help path: {loaded}"


def test_unknown_command_does_not_import_anthropic_stack():
    """Unknown commands must fail fast without importing the heavy anthropic/mcp stack."""
    loaded = _modules_after_running(["ai-assist", "/nope"])
    assert loaded == [], f"heavy modules imported on unknown-command path: {loaded}"
