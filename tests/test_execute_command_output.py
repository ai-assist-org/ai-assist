"""execute_command: output limits that replace `| head` / `| tail` / `| cut`, and loud pipeline failures."""

import re

import pytest

from ai_assist.config import AiAssistConfig
from ai_assist.filesystem_tools import FilesystemTools


@pytest.fixture
def tools(tmp_path, monkeypatch, isolate_config_dir):
    monkeypatch.chdir(tmp_path)
    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_commands=["seq", "cat", "head", "printf", "tail"],
        allowed_paths=[str(tmp_path)],
    )
    return FilesystemTools(config, load_user_config=False)


async def run(tools, command, **args):
    return await tools.execute_tool("execute_command", {"command": command, **args})


def stdout_lines(result):
    body = result.split("STDOUT:\n", 1)[1]
    body = body.split("\nSTDERR:\n", 1)[0]
    return [x for x in body.split("\n") if x and not x.startswith("[")]


# --- output limits ---


async def test_no_limit_keeps_the_output_unchanged(tools):
    result = await run(tools, "seq 1 300")
    assert stdout_lines(result) == [str(i) for i in range(1, 301)]
    assert "showing" not in result


async def test_max_output_lines_keeps_the_first_lines_and_saves_the_rest(tools):  # | head -20
    result = await run(tools, "seq 1 300", max_output_lines=20)
    assert stdout_lines(result) == [str(i) for i in range(1, 21)]
    note = re.search(
        r"\[showing lines 1-20 of 300; full output saved to (\S+): read it with internal__read_file", result
    )
    assert note, result
    saved = await tools.execute_tool("read_file", {"path": note.group(1), "line_start": 299, "line_end": 300})
    assert "299" in saved and "300" in saved


async def test_from_end_keeps_the_last_lines(tools):  # | tail -5
    result = await run(tools, "seq 1 300", max_output_lines=5, from_end=True)
    assert stdout_lines(result) == ["296", "297", "298", "299", "300"]
    assert "showing lines 296-300 of 300" in result


async def test_max_line_chars_cuts_long_lines(tools):  # | cut -c1-10
    result = await run(tools, "printf 'abcdefghijklmnopqrstuvwxyz\\nshort\\n'", max_line_chars=10)
    assert stdout_lines(result) == ["abcdefghij…", "short"]
    assert "1 line(s) cut at 10 characters" in result


async def test_limits_keep_the_real_exit_code(tools):
    """Unlike `cmd | head`, a failing command still reports its failure."""
    result = await run(tools, "cat missing.txt", max_output_lines=5)
    assert "Exit code: 1" in result


# --- loud pipeline failures ---


async def test_pipeline_hiding_a_failure_says_so(tools):
    result = await run(tools, "cat missing.txt | head -1")
    assert "Exit code: 0" in result
    assert "Warning: the exit code is that of the last command in the pipeline" in result
    assert "exit statuses: 1 0" in result


async def test_reader_closing_the_pipe_is_not_a_failure(tools):
    """`seq ... | head -1`: seq gets SIGPIPE (141) once head has enough; that is normal."""
    result = await run(tools, "seq 1 1000000 | head -1")
    assert stdout_lines(result) == ["1"]
    assert "Warning" not in result


async def test_successful_pipeline_has_no_warning_and_no_marker(tools):
    result = await run(tools, "seq 1 3 | head -2 # comment")
    assert stdout_lines(result) == ["1", "2"]
    assert "Warning" not in result
    assert "PIPESTATUS" not in result and "__ai_assist" not in result
    assert result.startswith("Command: seq 1 3 | head -2 # comment\n")


async def test_command_without_pipe_is_unchanged(tools):
    result = await run(tools, "cat missing.txt")
    assert "Exit code: 1" in result
    assert "Warning" not in result
