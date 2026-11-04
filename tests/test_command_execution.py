"""Agent commands must not read the terminal, block the event loop, or outlive cancellation."""

import asyncio
import os
import subprocess
import time

import pytest

from ai_assist.config import AiAssistConfig
from ai_assist.filesystem_tools import FilesystemTools


def _tools(tmp_path) -> FilesystemTools:
    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_commands=["cat", "sleep", "sh", "echo"],
        allowed_paths=[str(tmp_path)],
    )
    return FilesystemTools(config, load_user_config=False)


def _marker(base: int) -> str:
    """A sleep command line unique to this test run, so pgrep finds only our process."""
    return f"sleep {base}.{os.getpid()}"


def _running(marker: str) -> bool:
    """Whether a process whose command line contains ``marker`` is still alive."""
    return subprocess.run(["pgrep", "-f", marker], capture_output=True, check=False).returncode == 0


@pytest.mark.asyncio
async def test_command_reading_stdin_gets_end_of_input(tmp_path):
    """Like `grep pattern` with no file: it must not wait on the terminal."""
    tools = _tools(tmp_path)
    read_end, write_end = os.pipe()  # an "input" that never ends, standing in for the terminal
    saved_stdin = os.dup(0)
    os.dup2(read_end, 0)
    try:
        started = time.monotonic()
        result = await tools.execute_tool("execute_command", {"command": "cat", "timeout": 5})
        elapsed = time.monotonic() - started
    finally:
        os.dup2(saved_stdin, 0)
        for fd in (saved_stdin, read_end, write_end):
            os.close(fd)

    assert "Exit code: 0" in result
    assert elapsed < 3


@pytest.mark.asyncio
async def test_running_command_does_not_block_the_event_loop(tmp_path):
    tools = _tools(tmp_path)
    ticks = 0
    ticks_when_done = None

    async def ticker():
        nonlocal ticks
        for _ in range(15):
            await asyncio.sleep(0.1)
            ticks += 1

    async def command():
        nonlocal ticks_when_done
        await tools.execute_tool("execute_command", {"command": "sleep 1", "timeout": 5})
        ticks_when_done = ticks

    await asyncio.gather(command(), ticker())

    assert ticks_when_done >= 5  # the loop kept running while the command did


@pytest.mark.asyncio
async def test_cancelling_a_command_kills_it_and_its_children(tmp_path):
    """What Escape does: the query is cancelled, and the command must not keep running."""
    tools = _tools(tmp_path)
    marker = _marker(37)
    task = asyncio.create_task(
        tools.execute_tool("execute_command", {"command": f"sh -c '{marker}; true'", "timeout": 60})
    )
    await asyncio.sleep(0.5)
    assert _running(marker)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.2)

    assert not _running(marker)


@pytest.mark.asyncio
async def test_timed_out_command_is_killed_with_its_children(tmp_path):
    tools = _tools(tmp_path)
    marker = _marker(38)

    result = await tools.execute_tool("execute_command", {"command": f"sh -c '{marker}; true'", "timeout": 1})
    await asyncio.sleep(0.2)

    assert "timed out" in result
    assert not _running(marker)


@pytest.mark.asyncio
async def test_command_output_and_exit_code_unchanged(tmp_path):
    tools = _tools(tmp_path)

    result = await tools.execute_tool("execute_command", {"command": "echo hello; echo oops >&2; exit 3"})

    assert "Exit code: 3" in result
    assert "STDOUT:\nhello" in result
    assert "STDERR:\noops" in result
    assert "Command failed with exit code 3" in result
