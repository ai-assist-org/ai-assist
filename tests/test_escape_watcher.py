"""Tests for EscapeWatcher - detects bare Escape key during streaming"""

import os
import threading
import time
from unittest.mock import patch

import pytest

from ai_assist.escape_watcher import EscapeWatcher


@pytest.fixture
def cancel_event():
    return threading.Event()


@pytest.fixture
def fake_stdin():
    """Create a pipe-based fake stdin fd"""
    read_fd, write_fd = os.pipe()
    yield read_fd, write_fd
    os.close(read_fd)
    os.close(write_fd)


def test_standalone_escape_sets_event(cancel_event, fake_stdin):
    """Writing bare \\x1b (no follow-up) should set cancel_event"""
    read_fd, write_fd = fake_stdin

    with patch("sys.stdin") as mock_stdin:
        mock_stdin.fileno.return_value = read_fd
        mock_stdin.isatty.return_value = True

        with patch("ai_assist.escape_watcher.termios") as mock_termios, patch("ai_assist.escape_watcher.tty"):
            mock_termios.tcgetattr.return_value = [0] * 7

            watcher = EscapeWatcher(cancel_event)
            watcher.start()

            # Write bare escape
            os.write(write_fd, b"\x1b")
            # Wait for detection (50ms escape timeout). Poll instead of a fixed
            # sleep so the test doesn't flake when the watcher thread is slow to
            # be scheduled under heavy parallel load.
            detected = cancel_event.wait(timeout=2.0)

            watcher.stop()

    assert detected


def test_special_key_sequence_ignored(cancel_event, fake_stdin):
    """Writing \\x1b[A (arrow up) should NOT set cancel_event"""
    read_fd, write_fd = fake_stdin

    with patch("sys.stdin") as mock_stdin:
        mock_stdin.fileno.return_value = read_fd
        mock_stdin.isatty.return_value = True

        with patch("ai_assist.escape_watcher.termios") as mock_termios, patch("ai_assist.escape_watcher.tty"):
            mock_termios.tcgetattr.return_value = [0] * 7

            watcher = EscapeWatcher(cancel_event)
            watcher.start()

            # Write escape sequence (arrow up) all at once
            os.write(write_fd, b"\x1b[A")
            time.sleep(0.15)

            watcher.stop()

    assert not cancel_event.is_set()


def test_terminal_restored_on_stop(cancel_event, fake_stdin):
    """termios.tcsetattr should be called with original settings on stop"""
    read_fd, write_fd = fake_stdin
    original_attrs = [0, 1, 2, 3, 4, 5, []]

    with patch("sys.stdin") as mock_stdin:
        mock_stdin.fileno.return_value = read_fd
        mock_stdin.isatty.return_value = True

        with patch("ai_assist.escape_watcher.termios") as mock_termios, patch("ai_assist.escape_watcher.tty"):
            mock_termios.tcgetattr.return_value = original_attrs
            mock_termios.TCSADRAIN = 1

            watcher = EscapeWatcher(cancel_event)
            watcher.start()
            watcher.stop()

            mock_termios.tcsetattr.assert_called_with(read_fd, mock_termios.TCSADRAIN, original_attrs)


def test_context_manager_lifecycle(cancel_event, fake_stdin):
    """Using 'with' should start and stop the watcher"""
    read_fd, write_fd = fake_stdin

    with patch("sys.stdin") as mock_stdin:
        mock_stdin.fileno.return_value = read_fd
        mock_stdin.isatty.return_value = True

        with patch("ai_assist.escape_watcher.termios") as mock_termios, patch("ai_assist.escape_watcher.tty"):
            mock_termios.tcgetattr.return_value = [0] * 7

            with EscapeWatcher(cancel_event) as watcher:
                assert watcher._thread is not None
                assert watcher._thread.is_alive()

            # After exiting context, thread should be stopped
            assert not watcher._thread.is_alive()


def test_non_tty_skipped(cancel_event):
    """When stdin is not a tty, no thread should be started"""
    with patch("sys.stdin") as mock_stdin:
        mock_stdin.isatty.return_value = False

        watcher = EscapeWatcher(cancel_event)
        watcher.start()

        assert watcher._thread is None

        # stop() should not crash
        watcher.stop()


def test_main_restores_terminal_on_keyboard_interrupt():
    """main() should restore terminal state even on KeyboardInterrupt"""
    original_attrs = [0, 1, 2, 3, 4, 5, []]

    def _run_then_interrupt(coro):
        coro.close()
        raise KeyboardInterrupt()

    with patch("sys.stdin") as mock_stdin, patch("ai_assist.main.asyncio") as mock_asyncio:
        mock_stdin.isatty.return_value = True
        mock_stdin.fileno.return_value = 0
        mock_asyncio.run.side_effect = _run_then_interrupt

        with (
            patch("termios.tcgetattr", return_value=original_attrs),
            patch("termios.tcsetattr") as mock_set,
            patch("termios.TCSADRAIN", 1),
        ):
            from ai_assist.main import main

            main()
            mock_set.assert_called_with(0, 1, original_attrs)
