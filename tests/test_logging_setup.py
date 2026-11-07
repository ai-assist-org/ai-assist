"""Tests for logging setup."""

import logging
import logging.handlers
import os

import pytest

from ai_assist.config import _WatchedTimedRotatingFileHandler, setup_logging


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset root logger handlers before each test."""
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    root.handlers.clear()
    yield
    root.handlers.clear()
    root.handlers.extend(original_handlers)
    root.level = original_level


def test_setup_logging_creates_log_file(tmp_path):
    setup_logging(tmp_path)
    log_file = tmp_path / "logs" / "ai-assist.log"
    assert log_file.exists()


def test_setup_logging_pid_in_format(tmp_path):
    setup_logging(tmp_path)
    test_logger = logging.getLogger("test_pid_check")
    test_logger.info("pid marker")

    log_file = tmp_path / "logs" / "ai-assist.log"
    # Flush handlers
    for h in logging.getLogger().handlers:
        h.flush()
    content = log_file.read_text()
    assert str(os.getpid()) in content


def test_setup_logging_uses_watched_timed_rotating_handler(tmp_path):
    setup_logging(tmp_path)
    root = logging.getLogger()
    file_handlers = [h for h in root.handlers if isinstance(h, _WatchedTimedRotatingFileHandler)]
    assert len(file_handlers) == 1


def test_setup_logging_rotation_config(tmp_path):
    setup_logging(tmp_path)
    root = logging.getLogger()
    file_handlers = [h for h in root.handlers if isinstance(h, _WatchedTimedRotatingFileHandler)]
    handler = file_handlers[0]
    assert handler.when == "MIDNIGHT"
    assert handler.backupCount == 30


def test_setup_logging_stable_filename(tmp_path):
    """Log file should be ai-assist.log (no date in name); rotation adds date suffixes."""
    setup_logging(tmp_path)
    log_file = tmp_path / "logs" / "ai-assist.log"
    assert log_file.exists()
    log_dir = tmp_path / "logs"
    log_files = list(log_dir.iterdir())
    assert len(log_files) == 1
    assert log_files[0].name == "ai-assist.log"


def test_handler_reopens_after_rename(tmp_path):
    """After the log file is renamed, the handler should write to a new file."""
    setup_logging(tmp_path)
    log_file = tmp_path / "logs" / "ai-assist.log"
    rotated = tmp_path / "logs" / "ai-assist.log.old"

    test_logger = logging.getLogger("test_reopen")
    test_logger.info("before rename")
    for h in logging.getLogger().handlers:
        h.flush()
    assert "before rename" in log_file.read_text()

    os.rename(log_file, rotated)

    test_logger.info("after rename")
    for h in logging.getLogger().handlers:
        h.flush()

    assert log_file.exists(), "handler should have created a new log file"
    assert "after rename" in log_file.read_text()
    assert "after rename" not in rotated.read_text()


SHA = "0123456789abcdef0123456789abcdef01234567"


def _read_log(tmp_path):
    for h in logging.getLogger().handlers:
        h.flush()
    return (tmp_path / "logs" / "ai-assist.log").read_text()


def test_setup_logging_logs_source_commit(tmp_path):
    from unittest.mock import patch

    with patch("ai_assist.config.get_source_commit", return_value=f"{SHA} (uncommitted changes)"):
        setup_logging(tmp_path)
    assert f"ai-assist source commit {SHA} (uncommitted changes)" in _read_log(tmp_path)


def test_setup_logging_skips_commit_without_git(tmp_path):
    from unittest.mock import patch

    with patch("ai_assist.config.get_source_commit", return_value=None):
        setup_logging(tmp_path)
    assert "source commit" not in _read_log(tmp_path)


def _git(top_level, sha=SHA, status=""):
    """Fake subprocess.run answering the two git calls get_source_commit makes."""
    import subprocess

    def run(cmd, **kwargs):
        if cmd[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{top_level}\n{sha}\n")
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=status)
        raise AssertionError(cmd)

    return run


def _source_dir():
    from pathlib import Path

    import ai_assist

    return Path(ai_assist.__file__).resolve().parent.parent


def test_get_source_commit_clean_checkout():
    from unittest.mock import patch

    from ai_assist.config import get_source_commit

    with patch("ai_assist.config.subprocess.run", side_effect=_git(_source_dir())) as run:
        assert get_source_commit() == SHA
    assert run.call_args_list[0].kwargs["cwd"] == _source_dir()


def test_get_source_commit_reports_uncommitted_changes():
    from unittest.mock import patch

    from ai_assist.config import get_source_commit

    with patch("ai_assist.config.subprocess.run", side_effect=_git(_source_dir(), status=" M ai_assist/config.py\n")):
        assert get_source_commit() == f"{SHA} (uncommitted changes)"


def test_get_source_commit_ignores_enclosing_foreign_repo(tmp_path):
    """Installed in site-packages inside another repo: that repo's HEAD is not ours."""
    from unittest.mock import patch

    from ai_assist.config import get_source_commit

    with patch("ai_assist.config.subprocess.run", side_effect=_git(tmp_path)):
        assert get_source_commit() is None


@pytest.mark.parametrize("error", [FileNotFoundError("git"), OSError("boom")])
def test_get_source_commit_none_when_git_unavailable(error):
    from unittest.mock import patch

    from ai_assist.config import get_source_commit

    with patch("ai_assist.config.subprocess.run", side_effect=error):
        assert get_source_commit() is None


def test_get_source_commit_none_outside_a_checkout():
    import subprocess
    from unittest.mock import patch

    from ai_assist.config import get_source_commit

    completed = subprocess.CompletedProcess(args=[], returncode=128, stdout="")
    with patch("ai_assist.config.subprocess.run", return_value=completed):
        assert get_source_commit() is None
