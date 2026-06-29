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
