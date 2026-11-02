"""Tests must not reach the real repository through git's environment variables.

Git exports GIT_DIR (and GIT_INDEX_FILE, ...) to its hooks; from a linked
worktree GIT_DIR is the worktree's git directory. A test that ran ``git init``
in a scratch directory with that inherited would re-initialize the real
repository instead, writing core.bare = true into the config shared by all
worktrees.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_git_init_in_tmp_path_stays_in_tmp_path(tmp_path):
    """A scratch git init creates its repository where asked, nowhere else."""
    _git("init", "-q", cwd=tmp_path)

    assert (tmp_path / ".git").is_dir()


def test_leaked_worktree_git_dir_does_not_reach_another_repo(tmp_path):
    """Run a scratch git init under a leaked GIT_DIR: the leaked repo stays non-bare."""
    main = tmp_path / "main"
    main.mkdir()
    _git("init", "-q", cwd=main)
    _git("commit", "-q", "--allow-empty", "-m", "init", cwd=main)
    _git("worktree", "add", "-q", str(tmp_path / "wt"), cwd=main)

    env = {**os.environ, "GIT_DIR": str(main / ".git" / "worktrees" / "wt")}
    inner = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            f"{__file__}::test_git_init_in_tmp_path_stays_in_tmp_path",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,  # the return code is asserted below, with the output
    )

    assert inner.returncode == 0, inner.stdout + inner.stderr
    assert _git("config", "--get", "core.bare", cwd=main) == "false"
