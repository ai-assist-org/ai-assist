import gzip
import hashlib
import os
import zlib
from pathlib import Path

import pytest

from ai_assist.config import AiAssistConfig
from ai_assist.egress_guard import git_rejection, inspect_body
from ai_assist.filesystem_tools import FilesystemTools


@pytest.mark.parametrize(
    "profile,body,rule",
    [
        ("full_dlp", b"CONFIDENTIAL_INTERNAL_PROJECT", "internal-marker"),
        ("full_dlp", b"CILAB-123", "internal-ticket"),
        ("full_dlp", b"See [details](https://redhat.atlassian.net/browse/EXAMPLE-1)", None),
        ("full_dlp", b"https://REDHAT.ATLASSIAN.NET/", None),
        ("full_dlp", b"https://other.atlassian.net/", None),
        ("secrets_only", b"https://redhat.atlassian.net/", None),
        ("secrets_only", b"CILAB-123", None),
        ("secrets_only", b"AKIA" + b"A" * 16, "aws-key"),
        ("none", b"AKIA" + b"A" * 16, None),
    ],
)
def test_inspection_profiles(profile: str, body: bytes, rule: str | None) -> None:
    assert inspect_body(profile, gzip.compress(body), "gzip") == rule


def test_invalid_encoding_refused() -> None:
    assert inspect_body("full_dlp", b"invalid", "gzip") == "not-inspected"


def test_uninspectable_git_push_explains_how_to_submit_a_complete_pack() -> None:
    command = b"refs/heads/main\0report-status side-band-64k\n"
    body = f"{len(command) + 4:04x}".encode() + command + b"0000"
    rejection = git_rejection("not-inspected", body)
    assert rejection and b"--no-thin" in rejection
    content_rejection = git_rejection("internal-ticket", body)
    assert content_rejection and b"--no-thin" not in content_rejection


def test_pack_scans_reconstructed_delta() -> None:
    base = b"CILAB"
    blob = b"blob 5\0" + base
    ref = hashlib.sha1(blob, usedforsecurity=False).digest()
    delta = bytes([5, 9, 0x90, 5, 4]) + b"-123"
    pack = (
        b"PACK"
        + (2).to_bytes(4)
        + (2).to_bytes(4)
        + bytes([0x35])
        + zlib.compress(base)
        + bytes([0x70 | len(delta)])
        + ref
        + zlib.compress(delta)
    )
    pack += hashlib.sha1(pack, usedforsecurity=False).digest()
    assert inspect_body("full_dlp", b"0000" + pack, git_push=True) == "internal-ticket"


@pytest.mark.asyncio
@pytest.mark.parametrize("verb", ["api --input", "pr comment --body-file"])
async def test_allowed_gh_file_command_reaches_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verb: str
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    shim = tmp_path / "gh"
    shim.write_text('#!/bin/sh\nfor arg do :; done\ncat "$arg" > received\n')
    shim.chmod(0o755)
    body = b"public Markdown with `code` and $5"
    (tmp_path / "body.md").write_bytes(body)
    tools = FilesystemTools(
        AiAssistConfig(allowed_commands=["gh"], allowed_paths=[str(tmp_path)]), load_user_config=False
    )
    await tools.execute_tool("execute_command", {"command": f"gh {verb} body.md", "working_directory": str(tmp_path)})
    assert (tmp_path / "received").read_bytes() == body
