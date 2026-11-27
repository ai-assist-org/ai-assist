import gzip
import hashlib
import os
import zlib
from pathlib import Path

import pytest

from ai_assist.config import AiAssistConfig
from ai_assist.egress_guard import inspect_body
from ai_assist.filesystem_tools import FilesystemTools


@pytest.mark.parametrize(
    "profile,body,rule",
    [
        ("full_dlp", b"CONFIDENTIAL_INTERNAL_PROJECT", "internal-marker"),
        ("full_dlp", b"CILAB-123", "internal-ticket"),
        ("secrets_only", b"CILAB-123", None),
        ("secrets_only", b"AKIA" + b"A" * 16, "aws-key"),
        ("none", b"AKIA" + b"A" * 16, None),
    ],
)
def test_inspection_profiles(profile: str, body: bytes, rule: str | None) -> None:
    assert inspect_body(profile, gzip.compress(body), "gzip") == rule


def test_invalid_encoding_refused() -> None:
    assert inspect_body("full_dlp", b"invalid", "gzip") == "not-inspected"


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
@pytest.mark.parametrize("verb", ["api -X POST -f", "pr comment --body"])
async def test_gh_guard_before_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verb: str) -> None:
    monkeypatch.setenv("AI_ASSIST_SANDBOX", "false")
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    shim = tmp_path / "gh"
    shim.write_text('#!/bin/sh\nfor arg do :; done\nprintf "%s" "$arg" > received\n')
    shim.chmod(0o755)
    tools = FilesystemTools(
        AiAssistConfig(allowed_commands=["gh"], allowed_paths=[str(tmp_path)]), load_user_config=False
    )

    async def send(body: str) -> str:
        return await tools.execute_tool(
            "execute_command", {"command": f"gh {verb} 'body={body}'", "working_directory": str(tmp_path)}
        )

    result = await send("CILAB-123")
    assert "internal-ticket" in result and "CILAB-123" not in result
    assert not (tmp_path / "received").exists()
    await send("public text")
    assert (tmp_path / "received").read_bytes() == b"body=public text"
