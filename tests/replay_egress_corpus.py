"""Replay the published dcibot corpus with gh using read-only GitHub API requests."""

import hashlib
import json
import subprocess
from pathlib import Path

from ai_assist.egress_guard import inspect_body

INCIDENT = 7173906778
REPO = "repos/redhatci/ansible-collection-redhatci-ocp"
PR_DESCRIPTIONS = {
    1143: "b8eac1318c648b7dfeeac8989071845c9fab8b64aa268cf3ed539c666a874d96",
    1144: "f4ff45a8551e333e4528cbc1074a4513690eadf90e31f2a009a3ab2e69449dcb",
    1145: "838acec4bb7588e160a830096be24e814cb789657957ccfb482551b7c3477121",
    1146: "2ac6643a80e7adefc6d768ec42699d85f3506c2f0750b7e175161d7fc9eef3e0",
}


def main() -> None:
    hashes = json.loads((Path(__file__).parent / "fixtures/egress_corpus.json").read_text())
    response = subprocess.run(
        [
            "gh",
            "api",
            "--method",
            "GET",
            f"{REPO}/issues/comments",
            "-f",
            "since=2026-11-25T06:19:00Z",
            "-F",
            "per_page=100",
            "--paginate",
            "--jq",
            '.[] | select(.user.login=="dcibot" and .created_at <= "2026-11-26T18:33:59Z") | {id,body}',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    seen = set()
    for line in response.stdout.splitlines():
        row = json.loads(line)
        key = str(row["id"])
        if key not in hashes:
            continue
        body = row["body"].encode()
        if hashlib.sha256(body).hexdigest() != hashes[key]:
            raise ValueError(f"Published comment {key} changed")
        expected = "internal-ticket" if row["id"] == INCIDENT else None
        if inspect_body("full_dlp", body) != expected:
            raise ValueError(f"Unexpected verdict for comment {key}")
        seen.add(key)
    if seen != set(hashes):
        raise ValueError("Corpus incomplete")
    held_back = sum(int(key) % 2 for key in seen)
    print(f"{len(seen)} comment labels matched, {held_back} held back, 0 held-back false positives")
    for number, digest in PR_DESCRIPTIONS.items():
        row = json.loads(subprocess.check_output(["gh", "api", f"{REPO}/pulls/{number}"]))
        body = row["body"].encode()
        if row["user"]["login"] != "dcibot" or hashlib.sha256(body).hexdigest() != digest:
            raise ValueError(f"Published PR {number} changed")
        if inspect_body("full_dlp", body):
            raise ValueError(f"Unexpected refusal for PR {number}")
    print(f"{len(PR_DESCRIPTIONS)} additional held-back PR descriptions, 0 false positives")


if __name__ == "__main__":
    main()
