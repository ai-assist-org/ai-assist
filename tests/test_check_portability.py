"""scripts/check_portability.py: shell and awk code must run on Linux and macOS (see AGENTS.md)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("check_portability", ROOT / "scripts" / "check_portability.py")
assert _spec and _spec.loader
check_portability = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_portability)


def problems(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return check_portability.check_file(path)


@pytest.mark.parametrize(
    "line",
    [
        "mapfile -t files < <(ls)",
        "readarray -t x",
        "declare -A seen",
        'echo "${name,,}"',
        'echo "${name^^}"',
        "cmd &>> log",
        "cmd |& tee log",
        "now=$(date -Iseconds)",
        "stat -c '%y %n' f",
        "sed -i 's/a/b/' f",
        "readlink -f path",
        "grep -P '\\d+' f",
        "find . -printf '%p\\n'",
        "h=$(printf x | sha1sum)",
    ],
)
def test_shell_non_portable_forms_are_reported(tmp_path, line):
    found = problems(tmp_path, "s.sh", f"#!/usr/bin/env bash\n{line}\n")
    assert len(found) == 1 and found[0].startswith(f"{tmp_path / 's.sh'}:2: "), found


@pytest.mark.parametrize(
    "line",
    [
        'while IFS= read -r f; do files+=("$f"); done < <(ls)',
        "now=$(date '+%Y-%m-%dT%H:%M:%S%z')",
        "if command -v sha1sum >/dev/null; then h() { sha1sum; }; else h() { shasum -a 1; }; fi",
        "# a comment may mention mapfile or sed -i",
        "grep -n pattern f",
    ],
)
def test_shell_portable_forms_pass(tmp_path, line):
    assert problems(tmp_path, "s.sh", f"#!/usr/bin/env bash\n{line}\n") == []


@pytest.mark.parametrize(
    "line",
    [
        '{ t = mktime("2026 01 01 00 00 00") }',
        '{ print strftime("%Y") }',
        '{ x = gensub(/a/, "b", "g") }',
        "{ n = asort(a) }",
        "{ IGNORECASE = 1 }",
        "BEGINFILE { n++ }",
        '{ print > "/dev/stdout" }',
        '{ gsub(/\\\\/, "\\\\\\\\", s) }',
    ],
)
def test_awk_gawk_only_forms_are_reported(tmp_path, line):
    found = problems(tmp_path, "x.awk", f"{line}\n")
    assert len(found) == 1 and ":1: " in found[0], found


@pytest.mark.parametrize(
    "line",
    [
        '{ gsub(/[0-9]+/, "#", d) }',
        '{ n = split(s, parts, "\\\\") }',
        "# mktime() is gawk-only: plain arithmetic instead",
        "{ print $1 > out }",
    ],
)
def test_awk_portable_forms_pass(tmp_path, line):
    assert problems(tmp_path, "x.awk", f"{line}\n") == []


def test_shebang_shell_script_without_extension_is_checked(tmp_path):
    assert problems(tmp_path, "tool", "#!/bin/bash\nmapfile -t x\n")


def test_command_line_exit_status(tmp_path):
    good, bad = tmp_path / "good.sh", tmp_path / "bad.sh"
    good.write_text("#!/bin/bash\necho ok\n")
    bad.write_text("#!/bin/bash\nsed -i s/a/b/ f\n")
    script = str(ROOT / "scripts" / "check_portability.py")
    assert subprocess.run([sys.executable, script, str(good)], check=False).returncode == 0
    result = subprocess.run([sys.executable, script, str(good), str(bad)], capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert "bad.sh:2:" in result.stdout and "sed -i" in result.stdout


def test_repository_scripts_are_portable():
    tracked = subprocess.run(
        ["git", "ls-files", "*.sh", "*.awk", "scripts/*"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    found = [p for f in tracked for p in check_portability.check_file(ROOT / f)]
    assert found == [], "\n".join(found)
