"""Regression tests for execute_command allowlist bypasses.

Every command below must either be rejected (non-interactive, no callback) or
run normally. Bypass attempts try to create a marker file; it must not exist.
"""

import pytest

from ai_assist.config import AiAssistConfig
from ai_assist.filesystem_tools import FilesystemTools


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f").write_text("x\n")
    return tmp_path


def _tools(workdir, commands=("grep", "cat", "head", "sort", "yq", "find")):
    config = AiAssistConfig(anthropic_api_key="test", allowed_commands=list(commands), allowed_paths=[str(workdir)])
    return FilesystemTools(config, load_user_config=False)


BYPASSES = [
    # Builtins and keywords that run another command
    "eval 'touch {m}'",
    "exec touch {m}",
    "command touch {m}",
    "builtin source {m}.sh",
    "source {m}.sh",
    ". {m}.sh",
    "time touch {m}",
    "if true; then touch {m}; fi",
    "if true; then grep x f; else touch {m}; fi",
    "while false; do :; done; until true; do touch {m}; done",
    "! touch {m}",
    "{{ touch {m}; }}",
    "trap 'touch {m}' EXIT",
    "hash -p /usr/bin/touch grep; grep {m}",
    # Splitting
    "grep x f\ntouch {m}",
    "grep x f # comment\ntouch {m}",
    "grep $(touch {m}) f",
    "grep `touch {m}` f",
    'grep "$(touch {m})" f',
    "grep $(grep x $(touch {m})) f",
    "cat <(touch {m})",
    "echo $(( $(touch {m}) + 1 ))",
    "cat <<EOF\ndon't\nEOF\ngrep '#'; touch {m}",
    "cat <<EOF\n$(touch {m})\nEOF",
    "cat <<-EOF\n\t`touch {m}`\n\tEOF",
    "case x in x) touch {m};; esac",
    "((y = 1 << 2))\ntouch {m}",
    "(grep x f; touch {m})",
    # Writes outside allowed paths
    "find -H / -maxdepth 0 -newer {m}",
    "env find . -exec touch {m} ;",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("template", BYPASSES)
async def test_bypass_is_rejected(workdir, template):
    marker = workdir / "pwned"
    tools = _tools(workdir)

    result = await tools.execute_tool("execute_command", {"command": template.format(m=marker)})

    assert result.startswith("Error"), result
    assert not marker.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "echo hi > {outside}",
        "grep x f >> {outside}",
        "grep x f 2> {outside}",
        "grep x f &> {outside}",
        "echo hi >$HOME/x",
        "sort -o {outside} f",
        "sort --output={outside} f",
        "yq -i '.a = 1' {outside}",
        "echo '[]' > {protected}",
    ],
)
async def test_write_outside_allowed_paths_is_rejected(workdir, tmp_path_factory, isolate_config_dir, command):
    outside = tmp_path_factory.mktemp("outside") / "target"
    protected = isolate_config_dir / "allowed_commands.json"
    tools = _tools(workdir)

    result = await tools.execute_tool(
        "execute_command", {"command": command.format(outside=outside, protected=protected)}
    )

    assert result.startswith("Error"), result
    assert not outside.exists()
    assert not protected.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "cd {w} && echo hello && pwd",
        "for i in 1 2; do echo $i; done",
        "if [ -f f ]; then echo yes; else echo no; fi",
        "while false; do echo never; done",
        "grep -c x f 2>&1 | head -1",
        "grep x f 2>/dev/null",
        "echo $((1 + 2))",
        "command -v grep",
        "time grep x f",
        "grep x f\ngrep y f || true",
        "grep x \\\n  f",
        "{{ grep x f; }}",
        "! grep zzz f",
        "echo hi > {w}/out.txt",
        "sort -o {w}/sorted f",
        "cat > {w}/heredoc.txt <<'EOF'\nhello don't\n$(touch {w}/pwned)\nEOF",
        "x=$(grep x f); echo $x",
        "case x in x) echo match;; *) echo other;; esac",
        "for ((i = 0; i < 2; i++)); do echo $i; done",
        "(cd {w} && grep x f)",
    ],
)
async def test_legitimate_commands_still_run(workdir, command):
    tools = _tools(workdir)

    result = await tools.execute_tool("execute_command", {"command": command.format(w=workdir)})

    assert not result.startswith("Error"), result
    assert "Exit code: 0" in result
    assert not (workdir / "pwned").exists()


@pytest.mark.asyncio
async def test_code_running_builtin_allowed_after_confirmation(workdir):
    tools = _tools(workdir)
    prompts = []

    async def approve(description):
        prompts.append(description)
        return True

    tools.confirmation_callback = approve

    result = await tools.execute_tool("execute_command", {"command": "eval 'echo ran'"})

    assert prompts
    assert "ran" in result


@pytest.mark.asyncio
async def test_code_running_builtin_allowed_when_allowlisted(workdir):
    tools = _tools(workdir, commands=("source",))
    (workdir / "env.sh").write_text("echo sourced\n")

    result = await tools.execute_tool("execute_command", {"command": "source ./env.sh"})

    assert "sourced" in result
