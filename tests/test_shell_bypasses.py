"""Regression tests for execute_command allowlist bypasses.

Every command below must either be rejected (non-interactive, no callback) or
run normally. Bypass attempts try to create a marker file; it must not exist.
"""

from unittest.mock import AsyncMock

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
async def test_write_outside_allowed_paths_is_rejected(
    workdir, tmp_path_factory, isolate_config_dir, monkeypatch, command
):
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))  # `>$HOME/x` must not reach the real home
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


# --- Environment assignments that change which program runs ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "PATH=.:$PATH grep x f",
        "export PATH=.:$PATH; grep x f",
        "PATH=. ; grep x f",
        "env PATH=. grep x f",
        "declare -x PATH=.",
        "LD_PRELOAD=./evil.so grep x f",
        "BASH_ENV=./evil.sh grep x f",
        "GIT_SSH_COMMAND='touch pwned' grep x f",
    ],
)
async def test_program_hijacking_assignment_is_rejected(workdir, command):
    tools = _tools(workdir)

    result = await tools.execute_tool("execute_command", {"command": command})

    assert result.startswith("Error"), result


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["FOO=bar grep x f", "export FOO=1; grep x f", "LC_ALL=C sort f"])
async def test_ordinary_assignments_still_run(workdir, command):
    result = await _tools(workdir).execute_tool("execute_command", {"command": command})

    assert "Exit code: 0" in result, result


@pytest.mark.asyncio
async def test_program_hijacking_assignment_runs_after_confirmation(workdir):
    tools = _tools(workdir)
    prompts = []

    async def approve(description):
        prompts.append(description)
        return True

    tools.confirmation_callback = approve

    result = await tools.execute_tool("execute_command", {"command": "PATH=/usr/bin:/bin grep x f"})

    assert any("PATH" in p for p in prompts)
    assert "Exit code: 0" in result


# --- Sensitive locations ---


@pytest.fixture
def home(tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("home")
    (home / ".ssh").mkdir()
    (home / ".ssh" / "id_rsa").write_text("PRIVATE\n")
    (home / ".aws").mkdir()
    (home / ".aws" / "credentials").write_text("PRIVATE\n")
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "cat ~/.ssh/id_rsa",
        "cat $HOME/.ssh/id_rsa",
        "cat ${{HOME}}/.aws/credentials",
        "cat ~/.s*/id_rsa",
        "cat {home}/.ssh/id_rsa",
        "cd ~ && cat .ssh/id_rsa",
        "cat < ~/.ssh/id_rsa",
        "grep -f ~/.aws/credentials f",
        "grep --file={home}/.aws/credentials f",
        "cat {config}/.env",
    ],
)
async def test_sensitive_locations_blocked_for_commands(workdir, home, isolate_config_dir, command):
    (isolate_config_dir / ".env").write_text("PRIVATE\n")
    config = AiAssistConfig(
        anthropic_api_key="test", allowed_commands=["cat", "grep"], allowed_paths=[str(workdir), str(home)]
    )
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool(
        "execute_command", {"command": command.format(home=home, config=isolate_config_dir)}
    )

    assert result.startswith("Error"), result
    assert "PRIVATE" not in result


@pytest.mark.asyncio
async def test_sensitive_locations_blocked_for_file_tools(home):
    config = AiAssistConfig(anthropic_api_key="test", allowed_paths=[str(home)])
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool("read_file", {"path": "~/.ssh/id_rsa"})

    assert result.startswith("Error"), result
    assert "PRIVATE" not in result


@pytest.mark.asyncio
async def test_other_home_files_still_readable(workdir, home):
    (home / "notes.txt").write_text("hello\n")
    config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["cat"], allowed_paths=[str(home)])
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool("execute_command", {"command": "cat ~/notes.txt"})

    assert "hello" in result


# --- python -m and combined flags ---


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("python3 -m timeit 'x'", ("python3", "<module:timeit>")),
        ("python3 -mtimeit", ("python3", "<module:timeit>")),
        ("python3 -Im http.server", ("python3", "<module:http.server>")),
        ("python3 -Ic 'print(1)'", ("python3", "<inline-code>")),
        ("python3 -cprint(1)", ("python3", "<inline-code>")),
        ("python3 -W ignore script.py", ("python3", "<script:script.py>")),
        ("python3 -X dev -u script.py", ("python3", "<script:script.py>")),
        ("python3.12 -c 'print(1)'", ("python3.12", "<inline-code>")),
        ("/usr/bin/python3 -m pip list", ("python3", "<module:pip>")),
    ],
)
def test_python_invocations_classified(command, expected):
    from ai_assist.filesystem_tools import _extract_command_argument_paths

    assert _extract_command_argument_paths(command) == [expected]


def _python_tools(workdir, commands=("python3",)):
    config = AiAssistConfig(anthropic_api_key="test", allowed_commands=list(commands), allowed_paths=[str(workdir)])
    return FilesystemTools(config, load_user_config=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    ["python3 -m timeit -n1 'pass'", "python3 -Ic 'print(1)'", "python3 -Im timeit -n1 'pass'"],
)
async def test_python_module_and_combined_flags_prompt(workdir, command):
    tools = _python_tools(workdir)
    prompts = []

    async def deny(description):
        prompts.append(description)
        return False

    tools.confirmation_callback = deny

    result = await tools.execute_tool("execute_command", {"command": command})

    assert prompts, "expected a confirmation prompt"
    assert result.startswith("Error")


@pytest.mark.asyncio
async def test_python_c_variant_name_prompts(workdir):
    import shutil

    name = next((n for n in ("python3.14", "python3.13", "python3.12", "python3.11") if shutil.which(n)), None)
    if name is None:
        pytest.skip("no versioned python binary")
    tools = _python_tools(workdir, commands=(name,))
    tools.confirmation_callback = AsyncMock(return_value=False)

    result = await tools.execute_tool("execute_command", {"command": f"{name} -c 'print(1)'"})

    tools.confirmation_callback.assert_awaited()
    assert result.startswith("Error")


@pytest.mark.asyncio
async def test_explicitly_allowlisted_module_runs_without_prompt(workdir):
    tools = _python_tools(workdir, commands=("python3 -m timeit",))
    tools.confirmation_callback = AsyncMock(return_value=False)

    result = await tools.execute_tool("execute_command", {"command": "python3 -m timeit -n1 'pass'"})

    tools.confirmation_callback.assert_not_awaited()
    assert "Exit code: 0" in result


@pytest.mark.asyncio
async def test_auto_mode_still_runs_pytest_module(workdir):
    config = AiAssistConfig(anthropic_api_key="test", permission_mode="auto", allowed_paths=[str(workdir)])
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool("execute_command", {"command": "python3 -m pytest --version"})

    assert not result.startswith("Error"), result


def test_always_allow_prefix_keeps_module():
    from ai_assist.filesystem_tools import compute_allowlist_prefix

    assert compute_allowlist_prefix("python3 -m pytest -q tests/") == "python3 -m pytest"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/root/.ssh/id_rsa",
        "/tmp/dci/../../home/../root/.ssh/id_rsa",
        "/home/someone-else/.aws/credentials",
        "/home/someone-else/.config/gcloud/credentials.db",
        "/var/lib/svc/.docker/config.json",
        "/srv/user/.netrc",
    ],
)
async def test_credential_locations_blocked_under_any_home(tmp_path, path):
    """Blocked as sensitive before any prompt or existence check, whoever's home it is."""
    config = AiAssistConfig(anthropic_api_key="test", allowed_paths=["/"])
    tools = FilesystemTools(config, load_user_config=False)
    tools.path_confirmation_callback = AsyncMock(return_value=True)

    result = await tools.execute_tool("read_file", {"path": path})

    assert "sensitive location" in result, result
    tools.path_confirmation_callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_similar_names_not_blocked(tmp_path):
    (tmp_path / "ssh-notes.txt").write_text("ok\n")
    (tmp_path / ".sshrc-example").write_text("ok\n")
    config = AiAssistConfig(anthropic_api_key="test", allowed_paths=[str(tmp_path)])
    tools = FilesystemTools(config, load_user_config=False)

    for name in ("ssh-notes.txt", ".sshrc-example"):
        result = await tools.execute_tool("read_file", {"path": str(tmp_path / name)})
        assert "ok" in result, result


# --- F2: file-copying commands must validate their destination ---


def _copy_tools(workdir, extra=()):
    cmds = ["cp", "mv", "tee", "install", "dd", "ln", "echo", *extra]
    config = AiAssistConfig(anthropic_api_key="test", allowed_commands=cmds, allowed_paths=[str(workdir)])
    return FilesystemTools(config, load_user_config=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "cp {w}/src {out}/x",
        "mv {w}/src {out}/x",
        "install {w}/src {out}/x",
        "install -t {out} {w}/src",
        "cp -t {out} {w}/src",
        "tee {out}/x",
        "tee -a {w}/ok {out}/x",
        "dd if={w}/src of={out}/x",
        "ln -s {w}/src {out}/x",
    ],
)
async def test_copy_commands_destination_outside_allowlist_blocked(workdir, tmp_path_factory, command):
    out = tmp_path_factory.mktemp("outside")
    (workdir / "src").write_text("data")
    (workdir / "ok").write_text("")
    tools = _copy_tools(workdir)

    result = await tools.execute_tool("execute_command", {"command": command.format(w=workdir, out=out)})

    assert result.startswith("Error"), result
    assert not (out / "x").exists()


@pytest.mark.asyncio
async def test_cp_into_startup_location_blocked(workdir, tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("home")
    (home / "bin").mkdir()
    (home / "bin" / "git-save").write_text("#real\n")
    monkeypatch.setenv("HOME", str(home))
    (workdir / "evil").write_text("evil")
    tools = _copy_tools(workdir)

    result = await tools.execute_tool("execute_command", {"command": "cp evil ~/bin/git-save"})

    assert result.startswith("Error"), result
    assert (home / "bin" / "git-save").read_text() == "#real\n"


@pytest.mark.asyncio
async def test_cp_into_git_dir_blocked(workdir):
    (workdir / ".git").mkdir()
    (workdir / "evil").write_text("[core]\n  fsmonitor=/tmp/x\n")
    tools = _copy_tools(workdir)

    result = await tools.execute_tool("execute_command", {"command": "cp evil .git/config"})

    assert result.startswith("Error"), result
    assert not (workdir / ".git" / "config").exists()


@pytest.mark.asyncio
async def test_cp_within_allowlist_succeeds(workdir):
    (workdir / "src").write_text("data")
    tools = _copy_tools(workdir)

    result = await tools.execute_tool("execute_command", {"command": "cp src dest"})

    assert "Exit code: 0" in result
    assert (workdir / "dest").read_text() == "data"


@pytest.mark.asyncio
async def test_redirect_outside_workspace_blocked_with_empty_allowlist(tmp_path_factory, monkeypatch):
    workspace = tmp_path_factory.mktemp("workspace")
    monkeypatch.chdir(workspace)
    outside = tmp_path_factory.mktemp("outside") / "redir"
    config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["echo"], allowed_paths=[])
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool("execute_command", {"command": f"echo pwn > {outside}"})

    assert result.startswith("Error"), result
    assert not outside.exists()


# --- Virtual/special filesystems are never writable, even with a broad allowlist ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/proc/sys/kernel/core_pattern",
        "/proc/sysrq-trigger",
        "/proc/self/mem",
        "/sys/kernel/uevent_helper",
        "/sys/class/x",
        "/dev/mem",
        "/dev/sda",
    ],
)
async def test_virtual_filesystem_writes_blocked_even_with_root_allowlist(path):
    tools = FilesystemTools(AiAssistConfig(anthropic_api_key="test", allowed_paths=["/"]), load_user_config=False)

    assert await tools.validate_write_target(path) is not None


@pytest.mark.asyncio
async def test_virtual_filesystem_reads_still_allowed(tmp_path):
    # Reading /proc is legitimate (e.g. /proc/cpuinfo); only writes are blocked
    tools = FilesystemTools(AiAssistConfig(anthropic_api_key="test", allowed_paths=["/"]), load_user_config=False)

    assert await tools._validate_path("/proc/cpuinfo", for_write=False) is None


@pytest.mark.asyncio
async def test_virtual_filesystem_save_to_file_blocked(tmp_path):
    from unittest.mock import AsyncMock, MagicMock

    from ai_assist.agent import AiAssistAgent

    agent = AiAssistAgent(config=AiAssistConfig(anthropic_api_key="test-key", allowed_paths=["/"]))
    session = AsyncMock()
    res = MagicMock()
    res.content = [MagicMock(text="x")]
    session.call_tool.return_value = res
    agent.sessions = {"dci": session}

    result = await agent._execute_tool("dci__q", {"__save_to_file": "/proc/sys/kernel/core_pattern"})

    assert result.startswith("Error")


@pytest.mark.asyncio
async def test_cp_to_proc_blocked(workdir):
    (workdir / "evil").write_text("|/tmp/x")
    config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["cp"], allowed_paths=["/"])
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool("execute_command", {"command": "cp evil /proc/sys/kernel/core_pattern"})

    assert result.startswith("Error"), result


# --- Interpreters that run arbitrary code/scripts/packages must be confirm-gated ---


def _interp_tools(workdir, commands):
    config = AiAssistConfig(anthropic_api_key="test", allowed_commands=list(commands), allowed_paths=[str(workdir)])
    return FilesystemTools(config, load_user_config=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("allow", "command"),
    [
        (["python3"], "python3 {w}/s.py"),
        (["awk"], "awk 'BEGIN{{system(\"/bin/true\")}}'"),
        (["awk"], "awk -f {w}/prog.awk {w}/data"),
        (["gawk"], "gawk 'BEGIN{{print 1}}'"),
        (["perl"], "perl -e 'system(\"/bin/true\")'"),
        (["ruby"], "ruby -e 'puts 1'"),
        (["uvx"], "uvx ruff check ."),
        (["uv"], "uv run python -c 1"),
        (["npx"], "npx cowsay hi"),
        (["pipx"], "pipx run black ."),
    ],
)
async def test_interpreter_code_exec_denied_non_interactive(workdir, allow, command):
    (workdir / "s.py").write_text("print(1)")
    (workdir / "prog.awk").write_text("BEGIN{print 1}")
    (workdir / "data").write_text("x")
    tools = _interp_tools(workdir, allow)

    result = await tools.execute_tool("execute_command", {"command": command.format(w=workdir)})

    assert result.startswith("Error"), result


@pytest.mark.asyncio
async def test_interpreter_code_exec_prompts_interactive(workdir):
    (workdir / "s.py").write_text("print(1)")
    tools = _interp_tools(workdir, ["python3"])
    prompts = []

    async def approve(d):
        prompts.append(d)
        return True

    tools.confirmation_callback = approve

    result = await tools.execute_tool("execute_command", {"command": f"python3 {workdir}/s.py"})

    assert prompts
    assert "Exit code: 0" in result


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["awk --version", "python3 --version", "uvx --version", "npx --version"])
async def test_interpreter_version_is_benign(workdir, command):
    tools = _interp_tools(workdir, command.split()[:1])
    tools.confirmation_callback = AsyncMock(return_value=False)

    result = await tools.execute_tool("execute_command", {"command": command})

    tools.confirmation_callback.assert_not_awaited()
    assert "Exit code: 0" in result or "not allowed" not in result.lower()


@pytest.mark.asyncio
async def test_explicitly_allowlisted_interpreter_invocation_runs(workdir):
    (workdir / "s.py").write_text("print(1)")
    tools = _interp_tools(workdir, [f"python3 {workdir}/s.py"])
    tools.confirmation_callback = AsyncMock(return_value=False)

    result = await tools.execute_tool("execute_command", {"command": f"python3 {workdir}/s.py"})

    tools.confirmation_callback.assert_not_awaited()
    assert "Exit code: 0" in result


@pytest.mark.asyncio
async def test_python_m_pytest_still_auto_in_auto_mode(workdir, monkeypatch):
    monkeypatch.chdir(workdir)
    config = AiAssistConfig(anthropic_api_key="test", permission_mode="auto", allowed_paths=[str(workdir)])
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool("execute_command", {"command": "python3 -m pytest --version"})

    assert not result.startswith("Error"), result


# --- tar / unzip write destinations ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "tar -cf {out}/a.tar {w}/data",
        "tar cf {out}/a.tar {w}/data",
        "tar -xf {w}/a.tar -C {out}",
        "unzip {w}/a.zip -d {out}",
    ],
)
async def test_tar_unzip_destination_outside_blocked(workdir, tmp_path_factory, command):
    out = tmp_path_factory.mktemp("outside")
    (workdir / "data").write_text("x")
    (workdir / "a.tar").write_text("x")
    (workdir / "a.zip").write_text("x")
    tools = _interp_tools(workdir, ["tar", "unzip"])

    result = await tools.execute_tool("execute_command", {"command": command.format(w=workdir, out=out)})

    assert result.startswith("Error"), result


# --- Reads via shell commands must respect the path allowlist (#1) ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "cat {out}/data.txt",
        "head {out}/data.txt",
        "grep X {out}/data.txt",
        "cut -c1 {out}/data.txt",
        "base64 {out}/data.txt",
        "cp {out}/data.txt dest.txt",
        "wc -l {out}/data.txt",
    ],
)
async def test_shell_read_outside_allowlist_blocked(workdir, tmp_path_factory, monkeypatch, command):
    monkeypatch.chdir(workdir)
    outside = tmp_path_factory.mktemp("secret")
    (outside / "data.txt").write_text("SECRET-XYZ")
    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_commands=["cat", "head", "grep", "cut", "base64", "cp", "wc"],
        allowed_paths=[str(workdir)],
    )
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool("execute_command", {"command": command.format(out=outside)})

    assert result.startswith("Error"), result
    assert "SECRET-XYZ" not in result


@pytest.mark.asyncio
async def test_shell_read_inside_allowlist_ok(workdir):
    (workdir / "data.txt").write_text("hello-data")
    tools = FilesystemTools(
        AiAssistConfig(anthropic_api_key="test", allowed_commands=["cat"], allowed_paths=[str(workdir)]),
        load_user_config=False,
    )

    result = await tools.execute_tool("execute_command", {"command": f"cat {workdir}/data.txt"})

    assert "hello-data" in result


@pytest.mark.asyncio
async def test_grep_pattern_not_treated_as_path(workdir, monkeypatch):
    monkeypatch.chdir(workdir)
    (workdir / "f.txt").write_text("find-me\n")
    tools = FilesystemTools(
        AiAssistConfig(anthropic_api_key="test", allowed_commands=["grep"], allowed_paths=[str(workdir)]),
        load_user_config=False,
    )

    result = await tools.execute_tool("execute_command", {"command": "grep find-me f.txt"})

    assert "find-me" in result


@pytest.mark.asyncio
@pytest.mark.parametrize("name", [".env~", ".env.bak", ".env.local"])
async def test_env_backups_are_sensitive(workdir, monkeypatch, name):
    monkeypatch.chdir(workdir)
    (workdir / name).write_text("JEV_API_KEY=secret")
    tools = FilesystemTools(
        AiAssistConfig(anthropic_api_key="test", allowed_commands=["cat"], allowed_paths=[str(workdir)]),
        load_user_config=False,
    )

    result = await tools.execute_tool("execute_command", {"command": f"cat {name}"})

    assert result.startswith("Error"), result
    assert "secret" not in result


@pytest.mark.asyncio
async def test_awl_authorized_read_outside_allowed(workdir, tmp_path_factory):
    # A user-started AWL script may read its own data outside the allowlist without prompts
    outside = tmp_path_factory.mktemp("data")
    (outside / "x.txt").write_text("ok-content")
    tools = FilesystemTools(
        AiAssistConfig(anthropic_api_key="test", allowed_commands=["cat"], allowed_paths=[str(workdir)]),
        load_user_config=False,
    )
    tools.awl_authorized_commands = {"cat"}

    result = await tools.execute_tool("execute_command", {"command": f"cat {outside}/x.txt"})

    assert "ok-content" in result
