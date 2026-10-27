"""Auto-mode classification and git/ssh argument checks."""

import pytest

from ai_assist.config import AiAssistConfig
from ai_assist.filesystem_tools import FilesystemTools


@pytest.fixture
def auto_tools(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = AiAssistConfig(anthropic_api_key="test", permission_mode="auto", allowed_paths=[str(tmp_path)])
    return FilesystemTools(config, load_user_config=False)


@pytest.mark.parametrize(
    "command",
    [
        "sqlite3 :memory: \"select writefile('/tmp/x', 'data')\"",
        "sqlite3 :memory: \"select readfile('/etc/hostname')\"",
        "sqlite3 :memory: \"select edit('x', 'vi')\"",
        "sqlite3 /home/someone/.mozilla/cookies.sqlite 'select * from moz_cookies'",
        "gh api -X DELETE repos/o/r",
        "gh api -XPOST repos/o/r/issues",
        "gh api --method=PATCH repos/o/r",
        "gh api repos/o/r/issues -f title=x",
        "gh api repos/o/r/issues -F title=x",
        "gh api repos/o/r/issues --raw-field title=x",
        "gh api repos/o/r/issues --input body.json",
        "git diff --output=/tmp/x",
        "git log --ext-diff",
        "git show --textconv HEAD",
        "git branch -D main",
        "git branch new-branch",
        "git branch -m old new",
    ],
)
def test_not_auto_safe(auto_tools, command):
    assert not auto_tools._is_auto_command_safe(command)


@pytest.mark.parametrize(
    "command",
    [
        "sqlite3 :memory: 'select 1'",
        "sqlite3 data.db 'select count(*) from t'",
        "gh api repos/o/r/pulls",
        "gh api --method GET repos/o/r",
        "git diff --stat",
        "git log --oneline -5",
        "git branch",
        "git branch -a -v",
        "git branch --contains abc123",
    ],
)
def test_still_auto_safe(auto_tools, command):
    assert auto_tools._is_auto_command_safe(command)


@pytest.fixture
def manual_tools(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["git", "ssh"], allowed_paths=[str(tmp_path)])
    return FilesystemTools(config, load_user_config=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "git -c core.pager='touch pwned' log -1",
        "git --config-env=core.pager=X log",
        "git --exec-path=/tmp/x status",
        "git config core.fsmonitor 'touch pwned'",
        "git config --global alias.x '!touch pwned'",
        "ssh -o ProxyCommand='touch pwned' host",
        "ssh -oLocalCommand=x -oPermitLocalCommand=yes host",
        "ssh -F ./evil_config host",
    ],
)
async def test_program_running_git_ssh_args_need_confirmation(manual_tools, tmp_path, command):
    result = await manual_tools.execute_tool("execute_command", {"command": command})

    assert result.startswith("Error"), result
    assert not (tmp_path / "pwned").exists()


@pytest.mark.asyncio
async def test_git_config_read_is_fine(manual_tools):
    result = await manual_tools.execute_tool("execute_command", {"command": "git config --get user.name || true"})

    assert not result.startswith("Error"), result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("write_file", {"path": ".git/config", "content": "[core]\n\tfsmonitor = touch pwned\n"}),
        ("write_file", {"path": ".git/hooks/pre-commit", "content": "touch pwned"}),
        ("execute_command", {"command": "echo '[core]' >> .git/config"}),
    ],
)
async def test_git_internals_writes_need_confirmation(manual_tools, tmp_path, tool, args):
    result = await manual_tools.execute_tool(tool, args)

    assert result.startswith("Error"), result
    assert not (tmp_path / ".git" / "config").exists()
    assert not (tmp_path / ".git" / "hooks").exists()


@pytest.mark.asyncio
async def test_git_internals_write_after_confirmation(manual_tools, tmp_path):
    prompts = []

    async def approve(description):
        prompts.append(description)
        return True

    manual_tools.confirmation_callback = approve

    await manual_tools.execute_tool("write_file", {"path": ".git/info/exclude", "content": "x\n"})

    assert any(".git" in p for p in prompts)
    assert (tmp_path / ".git" / "info" / "exclude").exists()
