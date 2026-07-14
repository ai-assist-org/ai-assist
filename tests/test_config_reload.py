"""Tests for configuration auto-reload functionality"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_mcp_config_file_watching(tmp_path):
    """Test that mcp_servers.yaml changes are detected"""
    from ai_assist.file_watchdog import FileWatchdog

    config_file = tmp_path / "mcp_servers.yaml"
    config_file.write_text("servers: {}")

    callback_called = False

    async def callback():
        nonlocal callback_called
        callback_called = True

    watcher = FileWatchdog(config_file, callback)
    await watcher.start()

    # Modify file
    config_file.write_text("servers: {test: {}}")
    await asyncio.sleep(1.0)

    assert callback_called
    await watcher.stop()


@pytest.mark.asyncio
async def test_identity_file_watching(tmp_path):
    """Test that identity.yaml changes are detected"""
    from ai_assist.file_watchdog import FileWatchdog

    identity_file = tmp_path / "identity.yaml"
    identity_file.write_text("version: '1.0'\nuser:\n  name: 'Test'")

    callback_called = False

    async def callback():
        nonlocal callback_called
        callback_called = True

    watcher = FileWatchdog(identity_file, callback)
    await watcher.start()

    # Modify file
    identity_file.write_text("version: '1.0'\nuser:\n  name: 'Updated'")
    await asyncio.sleep(1.0)

    assert callback_called
    await watcher.stop()


@pytest.mark.asyncio
async def test_skill_md_file_watching(tmp_path):
    """Test that SKILL.md changes in skill directories are detected"""
    from ai_assist.config_watcher import ConfigWatcher
    from ai_assist.skills_manager import InstalledSkill

    # Create a fake skill directory with SKILL.md
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("---\nname: my-skill\n---\n# Original")

    # Create a mock agent with the skill installed
    skills_manager = MagicMock()
    skills_manager.installed_skills = [
        InstalledSkill(
            name="my-skill",
            source="/tmp/my-skill",
            source_type="local",
            branch="main",
            installed_at="2026-01-01T00:00:00",
            cache_path=str(skill_dir),
        )
    ]
    agent = MagicMock()
    agent.skills_manager = skills_manager

    watcher = ConfigWatcher(agent)
    await watcher._watch_skill_files()

    assert len(watcher._skill_watchers) == 1

    # Modify SKILL.md
    skill_md.write_text("---\nname: my-skill\n---\n# Updated instructions")
    await asyncio.sleep(1.5)

    # File watchers may trigger multiple events for a single write
    skills_manager.load_installed_skills.assert_called()
    assert skills_manager.load_installed_skills.call_count >= 1

    await watcher.stop()


@pytest.mark.asyncio
async def test_skills_file_watching(tmp_path):
    """Test that installed-skills.json changes are detected"""
    from ai_assist.file_watchdog import FileWatchdog

    skills_file = tmp_path / "installed-skills.json"
    skills_file.write_text('{"skills": []}')

    callback_called = False

    async def callback():
        nonlocal callback_called
        callback_called = True

    watcher = FileWatchdog(skills_file, callback)
    await watcher.start()

    # Modify file
    skills_file.write_text('{"skills": ["test"]}')
    await asyncio.sleep(1.0)

    assert callback_called
    await watcher.stop()


@pytest.mark.asyncio
async def test_reload_mcp_servers_add_server(tmp_path):
    """Test adding a new MCP server via config reload"""
    from ai_assist.agent import AiAssistAgent
    from ai_assist.config import AiAssistConfig

    # Setup agent with empty servers
    config = AiAssistConfig(mcp_servers={})
    with patch("ai_assist.agent.Anthropic"):
        agent = AiAssistAgent(config)

    assert len(agent.sessions) == 0

    # Create mcp_servers.yaml with a new server
    mcp_file = tmp_path / "mcp_servers.yaml"
    mcp_file.write_text("""
servers:
  test-server:
    command: echo
    args: ["hello"]
    enabled: true
""")

    # Mock the _run_server method to avoid actually starting a server
    agent._run_server = AsyncMock()

    # Reload MCP servers with the new config directory
    with patch("ai_assist.config.get_config_dir", return_value=tmp_path):
        await agent.reload_mcp_servers()

    # Verify the server was added
    assert "test-server" in agent.config.mcp_servers
    agent._run_server.assert_called_once()


@pytest.mark.asyncio
async def test_reload_mcp_servers_remove_server(tmp_path):
    """Test removing an MCP server via config reload"""
    from ai_assist.agent import AiAssistAgent
    from ai_assist.config import AiAssistConfig, MCPServerConfig

    # Setup agent with one server
    config = AiAssistConfig(
        mcp_servers={
            "test-server": MCPServerConfig(command="echo", args=["hello"], enabled=True),
        }
    )
    with patch("ai_assist.agent.Anthropic"):
        agent = AiAssistAgent(config)

    # Add a mock session
    mock_session = MagicMock()
    agent.sessions["test-server"] = mock_session

    # Add mock tools for this server
    agent.available_tools = [
        {"name": "test-server__tool1", "_server": "test-server"},
        {"name": "test-server__tool2", "_server": "test-server"},
    ]

    # Add mock server task
    mock_task = MagicMock()
    mock_task.cancel = MagicMock()
    agent._server_tasks = [mock_task]

    # Create empty mcp_servers.yaml
    mcp_file = tmp_path / "mcp_servers.yaml"
    mcp_file.write_text("""
servers: {}
""")

    # Reload MCP servers with the new config directory
    with patch("ai_assist.config.get_config_dir", return_value=tmp_path):
        await agent.reload_mcp_servers()

    # Verify the server was removed
    assert "test-server" not in agent.sessions
    assert len([t for t in agent.available_tools if t["_server"] == "test-server"]) == 0


@pytest.mark.asyncio
async def test_restart_mcp_server():
    """Test restarting a single MCP server"""
    from ai_assist.agent import AiAssistAgent
    from ai_assist.config import AiAssistConfig, MCPServerConfig

    # Setup agent with one server
    config = AiAssistConfig(
        mcp_servers={
            "test-server": MCPServerConfig(command="echo", args=["hello"], enabled=True),
        }
    )
    with patch("ai_assist.agent.Anthropic"):
        agent = AiAssistAgent(config)

    # Add a mock session
    mock_session = MagicMock()
    agent.sessions["test-server"] = mock_session

    # Add mock tools for this server
    agent.available_tools = [
        {"name": "test-server__tool1", "_server": "test-server"},
        {"name": "test-server__tool2", "_server": "test-server"},
    ]

    # Add mock prompts for this server
    agent.available_prompts["test-server"] = {"prompt1": MagicMock()}

    # Add mock server task
    mock_task = MagicMock()
    mock_task.get_name.return_value = "mcp_test-server"
    mock_task.cancel = MagicMock()
    agent._server_tasks = [mock_task]

    # Mock _run_server to simulate a successful reconnection
    async def fake_run_server(name, cfg):
        agent.sessions[name] = MagicMock()
        agent.available_tools.append({"name": "test-server__new_tool", "_server": name})

    agent._run_server = AsyncMock(side_effect=fake_run_server)

    await agent.restart_mcp_server("test-server")

    # Verify the old task was cancelled
    mock_task.cancel.assert_called_once()

    # Verify old session/tools/prompts were cleaned up before reconnection
    # (the fake_run_server added back a new session and tool)
    assert "test-server" in agent.sessions
    agent._run_server.assert_called_once()

    # Verify only the new tool exists (old ones were removed)
    server_tools = [t for t in agent.available_tools if t.get("_server") == "test-server"]
    assert len(server_tools) == 1
    assert server_tools[0]["name"] == "test-server__new_tool"

    # Verify prompts were cleaned
    assert "test-server" not in agent.available_prompts


@pytest.mark.asyncio
async def test_restart_mcp_server_unknown():
    """Test restarting an unknown MCP server raises ValueError"""
    from ai_assist.agent import AiAssistAgent
    from ai_assist.config import AiAssistConfig

    config = AiAssistConfig(mcp_servers={})
    with patch("ai_assist.agent.Anthropic"):
        agent = AiAssistAgent(config)

    with pytest.raises(ValueError, match="Unknown MCP server: nonexistent"):
        await agent.restart_mcp_server("nonexistent")


@pytest.mark.asyncio
async def test_run_server_retries_on_failure():
    """Test that _run_server retries with backoff after connection failure"""
    from ai_assist.agent import AiAssistAgent
    from ai_assist.config import AiAssistConfig, MCPServerConfig

    config = AiAssistConfig(
        mcp_servers={
            "test-server": MCPServerConfig(command="echo", args=["hello"], enabled=True),
        }
    )
    with patch("ai_assist.agent.Anthropic"):
        agent = AiAssistAgent(config)

    call_count = 0

    class FakeTransport:
        """Context manager that fails on first call, succeeds on second"""

        def __init__(self, fail):
            self.fail = fail

        async def __aenter__(self):
            if self.fail:
                raise ConnectionError("server down")
            return (MagicMock(), MagicMock())

        async def __aexit__(self, *args):
            pass

    def mock_transport(cfg):
        nonlocal call_count
        call_count += 1
        return FakeTransport(fail=(call_count == 1))

    agent._transport = mock_transport

    # Mock ClientSession to register the session and then wait
    async def fake_init():
        pass

    async def fake_list_tools():
        return MagicMock(tools=[])

    async def fake_list_prompts():
        return MagicMock(prompts=[])

    async def fake_list_resources():
        return MagicMock(resources=[])

    async def fake_list_resource_templates():
        return MagicMock(resourceTemplates=[])

    mock_session = MagicMock()
    mock_session.initialize = fake_init
    mock_session.list_tools = fake_list_tools
    mock_session.list_prompts = fake_list_prompts
    mock_session.list_resources = fake_list_resources
    mock_session.list_resource_templates = fake_list_resource_templates
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    sleep_calls = []
    original_sleep = asyncio.sleep

    async def mock_sleep(delay):
        sleep_calls.append(delay)
        # On backoff sleep, just continue immediately
        if delay >= 5:
            return
        await original_sleep(delay)

    with patch("ai_assist.agent.ClientSession", return_value=mock_session):
        with patch("asyncio.sleep", side_effect=mock_sleep):
            with patch("asyncio.Event") as mock_event:
                # Make Event().wait() raise CancelledError to exit after successful connect
                mock_event_instance = MagicMock()
                mock_event_instance.wait = AsyncMock(side_effect=asyncio.CancelledError)
                mock_event.return_value = mock_event_instance

                task = asyncio.create_task(agent._run_server("test-server", config.mcp_servers["test-server"]))
                await task

    assert call_count == 2
    assert 5 in sleep_calls
    assert "test-server" in agent.sessions


@pytest.mark.asyncio
async def test_run_server_stops_retry_on_cancel():
    """Test that _run_server stops retrying when cancelled during backoff"""
    from ai_assist.agent import AiAssistAgent
    from ai_assist.config import AiAssistConfig, MCPServerConfig

    config = AiAssistConfig(
        mcp_servers={
            "test-server": MCPServerConfig(command="echo", args=["hello"], enabled=True),
        }
    )
    with patch("ai_assist.agent.Anthropic"):
        agent = AiAssistAgent(config)

    class FailTransport:
        async def __aenter__(self):
            raise ConnectionError("server down")

        async def __aexit__(self, *args):
            pass

    agent._transport = lambda cfg: FailTransport()

    async def cancel_on_sleep(delay):
        if delay >= 5:
            raise asyncio.CancelledError

    with patch("asyncio.sleep", side_effect=cancel_on_sleep):
        task = asyncio.create_task(agent._run_server("test-server", config.mcp_servers["test-server"]))
        await task

    assert "test-server" not in agent.sessions
