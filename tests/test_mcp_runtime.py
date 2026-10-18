"""Tests for MCP runtime scope selection and exclusive leases."""

from ai_assist.agent import AiAssistAgent
from ai_assist.config import MCPScopeConfig, MCPServerConfig, load_mcp_servers_from_yaml
from ai_assist.mcp_runtime import MCPRuntimeIdentity, MCPServerLease


def test_scope_defaults_preserve_existing_unrestricted_behavior():
    scope = MCPServerConfig(command="server").scope

    assert scope.skip_reason("interactive", None) is None
    assert scope.skip_reason("monitor", None) is None
    assert scope.skip_reason("query", None) is None
    assert scope.skip_reason("run", None) is None


def test_scope_filters_mode_and_named_interactive_sessions():
    scope = MCPScopeConfig(modes={"interactive"}, interactive_sessions={"desk"})

    assert scope.skip_reason("monitor", None) == "mode 'monitor' is not allowed"
    assert scope.skip_reason("interactive", None) == "interactive session 'unnamed' is not allowed (allowed: desk)"
    assert scope.skip_reason("interactive", "laptop") == "interactive session 'laptop' is not allowed (allowed: desk)"
    assert scope.skip_reason("interactive", "desk") is None


def test_scope_is_loaded_from_yaml(tmp_path):
    path = tmp_path / "mcp_servers.yaml"
    path.write_text("""servers:
  telegram:
    command: telegram-mcp
    scope:
      modes: [monitor]
      exclusive: true
""")

    scope = load_mcp_servers_from_yaml(path)["telegram"].scope

    assert scope.modes == {"monitor"}
    assert scope.exclusive is True


def test_exclusive_lease_reports_holder_and_releases(tmp_path):
    runtime = MCPRuntimeIdentity(mode="interactive", session_name="desk")
    first = MCPServerLease(tmp_path, "telegram", runtime)
    second = MCPServerLease(tmp_path, "telegram", MCPRuntimeIdentity(mode="monitor"))

    assert first.acquire() is None
    owner = second.acquire()
    assert owner is not None
    assert owner["mode"] == "interactive"
    assert owner["session_name"] == "desk"

    first.release()
    assert MCPServerLease.current_owner(first.path) == {}
    assert second.acquire() is None
    second.release()


def test_agent_records_scope_skip_without_starting_server():
    agent = object.__new__(AiAssistAgent)
    agent.runtime_identity = MCPRuntimeIdentity(mode="interactive", session_name="desk")
    agent._mcp_connection_status = {}
    agent._mcp_leases = {}
    agent.sessions = {}
    config = MCPServerConfig(command="telegram", scope=MCPScopeConfig(modes={"monitor"}, exclusive=True))

    assert agent._may_connect_server("telegram", config) is False
    assert agent.get_mcp_connection_status() == {"telegram": "skipped: mode 'interactive' is not allowed"}
