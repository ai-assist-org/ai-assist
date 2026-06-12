"""Tests for SSE transport support in MCP server connections"""

from unittest.mock import MagicMock, patch

from ai_assist.config import MCPServerConfig


class TestMCPServerConfigUrl:
    def test_url_defaults_to_none(self):
        config = MCPServerConfig(command="test")
        assert config.url is None

    def test_url_field_accepted(self):
        config = MCPServerConfig(url="http://localhost:8001/sse")
        assert config.url == "http://localhost:8001/sse"
        assert config.command == ""

    def test_command_defaults_to_empty(self):
        config = MCPServerConfig(url="http://localhost:8001/sse")
        assert config.command == ""

    def test_both_url_and_command(self):
        config = MCPServerConfig(command="test-cmd", url="http://localhost:8001/sse")
        assert config.url == "http://localhost:8001/sse"
        assert config.command == "test-cmd"


class TestTransportSelection:
    def test_transport_returns_sse_when_url_set(self):
        from ai_assist.agent import AiAssistAgent

        config = MCPServerConfig(url="http://localhost:8001/sse")
        agent = MagicMock(spec=AiAssistAgent)
        agent._transport = AiAssistAgent._transport.__get__(agent)

        with patch("ai_assist.agent.sse_client", create=True):
            with patch("mcp.client.sse.sse_client") as mock_sse_import:
                agent._transport(config)
                mock_sse_import.assert_called_once_with("http://localhost:8001/sse", sse_read_timeout=3600)

    def test_transport_returns_stdio_when_no_url(self):
        from ai_assist.agent import AiAssistAgent

        config = MCPServerConfig(command="/path/to/server", args=["--flag"])
        agent = MagicMock(spec=AiAssistAgent)
        agent._transport = AiAssistAgent._transport.__get__(agent)

        with patch("ai_assist.agent.stdio_client_fixed") as mock_stdio:
            agent._transport(config)
            mock_stdio.assert_called_once()
            call_args = mock_stdio.call_args[0][0]
            assert call_args.command == "/path/to/server"
            assert call_args.args == ["--flag"]


class TestMCPServerConfigYaml:
    def test_parse_sse_config_from_dict(self):
        data = {
            "url": "http://dci-mcp-server:8001/sse",
            "enabled": True,
            "pagination": {
                "offset_param": "offset",
                "limit_param": "limit",
                "default_page_size": 200,
                "total_field": "_meta.count",
                "data_field": "auto",
            },
        }
        config = MCPServerConfig(**data)
        assert config.url == "http://dci-mcp-server:8001/sse"
        assert config.command == ""
        assert config.env == {}
        assert config.pagination is not None
        assert config.pagination.default_page_size == 200

    def test_parse_stdio_config_from_dict(self):
        data = {
            "command": "/path/to/server.sh",
            "env": {"API_KEY": "secret"},
        }
        config = MCPServerConfig(**data)
        assert config.url is None
        assert config.command == "/path/to/server.sh"
        assert config.env == {"API_KEY": "secret"}
