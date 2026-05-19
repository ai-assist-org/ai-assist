"""Tests for Slack webhook integration"""

import os
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from ai_assist.slack_tools import SlackTools


@pytest.fixture
def slack_tools_with_webhook():
    """SlackTools instance with default webhook URL configured"""
    with patch.dict(os.environ, {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/T00/B00/XXX"}):
        return SlackTools()


@pytest.fixture
def slack_tools_with_both_webhooks():
    """SlackTools instance with both default and team webhooks configured"""
    with patch.dict(
        os.environ,
        {
            "SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/T00/B00/XXX",
            "SLACK_TEAM_WEBHOOK_URL": "https://hooks.slack.com/services/T00/B01/YYY",
        },
    ):
        return SlackTools()


@pytest.fixture
def slack_tools_with_team_only():
    """SlackTools instance with only team webhook configured"""
    with patch.dict(os.environ, {"SLACK_TEAM_WEBHOOK_URL": "https://hooks.slack.com/services/T00/B01/YYY"}):
        return SlackTools()


@pytest.fixture
def slack_tools_without_webhook():
    """SlackTools instance without webhook URL"""
    with patch.dict(os.environ, {}, clear=True):
        return SlackTools()


class TestSlackTools:
    def test_get_tool_definitions_with_webhook(self, slack_tools_with_webhook):
        """Should return tool definitions when default webhook is configured"""
        tools = slack_tools_with_webhook.get_tool_definitions()
        assert len(tools) == 1
        assert tools[0]["name"] == "internal__post_slack_message"
        assert "input_schema" in tools[0]
        assert tools[0]["_server"] == "internal"
        # Should have channel parameter
        assert "channel" in tools[0]["input_schema"]["properties"]

    def test_get_tool_definitions_with_both_webhooks(self, slack_tools_with_both_webhooks):
        """Should return tool definitions when both webhooks are configured"""
        tools = slack_tools_with_both_webhooks.get_tool_definitions()
        assert len(tools) == 1
        # Description should mention both channels
        assert "team" in tools[0]["description"].lower()

    def test_get_tool_definitions_with_team_only(self, slack_tools_with_team_only):
        """Should return tool definitions when only team webhook is configured"""
        tools = slack_tools_with_team_only.get_tool_definitions()
        assert len(tools) == 1

    def test_get_tool_definitions_without_webhook(self, slack_tools_without_webhook):
        """Should return empty list when no webhook is configured"""
        tools = slack_tools_without_webhook.get_tool_definitions()
        assert len(tools) == 0

    @pytest.mark.asyncio
    async def test_post_message_success_default(self, slack_tools_with_webhook):
        """Should successfully post a message to default channel"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "ok"

        with patch("ai_assist.slack_tools.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

            result = await slack_tools_with_webhook.execute_tool(
                "internal__post_slack_message", {"text": "Hello from test!"}
            )

            assert "✓ Message posted successfully" in result
            assert "default channel" in result
            mock_client.return_value.__aenter__.return_value.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_message_success_team(self, slack_tools_with_both_webhooks):
        """Should successfully post a message to team channel"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "ok"

        with patch("ai_assist.slack_tools.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

            result = await slack_tools_with_both_webhooks.execute_tool(
                "internal__post_slack_message", {"text": "Hello team!", "channel": "team"}
            )

            assert "✓ Message posted successfully" in result
            assert "team channel" in result
            # Verify the team webhook URL was used
            call_args = mock_client.return_value.__aenter__.return_value.post.call_args
            assert "B01" in call_args[0][0]  # Team webhook has B01

    @pytest.mark.asyncio
    async def test_post_message_team_not_configured(self, slack_tools_with_webhook):
        """Should return error when team channel is requested but not configured"""
        result = await slack_tools_with_webhook.execute_tool(
            "internal__post_slack_message", {"text": "Hello team!", "channel": "team"}
        )

        assert "Error: SLACK_TEAM_WEBHOOK_URL not configured" in result

    @pytest.mark.asyncio
    async def test_post_message_without_default_webhook(self, slack_tools_with_team_only):
        """Should return error when default webhook is not configured"""
        result = await slack_tools_with_team_only.execute_tool("internal__post_slack_message", {"text": "Hello!"})

        assert "Error: SLACK_WEBHOOK_URL not configured" in result

    @pytest.mark.asyncio
    async def test_post_message_without_text(self, slack_tools_with_webhook):
        """Should return error when text parameter is missing"""
        result = await slack_tools_with_webhook.execute_tool("internal__post_slack_message", {})

        assert "Error: 'text' parameter is required" in result

    @pytest.mark.asyncio
    async def test_post_message_with_blocks(self, slack_tools_with_webhook):
        """Should post message with rich formatting blocks"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "ok"

        with patch("ai_assist.slack_tools.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

            blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "*Bold text*"}}]

            result = await slack_tools_with_webhook.execute_tool(
                "internal__post_slack_message", {"text": "Fallback text", "blocks": blocks}
            )

            assert "✓ Message posted successfully" in result

            # Verify the payload included blocks
            call_args = mock_client.return_value.__aenter__.return_value.post.call_args
            assert call_args[1]["json"]["blocks"] == blocks

    @pytest.mark.asyncio
    async def test_post_message_api_error(self, slack_tools_with_webhook):
        """Should handle Slack API errors gracefully"""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "invalid_payload"

        with patch("ai_assist.slack_tools.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

            result = await slack_tools_with_webhook.execute_tool("internal__post_slack_message", {"text": "Hello!"})

            assert "Error: Slack API error" in result
            assert "400" in result

    @pytest.mark.asyncio
    async def test_post_message_timeout(self, slack_tools_with_webhook):
        """Should handle timeout errors"""
        with patch("ai_assist.slack_tools.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("Timeout")
            )

            result = await slack_tools_with_webhook.execute_tool("internal__post_slack_message", {"text": "Hello!"})

            assert "Error: Timeout while posting to Slack" in result

    @pytest.mark.asyncio
    async def test_post_message_network_error(self, slack_tools_with_webhook):
        """Should handle network errors"""
        with patch("ai_assist.slack_tools.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.NetworkError("Network error")
            )

            result = await slack_tools_with_webhook.execute_tool("internal__post_slack_message", {"text": "Hello!"})

            assert "Error: Failed to post to Slack" in result

    @pytest.mark.asyncio
    async def test_unknown_tool(self, slack_tools_with_webhook):
        """Should return error for unknown tool name"""
        result = await slack_tools_with_webhook.execute_tool("internal__unknown_tool", {})

        assert "Unknown tool" in result
