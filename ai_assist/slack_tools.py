"""Agent tools for Slack webhook integration"""

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class SlackTools:
    """Agent tools for posting messages to Slack via webhooks"""

    def __init__(self) -> None:
        self.webhook_url = os.getenv("SLACK_WEBHOOK_URL")  # Default/personal channel
        self.team_webhook_url = os.getenv("SLACK_TEAM_WEBHOOK_URL")  # Team channel

    def get_tool_definitions(self) -> list[dict]:
        """Return tool definitions for Slack integration"""
        if not self.webhook_url and not self.team_webhook_url:
            # Don't expose tools if no webhook is configured
            logger.debug("No Slack webhooks configured - Slack tools disabled")
            return []

        # Build channel description based on configured webhooks
        channel_desc = 'Target channel: "default" (personal/logs)'
        if self.team_webhook_url:
            channel_desc += ' or "team" (team announcements/broadcasts)'
        channel_desc += ". Defaults to personal channel if not specified."

        return [
            {
                "name": "internal__post_slack_message",
                "description": (
                    "Post a message to Slack via webhook. Supports plain text and rich formatting "
                    "with markdown, emojis, and structured blocks. Use this to send notifications, "
                    "alerts, status updates, or any information that should be visible in Slack. "
                    f"{channel_desc}"
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The message text to post. Supports Slack markdown formatting.",
                        },
                        "channel": {
                            "type": "string",
                            "description": (
                                'Target channel: "default" for personal/logs (default), '
                                '"team" for team announcements/broadcasts.'
                            ),
                            "enum": ["default", "team"],
                        },
                        "blocks": {
                            "type": "array",
                            "description": (
                                "Optional: Rich message blocks for advanced formatting. "
                                "See https://api.slack.com/block-kit for structure."
                            ),
                            "items": {"type": "object"},
                        },
                    },
                    "required": ["text"],
                },
                "_server": "internal",
            },
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a Slack tool"""
        if tool_name == "internal__post_slack_message":
            return await self._post_message(arguments)
        return f"Unknown tool: {tool_name}"

    async def _post_message(self, args: dict[str, Any]) -> str:
        """Post a message to Slack via webhook"""
        text = args.get("text")
        if not text:
            return "Error: 'text' parameter is required"

        # Determine which webhook to use based on channel parameter
        channel = args.get("channel", "default")

        if channel == "team":
            if not self.team_webhook_url:
                return 'Error: SLACK_TEAM_WEBHOOK_URL not configured in .env file. Use channel="default" or configure the team webhook.'
            webhook_url = self.team_webhook_url
            channel_name = "team channel"
        else:  # default
            if not self.webhook_url:
                return "Error: SLACK_WEBHOOK_URL not configured in .env file"
            webhook_url = self.webhook_url
            channel_name = "default channel"

        # Build payload
        payload = {"text": text}

        # Add blocks if provided
        blocks = args.get("blocks")
        if blocks:
            payload["blocks"] = blocks

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    webhook_url,
                    json=payload,
                    timeout=10.0,
                )

                if response.status_code == 200 and response.text == "ok":
                    logger.info(f"Posted message to Slack {channel_name}: {text[:50]}...")
                    return f"✓ Message posted successfully to Slack ({channel_name})"
                else:
                    error_msg = f"Slack API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return f"Error: {error_msg}"

        except httpx.TimeoutException:
            error_msg = f"Timeout while posting to Slack {channel_name} (>10s)"
            logger.error(error_msg)
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"Failed to post to Slack {channel_name}: {e}"
            logger.error(error_msg)
            return f"Error: {error_msg}"
