#!/usr/bin/env python3
"""Quick test script for Slack webhook integration

Usage:
    python scripts/test_slack_webhook.py

    # Or with custom message:
    python scripts/test_slack_webhook.py "Hello from ai-assist!"
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path to import ai_assist modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from ai_assist.slack_tools import SlackTools


async def test_webhook(message: str = "🤖 Test message from ai-assist Slack webhook integration"):
    """Test the Slack webhook by posting a message"""

    # Load .env file
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✓ Loaded environment from {env_file}")
    else:
        print(f"⚠ No .env file found at {env_file}")

    # Check which webhooks are configured
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    team_webhook_url = os.getenv("SLACK_TEAM_WEBHOOK_URL")

    if not webhook_url and not team_webhook_url:
        print("\n❌ No Slack webhooks configured in .env file")
        print("\nTo configure:")
        print("1. Get your webhook URLs from: https://api.slack.com/apps → Your App → Incoming Webhooks")
        print("2. Add to .env file:")
        print("   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/... (personal/logs)")
        print("   SLACK_TEAM_WEBHOOK_URL=https://hooks.slack.com/services/... (team - optional)")
        print("3. Run this script again")
        return

    print("\n📋 Configured webhooks:")
    if webhook_url:
        print(f"   ✓ Default (personal): ...{webhook_url[-20:]}")
    else:
        print("   ⚠ Default (personal): not configured")

    if team_webhook_url:
        print(f"   ✓ Team: ...{team_webhook_url[-20:]}")
    else:
        print("   ℹ Team: not configured (optional)")

    # Create SlackTools instance and test
    slack = SlackTools()

    # Test default webhook if configured
    if webhook_url:
        print("\n📤 Testing DEFAULT webhook (personal/logs)...")
        print(f"   Message: {message}")

        result = await slack.execute_tool("internal__post_slack_message", {"text": message})
        print(f"   Result: {result}")

        if "✓" in result:
            print("   ✅ SUCCESS!")
        else:
            print("   ❌ FAILED!")

    # Test team webhook if configured
    if team_webhook_url:
        print("\n📤 Testing TEAM webhook...")
        team_message = f"{message} [TEAM CHANNEL TEST]"
        print(f"   Message: {team_message}")

        result = await slack.execute_tool("internal__post_slack_message", {"text": team_message, "channel": "team"})
        print(f"   Result: {result}")

        if "✓" in result:
            print("   ✅ SUCCESS!")
        else:
            print("   ❌ FAILED!")

    print("\n" + "=" * 60)
    print("Check your Slack channels to verify the messages arrived!")
    print("=" * 60)


if __name__ == "__main__":
    # Get custom message from command line if provided
    custom_message = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None

    if custom_message:
        asyncio.run(test_webhook(custom_message))
    else:
        asyncio.run(test_webhook())
