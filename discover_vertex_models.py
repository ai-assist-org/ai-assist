#!/usr/bin/env python3
"""Test all known Claude models on Vertex AI"""

import sys
from pathlib import Path

# Add ai_assist to path
sys.path.insert(0, str(Path(__file__).parent))

from ai_assist.config import get_config

# All known Claude models on Vertex AI (as of May 2026)
KNOWN_MODELS = [
    ("claude-opus-4-7@default", "Claude Opus 4.7 (newest, best for coding/agents)"),
    ("claude-opus-4-6@default", "Claude Opus 4.6"),
    ("claude-sonnet-4-6@default", "Claude Sonnet 4.6 (balanced, great for coding)"),
    ("claude-sonnet-4-5@20250929", "Claude Sonnet 4.5"),
    ("claude-haiku-4-5@20251001", "Claude Haiku 4.5 (efficient, fast)"),
    ("claude-opus-4-1@20250805", "Claude Opus 4.1 (agentic search)"),
    ("claude-opus-4@20250514", "Claude Opus 4"),
    ("claude-sonnet-4@20250514", "Claude Sonnet 4"),
    ("claude-3-7-sonnet@20250219", "Claude 3.7 Sonnet"),
    ("claude-3-5-sonnet-v2@20241022", "Claude 3.5 Sonnet v2"),
    ("claude-3-5-haiku@20241022", "Claude 3.5 Haiku"),
]


def test_models():
    """Test all known Claude models on Vertex AI"""
    config = get_config()

    if not config.use_vertex:
        print("❌ Vertex AI is not configured.")
        print("Please set ANTHROPIC_VERTEX_PROJECT_ID")
        return 1

    print("=" * 70)
    print("Testing All Known Claude Models on Vertex AI")
    print("=" * 70)
    print(f"\nProject: {config.vertex_project_id}")
    print(f"Region: {config.vertex_region or 'us-east5 (default)'}")
    print(f"\nCurrently configured: {config.model}")

    from anthropic import AnthropicVertex

    vertex_kwargs = {"project_id": config.vertex_project_id}
    if config.vertex_region:
        vertex_kwargs["region"] = config.vertex_region

    available_models = []
    unavailable_models = []

    print(f"\n🔍 Testing {len(KNOWN_MODELS)} known models...")
    print("-" * 70)

    for model_id, description in KNOWN_MODELS:
        try:
            print(f"\n{model_id}")
            print(f"  {description}")

            client = AnthropicVertex(**vertex_kwargs)
            client.messages.create(
                model=model_id,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )

            print("  ✅ Available")
            available_models.append((model_id, description))

        except Exception as e:
            error_str = str(e)
            if "404" in error_str or "NOT_FOUND" in error_str:
                print("  ❌ Not found")
                unavailable_models.append((model_id, "Not found"))
            elif "403" in error_str or "PERMISSION_DENIED" in error_str:
                print("  ⚠️  Permission denied")
                unavailable_models.append((model_id, "Permission denied"))
            else:
                print(f"  ❌ Error: {error_str[:60]}")
                unavailable_models.append((model_id, "Error"))

    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    if available_models:
        print(f"\n✅ Available models ({len(available_models)}):")
        for model_id, description in available_models:
            print(f"   • {model_id}")
            print(f"     {description}")

        print("\n💡 To use a different model, update your .env file:")
        print(f"   AI_ASSIST_MODEL={available_models[0][0]}")

    if unavailable_models:
        print(f"\n❌ Unavailable models ({len(unavailable_models)}):")
        for model_id, reason in unavailable_models:
            print(f"   • {model_id} ({reason})")

    if not available_models:
        print("\n❌ No models are accessible")
        print("\n💡 Check:")
        print("   • Vertex AI API enabled: gcloud services enable aiplatform.googleapis.com")
        print("   • Authentication: gcloud auth application-default login")
        print("   • Project has access to Claude models")
        print("   • Try different regions (set ANTHROPIC_VERTEX_REGION)")
        return 1

    print(f"\n📌 Currently configured: {config.model}")
    if config.model not in [m[0] for m in available_models]:
        print("   ⚠️  Your configured model is not in the available list!")

    return 0


if __name__ == "__main__":
    sys.exit(test_models())
