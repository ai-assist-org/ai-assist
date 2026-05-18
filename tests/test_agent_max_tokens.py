"""Tests for model max_tokens discovery"""

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig


def test_opus_46_max_tokens():
    """Claude Opus 4.6 should have 128K max output tokens"""
    config = AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-opus-4-6-20260205",
        mcp_servers={},
    )
    agent = AiAssistAgent(config)
    assert agent.get_max_tokens() == 128000


def test_opus_45_max_tokens():
    """Claude Opus 4.5 should have 64K max output tokens"""
    config = AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-opus-4-5-20251101",
        mcp_servers={},
    )
    agent = AiAssistAgent(config)
    assert agent.get_max_tokens() == 64000


def test_sonnet_4_max_tokens():
    """Claude Sonnet 4.5 should have 8192 max tokens"""
    config = AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-sonnet-4-5-20250929",
        mcp_servers={},
    )
    agent = AiAssistAgent(config)
    assert agent.get_max_tokens() == 8192


def test_sonnet_35_max_tokens():
    """Claude 3.5 Sonnet should have 8192 max tokens"""
    config = AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-3-5-sonnet-20241022",
        mcp_servers={},
    )
    agent = AiAssistAgent(config)
    assert agent.get_max_tokens() == 8192


def test_haiku_35_max_tokens():
    """Claude 3.5 Haiku should have 8192 max tokens"""
    config = AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-3-5-haiku-20241022",
        mcp_servers={},
    )
    agent = AiAssistAgent(config)
    assert agent.get_max_tokens() == 8192


def test_opus_3_max_tokens():
    """Claude 3 Opus should have 4096 max tokens"""
    config = AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-3-opus-20240229",
        mcp_servers={},
    )
    agent = AiAssistAgent(config)
    assert agent.get_max_tokens() == 4096


def test_unknown_model_default():
    """Unknown model should default to 4096 with warning"""
    config = AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-future-model-99",
        mcp_servers={},
    )
    agent = AiAssistAgent(config)
    # Should use conservative default
    assert agent.get_max_tokens() == 4096


def test_unknown_opus_4_pattern():
    """Unknown Opus 4.x model should infer 64K from name pattern (conservative)"""
    config = AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-opus-4-9-20991231",  # Future version
        mcp_servers={},
    )
    agent = AiAssistAgent(config)
    assert agent.get_max_tokens() == 64000


def test_unknown_sonnet_4_pattern():
    """Unknown Sonnet 4 model should infer 8192 from name pattern"""
    config = AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-sonnet-4-9-20991231",  # Future version
        mcp_servers={},
    )
    agent = AiAssistAgent(config)
    assert agent.get_max_tokens() == 8192


def test_vertex_model_format_opus_46():
    """Dateless model ID should work for Opus 4.6"""
    config = AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-opus-4-6",
        mcp_servers={},
    )
    agent = AiAssistAgent(config)
    assert agent.get_max_tokens() == 128000


def test_vertex_model_format_opus_45():
    """Vertex AI model format (with @) should work for Opus 4.5"""
    config = AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-opus-4-5@20251101",
        mcp_servers={},
    )
    agent = AiAssistAgent(config)
    assert agent.get_max_tokens() == 64000


def test_vertex_default_opus_46():
    """Vertex AI @default format for Opus 4.6"""
    config = AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-opus-4-6@default",
        mcp_servers={},
    )
    agent = AiAssistAgent(config)
    assert agent.get_max_tokens() == 128000


def test_vertex_default_opus_45():
    """Vertex AI @default format for Opus 4.5"""
    config = AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-opus-4-5@default",
        mcp_servers={},
    )
    agent = AiAssistAgent(config)
    assert agent.get_max_tokens() == 64000
