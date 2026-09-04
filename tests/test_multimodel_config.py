"""Tests for multi-model / custom-endpoint configuration"""

import os
from unittest.mock import MagicMock, patch

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig
from ai_assist.pricing import compute_turn_cost, is_known_model


def test_custom_endpoint_config_from_env():
    """Custom endpoint env vars are loaded and derived properties work"""
    with patch.dict(
        os.environ,
        {
            "ANTHROPIC_BASE_URL": "https://openrouter.ai/api",
            "AI_ASSIST_API_KEY": "sk-or-test",
            "AI_ASSIST_MODEL": "anthropic/claude-sonnet-4.6",
            "ANTHROPIC_API_KEY": "",
        },
        clear=True,
    ):
        config = AiAssistConfig.from_env()

        assert config.anthropic_base_url == "https://openrouter.ai/api"
        assert config.llm_api_key == "sk-or-test"
        assert config.use_custom_endpoint is True
        assert config.effective_api_key == "sk-or-test"
        assert config.model == "anthropic/claude-sonnet-4.6"


def test_effective_api_key_falls_back_to_anthropic_key():
    """effective_api_key falls back to anthropic_api_key when llm_api_key unset"""
    with patch.dict(
        os.environ,
        {
            "ANTHROPIC_BASE_URL": "http://localhost:11434",
            "ANTHROPIC_API_KEY": "sk-ant-test",
        },
        clear=True,
    ):
        config = AiAssistConfig.from_env()

        assert config.llm_api_key is None
        assert config.effective_api_key == "sk-ant-test"


def test_custom_endpoint_takes_precedence_over_vertex():
    """A custom endpoint disables Vertex even when a vertex project is set"""
    with patch.dict(
        os.environ,
        {
            "ANTHROPIC_BASE_URL": "http://localhost:8000",
            "ANTHROPIC_VERTEX_PROJECT_ID": "test-project-123",
            "ANTHROPIC_API_KEY": "",
        },
        clear=True,
    ):
        config = AiAssistConfig.from_env()

        assert config.use_custom_endpoint is True
        assert config.use_vertex is False


def test_agent_uses_custom_endpoint_client():
    """AiAssistAgent builds an Anthropic client with base_url for a custom endpoint"""
    with patch.dict(
        os.environ,
        {
            "ANTHROPIC_BASE_URL": "https://openrouter.ai/api",
            "AI_ASSIST_API_KEY": "sk-or-test",
            "ANTHROPIC_API_KEY": "",
        },
        clear=True,
    ):
        config = AiAssistConfig.from_env()

        with patch("ai_assist.agent.Anthropic") as mock_anthropic:
            mock_anthropic.return_value = MagicMock()

            _agent = AiAssistAgent(config)

            mock_anthropic.assert_called_once_with(
                api_key="sk-or-test",
                base_url="https://openrouter.ai/api",
                max_retries=5,
            )


def test_agent_uses_placeholder_key_when_none_provided():
    """A custom endpoint with no key uses a placeholder so the SDK does not error"""
    with patch.dict(
        os.environ,
        {
            "ANTHROPIC_BASE_URL": "http://localhost:11434",
            "ANTHROPIC_API_KEY": "",
        },
        clear=True,
    ):
        config = AiAssistConfig.from_env()

        with patch("ai_assist.agent.Anthropic") as mock_anthropic:
            mock_anthropic.return_value = MagicMock()

            _agent = AiAssistAgent(config)

            mock_anthropic.assert_called_once_with(
                api_key="not-needed",
                base_url="http://localhost:11434",
                max_retries=5,
            )


def _make_agent(env):
    """Build an agent with a mocked Anthropic client for the given env"""
    with patch.dict(os.environ, env, clear=True):
        config = AiAssistConfig.from_env()
    with patch("ai_assist.agent.Anthropic"), patch("ai_assist.agent.AnthropicVertex"):
        return AiAssistAgent(config)


def test_capability_overrides_honored():
    """Model max-tokens and context-window overrides win over the built-in tables"""
    agent = _make_agent(
        {
            "ANTHROPIC_BASE_URL": "http://localhost:8000",
            "AI_ASSIST_MODEL": "qwen3-coder",
            "AI_ASSIST_MODEL_MAX_TOKENS": "16384",
            "AI_ASSIST_MODEL_CONTEXT_WINDOW": "131072",
        }
    )

    assert agent.get_max_tokens() == 16384
    assert agent.get_context_window_size() == 131072


def test_unknown_model_falls_back_conservatively():
    """An unknown model with no override degrades to conservative defaults"""
    agent = _make_agent(
        {
            "ANTHROPIC_BASE_URL": "http://localhost:8000",
            "AI_ASSIST_MODEL": "some-unknown-model",
        }
    )

    assert agent.get_max_tokens() == 4096
    assert agent.get_context_window_size() == 200000


def test_prompt_caching_disabled_omits_cache_control():
    """With caching disabled, the static system block carries no cache_control"""
    agent = _make_agent(
        {
            "ANTHROPIC_BASE_URL": "http://localhost:8000",
            "AI_ASSIST_MODEL": "some-model",
            "AI_ASSIST_ENABLE_CACHE": "false",
        }
    )

    blocks = agent._build_system_prompt()

    assert blocks
    assert "cache_control" not in blocks[0]


def test_prompt_caching_enabled_sets_cache_control():
    """With caching enabled (default), the static system block sets cache_control"""
    agent = _make_agent({"ANTHROPIC_API_KEY": "sk-ant-test"})

    blocks = agent._build_system_prompt()

    assert blocks
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}


def test_per_role_model_routing():
    """Per-role overrides resolve, else fall back to the main model"""
    agent = _make_agent(
        {
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "AI_ASSIST_MODEL": "claude-sonnet-4-6",
            "AI_ASSIST_SYNTHESIS_MODEL": "claude-haiku-4-5",
        }
    )

    assert agent._model_for("synthesis") == "claude-haiku-4-5"
    assert agent._model_for("compaction") == "claude-sonnet-4-6"  # no override -> main
    assert agent._model_for("main") == "claude-sonnet-4-6"


def test_model_tiers_empty_by_default():
    """With no AWL tier env vars set, model_tiers is empty (levels fall back to model)."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True):
        config = AiAssistConfig.from_env()
        assert config.model_tiers == {}


def test_model_tiers_from_env():
    """AI_ASSIST_MODEL_LOW/_MEDIUM/_HIGH populate model_tiers; unset levels omitted."""
    with patch.dict(
        os.environ,
        {
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "AI_ASSIST_MODEL_LOW": "claude-haiku-4-5",
            "AI_ASSIST_MODEL_HIGH": "claude-opus-4-8",
        },
        clear=True,
    ):
        config = AiAssistConfig.from_env()
        assert config.model_tiers == {"low": "claude-haiku-4-5", "high": "claude-opus-4-8"}


def test_pricing_zero_for_unknown_custom_model():
    """zero_if_unknown reports $0 for a model with no known pricing"""
    entry = {"input_tokens": 1000, "output_tokens": 500}

    assert is_known_model("meta-llama-3.1-70b") is False
    assert compute_turn_cost("meta-llama-3.1-70b", entry, zero_if_unknown=True) == 0.0
    # Known models are still priced even with the flag on
    assert is_known_model("claude-sonnet-4-6") is True
    assert compute_turn_cost("claude-sonnet-4-6", entry, zero_if_unknown=True) > 0.0
