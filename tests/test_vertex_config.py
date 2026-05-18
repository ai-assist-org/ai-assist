"""Tests for Vertex AI configuration"""

import os
from unittest.mock import MagicMock, patch

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig


def test_vertex_config_detection():
    """Test that Vertex AI config is detected correctly"""
    with patch.dict(
        os.environ,
        {
            "ANTHROPIC_VERTEX_PROJECT_ID": "test-project-123",
            "ANTHROPIC_VERTEX_REGION": "us-central1",
            "ANTHROPIC_API_KEY": "",  # No API key
        },
        clear=True,
    ):
        config = AiAssistConfig.from_env()

        assert config.vertex_project_id == "test-project-123"
        assert config.vertex_region == "us-central1"
        assert config.use_vertex is True
        assert config.anthropic_api_key == ""


def test_direct_api_key_config():
    """Test that direct API key config works"""
    with patch.dict(
        os.environ,
        {
            "ANTHROPIC_API_KEY": "sk-ant-test123",
            "ANTHROPIC_VERTEX_PROJECT_ID": "",
        },
        clear=True,
    ):
        config = AiAssistConfig.from_env()

        assert config.anthropic_api_key == "sk-ant-test123"
        assert config.vertex_project_id == ""
        assert config.use_vertex is False


def test_vertex_takes_priority_when_both_set():
    """Test that if API key is set, Vertex is not used (API key takes priority)"""
    with patch.dict(
        os.environ,
        {
            "ANTHROPIC_API_KEY": "sk-ant-test123",
            "ANTHROPIC_VERTEX_PROJECT_ID": "test-project-123",
        },
        clear=True,
    ):
        config = AiAssistConfig.from_env()

        # When API key is set, use_vertex should be False
        # (API key takes priority)
        assert config.use_vertex is False


def test_vertex_region_default():
    """Test that Vertex region defaults to None (let SDK choose)"""
    with patch.dict(
        os.environ,
        {
            "ANTHROPIC_VERTEX_PROJECT_ID": "test-project-123",
            "ANTHROPIC_API_KEY": "",
        },
        clear=True,
    ):
        config = AiAssistConfig.from_env()

        assert config.vertex_region is None  # No default - SDK will choose


def test_agent_uses_vertex_client():
    """Test that AiAssistAgent uses AnthropicVertex when configured"""
    with patch.dict(
        os.environ,
        {
            "ANTHROPIC_VERTEX_PROJECT_ID": "test-project-123",
            "ANTHROPIC_VERTEX_REGION": "us-central1",
            "ANTHROPIC_API_KEY": "",
        },
        clear=True,
    ):
        config = AiAssistConfig.from_env()

        # Mock AnthropicVertex to avoid actual GCP calls
        with patch("ai_assist.agent.AnthropicVertex") as mock_vertex:
            mock_vertex.return_value = MagicMock()

            _agent = AiAssistAgent(config)

            # Verify AnthropicVertex was called with correct params
            mock_vertex.assert_called_once_with(project_id="test-project-123", region="us-central1", max_retries=5)


def test_agent_uses_vertex_client_without_region():
    """Test that AiAssistAgent uses AnthropicVertex without region (SDK default)"""
    with patch.dict(
        os.environ,
        {
            "ANTHROPIC_VERTEX_PROJECT_ID": "test-project-123",
            "ANTHROPIC_API_KEY": "",
        },
        clear=True,
    ):
        config = AiAssistConfig.from_env()

        # Mock AnthropicVertex to avoid actual GCP calls
        with patch("ai_assist.agent.AnthropicVertex") as mock_vertex:
            mock_vertex.return_value = MagicMock()

            _agent = AiAssistAgent(config)

            # Verify AnthropicVertex was called without region param
            mock_vertex.assert_called_once_with(project_id="test-project-123", max_retries=5)


def test_agent_uses_direct_api_client():
    """Test that AiAssistAgent uses Anthropic when using API key"""
    with patch.dict(
        os.environ,
        {
            "ANTHROPIC_API_KEY": "sk-ant-test123",
            "ANTHROPIC_VERTEX_PROJECT_ID": "",
        },
        clear=True,
    ):
        config = AiAssistConfig.from_env()

        # Mock Anthropic to avoid actual API calls
        with patch("ai_assist.agent.Anthropic") as mock_anthropic:
            mock_anthropic.return_value = MagicMock()

            _agent = AiAssistAgent(config)

            # Verify Anthropic was called with correct params
            mock_anthropic.assert_called_once_with(api_key="sk-ant-test123", max_retries=5)


def test_no_credentials_configured():
    """Test detection when neither method is configured"""
    with patch.dict(
        os.environ,
        {
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_VERTEX_PROJECT_ID": "",
        },
        clear=True,
    ):
        config = AiAssistConfig.from_env()

        assert config.anthropic_api_key == ""
        assert config.vertex_project_id == ""
        assert config.use_vertex is False


def test_env_var_loading():
    """Test that environment variables are loaded correctly"""
    test_env = {
        "ANTHROPIC_VERTEX_PROJECT_ID": "my-gcp-project",
        "ANTHROPIC_VERTEX_REGION": "europe-west1",
        "AI_ASSIST_MODEL": "claude-opus-4-5-20251101",
    }

    with patch.dict(os.environ, test_env, clear=True):
        config = AiAssistConfig.from_env()

        assert config.vertex_project_id == "my-gcp-project"
        assert config.vertex_region == "europe-west1"
        assert config.model == "claude-opus-4-5-20251101"
