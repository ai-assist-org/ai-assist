"""Pytest configuration and fixtures"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def isolate_config_dir(tmp_path, monkeypatch):
    """Prevent tests from writing to ~/.ai-assist by redirecting get_config_dir() to a temp directory.

    Sets AI_ASSIST_CONFIG_DIR (read live inside get_config_dir) so modules that bound
    the function at import time via ``from .config import get_config_dir`` are also
    isolated, and additionally patches the symbol for callers referencing it directly.
    """
    test_config_dir = tmp_path / ".ai-assist"
    test_config_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("AI_ASSIST_CONFIG_DIR", str(test_config_dir))
    monkeypatch.delenv("AI_ASSIST_PERMISSION_MODE", raising=False)
    with patch("ai_assist.config.get_config_dir", return_value=test_config_dir):
        yield test_config_dir


@pytest.fixture(autouse=True)
def isolate_llm_env(monkeypatch):
    """Prevent a real .env (e.g. a custom-gateway/model override for manual testing)
    from changing which model/endpoint tests resolve to.
    """
    for var in (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_VERTEX_PROJECT_ID",
        "ANTHROPIC_VERTEX_REGION",
        "AI_ASSIST_API_KEY",
        "AI_ASSIST_MODEL",
        "AI_ASSIST_MODEL_LOW",
        "AI_ASSIST_MODEL_MEDIUM",
        "AI_ASSIST_MODEL_HIGH",
        "AI_ASSIST_SYNTHESIS_MODEL",
        "AI_ASSIST_COMPACTION_MODEL",
        "AI_ASSIST_MODEL_MAX_TOKENS",
        "AI_ASSIST_MODEL_CONTEXT_WINDOW",
        "AI_ASSIST_QUERY_TIMEOUT",
        "AI_ASSIST_HTTP_REFERER",
        "AI_ASSIST_X_TITLE",
        "AI_ASSIST_JEV_API_KEY",
        "AI_ASSIST_JEV_URL",
        "AI_ASSIST_JEV_MODEL",
        "AI_ASSIST_JEV_ENABLED",
        "AI_ASSIST_JEV_VERIFY_TASKS",
        "AI_ASSIST_JEV_RERANK",
        "AI_ASSIST_JEV_VERIFY_NARRATION",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def isolate_reports_dir(tmp_path, monkeypatch):
    """Prevent tests from writing to ~/ai-assist/reports by redirecting via env var."""
    test_reports_dir = tmp_path / "reports"
    test_reports_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("AI_ASSIST_REPORTS_DIR", str(test_reports_dir))
    yield test_reports_dir


@pytest.fixture(autouse=True)
def no_desktop_notifications():
    """Prevent tests from sending real desktop notifications via notify-send."""
    with patch(
        "ai_assist.notification_channels.DesktopNotificationChannel.send",
        new_callable=AsyncMock,
        return_value=True,
    ):
        yield


@pytest.fixture(scope="session", autouse=True)
def preload_embedding_model():
    """Preload the embedding model once per worker so individual tests don't pay the cost."""
    from ai_assist.embedding import EmbeddingModel

    EmbeddingModel.get()._load()


@pytest.fixture
def make_replay_agent(tmp_path, monkeypatch):
    """Factory for an AiAssistAgent driven by a cassette (offline, deterministic).

    Usage:
        agent = await make_replay_agent(turns)
        result = await agent.query(prompt="...")

    ``turns`` is a list of cassette turn dicts (see ai_assist.testing.mock_llm).
    Internal tools run for real against an isolated temp KG; no MCP servers and no
    network are used.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from ai_assist.agent import AiAssistAgent
    from ai_assist.config import AiAssistConfig
    from ai_assist.knowledge_graph import KnowledgeGraph
    from ai_assist.testing import make_client

    created = []

    async def _factory(turns, *, model=None, max_output_tokens=None):
        kg = KnowledgeGraph(db_path=str(tmp_path / "replay_kg.db"))
        config_kwargs = {"anthropic_api_key": "test-key", "mcp_servers": {}}
        if model is not None:
            config_kwargs["model"] = model
        if max_output_tokens is not None:
            config_kwargs["model_max_output_tokens"] = max_output_tokens
        config = AiAssistConfig(**config_kwargs)
        agent = AiAssistAgent(config, knowledge_graph=kg)
        await agent.connect_to_servers()
        agent.anthropic = make_client(turns)
        created.append(kg)
        return agent

    yield _factory

    for kg in created:
        try:
            kg.conn.close()
        except Exception:
            pass


@pytest.fixture(scope="session", autouse=True)
def setup_skill_test_fixtures():
    """Create test skill fixtures for skills tests"""
    test_skills_dir = Path("/tmp/test-skills/hello")
    test_skills_dir.mkdir(parents=True, exist_ok=True)

    # Create SKILL.md for hello skill
    skill_md = test_skills_dir / "SKILL.md"
    skill_md.write_text("""---
name: hello
description: A test skill that greets users
license: MIT
---

# Hello Skill

This is a test skill for testing the skills system.

## Instructions

When greeting users, be warmly and enthusiastically welcoming.

## Examples

**User**: Hello!
**Assistant**: Hello! How can I help you today?
""")

    yield

    # Cleanup (optional - /tmp cleans itself)
