"""Tests for basic_interactive_mode in main.py"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_assist.main import basic_interactive_mode


@pytest.fixture
def mock_agent():
    agent = AsyncMock()
    agent.interactive_mode = False
    agent.on_inner_execution = None
    agent.renderer = MagicMock()
    agent.renderer.on_inner_execution = MagicMock()
    agent.available_prompts = {}
    return agent


@pytest.fixture
def mock_state_manager():
    sm = MagicMock()
    sm.load_conversation_context = MagicMock(return_value=None)
    sm.save_conversation_context = MagicMock()
    return sm


@pytest.mark.asyncio
async def test_eof_exits_cleanly(mock_agent, mock_state_manager):
    """EOFError (stdin closed, e.g. service mode) must exit the loop without infinite retries."""
    with patch("builtins.input", side_effect=EOFError):
        with patch("ai_assist.main.get_identity", return_value=MagicMock(get_greeting=lambda: "test")):
            # Should return without raising and without looping
            await basic_interactive_mode(mock_agent, mock_state_manager)


@pytest.mark.asyncio
async def test_exit_command_saves_context(mock_agent, mock_state_manager):
    """Typing /exit saves conversation context and exits."""
    with patch("builtins.input", side_effect=["/exit"]):
        with patch("ai_assist.main.get_identity", return_value=MagicMock(get_greeting=lambda: "test")):
            await basic_interactive_mode(mock_agent, mock_state_manager)

    mock_state_manager.save_conversation_context.assert_called_once()
