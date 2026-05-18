"""Tests for TUI interactive mode"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_assist.agent import AiAssistAgent
from ai_assist.output import PlainRenderer
from ai_assist.state import StateManager
from ai_assist.tui_interactive import (
    handle_clear_cache_command,
    handle_help_command,
    handle_history_command,
    handle_status_command,
    tui_interactive_mode,
)


@pytest.fixture
def mock_agent():
    """Create a mock agent"""
    agent = AsyncMock(spec=AiAssistAgent)
    agent.query = AsyncMock(return_value="Test response")

    # Mock streaming query to yield text and done signal
    async def mock_query_streaming(prompt=None, messages=None, progress_callback=None, cancel_event=None):
        if progress_callback:
            progress_callback("thinking", 0, 10, None)
            progress_callback("calling_claude", 1, 10, None)
            progress_callback("complete", 1, 10, None)
        yield "Test response"
        yield {"type": "done", "turns": 1}

    agent.query_streaming = mock_query_streaming

    # Mock new KG-related methods
    agent.get_last_kg_saved_count = MagicMock(return_value=0)
    agent.clear_tool_calls = MagicMock()
    agent.kg_save_enabled = True

    # Mock skills manager
    agent.skills_manager = MagicMock()
    agent.skills_manager.installed_skills = []

    # Mock filesystem tools for security confirmation callback
    agent.filesystem_tools = MagicMock()
    agent.filesystem_tools.confirmation_callback = None

    # Mock adaptive truncation limits
    agent.get_truncation_limits = MagicMock(
        return_value={
            "max_message_chars": 40000,
            "max_total_chars": 480000,
            "context_window_tokens": 200000,
            "usable_tokens": 150000,
        }
    )

    # Mock renderer
    from ai_assist.output import PlainRenderer

    agent.renderer = PlainRenderer()
    agent.on_inner_execution = agent.renderer.on_inner_execution

    return agent


@pytest.fixture
def mock_state_manager():
    """Create a mock state manager"""
    manager = MagicMock(spec=StateManager)
    manager.get_stats = MagicMock(return_value={"cache_entries": 10, "monitors": 2})
    manager.get_history = MagicMock(return_value=[{"timestamp": "2026-02-05"}])
    manager.cleanup_expired_cache = MagicMock(return_value=5)
    manager.save_conversation_context = MagicMock()
    manager.load_conversation_context = MagicMock(return_value=None)
    return manager


@pytest.mark.asyncio
async def test_status_command(mock_state_manager):
    """Test /status command displays statistics"""
    from io import StringIO

    from rich.console import Console

    output = StringIO()
    console = Console(file=output, force_terminal=True)

    await handle_status_command(mock_state_manager, console)

    mock_state_manager.get_stats.assert_called_once()
    output_text = output.getvalue()
    assert "State Statistics" in output_text


@pytest.mark.asyncio
async def test_history_command(mock_state_manager):
    """Test /history command displays recent checks"""
    from io import StringIO

    from rich.console import Console

    output = StringIO()
    console = Console(file=output, force_terminal=True)

    await handle_history_command(mock_state_manager, console)

    mock_state_manager.get_history.assert_called_once_with("jira_monitor", limit=5)
    output_text = output.getvalue()
    assert "Recent Jira checks" in output_text


@pytest.mark.asyncio
async def test_clear_cache_command(mock_state_manager):
    """Test /clear-cache command clears cache"""
    from io import StringIO

    from rich.console import Console

    output = StringIO()
    console = Console(file=output, force_terminal=False)  # Disable colors for testing

    await handle_clear_cache_command(mock_state_manager, console)

    mock_state_manager.cleanup_expired_cache.assert_called_once()
    output_text = output.getvalue()
    assert "Cleared 5 cache entries" in output_text


@pytest.mark.asyncio
async def test_help_command():
    """Test /help command displays help text"""
    from io import StringIO

    from rich.console import Console

    output = StringIO()
    console = Console(file=output, force_terminal=True)

    await handle_help_command(console)

    output_text = output.getvalue()
    assert "ai-assist Interactive Mode Help" in output_text
    assert "/status" in output_text
    assert "/history" in output_text


@pytest.mark.asyncio
async def test_tui_mode_initializes(mock_agent, mock_state_manager):
    """Test TUI mode initializes without errors"""
    with patch("ai_assist.tui_interactive.PromptSession") as mock_session_class:
        # Simulate user typing /exit
        mock_session = AsyncMock()
        mock_session.prompt_async = AsyncMock(side_effect=["/exit"])
        mock_session_class.return_value = mock_session

        await tui_interactive_mode(mock_agent, mock_state_manager)

        # Verify session was created
        mock_session_class.assert_called_once()
        # Verify conversation was saved
        mock_state_manager.save_conversation_context.assert_called_once()


@pytest.mark.asyncio
async def test_multiline_input_handling(mock_state_manager):
    """Test multi-line input is parsed correctly"""
    # Create mock agent with streaming support
    agent = AsyncMock(spec=AiAssistAgent)
    streaming_called = []

    async def mock_streaming(prompt=None, messages=None, progress_callback=None, cancel_event=None):
        # Track what was called (either prompt or last message)
        if messages:
            streaming_called.append(messages[-1]["content"])
        else:
            streaming_called.append(prompt)

        response_text = messages[-1]["content"] if messages else prompt
        yield "Response to: " + response_text
        yield {"type": "done", "turns": 1}

    agent.query_streaming = mock_streaming
    agent.get_last_kg_saved_count = MagicMock(return_value=0)
    agent.clear_tool_calls = MagicMock()
    agent.kg_save_enabled = True
    agent.skills_manager = MagicMock()
    agent.skills_manager.installed_skills = []
    agent.filesystem_tools = MagicMock()
    agent.filesystem_tools.confirmation_callback = None
    agent.renderer = PlainRenderer()
    agent.on_inner_execution = agent.renderer.on_inner_execution

    with patch("ai_assist.tui_interactive.PromptSession") as mock_session_class:
        # Simulate multi-line input then exit
        mock_session = AsyncMock()
        mock_session.prompt_async = AsyncMock(side_effect=["Line 1\nLine 2\nLine 3", "/exit"])
        mock_session_class.return_value = mock_session

        await tui_interactive_mode(agent, mock_state_manager)

        # Verify agent was called with multi-line input
        assert len(streaming_called) == 1
        assert streaming_called[0] == "Line 1\nLine 2\nLine 3"


@pytest.mark.asyncio
async def test_empty_input_ignored(mock_agent, mock_state_manager):
    """Test empty input is ignored"""
    with patch("ai_assist.tui_interactive.PromptSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session.prompt_async = AsyncMock(side_effect=["", "   ", "/exit"])  # Empty input  # Whitespace only
        mock_session_class.return_value = mock_session

        await tui_interactive_mode(mock_agent, mock_state_manager)

        # Agent should not be called for empty inputs
        mock_agent.query.assert_not_called()


@pytest.mark.asyncio
async def test_conversation_tracking(mock_state_manager):
    """Test conversation context is tracked"""
    # Create mock agent with streaming
    agent = AsyncMock(spec=AiAssistAgent)

    async def mock_streaming(prompt=None, messages=None, progress_callback=None, cancel_event=None):
        yield "Test response"
        yield {"type": "done", "turns": 1}

    agent.query_streaming = mock_streaming
    agent.get_last_kg_saved_count = MagicMock(return_value=0)
    agent.clear_tool_calls = MagicMock()
    agent.kg_save_enabled = True
    agent.skills_manager = MagicMock()
    agent.skills_manager.installed_skills = []
    agent.filesystem_tools = MagicMock()
    agent.filesystem_tools.confirmation_callback = None
    agent.get_truncation_limits = MagicMock(
        return_value={
            "max_message_chars": 40000,
            "max_total_chars": 480000,
            "context_window_tokens": 200000,
            "usable_tokens": 150000,
        }
    )
    agent.renderer = PlainRenderer()
    agent.on_inner_execution = agent.renderer.on_inner_execution

    with patch("ai_assist.tui_interactive.PromptSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session.prompt_async = AsyncMock(side_effect=["Test question", "/exit"])
        mock_session_class.return_value = mock_session

        await tui_interactive_mode(agent, mock_state_manager)

        # Verify conversation was saved with messages
        mock_state_manager.save_conversation_context.assert_called_once()
        call_args = mock_state_manager.save_conversation_context.call_args
        assert call_args[0][0] == "last_interactive_session"
        messages = call_args[0][1]["messages"]
        assert len(messages) == 1
        assert messages[0]["user"] == "Test question"
        assert messages[0]["assistant"] == "Test response"


@pytest.mark.asyncio
async def test_keyboard_interrupt_handling(mock_agent, mock_state_manager):
    """Test KeyboardInterrupt is handled gracefully"""
    with patch("ai_assist.tui_interactive.PromptSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session.prompt_async = AsyncMock(side_effect=KeyboardInterrupt)
        mock_session_class.return_value = mock_session

        await tui_interactive_mode(mock_agent, mock_state_manager)

        # Should save conversation before exiting
        mock_state_manager.save_conversation_context.assert_called_once()


@pytest.mark.asyncio
async def test_eoferror_handling(mock_agent, mock_state_manager):
    """Test EOFError (Ctrl-D) is handled gracefully"""
    with patch("ai_assist.tui_interactive.PromptSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session.prompt_async = AsyncMock(side_effect=EOFError)
        mock_session_class.return_value = mock_session

        await tui_interactive_mode(mock_agent, mock_state_manager)

        # Should save conversation before exiting
        mock_state_manager.save_conversation_context.assert_called_once()


@pytest.mark.asyncio
async def test_progress_feedback_callback():
    """Test progress callback is invoked during query"""
    from io import StringIO

    from rich.console import Console

    from ai_assist.tui_interactive import query_with_feedback

    output = StringIO()
    console = Console(file=output, force_terminal=False)

    # Create a mock agent with streaming
    mock_agent = AsyncMock(spec=AiAssistAgent)

    async def mock_streaming(prompt=None, messages=None, progress_callback=None, cancel_event=None):
        # Simulate calling the callback
        if progress_callback:
            progress_callback("thinking", 0, 10, None)
            progress_callback("calling_claude", 1, 10, None)
            progress_callback("executing_tool", 1, 10, "test_tool")
            progress_callback("complete", 1, 10, None)
        yield "Test response"
        yield {"type": "done", "turns": 1}

    mock_agent.query_streaming = mock_streaming
    mock_agent.get_last_kg_saved_count = MagicMock(return_value=0)
    mock_agent.clear_tool_calls = MagicMock()
    mock_agent.kg_save_enabled = True

    result = await query_with_feedback(mock_agent, "test prompt", console)

    assert result == "Test response"
    # Check that response was output
    output_text = output.getvalue()
    assert "Test response" in output_text


@pytest.mark.asyncio
async def test_feedback_with_tool_calls(mock_state_manager):
    """Test feedback shows tool calls"""
    # Create mock agent with streaming and tool calls
    agent = AsyncMock(spec=AiAssistAgent)

    async def mock_streaming_with_tools(prompt=None, messages=None, progress_callback=None, cancel_event=None):
        if progress_callback:
            progress_callback("thinking", 0, 10, None)
            progress_callback("calling_claude", 1, 10, None)
            progress_callback("executing_tool", 1, 10, "mcp__dci__search_dci_jobs")
            progress_callback("calling_claude", 2, 10, None)
            progress_callback("complete", 2, 10, None)
        # Simulate tool use and response
        yield {"type": "tool_use", "name": "mcp__dci__search_dci_jobs", "id": "1", "input": {}}
        yield "Found 5 failed jobs"
        yield {"type": "done", "turns": 2}

    agent.query_streaming = mock_streaming_with_tools
    agent.get_last_kg_saved_count = MagicMock(return_value=0)
    agent.clear_tool_calls = MagicMock()
    agent.kg_save_enabled = True
    agent.skills_manager = MagicMock()
    agent.skills_manager.installed_skills = []
    agent.filesystem_tools = MagicMock()
    agent.filesystem_tools.confirmation_callback = None
    agent.get_truncation_limits = MagicMock(
        return_value={
            "max_message_chars": 40000,
            "max_total_chars": 480000,
            "context_window_tokens": 200000,
            "usable_tokens": 150000,
        }
    )
    agent.renderer = PlainRenderer()
    agent.on_inner_execution = agent.renderer.on_inner_execution

    with patch("ai_assist.tui_interactive.PromptSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session.prompt_async = AsyncMock(side_effect=["Find failed jobs", "/exit"])
        mock_session_class.return_value = mock_session

        await tui_interactive_mode(agent, mock_state_manager)

        # Verify conversation was tracked
        mock_state_manager.save_conversation_context.assert_called()
        call_args = mock_state_manager.save_conversation_context.call_args
        messages = call_args[0][1]["messages"]
        assert len(messages) == 1
        assert messages[0]["user"] == "Find failed jobs"
        assert messages[0]["assistant"] == "Found 5 failed jobs"


@pytest.mark.asyncio
async def test_query_streaming_cancel_event():
    """Setting cancel_event mid-stream yields cancelled and stops"""
    import threading

    from ai_assist.agent import AiAssistAgent
    from ai_assist.config import AiAssistConfig

    mock_config = MagicMock(spec=AiAssistConfig)
    mock_config.use_vertex = False
    mock_config.anthropic_api_key = "test-key"
    mock_config.model = "claude-3-5-sonnet-20241022"
    mock_config.mcp_servers = {}
    mock_config.allow_skill_script_execution = False
    mock_config.allowed_commands = ["grep", "find", "wc", "sort", "head", "tail", "ls", "cat", "diff", "file", "stat"]
    mock_config.allowed_paths = ["~/.ai-assist", "/tmp/ai-assist"]
    mock_config.confirm_tools = ["internal__create_directory"]
    mock_config.message_limit_pct = 5.0
    mock_config.total_messages_pct = 60.0
    mock_config.reserve_pct = 25.0

    agent = AiAssistAgent(mock_config)

    cancel_event = threading.Event()

    # Mock the Anthropic streaming
    mock_stream = MagicMock()
    mock_text_delta = MagicMock()
    mock_text_delta.type = "content_block_delta"
    mock_text_delta.delta = MagicMock()
    mock_text_delta.delta.text = "partial"
    del mock_text_delta.delta.partial_json  # no partial_json attr

    # Simulate iteration: yield one text chunk, then set cancel
    def stream_iter(self_):
        yield mock_text_delta
        cancel_event.set()
        # Yield another text delta that should not be processed
        yield mock_text_delta

    mock_stream.__iter__ = stream_iter
    mock_stream.__enter__ = lambda s: s
    mock_stream.__exit__ = lambda s, *a: None

    mock_final = MagicMock()
    mock_final.content = []  # No tool calls
    mock_final.stop_reason = "end_turn"
    mock_stream.get_final_message.return_value = mock_final

    with patch.object(agent, "anthropic") as mock_anthropic:
        mock_anthropic.messages.stream.return_value = mock_stream

        chunks = []
        async for chunk in agent.query_streaming(prompt="test", cancel_event=cancel_event):
            chunks.append(chunk)

    # Should have text chunk and then cancelled signal
    assert any(isinstance(c, dict) and c.get("type") == "cancelled" for c in chunks)


@pytest.mark.asyncio
async def test_query_with_feedback_cancellation():
    """query_with_feedback handles cancelled chunk gracefully"""
    from io import StringIO

    from rich.console import Console

    from ai_assist.tui_interactive import query_with_feedback

    output = StringIO()
    console = Console(file=output, force_terminal=False)

    mock_agent = AsyncMock(spec=AiAssistAgent)

    async def mock_streaming(prompt=None, messages=None, progress_callback=None, cancel_event=None):
        yield "Partial response"
        yield {"type": "cancelled"}

    mock_agent.query_streaming = mock_streaming
    mock_agent.get_last_kg_saved_count = MagicMock(return_value=0)
    mock_agent.clear_tool_calls = MagicMock()

    with patch("ai_assist.tui_interactive.EscapeWatcher"):
        result = await query_with_feedback(mock_agent, "test prompt", console)

    assert result == "Partial response"
    output_text = output.getvalue()
    assert "cancelled" in output_text.lower()
