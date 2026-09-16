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

    # Mock background task support
    agent.background_task_tools = None
    agent._background_task_count = 0
    agent.available_tools = []

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
    agent.background_task_tools = None
    agent._background_task_count = 0
    agent.available_tools = []
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
    agent.background_task_tools = None
    agent._background_task_count = 0
    agent.available_tools = []
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
    agent.background_task_tools = None
    agent._background_task_count = 0
    agent.available_tools = []
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
    mock_config.use_custom_endpoint = False
    mock_config.anthropic_api_key = "test-key"
    mock_config.anthropic_base_url = None
    mock_config.model = "claude-3-5-sonnet-20241022"
    mock_config.synthesis_model = None
    mock_config.compaction_model = None
    mock_config.enable_prompt_caching = True
    mock_config.enable_mlflow = False
    mock_config.model_max_output_tokens = None
    mock_config.model_context_window = None
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


def _plain_console():
    """Build a Rich console writing to an in-memory buffer (no colors)"""
    from io import StringIO

    from rich.console import Console

    output = StringIO()
    return Console(file=output, force_terminal=False), output


# --- substitute_skill_args (pure helper) ---


def test_substitute_skill_args_full_arguments():
    from ai_assist.tui_interactive import substitute_skill_args

    result = substitute_skill_args("Run with $ARGUMENTS now", "alpha beta")
    assert result == "Run with alpha beta now"


def test_substitute_skill_args_positional_quoting():
    from ai_assist.tui_interactive import substitute_skill_args

    result = substitute_skill_args("first=$1 second=$2", 'one "two three"')
    assert result == "first=one second=two three"


def test_substitute_skill_args_missing_positions_empty():
    from ai_assist.tui_interactive import substitute_skill_args

    result = substitute_skill_args("a=$1 b=$2 c=$3", "only")
    assert result == "a=only b= c="


def test_substitute_skill_args_appends_when_no_placeholder():
    from ai_assist.tui_interactive import substitute_skill_args

    result = substitute_skill_args("Body text", "extra args")
    assert result == "Body text\n\nArguments: extra args"


def test_substitute_skill_args_no_placeholder_no_args():
    from ai_assist.tui_interactive import substitute_skill_args

    assert substitute_skill_args("Just body", "") == "Just body"


def test_substitute_skill_args_unbalanced_quotes_fallback():
    from ai_assist.tui_interactive import substitute_skill_args

    result = substitute_skill_args("x=$1 y=$2", 'broken "quote')
    assert result == 'x=broken y="quote'


# --- handle_skill_management_command + helpers ---


async def test_skill_command_ignores_other_commands(mock_agent):
    from ai_assist.tui_interactive import handle_skill_management_command

    console, _ = _plain_console()
    handled = await handle_skill_management_command("/status", mock_agent, console)
    assert handled is False


async def test_skill_command_empty_shows_usage(mock_agent):
    from ai_assist.tui_interactive import handle_skill_management_command

    console, output = _plain_console()
    handled = await handle_skill_management_command("/skill/", mock_agent, console)
    assert handled is True
    assert "Usage" in output.getvalue()


async def test_skill_install_without_source(mock_agent):
    from ai_assist.tui_interactive import handle_skill_management_command

    console, output = _plain_console()
    await handle_skill_management_command("/skill/install", mock_agent, console)
    assert "Usage" in output.getvalue()
    mock_agent.skills_manager.install_skill.assert_not_called()


async def test_skill_install_calls_manager(mock_agent):
    from ai_assist.tui_interactive import handle_skill_management_command

    console, output = _plain_console()
    mock_agent.skills_manager.install_skill = MagicMock(return_value="Installed pdf")
    await handle_skill_management_command("/skill/install owner/repo@main", mock_agent, console)
    mock_agent.skills_manager.install_skill.assert_called_once_with("owner/repo@main")
    assert "Installed pdf" in output.getvalue()


async def test_skill_install_reports_error(mock_agent):
    from ai_assist.tui_interactive import handle_skill_management_command

    console, output = _plain_console()
    mock_agent.skills_manager.install_skill = MagicMock(return_value="Error: bad source")
    await handle_skill_management_command("/skill/install bad", mock_agent, console)
    assert "Error: bad source" in output.getvalue()


async def test_skill_uninstall_calls_manager(mock_agent):
    from ai_assist.tui_interactive import handle_skill_management_command

    console, output = _plain_console()
    mock_agent.skills_manager.uninstall_skill = MagicMock(return_value="Removed pdf")
    await handle_skill_management_command("/skill/uninstall pdf", mock_agent, console)
    mock_agent.skills_manager.uninstall_skill.assert_called_once_with("pdf")
    assert "Removed pdf" in output.getvalue()


async def test_skill_uninstall_without_name(mock_agent):
    from ai_assist.tui_interactive import handle_skill_management_command

    console, output = _plain_console()
    await handle_skill_management_command("/skill/uninstall", mock_agent, console)
    assert "Usage" in output.getvalue()


async def test_skill_update_all_when_no_name(mock_agent):
    from ai_assist.tui_interactive import handle_skill_management_command

    console, output = _plain_console()
    skill = MagicMock()
    skill.name = "pdf"
    mock_agent.skills_manager.installed_skills = [skill]
    mock_agent.skills_manager.update_skill = MagicMock(return_value="Updated pdf")
    await handle_skill_management_command("/skill/update", mock_agent, console)
    mock_agent.skills_manager.update_skill.assert_called_once_with("pdf")
    assert "Updated pdf" in output.getvalue()


async def test_skill_update_no_skills(mock_agent):
    from ai_assist.tui_interactive import handle_skill_management_command

    console, output = _plain_console()
    mock_agent.skills_manager.installed_skills = []
    await handle_skill_management_command("/skill/update", mock_agent, console)
    assert "No skills installed" in output.getvalue()


async def test_skill_list_calls_manager(mock_agent):
    from ai_assist.tui_interactive import handle_skill_management_command

    console, output = _plain_console()
    mock_agent.skills_manager.list_installed = MagicMock(return_value="pdf, csv")
    await handle_skill_management_command("/skill/list", mock_agent, console)
    assert "pdf, csv" in output.getvalue()


async def test_skill_search_calls_loader(mock_agent):
    from ai_assist.tui_interactive import handle_skill_management_command

    console, output = _plain_console()
    loader = MagicMock()
    loader.search_clawhub = MagicMock(return_value="clawhub hit")
    loader.search_skills_sh = MagicMock(return_value="skills.sh hit")
    mock_agent.skills_manager.skills_loader = loader
    await handle_skill_management_command("/skill/search pdf", mock_agent, console)
    text = output.getvalue()
    assert "clawhub hit" in text
    assert "skills.sh hit" in text


async def test_skill_search_without_query(mock_agent):
    from ai_assist.tui_interactive import handle_skill_management_command

    console, output = _plain_console()
    await handle_skill_management_command("/skill/search", mock_agent, console)
    assert "Usage" in output.getvalue()


async def test_skill_add_env_saves(mock_agent):
    from ai_assist.tui_interactive import handle_skill_management_command

    console, output = _plain_console()
    with patch("ai_assist.script_execution_tools.ScriptExecutionTools.save_skill_env") as save:
        await handle_skill_management_command("/skill/add_env gog API_KEY", mock_agent, console)
    save.assert_called_once_with("gog", "API_KEY")
    assert "Allowed API_KEY" in output.getvalue()


async def test_skill_add_env_missing_args(mock_agent):
    from ai_assist.tui_interactive import handle_skill_management_command

    console, output = _plain_console()
    await handle_skill_management_command("/skill/add_env gog", mock_agent, console)
    assert "Usage" in output.getvalue()


async def test_skill_remove_env_removed(mock_agent):
    from ai_assist.tui_interactive import handle_skill_management_command

    console, output = _plain_console()
    with patch("ai_assist.script_execution_tools.ScriptExecutionTools.remove_skill_env", return_value=True):
        await handle_skill_management_command("/skill/remove_env gog API_KEY", mock_agent, console)
    assert "Removed API_KEY" in output.getvalue()


async def test_skill_remove_env_not_configured(mock_agent):
    from ai_assist.tui_interactive import handle_skill_management_command

    console, output = _plain_console()
    with patch("ai_assist.script_execution_tools.ScriptExecutionTools.remove_skill_env", return_value=False):
        await handle_skill_management_command("/skill/remove_env gog API_KEY", mock_agent, console)
    assert "was not configured" in output.getvalue()


async def test_skill_list_env_empty(mock_agent):
    from ai_assist.tui_interactive import handle_skill_management_command

    console, output = _plain_console()
    with patch("ai_assist.script_execution_tools.ScriptExecutionTools.list_skill_env", return_value={}):
        await handle_skill_management_command("/skill/list_env", mock_agent, console)
    assert "No environment variables" in output.getvalue()


async def test_skill_list_env_with_config(mock_agent):
    from ai_assist.tui_interactive import handle_skill_management_command

    console, output = _plain_console()
    with patch(
        "ai_assist.script_execution_tools.ScriptExecutionTools.list_skill_env",
        return_value={"gog": ["API_KEY", "TOKEN"]},
    ):
        await handle_skill_management_command("/skill/list_env", mock_agent, console)
    text = output.getvalue()
    assert "gog" in text
    assert "API_KEY" in text


async def test_skill_unknown_subcommand(mock_agent):
    from ai_assist.tui_interactive import handle_skill_management_command

    console, output = _plain_console()
    await handle_skill_management_command("/skill/bogus", mock_agent, console)
    assert "Unknown skill command" in output.getvalue()


# --- handle_plugin_management_command + helpers ---


async def test_plugin_command_ignores_other_commands(mock_agent):
    from ai_assist.tui_interactive import handle_plugin_management_command

    console, _ = _plain_console()
    handled = await handle_plugin_management_command("/status", mock_agent, console)
    assert handled is False


async def test_plugin_install_without_source(mock_agent):
    from ai_assist.tui_interactive import handle_plugin_management_command

    console, output = _plain_console()
    mock_agent.plugins_manager = MagicMock()
    await handle_plugin_management_command("/plugin/install", mock_agent, console)
    assert "Usage" in output.getvalue()
    mock_agent.plugins_manager.install_plugin.assert_not_called()


async def test_plugin_install_no_servers(mock_agent):
    from ai_assist.tui_interactive import handle_plugin_management_command

    console, output = _plain_console()
    mock_agent.plugins_manager = MagicMock()
    mock_agent.plugins_manager.install_plugin = MagicMock(return_value=("Installed my-plugin", []))
    await handle_plugin_management_command("/plugin/install owner/repo@main", mock_agent, console)
    mock_agent.plugins_manager.install_plugin.assert_called_once_with("owner/repo@main")
    assert "Installed my-plugin" in output.getvalue()


async def test_plugin_install_reports_error(mock_agent):
    from ai_assist.tui_interactive import handle_plugin_management_command

    console, output = _plain_console()
    mock_agent.plugins_manager = MagicMock()
    mock_agent.plugins_manager.install_plugin = MagicMock(return_value=("Error: nope", []))
    await handle_plugin_management_command("/plugin/install bad", mock_agent, console)
    assert "Error: nope" in output.getvalue()


async def test_plugin_uninstall_disconnects_servers(mock_agent):
    from ai_assist.tui_interactive import handle_plugin_management_command

    console, output = _plain_console()
    mock_agent.plugins_manager = MagicMock()
    mock_agent.plugins_manager.uninstall_plugin = MagicMock(return_value=("Removed my-plugin", ["srv1"]))
    mock_agent.config = MagicMock()
    mock_agent.config.mcp_servers = {"srv1": MagicMock()}
    await handle_plugin_management_command("/plugin/uninstall my-plugin", mock_agent, console)
    mock_agent._disconnect_server.assert_called_once_with("srv1")
    assert "Removed my-plugin" in output.getvalue()


async def test_plugin_uninstall_without_name(mock_agent):
    from ai_assist.tui_interactive import handle_plugin_management_command

    console, output = _plain_console()
    mock_agent.plugins_manager = MagicMock()
    await handle_plugin_management_command("/plugin/uninstall", mock_agent, console)
    assert "Usage" in output.getvalue()


async def test_plugin_update_no_plugins(mock_agent):
    from ai_assist.tui_interactive import handle_plugin_management_command

    console, output = _plain_console()
    mock_agent.plugins_manager = MagicMock()
    mock_agent.plugins_manager.installed_plugins = []
    await handle_plugin_management_command("/plugin/update", mock_agent, console)
    assert "No plugins installed" in output.getvalue()


async def test_plugin_list_calls_manager(mock_agent):
    from ai_assist.tui_interactive import handle_plugin_management_command

    console, output = _plain_console()
    mock_agent.plugins_manager = MagicMock()
    mock_agent.plugins_manager.list_installed = MagicMock(return_value="plugin-a")
    await handle_plugin_management_command("/plugin/list", mock_agent, console)
    assert "plugin-a" in output.getvalue()


async def test_plugin_search_calls_manager(mock_agent):
    from ai_assist.tui_interactive import handle_plugin_management_command

    console, output = _plain_console()
    mock_agent.plugins_manager = MagicMock()
    mock_agent.plugins_manager.search = MagicMock(return_value="search results")
    await handle_plugin_management_command("/plugin/search foo", mock_agent, console)
    mock_agent.plugins_manager.search.assert_called_once_with("foo")
    assert "search results" in output.getvalue()


async def test_plugin_search_without_query(mock_agent):
    from ai_assist.tui_interactive import handle_plugin_management_command

    console, output = _plain_console()
    mock_agent.plugins_manager = MagicMock()
    await handle_plugin_management_command("/plugin/search", mock_agent, console)
    assert "Usage" in output.getvalue()


async def test_plugin_marketplace_list(mock_agent):
    from ai_assist.tui_interactive import handle_plugin_management_command

    console, output = _plain_console()
    mock_agent.plugins_manager = MagicMock()
    mock_agent.plugins_manager.list_marketplaces = MagicMock(return_value="market-a")
    await handle_plugin_management_command("/plugin/marketplace list", mock_agent, console)
    assert "market-a" in output.getvalue()


async def test_plugin_marketplace_add(mock_agent):
    from ai_assist.tui_interactive import handle_plugin_management_command

    console, output = _plain_console()
    mock_agent.plugins_manager = MagicMock()
    mock_agent.plugins_manager.add_marketplace = MagicMock(return_value="Added market")
    await handle_plugin_management_command("/plugin/marketplace add owner/repo@main", mock_agent, console)
    mock_agent.plugins_manager.add_marketplace.assert_called_once_with("owner/repo@main", None)
    assert "Added market" in output.getvalue()


async def test_plugin_marketplace_without_action(mock_agent):
    from ai_assist.tui_interactive import handle_plugin_management_command

    console, output = _plain_console()
    mock_agent.plugins_manager = MagicMock()
    await handle_plugin_management_command("/plugin/marketplace", mock_agent, console)
    assert "Usage" in output.getvalue()


async def test_plugin_unknown_subcommand(mock_agent):
    from ai_assist.tui_interactive import handle_plugin_management_command

    console, output = _plain_console()
    mock_agent.plugins_manager = MagicMock()
    await handle_plugin_management_command("/plugin/bogus", mock_agent, console)
    assert "Unknown plugin command" in output.getvalue()


# --- handle_prompt_command ---


async def test_prompt_command_not_a_prompt(mock_agent):
    from ai_assist.tui_interactive import handle_prompt_command

    console, _ = _plain_console()
    handled = await handle_prompt_command("/status", mock_agent, [], console, MagicMock())
    assert handled is False


async def test_prompt_command_unknown_server(mock_agent):
    from ai_assist.tui_interactive import handle_prompt_command

    console, output = _plain_console()
    mock_agent.sessions = {}
    mock_agent.available_prompts = {}
    handled = await handle_prompt_command("/dci/rca", mock_agent, [], console, MagicMock())
    assert handled is True
    assert "Unknown MCP server" in output.getvalue()


async def test_prompt_command_server_no_prompts(mock_agent):
    from ai_assist.tui_interactive import handle_prompt_command

    console, output = _plain_console()
    mock_agent.sessions = {"dci": MagicMock()}
    mock_agent.available_prompts = {}
    handled = await handle_prompt_command("/dci/rca", mock_agent, [], console, MagicMock())
    assert handled is True
    assert "has no prompts" in output.getvalue()


async def test_prompt_command_unknown_prompt(mock_agent):
    from ai_assist.tui_interactive import handle_prompt_command

    console, output = _plain_console()
    mock_agent.sessions = {"dci": MagicMock()}
    mock_agent.available_prompts = {"dci": {"weekly": MagicMock()}}
    handled = await handle_prompt_command("/dci/rca", mock_agent, [], console, MagicMock())
    assert handled is True
    assert "Unknown prompt" in output.getvalue()


async def test_prompt_command_executes_no_arguments(mock_agent):
    from ai_assist.tui_interactive import handle_prompt_command

    console, output = _plain_console()

    prompt_def = MagicMock()
    prompt_def.arguments = None

    msg = MagicMock()
    msg.role = "user"
    msg.content = MagicMock()
    msg.content.text = "expert context here"
    result = MagicMock()
    result.messages = [msg]

    session = MagicMock()
    session.get_prompt = AsyncMock(return_value=result)

    mock_agent.sessions = {"dci": session}
    mock_agent.available_prompts = {"dci": {"rca": prompt_def}}

    history: list = []
    handled = await handle_prompt_command("/dci/rca", mock_agent, history, console, MagicMock())
    assert handled is True
    session.get_prompt.assert_awaited_once_with("rca", arguments=None)
    assert len(history) == 1
    assert "Injected prompt" in output.getvalue()


# --- handle_cost_command ---


async def test_cost_command_error_string():
    from ai_assist.tui_interactive import handle_cost_command

    console, output = _plain_console()
    with patch("ai_assist.eval.compute_cost_summary", return_value="Invalid period '5x'."):
        await handle_cost_command(console, "5x")
    assert "Invalid period" in output.getvalue()


async def test_cost_command_summary_table():
    from ai_assist.eval import CostSummary
    from ai_assist.tui_interactive import handle_cost_command

    console, output = _plain_console()
    summary = CostSummary(
        label="last 7 days",
        query_count=3,
        total_cost=1.2345,
        avg_cost=0.4115,
        total_input_tokens=1000,
        total_output_tokens=500,
        cost_by_model={"claude-sonnet-4-6": 1.2345},
        queries_by_model={"claude-sonnet-4-6": 3},
        cost_per_day=0.1764,
        num_days=7,
        cost_by_script={"report.awl": 0.5},
        queries_by_script={"report.awl": 1},
    )
    with patch("ai_assist.eval.compute_cost_summary", return_value=summary):
        await handle_cost_command(console, "7d")
    text = output.getvalue()
    assert "Cost Summary" in text
    assert "Cost by Model" in text
    assert "Cost by Script" in text


# --- handle_kg_viz_command ---


async def test_kg_viz_command_no_kg():
    from ai_assist.tui_interactive import handle_kg_viz_command

    console, output = _plain_console()
    await handle_kg_viz_command(None, console)
    assert "not available" in output.getvalue()


async def test_kg_viz_command_opens():
    from ai_assist.tui_interactive import handle_kg_viz_command

    console, output = _plain_console()
    with patch("ai_assist.kg_visualization.open_kg_visualization", return_value="/tmp/kg.html"):
        await handle_kg_viz_command(MagicMock(), console)
    text = output.getvalue()
    assert "visualization opened" in text
    assert "/tmp/kg.html" in text


async def test_prompt_command_cancels_on_missing_required_arg(mock_agent):
    from ai_assist.tui_interactive import handle_prompt_command

    console, output = _plain_console()

    arg = MagicMock()
    arg.name = "cluster"
    arg.required = True
    prompt_def = MagicMock()
    prompt_def.arguments = [arg]

    mock_agent.sessions = {"dci": MagicMock()}
    mock_agent.available_prompts = {"dci": {"rca": prompt_def}}

    arg_session = MagicMock()
    arg_session.prompt_async = AsyncMock(return_value="")
    with patch("prompt_toolkit.PromptSession", return_value=arg_session):
        handled = await handle_prompt_command("/dci/rca", mock_agent, [], console, MagicMock())

    assert handled is True
    assert "is required" in output.getvalue()


async def test_prompt_command_execution_error(mock_agent):
    from ai_assist.tui_interactive import handle_prompt_command

    console, output = _plain_console()

    prompt_def = MagicMock()
    prompt_def.arguments = None
    session = MagicMock()
    session.get_prompt = AsyncMock(side_effect=RuntimeError("boom"))

    mock_agent.sessions = {"dci": session}
    mock_agent.available_prompts = {"dci": {"rca": prompt_def}}

    handled = await handle_prompt_command("/dci/rca", mock_agent, [], console, MagicMock())
    assert handled is True
    assert "Error executing prompt" in output.getvalue()


async def test_help_command_lists_user_skills(mock_agent):
    from ai_assist.tui_interactive import handle_help_command

    console, output = _plain_console()

    skill = MagicMock()
    skill.metadata.user_invocable = True
    skill.metadata.argument_hint = "<name>"
    skill.metadata.description = "A demo skill"
    mock_agent.skills_manager.loaded_skills = {"demo": skill}

    await handle_help_command(console, mock_agent)
    text = output.getvalue()
    assert "Interactive Mode Help" in text
    assert "demo" in text
