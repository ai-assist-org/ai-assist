"""Tests for agent MCP prompt execution"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig


@pytest.fixture
def mock_config():
    """Create mock config"""
    config = MagicMock(spec=AiAssistConfig)
    config.use_vertex = False
    config.anthropic_api_key = "test-key"
    config.model = "claude-3-5-sonnet-20241022"
    config.mcp_servers = {}
    config.allow_skill_script_execution = False
    config.allowed_commands = ["grep", "find", "wc", "sort", "head", "tail", "ls", "cat", "diff", "file", "stat"]
    config.allowed_paths = ["~/.ai-assist", "/tmp/ai-assist"]
    config.confirm_tools = ["internal__create_directory"]
    return config


@pytest.fixture
def agent(mock_config):
    """Create agent instance"""
    return AiAssistAgent(mock_config)


@pytest.mark.asyncio
async def test_execute_mcp_prompt_server_not_found(agent):
    """Test error when server not connected"""
    with pytest.raises(ValueError, match="MCP server 'nonexistent' not connected"):
        await agent.execute_mcp_prompt("nonexistent", "rca", {})


@pytest.mark.asyncio
async def test_execute_mcp_prompt_server_no_prompts(agent):
    """Test error when server has no prompts"""
    # Add a mock session but no prompts
    agent.sessions["dci"] = MagicMock()

    with pytest.raises(ValueError, match="Server 'dci' has no prompts"):
        await agent.execute_mcp_prompt("dci", "rca", {})


@pytest.mark.asyncio
async def test_execute_mcp_prompt_prompt_not_found(agent):
    """Test error when prompt not found in server"""
    # Add mock session and prompts
    agent.sessions["dci"] = MagicMock()
    agent.available_prompts["dci"] = {"other_prompt": MagicMock(name="other_prompt", description="Other prompt")}

    with pytest.raises(ValueError, match="Prompt 'rca' not found in server 'dci'"):
        await agent.execute_mcp_prompt("dci", "rca", {})


@pytest.mark.asyncio
async def test_execute_mcp_prompt_missing_required_argument(agent):
    """Test error when required argument is missing"""
    # Setup mock session and prompt with required argument
    mock_session = MagicMock()
    agent.sessions["dci"] = mock_session

    # Create mock argument
    mock_arg = MagicMock()
    mock_arg.name = "days"
    mock_arg.required = True
    mock_arg.description = "Number of days"

    # Create mock prompt
    mock_prompt = MagicMock()
    mock_prompt.name = "rca"
    mock_prompt.arguments = [mock_arg]

    agent.available_prompts["dci"] = {"rca": mock_prompt}

    with pytest.raises(ValueError, match="Required argument 'days' missing"):
        await agent.execute_mcp_prompt("dci", "rca", {})


@pytest.mark.asyncio
async def test_execute_mcp_prompt_success(agent):
    """Test successful prompt execution feeds prompt to query_streaming()"""
    # Setup mock session
    mock_session = MagicMock()
    agent.sessions["dci"] = mock_session

    # Create mock prompt (no required arguments)
    mock_prompt = MagicMock()
    mock_prompt.name = "rca"
    mock_prompt.arguments = []

    agent.available_prompts["dci"] = {"rca": mock_prompt}

    # Mock get_prompt response
    mock_message = MagicMock()
    mock_message.role = "user"
    mock_message.content = MagicMock()
    mock_message.content.text = "Test prompt instructions"

    mock_result = MagicMock()
    mock_result.messages = [mock_message]

    mock_session.get_prompt = AsyncMock(return_value=mock_result)

    # Mock query_streaming to yield text chunks and done
    async def mock_streaming(**kwargs):
        yield "Claude "
        yield "executed "
        yield "the prompt"
        yield {"type": "done", "turns": 1}

    agent.query_streaming = mock_streaming

    # Execute
    result = await agent.execute_mcp_prompt("dci", "rca", None)

    assert result == "Claude executed the prompt"
    mock_session.get_prompt.assert_called_once_with("rca", arguments=None)


@pytest.mark.asyncio
async def test_execute_mcp_prompt_with_arguments(agent):
    """Test prompt execution with arguments"""
    # Setup mock session
    mock_session = MagicMock()
    agent.sessions["dci"] = mock_session

    # Create mock argument (optional)
    mock_arg = MagicMock()
    mock_arg.name = "days"
    mock_arg.required = False
    mock_arg.description = "Number of days"

    # Create mock prompt
    mock_prompt = MagicMock()
    mock_prompt.name = "rca"
    mock_prompt.arguments = [mock_arg]

    agent.available_prompts["dci"] = {"rca": mock_prompt}

    # Mock get_prompt response
    mock_message = MagicMock()
    mock_message.role = "user"
    mock_message.content = MagicMock()
    mock_message.content.text = "RCA for 7 days"

    mock_result = MagicMock()
    mock_result.messages = [mock_message]

    mock_session.get_prompt = AsyncMock(return_value=mock_result)

    # Mock query_streaming
    async def mock_streaming(**kwargs):
        yield "Claude analyzed RCA for 7 days"
        yield {"type": "done", "turns": 1}

    agent.query_streaming = mock_streaming

    # Execute with arguments
    result = await agent.execute_mcp_prompt("dci", "rca", {"days": "7"})

    assert result == "Claude analyzed RCA for 7 days"
    mock_session.get_prompt.assert_called_once_with("rca", arguments={"days": "7"})


@pytest.mark.asyncio
async def test_execute_mcp_prompt_multiple_messages(agent):
    """Test prompt execution with multiple messages passes all to query_streaming()"""
    # Setup mock session
    mock_session = MagicMock()
    agent.sessions["dci"] = mock_session

    # Create mock prompt
    mock_prompt = MagicMock()
    mock_prompt.name = "rca"
    mock_prompt.arguments = []

    agent.available_prompts["dci"] = {"rca": mock_prompt}

    # Mock get_prompt response with multiple messages
    mock_message1 = MagicMock()
    mock_message1.role = "user"
    mock_message1.content = MagicMock()
    mock_message1.content.text = "First message"

    mock_message2 = MagicMock()
    mock_message2.role = "assistant"
    mock_message2.content = MagicMock()
    mock_message2.content.text = "Second message"

    mock_result = MagicMock()
    mock_result.messages = [mock_message1, mock_message2]

    mock_session.get_prompt = AsyncMock(return_value=mock_result)

    # Track what messages query_streaming receives
    received_messages = []

    async def mock_streaming(**kwargs):
        received_messages.extend(kwargs.get("messages", []))
        yield "Claude processed both messages"
        yield {"type": "done", "turns": 1}

    agent.query_streaming = mock_streaming

    # Execute
    result = await agent.execute_mcp_prompt("dci", "rca", None)

    assert result == "Claude processed both messages"
    # Verify query_streaming was called with both messages
    assert len(received_messages) == 2
    assert received_messages[0]["role"] == "user"
    assert received_messages[0]["content"] == "First message"
    assert received_messages[1]["role"] == "assistant"
    assert received_messages[1]["content"] == "Second message"


@pytest.mark.asyncio
async def test_execute_mcp_prompt_calls_callback(agent):
    """Test that on_inner_execution callback receives chunks during execution"""
    # Setup mock session
    mock_session = MagicMock()
    agent.sessions["dci"] = mock_session

    # Create mock prompt
    mock_prompt = MagicMock()
    mock_prompt.name = "rca"
    mock_prompt.arguments = []

    agent.available_prompts["dci"] = {"rca": mock_prompt}

    # Mock get_prompt response
    mock_message = MagicMock()
    mock_message.role = "user"
    mock_message.content = MagicMock()
    mock_message.content.text = "Analyze failures"

    mock_result = MagicMock()
    mock_result.messages = [mock_message]
    mock_session.get_prompt = AsyncMock(return_value=mock_result)

    # Mock query_streaming to yield text, tool_use, and done
    async def mock_streaming(**kwargs):
        yield "Hello "
        yield "world"
        yield {"type": "tool_use", "name": "mcp__dci__search", "id": "t1", "input": {"query": "test"}}
        yield "Result text"
        yield {"type": "done", "turns": 1}

    agent.query_streaming = mock_streaming

    # Set up callback to collect received chunks
    received_chunks = []
    agent.on_inner_execution = received_chunks.append

    # Execute
    result = await agent.execute_mcp_prompt("dci", "rca", None)

    assert result == "Hello worldResult text"
    # Verify callback received text chunks and tool_use notification
    assert "Hello " in received_chunks
    assert "world" in received_chunks
    assert "Result text" in received_chunks
    tool_chunks = [c for c in received_chunks if isinstance(c, dict) and c.get("type") == "tool_use"]
    assert len(tool_chunks) == 1
    assert tool_chunks[0]["name"] == "mcp__dci__search"


@pytest.mark.asyncio
async def test_execute_mcp_prompt_no_callback(agent):
    """Test that execute_mcp_prompt works fine without a callback set"""
    # Setup mock session
    mock_session = MagicMock()
    agent.sessions["dci"] = mock_session

    # Create mock prompt
    mock_prompt = MagicMock()
    mock_prompt.name = "rca"
    mock_prompt.arguments = []

    agent.available_prompts["dci"] = {"rca": mock_prompt}

    # Mock get_prompt response
    mock_message = MagicMock()
    mock_message.role = "user"
    mock_message.content = MagicMock()
    mock_message.content.text = "Test"

    mock_result = MagicMock()
    mock_result.messages = [mock_message]
    mock_session.get_prompt = AsyncMock(return_value=mock_result)

    async def mock_streaming(**kwargs):
        yield "response"
        yield {"type": "done", "turns": 1}

    agent.query_streaming = mock_streaming

    # Default callback is the renderer's on_inner_execution (not None)
    assert agent.on_inner_execution is not None

    # Should work without errors
    result = await agent.execute_mcp_prompt("dci", "rca", None)
    assert result == "response"


@pytest.mark.asyncio
async def test_execute_mcp_prompt_error_callback(agent):
    """Test that error chunks are forwarded to callback"""
    # Setup mock session
    mock_session = MagicMock()
    agent.sessions["dci"] = mock_session

    # Create mock prompt
    mock_prompt = MagicMock()
    mock_prompt.name = "rca"
    mock_prompt.arguments = []

    agent.available_prompts["dci"] = {"rca": mock_prompt}

    # Mock get_prompt response
    mock_message = MagicMock()
    mock_message.role = "user"
    mock_message.content = MagicMock()
    mock_message.content.text = "Test"

    mock_result = MagicMock()
    mock_result.messages = [mock_message]
    mock_session.get_prompt = AsyncMock(return_value=mock_result)

    async def mock_streaming(**kwargs):
        yield "partial"
        yield {"type": "error", "message": "Something went wrong"}

    agent.query_streaming = mock_streaming

    received_chunks = []
    agent.on_inner_execution = received_chunks.append

    result = await agent.execute_mcp_prompt("dci", "rca", None)

    assert result == "partial"
    error_chunks = [c for c in received_chunks if isinstance(c, dict) and c.get("type") == "error"]
    assert len(error_chunks) == 1
    assert error_chunks[0]["message"] == "Something went wrong"


@pytest.mark.asyncio
async def test_execute_mcp_prompt_forwards_cancel_event(agent):
    """Test that execute_mcp_prompt forwards the active cancel_event"""
    import threading

    mock_session = MagicMock()
    agent.sessions["dci"] = mock_session

    mock_prompt = MagicMock()
    mock_prompt.name = "rca"
    mock_prompt.arguments = []
    agent.available_prompts["dci"] = {"rca": mock_prompt}

    mock_message = MagicMock()
    mock_message.role = "user"
    mock_message.content = MagicMock()
    mock_message.content.text = "Test prompt"
    mock_result = MagicMock()
    mock_result.messages = [mock_message]
    mock_session.get_prompt = AsyncMock(return_value=mock_result)

    # Set a cancel_event on the agent (simulating outer query_streaming having set it)
    cancel_event = threading.Event()
    agent._cancel_event = cancel_event

    # Mock query_streaming to capture cancel_event and yield cancelled
    received_cancel_event = []

    async def mock_streaming(**kwargs):
        received_cancel_event.append(kwargs.get("cancel_event"))
        yield "partial"
        yield {"type": "cancelled"}

    agent.query_streaming = mock_streaming

    received_chunks = []
    agent.on_inner_execution = received_chunks.append

    result = await agent.execute_mcp_prompt("dci", "rca", None)

    # Verify cancel_event was forwarded
    assert received_cancel_event[0] is cancel_event
    assert result == "partial"
    cancelled_chunks = [c for c in received_chunks if isinstance(c, dict) and c.get("type") == "cancelled"]
    assert len(cancelled_chunks) == 1
