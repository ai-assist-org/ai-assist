"""Tests for __save_to_file parameter in tool execution."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig


@pytest.mark.asyncio
async def test_save_to_file_mcp_tool():
    """Test that __save_to_file saves MCP tool results to file."""
    # Create agent with mocked MCP session
    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"], allowed_paths=[tempfile.gettempdir()])
    agent = AiAssistAgent(config=config)

    # Mock MCP session
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.content = [MagicMock(text='{"hits": [{"id": "job1"}, {"id": "job2"}]}')]
    mock_session.call_tool.return_value = mock_result
    agent.sessions = {"test": mock_session}

    # Create temp file path
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "result.json"

        # Execute tool with __save_to_file
        result = await agent._execute_tool("test__some_tool", {"query": "test", "__save_to_file": str(output_file)})

        # Verify summary returned
        assert "Result saved to" in result
        assert str(output_file) in result
        assert "bytes" in result

        # Verify file was created and contains the full result
        assert output_file.exists()
        saved_data = json.loads(output_file.read_text())
        assert saved_data == {"hits": [{"id": "job1"}, {"id": "job2"}]}


@pytest.mark.asyncio
async def test_save_to_file_creates_parent_dirs():
    """Test that __save_to_file creates parent directories."""
    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"], allowed_paths=[tempfile.gettempdir()])
    agent = AiAssistAgent(config=config)

    # Mock MCP session
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.content = [MagicMock(text='{"data": "test"}')]
    mock_session.call_tool.return_value = mock_result
    agent.sessions = {"test": mock_session}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Use nested path that doesn't exist
        output_file = Path(tmpdir) / "deep" / "nested" / "path" / "result.json"

        result = await agent._execute_tool("test__some_tool", {"__save_to_file": str(output_file)})

        assert "Result saved to" in result
        assert output_file.exists()
        assert output_file.read_text() == '{"data": "test"}'


@pytest.mark.asyncio
async def test_save_to_file_large_result():
    """Test that __save_to_file handles large results efficiently."""
    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"], allowed_paths=[tempfile.gettempdir()])
    agent = AiAssistAgent(config=config)

    # Create large result (>20KB)
    large_result = json.dumps({"data": "x" * 30000})

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.content = [MagicMock(text=large_result)]
    mock_session.call_tool.return_value = mock_result
    agent.sessions = {"test": mock_session}

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "large.json"

        result = await agent._execute_tool("test__some_tool", {"__save_to_file": str(output_file)})

        # Verify summary is short
        assert len(result) < 200

        # Verify full result was saved
        saved_data = output_file.read_text()
        assert len(saved_data) > 20000
        assert json.loads(saved_data) == {"data": "x" * 30000}


@pytest.mark.asyncio
async def test_save_to_file_error_handling():
    """Test error handling when file save fails."""
    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"], allowed_paths=[tempfile.gettempdir()])
    agent = AiAssistAgent(config=config)

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.content = [MagicMock(text='{"data": "test"}')]
    mock_session.call_tool.return_value = mock_result
    agent.sessions = {"test": mock_session}

    # Use a path whose parent is a regular file, so the write fails
    blocker = tempfile.NamedTemporaryFile()
    invalid_path = f"{blocker.name}/file.json"

    result = await agent._execute_tool("test__some_tool", {"__save_to_file": invalid_path})

    assert "Error saving result" in result
    assert invalid_path in result


@pytest.mark.asyncio
async def test_without_save_to_file_returns_normal():
    """Test that tools without __save_to_file work normally."""
    config = AiAssistConfig(anthropic_api_key="test-key", working_dirs=["/tmp"], allowed_paths=[tempfile.gettempdir()])
    agent = AiAssistAgent(config=config)

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.content = [MagicMock(text='{"data": "test"}')]
    mock_session.call_tool.return_value = mock_result
    agent.sessions = {"test": mock_session}

    result = await agent._execute_tool("test__some_tool", {"query": "test"})

    # Should return the actual result, not a summary
    assert result == '{"data": "test"}'


def _make_agent(allowed_paths, result_text='{"data": "test"}'):
    config = AiAssistConfig(anthropic_api_key="test-key", allowed_paths=allowed_paths)
    agent = AiAssistAgent(config=config)
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.content = [MagicMock(text=result_text)]
    mock_session.call_tool.return_value = mock_result
    agent.sessions = {"test": mock_session}
    return agent, mock_session


@pytest.mark.asyncio
async def test_save_to_file_rejects_path_outside_allowed_paths(tmp_path):
    """__save_to_file must obey the path allowlist like internal__write_file."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    agent, session = _make_agent([str(allowed)])
    target = tmp_path / "outside" / "result.json"

    result = await agent._execute_tool("test__some_tool", {"__save_to_file": str(target)})

    assert "Error" in result
    assert not target.exists()
    session.call_tool.assert_not_called()


@pytest.mark.asyncio
async def test_save_to_file_rejected_by_path_confirmation(tmp_path):
    """A denied path-approval prompt blocks the write."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    agent, session = _make_agent([str(allowed)])
    agent.filesystem_tools.path_confirmation_callback = AsyncMock(return_value=False)
    target = tmp_path / "outside.json"

    result = await agent._execute_tool("test__some_tool", {"__save_to_file": str(target)})

    assert "rejected" in result
    assert not target.exists()
    session.call_tool.assert_not_called()


@pytest.mark.asyncio
async def test_save_to_file_allowed_path_succeeds(tmp_path):
    agent, _ = _make_agent([str(tmp_path)])
    target = tmp_path / "result.json"

    result = await agent._execute_tool("test__some_tool", {"__save_to_file": str(target)})

    assert "Result saved to" in result
    assert target.read_text() == '{"data": "test"}'


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["allowed_commands.json", "mcp_servers.yaml", ".env", "schedules.json"])
async def test_save_to_file_cannot_overwrite_protected_config(isolate_config_dir, name):
    """Even with the config dir allowed, protected config files cannot be written."""
    agent, session = _make_agent([str(isolate_config_dir)])
    target = isolate_config_dir / name

    result = await agent._execute_tool("test__some_tool", {"__save_to_file": str(target)})

    assert "protected" in result
    assert not target.exists()
    session.call_tool.assert_not_called()
