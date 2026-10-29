"""Tests for __save_to_file parameter in tool execution."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig
from ai_assist.filesystem_tools import FilesystemTools


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


# --- Path traversal / symlink regression matrix for __save_to_file ---


def _agent_scoped_to(dirpath):
    config = AiAssistConfig(anthropic_api_key="test-key", allowed_paths=[str(dirpath)])
    agent = AiAssistAgent(config=config)
    session = AsyncMock()
    result = MagicMock()
    result.content = [MagicMock(text="PWNED")]
    session.call_tool.return_value = result
    agent.sessions = {"dci": session}
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize("relative", ["../escape.txt", "../../evil.txt", "./../escape2.txt", "sub/../../out.txt"])
async def test_save_to_file_traversal_rejected(tmp_path, relative):
    allowed = tmp_path / "dci"
    allowed.mkdir()
    agent = _agent_scoped_to(allowed)

    result = await agent._execute_tool("dci__q", {"__save_to_file": f"{allowed}/{relative}"})

    assert result.startswith("Error"), result
    assert not any(p.read_text() == "PWNED" for p in tmp_path.rglob("*") if p.is_file())


@pytest.mark.asyncio
@pytest.mark.parametrize("relative", ["ok.txt", "double//ok.txt", "sub/../ok2.txt"])
async def test_save_to_file_inside_allowed_succeeds(tmp_path, relative):
    allowed = tmp_path / "dci"
    allowed.mkdir()
    agent = _agent_scoped_to(allowed)

    result = await agent._execute_tool("dci__q", {"__save_to_file": f"{allowed}/{relative}"})

    assert "Result saved to" in result
    assert (allowed / relative).resolve().read_text() == "PWNED"


@pytest.mark.asyncio
async def test_save_to_file_symlink_escaping_allowed_dir_rejected(tmp_path):
    allowed = tmp_path / "dci"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (allowed / "link").symlink_to(outside)
    agent = _agent_scoped_to(allowed)

    result = await agent._execute_tool("dci__q", {"__save_to_file": f"{allowed}/link/x.txt"})

    assert result.startswith("Error"), result
    assert not (outside / "x.txt").exists()


@pytest.mark.asyncio
async def test_save_to_file_homoglyph_dir_not_conflated_with_allowed(tmp_path):
    (tmp_path / "dci").mkdir()
    agent = _agent_scoped_to(tmp_path / "dci")

    # Cyrillic 'с' (U+0441) looks like Latin 'c' but is a different directory
    result = await agent._execute_tool("dci__q", {"__save_to_file": f"{tmp_path}/dсi/x.txt"})

    assert result.startswith("Error"), result
    assert not (tmp_path / "dсi").exists()


@pytest.mark.asyncio
async def test_save_to_file_logs_original_and_resolved_path(tmp_path, caplog):
    import logging

    allowed = tmp_path / "dci"
    allowed.mkdir()
    agent = _agent_scoped_to(allowed)

    with caplog.at_level(logging.INFO, logger="ai_assist.tool_result_router"):
        await agent._execute_tool("dci__q", {"__save_to_file": f"{allowed}/sub/../saved.txt"})

    saved_logs = [r.getMessage() for r in caplog.records if "Saved tool result" in r.getMessage()]
    assert saved_logs
    assert "sub/../saved.txt" in saved_logs[0]  # original, as the model gave it
    assert str((allowed / "saved.txt").resolve()) in saved_logs[0]  # resolved


# --- Empty allowlist must not let writes escape silently (reads stay open) ---


@pytest.mark.asyncio
async def test_empty_allowlist_write_outside_denied_non_interactive(tmp_path_factory, monkeypatch):
    workspace = tmp_path_factory.mktemp("workspace")
    monkeypatch.chdir(workspace)  # workspace = cwd; target is elsewhere
    config = AiAssistConfig(anthropic_api_key="test-key", allowed_paths=[])
    agent = AiAssistAgent(config=config)
    session = AsyncMock()
    res = MagicMock()
    res.content = [MagicMock(text="PWNED")]
    session.call_tool.return_value = res
    agent.sessions = {"dci": session}
    target = tmp_path_factory.mktemp("outside") / "x.txt"

    result = await agent._execute_tool("dci__q", {"__save_to_file": str(target)})

    assert result.startswith("Error"), result
    assert not target.exists()


@pytest.mark.asyncio
async def test_empty_allowlist_write_outside_prompts_interactive(tmp_path):
    config = AiAssistConfig(anthropic_api_key="test", allowed_paths=[])
    tools = FilesystemTools(config, load_user_config=False)
    prompts = []

    async def approve(desc):
        prompts.append(desc)
        return True

    tools.path_confirmation_callback = approve
    target = tmp_path / "anywhere" / "x.txt"

    await tools.execute_tool("write_file", {"path": str(target), "content": "x"})

    assert prompts, "expected a path confirmation prompt"
    assert target.read_text() == "x"


@pytest.mark.asyncio
async def test_empty_allowlist_read_still_unrestricted(tmp_path):
    f = tmp_path / "anywhere.txt"
    f.write_text("accessible")
    tools = FilesystemTools(AiAssistConfig(anthropic_api_key="test", allowed_paths=[]), load_user_config=False)

    result = await tools.execute_tool("read_file", {"path": str(f)})

    assert "accessible" in result


@pytest.mark.asyncio
async def test_empty_allowlist_write_to_workspace_ok(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tools = FilesystemTools(AiAssistConfig(anthropic_api_key="test", allowed_paths=[]), load_user_config=False)

    result = await tools.execute_tool("write_file", {"path": str(tmp_path / "note.txt"), "content": "x"})

    assert "File written" in result
    assert (tmp_path / "note.txt").read_text() == "x"
