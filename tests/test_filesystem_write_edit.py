"""Tests for write_file and edit_file filesystem tools"""

import pytest

from ai_assist.config import AiAssistConfig
from ai_assist.filesystem_tools import FilesystemTools


@pytest.fixture
def tools(tmp_path):
    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_paths=[str(tmp_path)],
    )
    return FilesystemTools(config, load_user_config=False)


# --- write_file ---


@pytest.mark.asyncio
async def test_write_file_creates_new_file(tools, tmp_path):
    target = tmp_path / "hello.txt"
    result = await tools.execute_tool("write_file", {"path": str(target), "content": "hello world"})

    assert "File written" in result
    assert target.read_text() == "hello world"


@pytest.mark.asyncio
async def test_write_file_overwrites_existing(tools, tmp_path):
    target = tmp_path / "existing.txt"
    target.write_text("old content")

    result = await tools.execute_tool("write_file", {"path": str(target), "content": "new content"})

    assert "File written" in result
    assert target.read_text() == "new content"


@pytest.mark.asyncio
async def test_write_file_creates_parent_dirs(tools, tmp_path):
    target = tmp_path / "sub" / "dir" / "file.txt"
    result = await tools.execute_tool("write_file", {"path": str(target), "content": "nested"})

    assert "File written" in result
    assert target.read_text() == "nested"


@pytest.mark.asyncio
async def test_write_file_missing_path(tools):
    result = await tools.execute_tool("write_file", {"content": "hello"})
    assert "Error" in result


@pytest.mark.asyncio
async def test_write_file_missing_content(tools, tmp_path):
    target = tmp_path / "file.txt"
    result = await tools.execute_tool("write_file", {"path": str(target)})
    assert "Error" in result


@pytest.mark.asyncio
async def test_write_file_path_validation(tmp_path):
    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_paths=[str(tmp_path / "allowed")],
    )
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool("write_file", {"path": str(tmp_path / "forbidden" / "file.txt"), "content": "x"})
    assert "Error" in result


@pytest.mark.asyncio
async def test_write_file_expands_tilde(tools, tmp_path):
    target = tmp_path / "tilde_test.txt"
    result = await tools.execute_tool("write_file", {"path": str(target), "content": "ok"})
    assert "File written" in result


@pytest.mark.asyncio
async def test_write_file_empty_content(tools, tmp_path):
    target = tmp_path / "empty.txt"
    result = await tools.execute_tool("write_file", {"path": str(target), "content": ""})
    assert "File written" in result
    assert target.read_text() == ""


@pytest.mark.asyncio
async def test_write_file_confirmation_rejected(tools, tmp_path):
    tools.confirm_tools = {"internal__write_file"}

    async def reject(desc: str) -> bool:
        return False

    tools.confirmation_callback = reject

    target = tmp_path / "rejected.txt"
    result = await tools.execute_tool("write_file", {"path": str(target), "content": "x"})
    assert "rejected" in result.lower()
    assert not target.exists()


# --- edit_file ---


@pytest.mark.asyncio
async def test_edit_file_replaces_unique_string(tools, tmp_path):
    target = tmp_path / "code.py"
    target.write_text("def hello():\n    return 'hello'\n")

    result = await tools.execute_tool(
        "edit_file",
        {"path": str(target), "old_string": "return 'hello'", "new_string": "return 'world'"},
    )

    assert "File edited" in result
    assert "return 'world'" in target.read_text()


@pytest.mark.asyncio
async def test_edit_file_multiline(tools, tmp_path):
    target = tmp_path / "config.yaml"
    target.write_text("key: value\nother: stuff\n")

    result = await tools.execute_tool(
        "edit_file",
        {
            "path": str(target),
            "old_string": "key: value\nother: stuff",
            "new_string": "key: new_value\nother: new_stuff",
        },
    )

    assert "File edited" in result
    content = target.read_text()
    assert "key: new_value" in content
    assert "other: new_stuff" in content


@pytest.mark.asyncio
async def test_edit_file_not_found(tools, tmp_path):
    target = tmp_path / "missing.txt"
    result = await tools.execute_tool(
        "edit_file",
        {"path": str(target), "old_string": "x", "new_string": "y"},
    )
    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_edit_file_old_string_missing(tools, tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("hello world")

    result = await tools.execute_tool(
        "edit_file",
        {"path": str(target), "old_string": "nonexistent", "new_string": "y"},
    )
    assert "not found" in result.lower()
    assert target.read_text() == "hello world"


@pytest.mark.asyncio
async def test_edit_file_duplicate_match(tools, tmp_path):
    target = tmp_path / "dup.txt"
    target.write_text("foo bar foo baz")

    result = await tools.execute_tool(
        "edit_file",
        {"path": str(target), "old_string": "foo", "new_string": "qux"},
    )
    assert "appears 2 times" in result
    assert target.read_text() == "foo bar foo baz"


@pytest.mark.asyncio
async def test_edit_file_identical_strings(tools, tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("content")

    result = await tools.execute_tool(
        "edit_file",
        {"path": str(target), "old_string": "content", "new_string": "content"},
    )
    assert "identical" in result.lower()


@pytest.mark.asyncio
async def test_edit_file_missing_params(tools):
    result = await tools.execute_tool("edit_file", {"path": "/tmp/x"})
    assert "Error" in result

    result = await tools.execute_tool("edit_file", {"old_string": "x", "new_string": "y"})
    assert "Error" in result


@pytest.mark.asyncio
async def test_edit_file_path_validation(tmp_path):
    config = AiAssistConfig(
        anthropic_api_key="test",
        allowed_paths=[str(tmp_path / "allowed")],
    )
    tools = FilesystemTools(config, load_user_config=False)

    result = await tools.execute_tool(
        "edit_file",
        {"path": str(tmp_path / "forbidden" / "file.txt"), "old_string": "x", "new_string": "y"},
    )
    assert "Error" in result


@pytest.mark.asyncio
async def test_edit_file_confirmation_rejected(tools, tmp_path):
    tools.confirm_tools = {"internal__edit_file"}

    async def reject(desc: str) -> bool:
        return False

    tools.confirmation_callback = reject

    target = tmp_path / "file.txt"
    target.write_text("original")

    result = await tools.execute_tool(
        "edit_file",
        {"path": str(target), "old_string": "original", "new_string": "changed"},
    )
    assert "rejected" in result.lower()
    assert target.read_text() == "original"


@pytest.mark.asyncio
async def test_edit_file_not_a_file(tools, tmp_path):
    target = tmp_path / "subdir"
    target.mkdir()

    result = await tools.execute_tool(
        "edit_file",
        {"path": str(target), "old_string": "x", "new_string": "y"},
    )
    assert "Not a file" in result


@pytest.mark.asyncio
async def test_edit_file_not_found_message(tools, tmp_path):
    target = tmp_path / "missing.txt"
    result = await tools.execute_tool(
        "edit_file",
        {"path": str(target), "old_string": "x", "new_string": "y"},
    )
    assert "File not found" in result
