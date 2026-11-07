"""Answering "always allow" must save an allowlist entry that really permits the invocation."""

import pytest

from ai_assist.config import AiAssistConfig
from ai_assist.filesystem_tools import FilesystemTools


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "hello.py").write_text("print('ok')\n")
    return tmp_path


def _tools(workdir, commands=("python3",)):
    config = AiAssistConfig(anthropic_api_key="test", allowed_commands=list(commands), allowed_paths=[str(workdir)])
    return FilesystemTools(config, load_user_config=False)


async def _question_and_entry(tools, command):
    """Run a command interactively; return the question asked and the entry "always" would save."""
    asked = []

    async def callback(question):
        asked.append(question)
        return True

    tools.confirmation_callback = callback
    await tools.execute_tool("execute_command", {"command": command})
    assert asked, "no question was asked"
    return asked[0], tools.allowlist_entry_for(asked[0])


@pytest.mark.asyncio
async def test_always_on_a_script_saves_the_interpreter_and_resolved_path(workdir):
    tools = _tools(workdir)
    _, entry = await _question_and_entry(tools, "python3 tools/hello.py --flag")
    assert entry == f"python3 {workdir / 'tools' / 'hello.py'}"


@pytest.mark.asyncio
async def test_saved_script_entry_runs_it_non_interactively(workdir):
    tools = _tools(workdir)
    _, entry = await _question_and_entry(tools, "python3 tools/hello.py")

    later = _tools(workdir, commands=("python3", entry))  # next run, nobody to ask
    result = await later.execute_tool("execute_command", {"command": f"python3 {workdir}/tools/hello.py"})
    assert "Exit code: 0" in result, result
    assert "ok" in result


@pytest.mark.asyncio
async def test_always_on_a_module_saves_the_module_invocation(workdir):
    tools = _tools(workdir)
    _, entry = await _question_and_entry(tools, "python3 -m json.tool --help")
    assert entry == "python3 -m json.tool"


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["python3 -c 'print(1)'", "python3 - <<'EOF'\nprint(1)\nEOF"])
async def test_arbitrary_code_cannot_be_allowed_permanently(workdir, command):
    tools = _tools(workdir)
    _, entry = await _question_and_entry(tools, command)
    assert entry is None


@pytest.mark.asyncio
async def test_ordinary_command_question_keeps_its_prefix(workdir):
    tools = _tools(workdir, commands=())
    from ai_assist.filesystem_tools import compute_allowlist_prefix

    question, entry = await _question_and_entry(tools, "tool-x --verbose run")
    assert entry == compute_allowlist_prefix(question) == "tool-x run"
    assert tools.allowlist_entry_for("tool-y build now") == compute_allowlist_prefix("tool-y build now")


@pytest.mark.asyncio
@pytest.mark.parametrize("form", ["{abs}", "script:{abs}", "script:~/tools/hello.py", "script:tools/hello.py"])
async def test_script_entries_saved_by_older_versions_are_honored(workdir, monkeypatch, form):
    """Older versions saved "python3 script:<path>" on "always": the user approved those."""
    monkeypatch.setenv("HOME", str(workdir))
    entry = "python3 " + form.format(abs=workdir / "tools" / "hello.py")
    tools = _tools(workdir, commands=("python3", entry))

    result = await tools.execute_tool("execute_command", {"command": "python3 tools/hello.py"})
    assert "Exit code: 0" in result, result


@pytest.mark.asyncio
async def test_script_entry_does_not_allow_another_script(workdir):
    (workdir / "tools" / "other.py").write_text("print('other')\n")
    tools = _tools(workdir, commands=("python3", f"python3 script:{workdir / 'tools' / 'hello.py'}"))

    result = await tools.execute_tool("execute_command", {"command": "python3 tools/other.py"})
    assert "not allowed in non-interactive mode" in result
