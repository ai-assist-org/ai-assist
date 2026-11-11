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
async def test_command_with_options_before_its_arguments_gets_no_entry(workdir):
    """`tool-x --verbose run`: "tool-x run" would never match it again, "tool-x" would allow more."""
    tools = _tools(workdir, commands=())
    assert await _entry_for_refused(tools, "tool-x --verbose run") is None


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


# --- "always allow" never saves an entry that is junk or allows more than approved ---


@pytest.mark.parametrize(
    "entry",
    [
        "git status",
        "whoami",
        "git worktree list",
        "python3 -m pytest",
        "python3 /home/u/tools/x.py",
        "uv run garmin-golf",
        ".venv/bin/python scripts/health_query.py",
        "sqlite3 ~/data/app.db",
        "grep ^#",
    ],
)
def test_valid_allowlist_entries(entry):
    from ai_assist.filesystem_tools import is_valid_allowlist_entry

    assert is_valid_allowlist_entry(entry)


@pytest.mark.parametrize(
    "entry",
    [
        "$(grep ^#",
        "-r .access_token",
        "bad =",
        "2>&1",
        "ARGS+=",
        "awk NR==2{print",
        "wc $f",
        "comm $__substitution__",
        "(which pdftotext",
        "sort)",
        "mapfile ids",
        "pl.col(putts).is_null()).sum() df",
        "\nfor f",
        "",
    ],
)
def test_junk_allowlist_entries(entry):
    from ai_assist.filesystem_tools import is_valid_allowlist_entry

    assert not is_valid_allowlist_entry(entry)


async def _entry_for_refused(tools, command):
    """Ask about a command (answered "no": nothing runs); return the entry "always" would save."""
    asked = []

    async def callback(question):
        asked.append(question)
        return False

    tools.confirmation_callback = callback
    await tools.execute_tool("execute_command", {"command": command})
    assert asked, "no question was asked"
    return tools.allowlist_entry_for(asked[0])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "entry"),
    [
        ("git worktree list --porcelain", "git worktree list"),
        ("whoami", "whoami"),
        ("docker ps -a", "docker ps"),
        ("wc -l notes.txt", None),  # `wc` alone would allow more than approved
        ("rm -rf build", None),
        ("LC_ALL=C hostname -f", None),
    ],
)
async def test_always_saves_only_an_entry_matching_the_approved_command(workdir, command, entry):
    tools = _tools(workdir, commands=())
    assert await _entry_for_refused(tools, command) == entry


def test_questions_that_are_not_commands_get_no_entry(workdir):
    tools = _tools(workdir)
    assert tools.allowlist_entry_for("Remember as a user preference (added to every future prompt): x") is None
    assert tools.allowlist_entry_for("Schedule action 'a1' running AWL script x.awl") is None


def test_matching_allowlist_entries_uses_the_allowlist_check_rules():
    from ai_assist.filesystem_tools import matching_allowlist_entries

    entries = ["git", "git worktree list", "ls", "sqlite3 ~/a.db", "bash"]
    command = "cd /srv && git worktree list --porcelain | LC_ALL=C sort; timeout 5 ls -la"
    # the longest entry matching each segment; sort is not allowlisted, cd is inert
    assert matching_allowlist_entries(command, entries) == {"git worktree list", "ls"}


def test_matching_allowlist_entries_resolves_script_paths(tmp_path, monkeypatch):
    """Script entries match by resolved path, as the code-execution check does (see _is_script_allowlisted)."""
    from ai_assist.filesystem_tools import matching_allowlist_entries

    real = tmp_path / "real" / "tool.py"
    real.parent.mkdir()
    real.write_text("print(1)\n")
    (tmp_path / "link").symlink_to(real.parent)
    monkeypatch.setenv("HOME", str(tmp_path))
    entry = f"python3 {real}"
    assert matching_allowlist_entries("python3 ~/link/tool.py --all", [entry]) == {entry}
    assert matching_allowlist_entries("python3 ~/link/other.py", [entry]) == set()
    same_file = f"python3 {tmp_path}/link/tool.py"  # two entries for one file: both are in use
    assert matching_allowlist_entries("python3 ~/link/tool.py", [entry, same_file]) == {entry, same_file}
