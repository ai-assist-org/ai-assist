"""internal__search_in_file and internal__list_directory: the grep / ls forms they replace."""

import os
import re

import pytest

from ai_assist.config import AiAssistConfig
from ai_assist.filesystem_tools import FilesystemTools


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "app.log").write_text("start\nERROR disk full\nok\nerror: retry 3\nok\nERROR net down\nend\n")
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("import os\n# TODO: fix\nprint('x')\n")
    (src / "util.py").write_text("def f():\n    return 1  # todo later\n")
    (src / "notes.txt").write_text("TODO in a text file\n")
    (src / "blob.bin").write_bytes(b"TODO\x00\x01\x02")
    hidden = src / ".git"
    hidden.mkdir()
    (hidden / "config").write_text("TODO inside git\n")
    return tmp_path


@pytest.fixture
def tools(tree):
    config = AiAssistConfig(anthropic_api_key="test", allowed_paths=[str(tree)])
    return FilesystemTools(config, load_user_config=False)


async def search(tools, **args):
    return await tools.execute_tool("search_in_file", args)


# --- search_in_file: unchanged default ---


async def test_default_lists_matching_lines_with_numbers(tools, tree):
    out = await search(tools, path=str(tree / "app.log"), pattern="ERROR")
    assert "2: ERROR disk full" in out
    assert "6: ERROR net down" in out
    assert "retry" not in out


# --- search_in_file: grep options ---


async def test_ignore_case(tools, tree):  # grep -i
    out = await search(tools, path=str(tree / "app.log"), pattern="error", ignore_case=True)
    assert "2: ERROR disk full" in out and "4: error: retry 3" in out


async def test_invert(tools, tree):  # grep -v
    out = await search(tools, path=str(tree / "app.log"), pattern="ok|ERROR", invert=True)
    assert "1: start" in out and "4: error: retry 3" in out
    assert "ERROR" not in out.split(":\n\n", 1)[1]


async def test_count(tools, tree):  # grep -c
    out = await search(tools, path=str(tree / "app.log"), pattern="ERROR", output="count")
    assert out.strip().endswith("2")


async def test_only_matching(tools, tree):  # grep -o
    out = await search(tools, path=str(tree / "app.log"), pattern=r"retry \d+|disk \w+", output="only_matching")
    assert "2: disk full" in out and "4: retry 3" in out
    assert "ERROR" not in out.split(":\n\n", 1)[1]


async def test_context_lines(tools, tree):  # grep -A/-B/-C
    out = await search(tools, path=str(tree / "app.log"), pattern="retry", context=1)
    assert "3- ok" in out and "4: error: retry 3" in out and "5- ok" in out
    assert "start" not in out


# --- search_in_file: a directory tree (grep -r, --include, -l) ---


async def test_directory_search_is_recursive_and_skips_hidden_and_binary(tools, tree):
    out = await search(tools, path=str(tree / "src"), pattern="TODO")
    assert "main.py:2: # TODO: fix" in out
    assert "notes.txt:1: TODO in a text file" in out
    assert ".git" not in out and "blob.bin" not in out


async def test_directory_search_glob_filter(tools, tree):  # grep -r --include='*.py'
    out = await search(tools, path=str(tree / "src"), pattern="todo", ignore_case=True, glob="*.py")
    assert "main.py:2:" in out and "util.py:2:" in out
    assert "notes.txt" not in out


async def test_files_with_matches(tools, tree):  # grep -rl
    out = await search(tools, path=str(tree / "src"), pattern="TODO", output="files")
    assert "main.py" in out and "notes.txt" in out
    assert "util.py" not in out and ":2:" not in out


async def test_count_per_file(tools, tree):  # grep -rc
    out = await search(tools, path=str(tree / "src"), pattern="todo", ignore_case=True, output="count")
    assert "main.py: 1" in out and "util.py: 1" in out and "notes.txt: 1" in out


async def test_directory_search_does_not_follow_symlinks_out_of_the_tree(tools, tree, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "secret.txt"
    outside.write_text("TODO secret\n")
    os.symlink(outside, tree / "src" / "link.txt")
    out = await search(tools, path=str(tree / "src"), pattern="TODO")
    assert "link.txt" not in out
    assert "main.py" in out


async def test_directory_search_skips_secret_files(tools, tree):
    """A tree search must not leak a .env sitting in an allowed directory."""
    (tree / "src" / ".env").write_text("TOKEN=TODO-secret-value\n")
    out = await search(tools, path=str(tree / "src"), pattern="TODO")
    assert "secret-value" not in out


async def test_invalid_output_mode(tools, tree):
    out = await search(tools, path=str(tree / "app.log"), pattern="x", output="weird")
    assert out.startswith("Error") and "lines, count, only_matching, files" in out


# --- list_directory: ls -l, ls -t, | head ---


async def listing(tools, **args):
    return await tools.execute_tool("list_directory", args)


async def test_listing_shows_hidden_files(tools, tree):  # ls -a (already the case)
    out = await listing(tools, path=str(tree / "src"))
    assert ".git/" in out


async def test_long_listing_has_times_and_permissions(tools, tree):  # ls -l
    os.utime(tree / "app.log", (1_700_000_000, 1_700_000_000))
    out = await listing(tools, path=str(tree), long=True)
    line = next(x for x in out.splitlines() if "app.log" in x)
    assert "-rw" in line and "2023-11-1" in line


def names(out):
    return re.findall(r"\[(?:FILE|DIR)\]\s+(\S+)", out)


async def test_sort_by_time_newest_first_with_limit(tools, tree):  # ls -t | head -2
    for i, name in enumerate(["a.txt", "b.txt", "c.txt"]):
        (tree / name).write_text(name)
        os.utime(tree / name, (2_000_000_000 + i, 2_000_000_000 + i))
    out = await listing(tools, path=str(tree), sort="time", max_results=2)
    assert names(out) == ["c.txt", "b.txt"]
    assert "more" in out


async def test_sort_by_size(tools, tree):  # ls -S
    (tree / "src" / "big.txt").write_text("x" * 5000)
    out = await listing(tools, path=str(tree / "src"), sort="size")
    files = [n for n in names(out) if not n.endswith("/")]
    assert files[0] == "big.txt"


async def test_invalid_sort(tools, tree):
    out = await listing(tools, path=str(tree), sort="random")
    assert out.startswith("Error") and "name, time, size" in out


async def test_directory_with_no_searchable_file(tools, tree):
    (tree / "empty").mkdir()
    out = await search(tools, path=str(tree / "empty"), pattern="x")
    assert out == "No matches found for pattern: x in 0 file(s)"


# --- line length: awk 'length > 100' ---


async def test_long_lines_with_a_length_pattern(tools, tree):
    lines = ["a" * 100, "b" * 101, "c" * 500, "d" * 101 + "\r"]
    (tree / "wide.txt").write_bytes("\n".join(lines).encode() + b"\n")
    out = await search(tools, path=str(tree / "wide.txt"), pattern=".{101,}")
    numbers = sorted(int(m) for m in re.findall(r"^(\d+):", out, re.MULTILINE))
    assert numbers == [2, 3, 4], out


def test_description_teaches_the_line_length_pattern(tools):
    [definition] = [t for t in tools.get_tool_definitions() if t["name"] == "internal__search_in_file"]
    assert ".{101,}" in definition["description"] and "output=count" in definition["description"]
