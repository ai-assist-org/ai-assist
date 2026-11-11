"""internal__transform: run a short payload over one file, with preloaded modules and no imports."""

import json
import re

import pytest

from ai_assist.config import AiAssistConfig
from ai_assist.filesystem_tools import FilesystemTools
from ai_assist.redaction import Redactor
from ai_assist.transform_tools import TransformTools


@pytest.fixture
def workdir(tmp_path, isolate_config_dir):
    return tmp_path


@pytest.fixture
def tools(workdir):
    fs = FilesystemTools(AiAssistConfig(anthropic_api_key="test", allowed_paths=[str(workdir)]), load_user_config=False)
    return TransformTools(fs, Redactor({"s3cr3t-value-123": "API_TOKEN"}, []))


async def transform(tools, workdir, content, code, name="data.txt", **extra):
    path = workdir / name
    path.write_text(content)
    return await tools.execute_tool("transform", {"file": str(path), "code": code, **extra})


# --- representative payloads (what agents did with python3 -c and awk) ---


async def test_regex_over_lines(tools, workdir):
    log = "start\nERROR disk full\nok\nERROR net down\n"
    out = await transform(
        tools, workdir, log, "for l in lines:\n    m = re.search(r'ERROR (\\w+)', l)\n    if m: print(m.group(1))"
    )
    assert out.split() == ["disk", "net"]


async def test_csv_field_sum(tools, workdir):
    out = await transform(
        tools,
        workdir,
        "name,amount\na,3\nb,4\n",
        "rows = csv.DictReader(io.StringIO(text))\nprint(sum(int(r['amount']) for r in rows))",
    )
    assert out.strip() == "7"


async def test_json_and_yaml(tools, workdir):
    assert (
        await transform(tools, workdir, '{"a": [1, 2]}', "print(len(json.loads(text)['a']))", "d.json")
    ).strip() == "2"
    assert (
        await transform(tools, workdir, "x:\n  y: 5\n", "print(yaml.safe_load(text)['x']['y'])", "d.yaml")
    ).strip() == "5"


async def test_counter_dates_and_sorting(tools, workdir):
    code = (
        "c = collections.Counter(l.split()[0] for l in lines)\n"
        "print(sorted(c.items()))\n"
        "d = datetime.date(2026, 11, 11) - datetime.timedelta(days=10)\n"
        "print(d.isoformat())"
    )
    out = await transform(tools, workdir, "b x\na y\nb z\n", code)
    assert out.splitlines() == ["[('a', 1), ('b', 2)]", "2026-11-01"]


async def test_path_and_tilde_expansion(tools, workdir, monkeypatch):
    monkeypatch.setenv("HOME", str(workdir))
    (workdir / "notes.txt").write_text("one\ntwo\n")
    out = await tools.execute_tool("transform", {"file": "~/notes.txt", "code": "print(path, len(lines))"})
    assert out.strip() == "notes.txt 2"


# --- the boundary: no imports, no files, no escapes ---


@pytest.mark.parametrize(
    "code",
    [
        "import os",
        "from subprocess import run",
        "__import__('os')",
        "open('/etc/passwd')",
        "eval('1')",
        "exec('x=1')",
        "print(yaml.load(text))",
        "print(io.open)",
        "print(().__class__.__base__.__subclasses__())",
        "print(json.loads.__globals__)",
        "print((x for x in [1]).gi_frame.f_back)",
        "print('{0.__class__}'.format(1))",
        "print(getattr(json, '__dict__'))",
    ],
)
async def test_payload_cannot_import_open_or_escape(tools, workdir, code):
    out = await transform(tools, workdir, "x\n", code)
    assert out.startswith("transform error:"), out


@pytest.mark.parametrize(
    "code",
    [
        "import re\nprint(re.findall(r'\\w+', text))",
        "import json as j\nprint(j.dumps(lines))",
        "from collections import Counter\nprint(Counter(lines))",
        "import re, json\nprint(json.dumps(re.findall(r'x', text)))",
    ],
)
async def test_importing_a_preloaded_module_just_works(tools, workdir, code):
    """Agents write imports out of habit: for the preloaded modules they are no-ops."""
    out = await transform(tools, workdir, "x\n", code)
    assert not out.startswith("transform error"), out


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (
            "counts = {}\nfor l in lines:\n    counts[l] = counts.get(l, 0)\n    counts[l] += 1\nprint(counts)",
            "{'a': 2, 'b': 1}",
        ),
        ("c = collections.Counter()\nfor l in lines:\n    c[l] += 1\nprint(dict(c))", "{'a': 2, 'b': 1}"),
        ("d = collections.defaultdict(int)\nfor l in lines:\n    d[l] += 1\nprint(dict(d))", "{'a': 2, 'b': 1}"),
        ("x = [0, 0]\nx[1] += 5\nprint(x)", "[0, 5]"),
    ],
)
async def test_counting_idioms_work(tools, workdir, code, expected):
    """`counts[k] += 1` on dicts, Counter and defaultdict: what agents write first."""
    out = await transform(tools, workdir, "a\nb\na\n", code)
    assert out.strip() == expected, out


async def test_augmented_assignment_errors_keep_their_line(tools, workdir):
    out = await transform(tools, workdir, "x\n", "d = {}\nd['missing'] += 1\n")
    assert out.startswith("transform error: line 2: KeyError"), out


@pytest.mark.parametrize("code", ["import collections.abc", "from os import path", "import re, os"])
async def test_importing_anything_else_still_fails(tools, workdir, code):
    out = await transform(tools, workdir, "x\n", code)
    assert out.startswith("transform error:") and "ImportError" in out, out


async def test_import_error_names_the_preloaded_modules(tools, workdir):
    out = await transform(tools, workdir, "x\n", "import os")
    assert "re, json, csv" in out


async def test_secrets_are_redacted_before_the_payload_sees_them(tools, workdir):
    out = await transform(tools, workdir, "token=s3cr3t-value-123\n", "print(text)")
    assert "s3cr3t-value-123" not in out and "REDACTED" in out


async def test_file_outside_allowed_paths_is_refused(tools, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "f.txt"
    outside.write_text("x\n")
    out = await tools.execute_tool("transform", {"file": str(outside), "code": "print(text)"})
    assert out.startswith("Error") and "x\n" not in out


# --- results ---


async def test_error_reports_the_payload_line_without_paths(tools, workdir):
    out = await transform(tools, workdir, "x\n", "a = 1\nb = a / 0\n")
    assert out.startswith("transform error: line 2: ZeroDivisionError")
    assert "/" not in out.split("ZeroDivisionError", 1)[1].replace("division by zero", "")


async def test_syntax_error_is_reported(tools, workdir):
    out = await transform(tools, workdir, "x\n", "print(\n")
    assert out.startswith("transform error:") and "line" in out


async def test_empty_output_is_said(tools, workdir):
    assert await transform(tools, workdir, "x\n", "n = len(lines)") == "(no output)"


async def test_output_limits_save_the_full_output(tools, workdir):
    out = await transform(tools, workdir, "x\n", "for i in range(300): print(i)", max_output_lines=5)
    assert out.splitlines()[:5] == ["0", "1", "2", "3", "4"]
    saved = re.search(r"full output saved to (\S+):", out)
    assert saved, out
    assert len(open(saved.group(1)).read().splitlines()) == 300


async def test_endless_payload_times_out(tools, workdir):
    tools.timeout_seconds = 2
    out = await transform(tools, workdir, "x\n", "while True: pass")
    assert out == "transform error: timed out after 2 seconds"


def test_tool_definition_is_read_only_and_lists_the_modules(tools):
    [definition] = tools.get_tool_definitions()
    assert definition["name"] == "internal__transform" and definition["_readonly"] is True
    assert set(definition["input_schema"]["required"]) == {"file", "code"}
    assert "re, json, csv" in definition["description"] and "any other import fails" in definition["description"]
    json.dumps(definition)


# --- registration in the agent ---


@pytest.fixture
async def agent(workdir):
    from ai_assist.agent import AiAssistAgent

    agent = AiAssistAgent(AiAssistConfig(anthropic_api_key="test", mcp_servers={}, allowed_paths=[str(workdir)]))
    await agent.connect_to_servers()
    return agent


async def test_agent_offers_and_dispatches_the_tool(agent, workdir):
    from ai_assist.plan_mode import get_planning_tools

    names = [t["name"] for t in agent.available_tools]
    assert "internal__transform" in names
    assert "internal__transform" in [t["name"] for t in get_planning_tools(agent.available_tools)]
    (workdir / "f.txt").write_text("a\nb\n")
    out = await agent._execute_tool(
        "internal__transform", {"file": str(workdir / "f.txt"), "code": "print(len(lines))"}
    )
    assert out.strip() == "2"


async def test_system_prompt_steers_to_the_tool(agent):
    prompt = "".join(block["text"] for block in agent._build_system_prompt())
    assert "## Text processing" in prompt
    assert "internal__transform" in prompt and "python3 -c" in prompt


def test_transform_results_can_be_saved_to_a_file():
    from ai_assist.agent import _meta_params_for_tool

    assert "__save_to_file" in _meta_params_for_tool("internal__transform", jq_available=False)


# --- weaker layers when the kernel confinement is not available (user choice) ---


@pytest.mark.parametrize(
    "confinement",
    [
        "unavailable: no kernel confinement on darwin",
        "landlock ABI 3: no files; no seccomp filter for riscv64: sockets are not blocked",
    ],
)
async def test_missing_kernel_confinement_is_said_in_the_result(tools, workdir, monkeypatch, caplog, confinement):
    import logging

    async def child(request, timeout):
        return {"output": "42\n", "error": None, "confinement": confinement}

    monkeypatch.setattr("ai_assist.transform_tools.run_child", child)
    monkeypatch.setattr(TransformTools, "_warned", False)
    with caplog.at_level(logging.WARNING, logger="ai_assist.transform_tools"):
        first = await transform(tools, workdir, "x\n", "print(42)")
        await transform(tools, workdir, "x\n", "print(42)")
    assert first.startswith("42\n")
    assert "note: kernel confinement" in first and confinement.split(":", 1)[1].strip()[:20] in first
    assert len([r for r in caplog.records if "confinement" in r.getMessage()]) == 1


async def test_full_confinement_adds_no_note(tools, workdir):
    out = await transform(tools, workdir, "x\n", "print(1)")
    assert out == "1\n"


# --- macOS: the child runs under sandbox-exec (built here, executed only on a Mac) ---


def test_macos_child_runs_under_sandbox_exec(monkeypatch):
    import ai_assist.transform_tools as mod

    monkeypatch.setattr(mod.sys, "platform", "darwin")
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/sandbox-exec" if name == "sandbox-exec" else None)
    command, sandboxed = mod.child_command()
    assert command[:2] == ["/usr/bin/sandbox-exec", "-p"] and sandboxed
    assert command[3:] == [mod.sys.executable, "-I", "-m", "ai_assist.transform_harness"]
    profile = command[2]
    for rule in ("(deny network*)", "(deny process-exec*)", "(deny process-fork)", "(deny file-write*)"):
        assert rule in profile
    assert f'(subpath "{mod.sys.prefix}")' in profile and "(deny file-read* (subpath" in profile


def test_linux_child_needs_no_wrapper(monkeypatch):
    import ai_assist.transform_tools as mod

    monkeypatch.setattr(mod.sys, "platform", "linux")
    command, sandboxed = mod.child_command()
    assert command == [mod.sys.executable, "-I", "-m", "ai_assist.transform_harness"] and not sandboxed


async def test_confinement_status_reports_the_child_confinement():
    from ai_assist.transform_tools import confinement_status

    status = await confinement_status()
    assert status.startswith(("landlock", "unavailable:", "sandbox-exec"))


# --- common idioms work; what stays refused is explained ---


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("class C:\n    pass\nc = C()\nc.n = 1\nc.n += 2\nprint(c.n)", "3"),
        ("class P:\n    def __init__(self, x):\n        self.x = x\nprint(P(4).x)", "4"),
        ("print(hasattr(lines, 'append'), getattr(text, 'upper')())", "True A\nB"),
        ("print(type(1) is int, type(lines) is list)", "True True"),
        ("print(datetime.datetime.strptime('2026-11-11', '%Y-%m-%d').date())", "2026-11-11"),
    ],
)
async def test_more_idioms_work(tools, workdir, code, expected):
    out = await transform(tools, workdir, "a\nb\n", code)
    assert out.strip() == expected, out


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("print(getattr(text, '__class__'))", "starts with"),
        ("print(hasattr(text, '__class__'))", "False"),  # a guarded hasattr: never reveals `_` names
        ("print(type('X', (), {}))", "type() takes 1 argument here"),
        ("print(open('x'))", "open is not available"),
        ("eval('1')", "Eval calls are not allowed"),  # RestrictedPython refuses it at compile time
        ("match lines:\n    case _: pass", "Match statements are not allowed"),
        ("def f():\n    x = 0\n    def g():\n        nonlocal x\n    return g", "Nonlocal statements are not allowed"),
    ],
)
async def test_refusals_are_explained(tools, workdir, code, message):
    out = await transform(tools, workdir, "a\n", code)
    assert message in out, out


def test_description_lists_what_is_not_supported(tools):
    [definition] = tools.get_tool_definitions()
    for construct in ("match", "nonlocal", "async", "open"):
        assert construct in definition["description"]
