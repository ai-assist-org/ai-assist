"""Tests for AWL local variables (built-in + user-defined from variables.yaml)"""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_assist.awl_executor import (
    _generate_builtin_variables,
    _merge_all_variables,
    get_missing_variables,
)
from ai_assist.config import load_local_variables

# ── load_local_variables ─────────────────────────────────────────


def test_load_local_variables_missing_file(tmp_path):
    result = load_local_variables(tmp_path)
    assert result == {}


def test_load_local_variables_valid_yaml(tmp_path):
    (tmp_path / "variables.yaml").write_text("workspace: /opt/work\nteam: dci\n")
    result = load_local_variables(tmp_path)
    assert result == {"workspace": "/opt/work", "team": "dci"}


def test_load_local_variables_tilde_expansion(tmp_path):
    (tmp_path / "variables.yaml").write_text("workspace: ~/my-workspace\n")
    result = load_local_variables(tmp_path)
    assert result["workspace"] == str(Path.home() / "my-workspace")


def test_load_local_variables_env_var_expansion(tmp_path):
    (tmp_path / "variables.yaml").write_text("home: $HOME/work\n")
    result = load_local_variables(tmp_path)
    assert result["home"] == os.environ["HOME"] + "/work"


def test_load_local_variables_none_value(tmp_path):
    (tmp_path / "variables.yaml").write_text("empty_val:\n")
    result = load_local_variables(tmp_path)
    assert result["empty_val"] == ""


def test_load_local_variables_numeric_value(tmp_path):
    (tmp_path / "variables.yaml").write_text("count: 42\n")
    result = load_local_variables(tmp_path)
    assert result["count"] == "42"


def test_load_local_variables_malformed_yaml(tmp_path):
    (tmp_path / "variables.yaml").write_text("{{invalid yaml")
    result = load_local_variables(tmp_path)
    assert result == {}


def test_load_local_variables_yaml_list_instead_of_dict(tmp_path):
    (tmp_path / "variables.yaml").write_text("- one\n- two\n")
    result = load_local_variables(tmp_path)
    assert result == {}


def test_load_local_variables_empty_file(tmp_path):
    (tmp_path / "variables.yaml").write_text("")
    result = load_local_variables(tmp_path)
    assert result == {}


def test_load_local_variables_skips_non_string_keys(tmp_path):
    (tmp_path / "variables.yaml").write_text("123: numeric_key\nvalid_key: value\n")
    result = load_local_variables(tmp_path)
    assert "valid_key" in result
    assert len(result) == 1


# ── _generate_builtin_variables ──────────────────────────────────


def test_generate_builtin_variables_keys():
    result = _generate_builtin_variables()
    expected_keys = {
        "date",
        "time",
        "datetime",
        "utc_date",
        "utc_time",
        "utc_datetime",
        "timezone",
        "config_dir",
        "reports_dir",
        "logs_dir",
    }
    assert set(result.keys()) == expected_keys


def test_generate_builtin_variables_date_format():
    result = _generate_builtin_variables()
    assert len(result["date"]) == 10  # YYYY-MM-DD
    assert result["date"][4] == "-"
    assert len(result["utc_date"]) == 10
    assert len(result["time"]) == 8  # HH:MM:SS
    assert result["time"][2] == ":"


def test_generate_builtin_variables_paths():
    result = _generate_builtin_variables()
    assert Path(result["config_dir"]).is_absolute()
    assert Path(result["reports_dir"]).is_absolute()
    assert result["logs_dir"].endswith("/logs")


# ── _merge_all_variables ─────────────────────────────────────────


def test_merge_all_variables_cli_overrides_local(tmp_path):
    (tmp_path / "variables.yaml").write_text("team: local_team\n")
    with patch("ai_assist.awl_executor.load_local_variables", return_value={"team": "local_team"}):
        result = _merge_all_variables({"team": "cli_team"})
    assert result["team"] == "cli_team"


def test_merge_all_variables_local_overrides_builtin(tmp_path):
    with patch("ai_assist.awl_executor.load_local_variables", return_value={"date": "custom-date"}):
        result = _merge_all_variables()
    assert result["date"] == "custom-date"


def test_merge_all_variables_includes_builtins():
    with patch("ai_assist.awl_executor.load_local_variables", return_value={}):
        result = _merge_all_variables()
    assert "date" in result
    assert "config_dir" in result


# ── get_missing_variables with local vars ────────────────────────


def test_builtin_vars_not_reported_as_missing(tmp_path):
    """AWL script using ${date} should not report it as missing."""
    from ai_assist.awl_parser import AWLParser

    source = "@start\n@set msg = Today is ${date}\n@end\n"
    workflow = AWLParser(source).parse()

    with patch("ai_assist.awl_executor.load_local_variables", return_value={}):
        missing = get_missing_variables(workflow)
    assert "date" not in missing


def test_local_vars_not_reported_as_missing():
    """AWL script using ${workspace} with it in variables.yaml should not report missing."""
    from ai_assist.awl_parser import AWLParser

    source = "@start\n@set path = ${workspace}/data\n@end\n"
    workflow = AWLParser(source).parse()

    with patch("ai_assist.awl_executor.load_local_variables", return_value={"workspace": "/opt/work"}):
        missing = get_missing_variables(workflow)
    assert "workspace" not in missing


def test_truly_missing_var_still_reported():
    """AWL script using an undefined variable should still report it as missing."""
    from ai_assist.awl_parser import AWLParser

    source = "@start\n@set msg = Hello ${unknown_var}\n@end\n"
    workflow = AWLParser(source).parse()

    with patch("ai_assist.awl_executor.load_local_variables", return_value={}):
        missing = get_missing_variables(workflow)
    assert "unknown_var" in missing


# ── CLI error reporting ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_cli_run_shows_builtin_vars_as_provided(tmp_path, capsys):
    """Error message should show builtin vars in the provided set."""
    from ai_assist.main import run_awl_script

    script = tmp_path / "test.awl"
    script.write_text("@start\n@set msg = ${custom_var}\n@end\n")

    agent = MagicMock()

    with (
        patch("ai_assist.awl_executor.load_local_variables", return_value={}),
        pytest.raises(SystemExit),
    ):
        await run_awl_script(agent, str(script), variables={})

    captured = capsys.readouterr()
    assert "custom_var" in captured.out
    assert "date" in captured.out  # builtin should appear in provided


@pytest.mark.asyncio
async def test_cli_run_succeeds_with_builtin_date_var(tmp_path):
    """AWL script using only ${date} should not fail validation."""
    from ai_assist.main import run_awl_script

    script = tmp_path / "test.awl"
    script.write_text("@start\n@set msg = Today is ${date}\n@end\n")

    agent = AsyncMock()
    agent.query = AsyncMock(return_value="done")

    with patch("ai_assist.awl_executor.load_local_variables", return_value={}):
        # Should not raise SystemExit for missing variables
        # (it will proceed to actual execution which we mock)
        with patch("ai_assist.awl_runtime.AWLRuntime.execute") as mock_exec:
            from ai_assist.awl_runtime import WorkflowResult

            mock_exec.return_value = WorkflowResult(success=True, return_value="ok")
            await run_awl_script(agent, str(script), variables={})
