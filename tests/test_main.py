"""Tests for main.py synchronous command functions and pure helpers"""

from datetime import datetime
from unittest.mock import patch

import pytest

from ai_assist.identity import AssistantIdentity, Identity, UserIdentity
from ai_assist.knowledge_graph import KnowledgeGraph
from ai_assist.main import (
    cost_command,
    eval_stats_command,
    identity_init_command,
    identity_show_command,
    kg_asof_command,
    kg_changes_command,
    kg_late_command,
    kg_show_command,
    kg_stats_command,
    reset_terminal_title,
    set_terminal_title,
    should_use_tui,
)


@pytest.fixture
def kg(tmp_path):
    """Knowledge graph backed by a temp database"""
    graph = KnowledgeGraph(db_path=str(tmp_path / "kg.db"))
    yield graph
    graph.close()


@pytest.fixture
def populated_kg(kg):
    """Knowledge graph with a task, related resource and person"""
    kg.insert_entity(
        entity_type="task",
        entity_id="task-1",
        valid_from=datetime(2026, 2, 4, 10, 0),
        tx_from=datetime(2026, 2, 4, 10, 5),
        data={"status": "blocked"},
    )
    kg.insert_entity(
        entity_type="resource",
        entity_id="res-1",
        valid_from=datetime(2026, 2, 4, 0, 0),
        tx_from=datetime(2026, 2, 4, 0, 0),
        data={"name": "shared-pool"},
    )
    kg.insert_relationship(
        rel_type="depends_on",
        source_id="task-1",
        target_id="res-1",
        valid_from=datetime(2026, 2, 4, 10, 0),
    )
    return kg


# --- pure helpers -----------------------------------------------------------


def test_set_terminal_title_not_tty(capsys):
    with patch("ai_assist.main.sys.stdout") as out:
        out.isatty.return_value = False
        set_terminal_title("Title")
    out.write.assert_not_called()


def test_set_terminal_title_tty(monkeypatch):
    for var in ("TMUX", "ZELLIJ", "STY"):
        monkeypatch.delenv(var, raising=False)
    with patch("ai_assist.main.sys.stdout") as out:
        out.isatty.return_value = True
        set_terminal_title("Title")
    assert any("Title" in str(c.args[0]) for c in out.write.call_args_list)


def test_reset_terminal_title_not_tty():
    with patch("ai_assist.main.sys.stdout") as out:
        out.isatty.return_value = False
        reset_terminal_title()
    out.write.assert_not_called()


def test_reset_terminal_title_tty(monkeypatch):
    for var in ("TMUX", "ZELLIJ", "STY"):
        monkeypatch.delenv(var, raising=False)
    with patch("ai_assist.main.sys.stdout") as out:
        out.isatty.return_value = True
        reset_terminal_title()
    out.write.assert_called()


def test_should_use_tui_not_tty():
    with patch("sys.stdin") as stdin:
        stdin.isatty.return_value = False
        assert should_use_tui() is False


def test_should_use_tui_basic_mode(monkeypatch):
    monkeypatch.setenv("AI_ASSIST_INTERACTIVE_MODE", "basic")
    with patch("sys.stdin") as stdin:
        stdin.isatty.return_value = True
        assert should_use_tui() is False


def test_should_use_tui_enabled(monkeypatch):
    monkeypatch.setenv("AI_ASSIST_INTERACTIVE_MODE", "tui")
    with patch("sys.stdin") as stdin:
        stdin.isatty.return_value = True
        assert should_use_tui() is True


# --- knowledge graph commands ----------------------------------------------


def test_kg_stats_command(kg, capsys):
    kg_stats_command(kg)
    out = capsys.readouterr().out
    assert "Knowledge Graph Statistics" in out
    assert "Total entities" in out


def test_kg_stats_command_populated(populated_kg, capsys):
    kg_stats_command(populated_kg)
    out = capsys.readouterr().out
    assert "task" in out
    assert "depends_on" in out


def test_kg_changes_command_with_entities(kg, capsys):
    now = datetime.now()
    kg.insert_entity(
        entity_type="task",
        entity_id="recent-task",
        valid_from=now,
        tx_from=now,
        data={"status": "open"},
    )
    kg_changes_command(kg, hours=1)
    out = capsys.readouterr().out
    assert "New entities:" in out
    assert "recent-task" in out


def test_kg_asof_command_invalid_time(kg, capsys):
    kg_asof_command(kg, "not-a-date")
    out = capsys.readouterr().out
    assert "Invalid time format" in out


def test_kg_asof_command_valid(populated_kg, capsys):
    kg_asof_command(populated_kg, "2026-02-05 00:00")
    out = capsys.readouterr().out
    assert "What ai-assist knew at" in out
    assert "Total entities" in out


def test_kg_late_command(populated_kg, capsys):
    kg_late_command(populated_kg, min_delay=1)
    out = capsys.readouterr().out
    assert "discovered" in out
    assert "Total:" in out


def test_kg_changes_command(kg, capsys):
    kg_changes_command(kg, hours=1)
    out = capsys.readouterr().out
    assert "Changes in the last 1 hour" in out
    assert "New entities" in out


def test_kg_show_command_not_found(kg, capsys):
    kg_show_command(kg, "missing-id")
    out = capsys.readouterr().out
    assert "Entity not found" in out


def test_kg_show_command_found(populated_kg, capsys):
    kg_show_command(populated_kg, "task-1")
    out = capsys.readouterr().out
    assert "task-1" in out
    assert "depends_on" in out


# --- eval / cost commands ---------------------------------------------------


def _write_trace():
    from ai_assist.eval import QueryTrace, TraceStore

    trace = QueryTrace(
        query_text="hello",
        timestamp=datetime.now().isoformat(),
        turn_count=1,
        response_text="hi",
        total_input_tokens=100,
        total_output_tokens=50,
        total_cost_usd=0.01,
        duration_seconds=1.5,
        model="claude-sonnet-4-6",
        tools_available_count=3,
    )
    TraceStore().append(trace)


def test_eval_stats_command_empty(capsys):
    eval_stats_command()
    out = capsys.readouterr().out
    assert "No query traces found" in out


def test_eval_stats_command_with_traces(capsys):
    _write_trace()
    eval_stats_command()
    out = capsys.readouterr().out
    assert "Evaluation Metrics" in out
    assert "Total cost" in out


def test_cost_command_empty(capsys):
    cost_command()
    out = capsys.readouterr().out
    assert "No query traces found" in out


def test_cost_command_invalid_period(capsys):
    cost_command("7x")
    out = capsys.readouterr().out
    assert "Invalid period" in out


def test_cost_command_with_traces(capsys):
    _write_trace()
    cost_command()
    out = capsys.readouterr().out
    assert "Cost Summary" in out
    assert "By model" in out


# --- identity commands ------------------------------------------------------


def test_identity_show_command(capsys):
    identity = Identity(
        user=UserIdentity(name="Alex", role="Engineer", organization="Acme"),
        assistant=AssistantIdentity(nickname="Helper"),
    )
    with patch("ai_assist.main.get_identity", return_value=identity):
        identity_show_command()
    out = capsys.readouterr().out
    assert "Current Identity" in out
    assert "Alex" in out
    assert "Helper" in out


def test_identity_show_command_full(capsys):
    identity = Identity(
        user=UserIdentity(
            name="Alex",
            role="Engineer",
            organization="Acme",
            timezone="Europe/Paris",
            context="x" * 300,
        ),
        assistant=AssistantIdentity(nickname="Helper", personality="friendly and terse"),
    )
    with patch("ai_assist.main.get_identity", return_value=identity):
        identity_show_command()
    out = capsys.readouterr().out
    assert "Europe/Paris" in out
    assert "User Context" in out
    assert "Custom Personality: Yes" in out


def test_identity_init_command(capsys, tmp_path):
    inputs = iter(["Alex", "Engineer", "Acme", "Helper"])
    with (
        patch("ai_assist.identity.get_config_dir", return_value=tmp_path),
        patch("ai_assist.main.get_config_dir", return_value=tmp_path),
        patch("builtins.input", lambda _prompt="": next(inputs)),
    ):
        identity_init_command()
    out = capsys.readouterr().out
    assert "Identity saved" in out
    assert (tmp_path / "identity.yaml").exists()
