"""Tests for user-invoked Agent Skills as slash commands (/<skill-name>)."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_assist.skills_loader import SkillContent, SkillMetadata
from ai_assist.tui_interactive import handle_skill_invocation, substitute_skill_args


def _make_skill(name="deploy", body="Deploy the app.", user_invocable=True, argument_hint=None):
    return SkillContent(
        metadata=SkillMetadata(
            name=name,
            description=f"{name} skill",
            skill_path=Path(f"/tmp/skills-cache/{name}"),
            source_type="local",
            user_invocable=user_invocable,
            argument_hint=argument_hint,
        ),
        body=body,
    )


# --- substitute_skill_args ---


def test_substitute_arguments_placeholder():
    assert substitute_skill_args("Deploy to $ARGUMENTS", "prod v2") == "Deploy to prod v2"


def test_substitute_positional():
    assert substitute_skill_args("Move $1 to $2", "src dst") == "Move src to dst"


def test_substitute_quoted_positional_stays_grouped():
    result = substitute_skill_args("Deploy to $1 version $2", '"us east" v2')
    assert result == "Deploy to us east version v2"


def test_substitute_missing_positional_is_empty():
    assert substitute_skill_args("A=$1 B=$2", "only") == "A=only B="


def test_substitute_unbalanced_quote_falls_back_to_whitespace():
    # A stray quote must not raise; falls back to whitespace-split.
    assert substitute_skill_args("First=$1", 'a"b c') == 'First=a"b'


def test_substitute_no_placeholder_appends_args():
    assert substitute_skill_args("Run the thing.", "extra ctx") == "Run the thing.\n\nArguments: extra ctx"


def test_substitute_no_placeholder_no_args_unchanged():
    assert substitute_skill_args("Run the thing.", "") == "Run the thing."


def test_substitute_arguments_is_raw_string():
    # $ARGUMENTS keeps quotes verbatim, unlike positional splitting.
    assert substitute_skill_args("Got: $ARGUMENTS", '"us east"') == 'Got: "us east"'


# --- handle_skill_invocation ---


@pytest.fixture
def agent():
    a = MagicMock()
    a.skills_manager.loaded_skills = {}
    return a


@pytest.mark.asyncio
async def test_invokes_matching_skill(agent):
    agent.skills_manager.loaded_skills = {"deploy": _make_skill("deploy", "Deploy $1")}
    console = MagicMock()

    with patch("ai_assist.tui_interactive.query_with_feedback", new=AsyncMock()) as mock_query:
        handled = await handle_skill_invocation(
            "/deploy prod", agent, console, conversation_memory=MagicMock(), kg_context=None
        )

    assert handled is True
    mock_query.assert_awaited_once()
    # Body was substituted before being sent as the prompt.
    assert mock_query.await_args.args[1] == "Deploy prod"


@pytest.mark.asyncio
async def test_unknown_skill_not_handled(agent):
    console = MagicMock()
    with patch("ai_assist.tui_interactive.query_with_feedback", new=AsyncMock()) as mock_query:
        handled = await handle_skill_invocation(
            "/nope", agent, console, conversation_memory=MagicMock(), kg_context=None
        )
    assert handled is False
    mock_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_opted_out_skill_not_handled(agent):
    agent.skills_manager.loaded_skills = {"secret": _make_skill("secret", user_invocable=False)}
    console = MagicMock()
    with patch("ai_assist.tui_interactive.query_with_feedback", new=AsyncMock()) as mock_query:
        handled = await handle_skill_invocation(
            "/secret", agent, console, conversation_memory=MagicMock(), kg_context=None
        )
    assert handled is False
    mock_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_server_prompt_token_not_handled_as_skill(agent):
    # /server/prompt tokens have a slash in the name and belong to handle_prompt_command.
    agent.skills_manager.loaded_skills = {"dci": _make_skill("dci")}
    console = MagicMock()
    with patch("ai_assist.tui_interactive.query_with_feedback", new=AsyncMock()) as mock_query:
        handled = await handle_skill_invocation(
            "/dci/rca", agent, console, conversation_memory=MagicMock(), kg_context=None
        )
    assert handled is False
    mock_query.assert_not_awaited()


# --- namespaced plugin skills ---


@pytest.mark.asyncio
async def test_namespaced_plugin_skill_invocation(agent):
    agent.skills_manager.loaded_skills = {"acme:review": _make_skill("review", "Review $1")}
    console = MagicMock()
    with patch("ai_assist.tui_interactive.query_with_feedback", new=AsyncMock()) as mock_query:
        handled = await handle_skill_invocation(
            "/acme:review src", agent, console, conversation_memory=MagicMock(), kg_context=None
        )
    assert handled is True
    assert mock_query.await_args.args[1] == "Review src"


@pytest.mark.asyncio
async def test_bare_name_resolves_unique_plugin_skill(agent):
    agent.skills_manager.loaded_skills = {"acme:review": _make_skill("review", "Review $1")}
    console = MagicMock()
    with patch("ai_assist.tui_interactive.query_with_feedback", new=AsyncMock()) as mock_query:
        handled = await handle_skill_invocation(
            "/review src", agent, console, conversation_memory=MagicMock(), kg_context=None
        )
    assert handled is True
    assert mock_query.await_args.args[1] == "Review src"


@pytest.mark.asyncio
async def test_bare_name_ambiguous_plugin_skill_reports(agent):
    agent.skills_manager.loaded_skills = {
        "acme:review": _make_skill("review"),
        "other:review": _make_skill("review"),
    }
    console = MagicMock()
    with patch("ai_assist.tui_interactive.query_with_feedback", new=AsyncMock()) as mock_query:
        handled = await handle_skill_invocation(
            "/review", agent, console, conversation_memory=MagicMock(), kg_context=None
        )
    # Handled (error shown) but not run.
    assert handled is True
    mock_query.assert_not_awaited()
