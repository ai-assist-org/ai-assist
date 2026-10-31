"""Callers must not mistake a model API failure for the model's answer."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_assist.agent import QueryAPIError

API_ERROR_TEXT = "API Rate Limit Error: usage limit reached\n\nPlease wait before retrying."


async def _failing_query(*args, raise_on_api_error=False, **kwargs):
    """Behave like AiAssistAgent.query when every model call fails."""
    if raise_on_api_error:
        raise QueryAPIError(API_ERROR_TEXT)
    return API_ERROR_TEXT


@pytest.mark.asyncio
async def test_background_task_fails(tmp_path):
    from ai_assist.background_tasks import BackgroundTask, BackgroundTaskManager

    agent = MagicMock()
    agent.query = AsyncMock(side_effect=_failing_query)
    agent._background_task_count = 0
    notifications = tmp_path / "notifications.jsonl"
    manager = BackgroundTaskManager(agent=agent, console=MagicMock(), notifications_file=notifications)
    task = BackgroundTask(description="Summarize", prompt="Summarize the logs")

    await manager._run_task(task, None, max_turns=5, max_time_seconds=60)

    assert task.status == "failed"
    assert "Rate Limit" in task.error
    assert json.loads(notifications.read_text().splitlines()[-1])["level"] == "error"


@pytest.mark.asyncio
async def test_create_google_doc_action_reports_failure(tmp_path, caplog):
    from ai_assist.conditions import ActionExecutor
    from ai_assist.state import StateManager

    agent = MagicMock()
    agent.query = AsyncMock(side_effect=_failing_query)
    executor = ActionExecutor(agent, StateManager(state_dir=tmp_path / "state"))

    with caplog.at_level(logging.INFO):
        await executor._create_doc({"title": "Daily report"}, {"result": "content"})

    assert "Created Google Doc" not in caplog.text
    assert "Error creating Google Doc" in caplog.text


@pytest.mark.asyncio
async def test_awl_goal_judge_does_not_decide_on_api_error():
    from ai_assist.awl_ast import GoalNode
    from ai_assist.awl_runtime import AWLRuntime

    agent = MagicMock()
    agent.query = AsyncMock(side_effect=_failing_query)
    runtime = AWLRuntime(agent)
    goal = GoalNode(goal_id="green", success_criteria="The build is green", body=[])

    with pytest.raises(QueryAPIError):
        await runtime._evaluate_goal_success_llm(goal, "  build = red")
    assert "_goal_success_met" not in runtime._variables


@pytest.mark.asyncio
async def test_execute_mcp_prompt_tool_reports_failure():
    from ai_assist.introspection_tools import IntrospectionTools

    async def failing_prompt(*args, raise_on_api_error=False, **kwargs):
        if raise_on_api_error:
            raise QueryAPIError(API_ERROR_TEXT)
        return ""

    agent = MagicMock()
    agent.sessions = {"srv": MagicMock()}
    agent.available_prompts = {"srv": {"report": MagicMock(arguments=[])}}
    agent.execute_mcp_prompt = AsyncMock(side_effect=failing_prompt)
    tools = IntrospectionTools()
    tools.agent = agent

    result = json.loads(await tools._execute_mcp_prompt({"server": "srv", "prompt": "report"}))

    assert result["success"] is False
    assert "Rate Limit" in result["error"]


@pytest.mark.asyncio
async def test_one_off_query_exits_non_zero(capsys):
    from ai_assist.main import run_query

    agent = MagicMock()
    agent.query = AsyncMock(side_effect=_failing_query)

    with pytest.raises(SystemExit) as exc:
        await run_query(agent, "What failed today?")

    assert exc.value.code == 1
    assert "Rate Limit" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_basic_prompt_command_keeps_api_error_out_of_history(capsys):
    from ai_assist.main import handle_prompt_command_basic

    message = MagicMock(role="user")
    message.content.text = "Write the daily report."
    session = MagicMock()
    session.get_prompt = AsyncMock(return_value=MagicMock(messages=[message]))
    agent = MagicMock()
    agent.sessions = {"srv": session}
    agent.available_prompts = {"srv": {"report": MagicMock(arguments=[])}}
    agent.get_truncation_limits.return_value = {"max_message_chars": 100_000, "max_total_chars": 1_000_000}
    agent.query = AsyncMock(side_effect=_failing_query)
    history: list[dict] = []

    await handle_prompt_command_basic("/srv/report", agent, history, MagicMock())

    assert all(m["role"] != "assistant" for m in history)
    assert "Rate Limit" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_plan_mode_does_not_offer_api_error_as_plan():
    from ai_assist.tui_interactive import handle_plan_command

    memory = MagicMock()
    console = MagicMock()

    with (
        patch("ai_assist.tui_interactive.query_with_feedback", AsyncMock(return_value=API_ERROR_TEXT)),
        patch("prompt_toolkit.PromptSession") as session_cls,
    ):
        await handle_plan_command("Refactor the parser", MagicMock(), console, conversation_memory=memory)

    memory.add_exchange.assert_not_called()
    session_cls.assert_not_called()
    assert not any("Plan Ready" in str(call) for call in console.print.call_args_list)


@pytest.mark.asyncio
async def test_plan_execution_api_error_not_saved():
    from ai_assist.tui_interactive import handle_plan_command

    memory = MagicMock()
    session = MagicMock()
    session.prompt_async = AsyncMock(return_value="a")

    with (
        patch(
            "ai_assist.tui_interactive.query_with_feedback",
            AsyncMock(side_effect=["1. Do the thing", API_ERROR_TEXT]),
        ),
        patch("prompt_toolkit.PromptSession", return_value=session),
    ):
        await handle_plan_command("Refactor the parser", MagicMock(), MagicMock(), conversation_memory=memory)

    saved = [call.args[0] for call in memory.add_exchange.call_args_list]
    assert saved == ["/plan Refactor the parser"]
