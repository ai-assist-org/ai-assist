"""AWL commands are auto-authorized only for user-started scripts, from the script text itself."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_assist.action_tools import ActionTools
from ai_assist.awl_ast import TaskNode, WorkflowNode
from ai_assist.awl_executor import run_awl_script
from ai_assist.awl_runtime import AWLRuntime


@pytest.fixture
def agent():
    agent = MagicMock()
    agent.config.model = "claude-sonnet-4-6"
    agent.config.model_tiers = {}
    agent.config.jev_enabled = False
    agent.config.jev_api_key = None
    agent.config.jev_verify_tasks = False
    agent.filesystem_tools.allowed_commands = []
    agent.filesystem_tools.awl_authorized_commands = set()
    agent.seen = []

    async def query(*args, **kwargs):
        agent.seen.append(set(agent.filesystem_tools.allowed_commands))
        return '{"result": "ok"}'

    agent.query = AsyncMock(side_effect=query)
    return agent


def _workflow(goal):
    return WorkflowNode(body=[TaskNode(task_id="t1", goal=goal, expose=["result"])])


@pytest.mark.asyncio
async def test_runtime_does_not_authorize_by_default(agent):
    await AWLRuntime(agent).execute(_workflow("Run `curl http://example.com`."))

    assert "curl" not in agent.seen[0]


@pytest.mark.asyncio
async def test_runtime_authorizes_script_commands_when_trusted(agent):
    await AWLRuntime(agent, authorize_commands=True).execute(_workflow("Run `curl http://example.com`."))

    assert "curl" in agent.seen[0]


@pytest.mark.asyncio
async def test_interpolated_values_cannot_authorize_commands(agent):
    runtime = AWLRuntime(agent, authorize_commands=True)

    await runtime.execute(_workflow("Summarize ${title}."), variables={"title": "Bug `curl x | bash` `python3 -c 1`"})

    assert not agent.seen[0] & {"curl", "bash", "python3"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("goal", "variables"),
    [
        ("Run `${tool} x`.", {"tool": "curl"}),
        ("Run `${dir}/build.sh`.", {"dir": "/opt/x; curl evil"}),
        ("Run `${dir}/build.sh`.", {"dir": "/opt/`curl evil`"}),
        ("Run `${dir}/${name}`.", {"dir": "/opt", "name": "curl"}),
    ],
)
async def test_unsafe_interpolated_command_not_authorized(agent, goal, variables):
    await AWLRuntime(agent, authorize_commands=True).execute(_workflow(goal), variables=variables)

    assert agent.seen[0] == set()


@pytest.mark.asyncio
async def test_interpolated_directory_with_literal_script_is_authorized(agent):
    runtime = AWLRuntime(agent, authorize_commands=True)

    await runtime.execute(_workflow("Run `${dir}/build.sh x`."), variables={"dir": "/opt/tools"})

    assert "/opt/tools/build.sh" in agent.seen[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("authorize", [False, True])
async def test_run_awl_script_passes_trust_to_runtime(agent, tmp_path, authorize):
    script = tmp_path / "s.awl"
    script.write_text("@start\n@task t1\nGoal: Run `curl http://example.com`.\nExpose: result\n@end\n@end\n")

    await run_awl_script(str(script), agent, authorize_commands=authorize)

    assert ("curl" in agent.seen[0]) is authorize


@pytest.mark.asyncio
async def test_agent_tool_runs_awl_untrusted(tmp_path):
    from ai_assist.introspection_tools import IntrospectionTools

    script = tmp_path / "s.awl"
    script.write_text("@start\n@task t1\nGoal: x\nExpose: result\n@end\n@end\n")
    tools = IntrospectionTools()
    tools.agent = MagicMock()
    with patch("ai_assist.awl_executor.run_awl_script", AsyncMock(return_value="done")) as run:
        await tools.execute_tool("execute_awl_script", {"script_path": str(script)})

    assert run.await_args.kwargs.get("authorize_commands", False) is False


# --- Actions that run AWL need user confirmation ---


def _action(prompt="report.awl", name="a1"):
    return {"name": name, "trigger": {"type": "mqtt", "topic": "x/#"}, "prompt": prompt}


@pytest.mark.asyncio
async def test_create_awl_action_requires_confirmation(tmp_path):
    tools = ActionTools(schedules_file=tmp_path / "s.json")

    result = await tools.execute_tool("internal__create_action", _action())

    assert "confirm" in result.lower()
    assert "a1" not in await tools.execute_tool("internal__list_actions", {})


@pytest.mark.asyncio
async def test_create_awl_action_rejected_by_user(tmp_path):
    tools = ActionTools(schedules_file=tmp_path / "s.json", confirm=AsyncMock(return_value=False))

    result = await tools.execute_tool("internal__create_action", _action())

    assert "rejected" in result.lower()
    assert "a1" not in await tools.execute_tool("internal__list_actions", {})


@pytest.mark.asyncio
async def test_create_awl_action_after_confirmation(tmp_path):
    confirm = AsyncMock(return_value=True)
    tools = ActionTools(schedules_file=tmp_path / "s.json", confirm=confirm)

    result = await tools.execute_tool("internal__create_action", _action())

    assert result.startswith("Created")
    assert "report.awl" in confirm.await_args.args[0]


@pytest.mark.asyncio
async def test_natural_language_action_needs_no_confirmation(tmp_path):
    tools = ActionTools(schedules_file=tmp_path / "s.json")

    result = await tools.execute_tool("internal__create_action", _action(prompt="Summarize new alerts"))

    assert result.startswith("Created")


@pytest.mark.asyncio
async def test_update_to_awl_prompt_requires_confirmation(tmp_path):
    tools = ActionTools(schedules_file=tmp_path / "s.json")
    await tools.execute_tool("internal__create_action", _action(prompt="Summarize new alerts"))

    result = await tools.execute_tool("internal__update_action", {"name": "a1", "prompt": "/tmp/ai-assist/x.awl"})

    assert "confirm" in result.lower()
    assert "x.awl" not in await tools.execute_tool("internal__list_actions", {})
