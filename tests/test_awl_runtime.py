"""Tests for AWL runtime engine"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_assist.awl_ast import (
    FailNode,
    GoalNode,
    IfNode,
    LoopNode,
    NotifyNode,
    ReturnNode,
    SetNode,
    TaskNode,
    WaitNode,
    WhileNode,
    WorkflowNode,
)
from ai_assist.awl_runtime import (
    AWLRuntime,
    AWLRuntimeError,
    RuntimeLimits,
    _compute_input_variables,
    _extract_commands_from_workflow,
    _resolve_model_aliases,
    _validate_workflow_models,
    validate_workflow_variables,
)


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.query = AsyncMock()
    return agent


@pytest.fixture
def runtime(mock_agent):
    return AWLRuntime(mock_agent)


@pytest.mark.asyncio
async def test_execute_single_task(mock_agent, runtime):
    mock_agent.query.return_value = 'Found the server.\n```json\n{"server_file": "cmd/server/main.go"}\n```'
    workflow = WorkflowNode(
        body=[
            TaskNode(
                task_id="find_server",
                goal="Find the HTTP server.",
                success="Identify the file.",
                expose=["server_file"],
            ),
        ]
    )
    result = await runtime.execute(workflow)
    assert result.success is True
    assert result.variables["server_file"] == "cmd/server/main.go"
    assert len(result.task_outcomes) == 1
    assert result.task_outcomes[0].status == "success"


@pytest.mark.asyncio
async def test_execute_task_with_return(mock_agent, runtime):
    mock_agent.query.return_value = '```json\n{"server_file": "main.go"}\n```'
    workflow = WorkflowNode(
        body=[
            TaskNode(task_id="t1", goal="Find server.", expose=["server_file"]),
            ReturnNode(expression="server_file"),
        ]
    )
    result = await runtime.execute(workflow)
    assert result.return_value == "main.go"


@pytest.mark.asyncio
async def test_execute_set_and_interpolation(mock_agent, runtime):
    mock_agent.query.return_value = '```json\n{"target_file": "main.go"}\n```'
    workflow = WorkflowNode(
        body=[
            SetNode(variable="target", value="HTTP server"),
            TaskNode(task_id="find", goal="Find ${target}.", expose=["target_file"]),
        ]
    )
    result = await runtime.execute(workflow)
    assert result.variables["target"] == "HTTP server"
    assert "HTTP server" in mock_agent.query.call_args[1].get("prompt", mock_agent.query.call_args[0][0])


@pytest.mark.asyncio
async def test_execute_if_true_branch(mock_agent, runtime):
    mock_agent.query.return_value = '```json\n{"analysis": "done"}\n```'
    workflow = WorkflowNode(
        body=[
            IfNode(
                expression="handlers",
                then_body=[TaskNode(task_id="analyze", goal="Analyze.", expose=["analysis"])],
                else_body=[TaskNode(task_id="search", goal="Search.", expose=["found"])],
            ),
        ]
    )
    result = await runtime.execute(workflow, variables={"handlers": ["a", "b"]})
    assert result.variables.get("analysis") == "done"
    assert "found" not in result.variables


@pytest.mark.asyncio
async def test_execute_if_false_branch(mock_agent, runtime):
    mock_agent.query.return_value = '```json\n{"found": "yes"}\n```'
    workflow = WorkflowNode(
        body=[
            IfNode(
                expression="handlers",
                then_body=[TaskNode(task_id="analyze", goal="Analyze.", expose=["analysis"])],
                else_body=[TaskNode(task_id="search", goal="Search.", expose=["found"])],
            ),
        ]
    )
    result = await runtime.execute(workflow, variables={"handlers": []})
    assert result.variables.get("found") == "yes"
    assert "analysis" not in result.variables


@pytest.mark.asyncio
async def test_execute_loop(mock_agent, runtime):
    call_count = 0

    async def mock_query(prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        return f'{{"handler_summary": "summary_{call_count}"}}'

    mock_agent.query = AsyncMock(side_effect=mock_query)
    workflow = WorkflowNode(
        body=[
            LoopNode(
                collection="handlers",
                item_var="handler",
                limit=None,
                body=[TaskNode(task_id="inspect", goal="Inspect ${handler}.", expose=["handler_summary"])],
            ),
        ]
    )
    await runtime.execute(workflow, variables={"handlers": ["h1", "h2", "h3"]})
    assert mock_agent.query.call_count == 3


@pytest.mark.asyncio
async def test_execute_loop_with_limit(mock_agent, runtime):
    mock_agent.query.return_value = '{"handler_summary": "done"}'
    workflow = WorkflowNode(
        body=[
            LoopNode(
                collection="handlers",
                item_var="handler",
                limit=2,
                body=[TaskNode(task_id="inspect", goal="Inspect ${handler}.", expose=["handler_summary"])],
            ),
        ]
    )
    await runtime.execute(workflow, variables={"handlers": ["h1", "h2", "h3", "h4", "h5"]})
    assert mock_agent.query.call_count == 2


@pytest.mark.asyncio
async def test_execute_loop_collect(mock_agent, runtime):
    call_count = 0

    async def mock_query(prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        return f'{{"summary": "summary_{call_count}", "priority": "p{call_count}"}}'

    mock_agent.query = AsyncMock(side_effect=mock_query)
    workflow = WorkflowNode(
        body=[
            LoopNode(
                collection="items",
                item_var="item",
                collect="results",
                body=[TaskNode(task_id="analyze", goal="Analyze ${item}.", expose=["summary", "priority"])],
            ),
        ]
    )
    result = await runtime.execute(workflow, variables={"items": ["a", "b", "c"]})
    assert "results" in result.variables
    results = result.variables["results"]
    assert len(results) == 3
    assert results[0] == {"summary": "summary_1", "priority": "p1"}
    assert results[1] == {"summary": "summary_2", "priority": "p2"}
    assert results[2] == {"summary": "summary_3", "priority": "p3"}


@pytest.mark.asyncio
async def test_execute_loop_collect_with_limit(mock_agent, runtime):
    mock_agent.query.return_value = '{"summary": "done"}'
    workflow = WorkflowNode(
        body=[
            LoopNode(
                collection="items",
                item_var="item",
                limit=2,
                collect="results",
                body=[TaskNode(task_id="analyze", goal="Analyze ${item}.", expose=["summary"])],
            ),
        ]
    )
    result = await runtime.execute(workflow, variables={"items": ["a", "b", "c", "d"]})
    assert len(result.variables["results"]) == 2


@pytest.mark.asyncio
async def test_execute_loop_collect_fields(mock_agent, runtime):
    call_count = 0

    async def mock_query(prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        return f'{{"summary": "summary_{call_count}", "priority": "p{call_count}", "extra": "ignore"}}'

    mock_agent.query = AsyncMock(side_effect=mock_query)
    workflow = WorkflowNode(
        body=[
            LoopNode(
                collection="items",
                item_var="item",
                collect="results",
                collect_fields=["summary"],
                body=[TaskNode(task_id="analyze", goal="Analyze ${item}.", expose=["summary", "priority", "extra"])],
            ),
        ]
    )
    result = await runtime.execute(workflow, variables={"items": ["a", "b"]})
    results = result.variables["results"]
    assert len(results) == 2
    assert results[0] == {"summary": "summary_1"}
    assert results[1] == {"summary": "summary_2"}


@pytest.mark.asyncio
async def test_execute_loop_collect_skips_failed_tasks(mock_agent, runtime):
    call_count = 0

    async def mock_query(prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise Exception("API Error")
        return f'{{"summary": "ok_{call_count}"}}'

    mock_agent.query = AsyncMock(side_effect=mock_query)
    workflow = WorkflowNode(
        body=[
            LoopNode(
                collection="items",
                item_var="item",
                collect="results",
                body=[TaskNode(task_id="analyze", goal="Analyze ${item}.", expose=["summary"])],
            ),
        ]
    )
    result = await runtime.execute(workflow, variables={"items": ["a", "b", "c"]})
    results = result.variables["results"]
    assert len(results) == 2
    assert results[0] == {"summary": "ok_1"}
    assert results[1] == {"summary": "ok_3"}


@pytest.mark.asyncio
async def test_task_failure_marks_workflow_failed(mock_agent, runtime):
    mock_agent.query.side_effect = Exception("API Error")
    workflow = WorkflowNode(
        body=[TaskNode(task_id="failing", goal="This will fail.", expose=["result"])],
    )
    result = await runtime.execute(workflow)
    assert result.success is False
    assert result.task_outcomes[0].status == "failed"


@pytest.mark.asyncio
async def test_max_steps_exceeded(mock_agent):
    mock_agent.query.return_value = '{"r": "ok"}'
    runtime = AWLRuntime(mock_agent, limits=RuntimeLimits(max_steps=2))
    workflow = WorkflowNode(
        body=[
            TaskNode(task_id="t1", goal="Step 1.", expose=["r"]),
            TaskNode(task_id="t2", goal="Step 2.", expose=["r"]),
            TaskNode(task_id="t3", goal="Step 3.", expose=["r"]),
        ]
    )
    with pytest.raises(AWLRuntimeError, match="max_steps"):
        await runtime.execute(workflow)


@pytest.mark.asyncio
async def test_task_no_expose(mock_agent, runtime):
    mock_agent.query.return_value = "Task completed successfully."
    workflow = WorkflowNode(
        body=[TaskNode(task_id="simple", goal="Just do it.")],
    )
    result = await runtime.execute(workflow)
    assert result.success is True
    assert result.task_outcomes[0].status == "success"


@pytest.mark.asyncio
async def test_no_kg_hint(mock_agent, runtime):
    mock_agent.query.return_value = '{"result": "found"}'
    workflow = WorkflowNode(
        body=[TaskNode(task_id="t1", hints=["no-kg"], goal="Find something.", expose=["result"])],
    )
    await runtime.execute(workflow)
    prompt = mock_agent.query.call_args[0][0]
    assert prompt.startswith("@no-kg")


@pytest.mark.asyncio
async def test_if_with_none_comparison_no_crash(mock_agent, runtime, caplog):
    """@if count > 0 should not crash when count is undefined (None), but should warn."""
    mock_agent.query.return_value = '{"r": "ok"}'
    workflow = WorkflowNode(
        body=[
            IfNode(
                expression="count > 0",
                then_body=[TaskNode(task_id="t1", goal="Do.", expose=["r"])],
            ),
        ]
    )
    result = await runtime.execute(workflow)
    assert result.success is True
    assert mock_agent.query.call_count == 0
    assert "undefined variable(s): count" in caplog.text


@pytest.mark.asyncio
async def test_missing_exposed_var_defaults_to_none(mock_agent, runtime, capsys):
    """Missing exposed vars should default to None so @if evaluates to False without 'undefined' warning."""
    # Task exposes has_improvements but Claude's response doesn't include it
    mock_agent.query.return_value = '{"clusters_summary": "3 clusters found"}'
    workflow = WorkflowNode(
        body=[
            TaskNode(
                task_id="cluster",
                goal="Cluster results.",
                expose=["clusters_summary", "has_improvements"],
            ),
            IfNode(
                expression="has_improvements",
                then_body=[TaskNode(task_id="apply", goal="Apply changes.")],
                else_body=[TaskNode(task_id="report", goal="Report no changes.")],
            ),
        ]
    )
    result = await runtime.execute(workflow)
    assert result.success is True
    # The @if should take the else branch (has_improvements is None)
    # 3 calls: cluster + nudge (retry for missing has_improvements) + report (not apply)
    assert mock_agent.query.call_count == 3
    captured = capsys.readouterr()
    assert "missing exposed vars" in captured.out
    assert "has_improvements" in captured.out
    # Should NOT have an "undefined variable" warning
    assert "undefined variable" not in captured.out


@pytest.mark.asyncio
async def test_nudge_recovers_missing_exposed_vars(mock_agent, runtime, capsys):
    """Nudge should recover missing exposed vars on retry."""
    # First call returns partial vars, nudge call returns the missing one
    mock_agent.query.side_effect = [
        '{"clusters_summary": "3 clusters found"}',  # cluster task — missing has_improvements
        '{"has_improvements": true}',  # nudge retry — recovers it
        '{"files_changed": "log_map.yml"}',  # apply task (then branch)
    ]
    workflow = WorkflowNode(
        body=[
            TaskNode(
                task_id="cluster",
                goal="Cluster results.",
                expose=["clusters_summary", "has_improvements"],
            ),
            IfNode(
                expression="has_improvements",
                then_body=[TaskNode(task_id="apply", goal="Apply changes.", expose=["files_changed"])],
                else_body=[TaskNode(task_id="report", goal="Report no changes.")],
            ),
        ]
    )
    result = await runtime.execute(workflow)
    assert result.success is True
    # 3 calls: cluster + nudge + apply (then branch, not else)
    assert mock_agent.query.call_count == 3
    captured = capsys.readouterr()
    assert "nudge recovered" in captured.out
    assert "missing exposed vars" not in captured.out


@pytest.mark.asyncio
async def test_nudge_includes_response_as_context(mock_agent, runtime, capsys):
    """Nudge should include the task's response text so the agent can extract values."""
    call_count = 0

    async def track_query(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "The root cause is a stuck image pull on worker-2."
        elif call_count == 2:
            # Nudge: verify the prompt includes the original response
            prompt = args[0] if args else kwargs.get("prompt", "")
            assert "stuck image pull" in prompt
            assert "JSON block" in prompt
            return '{"partial": "data", "full_result": "stuck image pull"}'
        return "{}"

    mock_agent.query.side_effect = track_query
    workflow = WorkflowNode(
        body=[TaskNode(task_id="t1", goal="Do analysis.", expose=["partial", "full_result"])],
    )
    result = await runtime.execute(workflow)
    assert result.success is True
    assert call_count == 2
    captured = capsys.readouterr()
    assert "nudge recovered" in captured.out


@pytest.mark.asyncio
async def test_fail_directive_raises(runtime):
    workflow = WorkflowNode(body=[FailNode(message="Jira is down")])
    with pytest.raises(AWLRuntimeError, match="Jira is down"):
        await runtime.execute(workflow)


@pytest.mark.asyncio
async def test_fail_inside_if_aborts(runtime):
    workflow = WorkflowNode(
        body=[
            IfNode(
                expression="not jira_report",
                then_body=[FailNode(message="Jira unavailable")],
                else_body=[],
            )
        ]
    )
    with pytest.raises(AWLRuntimeError, match="Jira unavailable"):
        await runtime.execute(workflow)


@pytest.mark.asyncio
async def test_fail_not_reached_when_condition_false(runtime):
    workflow = WorkflowNode(
        body=[
            SetNode(variable="jira_report", value="report.jsonl"),
            IfNode(
                expression="not jira_report",
                then_body=[FailNode(message="should not fail")],
                else_body=[],
            ),
            ReturnNode(expression="jira_report"),
        ]
    )
    result = await runtime.execute(workflow)
    assert result.success is True
    assert result.return_value == "report.jsonl"


@pytest.mark.asyncio
async def test_if_with_out_of_range_index_no_crash(mock_agent, runtime):
    """@if handlers[99] should not crash when index is out of range."""
    mock_agent.query.return_value = '{"r": "ok"}'
    workflow = WorkflowNode(
        body=[
            IfNode(
                expression="handlers[99]",
                then_body=[TaskNode(task_id="t1", goal="Do.", expose=["r"])],
            ),
        ]
    )
    result = await runtime.execute(workflow, variables={"handlers": ["a"]})
    assert result.success is True
    assert mock_agent.query.call_count == 0


@pytest.mark.asyncio
async def test_loop_undefined_collection_warns(mock_agent, runtime, caplog):
    """@loop on undefined collection should warn."""
    workflow = WorkflowNode(
        body=[
            LoopNode(
                collection="missing_var",
                item_var="item",
                body=[TaskNode(task_id="t1", goal="Do.", expose=["r"])],
            ),
        ]
    )
    result = await runtime.execute(workflow)
    assert result.success is True
    assert mock_agent.query.call_count == 0  # Empty collection, no iterations
    assert "collection variable 'missing_var' is not defined" in caplog.text


@pytest.mark.asyncio
async def test_if_defined_variable_no_warning(mock_agent, runtime, caplog):
    """@if with a defined variable should not produce undefined variable warnings."""
    mock_agent.query.return_value = '{"r": "ok"}'
    workflow = WorkflowNode(
        body=[
            IfNode(
                expression="len(items) > 0",
                then_body=[TaskNode(task_id="t1", goal="Do.", expose=["r"])],
            ),
        ]
    )
    await runtime.execute(workflow, variables={"items": ["a"]})
    assert "undefined variable" not in caplog.text


@pytest.mark.asyncio
async def test_task_unresolved_interpolation_warns(mock_agent, runtime, caplog):
    """Task goal with ${var} referencing undefined variable should warn."""
    mock_agent.query.return_value = '{"new_issues": "none"}'
    workflow = WorkflowNode(
        body=[
            TaskNode(
                task_id="check",
                goal="Analyze ${results} for issues.",
                expose=["new_issues"],
            ),
        ]
    )
    await runtime.execute(workflow)
    assert "unresolved variable(s)" in caplog.text
    assert "results" in caplog.text


@pytest.mark.asyncio
async def test_task_resolved_interpolation_no_warning(mock_agent, runtime, caplog):
    """Task goal with ${var} referencing a defined variable should not warn."""
    mock_agent.query.return_value = '{"summary": "ok"}'
    workflow = WorkflowNode(
        body=[
            TaskNode(
                task_id="check",
                goal="Analyze ${data} for issues.",
                expose=["summary"],
            ),
        ]
    )
    await runtime.execute(workflow, variables={"data": [1, 2, 3]})
    assert "unresolved variable" not in caplog.text


@pytest.mark.asyncio
async def test_substituted_content_dollar_vars_no_warning(mock_agent, runtime, caplog):
    """Shell-style ${VAR} inside substituted content should not trigger warnings."""
    mock_agent.query.return_value = '{"status": "done"}'
    workflow = WorkflowNode(
        body=[
            TaskNode(
                task_id="deploy",
                goal="Write the script from ${script_content} to disk.",
                expose=["status"],
            ),
        ]
    )
    script = 'echo "version=${GL_VERSION}"\nsha="${GL_SHA256}"'
    await runtime.execute(workflow, variables={"script_content": script})
    assert "unresolved variable" not in caplog.text


@pytest.mark.asyncio
async def test_loop_with_json_string_variable(mock_agent, runtime):
    """Test that @loop parses JSON string variables into lists"""
    mock_agent.query = AsyncMock(return_value='{"summary": "done"}')
    workflow = WorkflowNode(
        body=[
            LoopNode(
                collection="jobs",
                item_var="job",
                body=[TaskNode(task_id="process", goal="Process ${job}.", expose=["summary"])],
            ),
        ]
    )
    # Pass jobs as a JSON string (as agent might expose it)
    result = await runtime.execute(workflow, variables={"jobs": '[{"id": "a"}, {"id": "b"}]'})
    assert result.success is True
    assert mock_agent.query.call_count == 2  # 2 items in the JSON array


@pytest.mark.asyncio
async def test_loop_with_single_dict_wraps_in_list(mock_agent, runtime):
    """Test that @loop wraps a single dict in a list"""
    mock_agent.query = AsyncMock(return_value='{"summary": "done"}')
    workflow = WorkflowNode(
        body=[
            LoopNode(
                collection="job",
                item_var="j",
                body=[TaskNode(task_id="process", goal="Process ${j}.", expose=["summary"])],
            ),
        ]
    )
    result = await runtime.execute(workflow, variables={"job": {"id": "abc", "name": "test"}})
    assert result.success is True
    assert mock_agent.query.call_count == 1  # Single dict wrapped in list = 1 iteration


@pytest.mark.asyncio
async def test_loop_with_python_repr_string(mock_agent, runtime):
    """Test that @loop parses Python-repr strings (single quotes)"""
    mock_agent.query = AsyncMock(return_value='{"summary": "done"}')
    workflow = WorkflowNode(
        body=[
            LoopNode(
                collection="jobs",
                item_var="job",
                body=[TaskNode(task_id="process", goal="Process ${job}.", expose=["summary"])],
            ),
        ]
    )
    result = await runtime.execute(workflow, variables={"jobs": "[{'id': 'a'}, {'id': 'b'}]"})
    assert result.success is True
    assert mock_agent.query.call_count == 2


# ── @goal directive runtime tests ────────────────────────────────


@pytest.mark.asyncio
async def test_execute_goal_runs_body(mock_agent, runtime):
    """Test that @goal executes its body tasks"""
    call_count = 0

    async def mock_query(prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return '```json\n{"failure_rate": 5}\n```'
        # Success evaluation
        return '```json\n{"success_met": false, "reason": "still monitoring"}\n```'

    mock_agent.query = AsyncMock(side_effect=mock_query)
    workflow = WorkflowNode(
        body=[
            GoalNode(
                goal_id="test_goal",
                success_criteria="Failure rate below 10%",
                body=[
                    TaskNode(task_id="check", goal="Check rate.", expose=["failure_rate"]),
                ],
            ),
        ]
    )
    result = await runtime.execute(workflow)
    assert result.success is True
    assert result.variables["failure_rate"] == 5
    assert result.variables.get("_goal_success_met") is False


@pytest.mark.asyncio
async def test_execute_goal_success_met(mock_agent, runtime):
    """Test that success evaluation sets _goal_success_met"""
    call_count = 0

    async def mock_query(prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return '```json\n{"failure_rate": 2}\n```'
        # Success evaluation
        return '```json\n{"success_met": true, "reason": "Rate is 2%, well below 10%"}\n```'

    mock_agent.query = AsyncMock(side_effect=mock_query)
    workflow = WorkflowNode(
        body=[
            GoalNode(
                goal_id="test_goal",
                success_criteria="Failure rate below 10%",
                body=[
                    TaskNode(task_id="check", goal="Check rate.", expose=["failure_rate"]),
                ],
            ),
        ]
    )
    result = await runtime.execute(workflow)
    assert result.variables["_goal_success_met"] is True
    assert "2%" in result.variables["_goal_success_reason"]


@pytest.mark.asyncio
async def test_execute_goal_max_actions_limit(mock_agent):
    """Test that max_actions limits tool calls within the goal body"""
    runtime = AWLRuntime(mock_agent, limits=RuntimeLimits(max_tool_calls=100))
    mock_agent.query = AsyncMock(return_value='```json\n{"r": "ok"}\n```')

    workflow = WorkflowNode(
        body=[
            GoalNode(
                goal_id="limited",
                success_criteria="Done",
                max_actions=2,
                body=[
                    TaskNode(task_id="t1", goal="Task 1.", expose=["r"]),
                ],
            ),
        ]
    )
    r = await runtime.execute(workflow)
    # Task should execute with max_turns=2 (from max_actions)
    assert r.success is True


@pytest.mark.asyncio
async def test_execute_goal_with_conditional_body(mock_agent, runtime):
    """Test @goal with @if inside body"""
    call_count = 0

    async def mock_query(prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return '```json\n{"failure_rate": 15}\n```'
        elif call_count == 2:
            return "Alert created."
        # Success evaluation
        return '```json\n{"success_met": false, "reason": "rate still high"}\n```'

    mock_agent.query = AsyncMock(side_effect=mock_query)
    workflow = WorkflowNode(
        body=[
            GoalNode(
                goal_id="test_cond",
                success_criteria="Failure rate below 10%",
                body=[
                    TaskNode(task_id="check", goal="Check rate.", expose=["failure_rate"]),
                    IfNode(
                        expression="failure_rate > 10",
                        then_body=[TaskNode(task_id="alert", goal="Create alert.")],
                    ),
                ],
            ),
        ]
    )
    await runtime.execute(workflow, variables={"failure_rate": 0})
    # Both tasks should have executed (check + alert)
    assert mock_agent.query.call_count == 3  # check + alert + success eval


# ── Task failure behavior tests ────────────────────────────────


@pytest.mark.asyncio
async def test_task_failure_stops_workflow_outside_loop(mock_agent, runtime):
    """Top-level task failure should stop the workflow; second task never runs."""
    call_count = 0

    async def mock_query(prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("First task failed")
        return '{"result": "ok"}'

    mock_agent.query = AsyncMock(side_effect=mock_query)
    workflow = WorkflowNode(
        body=[
            TaskNode(task_id="t1", goal="Fail.", expose=["result"]),
            TaskNode(task_id="t2", goal="Should not run.", expose=["result"]),
        ],
    )
    result = await runtime.execute(workflow)
    assert result.success is False
    assert len(result.task_outcomes) == 1
    assert result.task_outcomes[0].status == "failed"


@pytest.mark.asyncio
async def test_continue_on_failure_hint(mock_agent, runtime):
    """Task with @continue-on-failure hint should not stop the workflow."""
    call_count = 0

    async def mock_query(prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("First task failed")
        return '{"result": "ok"}'

    mock_agent.query = AsyncMock(side_effect=mock_query)
    workflow = WorkflowNode(
        body=[
            TaskNode(task_id="t1", hints=["continue-on-failure"], goal="Fail.", expose=["result"]),
            TaskNode(task_id="t2", goal="Should still run.", expose=["result"]),
        ],
    )
    result = await runtime.execute(workflow)
    assert result.success is False
    assert len(result.task_outcomes) == 2
    assert result.task_outcomes[0].status == "failed"
    assert result.task_outcomes[1].status == "success"


# ── AWL command auto-authorization tests ─────────────────────────


def test_extract_commands_from_workflow_backticks():
    """Commands in backticks in task goals should be extracted."""
    workflow = WorkflowNode(
        body=[
            TaskNode(
                task_id="t1",
                goal="Run `git -C /tmp/repo fetch origin master` to get latest.",
            ),
            TaskNode(
                task_id="t2",
                goal='Run `python3 -c "print(1)"` to compute.',
            ),
        ]
    )
    commands = _extract_commands_from_workflow(workflow)
    assert "git" in commands
    assert "python3" in commands


def test_extract_commands_from_workflow_full_path():
    """Full path commands in backticks should preserve the full path."""
    workflow = WorkflowNode(
        body=[
            TaskNode(
                task_id="t1",
                goal="Run `/home/user/scripts/process.sh arg1`.",
            ),
        ]
    )
    commands = _extract_commands_from_workflow(workflow)
    assert "/home/user/scripts/process.sh" in commands


def test_extract_commands_from_nested_structures():
    """Commands in loops and conditionals should be extracted."""
    workflow = WorkflowNode(
        body=[
            LoopNode(
                collection="items",
                item_var="item",
                body=[
                    IfNode(
                        expression="item",
                        then_body=[
                            TaskNode(task_id="t1", goal="Run `gh pr create`."),
                        ],
                    ),
                ],
            ),
        ]
    )
    commands = _extract_commands_from_workflow(workflow)
    assert "gh" in commands


def test_extract_commands_from_context():
    """Commands in task context should also be extracted."""
    workflow = WorkflowNode(
        body=[
            TaskNode(
                task_id="t1",
                goal="Do something.",
                context="Use `make build` to compile.",
            ),
        ]
    )
    commands = _extract_commands_from_workflow(workflow)
    assert "make" in commands


def test_extract_commands_empty_workflow():
    """Empty workflow should return no commands."""
    workflow = WorkflowNode(body=[])
    commands = _extract_commands_from_workflow(workflow)
    assert commands == set()


@pytest.mark.asyncio
async def test_task_max_tool_calls_override(mock_agent):
    """Task with max_tool_calls should override the default limit."""
    mock_agent.query.return_value = '{"result": "ok"}'

    runtime = AWLRuntime(mock_agent, limits=RuntimeLimits(max_tool_calls=100))
    workflow = WorkflowNode(
        body=[
            TaskNode(task_id="t1", goal="Heavy task.", max_tool_calls=200, expose=["result"]),
        ]
    )
    await runtime.execute(workflow)
    # Check that query was called with max_turns=200
    assert (
        mock_agent.query.call_args[1].get(
            "max_turns", mock_agent.query.call_args[0][1] if len(mock_agent.query.call_args[0]) > 1 else None
        )
        == 200
    )


@pytest.mark.asyncio
async def test_task_default_tool_calls(mock_agent):
    """Task without max_tool_calls should use the runtime default."""
    mock_agent.query.return_value = '{"result": "ok"}'

    runtime = AWLRuntime(mock_agent, limits=RuntimeLimits(max_tool_calls=100))
    workflow = WorkflowNode(
        body=[
            TaskNode(task_id="t1", goal="Normal task.", expose=["result"]),
        ]
    )
    await runtime.execute(workflow)
    _, kwargs = mock_agent.query.call_args
    assert kwargs.get("max_turns") == 100


@pytest.mark.asyncio
async def test_awl_auto_authorizes_commands(mock_agent):
    """AWL runtime should temporarily add script commands to allowed list."""
    # Set up filesystem_tools mock
    fs_tools = MagicMock()
    fs_tools.allowed_commands = ["grep", "find"]
    fs_tools.awl_authorized_commands = set()
    mock_agent.filesystem_tools = fs_tools
    mock_agent.query.return_value = '{"result": "ok"}'

    runtime = AWLRuntime(mock_agent)
    workflow = WorkflowNode(
        body=[
            TaskNode(task_id="t1", goal="Run `python3 -c 'print(1)'`.", expose=["result"]),
        ]
    )
    await runtime.execute(workflow)

    # After execution, temporarily added commands should be cleaned up
    assert "python3" not in fs_tools.allowed_commands
    assert len(fs_tools.awl_authorized_commands) == 0


@pytest.mark.asyncio
async def test_awl_auto_authorize_cleanup_on_failure(mock_agent):
    """Cleanup should happen even if the workflow fails."""
    fs_tools = MagicMock()
    fs_tools.allowed_commands = ["grep"]
    fs_tools.awl_authorized_commands = set()
    mock_agent.filesystem_tools = fs_tools
    mock_agent.query.side_effect = Exception("fail")

    runtime = AWLRuntime(mock_agent)
    workflow = WorkflowNode(
        body=[
            TaskNode(task_id="t1", goal="Run `custom_tool arg`."),
        ]
    )
    await runtime.execute(workflow)

    assert "custom_tool" not in fs_tools.allowed_commands
    assert len(fs_tools.awl_authorized_commands) == 0


# ── Static variable validation tests ─────────────────────────────


class TestValidateWorkflowVariables:
    def test_no_warnings_for_valid_workflow(self):
        """Well-formed workflow produces no warnings."""
        workflow = WorkflowNode(
            body=[
                TaskNode(task_id="t1", goal="Find items.", expose=["items"]),
                IfNode(
                    expression="len(items) > 0",
                    then_body=[TaskNode(task_id="t2", goal="Process ${items}.")],
                ),
            ]
        )
        assert validate_workflow_variables(workflow) == []

    def test_undefined_variable_in_if(self):
        """@if referencing undefined variable produces a warning."""
        workflow = WorkflowNode(
            body=[
                IfNode(
                    expression="count > 0",
                    then_body=[TaskNode(task_id="t1", goal="Do.")],
                ),
            ]
        )
        warnings = validate_workflow_variables(workflow)
        assert len(warnings) == 1
        assert "count" in warnings[0]

    def test_undefined_variable_in_task_goal(self):
        """Task goal referencing undefined ${var} produces a warning."""
        workflow = WorkflowNode(
            body=[
                TaskNode(task_id="t1", goal="Analyze ${results}."),
            ]
        )
        warnings = validate_workflow_variables(workflow)
        assert len(warnings) == 1
        assert "results" in warnings[0]

    def test_loop_collect_not_available_inside_body(self):
        """collect variable should NOT be available inside the loop body."""
        workflow = WorkflowNode(
            body=[
                TaskNode(task_id="t1", goal="Find jobs.", expose=["jobs"]),
                LoopNode(
                    collection="jobs",
                    item_var="job",
                    collect="analyses",
                    body=[
                        TaskNode(task_id="t2", goal="Analyze ${job}.", expose=["result"]),
                        TaskNode(task_id="t3", goal="Check ${analyses}."),
                    ],
                ),
            ]
        )
        warnings = validate_workflow_variables(workflow)
        assert any("analyses" in w for w in warnings)

    def test_loop_collect_available_after_loop(self):
        """collect variable should be available after the loop."""
        workflow = WorkflowNode(
            body=[
                TaskNode(task_id="t1", goal="Find jobs.", expose=["jobs"]),
                LoopNode(
                    collection="jobs",
                    item_var="job",
                    collect="analyses",
                    body=[TaskNode(task_id="t2", goal="Analyze ${job}.", expose=["result"])],
                ),
                TaskNode(task_id="t3", goal="Report on ${analyses}."),
            ]
        )
        assert validate_workflow_variables(workflow) == []

    def test_undefined_loop_collection(self):
        """@loop referencing undefined collection produces a warning."""
        workflow = WorkflowNode(
            body=[
                LoopNode(
                    collection="missing",
                    item_var="item",
                    body=[TaskNode(task_id="t1", goal="Do.")],
                ),
            ]
        )
        warnings = validate_workflow_variables(workflow)
        assert any("missing" in w for w in warnings)

    def test_initial_variables_suppress_warnings(self):
        """Variables passed as initial should not trigger warnings."""
        workflow = WorkflowNode(
            body=[
                TaskNode(task_id="t1", goal="Use ${name}."),
            ]
        )
        assert validate_workflow_variables(workflow, {"name"}) == []

    def test_set_defines_variable(self):
        """@set should define variable for subsequent nodes."""
        workflow = WorkflowNode(
            body=[
                SetNode(variable="target", value="server"),
                TaskNode(task_id="t1", goal="Find ${target}."),
            ]
        )
        assert validate_workflow_variables(workflow) == []

    def test_expose_defines_variable(self):
        """Expose: from a task should define variables for subsequent nodes."""
        workflow = WorkflowNode(
            body=[
                TaskNode(task_id="t1", goal="Find.", expose=["server_file"]),
                TaskNode(task_id="t2", goal="Read ${server_file}."),
            ]
        )
        assert validate_workflow_variables(workflow) == []

    def test_input_variables_excluded_when_passed(self):
        """Input variables (used but never produced) should not warn when passed as initial."""
        workflow = WorkflowNode(
            body=[
                TaskNode(task_id="t1", goal="Process ${subject} for ${quarter}."),
            ]
        )
        # Without initial_variables, these are flagged
        warnings = validate_workflow_variables(workflow)
        assert len(warnings) == 2

        # With computed input variables, no warnings
        input_vars = _compute_input_variables(workflow)
        assert input_vars == {"subject", "quarter"}
        assert validate_workflow_variables(workflow, input_vars) == []

    def test_scoping_bug_still_caught_with_input_vars(self):
        """Scoping bugs (collect used inside loop) should still be caught even with input var exclusion."""
        workflow = WorkflowNode(
            body=[
                TaskNode(task_id="t1", goal="Find ${subject}.", expose=["jobs"]),
                LoopNode(
                    collection="jobs",
                    item_var="job",
                    collect="analyses",
                    body=[
                        TaskNode(task_id="t2", goal="Analyze ${job}.", expose=["result"]),
                        TaskNode(task_id="t3", goal="Check ${analyses}."),
                    ],
                ),
            ]
        )
        input_vars = _compute_input_variables(workflow)
        warnings = validate_workflow_variables(workflow, input_vars)
        # subject is an input var and should NOT be flagged
        assert not any("subject" in w for w in warnings)
        # analyses is a scoping bug and SHOULD be flagged
        assert any("analyses" in w for w in warnings)

    def test_loop_item_var_available_in_body(self):
        """@loop item variable should be available inside the body."""
        workflow = WorkflowNode(
            body=[
                SetNode(variable="items", value="list"),
                LoopNode(
                    collection="items",
                    item_var="item",
                    body=[TaskNode(task_id="t1", goal="Process ${item}.")],
                ),
            ]
        )
        assert validate_workflow_variables(workflow) == []

    def test_while_undefined_variable_warns(self):
        """@while expression with undefined variable should warn."""
        workflow = WorkflowNode(
            body=[
                WhileNode(
                    expression="missing > 0",
                    body=[SetNode(variable="x", value="1")],
                ),
            ]
        )
        warnings = validate_workflow_variables(workflow)
        assert any("missing" in w for w in warnings)

    def test_notify_undefined_variable_warns(self):
        """@notify with undefined ${var} should warn."""
        workflow = WorkflowNode(
            body=[
                NotifyNode(message="${undefined_var} items"),
            ]
        )
        warnings = validate_workflow_variables(workflow)
        assert any("undefined_var" in w for w in warnings)


# ── _extract_exposed unit tests ──────────────────────────────────


class TestExtractExposed:
    """Tests for _extract_exposed accumulation across multiple JSON blocks."""

    def _make_runtime(self, mock_agent):
        return AWLRuntime(mock_agent)

    def test_single_block_all_vars(self, mock_agent):
        """Single JSON block containing all vars returns them all."""
        rt = self._make_runtime(mock_agent)
        response = '```json\n{"a": 1, "b": true}\n```'
        result = rt._extract_exposed(response, ["a", "b"])
        assert result == {"a": 1, "b": True}

    def test_split_across_two_blocks(self, mock_agent):
        """Vars split across two JSON blocks should all be found."""
        rt = self._make_runtime(mock_agent)
        response = (
            'Here is the summary:\n```json\n{"clusters_summary": "3 clusters"}\n```\n'
            'And the flag:\n```json\n{"has_improvements": true}\n```'
        )
        result = rt._extract_exposed(response, ["clusters_summary", "has_improvements"])
        assert result == {"clusters_summary": "3 clusters", "has_improvements": True}

    def test_first_block_wins_on_conflict(self, mock_agent):
        """When same key appears in multiple blocks, first value wins."""
        rt = self._make_runtime(mock_agent)
        response = '```json\n{"x": "first"}\n```\n' '```json\n{"x": "second", "y": 2}\n```'
        result = rt._extract_exposed(response, ["x", "y"])
        assert result == {"x": "first", "y": 2}

    def test_partial_match_not_returned_early(self, mock_agent):
        """A block with only some vars should not cause early return."""
        rt = self._make_runtime(mock_agent)
        response = '```json\n{"a": 1}\n```\n' "Some text in between.\n" '```json\n{"b": 2, "c": 3}\n```'
        result = rt._extract_exposed(response, ["a", "b", "c"])
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_inline_json_fallback(self, mock_agent):
        """Inline JSON (no fences) should also be found."""
        rt = self._make_runtime(mock_agent)
        response = 'The result is {"flag": false} and that is final.'
        result = rt._extract_exposed(response, ["flag"])
        assert result == {"flag": False}

    def test_empty_expose_returns_empty(self, mock_agent):
        """No expose vars means empty result."""
        rt = self._make_runtime(mock_agent)
        result = rt._extract_exposed('{"a": 1}', [])
        assert result == {}


# ── @wait runtime tests ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_wait_calls_sleep(mock_agent, runtime):
    """@wait should call asyncio.sleep with the correct duration."""
    workflow = WorkflowNode(body=[WaitNode(duration_seconds=300)])
    with patch("ai_assist.awl_runtime.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await runtime.execute(workflow)
        mock_sleep.assert_called_once_with(300)


# ── @while runtime tests ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_while_loops_until_false(mock_agent, runtime):
    """@while should loop until expression is falsy."""
    call_count = 0

    async def mock_query(prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        count = max(0, 3 - call_count)
        return f'{{"running_count": {count}}}'

    mock_agent.query = AsyncMock(side_effect=mock_query)
    workflow = WorkflowNode(
        body=[
            SetNode(variable="running_count", value="3"),
            WhileNode(
                expression="running_count > 0",
                body=[
                    TaskNode(task_id="check", goal="Check.", expose=["running_count"]),
                ],
            ),
        ]
    )
    result = await runtime.execute(workflow)
    assert result.success is True
    assert mock_agent.query.call_count == 3


@pytest.mark.asyncio
async def test_while_respects_max_iterations(mock_agent, runtime):
    """@while should stop at max_iterations even if condition is still true."""
    mock_agent.query.return_value = '{"r": "ok"}'
    workflow = WorkflowNode(
        body=[
            SetNode(variable="always_true", value="yes"),
            WhileNode(
                expression="always_true",
                max_iterations=3,
                body=[
                    TaskNode(task_id="t1", goal="Do.", expose=["r"]),
                ],
            ),
        ]
    )
    await runtime.execute(workflow)
    assert mock_agent.query.call_count == 3


@pytest.mark.asyncio
async def test_while_falsy_from_start(mock_agent, runtime):
    """@while with falsy condition should never enter body."""
    mock_agent.query.return_value = '{"r": "ok"}'
    workflow = WorkflowNode(
        body=[
            SetNode(variable="count", value="0"),
            WhileNode(
                expression="count > 0",
                body=[TaskNode(task_id="t1", goal="Do.", expose=["r"])],
            ),
        ]
    )
    await runtime.execute(workflow)
    assert mock_agent.query.call_count == 0


@pytest.mark.asyncio
async def test_while_tasks_dont_count_max_steps(mock_agent):
    """Tasks inside @while should not count against max_steps."""
    call_count = 0

    async def mock_query(prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        count = max(0, 5 - call_count)
        return f'{{"running_count": {count}}}'

    mock_agent.query = AsyncMock(side_effect=mock_query)
    runtime_obj = AWLRuntime(mock_agent, limits=RuntimeLimits(max_steps=2))
    workflow = WorkflowNode(
        body=[
            SetNode(variable="running_count", value="5"),
            WhileNode(
                expression="running_count > 0",
                max_iterations=10,
                body=[
                    TaskNode(task_id="t1", goal="Check.", expose=["running_count"]),
                ],
            ),
        ]
    )
    # Should NOT raise max_steps error - while uses _loop_depth
    result = await runtime_obj.execute(workflow)
    assert result.success is True
    assert mock_agent.query.call_count == 5


# ── @notify runtime tests ────────────────────────────────────────


@pytest.mark.asyncio
async def test_notify_interpolates_and_prints(mock_agent, runtime, capsys):
    """@notify should interpolate variables and print."""
    workflow = WorkflowNode(
        body=[
            SetNode(variable="count", value="5"),
            NotifyNode(message="${count} jobs still running"),
        ]
    )
    await runtime.execute(workflow)
    captured = capsys.readouterr()
    assert "5 jobs still running" in captured.out


# ── model= override tests ────────────────────────────────────────


def test_validate_workflow_models_valid():
    """Known model prefixes should pass validation."""
    workflow = WorkflowNode(body=[TaskNode(task_id="t1", goal="Do.", model="claude-opus-4-6-20260205")])
    errors = _validate_workflow_models(workflow, ["claude-opus-4-6", "claude-sonnet-4-6"])
    assert errors == []


def test_validate_workflow_models_invalid():
    """Unknown model should produce an error."""
    workflow = WorkflowNode(body=[TaskNode(task_id="t1", goal="Do.", model="gpt-4o")])
    errors = _validate_workflow_models(workflow, ["claude-opus-4-6", "claude-sonnet-4-6"])
    assert len(errors) == 1
    assert "gpt-4o" in errors[0]
    assert "t1" in errors[0]


def test_validate_workflow_models_no_override():
    """Tasks without model= should not produce errors."""
    workflow = WorkflowNode(body=[TaskNode(task_id="t1", goal="Do.")])
    errors = _validate_workflow_models(workflow, ["claude-opus-4-6"])
    assert errors == []


def test_validate_workflow_models_nested():
    """Validation should walk into if/loop/while bodies."""
    workflow = WorkflowNode(
        body=[
            IfNode(
                expression="True",
                then_body=[TaskNode(task_id="t1", goal="Do.", model="bad-model")],
            )
        ]
    )
    errors = _validate_workflow_models(workflow, ["claude-opus-4-6"])
    assert len(errors) == 1
    assert "bad-model" in errors[0]


@pytest.mark.asyncio
async def test_task_model_override(mock_agent):
    """Task with model= should temporarily swap the agent's config.model."""
    mock_agent.query.return_value = '{"result": "ok"}'
    mock_agent.config.model = "claude-sonnet-4-6"

    # Bypass model validation by not setting MODEL_CONTEXT_WINDOWS on the mock
    runtime = AWLRuntime(mock_agent, limits=RuntimeLimits(max_tool_calls=100))
    workflow = WorkflowNode(body=[TaskNode(task_id="t1", goal="Do.", model="claude-opus-4-6", expose=["result"])])
    await runtime.execute(workflow)

    # Model should be restored after task
    assert mock_agent.config.model == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_task_model_override_restored_on_failure(mock_agent):
    """Model should be restored even when the task raises."""
    mock_agent.query.side_effect = Exception("boom")
    mock_agent.config.model = "claude-sonnet-4-6"

    runtime = AWLRuntime(mock_agent, limits=RuntimeLimits(max_tool_calls=100))
    workflow = WorkflowNode(
        body=[
            TaskNode(
                task_id="t1",
                goal="Do.",
                model="claude-opus-4-6",
                expose=["result"],
                hints=["continue-on-failure"],
            )
        ]
    )
    await runtime.execute(workflow)
    assert mock_agent.config.model == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_task_model_validation_rejects_unknown(mock_agent):
    """Workflow should fail before execution if a task has an unknown model."""
    mock_agent.query.return_value = '{"result": "ok"}'
    mock_agent.config.model = "claude-sonnet-4-6"
    # Set MODEL_CONTEXT_WINDOWS on the mock's class to trigger validation
    type(mock_agent).MODEL_CONTEXT_WINDOWS = {"claude-opus-4-6": 1000000, "claude-sonnet-4-6": 1000000}

    runtime = AWLRuntime(mock_agent, limits=RuntimeLimits(max_tool_calls=100))
    workflow = WorkflowNode(body=[TaskNode(task_id="t1", goal="Do.", model="bad-model", expose=["result"])])
    with pytest.raises(AWLRuntimeError, match="Model validation failed"):
        await runtime.execute(workflow)

    # query should never have been called
    mock_agent.query.assert_not_called()

    # Clean up class attribute
    del type(mock_agent).MODEL_CONTEXT_WINDOWS


def test_resolve_model_aliases():
    """Short aliases should be resolved to full model names."""
    workflow = WorkflowNode(
        body=[
            TaskNode(task_id="t1", goal="Do.", model="haiku"),
            TaskNode(task_id="t2", goal="Do.", model="sonnet"),
            TaskNode(task_id="t3", goal="Do.", model="opus"),
        ]
    )
    _resolve_model_aliases(workflow)
    assert workflow.body[0].model == "claude-haiku-4-5"
    assert workflow.body[1].model == "claude-sonnet-4-6"
    assert workflow.body[2].model == "claude-opus-4-6"


def test_resolve_model_aliases_leaves_full_names():
    """Full model names should pass through unchanged."""
    workflow = WorkflowNode(body=[TaskNode(task_id="t1", goal="Do.", model="claude-opus-4-6-20260205")])
    _resolve_model_aliases(workflow)
    assert workflow.body[0].model == "claude-opus-4-6-20260205"


def test_resolve_model_aliases_nested():
    """Aliases in nested structures should be resolved."""
    workflow = WorkflowNode(
        body=[
            LoopNode(
                collection="items",
                item_var="item",
                body=[TaskNode(task_id="t1", goal="Do.", model="haiku")],
            )
        ]
    )
    _resolve_model_aliases(workflow)
    assert workflow.body[0].body[0].model == "claude-haiku-4-5"
