"""Tests for AWL parser"""

import pytest

from ai_assist.awl_ast import (
    BreakNode,
    ContinueNode,
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
from ai_assist.awl_parser import AWLParser, ParseError


def test_parse_empty_workflow():
    script = "@start\n@end"
    result = AWLParser(script).parse()
    assert isinstance(result, WorkflowNode)
    assert result.body == []


def test_parse_single_task():
    script = """\
@start

@task find_server
Goal: Find where the HTTP server is initialized.
Success: Identify the file and function responsible for startup.
Expose: server_file, server_function
@end

@end"""
    result = AWLParser(script).parse()
    assert len(result.body) == 1
    task = result.body[0]
    assert isinstance(task, TaskNode)
    assert task.task_id == "find_server"
    assert task.goal == "Find where the HTTP server is initialized."
    assert task.success == "Identify the file and function responsible for startup."
    assert task.expose == ["server_file", "server_function"]
    assert task.hints == []


def test_parse_task_with_all_fields():
    script = """\
@start

@task locate_config
Goal: Find where the server configuration is loaded.
Context: Focus on startup code and configuration parsing.
Constraints: Only look in src/ directory.
Success: Identify the entrypoint and parser.
Expose: config_entrypoint, config_parser
@end

@end"""
    result = AWLParser(script).parse()
    task = result.body[0]
    assert isinstance(task, TaskNode)
    assert task.context == "Focus on startup code and configuration parsing."
    assert task.constraints == "Only look in src/ directory."


def test_parse_task_with_hints():
    script = """\
@start

@task fresh_search @no-history @no-kg
Goal: Re-evaluate where the HTTP server is initialized.
@end

@end"""
    result = AWLParser(script).parse()
    task = result.body[0]
    assert isinstance(task, TaskNode)
    assert task.task_id == "fresh_search"
    assert task.hints == ["no-history", "no-kg"]


def test_parse_return():
    script = """\
@start

@task t1
Goal: Do something.
Expose: result
@end

@return result

@end"""
    result = AWLParser(script).parse()
    assert len(result.body) == 2
    ret = result.body[1]
    assert isinstance(ret, ReturnNode)
    assert ret.expression == "result"


def test_parse_set():
    script = """\
@start

@set target = "HTTP server"

@task find_target
Goal: Find ${target}.
@end

@end"""
    result = AWLParser(script).parse()
    assert len(result.body) == 2
    set_node = result.body[0]
    assert isinstance(set_node, SetNode)
    assert set_node.variable == "target"
    assert set_node.value == "HTTP server"


def test_parse_if_else():
    script = """\
@start

@if handlers

@task analyze
Goal: Analyze handlers.
@end

@else

@task search
Goal: Search for handlers.
@end

@end

@end"""
    result = AWLParser(script).parse()
    assert len(result.body) == 1
    if_node = result.body[0]
    assert isinstance(if_node, IfNode)
    assert if_node.expression == "handlers"
    assert len(if_node.then_body) == 1
    assert isinstance(if_node.then_body[0], TaskNode)
    assert len(if_node.else_body) == 1
    assert isinstance(if_node.else_body[0], TaskNode)


def test_parse_if_without_else():
    script = """\
@start

@if handlers

@task analyze
Goal: Analyze handlers.
@end

@end

@end"""
    result = AWLParser(script).parse()
    if_node = result.body[0]
    assert isinstance(if_node, IfNode)
    assert len(if_node.then_body) == 1
    assert if_node.else_body == []


def test_parse_loop():
    script = """\
@start

@loop handlers as handler

@task inspect
Goal: Inspect ${handler}.
@end

@end

@end"""
    result = AWLParser(script).parse()
    loop = result.body[0]
    assert isinstance(loop, LoopNode)
    assert loop.collection == "handlers"
    assert loop.item_var == "handler"
    assert loop.limit is None
    assert len(loop.body) == 1


def test_parse_loop_with_limit():
    script = """\
@start

@loop handlers as handler limit=5

@task inspect
Goal: Inspect ${handler}.
@end

@end

@end"""
    result = AWLParser(script).parse()
    loop = result.body[0]
    assert isinstance(loop, LoopNode)
    assert loop.limit == 5


def test_parse_loop_with_collect():
    script = """\
@start

@loop handlers as handler collect=results

@task inspect
Goal: Inspect ${handler}.
Expose: summary
@end

@end

@end"""
    result = AWLParser(script).parse()
    loop = result.body[0]
    assert isinstance(loop, LoopNode)
    assert loop.collect == "results"


def test_parse_loop_with_collect_fields():
    script = """\
@start

@loop handlers as handler collect=results(summary)

@task inspect
Goal: Inspect ${handler}.
Expose: summary
@end

@end

@end"""
    result = AWLParser(script).parse()
    loop = result.body[0]
    assert isinstance(loop, LoopNode)
    assert loop.collect == "results"
    assert loop.collect_fields == ["summary"]


def test_parse_loop_with_collect_multiple_fields():
    script = """\
@start

@loop handlers as handler collect=results(summary, priority)

@task inspect
Goal: Inspect ${handler}.
Expose: summary, priority
@end

@end

@end"""
    result = AWLParser(script).parse()
    loop = result.body[0]
    assert isinstance(loop, LoopNode)
    assert loop.collect == "results"
    assert loop.collect_fields == ["summary", "priority"]


def test_parse_loop_with_limit_and_collect():
    script = """\
@start

@loop handlers as handler limit=5 collect=reports

@task inspect
Goal: Inspect ${handler}.
@end

@end

@end"""
    result = AWLParser(script).parse()
    loop = result.body[0]
    assert isinstance(loop, LoopNode)
    assert loop.limit == 5
    assert loop.collect == "reports"


def test_parse_full_example():
    script = """\
@start

@task find_handlers @no-kg
Goal: Find HTTP handlers defined in the repository.
Expose: handlers
@end

@if len(handlers) > 0

@loop handlers as handler limit=5

@task inspect_handler @no-history
Goal: Understand what ${handler} does.
Expose: handler_summary
@end

@end

@else

@task fallback_search
Goal: Search more broadly for request entry points.
Expose: handlers
@end

@end

@return handlers

@end"""
    result = AWLParser(script).parse()
    assert len(result.body) == 3
    assert isinstance(result.body[0], TaskNode)
    assert isinstance(result.body[1], IfNode)
    assert isinstance(result.body[2], ReturnNode)

    if_node = result.body[1]
    assert if_node.expression == "len(handlers) > 0"
    assert len(if_node.then_body) == 1
    assert isinstance(if_node.then_body[0], LoopNode)
    assert len(if_node.else_body) == 1
    assert isinstance(if_node.else_body[0], TaskNode)


def test_parse_comments_ignored():
    script = """\
# This is a workflow
@start

# Find the server
@task find_server
Goal: Find the HTTP server.  # inline comment
Expose: server_file
@end

@end"""
    result = AWLParser(script).parse()
    assert len(result.body) == 1
    task = result.body[0]
    assert task.task_id == "find_server"
    assert "# inline comment" not in task.goal


def test_parse_error_missing_start():
    with pytest.raises(ParseError) as exc_info:
        AWLParser("@task t1\nGoal: Do.\n@end").parse()
    assert exc_info.value.line >= 1


def test_parse_error_missing_end():
    with pytest.raises(ParseError):
        AWLParser("@start\n@task t1\nGoal: Do.\n@end").parse()


def test_parse_error_unknown_directive():
    with pytest.raises(ParseError):
        AWLParser("@start\n@unknown\n@end").parse()


def test_parse_error_task_without_goal():
    with pytest.raises(ParseError):
        AWLParser("@start\n@task t1\n@end\n@end").parse()


def test_parse_error_loop_without_as():
    with pytest.raises(ParseError):
        AWLParser("@start\n@loop handlers\n@end\n@end").parse()


def test_parse_error_set_without_equals():
    with pytest.raises(ParseError):
        AWLParser("@start\n@set target\n@end").parse()


def test_parse_error_task_no_id():
    with pytest.raises(ParseError):
        AWLParser("@start\n@task\nGoal: Do.\n@end\n@end").parse()


def test_parse_error_unknown_hint():
    with pytest.raises(ParseError):
        AWLParser("@start\n@task t1 @bogus\nGoal: Do.\n@end\n@end").parse()


def test_parse_error_duplicate_task_id():
    with pytest.raises(ParseError):
        AWLParser("@start\n@task t1\nGoal: Do.\n@end\n@task t1\nGoal: Again.\n@end\n@end").parse()


def test_parse_error_if_empty_expression():
    with pytest.raises(ParseError):
        AWLParser("@start\n@if\n@end\n@end").parse()


def test_parse_error_return_empty_expression():
    with pytest.raises(ParseError):
        AWLParser("@start\n@return\n@end").parse()


def test_parse_error_invalid_if_expression():
    with pytest.raises(ParseError):
        AWLParser("@start\n@if !!!\n@end\n@end").parse()


def test_parse_error_invalid_return_expression():
    with pytest.raises(ParseError):
        AWLParser("@start\n@return len(\n@end").parse()


def test_parse_error_trailing_content_after_end():
    with pytest.raises(ParseError):
        AWLParser("@start\n@task t1\nGoal: Do.\n@end\n@end\n@task t2\nGoal: Extra.\n@end").parse()


def test_valid_hints_accepted():
    script = "@start\n@task t1 @no-history @no-kg\nGoal: Do.\n@end\n@end"
    result = AWLParser(script).parse()
    task = result.body[0]
    assert isinstance(task, TaskNode)
    assert task.hints == ["no-history", "no-kg"]


def test_parse_task_with_max_tool_calls():
    script = "@start\n@task t1 @no-kg max_tool_calls=200\nGoal: Do heavy analysis.\n@end\n@end"
    result = AWLParser(script).parse()
    task = result.body[0]
    assert isinstance(task, TaskNode)
    assert task.max_tool_calls == 200
    assert task.hints == ["no-kg"]


def test_parse_task_with_max_time():
    script = "@start\n@task t1 max_time=1800\nGoal: Long running task.\n@end\n@end"
    result = AWLParser(script).parse()
    task = result.body[0]
    assert isinstance(task, TaskNode)
    assert task.max_time == 1800


def test_parse_task_with_max_tool_calls_and_max_time():
    script = "@start\n@task t1 @no-kg max_tool_calls=200 max_time=1200\nGoal: Do.\n@end\n@end"
    result = AWLParser(script).parse()
    task = result.body[0]
    assert task.max_tool_calls == 200
    assert task.max_time == 1200
    assert task.hints == ["no-kg"]


def test_parse_task_without_max_tool_calls():
    script = "@start\n@task t1\nGoal: Simple task.\n@end\n@end"
    result = AWLParser(script).parse()
    task = result.body[0]
    assert isinstance(task, TaskNode)
    assert task.max_tool_calls is None


def test_parse_task_with_model():
    script = "@start\n@task t1 model=claude-opus-4-6\nGoal: Deep analysis.\n@end\n@end"
    result = AWLParser(script).parse()
    task = result.body[0]
    assert isinstance(task, TaskNode)
    assert task.model == "claude-opus-4-6"


def test_parse_task_with_model_and_hints():
    script = (
        "@start\n@task t1 @no-kg model=claude-haiku-4-5-20251001 max_tool_calls=50\nGoal: Quick search.\n@end\n@end"
    )
    result = AWLParser(script).parse()
    task = result.body[0]
    assert isinstance(task, TaskNode)
    assert task.model == "claude-haiku-4-5-20251001"
    assert task.max_tool_calls == 50
    assert task.hints == ["no-kg"]


def test_parse_task_without_model():
    script = "@start\n@task t1\nGoal: Normal task.\n@end\n@end"
    result = AWLParser(script).parse()
    task = result.body[0]
    assert isinstance(task, TaskNode)
    assert task.model is None


def test_parse_fail_basic():
    script = "@start\n@fail Jira is down\n@end\n"
    result = AWLParser(script).parse()
    assert len(result.body) == 1
    assert isinstance(result.body[0], FailNode)
    assert result.body[0].message == "Jira is down"


def test_parse_fail_inside_if():
    script = "@start\n@if not jira_report\n@fail Jira unavailable\n@end\n@end\n"
    result = AWLParser(script).parse()
    assert len(result.body) == 1
    node = result.body[0]
    assert isinstance(node, IfNode)
    assert len(node.then_body) == 1
    assert isinstance(node.then_body[0], FailNode)
    assert node.then_body[0].message == "Jira unavailable"


def test_parse_fail_missing_message():
    with pytest.raises(ParseError):
        AWLParser("@start\n@fail\n@end\n").parse()


# ── @continue / @break directive tests ──────────────────────────


def test_parse_continue_basic():
    script = "@start\n@continue skipping this item\n@end\n"
    result = AWLParser(script).parse()
    assert len(result.body) == 1
    assert isinstance(result.body[0], ContinueNode)
    assert result.body[0].message == "skipping this item"


def test_parse_continue_inside_loop():
    script = "@start\n@loop items as item\n@if item == 'skip'\n@continue item is bad\n@end\n@end\n@end\n"
    result = AWLParser(script).parse()
    loop = result.body[0]
    assert isinstance(loop, LoopNode)
    if_node = loop.body[0]
    assert isinstance(if_node, IfNode)
    assert if_node.expression == "item == 'skip'"
    assert isinstance(if_node.then_body[0], ContinueNode)
    assert if_node.then_body[0].message == "item is bad"


def test_parse_continue_missing_message():
    with pytest.raises(ParseError):
        AWLParser("@start\n@continue\n@end\n").parse()


def test_parse_break_basic():
    script = "@start\n@break no more items\n@end\n"
    result = AWLParser(script).parse()
    assert len(result.body) == 1
    assert isinstance(result.body[0], BreakNode)
    assert result.body[0].message == "no more items"


def test_parse_break_inside_while():
    script = "@start\n@while counter > 0\n@break done early\n@end\n@end\n"
    result = AWLParser(script).parse()
    while_node = result.body[0]
    assert isinstance(while_node, WhileNode)
    assert isinstance(while_node.body[0], BreakNode)
    assert while_node.body[0].message == "done early"


def test_parse_break_missing_message():
    with pytest.raises(ParseError):
        AWLParser("@start\n@break\n@end\n").parse()


# ── @goal directive tests ────────────────────────────────────────


def test_parse_goal_basic():
    script = """\
@start

@goal track_failures
  Success: Failure rate below 10%

  @task check
  Goal: Check failure rate.
  Expose: failure_rate
  @end

@end

@end"""
    result = AWLParser(script).parse()
    assert len(result.body) == 1
    goal = result.body[0]
    assert isinstance(goal, GoalNode)
    assert goal.goal_id == "track_failures"
    assert goal.success_criteria == "Failure rate below 10%"
    assert goal.max_actions == 5  # default
    assert len(goal.body) == 1
    assert isinstance(goal.body[0], TaskNode)


def test_parse_goal_with_max_actions():
    script = """\
@start

@goal weekly_report max_actions=3
  Success: Report generated and sent

  @task generate
  Goal: Generate weekly report.
  @end

@end

@end"""
    result = AWLParser(script).parse()
    goal = result.body[0]
    assert isinstance(goal, GoalNode)
    assert goal.max_actions == 3


def test_parse_goal_with_complex_body():
    script = """\
@start

@goal monitor_ci max_actions=5
  Success: All CI pipelines green for 24 hours

  @task check_pipelines @no-history
  Goal: Check CI pipeline status.
  Expose: failed_pipelines, total
  @end

  @if len(failed_pipelines) > 0
    @task create_report
    Goal: Write report about ${failed_pipelines}.
    @end
  @end

@end

@end"""
    result = AWLParser(script).parse()
    goal = result.body[0]
    assert isinstance(goal, GoalNode)
    assert len(goal.body) == 2
    assert isinstance(goal.body[0], TaskNode)
    assert isinstance(goal.body[1], IfNode)


def test_parse_goal_missing_success():
    script = """\
@start
@goal my_goal
  @task t1
  Goal: Do something.
  @end
@end
@end"""
    with pytest.raises(ParseError, match="Success"):
        AWLParser(script).parse()


# ── @wait directive tests ───────────────────────────────────────


def test_parse_wait_seconds():
    result = AWLParser("@start\n@wait 30s\n@end").parse()
    assert len(result.body) == 1
    assert isinstance(result.body[0], WaitNode)
    assert result.body[0].duration_seconds == 30


def test_parse_wait_minutes():
    result = AWLParser("@start\n@wait 5m\n@end").parse()
    node = result.body[0]
    assert isinstance(node, WaitNode)
    assert node.duration_seconds == 300


def test_parse_wait_hours():
    result = AWLParser("@start\n@wait 1h\n@end").parse()
    node = result.body[0]
    assert isinstance(node, WaitNode)
    assert node.duration_seconds == 3600


def test_parse_wait_invalid_no_unit():
    with pytest.raises(ParseError):
        AWLParser("@start\n@wait 30\n@end").parse()


def test_parse_wait_invalid_format():
    with pytest.raises(ParseError):
        AWLParser("@start\n@wait forever\n@end").parse()


# ── @while directive tests ──────────────────────────────────────


def test_parse_while_basic():
    script = "@start\n@while count > 0\n@set x = 1\n@end\n@end"
    result = AWLParser(script).parse()
    assert len(result.body) == 1
    node = result.body[0]
    assert isinstance(node, WhileNode)
    assert node.expression == "count > 0"
    assert node.max_iterations == 100  # default
    assert len(node.body) == 1


def test_parse_while_with_max_iterations():
    script = "@start\n@while running max_iterations=24\n@set x = 1\n@end\n@end"
    result = AWLParser(script).parse()
    node = result.body[0]
    assert isinstance(node, WhileNode)
    assert node.expression == "running"
    assert node.max_iterations == 24


def test_parse_while_with_task():
    script = """\
@start
@while len(jobs) > 0 max_iterations=10
  @task check @no-history @no-kg
  Goal: Check status.
  Expose: jobs
  @end
@end
@end"""
    result = AWLParser(script).parse()
    node = result.body[0]
    assert isinstance(node, WhileNode)
    assert len(node.body) == 1
    assert isinstance(node.body[0], TaskNode)


def test_parse_while_missing_expression():
    with pytest.raises(ParseError):
        AWLParser("@start\n@while\n@end\n@end").parse()


def test_parse_while_invalid_expression():
    with pytest.raises(ParseError):
        AWLParser("@start\n@while !!!\n@end\n@end").parse()


# ── @notify directive tests ─────────────────────────────────────


def test_parse_notify_basic():
    result = AWLParser("@start\n@notify All jobs complete\n@end").parse()
    assert len(result.body) == 1
    node = result.body[0]
    assert isinstance(node, NotifyNode)
    assert node.message == "All jobs complete"


def test_parse_notify_with_variable():
    result = AWLParser("@start\n@notify ${count} jobs still running\n@end").parse()
    node = result.body[0]
    assert isinstance(node, NotifyNode)
    assert "${count}" in node.message


def test_parse_notify_missing_message():
    with pytest.raises(ParseError):
        AWLParser("@start\n@notify\n@end").parse()


def test_parse_goal_with_set_before():
    """Test that @set nodes can appear before @goal in the workflow"""
    script = """\
@start

@set product = "OCP 4.19"

@goal track
  Success: Product stable

  @task check
  Goal: Check ${product} status.
  Expose: status
  @end

@end

@end"""
    result = AWLParser(script).parse()
    assert len(result.body) == 2
    assert isinstance(result.body[0], SetNode)
    assert isinstance(result.body[1], GoalNode)
