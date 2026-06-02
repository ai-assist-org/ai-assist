"""AWL runtime engine - executes AWL ASTs via agent reasoning"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .awl_ast import (
    ASTNode,
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
from .awl_expressions import AWLExpressionEvaluator
from .filesystem_tools import extract_command_names

logger = logging.getLogger(__name__)


class AWLRuntimeError(Exception):
    pass


class _ReturnSignal(Exception):
    def __init__(self, value: Any):
        self.value = value


class _TaskFailedError(Exception):
    """Internal signal: top-level task failed, stop workflow."""

    pass


@dataclass
class TaskOutcome:
    status: str
    summary: str
    exposed: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    success: bool
    return_value: Any = None
    variables: dict[str, Any] = field(default_factory=dict)
    task_outcomes: list[TaskOutcome] = field(default_factory=list)


@dataclass
class RuntimeLimits:
    max_steps: int = 50
    max_tool_calls: int = 100
    timeout: float = 300.0


def _extract_commands_from_workflow(workflow: WorkflowNode) -> set[str]:
    """Extract command names from commands in AWL task/goal text.

    Looks for commands in two forms:
    - Backtick-enclosed: `git fetch origin`
    - Indented command lines: lines starting with whitespace followed by a command

    Commands explicitly written in AWL scripts are trusted and should be
    auto-authorized during workflow execution.
    """
    commands: set[str] = set()

    def _extract_from_text(text: str):
        for match in re.findall(r"`([^`]+)`", text):
            commands.update(extract_command_names(match))

    def _visit(nodes: list[ASTNode]):
        for node in nodes:
            if isinstance(node, TaskNode):
                _extract_from_text(node.goal)
                if node.context:
                    _extract_from_text(node.context)
            elif isinstance(node, GoalNode):
                _extract_from_text(node.success_criteria)
                _visit(node.body)
            elif isinstance(node, IfNode):
                _visit(node.then_body)
                _visit(node.else_body)
            elif isinstance(node, LoopNode):
                _visit(node.body)
            elif isinstance(node, WhileNode):
                _visit(node.body)

    _visit(workflow.body)
    # Filter out markers like <shell-expansion>
    return {cmd for cmd in commands if not cmd.startswith("<")}


def _compute_input_variables(workflow: WorkflowNode) -> set[str]:
    """Compute variables that are expected as external inputs.

    These are variables referenced in the workflow but never produced
    internally (via Expose:, @set, or @loop item_var/collect).
    They are expected to be injected at runtime via CLI or the variables dict.
    """
    interpolation_re = re.compile(r"\$\{(\w+)\}")
    from .awl_expressions import AWLExpressionEvaluator

    expr_eval = AWLExpressionEvaluator()
    used: set[str] = set()
    defined: set[str] = set()

    def _collect_from_text(text: str | None) -> None:
        if text:
            used.update(interpolation_re.findall(text))

    def _collect_from_expr(expression: str) -> None:
        used.update(expr_eval.extract_variables(expression))

    def _walk(nodes: list[ASTNode]) -> None:
        for node in nodes:
            if isinstance(node, TaskNode):
                for text in (node.goal, node.context, node.constraints, node.success):
                    _collect_from_text(text)
                defined.update(node.expose)
            elif isinstance(node, SetNode):
                _collect_from_text(node.value)
                defined.add(node.variable)
            elif isinstance(node, IfNode):
                _collect_from_expr(node.expression)
                _walk(node.then_body)
                _walk(node.else_body)
            elif isinstance(node, LoopNode):
                used.add(node.collection)
                defined.add(node.item_var)
                if node.collect:
                    defined.add(node.collect)
                _walk(node.body)
            elif isinstance(node, ReturnNode):
                _collect_from_expr(node.expression)
            elif isinstance(node, GoalNode):
                _walk(node.body)
            elif isinstance(node, WhileNode):
                _collect_from_expr(node.expression)
                _walk(node.body)
            elif isinstance(node, NotifyNode):
                _collect_from_text(node.message)

    _walk(workflow.body)
    return used - defined


def validate_workflow_variables(workflow: WorkflowNode, initial_variables: set[str] | None = None) -> list[str]:
    """Static analysis: detect variables used before they are defined.

    Walks the AST in execution order, tracking which variables have been
    defined at each point.  Returns a list of warning strings for any
    variable that is referenced before it could have been set.

    When ``initial_variables`` is not provided, the function automatically
    identifies expected input variables (those referenced but never produced
    internally) and excludes them from warnings.

    Key rules:
    - ``@set`` and ``Expose:`` define variables for subsequent nodes.
    - ``@loop collection as item collect=result`` makes ``item`` available
      inside the body, but ``result`` is only available *after* the loop.
    - ``@if`` branches may or may not run, so variables defined inside are
      treated as *possibly* defined (warnings are still emitted if a variable
      is used only after an @if branch that might not execute, but that level
      of analysis is deferred — we optimistically add them).
    """
    interpolation_re = re.compile(r"\$\{(\w+)")
    from .awl_expressions import AWLExpressionEvaluator

    expr_eval = AWLExpressionEvaluator()

    defined: set[str] = set(initial_variables) if initial_variables else set()
    warnings: list[str] = []

    def _check_used(text: str | None, context: str) -> None:
        if not text:
            return
        for var in interpolation_re.findall(text):
            if var not in defined:
                warnings.append(f"{context}: variable '{var}' used before definition")

    def _check_expr(expression: str, context: str) -> None:
        for var in expr_eval.extract_variables(expression):
            if var not in defined:
                warnings.append(f"{context}: variable '{var}' used before definition")

    def _walk(nodes: list[ASTNode]) -> None:
        for node in nodes:
            if isinstance(node, TaskNode):
                for text in (node.goal, node.context, node.constraints, node.success):
                    _check_used(text, f"@task '{node.task_id}'")
                defined.update(node.expose)
            elif isinstance(node, SetNode):
                _check_used(node.value, f"@set '{node.variable}'")
                defined.add(node.variable)
            elif isinstance(node, IfNode):
                _check_expr(node.expression, "@if")
                _walk(node.then_body)
                _walk(node.else_body)
            elif isinstance(node, LoopNode):
                if node.collection not in defined:
                    warnings.append(f"@loop: collection variable '{node.collection}' used before definition")
                # item_var is available inside the body
                defined.add(node.item_var)
                # collect is NOT available inside the body
                _walk(node.body)
                # collect becomes available after the loop
                if node.collect:
                    defined.add(node.collect)
            elif isinstance(node, ReturnNode):
                _check_expr(node.expression, "@return")
            elif isinstance(node, GoalNode):
                _walk(node.body)
            elif isinstance(node, WhileNode):
                _check_expr(node.expression, "@while")
                _walk(node.body)
            elif isinstance(node, NotifyNode):
                _check_used(node.message, "@notify")

    _walk(workflow.body)
    return warnings


class AWLRuntime:
    def __init__(self, agent: Any, limits: RuntimeLimits | None = None, verbose: bool = False):
        self._agent = agent
        self._limits = limits or RuntimeLimits()
        self._variables: dict[str, Any] = {}
        self._expr = AWLExpressionEvaluator()
        self._steps = 0
        self._loop_depth = 0
        self._task_outcomes: list[TaskOutcome] = []
        self._verbose = verbose

    async def execute(self, workflow: WorkflowNode, variables: dict[str, Any] | None = None) -> WorkflowResult:
        self._variables = dict(variables) if variables else {}
        self._steps = 0
        self._task_outcomes = []
        return_value = None

        # Static validation: warn about variables used before definition
        var_warnings = validate_workflow_variables(workflow, set(self._variables.keys()))
        for w in var_warnings:
            logger.warning("AWL validation: %s", w)
            print(f"  [!] {w}")

        if workflow.max_steps is not None:
            self._limits.max_steps = workflow.max_steps

        # Auto-authorize commands explicitly mentioned in the AWL script
        awl_commands = _extract_commands_from_workflow(workflow)
        added_commands: list[str] = []
        fs_tools = getattr(self._agent, "filesystem_tools", None)
        if fs_tools and awl_commands:
            for cmd in awl_commands:
                if cmd not in fs_tools.allowed_commands:
                    fs_tools.allowed_commands.append(cmd)
                    added_commands.append(cmd)
            fs_tools.awl_authorized_commands |= awl_commands
            if added_commands:
                logger.info("AWL auto-authorized commands: %s", added_commands)

        try:
            await self._execute_body(workflow.body)
        except _ReturnSignal as ret:
            return_value = ret.value
        except _TaskFailedError:
            pass  # outcome already recorded; fall through to result
        finally:
            # Remove temporarily added commands
            if fs_tools and added_commands:
                for cmd in added_commands:
                    try:
                        fs_tools.allowed_commands.remove(cmd)
                    except ValueError:
                        pass
                fs_tools.awl_authorized_commands -= awl_commands

        all_succeeded = all(o.status == "success" for o in self._task_outcomes)
        return WorkflowResult(
            success=all_succeeded,
            return_value=return_value,
            variables=dict(self._variables),
            task_outcomes=list(self._task_outcomes),
        )

    async def _execute_body(self, body: list[ASTNode]):
        for node in body:
            await self._execute_node(node)

    async def _execute_node(self, node: ASTNode):
        # Only count LLM-calling nodes (task, goal) outside loops as steps.
        # Loop iterations are naturally bounded by the collection size
        # (and optionally by limit=N), so they don't need step limits.
        if isinstance(node, TaskNode | GoalNode) and self._loop_depth == 0:
            self._steps += 1
            if self._steps > self._limits.max_steps:
                raise AWLRuntimeError(f"Exceeded max_steps ({self._limits.max_steps})")

        if isinstance(node, TaskNode):
            await self._execute_task(node)
        elif isinstance(node, SetNode):
            self._execute_set(node)
        elif isinstance(node, IfNode):
            await self._execute_if(node)
        elif isinstance(node, LoopNode):
            await self._execute_loop(node)
        elif isinstance(node, ReturnNode):
            self._execute_return(node)
        elif isinstance(node, GoalNode):
            await self._execute_goal(node)
        elif isinstance(node, WaitNode):
            await self._execute_wait(node)
        elif isinstance(node, WhileNode):
            await self._execute_while(node)
        elif isinstance(node, NotifyNode):
            await self._execute_notify(node)
        elif isinstance(node, FailNode):
            raise AWLRuntimeError(node.message)

    async def _execute_task(self, task: TaskNode):
        prompt = self._build_task_prompt(task)

        if "no-kg" in task.hints:
            prompt = "@no-kg " + prompt

        logger.info("AWL task '%s' starting", task.task_id)
        print(f"  > task '{task.task_id}' ...", flush=True)
        try:
            callback = self._progress_callback if self._verbose else None
            # During AWL tasks, use renderer for inner execution (no noisy text)
            prev_inner = getattr(self._agent, "on_inner_execution", None)
            renderer = getattr(self._agent, "renderer", None)
            if self._verbose and renderer:
                self._agent.on_inner_execution = renderer.on_inner_execution
            else:
                self._agent.on_inner_execution = None
            try:
                max_turns = task.max_tool_calls or self._limits.max_tool_calls
                response = await self._agent.query(
                    prompt,
                    max_turns=max_turns,
                    progress_callback=callback,
                    max_time_seconds=task.max_time,
                )
            finally:
                self._agent.on_inner_execution = prev_inner
            exposed = self._extract_exposed(response, task.expose)

            # Retry: if exposed vars are missing, nudge the agent to emit the JSON block
            if task.expose:
                missing_vars = [v for v in task.expose if v not in exposed]
                if missing_vars:
                    exposed = await self._retry_expose_extraction(task, response, exposed, missing_vars)

            self._variables.update(exposed)
            outcome = TaskOutcome(status="success", summary=response[:200], exposed=exposed)
            self._task_outcomes.append(outcome)

            # Log exposed variables and default missing ones to None
            if task.expose:
                missing_vars = [v for v in task.expose if v not in exposed]
                if missing_vars:
                    logger.warning(
                        "AWL task '%s': expected exposed vars %s but got %s (missing: %s)",
                        task.task_id,
                        task.expose,
                        list(exposed.keys()),
                        missing_vars,
                    )
                    print(f"    [!] missing exposed vars: {missing_vars}")
                    # Default missing vars to None so downstream @if checks
                    # evaluate to False instead of triggering "undefined variable"
                    for var in missing_vars:
                        self._variables[var] = None

            if exposed:
                for key, val in exposed.items():
                    val_str = str(val)
                    if len(val_str) > 200:
                        val_str = val_str[:200] + "..."
                    print(f"    [+] {key} = {val_str}")
                logger.info(
                    "AWL task '%s' succeeded, exposed: %s",
                    task.task_id,
                    {k: repr(v)[:100] for k, v in exposed.items()},
                )
            else:
                print("    [+] success")
                logger.info("AWL task '%s' succeeded (no exposed vars)", task.task_id)
        except Exception as e:
            outcome = TaskOutcome(status="failed", summary=str(e))
            self._task_outcomes.append(outcome)
            logger.error("AWL task '%s' failed: %s", task.task_id, e)
            print(f"    [-] failed: {e}")
            if self._loop_depth == 0 and "continue-on-failure" not in task.hints:
                raise _TaskFailedError(f"Task '{task.task_id}' failed: {e}") from e

    async def _retry_expose_extraction(
        self,
        task: TaskNode,
        response: str,
        partial_exposed: dict[str, Any],
        missing_vars: list[str],
    ) -> dict[str, Any]:
        """Send a follow-up to extract missing Expose: variables.

        After complex tasks (e.g. RCA via MCP prompt), the agent may forget to
        emit the JSON block.  We send a fresh query with the full response text
        as context — the response is small (just the agent's final text), whereas
        the conversation history (with downloaded log files in tool results) can
        be hundreds of KB and would cause a 400 error.
        """
        keys = ", ".join(missing_vars)
        example = ", ".join(f'"{k}": "<value>"' for k in missing_vars)
        prompt = (
            f"A task just completed and produced the following output:\n"
            f"---\n{response}\n---\n\n"
            f"Extract the requested information from the output above and return "
            f"ONLY a JSON block with these keys: {keys}\n"
            f"```json\n{{{example}}}\n```\n"
            f"If a value is not present in the output, use your best judgement to "
            f"infer it from context. Do NOT call any tools. Output nothing else."
        )
        logger.info("AWL task '%s': nudging for missing exposed vars: %s", task.task_id, missing_vars)
        print(f"    [~] nudging for missing vars: {missing_vars}")
        try:
            retry_response = await self._agent.query(prompt, max_turns=2)
            retry_exposed = self._extract_exposed(retry_response, task.expose)
            if retry_exposed:
                merged = {**partial_exposed, **retry_exposed}
                still_missing = [v for v in task.expose if v not in merged]
                if not still_missing:
                    logger.info("AWL task '%s': nudge recovered all exposed vars", task.task_id)
                    print(f"    [+] nudge recovered: {list(retry_exposed.keys())}")
                else:
                    logger.warning(
                        "AWL task '%s': nudge recovered %s but still missing %s",
                        task.task_id,
                        list(retry_exposed.keys()),
                        still_missing,
                    )
                    print(f"    [~] nudge recovered {list(retry_exposed.keys())}, still missing {still_missing}")
                return merged
        except Exception as e:
            logger.warning("AWL task '%s': nudge failed: %s", task.task_id, e)
        return partial_exposed

    def _execute_set(self, node: SetNode):
        self._variables[node.variable] = self._expr.interpolate(node.value, self._variables)

    def _check_undefined_variables(self, expression: str, directive: str) -> None:
        """Warn about variables referenced in an expression that are not defined."""
        referenced = self._expr.extract_variables(expression)
        undefined = referenced - set(self._variables.keys())
        if undefined:
            names = ", ".join(sorted(undefined))
            logger.warning(
                "AWL %s '%s': undefined variable(s): %s",
                directive,
                expression,
                names,
            )
            print(f"  [!] {directive} '{expression}': undefined variable(s): {names}")

    async def _execute_if(self, node: IfNode):
        self._check_undefined_variables(node.expression, "@if")
        value = self._expr.evaluate(node.expression, self._variables)
        truthy = self._expr.is_truthy(value)
        logger.info("AWL @if '%s' evaluated to %s (raw: %s)", node.expression, truthy, value)
        if truthy:
            await self._execute_body(node.then_body)
        else:
            await self._execute_body(node.else_body)

    async def _execute_loop(self, node: LoopNode):
        if node.collection not in self._variables:
            logger.warning(
                "AWL @loop '%s': collection variable '%s' is not defined",
                node.collection,
                node.collection,
            )
            print(f"  [!] @loop '{node.collection}': collection variable is not defined")
        collection = self._variables.get(node.collection, [])
        collection = self._coerce_to_list(collection, node.collection)
        if not isinstance(collection, list | tuple):
            raise AWLRuntimeError(
                f"@loop '{node.collection}' is not a list (got {type(collection).__name__}). "
                f"Ensure the task exposes it as a JSON array."
            )

        items = collection
        if node.limit is not None:
            items = items[: node.limit]

        logger.info(
            "AWL @loop '%s': %d items%s",
            node.collection,
            len(items),
            f" (collect={node.collect})" if node.collect else "",
        )

        collected: list[dict[str, Any]] = []
        self._loop_depth += 1

        try:
            for i, item in enumerate(items, 1):
                item_full = str(item)
                item_short = item_full[:100] + "..." if len(item_full) > 100 else item_full
                print(f"  loop {node.collection} [{i}/{len(items)}]: {node.item_var} = {item_short}")
                logger.info("AWL @loop iteration %d/%d: %s = %s", i, len(items), node.item_var, item_full)
                self._variables[node.item_var] = item
                outcomes_before = len(self._task_outcomes)
                await self._execute_body(node.body)

                if node.collect is not None:
                    new_outcomes = self._task_outcomes[outcomes_before:]
                    all_succeeded = all(o.status == "success" for o in new_outcomes)
                    if not new_outcomes:
                        logger.warning("AWL @loop collect: iteration %d produced no outcomes", i)
                    elif not all_succeeded:
                        failed = [o for o in new_outcomes if o.status != "success"]
                        logger.warning(
                            "AWL @loop collect: iteration %d had %d failed task(s), skipping", i, len(failed)
                        )
                        print(f"    [!] iteration {i} failed, skipping collect")
                    else:
                        iteration_exposed: dict[str, Any] = {}
                        for outcome in new_outcomes:
                            iteration_exposed.update(outcome.exposed)
                        if not iteration_exposed:
                            # Task succeeded but no vars extracted -- still include with item context
                            logger.warning(
                                "AWL @loop collect: iteration %d succeeded but no exposed vars extracted, "
                                "including with _item only",
                                i,
                            )
                            print(f"    [!] iteration {i}: no exposed vars, including with _item")
                            iteration_exposed = {"_item": item}
                        if node.collect_fields:
                            iteration_exposed = {k: v for k, v in iteration_exposed.items() if k in node.collect_fields}
                        collected.append(iteration_exposed)
        finally:
            self._loop_depth -= 1

        if node.collect is not None:
            self._variables[node.collect] = collected
            logger.info("AWL @loop collected %d/%d items into '%s'", len(collected), len(items), node.collect)
            if len(collected) < len(items):
                print(f"  [!] collect: {len(collected)}/{len(items)} iterations produced results")
                logger.warning(
                    "AWL @loop '%s' collect: only %d/%d iterations produced results",
                    node.collection,
                    len(collected),
                    len(items),
                )

    async def _execute_wait(self, node: WaitNode):
        if node.duration_seconds >= 3600:
            label = f"{node.duration_seconds // 3600}h"
        elif node.duration_seconds >= 60:
            label = f"{node.duration_seconds // 60}m"
        else:
            label = f"{node.duration_seconds}s"
        logger.info("AWL @wait %s (%d seconds)", label, node.duration_seconds)
        print(f"  > waiting {label} ...", flush=True)
        await asyncio.sleep(node.duration_seconds)

    async def _execute_while(self, node: WhileNode):
        self._check_undefined_variables(node.expression, "@while")
        iteration = 0
        self._loop_depth += 1
        try:
            while iteration < node.max_iterations:
                value = self._expr.evaluate(node.expression, self._variables)
                truthy = self._expr.is_truthy(value)
                if not truthy:
                    break
                iteration += 1
                logger.info(
                    "AWL @while iteration %d/%d: '%s' = %s",
                    iteration,
                    node.max_iterations,
                    node.expression,
                    value,
                )
                print(
                    f"  while [{iteration}/{node.max_iterations}]: {node.expression} = {value}",
                    flush=True,
                )
                await self._execute_body(node.body)
        finally:
            self._loop_depth -= 1

        if iteration >= node.max_iterations:
            logger.warning("AWL @while reached max_iterations (%d)", node.max_iterations)
            print(f"  [!] @while reached max_iterations ({node.max_iterations})")

        logger.info("AWL @while exited after %d iteration(s)", iteration)

    async def _execute_notify(self, node: NotifyNode):
        message = self._expr.interpolate(node.message, self._variables)
        self._check_unresolved_interpolations("notify", [node.message])
        logger.info("AWL @notify: %s", message)
        print(f"  [*] {message}", flush=True)

        # Send via notification dispatcher (desktop)
        from datetime import datetime

        from .notification_dispatcher import Notification, NotificationDispatcher

        dispatcher = NotificationDispatcher()
        notification = Notification(
            id=f"awl-notify-{datetime.now().timestamp()}",
            action_id="awl-workflow",
            title="AWL Notification",
            message=message,
            level="info",
            timestamp=datetime.now(),
            channels=["desktop"],
        )
        result = await dispatcher.dispatch(notification)
        logger.info("AWL @notify dispatch result: %s", result)

    def _progress_callback(self, status: str, turn: int, max_turns: int, tool_name: str | None):
        """Print agent progress during task execution (verbose mode)."""
        if status == "executing_tool":
            print(f"    [{turn}/{max_turns}] calling {tool_name}", flush=True)
        elif status == "calling_claude":
            print(f"    [{turn}/{max_turns}] thinking...", flush=True)

    def _coerce_to_list(self, value: Any, var_name: str) -> Any:
        """Try to coerce a value to a list for @loop iteration.

        Handles: JSON strings, Python-repr strings, single dicts.
        """
        import ast

        # Already a list
        if isinstance(value, list | tuple):
            return value

        # Single dict → wrap in list
        if isinstance(value, dict):
            result = [value]
            self._variables[var_name] = result
            return result

        # String → try parsing as JSON, then Python literal
        if isinstance(value, str):
            # Try JSON first (double quotes)
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    self._variables[var_name] = parsed
                    return parsed
                if isinstance(parsed, dict):
                    result = [parsed]
                    self._variables[var_name] = result
                    return result
            except json.JSONDecodeError, ValueError:
                pass

            # Try Python literal (single quotes)
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    self._variables[var_name] = parsed
                    return parsed
                if isinstance(parsed, dict):
                    result = [parsed]
                    self._variables[var_name] = result
                    return result
            except ValueError, SyntaxError:
                pass

        return value

    def _execute_return(self, node: ReturnNode):
        value = self._expr.evaluate(node.expression, self._variables)
        raise _ReturnSignal(value)

    async def _execute_goal(self, goal: GoalNode):
        """Execute one cycle of a goal block."""
        saved_limit = self._limits.max_tool_calls
        self._limits.max_tool_calls = min(goal.max_actions, saved_limit)

        try:
            await self._execute_body(goal.body)
        finally:
            self._limits.max_tool_calls = saved_limit

        # Evaluate success criteria
        await self._evaluate_goal_success(goal)

    async def _evaluate_goal_success(self, goal: GoalNode):
        """Ask the agent whether the goal's success criterion is met."""
        var_summary = "\n".join(f"  {k} = {v}" for k, v in self._variables.items() if not k.startswith("_goal_"))
        prompt = (
            f"You are evaluating whether a goal's success criterion has been met.\n\n"
            f"Goal: {goal.goal_id}\n"
            f"Success criterion: {goal.success_criteria}\n\n"
            f"Current state after this cycle:\n{var_summary}\n\n"
            f"Has the success criterion been met? Respond with JSON:\n"
            f'{{"success_met": true/false, "reason": "brief explanation"}}'
        )
        response = await self._agent.query(prompt, max_turns=1)
        result = self._extract_exposed(response, ["success_met", "reason"])

        self._variables["_goal_success_met"] = result.get("success_met", False)
        self._variables["_goal_success_reason"] = result.get("reason", "")

    def _check_unresolved_interpolations(self, task_id: str, raw_templates: list[str | None]) -> None:
        """Warn about ${var} references in raw task templates that cannot be resolved.

        Scans the original template strings (before interpolation) so that
        shell-style variables introduced by substituted content (e.g. bash
        ``${GL_VERSION}`` embedded in a file read into an AWL variable) are
        not falsely flagged.
        """
        all_refs: list[str] = []
        for text in raw_templates:
            if text:
                all_refs.extend(re.findall(r"\$\{(\w+)(?:\.\w+)*\}", text))
        unresolved = [name for name in dict.fromkeys(all_refs) if name not in self._variables]
        if unresolved:
            names = ", ".join(unresolved)
            logger.warning(
                "AWL task '%s': unresolved variable(s) in prompt: %s",
                task_id,
                names,
            )
            print(f"  [!] task '{task_id}': unresolved variable(s): {names}")

    def _build_task_prompt(self, task: TaskNode) -> str:
        interp = self._expr.interpolate
        v = self._variables

        parts = [f"You are executing task '{task.task_id}'."]

        goal_text = interp(task.goal, v)
        parts.append(f"\nGoal: {goal_text}")

        if task.context:
            parts.append(f"Context: {interp(task.context, v)}")
        if task.constraints:
            parts.append(f"Constraints: {interp(task.constraints, v)}")
        if task.success:
            parts.append(f"\nSuccess criteria: {interp(task.success, v)}")

        if task.expose:
            keys = ", ".join(task.expose)
            example = ", ".join(f'"{k}": "<value>"' for k in task.expose)
            parts.append(f"\nIMPORTANT: Your final message MUST end with a JSON block containing these keys: {keys}")
            parts.append(f"```json\n{{{example}}}\n```")
            parts.append(
                "This JSON block is mandatory — without it the workflow cannot continue. "
                "Do NOT put the JSON inside a tool call or think block."
            )

        # If the goal references an MCP prompt (e.g. /server/prompt), remind the agent
        # to use introspection__execute_mcp_prompt to run it.
        combined = goal_text + (" " + interp(task.context, v) if task.context else "")
        if re.search(r"/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", combined):
            parts.append(
                "\nNote: To run an MCP prompt referenced as /server/prompt_name, "
                "call the introspection__execute_mcp_prompt tool with the server and prompt name."
            )

        prompt = "\n".join(parts)
        self._check_unresolved_interpolations(task.task_id, [task.goal, task.context, task.constraints, task.success])
        return prompt

    def _extract_exposed(self, response: str, expose_vars: list[str]) -> dict[str, Any]:
        if not expose_vars:
            return {}

        expected = set(expose_vars)
        result: dict[str, Any] = {}

        # Try fenced JSON blocks first — accumulate across all blocks
        json_blocks = re.findall(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
        for block in json_blocks:
            try:
                data = json.loads(block.strip())
                if isinstance(data, dict):
                    for k in expose_vars:
                        if k in data and k not in result:
                            result[k] = data[k]
                    if expected <= set(result.keys()):
                        return result
            except json.JSONDecodeError:
                continue

        if expected <= set(result.keys()):
            return result

        # Try parsing the entire response as JSON
        try:
            data = json.loads(response.strip())
            if isinstance(data, dict):
                for k in expose_vars:
                    if k in data and k not in result:
                        result[k] = data[k]
                if expected <= set(result.keys()):
                    return result
        except json.JSONDecodeError:
            pass

        # Fallback: find inline JSON objects (non-nested)
        for json_match in re.finditer(r"\{[^{}]*\}", response):
            try:
                data = json.loads(json_match.group())
                if isinstance(data, dict):
                    for k in expose_vars:
                        if k in data and k not in result:
                            result[k] = data[k]
                    if expected <= set(result.keys()):
                        return result
            except json.JSONDecodeError:
                continue

        return result
