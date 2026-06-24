"""AWL (Agent Workflow Language) parser - line-oriented recursive descent"""

import re

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


class ParseError(Exception):
    def __init__(self, line: int, message: str):
        self.line = line
        super().__init__(f"Line {line}: {message}")


VALID_HINTS = {"no-history", "no-kg", "continue-on-failure"}


class AWLParser:
    def __init__(self, source: str):
        self._lines = source.splitlines()
        self._pos = 0
        self._task_ids: set[str] = set()
        self._expr = AWLExpressionEvaluator()

    def parse(self) -> WorkflowNode:
        self._skip_blank()
        line = self._require_line()
        if not line.startswith("@start"):
            raise ParseError(self._pos + 1, f"Expected '@start', got '{line}'")
        max_steps = None
        max_steps_match = re.search(r"max_steps=(\d+)", line)
        if max_steps_match:
            max_steps = int(max_steps_match.group(1))
        self._advance()
        body = self._parse_body()
        self._expect("@end")
        trailing = self._current_line()
        if trailing is not None:
            raise ParseError(self._pos + 1, f"Unexpected content after final @end: '{trailing}'")
        return WorkflowNode(body=body, max_steps=max_steps)

    def _current_line(self) -> str | None:
        while self._pos < len(self._lines):
            line = self._lines[self._pos].strip()
            if line == "" or line.startswith("#"):
                self._pos += 1
                continue
            if " #" in line:
                line = line[: line.index(" #")].strip()
            return line
        return None

    def _require_line(self) -> str:
        line = self._current_line()
        if line is None:
            raise ParseError(self._pos + 1, "Unexpected end of input")
        return line

    def _advance(self) -> None:
        self._pos += 1

    def _skip_blank(self) -> None:
        while self._pos < len(self._lines) and self._lines[self._pos].strip() == "":
            self._pos += 1

    def _expect(self, prefix: str) -> None:
        line = self._current_line()
        if line is None:
            raise ParseError(self._pos + 1, f"Expected '{prefix}', got end of input")
        if line != prefix and not line.startswith(prefix + " "):
            if line != prefix:
                raise ParseError(self._pos + 1, f"Expected '{prefix}', got '{line}'")
        self._advance()

    def _parse_body(self) -> list[ASTNode]:
        nodes: list[ASTNode] = []
        while True:
            line = self._current_line()
            if line is None or line in {"@end", "@else"}:
                break
            if line.startswith("@task"):
                nodes.append(self._parse_task())
            elif line.startswith("@set "):
                nodes.append(self._parse_set())
            elif line.startswith("@if "):
                nodes.append(self._parse_if())
            elif line.startswith("@loop "):
                nodes.append(self._parse_loop())
            elif line.startswith("@return"):
                nodes.append(self._parse_return())
            elif line.startswith("@fail"):
                nodes.append(self._parse_fail())
            elif line.startswith("@goal "):
                nodes.append(self._parse_goal())
            elif line.startswith("@wait "):
                nodes.append(self._parse_wait())
            elif line.startswith("@while "):
                nodes.append(self._parse_while())
            elif line.startswith("@notify "):
                nodes.append(self._parse_notify())
            else:
                raise ParseError(self._pos + 1, f"Unexpected: '{line}'")
        return nodes

    def _parse_task(self) -> TaskNode:
        line = self._require_line()
        parts = line.split()

        if len(parts) < 2:
            raise ParseError(self._pos + 1, "Expected task ID after @task")

        task_id = parts[1]
        if task_id in self._task_ids:
            raise ParseError(self._pos + 1, f"Duplicate task ID '{task_id}'")
        self._task_ids.add(task_id)

        hints = [p.removeprefix("@") for p in parts[2:] if p.startswith("@")]
        for hint in hints:
            if hint not in VALID_HINTS:
                raise ParseError(
                    self._pos + 1,
                    f"Unknown hint '@{hint}'. Valid hints: {', '.join('@' + h for h in sorted(VALID_HINTS))}",
                )

        # Extract max_tool_calls=N (optional)
        max_tool_calls = None
        max_tc_match = re.search(r"max_tool_calls=(\d+)", line)
        if max_tc_match:
            max_tool_calls = int(max_tc_match.group(1))

        # Extract max_time=N (optional, in seconds)
        max_time = None
        max_time_match = re.search(r"max_time=(\d+)", line)
        if max_time_match:
            max_time = int(max_time_match.group(1))

        # Extract model=<name> (optional)
        model = None
        model_match = re.search(r"model=([\w.\-]+)", line)
        if model_match:
            model = model_match.group(1)

        self._advance()

        goal = None
        context = None
        constraints = None
        success = None
        expose: list[str] = []

        task_fields = {"Goal:", "Context:", "Constraints:", "Success:", "Expose:"}

        while True:
            field_line = self._current_line()
            if field_line is None or field_line == "@end":
                break

            if field_line.startswith("Goal:"):
                goal = self._parse_field_value("Goal:", task_fields)
            elif field_line.startswith("Context:"):
                context = self._parse_field_value("Context:", task_fields)
            elif field_line.startswith("Constraints:"):
                constraints = self._parse_field_value("Constraints:", task_fields)
            elif field_line.startswith("Success:"):
                success = self._parse_field_value("Success:", task_fields)
            elif field_line.startswith("Expose:"):
                expose_str = field_line.removeprefix("Expose:").strip()
                expose = [v.strip() for v in expose_str.split(",")]
                self._advance()
            else:
                break

        if goal is None:
            raise ParseError(self._pos + 1, f"Task '{task_id}' missing required Goal field")

        self._expect("@end")

        return TaskNode(
            task_id=task_id,
            hints=hints,
            goal=goal,
            context=context,
            constraints=constraints,
            success=success,
            expose=expose,
            max_tool_calls=max_tool_calls,
            max_time=max_time,
            model=model,
        )

    def _parse_field_value(self, prefix: str, all_fields: set[str]) -> str:
        value = self._require_line().removeprefix(prefix).strip()
        self._advance()

        while True:
            cont = self._current_line()
            if cont is None or cont.startswith("@"):
                break
            if any(cont.startswith(f) for f in all_fields):
                break
            value += " " + cont
            self._advance()

        return value

    def _parse_set(self) -> SetNode:
        line = self._require_line()
        match = re.match(r"@set\s+(\w+)\s*=\s*(.*)", line)
        if not match:
            raise ParseError(self._pos + 1, "Invalid @set syntax. Expected: @set <var> = <value>")
        variable = match.group(1)
        value = match.group(2).strip().strip('"').strip("'")
        self._advance()
        return SetNode(variable=variable, value=value)

    def _parse_if(self) -> IfNode:
        line = self._require_line()
        expression = line.removeprefix("@if").strip()
        self._validate_expr(expression, "after @if")
        self._advance()

        then_body = self._parse_body()
        else_body: list[ASTNode] = []

        if self._current_line() == "@else":
            self._advance()
            else_body = self._parse_body()

        self._expect("@end")
        return IfNode(expression=expression, then_body=then_body, else_body=else_body)

    def _parse_loop(self) -> LoopNode:
        line = self._require_line()
        match = re.match(r"@loop\s+(\w+)\s+as\s+(\w+)(.*)", line)
        if not match:
            raise ParseError(self._pos + 1, "Invalid @loop syntax. Expected: @loop <collection> as <item> [limit=N]")
        collection = match.group(1)
        item_var = match.group(2)
        opts = match.group(3)
        limit_match = re.search(r"limit=(\d+)", opts)
        limit = int(limit_match.group(1)) if limit_match else None
        collect_match = re.search(r"collect=(\w+)(?:\(([^)]+)\))?", opts)
        collect = collect_match.group(1) if collect_match else None
        collect_fields: list[str] = []
        if collect_match and collect_match.group(2):
            collect_fields = [f.strip() for f in collect_match.group(2).split(",")]
        self._advance()

        body = self._parse_body()
        self._expect("@end")
        return LoopNode(
            collection=collection,
            item_var=item_var,
            limit=limit,
            collect=collect,
            collect_fields=collect_fields,
            body=body,
        )

    def _parse_goal(self) -> GoalNode:
        line = self._require_line()
        match = re.match(r"@goal\s+(\w+)(.*)", line)
        if not match:
            raise ParseError(self._pos + 1, "Invalid @goal syntax. Expected: @goal <id> [max_actions=N]")

        goal_id = match.group(1)
        opts = match.group(2)

        # Extract max_actions (optional, default 5)
        max_actions_match = re.search(r"max_actions=(\d+)", opts)
        max_actions = int(max_actions_match.group(1)) if max_actions_match else 5

        self._advance()

        # Parse required Success: field
        goal_fields = {"Success:"}
        success_criteria = None
        current = self._current_line()
        if current is not None and current.startswith("Success:"):
            success_criteria = self._parse_field_value("Success:", goal_fields)

        if success_criteria is None:
            raise ParseError(self._pos + 1, f"@goal '{goal_id}' missing required Success field")

        body = self._parse_body()
        self._expect("@end")

        return GoalNode(
            goal_id=goal_id,
            success_criteria=success_criteria,
            max_actions=max_actions,
            body=body,
        )

    _DURATION_UNITS = {"s": 1, "m": 60, "h": 3600}

    def _parse_wait(self) -> WaitNode:
        line = self._require_line()
        match = re.match(r"@wait\s+(\d+)(s|m|h)$", line)
        if not match:
            raise ParseError(
                self._pos + 1,
                "Invalid @wait syntax. Expected: @wait <N>s|m|h (e.g. @wait 5m)",
            )
        amount = int(match.group(1))
        unit = match.group(2)
        self._advance()
        return WaitNode(duration_seconds=amount * self._DURATION_UNITS[unit])

    def _parse_while(self) -> WhileNode:
        line = self._require_line()
        # Strip @while prefix, then extract optional max_iterations from the end
        rest = line.removeprefix("@while").strip()
        max_iterations = 100
        max_iter_match = re.search(r"\s+max_iterations=(\d+)$", rest)
        if max_iter_match:
            max_iterations = int(max_iter_match.group(1))
            rest = rest[: max_iter_match.start()].strip()
        expression = rest
        if not expression:
            raise ParseError(self._pos + 1, "Missing expression after @while")
        self._validate_expr(expression, "after @while")
        self._advance()
        body = self._parse_body()
        self._expect("@end")
        return WhileNode(expression=expression, max_iterations=max_iterations, body=body)

    def _parse_notify(self) -> NotifyNode:
        line = self._require_line()
        message = line.removeprefix("@notify").strip()
        if not message:
            raise ParseError(self._pos + 1, "@notify requires a message")
        self._advance()
        return NotifyNode(message=message)

    def _validate_expr(self, expression: str, context: str) -> None:
        try:
            self._expr.validate_expression(expression)
        except ValueError as e:
            raise ParseError(self._pos + 1, f"Invalid expression {context}: {e}") from None

    def _parse_return(self) -> ReturnNode:
        line = self._require_line()
        expression = line.removeprefix("@return").strip()
        self._validate_expr(expression, "after @return")
        self._advance()
        return ReturnNode(expression=expression)

    def _parse_fail(self) -> FailNode:
        line = self._require_line()
        message = line.removeprefix("@fail").strip()
        if not message:
            raise ParseError(self._pos + 1, "@fail requires a message")
        self._advance()
        return FailNode(message=message)
