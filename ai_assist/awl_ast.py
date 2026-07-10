"""AWL (Agent Workflow Language) AST node definitions"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskNode:
    task_id: str
    goal: str
    hints: list[str] = field(default_factory=list)
    context: str | None = None
    constraints: str | None = None
    success: str | None = None
    expose: list[str] = field(default_factory=list)
    max_tool_calls: int | None = None
    max_time: int | None = None
    model: str | None = None


@dataclass
class SetNode:
    variable: str
    value: str


@dataclass
class ReturnNode:
    expression: str


@dataclass
class IfNode:
    expression: str
    then_body: list[Any] = field(default_factory=list)
    else_body: list[Any] = field(default_factory=list)


@dataclass
class LoopNode:
    collection: str
    item_var: str
    body: list[Any] = field(default_factory=list)
    limit: int | None = None
    collect: str | None = None
    collect_fields: list[str] = field(default_factory=list)


@dataclass
class WorkflowNode:
    body: list[Any] = field(default_factory=list)
    max_steps: int | None = None
    idempotent: bool = False


@dataclass
class FailNode:
    message: str


@dataclass
class GoalNode:
    goal_id: str
    success_criteria: str
    max_actions: int = 5
    body: list[Any] = field(default_factory=list)


@dataclass
class WaitNode:
    duration_seconds: int


@dataclass
class WhileNode:
    expression: str
    max_iterations: int = 100
    body: list[Any] = field(default_factory=list)


@dataclass
class NotifyNode:
    message: str


ASTNode = (
    WorkflowNode
    | TaskNode
    | SetNode
    | IfNode
    | LoopNode
    | ReturnNode
    | FailNode
    | GoalNode
    | WaitNode
    | WhileNode
    | NotifyNode
)
