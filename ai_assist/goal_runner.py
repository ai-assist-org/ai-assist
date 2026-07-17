"""Goal runner - executes AWL @goal scripts with state persistence"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from .awl_ast import GoalNode, WorkflowNode
from .awl_parser import AWLParser
from .awl_runtime import AWLRuntime, RuntimeLimits
from .goal_state import GoalStateManager

logger = logging.getLogger(__name__)


class GoalRunner:
    """Run a @goal AWL script with persisted state between cycles."""

    def __init__(
        self,
        awl_path: Path,
        agent,
        state_manager: GoalStateManager,
        initial_variables: dict | None = None,
    ):
        self.awl_path = awl_path
        self.agent = agent
        self.state_manager = state_manager
        self._initial_variables = initial_variables or {}
        self._workflow: WorkflowNode | None = None
        self._goal_node: GoalNode | None = None

    def load(self):
        """Parse the AWL file and extract the GoalNode."""
        source = self.awl_path.read_text()
        workflow = AWLParser(source).parse()

        goal_nodes = [n for n in workflow.body if isinstance(n, GoalNode)]
        if len(goal_nodes) != 1:
            raise ValueError(f"Expected exactly 1 @goal in {self.awl_path}, found {len(goal_nodes)}")

        self._workflow = workflow
        self._goal_node = goal_nodes[0]

    @property
    def goal_id(self) -> str:
        assert self._goal_node is not None
        return self._goal_node.goal_id

    async def run_cycle(self):
        """Execute one cycle of the goal."""
        assert self._workflow is not None and self._goal_node is not None

        state = self.state_manager.load(self.goal_id)

        if state.status != "active":
            print(f"Goal {self.goal_id} is {state.status}, stopping cycle")
            raise asyncio.CancelledError()

        # Merge: builtin/local vars < persisted state < initial (caller) vars
        variables = dict(self._initial_variables)
        variables.update(state.variables)

        # Execute the full workflow (may include @set nodes before @goal)
        runtime = AWLRuntime(
            self.agent,
            limits=RuntimeLimits(max_tool_calls=self._goal_node.max_actions),
        )
        result = await runtime.execute(self._workflow, variables=variables)

        # Check if success was met
        success_met = result.variables.get("_goal_success_met", False)

        # Persist state (exclude internal variables)
        state.variables = {k: v for k, v in result.variables.items() if not k.startswith("_goal_")}
        state.cycle_count += 1
        state.last_run = datetime.now().isoformat()

        if success_met:
            state.status = "completed"
            state.success_reason = result.variables.get("_goal_success_reason", "")
            print(f"Goal {self.goal_id} completed: {state.success_reason}")

        self.state_manager.save(self.goal_id, state)
        return result
