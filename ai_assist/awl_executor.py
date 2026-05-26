"""Shared AWL script execution for action engine and task runner"""

from pathlib import Path

from .config import get_config_dir


async def run_awl_script(prompt: str, agent: object) -> str:
    """Execute an AWL script, handling both @goal and @start workflows."""
    from .awl_ast import GoalNode
    from .awl_parser import AWLParser

    awl_path = Path(prompt).expanduser()
    if not awl_path.is_absolute():
        awl_path = get_config_dir() / prompt

    if not awl_path.exists():
        raise FileNotFoundError(f"AWL script not found: {awl_path}")

    source = awl_path.read_text()
    workflow = AWLParser(source).parse()
    has_goal = any(isinstance(n, GoalNode) for n in workflow.body)

    if has_goal:
        from .goal_runner import GoalRunner
        from .goal_state import GoalStateManager

        state_manager = GoalStateManager(get_config_dir() / "state")
        runner = GoalRunner(awl_path, agent, state_manager)
        runner.load()
        await runner.run_cycle()

        lines = [f"Goal '{runner.goal_id}' cycle completed."]
        state = state_manager.load(runner.goal_id)
        lines.append(f"Status: {state.status} | Cycles: {state.cycle_count}")
        if state.success_reason:
            lines.append(f"Success: {state.success_reason}")
        return "\n".join(lines)

    from .awl_runtime import AWLRuntime

    runtime = AWLRuntime(agent)
    result = await runtime.execute(workflow)
    if not result.success:
        raise RuntimeError(
            f"AWL workflow failed: {result.task_outcomes[-1].summary if result.task_outcomes else 'unknown error'}"
        )
    return result.return_value or "Workflow completed successfully."
