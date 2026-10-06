"""Tests for GoalRunner"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_assist.goal_runner import GoalRunner
from ai_assist.goal_state import GoalState, GoalStateManager

AWL_BASIC = """\
@start

@goal test_goal
  Success: Rate below 10%

  @task check
  Goal: Check failure rate.
  Expose: failure_rate
  @end

@end

@end
"""

AWL_WITH_SET = """\
@start

@set product = "OCP 4.19"

@goal track
  Success: Product is stable

  @task check
  Goal: Check ${product} status.
  Expose: status
  @end

@end

@end
"""


def _make_mock_agent():
    agent = MagicMock()
    agent.config.jev_enabled = False
    agent.config.jev_api_key = None
    agent.query = AsyncMock(return_value='```json\n{"failure_rate": 5}\n```')
    return agent


class TestGoalRunner:
    def test_load_and_properties(self, tmp_path):
        awl_path = tmp_path / "test.awl"
        awl_path.write_text(AWL_BASIC)

        agent = _make_mock_agent()
        state_mgr = GoalStateManager(tmp_path / "state")
        runner = GoalRunner(awl_path, agent, state_mgr)
        runner.load()

        assert runner.goal_id == "test_goal"

    def test_load_no_goal_raises(self, tmp_path):
        awl_path = tmp_path / "test.awl"
        awl_path.write_text("@start\n@task t1\nGoal: Do.\n@end\n@end")

        agent = _make_mock_agent()
        state_mgr = GoalStateManager(tmp_path / "state")
        runner = GoalRunner(awl_path, agent, state_mgr)

        with pytest.raises(ValueError, match="Expected exactly 1"):
            runner.load()

    @pytest.mark.asyncio
    async def test_run_cycle_persists_variables(self, tmp_path):
        awl_path = tmp_path / "test.awl"
        awl_path.write_text(AWL_BASIC)

        call_count = 0

        async def mock_query(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return '```json\n{"failure_rate": 12}\n```'
            return '```json\n{"success_met": false, "reason": "rate is 12%"}\n```'

        agent = MagicMock()
        agent.config.jev_enabled = False
        agent.config.jev_api_key = None
        agent.query = AsyncMock(side_effect=mock_query)

        state_mgr = GoalStateManager(tmp_path / "state")
        runner = GoalRunner(awl_path, agent, state_mgr)
        runner.load()

        await runner.run_cycle()

        # Check persisted state
        state = state_mgr.load("test_goal")
        assert state.cycle_count == 1
        assert state.variables["failure_rate"] == 12
        assert state.status == "active"
        assert state.last_run is not None

    @pytest.mark.asyncio
    async def test_run_cycle_completes_on_success(self, tmp_path):
        awl_path = tmp_path / "test.awl"
        awl_path.write_text(AWL_BASIC)

        call_count = 0

        async def mock_query(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return '```json\n{"failure_rate": 3}\n```'
            return '```json\n{"success_met": true, "reason": "Rate is 3%, below 10%"}\n```'

        agent = MagicMock()
        agent.config.jev_enabled = False
        agent.config.jev_api_key = None
        agent.query = AsyncMock(side_effect=mock_query)

        state_mgr = GoalStateManager(tmp_path / "state")
        runner = GoalRunner(awl_path, agent, state_mgr)
        runner.load()

        await runner.run_cycle()

        state = state_mgr.load("test_goal")
        assert state.status == "completed"
        assert "3%" in state.success_reason

    @pytest.mark.asyncio
    async def test_run_cycle_skips_inactive_goal(self, tmp_path):
        awl_path = tmp_path / "test.awl"
        awl_path.write_text(AWL_BASIC)

        agent = _make_mock_agent()
        state_mgr = GoalStateManager(tmp_path / "state")

        # Pre-set state to paused
        state_mgr.save("test_goal", GoalState(status="paused"))

        runner = GoalRunner(awl_path, agent, state_mgr)
        runner.load()

        with pytest.raises(asyncio.CancelledError):
            await runner.run_cycle()

        # Agent should not have been called
        agent.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_cycle_preserves_variables_across_cycles(self, tmp_path):
        """Variables from cycle 1 are available in cycle 2"""
        awl_path = tmp_path / "test.awl"
        awl_path.write_text(AWL_BASIC)

        cycle = 0

        async def mock_query(prompt, **kwargs):
            nonlocal cycle
            cycle += 1
            if cycle <= 2:
                # Cycle 1: task + success eval
                if cycle == 1:
                    return '```json\n{"failure_rate": 15}\n```'
                return '```json\n{"success_met": false, "reason": "still high"}\n```'
            else:
                # Cycle 2: task + success eval
                if cycle == 3:
                    return '```json\n{"failure_rate": 8}\n```'
                return '```json\n{"success_met": true, "reason": "below 10%"}\n```'

        agent = MagicMock()
        agent.config.jev_enabled = False
        agent.config.jev_api_key = None
        agent.query = AsyncMock(side_effect=mock_query)

        state_mgr = GoalStateManager(tmp_path / "state")
        runner = GoalRunner(awl_path, agent, state_mgr)
        runner.load()

        # Cycle 1
        await runner.run_cycle()
        state = state_mgr.load("test_goal")
        assert state.variables["failure_rate"] == 15
        assert state.status == "active"

        # Cycle 2
        await runner.run_cycle()
        state = state_mgr.load("test_goal")
        assert state.variables["failure_rate"] == 8
        assert state.status == "completed"
