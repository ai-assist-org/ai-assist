"""Tests for AWL executor script-level locking."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_assist.awl_executor import _try_lock_script, run_awl_script


@pytest.fixture
def locks_dir(tmp_path):
    with patch("ai_assist.awl_executor.get_config_dir", return_value=tmp_path):
        yield tmp_path / "locks"


class TestTryLockScript:
    def test_acquire_lock_succeeds(self, locks_dir):
        script_path = Path("/home/user/scripts/triage.awl")
        fh = _try_lock_script(script_path)
        assert fh is not None
        assert not fh.closed
        fh.close()

    def test_second_lock_same_path_fails(self, locks_dir):
        script_path = Path("/home/user/scripts/triage.awl")
        fh1 = _try_lock_script(script_path)
        assert fh1 is not None

        fh2 = _try_lock_script(script_path)
        assert fh2 is None

        fh1.close()

    def test_lock_released_after_close(self, locks_dir):
        script_path = Path("/home/user/scripts/triage.awl")
        fh1 = _try_lock_script(script_path)
        assert fh1 is not None
        fh1.close()

        fh2 = _try_lock_script(script_path)
        assert fh2 is not None
        fh2.close()

    def test_different_paths_independent(self, locks_dir):
        path_a = Path("/home/user/scripts/triage.awl")
        path_b = Path("/home/user/scripts/ci_duty.awl")

        fh_a = _try_lock_script(path_a)
        fh_b = _try_lock_script(path_b)

        assert fh_a is not None
        assert fh_b is not None

        fh_a.close()
        fh_b.close()

    def test_same_basename_different_dirs(self, locks_dir):
        path_a = Path("/home/user/project1/triage.awl")
        path_b = Path("/home/user/project2/triage.awl")

        fh_a = _try_lock_script(path_a)
        fh_b = _try_lock_script(path_b)

        assert fh_a is not None
        assert fh_b is not None

        fh_a.close()
        fh_b.close()

    def test_lock_file_contains_script_path(self, locks_dir):
        script_path = Path("/home/user/scripts/triage.awl")
        fh = _try_lock_script(script_path)
        assert fh is not None

        lock_files = list(locks_dir.glob("*.lock"))
        assert len(lock_files) == 1
        assert lock_files[0].read_text() == str(script_path)

        fh.close()


class TestRunAwlScriptLocking:
    def _mock_workflow(self, tmp_path):
        """Return a mock workflow with no GoalNodes and its resolved path."""
        workflow = MagicMock()
        workflow.body = []
        awl_path = tmp_path / "test.awl"
        return workflow, awl_path

    @pytest.mark.asyncio
    async def test_skips_when_locked(self, tmp_path):
        workflow, awl_path = self._mock_workflow(tmp_path)

        with (
            patch("ai_assist.awl_executor.get_config_dir", return_value=tmp_path),
            patch("ai_assist.awl_executor.load_awl_workflow", return_value=(workflow, awl_path)),
            patch("ai_assist.awl_executor.load_local_variables", return_value={}),
            patch("ai_assist.awl_executor.get_reports_dir", return_value=tmp_path / "reports"),
        ):
            fh = _try_lock_script(awl_path)
            assert fh is not None

            try:
                result = await run_awl_script("test.awl", MagicMock())
                assert "Skipped" in result
                assert "already running" in result
            finally:
                fh.close()

    @pytest.mark.asyncio
    async def test_releases_lock_on_success(self, tmp_path):
        workflow, awl_path = self._mock_workflow(tmp_path)

        with (
            patch("ai_assist.awl_executor.get_config_dir", return_value=tmp_path),
            patch("ai_assist.awl_executor.load_awl_workflow", return_value=(workflow, awl_path)),
            patch("ai_assist.awl_executor.load_local_variables", return_value={}),
            patch("ai_assist.awl_executor.get_reports_dir", return_value=tmp_path / "reports"),
            patch("ai_assist.awl_runtime.AWLRuntime.execute", new_callable=AsyncMock) as mock_execute,
        ):
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.return_value = "Done"
            mock_execute.return_value = mock_result

            result = await run_awl_script("test.awl", MagicMock())
            assert result == "Done"

            fh = _try_lock_script(awl_path)
            assert fh is not None, "Lock should be released after execution"
            fh.close()

    @pytest.mark.asyncio
    async def test_releases_lock_on_error(self, tmp_path):
        workflow, awl_path = self._mock_workflow(tmp_path)

        with (
            patch("ai_assist.awl_executor.get_config_dir", return_value=tmp_path),
            patch("ai_assist.awl_executor.load_awl_workflow", return_value=(workflow, awl_path)),
            patch("ai_assist.awl_executor.load_local_variables", return_value={}),
            patch("ai_assist.awl_executor.get_reports_dir", return_value=tmp_path / "reports"),
            patch("ai_assist.awl_runtime.AWLRuntime.execute", new_callable=AsyncMock) as mock_execute,
        ):
            from ai_assist.awl_runtime import AWLRuntimeError

            mock_execute.side_effect = AWLRuntimeError("boom")

            with pytest.raises(RuntimeError):
                await run_awl_script("test.awl", MagicMock())

            fh = _try_lock_script(awl_path)
            assert fh is not None, "Lock should be released after error"
            fh.close()


class TestRunAwlScriptReturnValue:
    """run_awl_script returns text: callers (action engine, tools) treat it as a string."""

    async def _run(self, tmp_path, return_value):
        workflow = MagicMock()
        workflow.body = []
        with (
            patch("ai_assist.awl_executor.get_config_dir", return_value=tmp_path),
            patch("ai_assist.awl_executor.load_awl_workflow", return_value=(workflow, tmp_path / "test.awl")),
            patch("ai_assist.awl_executor.load_local_variables", return_value={}),
            patch("ai_assist.awl_executor.get_reports_dir", return_value=tmp_path / "reports"),
            patch("ai_assist.awl_runtime.AWLRuntime.execute", new_callable=AsyncMock) as mock_execute,
        ):
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.return_value = return_value
            mock_execute.return_value = mock_result
            return await run_awl_script("test.awl", MagicMock())

    @pytest.mark.asyncio
    async def test_list_return_value_is_json_text(self, tmp_path):
        value = [{"job_id": "j1", "ticket_key": "CNF-1"}, {"job_id": "j2", "ticket_key": "CNF-2"}]
        result = await self._run(tmp_path, value)
        assert isinstance(result, str)
        assert json.loads(result) == value

    @pytest.mark.asyncio
    async def test_string_return_value_unchanged(self, tmp_path):
        assert await self._run(tmp_path, "https://example.com/pull/1") == "https://example.com/pull/1"

    @pytest.mark.asyncio
    async def test_empty_return_value_gives_default_message(self, tmp_path):
        assert await self._run(tmp_path, None) == "Workflow completed successfully."
        assert await self._run(tmp_path, []) == "Workflow completed successfully."


class TestAllowSessionTmpdir:
    def test_whitelists_session_tmpdir(self, tmp_path):
        from ai_assist.awl_executor import _allow_session_tmpdir

        agent = MagicMock()
        agent.filesystem_tools.allowed_paths = [Path("/tmp/ai-assist")]
        session_dir = tmp_path / "ai-assist-xyz"
        session_dir.mkdir()

        with patch("ai_assist.awl_executor._get_session_tmpdir", return_value=str(session_dir)):
            _allow_session_tmpdir(agent)

        assert session_dir.resolve() in agent.filesystem_tools.allowed_paths

    def test_no_duplicate_entry(self, tmp_path):
        from ai_assist.awl_executor import _allow_session_tmpdir

        agent = MagicMock()
        session_dir = tmp_path / "ai-assist-xyz"
        session_dir.mkdir()
        agent.filesystem_tools.allowed_paths = [session_dir.resolve()]

        with patch("ai_assist.awl_executor._get_session_tmpdir", return_value=str(session_dir)):
            _allow_session_tmpdir(agent)

        assert agent.filesystem_tools.allowed_paths.count(session_dir.resolve()) == 1

    def test_no_filesystem_tools_is_noop(self):
        from ai_assist.awl_executor import _allow_session_tmpdir

        agent = MagicMock(spec=[])  # no filesystem_tools attribute
        _allow_session_tmpdir(agent)  # must not raise
