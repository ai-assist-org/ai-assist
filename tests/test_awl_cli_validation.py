"""Tests for AWL CLI input variable validation"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_assist.main import run_awl_script


@pytest.mark.asyncio
async def test_run_awl_script_validates_missing_variables(tmp_path, capsys):
    """Test that run_awl_script validates and reports missing input variables"""
    # Create AWL script with required variables
    script = tmp_path / "test.awl"
    script.write_text("""@start
@set greeting = Hello ${name}
@set location = ${city}
@end
""")

    agent = MagicMock()

    # Call without providing required variables should exit
    with pytest.raises(SystemExit) as exc_info:
        await run_awl_script(agent, str(script), variables={})

    assert exc_info.value.code == 1

    # Check error message
    captured = capsys.readouterr()
    assert "Missing required input variables" in captured.out
    assert "name" in captured.out
    assert "city" in captured.out
    assert "Usage:" in captured.out


@pytest.mark.asyncio
async def test_run_awl_script_validates_typo_in_variable_name(tmp_path, capsys):
    """Test that run_awl_script detects typos in variable names"""
    script = tmp_path / "test.awl"
    script.write_text("""@start
@set msg = Processing ${subject}
@end
""")

    agent = MagicMock()

    # Provide 'subjec' instead of 'subject' (typo)
    with pytest.raises(SystemExit) as exc_info:
        await run_awl_script(agent, str(script), variables={"subjec": "Test"})

    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert "Missing required input variables" in captured.out
    assert "subject" in captured.out
    assert "Required variables: ['subject']" in captured.out
    assert "Provided variables: ['subjec']" in captured.out


@pytest.mark.asyncio
async def test_run_awl_script_passes_with_all_variables(tmp_path):
    """Test that run_awl_script executes when all variables are provided"""
    script = tmp_path / "test.awl"
    script.write_text("""@start
@set msg = Hello ${name}
@end
""")

    agent = MagicMock()

    # Mock AWLRuntime to avoid actual execution
    with patch("ai_assist.awl_runtime.AWLRuntime") as mock_runtime:
        mock_instance = AsyncMock()
        mock_instance.execute = AsyncMock(return_value=MagicMock(success=True, return_value=None, task_outcomes=[]))
        mock_runtime.return_value = mock_instance

        # Should not raise when all variables provided
        await run_awl_script(agent, str(script), variables={"name": "World"})

        # Verify runtime was called
        mock_instance.execute.assert_called_once()


@pytest.mark.asyncio
async def test_run_awl_script_no_variables_required(tmp_path):
    """Test that run_awl_script works when no variables are required"""
    script = tmp_path / "test.awl"
    script.write_text("""@start
@set x = 42
@end
""")

    agent = MagicMock()

    with patch("ai_assist.awl_runtime.AWLRuntime") as mock_runtime:
        mock_instance = AsyncMock()
        mock_instance.execute = AsyncMock(return_value=MagicMock(success=True, return_value=None, task_outcomes=[]))
        mock_runtime.return_value = mock_instance

        # Should work with no variables
        await run_awl_script(agent, str(script), variables={})
        mock_instance.execute.assert_called_once()


@pytest.mark.asyncio
async def test_run_awl_script_extra_variables_allowed(tmp_path):
    """Test that providing extra variables is allowed"""
    script = tmp_path / "test.awl"
    script.write_text("""@start
@set msg = Hello ${name}
@end
""")

    agent = MagicMock()

    with patch("ai_assist.awl_runtime.AWLRuntime") as mock_runtime:
        mock_instance = AsyncMock()
        mock_instance.execute = AsyncMock(return_value=MagicMock(success=True, return_value=None, task_outcomes=[]))
        mock_runtime.return_value = mock_instance

        # Extra variables should be allowed
        await run_awl_script(agent, str(script), variables={"name": "World", "extra": "value"})
        mock_instance.execute.assert_called_once()
