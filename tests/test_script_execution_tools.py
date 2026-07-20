"""Tests for script execution tools with security focus"""

import os
from unittest.mock import MagicMock

import pytest

from ai_assist.config import AiAssistConfig
from ai_assist.script_execution_tools import ScriptExecutionTools
from ai_assist.skills_loader import SkillsLoader
from ai_assist.skills_manager import SkillsManager


@pytest.fixture
def temp_skill_dir(tmp_path):
    """Create a temporary skill directory with scripts"""
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()

    # Create SKILL.md
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("""---
name: test-skill
description: Test skill for script execution
allowed-tools: "internal__execute_skill_script"
---
Test skill with scripts.
""")

    # Create scripts directory
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()

    # Create a simple test script
    hello_script = scripts_dir / "hello.sh"
    hello_script.write_text("""#!/bin/bash
echo "Hello from script!"
""")
    hello_script.chmod(0o755)

    # Create a script that uses env vars
    env_script = scripts_dir / "print_env.sh"
    env_script.write_text("""#!/bin/bash
env | sort
""")
    env_script.chmod(0o755)

    # Create a script that sleeps
    sleep_script = scripts_dir / "sleep.sh"
    sleep_script.write_text("""#!/bin/bash
sleep 60
""")
    sleep_script.chmod(0o755)

    # Create a script with large output
    large_output_script = scripts_dir / "large_output.sh"
    large_output_script.write_text("""#!/bin/bash
python3 -c "print('x' * 30000)"
""")
    large_output_script.chmod(0o755)

    # Create a Python script with args
    args_script = scripts_dir / "echo_args.py"
    args_script.write_text("""#!/usr/bin/env python3
import sys
print(' '.join(sys.argv[1:]))
""")
    args_script.chmod(0o755)

    # Create a script that outputs UTF-8 text
    utf8_script = scripts_dir / "utf8_output.py"
    utf8_script.write_text("""#!/usr/bin/env python3
import sys
sys.stdout.buffer.write("Résultats: café, naïve, Ωmega, 日本語\\n".encode("utf-8"))
""")
    utf8_script.chmod(0o755)

    return skill_dir


@pytest.fixture
def skills_manager_with_skill(temp_skill_dir):
    """Create a skills manager with the test skill installed"""
    loader = SkillsLoader()
    manager = SkillsManager(loader)

    # Manually load the skill
    content = loader.load_skill_from_local(temp_skill_dir)
    manager.loaded_skills["test-skill"] = content
    manager.installed_skills.append(
        MagicMock(name="test-skill", source=str(temp_skill_dir), source_type="local", branch="main")
    )

    return manager


def test_script_execution_disabled_by_default():
    """Verify scripts disabled by default for security"""
    # Ensure env var is not set
    if "AI_ASSIST_ALLOW_SCRIPT_EXECUTION" in os.environ:
        del os.environ["AI_ASSIST_ALLOW_SCRIPT_EXECUTION"]

    config = AiAssistConfig(anthropic_api_key="test")
    assert config.allow_skill_script_execution is False


def test_script_execution_enabled_via_env():
    """Verify scripts can be enabled via environment variable"""
    # Save original value
    original = os.environ.get("AI_ASSIST_ALLOW_SCRIPT_EXECUTION")

    try:
        os.environ["AI_ASSIST_ALLOW_SCRIPT_EXECUTION"] = "true"
        config = AiAssistConfig(anthropic_api_key="test")
        assert config.allow_skill_script_execution is True
    finally:
        # Restore original value
        if original is None:
            os.environ.pop("AI_ASSIST_ALLOW_SCRIPT_EXECUTION", None)
        else:
            os.environ["AI_ASSIST_ALLOW_SCRIPT_EXECUTION"] = original


def test_get_tool_definitions_disabled(skills_manager_with_skill):
    """Verify no tools returned when disabled"""
    config = AiAssistConfig(anthropic_api_key="test", allow_skill_script_execution=False)
    tools = ScriptExecutionTools(skills_manager_with_skill, config)

    definitions = tools.get_tool_definitions()
    assert definitions == []


def test_get_tool_definitions_enabled(skills_manager_with_skill):
    """Verify tool definitions returned when enabled"""
    config = AiAssistConfig(anthropic_api_key="test", allow_skill_script_execution=True)
    tools = ScriptExecutionTools(skills_manager_with_skill, config)

    definitions = tools.get_tool_definitions()
    assert len(definitions) == 1
    assert definitions[0]["name"] == "internal__execute_skill_script"
    assert "skill_name" in definitions[0]["input_schema"]["properties"]
    assert "script_name" in definitions[0]["input_schema"]["properties"]


@pytest.mark.asyncio
async def test_execute_when_disabled(skills_manager_with_skill):
    """Verify execution blocked when disabled"""
    config = AiAssistConfig(anthropic_api_key="test", allow_skill_script_execution=False)
    tools = ScriptExecutionTools(skills_manager_with_skill, config)

    result = await tools.execute_tool("execute_skill_script", {"skill_name": "test-skill", "script_name": "hello.sh"})

    assert "Error" in result
    assert "disabled" in result.lower()


@pytest.mark.asyncio
async def test_execute_utf8_output(skills_manager_with_skill):
    """Test that UTF-8 output from scripts is correctly decoded"""
    config = AiAssistConfig(anthropic_api_key="test", allow_skill_script_execution=True)
    tools = ScriptExecutionTools(skills_manager_with_skill, config)

    result = await tools.execute_tool(
        "execute_skill_script", {"skill_name": "test-skill", "script_name": "utf8_output.py"}
    )

    assert "Résultats" in result
    assert "café" in result
    assert "naïve" in result
    assert "Ωmega" in result
    assert "日本語" in result


@pytest.mark.asyncio
async def test_execute_simple_script(skills_manager_with_skill):
    """Test executing a basic script"""
    config = AiAssistConfig(anthropic_api_key="test", allow_skill_script_execution=True)
    tools = ScriptExecutionTools(skills_manager_with_skill, config)

    result = await tools.execute_tool("execute_skill_script", {"skill_name": "test-skill", "script_name": "hello.sh"})

    assert "Hello from script!" in result
    assert "Error" not in result


@pytest.mark.asyncio
async def test_script_with_arguments(skills_manager_with_skill):
    """Test executing a script with command-line arguments"""
    config = AiAssistConfig(anthropic_api_key="test", allow_skill_script_execution=True)
    tools = ScriptExecutionTools(skills_manager_with_skill, config)

    result = await tools.execute_tool(
        "execute_skill_script",
        {"skill_name": "test-skill", "script_name": "echo_args.py", "args": ["hello", "world"]},
    )

    assert "hello world" in result


@pytest.mark.asyncio
async def test_path_validation_blocks_traversal(skills_manager_with_skill, tmp_path):
    """Test directory traversal is blocked"""
    config = AiAssistConfig(anthropic_api_key="test", allow_skill_script_execution=True)
    tools = ScriptExecutionTools(skills_manager_with_skill, config)

    # Try to execute a script outside the skill directory
    result = await tools.execute_tool(
        "execute_skill_script", {"skill_name": "test-skill", "script_name": "../../../etc/passwd"}
    )

    assert "Error" in result
    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_permission_enforcement_no_allowed_tools(temp_skill_dir):
    """Test skills with scripts but no allowed-tools field are permitted (default allow)"""
    # Create a skill without allowed-tools but with scripts
    skill_dir = temp_skill_dir.parent / "no-permission-skill"
    skill_dir.mkdir()

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("""---
name: no-permission-skill
description: Skill without explicit script execution permission
---
This skill has scripts but no allowed-tools declaration.
""")

    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    script = scripts_dir / "test.sh"
    script.write_text("#!/bin/bash\necho hello")
    script.chmod(0o755)

    # Load skill
    loader = SkillsLoader()
    manager = SkillsManager(loader)
    content = loader.load_skill_from_local(skill_dir)
    manager.loaded_skills["no-permission-skill"] = content

    config = AiAssistConfig(anthropic_api_key="test", allow_skill_script_execution=True)
    tools = ScriptExecutionTools(manager, config)

    result = await tools.execute_tool(
        "execute_skill_script", {"skill_name": "no-permission-skill", "script_name": "test.sh"}
    )

    # Should be allowed - user already opted in globally via allow_skill_script_execution
    assert "Error" not in result or "not allowed" not in result


@pytest.mark.asyncio
async def test_permission_enforcement_explicit_denial(temp_skill_dir):
    """Test that skills with allowed-tools that excludes script execution are blocked"""
    # Create a skill that explicitly allows other tools but not script execution
    skill_dir = temp_skill_dir.parent / "explicit-denial-skill"
    skill_dir.mkdir()

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("""---
name: explicit-denial-skill
description: Skill that explicitly denies script execution
allowed-tools: "some_other_tool"
---
This skill explicitly sets allowed-tools without script execution.
""")

    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    script = scripts_dir / "test.sh"
    script.write_text("#!/bin/bash\necho hello")
    script.chmod(0o755)

    # Load skill
    loader = SkillsLoader()
    manager = SkillsManager(loader)
    content = loader.load_skill_from_local(skill_dir)
    manager.loaded_skills["explicit-denial-skill"] = content

    config = AiAssistConfig(anthropic_api_key="test", allow_skill_script_execution=True)
    tools = ScriptExecutionTools(manager, config)

    result = await tools.execute_tool(
        "execute_skill_script", {"skill_name": "explicit-denial-skill", "script_name": "test.sh"}
    )

    # Should fail - skill explicitly excludes script execution
    assert "Error" in result
    assert "not allowed" in result.lower()


@pytest.mark.asyncio
async def test_timeout_enforcement(skills_manager_with_skill):
    """Test scripts timeout after configured timeout"""
    config = AiAssistConfig(anthropic_api_key="test", allow_skill_script_execution=True)
    tools = ScriptExecutionTools(skills_manager_with_skill, config, timeout=3)

    result = await tools.execute_tool("execute_skill_script", {"skill_name": "test-skill", "script_name": "sleep.sh"})

    assert "Error" in result
    assert "timed out" in result.lower()


@pytest.mark.asyncio
async def test_environment_filtering(skills_manager_with_skill):
    """Test API keys filtered from environment"""
    # Set sensitive env vars
    os.environ["ANTHROPIC_API_KEY"] = "sk-test-secret-key"
    os.environ["GITHUB_TOKEN"] = "ghp_test_token"
    os.environ["SAFE_VAR"] = "safe_value"

    try:
        config = AiAssistConfig(anthropic_api_key="test", allow_skill_script_execution=True)
        tools = ScriptExecutionTools(skills_manager_with_skill, config)

        result = await tools.execute_tool(
            "execute_skill_script", {"skill_name": "test-skill", "script_name": "print_env.sh"}
        )

        # Sensitive vars should NOT be in output
        assert "sk-test-secret-key" not in result
        assert "ghp_test_token" not in result
        assert "ANTHROPIC_API_KEY" not in result
        assert "GITHUB_TOKEN" not in result

        # Safe var should be in output
        assert "SAFE_VAR" in result
        assert "safe_value" in result

        # PATH should be available
        assert "PATH" in result

    finally:
        del os.environ["ANTHROPIC_API_KEY"]
        del os.environ["GITHUB_TOKEN"]
        del os.environ["SAFE_VAR"]


@pytest.mark.asyncio
async def test_nonexistent_skill(skills_manager_with_skill):
    """Test error when skill not installed"""
    config = AiAssistConfig(anthropic_api_key="test", allow_skill_script_execution=True)
    tools = ScriptExecutionTools(skills_manager_with_skill, config)

    result = await tools.execute_tool(
        "execute_skill_script", {"skill_name": "nonexistent-skill", "script_name": "hello.sh"}
    )

    assert "Error" in result
    assert "not installed" in result.lower()


@pytest.mark.asyncio
async def test_nonexistent_script(skills_manager_with_skill):
    """Test error when script not found"""
    config = AiAssistConfig(anthropic_api_key="test", allow_skill_script_execution=True)
    tools = ScriptExecutionTools(skills_manager_with_skill, config)

    result = await tools.execute_tool(
        "execute_skill_script", {"skill_name": "test-skill", "script_name": "nonexistent.sh"}
    )

    assert "Error" in result
    assert "not found" in result.lower()
    # Should list available scripts
    assert "Available scripts:" in result


@pytest.mark.asyncio
async def test_script_exit_code_nonzero(skills_manager_with_skill, temp_skill_dir):
    """Test handling of script that fails"""
    # Create a script that fails
    scripts_dir = temp_skill_dir / "scripts"
    fail_script = scripts_dir / "fail.sh"
    fail_script.write_text("""#!/bin/bash
echo "Error message" >&2
exit 1
""")
    fail_script.chmod(0o755)

    # Reload skill to pick up new script
    loader = SkillsLoader()
    content = loader.load_skill_from_local(temp_skill_dir)
    skills_manager_with_skill.loaded_skills["test-skill"] = content

    config = AiAssistConfig(anthropic_api_key="test", allow_skill_script_execution=True)
    tools = ScriptExecutionTools(skills_manager_with_skill, config)

    result = await tools.execute_tool("execute_skill_script", {"skill_name": "test-skill", "script_name": "fail.sh"})

    assert "Script failed" in result
    assert "exit code 1" in result
    assert "Error message" in result


def test_system_prompt_lists_skill_without_script_details(skills_manager_with_skill):
    """Test that system prompt lists skill with script names for tool selection guidance"""
    result = skills_manager_with_skill.get_system_prompt_section()

    # Should list the skill with its script names
    assert "test-skill" in result
    assert "hello.sh" in result
    # Full skill body is still behind progressive disclosure
    assert "Script Execution" not in result


# --- Skill environment variable allowlist tests ---


def test_skill_env_save_and_load(tmp_path, monkeypatch):
    """Test saving and loading skill env var config"""
    monkeypatch.setattr("ai_assist.script_execution_tools.get_config_dir", lambda: tmp_path)

    ScriptExecutionTools.save_skill_env("my-skill", "MY_API_KEY")
    ScriptExecutionTools.save_skill_env("my-skill", "MY_SECRET")

    config = ScriptExecutionTools.list_skill_env("my-skill")
    assert "MY_API_KEY" in config["my-skill"]
    assert "MY_SECRET" in config["my-skill"]


def test_skill_env_no_duplicates(tmp_path, monkeypatch):
    """Test that saving same var twice doesn't duplicate"""
    monkeypatch.setattr("ai_assist.script_execution_tools.get_config_dir", lambda: tmp_path)

    ScriptExecutionTools.save_skill_env("my-skill", "MY_KEY")
    ScriptExecutionTools.save_skill_env("my-skill", "MY_KEY")

    config = ScriptExecutionTools.list_skill_env("my-skill")
    assert config["my-skill"].count("MY_KEY") == 1


def test_skill_env_remove(tmp_path, monkeypatch):
    """Test removing a skill env var"""
    monkeypatch.setattr("ai_assist.script_execution_tools.get_config_dir", lambda: tmp_path)

    ScriptExecutionTools.save_skill_env("my-skill", "MY_KEY")
    assert ScriptExecutionTools.remove_skill_env("my-skill", "MY_KEY") is True

    config = ScriptExecutionTools.list_skill_env("my-skill")
    assert config["my-skill"] == []


def test_skill_env_remove_nonexistent(tmp_path, monkeypatch):
    """Test removing a var that doesn't exist returns False"""
    monkeypatch.setattr("ai_assist.script_execution_tools.get_config_dir", lambda: tmp_path)

    assert ScriptExecutionTools.remove_skill_env("no-skill", "NO_VAR") is False


def test_skill_env_list_empty(tmp_path, monkeypatch):
    """Test listing when no config exists"""
    monkeypatch.setattr("ai_assist.script_execution_tools.get_config_dir", lambda: tmp_path)

    config = ScriptExecutionTools.list_skill_env()
    assert config == {}


def test_skill_env_filtering_with_allowlist(skills_manager_with_skill, tmp_path, monkeypatch):
    """Test that allowed env vars pass through to scripts"""
    monkeypatch.setattr("ai_assist.script_execution_tools.get_config_dir", lambda: tmp_path)

    # Save an allowed var for the skill
    ScriptExecutionTools.save_skill_env("test-skill", "GOOGLE_APPLICATION_CREDENTIALS")

    # Set the env var
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/creds.json"

    try:
        config = AiAssistConfig(anthropic_api_key="test", allow_skill_script_execution=True)
        tools = ScriptExecutionTools(skills_manager_with_skill, config)

        # Get the safe environment for this skill
        safe_env = tools._get_safe_environment("test-skill")
        assert "GOOGLE_APPLICATION_CREDENTIALS" in safe_env
        assert safe_env["GOOGLE_APPLICATION_CREDENTIALS"] == "/path/to/creds.json"

        # Without skill name, it should be filtered
        safe_env_no_skill = tools._get_safe_environment()
        assert "GOOGLE_APPLICATION_CREDENTIALS" not in safe_env_no_skill
    finally:
        del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]


def test_skill_env_filtering_other_skill_not_allowed(skills_manager_with_skill, tmp_path, monkeypatch):
    """Test that env vars allowed for one skill don't leak to another"""
    monkeypatch.setattr("ai_assist.script_execution_tools.get_config_dir", lambda: tmp_path)

    ScriptExecutionTools.save_skill_env("other-skill", "GOOGLE_API_KEY")

    os.environ["GOOGLE_API_KEY"] = "secret-key"

    try:
        config = AiAssistConfig(anthropic_api_key="test", allow_skill_script_execution=True)
        tools = ScriptExecutionTools(skills_manager_with_skill, config)

        # test-skill should NOT get other-skill's allowed vars
        safe_env = tools._get_safe_environment("test-skill")
        assert "GOOGLE_API_KEY" not in safe_env
    finally:
        del os.environ["GOOGLE_API_KEY"]
