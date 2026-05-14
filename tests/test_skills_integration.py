"""Integration tests for skills system"""

from pathlib import Path

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig


def _prompt_text(blocks: list[dict]) -> str:
    return "\n\n".join(b["text"] for b in blocks)


def test_skills_integration_with_agent():
    """Test that skills are loaded and integrated into agent"""
    import tempfile

    # Create minimal config
    config = AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-3-5-sonnet-20241022",
        mcp_servers={},
    )

    # Create agent
    agent = AiAssistAgent(config)

    # Use temporary file for testing to avoid overwriting user's config
    temp_dir = Path(tempfile.mkdtemp())
    installed_file = temp_dir / "installed-skills.json"
    agent.skills_manager.installed_skills_file = installed_file

    # Verify skills system is initialized
    assert agent.skills_loader is not None
    assert agent.skills_manager is not None

    # Install test skill
    result = agent.skills_manager.install_skill("/tmp/test-skills/hello@main")
    assert "installed successfully" in result

    # Verify skill is loaded
    assert len(agent.skills_manager.installed_skills) == 1
    assert "hello" in agent.skills_manager.loaded_skills

    # Verify system prompt includes skill summary (progressive disclosure)
    system_prompt = _prompt_text(agent._build_system_prompt())
    assert "Agent Skills" in system_prompt
    assert "hello" in system_prompt
    assert "introspection__get_skill_help" in system_prompt
    # Body content should NOT be in the prompt (progressive disclosure)
    assert "Hello Skill" not in system_prompt

    # Clean up
    agent.skills_manager.uninstall_skill("hello")
    if installed_file.exists():
        installed_file.unlink()
    temp_dir.rmdir()


def test_skills_persistence_across_sessions():
    """Test that skills persist across multiple agent instances"""
    # Create temp installed skills file
    import tempfile

    temp_dir = Path(tempfile.mkdtemp())
    installed_file = temp_dir / "installed-skills.json"

    # Create first agent and install skill
    config1 = AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-3-5-sonnet-20241022",
        mcp_servers={},
    )
    agent1 = AiAssistAgent(config1)
    agent1.skills_manager.installed_skills_file = installed_file

    agent1.skills_manager.install_skill("/tmp/test-skills/hello@main")
    assert len(agent1.skills_manager.installed_skills) == 1

    # Create second agent and verify skill is loaded
    config2 = AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-3-5-sonnet-20241022",
        mcp_servers={},
    )
    agent2 = AiAssistAgent(config2)
    agent2.skills_manager.installed_skills_file = installed_file
    agent2.skills_manager.load_installed_skills()

    assert len(agent2.skills_manager.installed_skills) == 1
    assert "hello" in agent2.skills_manager.loaded_skills

    # Clean up
    agent2.skills_manager.uninstall_skill("hello")
    installed_file.unlink()
    temp_dir.rmdir()
