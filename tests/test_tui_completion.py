"""Tests for TUI completion with skills"""

from pathlib import Path

from prompt_toolkit.document import Document

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig
from ai_assist.skills_loader import SkillContent, SkillMetadata
from ai_assist.tui import AiAssistCompleter


def _make_skill(name, user_invocable=True, argument_hint=None):
    return SkillContent(
        metadata=SkillMetadata(
            name=name,
            description=f"{name} skill",
            skill_path=Path(f"/tmp/skills-cache/{name}"),
            source_type="local",
            user_invocable=user_invocable,
            argument_hint=argument_hint,
        ),
        body="body",
    )


def test_skill_command_completion():
    """Test that skill commands appear in completion"""
    # Create agent without real MCP connections
    config = AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-3-5-sonnet-20241022",
        mcp_servers={},
    )
    agent = AiAssistAgent(config)

    completer = AiAssistCompleter(agent=agent)

    # Test /skill prefix completion
    doc = Document("/ski", cursor_position=4)
    completions = list(completer.get_completions(doc, None))

    # Should suggest skill commands
    commands = [c.text for c in completions]
    assert "/skill/install" in commands
    assert "/skill/uninstall" in commands
    assert "/skill/list" in commands


def test_skill_install_completion():
    """Test /skill/install shows examples"""
    config = AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-3-5-sonnet-20241022",
        mcp_servers={},
    )
    agent = AiAssistAgent(config)

    completer = AiAssistCompleter(agent=agent)

    # Test /skill/install completion
    doc = Document("/skill/install ", cursor_position=15)  # "/skill/install " is 15 chars
    completions = list(completer.get_completions(doc, None))

    # Should suggest example patterns
    commands = [c.text for c in completions]
    assert any("anthropics/skills" in cmd for cmd in commands)
    assert any("/path/to/skill" in cmd for cmd in commands)


def test_skill_uninstall_completion():
    """Test /skill/uninstall completes with installed skills"""
    config = AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-3-5-sonnet-20241022",
        mcp_servers={},
    )
    agent = AiAssistAgent(config)

    # Set test file for skills
    agent.skills_manager.installed_skills_file = Path("/tmp/test-completion-skills.json")

    # Install test skill
    agent.skills_manager.install_skill("/tmp/test-skills/hello@main")

    completer = AiAssistCompleter(agent=agent)

    # Test /skill/uninstall completion
    doc = Document("/skill/uninstall ", cursor_position=17)
    completions = list(completer.get_completions(doc, None))

    # Should suggest installed skill
    commands = [c.text for c in completions]
    assert any("hello" in cmd for cmd in commands)

    # Clean up
    agent.skills_manager.uninstall_skill("hello")
    Path("/tmp/test-completion-skills.json").unlink()


def test_skill_list_completion():
    """Test /skill/list completes"""
    config = AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-3-5-sonnet-20241022",
        mcp_servers={},
    )
    agent = AiAssistAgent(config)

    completer = AiAssistCompleter(agent=agent)

    # Test /skill/list completion
    doc = Document("/skill/lis", cursor_position=10)
    completions = list(completer.get_completions(doc, None))

    # Should suggest /skill/list
    commands = [c.text for c in completions]
    assert "/skill/list" in commands


def _completer_with_skills(skills):
    config = AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-3-5-sonnet-20241022",
        mcp_servers={},
    )
    agent = AiAssistAgent(config)
    agent.skills_manager.loaded_skills = {s.metadata.name: s for s in skills}
    return AiAssistCompleter(agent=agent)


def test_user_invocable_skill_completion():
    """Installed user-invocable skills appear as /<skill-name>"""
    completer = _completer_with_skills([_make_skill("deploy")])

    doc = Document("/dep", cursor_position=4)
    commands = [c.text for c in completer.get_completions(doc, None)]

    assert "/deploy" in commands


def test_opted_out_skill_not_completed():
    """Skills with user-invocable: false are excluded from completion"""
    completer = _completer_with_skills([_make_skill("secret", user_invocable=False)])

    doc = Document("/sec", cursor_position=4)
    commands = [c.text for c in completer.get_completions(doc, None)]

    assert "/secret" not in commands


def test_skill_colliding_with_builtin_not_completed():
    """A skill named like a built-in must not shadow the built-in command"""
    completer = _completer_with_skills([_make_skill("status")])

    doc = Document("/stat", cursor_position=5)
    completions = list(completer.get_completions(doc, None))
    # Built-in /status is present exactly once (skill does not add a duplicate)
    assert [c.text for c in completions].count("/status") == 1
