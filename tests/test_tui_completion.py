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


def test_skill_update_completion():
    """Test /skill/update completes with installed skills"""
    config = AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-3-5-sonnet-20241022",
        mcp_servers={},
    )
    agent = AiAssistAgent(config)
    agent.skills_manager.installed_skills_file = Path("/tmp/test-completion-skills-update.json")
    agent.skills_manager.install_skill("/tmp/test-skills/hello@main")

    completer = AiAssistCompleter(agent=agent)
    doc = Document("/skill/update ", cursor_position=14)
    commands = [c.text for c in completer.get_completions(doc, None)]
    assert "/skill/update hello" in commands

    agent.skills_manager.uninstall_skill("hello")
    Path("/tmp/test-completion-skills-update.json").unlink()


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


# --- plugin completion ---


def _agent():
    config = AiAssistConfig(
        anthropic_api_key="test-key",
        model="claude-3-5-sonnet-20241022",
        mcp_servers={},
    )
    return AiAssistAgent(config)


def test_plugin_command_completion():
    """/plugin/* commands appear in completion"""
    completer = AiAssistCompleter(agent=_agent())
    doc = Document("/plug", cursor_position=5)
    commands = [c.text for c in completer.get_completions(doc, None)]
    assert "/plugin/install" in commands
    assert "/plugin/list" in commands
    assert "/plugin/uninstall" in commands


def test_plugin_install_completion_shows_examples():
    completer = AiAssistCompleter(agent=_agent())
    doc = Document("/plugin/install ", cursor_position=16)
    commands = [c.text for c in completer.get_completions(doc, None)]
    assert any("owner/repo" in cmd for cmd in commands)
    assert any("/path/to/plugin" in cmd for cmd in commands)


def test_plugin_uninstall_completion_lists_installed():
    from ai_assist.plugins_manager import InstalledPlugin

    agent = _agent()
    agent.plugins_manager.installed_plugins = [
        InstalledPlugin(
            name="acme",
            source="owner/acme",
            source_type="git",
            branch="main",
            installed_at="now",
            cache_path="/tmp/acme",
        )
    ]
    completer = AiAssistCompleter(agent=agent)
    doc = Document("/plugin/uninstall ", cursor_position=18)
    commands = [c.text for c in completer.get_completions(doc, None)]
    assert any("acme" in cmd for cmd in commands)


def test_plugin_update_completion_lists_installed():
    """/plugin/update completes with installed plugin names"""
    from ai_assist.plugins_manager import InstalledPlugin

    agent = _agent()
    agent.plugins_manager.installed_plugins = [
        InstalledPlugin(
            name="acme",
            source="owner/acme",
            source_type="git",
            branch="main",
            installed_at="now",
            cache_path="/tmp/acme",
        )
    ]
    completer = AiAssistCompleter(agent=agent)
    doc = Document("/plugin/update ", cursor_position=15)
    commands = [c.text for c in completer.get_completions(doc, None)]
    assert "/plugin/update acme" in commands


def test_plugin_install_completion_lists_marketplace_names(tmp_path):
    """/plugin/install completes with plugin names from registered marketplaces"""
    import json

    from ai_assist.plugins_manager import Marketplace

    manifest_dir = tmp_path / "market" / ".claude-plugin"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "marketplace.json").write_text(
        json.dumps(
            {
                "name": "acme-market",
                "plugins": [{"name": "acme-tools", "source": {"source": "github", "repo": "acme/tools"}}],
            }
        )
    )

    agent = _agent()
    agent.plugins_manager.marketplaces = [
        Marketplace(name="acme-market", source="acme/market", branch="main", cache_path=str(tmp_path / "market"))
    ]
    completer = AiAssistCompleter(agent=agent)
    doc = Document("/plugin/install acme", cursor_position=20)
    commands = [c.text for c in completer.get_completions(doc, None)]
    assert "/plugin/install acme-tools" in commands


def test_plugin_marketplace_add_completion_lists_well_known():
    """/plugin/marketplace add suggests the well-known public marketplaces"""
    completer = AiAssistCompleter(agent=_agent())
    doc = Document("/plugin/marketplace add ", cursor_position=24)
    commands = [c.text for c in completer.get_completions(doc, None)]
    assert any("anthropics/claude-plugins-official" in cmd for cmd in commands)
    assert any("anthropics/claude-plugins-community" in cmd for cmd in commands)


def test_plugin_marketplace_subcommand_completion():
    """/plugin/marketplace <space> suggests the add|update|list subcommands"""
    completer = AiAssistCompleter(agent=_agent())
    doc = Document("/plugin/marketplace ", cursor_position=20)
    commands = [c.text for c in completer.get_completions(doc, None)]
    assert "/plugin/marketplace add" in commands
    assert "/plugin/marketplace update" in commands
    assert "/plugin/marketplace list" in commands


def test_plugin_marketplace_subcommand_completion_prefix():
    """/plugin/marketplace u<partial> narrows to matching subcommands"""
    completer = AiAssistCompleter(agent=_agent())
    doc = Document("/plugin/marketplace u", cursor_position=21)
    commands = [c.text for c in completer.get_completions(doc, None)]
    assert commands == ["/plugin/marketplace update"]


def test_plugin_marketplace_update_completion_lists_registered():
    """/plugin/marketplace update completes with registered marketplace names"""
    from ai_assist.plugins_manager import Marketplace

    agent = _agent()
    agent.plugins_manager.marketplaces = [
        Marketplace(name="ai-helpers", source="openshift-eng/ai-helpers", branch="main", cache_path="/tmp/ai-helpers")
    ]
    completer = AiAssistCompleter(agent=agent)
    doc = Document("/plugin/marketplace update ", cursor_position=27)
    commands = [c.text for c in completer.get_completions(doc, None)]
    assert "/plugin/marketplace update ai-helpers" in commands


def test_plugin_namespace_completion_lists_subcommands():
    """/plugin <space> suggests all /plugin/* commands"""
    completer = AiAssistCompleter(agent=_agent())
    doc = Document("/plugin ", cursor_position=8)
    commands = [c.text for c in completer.get_completions(doc, None)]
    assert "/plugin/install" in commands
    assert "/plugin/uninstall" in commands
    assert "/plugin/update" in commands
    assert "/plugin/list" in commands
    assert "/plugin/marketplace" in commands


def test_skill_namespace_completion_lists_subcommands():
    """/skill <space> suggests all /skill/* commands"""
    completer = AiAssistCompleter(agent=_agent())
    doc = Document("/skill ", cursor_position=7)
    commands = [c.text for c in completer.get_completions(doc, None)]
    assert "/skill/install" in commands
    assert "/skill/uninstall" in commands
    assert "/skill/update" in commands
    assert "/skill/list" in commands


def test_mcp_namespace_completion_lists_subcommands():
    """/mcp <space> suggests all /mcp/* commands"""
    completer = AiAssistCompleter(agent=_agent())
    doc = Document("/mcp ", cursor_position=5)
    commands = [c.text for c in completer.get_completions(doc, None)]
    assert any(cmd.startswith("/mcp/") for cmd in commands)


def test_namespace_completion_narrows_on_partial():
    """/plugin ins narrows to /plugin/install"""
    completer = AiAssistCompleter(agent=_agent())
    doc = Document("/plugin ins", cursor_position=11)
    commands = [c.text for c in completer.get_completions(doc, None)]
    assert commands == ["/plugin/install"]


def test_namespaced_plugin_skill_completion():
    """A namespaced plugin skill in loaded_skills is completed as /<plugin>:<skill>"""
    agent = _agent()
    agent.skills_manager.loaded_skills = {"acme:review": _make_skill("review")}
    completer = AiAssistCompleter(agent=agent)
    doc = Document("/acme", cursor_position=5)
    commands = [c.text for c in completer.get_completions(doc, None)]
    assert "/acme:review" in commands
