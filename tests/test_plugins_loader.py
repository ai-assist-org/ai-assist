"""Tests for the Claude Code plugin loader"""

import json

import pytest

from ai_assist.plugins_loader import PluginsLoader
from ai_assist.skills_loader import SkillsLoader


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def plugin_dir(tmp_path):
    """Build a representative plugin directory."""
    root = tmp_path / "my-plugin"
    _write(
        root / ".claude-plugin" / "plugin.json",
        json.dumps({"name": "my-plugin", "version": "1.0.0", "description": "Test plugin"}),
    )
    _write(
        root / "skills" / "review" / "SKILL.md",
        "---\nname: review\ndescription: Review code carefully\n---\n\n# Review\n\nReview $ARGUMENTS.\n",
    )
    _write(
        root / "skills" / "deploy" / "SKILL.md",
        "---\nname: deploy\ndescription: Deploy the app\nargument-hint: '[env]'\n---\n\n# Deploy\n\nDeploy to $1.\n",
    )
    _write(
        root / "commands" / "greet.md",
        "---\ndescription: Greet the user\n---\n\nSay hello to $1.\n",
    )
    _write(root / "commands" / "plain.md", "Just do the thing.\n")
    _write(
        root / ".mcp.json",
        json.dumps(
            {
                "mcpServers": {
                    "local": {"command": "python", "args": ["-m", "server"], "env": {"KEY": "val"}},
                    "remote": {"type": "http", "url": "https://example.com/mcp"},
                }
            }
        ),
    )
    _write(root / "agents" / "helper.md", "# Helper agent\n")
    _write(root / "hooks" / "hooks.json", json.dumps({"hooks": {}}))
    return root


@pytest.fixture
def loader():
    return PluginsLoader(SkillsLoader())


def test_load_manifest(loader, plugin_dir):
    plugin = loader.load_plugin_from_local(plugin_dir)
    assert plugin.manifest.name == "my-plugin"
    assert plugin.manifest.version == "1.0.0"


def test_scan_multiple_skills(loader, plugin_dir):
    plugin = loader.load_plugin_from_local(plugin_dir)
    assert set(plugin.skills) >= {"review", "deploy"}
    assert plugin.skills["deploy"].metadata.argument_hint == "[env]"


def test_commands_loaded_as_skills(loader, plugin_dir):
    plugin = loader.load_plugin_from_local(plugin_dir)
    # Frontmatter command
    assert "greet" in plugin.skills
    assert plugin.skills["greet"].metadata.description == "Greet the user"
    # Plain markdown command (name from file stem)
    assert "plain" in plugin.skills
    assert "do the thing" in plugin.skills["plain"].body


def test_mcp_servers_parsed_and_namespaced(loader, plugin_dir):
    plugin = loader.load_plugin_from_local(plugin_dir)
    assert "my-plugin__local" in plugin.mcp_servers
    assert "my-plugin__remote" in plugin.mcp_servers
    assert plugin.mcp_servers["my-plugin__local"].command == "python"
    assert plugin.mcp_servers["my-plugin__local"].env == {"KEY": "val"}
    # type "http" maps to streamablehttp transport
    assert plugin.mcp_servers["my-plugin__remote"].transport == "streamablehttp"
    assert plugin.mcp_servers["my-plugin__remote"].url == "https://example.com/mcp"


def test_agents_and_hooks_counted_as_skipped(loader, plugin_dir):
    plugin = loader.load_plugin_from_local(plugin_dir)
    assert plugin.skipped["agents"] == 1
    assert plugin.skipped["hooks"] == 1


def test_missing_manifest_raises(loader, tmp_path):
    with pytest.raises(FileNotFoundError):
        loader.load_plugin_from_local(tmp_path / "empty")
