"""Tests for the Claude Code plugins manager"""

import json

import pytest

from ai_assist.plugins_loader import PluginsLoader
from ai_assist.plugins_manager import PluginsManager
from ai_assist.skills_loader import SkillsLoader
from ai_assist.skills_manager import SkillsManager


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def plugin_dir(tmp_path):
    root = tmp_path / "acme-plugin"
    _write(root / ".claude-plugin" / "plugin.json", json.dumps({"name": "acme", "description": "Acme tools"}))
    _write(
        root / "skills" / "review" / "SKILL.md",
        "---\nname: review\ndescription: Review code\n---\n\nReview $ARGUMENTS.\n",
    )
    _write(
        root / ".mcp.json",
        json.dumps({"mcpServers": {"srv": {"command": "echo", "args": ["hi"]}}}),
    )
    return root


@pytest.fixture
def manager(tmp_path):
    skills_loader = SkillsLoader()
    skills_manager = SkillsManager(skills_loader)
    mgr = PluginsManager(PluginsLoader(skills_loader), skills_manager)
    mgr.installed_plugins_file = tmp_path / "installed-plugins.json"
    mgr.marketplaces_file = tmp_path / "plugin-marketplaces.json"
    return mgr


def test_install_local_plugin(manager, plugin_dir):
    message, servers = manager.install_plugin(str(plugin_dir))

    assert "installed" in message
    assert servers == ["acme__srv"]
    # Skill registered under a namespaced key in the shared loaded_skills dict
    assert "acme:review" in manager.skills_manager.loaded_skills
    assert "acme__srv" in manager.plugin_mcp_servers
    assert manager.installed_plugins_file.exists()


def test_install_duplicate_rejected(manager, plugin_dir):
    manager.install_plugin(str(plugin_dir))
    message, _ = manager.install_plugin(str(plugin_dir))
    assert "already installed" in message


def test_uninstall_removes_skills_and_servers(manager, plugin_dir):
    manager.install_plugin(str(plugin_dir))
    message, servers = manager.uninstall_plugin("acme")

    assert "uninstalled" in message
    assert servers == ["acme__srv"]
    assert "acme:review" not in manager.skills_manager.loaded_skills
    assert "acme__srv" not in manager.plugin_mcp_servers


def test_persistence_round_trip(manager, plugin_dir):
    manager.install_plugin(str(plugin_dir))

    # New manager over the same files reloads the plugin and re-registers it
    skills_loader = SkillsLoader()
    skills_manager = SkillsManager(skills_loader)
    reloaded = PluginsManager(PluginsLoader(skills_loader), skills_manager)
    reloaded.installed_plugins_file = manager.installed_plugins_file
    reloaded.marketplaces_file = manager.marketplaces_file
    reloaded.load_installed_plugins()

    assert [p.name for p in reloaded.installed_plugins] == ["acme"]
    assert "acme:review" in skills_manager.loaded_skills
    assert "acme__srv" in reloaded.plugin_mcp_servers


def test_reapply_survives_skills_reload(manager, plugin_dir):
    manager.install_plugin(str(plugin_dir))
    # Simulate load_installed_skills() wiping loaded_skills
    manager.skills_manager.loaded_skills = {}
    manager.reapply_to_loaded_skills()
    assert "acme:review" in manager.skills_manager.loaded_skills


def test_uninstall_missing_plugin(manager):
    message, servers = manager.uninstall_plugin("nope")
    assert "not installed" in message
    assert servers == []


# --- marketplace ---


@pytest.fixture
def marketplace_dir(tmp_path, plugin_dir):
    root = tmp_path / "market"
    _write(
        root / ".claude-plugin" / "marketplace.json",
        json.dumps(
            {
                "name": "acme-market",
                "plugins": [
                    {"name": "acme", "source": str(plugin_dir), "description": "Acme tools bundle"},
                ],
            }
        ),
    )
    return root


def test_add_marketplace_and_search(manager, marketplace_dir):
    result = manager.add_marketplace(str(marketplace_dir))
    assert "added" in result

    listing = manager.list_marketplaces()
    assert "acme-market" in listing

    found = manager.search("acme")
    assert "acme" in found
    assert "Acme tools bundle" in found


def test_install_by_name_via_marketplace(manager, marketplace_dir):
    manager.add_marketplace(str(marketplace_dir))
    message, servers = manager.install_plugin("acme")
    assert "installed" in message
    assert "acme:review" in manager.skills_manager.loaded_skills
