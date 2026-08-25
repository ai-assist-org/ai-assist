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


def test_add_marketplace_with_nickname(manager, marketplace_dir):
    result = manager.add_marketplace(str(marketplace_dir), "redhat")
    assert "'redhat' added" in result
    # Registered under the nickname, not the manifest name
    assert [m.name for m in manager.marketplaces] == ["redhat"]
    assert "redhat" in manager.list_marketplaces()
    # The nickname is usable for update
    assert "updated" in manager.update_marketplace("redhat")


def test_nickname_avoids_name_collision(manager, marketplace_dir):
    # Same manifest name added twice under distinct nicknames coexist
    manager.add_marketplace(str(marketplace_dir), "one")
    manager.add_marketplace(str(marketplace_dir), "two")
    assert sorted(m.name for m in manager.marketplaces) == ["one", "two"]


@pytest.fixture
def other_marketplace_dir(tmp_path, plugin_dir):
    # A second, different source that declares the SAME manifest name.
    root = tmp_path / "market2"
    _write(
        root / ".claude-plugin" / "marketplace.json",
        json.dumps({"name": "acme-market", "plugins": [{"name": "acme", "source": str(plugin_dir)}]}),
    )
    return root


def test_add_refuses_masking_marketplace(manager, marketplace_dir, other_marketplace_dir):
    manager.add_marketplace(str(marketplace_dir))
    result = manager.add_marketplace(str(other_marketplace_dir))
    assert result.startswith("Error")
    assert "already refers to" in result
    assert "nickname" in result
    # The original registration is untouched
    assert [m.source for m in manager.marketplaces] == [str(marketplace_dir)]


def test_add_refuses_masking_nickname(manager, marketplace_dir, other_marketplace_dir):
    manager.add_marketplace(str(marketplace_dir), "redhat")
    result = manager.add_marketplace(str(other_marketplace_dir), "redhat")
    assert result.startswith("Error")
    assert "choose a different nickname" in result


def test_readd_same_source_is_allowed(manager, marketplace_dir):
    manager.add_marketplace(str(marketplace_dir))
    result = manager.add_marketplace(str(marketplace_dir))
    assert "added" in result
    # Still exactly one registration, not a duplicate
    assert len(manager.marketplaces) == 1


def test_update_marketplace(manager, marketplace_dir):
    manager.add_marketplace(str(marketplace_dir))
    result = manager.update_marketplace("acme-market")
    assert "updated" in result
    assert "1 plugins" in result


def test_update_unknown_marketplace(manager):
    result = manager.update_marketplace("nope")
    assert result.startswith("Error")
    assert "not registered" in result


def test_install_by_name_via_marketplace(manager, marketplace_dir):
    manager.add_marketplace(str(marketplace_dir))
    message, servers = manager.install_plugin("acme")
    assert "installed" in message
    assert "acme:review" in manager.skills_manager.loaded_skills


# --- real-world marketplace source schemas ---


def _market_with_plugin(tmp_path, entry, plugin_root=None):
    """Build a marketplace dir whose manifest declares a single plugin entry."""
    root = tmp_path / "real-market"
    manifest = {"name": "real-market", "plugins": [entry]}
    if plugin_root is not None:
        manifest["pluginRoot"] = plugin_root
    _write(root / ".claude-plugin" / "marketplace.json", json.dumps(manifest))
    return root


def test_resolve_github_object_source(manager, tmp_path):
    market = _market_with_plugin(
        tmp_path,
        {"name": "cc-suite", "source": {"source": "github", "repo": "xiaolai/cc-suite"}},
    )
    manager.add_marketplace(str(market))
    assert manager._resolve_from_marketplaces("cc-suite") == "xiaolai/cc-suite"


def test_resolve_github_object_source_with_path_and_ref(manager, tmp_path):
    market = _market_with_plugin(
        tmp_path,
        {
            "name": "airtable",
            "source": {"source": "github", "repo": "Airtable/skills", "path": "plugins/airtable", "ref": "v2"},
        },
    )
    manager.add_marketplace(str(market))
    assert manager._resolve_from_marketplaces("airtable") == "Airtable/skills/plugins/airtable@v2"


def test_resolve_git_subdir_object_source(manager, tmp_path):
    market = _market_with_plugin(
        tmp_path,
        {
            "name": "airtable",
            "source": {
                "source": "git-subdir",
                "url": "https://github.com/Airtable/skills.git",
                "path": "plugins/airtable",
            },
        },
    )
    manager.add_marketplace(str(market))
    assert manager._resolve_from_marketplaces("airtable") == "Airtable/skills/plugins/airtable"


def test_resolve_non_github_git_source_unsupported(manager, tmp_path):
    market = _market_with_plugin(
        tmp_path,
        {"name": "gl", "source": {"source": "git", "url": "https://gitlab.com/foo/bar.git"}},
    )
    manager.add_marketplace(str(market))
    assert manager._resolve_from_marketplaces("gl") is None

    message, servers = manager.install_plugin("gl")
    assert "unsupported source type" in message
    assert servers == []


def test_resolve_relative_string_source_honors_plugin_root(manager, tmp_path, plugin_dir):
    # Place the plugin under <market>/bundles/acme and point pluginRoot at "bundles".
    market = tmp_path / "rel-market"
    (market / "bundles").mkdir(parents=True)
    plugin_dir.rename(market / "bundles" / "acme")
    _write(
        market / ".claude-plugin" / "marketplace.json",
        json.dumps(
            {
                "name": "rel-market",
                "pluginRoot": "bundles",
                "plugins": [{"name": "acme", "source": "acme"}],
            }
        ),
    )

    manager.add_marketplace(str(market))
    resolved = manager._resolve_from_marketplaces("acme")
    assert resolved == str((market / "bundles" / "acme").resolve())

    message, _ = manager.install_plugin("acme")
    assert "installed" in message
    assert "acme:review" in manager.skills_manager.loaded_skills
