"""Manage Claude Code plugin installation, availability, and marketplaces.

Plugins bundle Agent Skills and MCP servers. Their skills are registered into
the shared ``SkillsManager.loaded_skills`` dict under namespaced keys
(``plugin:skill``) so every existing skill consumer (system prompt, slash
invocation, completion, ``introspection__get_skill_help``) works unchanged.
Bundled MCP servers are exposed via ``plugin_mcp_servers`` for the agent to
connect.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from .config import MCPServerConfig, get_config_dir
from .plugins_loader import LoadedPlugin, PluginsLoader
from .skills_loader import SkillContent
from .skills_manager import SkillsManager

logger = logging.getLogger(__name__)


class InstalledPlugin(BaseModel):
    """Record of an installed plugin."""

    name: str
    source: str  # Original source spec
    source_type: str  # 'git' or 'local'
    skill_keys: list[str] = []  # Namespaced keys registered in loaded_skills
    mcp_server_names: list[str] = []  # Namespaced MCP server names
    branch: str
    installed_at: str  # ISO timestamp
    cache_path: str  # Path to the plugin directory


class Marketplace(BaseModel):
    """A registered plugin marketplace (a repo/dir with marketplace.json)."""

    name: str
    source: str
    branch: str
    cache_path: str


class PluginsManager:
    """Install and manage Claude Code plugins and marketplaces."""

    def __init__(self, plugins_loader: PluginsLoader, skills_manager: SkillsManager):
        self.plugins_loader = plugins_loader
        self.skills_manager = skills_manager
        self.installed_plugins_file = get_config_dir() / "installed-plugins.json"
        self.marketplaces_file = get_config_dir() / "plugin-marketplaces.json"

        self.installed_plugins: list[InstalledPlugin] = []
        self.loaded_plugin_skills: dict[str, SkillContent] = {}  # namespaced keys
        self.plugin_mcp_servers: dict[str, MCPServerConfig] = {}  # namespaced names
        self.marketplaces: list[Marketplace] = []

    # ------------------------------------------------------------------ load

    def load_installed_plugins(self):
        """Load installed plugins from disk and register their skills/MCP servers."""
        # Drop previously-registered plugin skills before rebuilding
        for key in self.loaded_plugin_skills:
            self.skills_manager.loaded_skills.pop(key, None)

        self.installed_plugins = []
        self.loaded_plugin_skills = {}
        self.plugin_mcp_servers = {}
        self._load_marketplaces()

        if not self.installed_plugins_file.exists():
            return

        try:
            data = json.loads(self.installed_plugins_file.read_text())
        except json.JSONDecodeError as e:
            logger.error("Failed to parse %s: %s", self.installed_plugins_file, e)
            return

        for record in data.get("plugins", []):
            try:
                plugin = InstalledPlugin(**record)
                cache_path = Path(plugin.cache_path)
                if not cache_path.exists():
                    logger.warning("Plugin cache not found for '%s' at %s", plugin.name, cache_path)
                    continue
                loaded = self.plugins_loader.load_plugin_from_local(cache_path)
                self._register(plugin.name, loaded)
                self.installed_plugins.append(plugin)
            except Exception as e:
                logger.warning("Failed to load installed plugin: %s", e)

        self.reapply_to_loaded_skills()

    def reapply_to_loaded_skills(self):
        """Re-merge plugin skills into loaded_skills (after a skills reload wipes it)."""
        self.skills_manager.loaded_skills.update(self.loaded_plugin_skills)

    # --------------------------------------------------------------- install

    def install_plugin(self, source_spec: str) -> tuple[str, list[str]]:
        """Install a plugin from a source spec or a marketplace plugin name.

        Returns (message, newly_added_mcp_server_names). The caller connects the
        returned MCP servers.
        """
        try:
            resolved = self._resolve_from_marketplaces(source_spec)
            if resolved is None and "/" not in source_spec and not Path(source_spec).expanduser().is_absolute():
                if self._find_marketplace_plugin(source_spec) is not None:
                    return (
                        f"Error: Plugin '{source_spec}' uses an unsupported source type "
                        "(only GitHub git sources and in-repo paths are supported).",
                        [],
                    )
            resolved = resolved or source_spec
            source, branch = self.skills_manager._parse_source_spec(resolved)

            source_path = Path(source).expanduser()
            if source_path.is_absolute() and source_path.exists():
                source_type = "local"
                loaded = self.plugins_loader.load_plugin_from_local(source_path)
                cache_path = str(source_path)
            else:
                source_type = "git"
                source, url_branch = self.skills_manager._normalize_github_url(source)
                if url_branch and branch == "main":
                    branch = url_branch
                parts = source.split("/")
                if len(parts) < 2:
                    return (f"Error: Invalid git source '{source}'. Expected owner/repo[/path]", [])
                repo_url = f"{parts[0]}/{parts[1]}"
                subpath = "/".join(parts[2:]) if len(parts) > 2 else ""
                loaded = self.plugins_loader.load_plugin_from_git(repo_url, subpath, branch)
                cache_path = str(loaded.plugin_path)

            plugin_name = loaded.manifest.name
            if any(p.name == plugin_name for p in self.installed_plugins):
                return (f"Error: Plugin '{plugin_name}' is already installed. Uninstall first to reinstall.", [])

            skill_keys = self._register(plugin_name, loaded)
            server_names = list(loaded.mcp_servers.keys())

            self.installed_plugins.append(
                InstalledPlugin(
                    name=plugin_name,
                    source=source,
                    source_type=source_type,
                    branch=branch,
                    installed_at=datetime.now().isoformat(),
                    cache_path=cache_path,
                    skill_keys=skill_keys,
                    mcp_server_names=server_names,
                )
            )
            self.reapply_to_loaded_skills()
            self._save_installed_plugins()

            summary = self._install_summary(loaded, len(skill_keys), server_names)
            return (summary, server_names)

        except (FileNotFoundError, ValueError) as e:
            return (f"Error: {e}", [])
        except Exception as e:
            logger.exception("Failed to install plugin")
            return (f"Error: Failed to install plugin: {e}", [])

    def uninstall_plugin(self, name: str) -> tuple[str, list[str]]:
        """Uninstall a plugin. Returns (message, removed_mcp_server_names)."""
        plugin = next((p for p in self.installed_plugins if p.name == name), None)
        if not plugin:
            return (f"Error: Plugin '{name}' is not installed", [])

        for key in plugin.skill_keys:
            self.loaded_plugin_skills.pop(key, None)
            self.skills_manager.loaded_skills.pop(key, None)
        for server_name in plugin.mcp_server_names:
            self.plugin_mcp_servers.pop(server_name, None)

        self.installed_plugins.remove(plugin)
        self._save_installed_plugins()
        return (f"Plugin '{name}' uninstalled successfully", plugin.mcp_server_names)

    def update_plugin(self, name: str) -> tuple[str, list[str], list[str]]:
        """Reinstall a plugin from its recorded source to pick up upstream changes.

        Returns (message, servers_to_disconnect, servers_to_connect). The caller
        disconnects the first list and connects the second (plugin MCP servers
        launch processes, so connection stays in the interactive layer). On
        failure the previous plugin is restored and both lists are empty.
        """
        plugin = next((p for p in self.installed_plugins if p.name == name), None)
        if plugin is None:
            return (f"Error: Plugin '{name}' is not installed", [], [])

        source_spec = f"{plugin.source}@{plugin.branch}"
        old_servers = list(plugin.mcp_server_names)
        self.uninstall_plugin(name)
        message, new_servers = self.install_plugin(source_spec)
        if message.startswith("Error"):
            # Restore the previous plugin so a failed update is not destructive.
            try:
                loaded = self.plugins_loader.load_plugin_from_local(Path(plugin.cache_path))
                self._register(plugin.name, loaded)
                self.installed_plugins.append(plugin)
                self.reapply_to_loaded_skills()
                self._save_installed_plugins()
            except Exception:
                logger.exception("Failed to roll back plugin update for '%s'", name)
            return (f"Error: Update of '{name}' failed, kept previous version. {message}", [], [])
        return (f"Plugin '{name}' updated ({message})", old_servers, new_servers)

    def list_installed(self) -> str:
        """Return a formatted list of installed plugins."""
        if not self.installed_plugins:
            return "No plugins installed.\n\nInstall with: /plugin/install <owner/repo|/path>@<branch>"

        lines = ["Installed plugins:\n"]
        for plugin in self.installed_plugins:
            lines.append(f"  {plugin.name}")
            lines.append(f"    Source: {plugin.source}@{plugin.branch}")
            skills = [k.split(":", 1)[1] for k in plugin.skill_keys]
            if skills:
                lines.append(f"    Skills: {', '.join(skills)}")
            if plugin.mcp_server_names:
                lines.append(f"    MCP servers: {', '.join(plugin.mcp_server_names)}")
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------ marketplace

    def add_marketplace(self, source_spec: str, nickname: str | None = None) -> str:
        """Register a marketplace from a git repo or local directory.

        An optional ``nickname`` overrides the manifest name, letting the user
        register several marketplaces that share a name and refer to each by a
        short handle in ``update``/``search``/``list``.
        """
        try:
            source, branch = self.skills_manager._parse_source_spec(source_spec)
            source_path = Path(source).expanduser()
            if source_path.is_absolute() and source_path.exists():
                cache_path = source_path
            else:
                source, _ = self.skills_manager._normalize_github_url(source)
                cache_path = self.plugins_loader.skills_loader._ensure_repo_cached(source, branch)

            manifest = self._read_marketplace_manifest(cache_path)
            name = nickname or manifest.get("name") or source

            existing = next((m for m in self.marketplaces if m.name == name), None)
            if existing is not None and (existing.source != source or existing.branch != branch):
                hint = (
                    "choose a different nickname"
                    if nickname
                    else f"register it under a nickname: /plugin/marketplace add {source_spec} <nickname>"
                )
                return (
                    f"Error: '{name}' already refers to {existing.source}@{existing.branch}. " f"To keep both, {hint}."
                )

            self.marketplaces = [m for m in self.marketplaces if m.name != name]
            self.marketplaces.append(Marketplace(name=name, source=source, branch=branch, cache_path=str(cache_path)))
            self._save_marketplaces()
            count = len(manifest.get("plugins", []))
            return f"Marketplace '{name}' added ({count} plugins available)"
        except (FileNotFoundError, ValueError) as e:
            return f"Error: {e}"

    def update_marketplace(self, name: str) -> str:
        """Refresh a registered marketplace's cached repo (git pull)."""
        market = next((m for m in self.marketplaces if m.name == name), None)
        if market is None:
            return f"Error: Marketplace '{name}' is not registered"
        try:
            source_path = Path(market.source).expanduser()
            if source_path.is_absolute() and source_path.exists():
                cache_path = source_path  # local directory, always current
            else:
                cache_path = self.plugins_loader.skills_loader._ensure_repo_cached(market.source, market.branch)
            manifest = self._read_marketplace_manifest(cache_path)
        except (FileNotFoundError, ValueError) as e:
            return f"Error: {e}"
        count = len(manifest.get("plugins", []))
        return f"Marketplace '{name}' updated ({count} plugins available)"

    def list_marketplaces(self) -> str:
        """Return a formatted list of registered marketplaces."""
        if not self.marketplaces:
            return "No marketplaces registered.\n\nAdd one with: /plugin/marketplace add <owner/repo>"
        lines = ["Registered marketplaces:\n"]
        for m in self.marketplaces:
            lines.append(f"  {m.name} ({m.source}@{m.branch})")
        return "\n".join(lines)

    def search(self, query: str) -> str:
        """Search registered marketplaces for plugins matching a query."""
        if not self.marketplaces:
            return "No marketplaces registered. Add one with: /plugin/marketplace add <owner/repo>"

        query_lower = query.lower()
        lines: list[str] = []
        for market in self.marketplaces:
            try:
                manifest = self._read_marketplace_manifest(Path(market.cache_path))
            except FileNotFoundError, ValueError:
                continue
            for plugin in manifest.get("plugins", []):
                name = plugin.get("name", "")
                description = plugin.get("description", "")
                if query_lower in name.lower() or query_lower in description.lower():
                    lines.append(f"  {name}  [{market.name}]")
                    if description:
                        lines.append(f"    {description}")
                    lines.append(f"    Install: /plugin/install {name}")
                    lines.append("")

        if not lines:
            return f"No plugins found matching '{query}'"
        return "\n".join([f"Plugin search results for '{query}':\n", *lines])

    def _resolve_from_marketplaces(self, name: str) -> str | None:
        """Resolve a bare plugin name to a concrete install spec via marketplaces.

        A bare name may carry a ``@marketplace`` disambiguator to pick between
        marketplaces that expose the same plugin name. Raises ValueError when a
        name is ambiguous or the requested marketplace does not have it.
        """
        if "/" in name or Path(name).expanduser().is_absolute():
            return None  # Already a direct source spec
        plugin_name, _, market_hint = name.partition("@")
        matches = self._find_marketplace_plugins(plugin_name)
        if market_hint:
            matches = [m for m in matches if m[0].name == market_hint]
            if not matches:
                raise ValueError(f"No plugin '{plugin_name}' found in marketplace '{market_hint}'")
        if not matches:
            return None
        if len(matches) > 1:
            options = "\n".join(f"  {plugin_name}@{market.name}" for market, _, _, _ in matches)
            raise ValueError(
                f"'{plugin_name}' is offered by {len(matches)} marketplaces:\n{options}\n"
                f"Pick one, e.g. /plugin/install {plugin_name}@{matches[0][0].name}"
            )
        _market, plugin, cache_path, plugin_root = matches[0]
        return self._marketplace_source_to_spec(plugin.get("source"), cache_path, plugin_root)

    def _find_marketplace_plugin(self, name: str) -> tuple[dict, Path, str] | None:
        """Return the first (plugin_entry, cache_path, plugin_root) matching name, or None."""
        plugin_name, _, _ = name.partition("@")
        matches = self._find_marketplace_plugins(plugin_name)
        if not matches:
            return None
        _market, plugin, cache_path, plugin_root = matches[0]
        return plugin, cache_path, plugin_root

    def _find_marketplace_plugins(self, name: str) -> list[tuple[Marketplace, dict, Path, str]]:
        """Find every plugin entry named ``name`` across registered marketplaces.

        Returns a list of (marketplace, plugin_entry, cache_path, plugin_root).
        """
        results = []
        for market in self.marketplaces:
            try:
                manifest = self._read_marketplace_manifest(Path(market.cache_path))
            except FileNotFoundError, ValueError:
                continue
            plugin_root = str(manifest.get("pluginRoot", ""))
            for plugin in manifest.get("plugins", []):
                if plugin.get("name") == name:
                    results.append((market, plugin, Path(market.cache_path), plugin_root))
        return results

    def _marketplace_source_to_spec(self, source, cache_path: Path, plugin_root: str) -> str | None:
        """Convert a Claude Code marketplace plugin ``source`` to an install spec.

        Supports the object forms used by real marketplaces
        (``github`` / ``git`` / ``git-subdir``) and string sources (a path
        relative to the marketplace repo, honoring ``pluginRoot``). Returns None
        for unsupported sources (e.g. non-GitHub git URLs).
        """
        if isinstance(source, dict):
            kind = source.get("source")
            ref = source.get("ref") or source.get("branch")
            path = str(source.get("path", "")).strip("/")
            if kind == "github":
                repo = str(source.get("repo", "")).strip("/")
                if not repo:
                    return None
            elif kind in ("git", "git-subdir"):
                repo, _ = self.skills_manager._normalize_github_url(str(source.get("url", "")))
                if "://" in repo or repo.count("/") != 1:
                    return None  # non-GitHub git URLs are not supported
            else:
                return None
            spec = f"{repo}/{path}" if path else repo
            return f"{spec}@{ref}" if ref else spec

        if isinstance(source, str) and source:
            expanded = Path(source).expanduser()
            if expanded.is_absolute():
                return str(expanded)
            # A string source is a path relative to the marketplace repo.
            rel = source.lstrip("./")
            root = plugin_root.strip("/")
            rel = f"{root}/{rel}" if root else rel
            return str((cache_path / rel).resolve())

        return None

    @staticmethod
    def _read_marketplace_manifest(cache_path: Path) -> dict:
        """Read .claude-plugin/marketplace.json from a marketplace directory."""
        manifest_file = cache_path / ".claude-plugin" / "marketplace.json"
        if not manifest_file.exists():
            raise FileNotFoundError(f"marketplace.json not found in {cache_path}")
        try:
            return json.loads(manifest_file.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid marketplace.json: {e}") from e

    # ---------------------------------------------------------------- helpers

    def _register(self, plugin_name: str, loaded: LoadedPlugin) -> list[str]:
        """Register a loaded plugin's skills and MCP servers in memory."""
        skill_keys: list[str] = []
        for short_name, content in loaded.skills.items():
            key = f"{plugin_name}:{short_name}"
            self.loaded_plugin_skills[key] = content
            skill_keys.append(key)
        self.plugin_mcp_servers.update(loaded.mcp_servers)
        return skill_keys

    @staticmethod
    def _install_summary(loaded: LoadedPlugin, skill_count: int, server_names: list[str]) -> str:
        """Build a human-readable install summary, noting skipped components."""
        parts = [f"Plugin '{loaded.manifest.name}' installed: {skill_count} skill(s)"]
        if server_names:
            parts.append(f"{len(server_names)} MCP server(s)")
        summary = ", ".join(parts)
        skipped = [f"{n} {kind}" for kind, n in loaded.skipped.items() if n]
        if skipped:
            summary += f" (skipped {', '.join(skipped)} — not supported)"
        return summary

    def _save_installed_plugins(self):
        """Persist installed plugins to JSON."""
        data = {"plugins": [p.model_dump() for p in self.installed_plugins]}
        self.installed_plugins_file.write_text(json.dumps(data, indent=2))

    def _load_marketplaces(self):
        """Load registered marketplaces from JSON."""
        self.marketplaces = []
        if not self.marketplaces_file.exists():
            return
        try:
            data = json.loads(self.marketplaces_file.read_text())
        except json.JSONDecodeError as e:
            logger.error("Failed to parse %s: %s", self.marketplaces_file, e)
            return
        for record in data.get("marketplaces", []):
            try:
                self.marketplaces.append(Marketplace(**record))
            except Exception as e:
                logger.warning("Failed to load marketplace: %s", e)

    def _save_marketplaces(self):
        """Persist registered marketplaces to JSON."""
        data = {"marketplaces": [m.model_dump() for m in self.marketplaces]}
        self.marketplaces_file.write_text(json.dumps(data, indent=2))
