"""Load Claude Code plugins (skills + bundled MCP servers).

A Claude Code plugin is a directory (or git repo) containing a
``.claude-plugin/plugin.json`` manifest plus, optionally:

- ``skills/<name>/SKILL.md`` — one or more Agent Skills
- ``commands/<name>.md`` — legacy slash-command definitions (loaded as skills)
- ``.mcp.json`` — bundled MCP server definitions

``agents/`` (subagents) and ``hooks/`` are intentionally NOT loaded — ai-assist
has no equivalent subsystem. Their presence is counted and reported so the user
knows they were skipped.
"""

import json
import logging
import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .config import MCPServerConfig
from .security import validate_tool_description
from .skills_loader import SkillContent, SkillsLoader

logger = logging.getLogger(__name__)


class PluginManifest(BaseModel):
    """Contents of a plugin's .claude-plugin/plugin.json manifest."""

    name: str
    version: str | None = None
    description: str | None = None
    # author may be a string or an object ({name, email, url}) in Claude Code
    author: object | None = None


class LoadedPlugin(BaseModel):
    """A fully parsed plugin ready to be registered."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    manifest: PluginManifest
    plugin_path: Path
    skills: dict[str, SkillContent] = Field(default_factory=dict)  # keyed by skill short name
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)  # keyed by server short name
    skipped: dict[str, int] = Field(default_factory=dict)  # {"agents": N, "hooks": N}


class PluginsLoader:
    """Load Claude Code plugins from a local path or a git repository."""

    def __init__(self, skills_loader: SkillsLoader):
        """Reuse the skills loader for SKILL.md parsing and git caching."""
        self.skills_loader = skills_loader

    def load_plugin_from_local(self, plugin_path: Path) -> LoadedPlugin:
        """Load a plugin from a local directory."""
        return self._load_plugin_dir(plugin_path, source_url=None)

    def load_plugin_from_git(self, repo_url: str, subpath: str, branch: str = "main") -> LoadedPlugin:
        """Load a plugin from a git repository (optionally a subdirectory)."""
        repo_dir = self.skills_loader._ensure_repo_cached(repo_url, branch)
        plugin_path = repo_dir / subpath if subpath else repo_dir
        if not plugin_path.exists():
            raise FileNotFoundError(f"Plugin path '{subpath}' not found in repository")
        return self._load_plugin_dir(plugin_path, source_url=repo_url)

    def _load_plugin_dir(self, plugin_path: Path, source_url: str | None) -> LoadedPlugin:
        """Parse a plugin directory into a LoadedPlugin."""
        manifest_file = plugin_path / ".claude-plugin" / "plugin.json"
        if not manifest_file.exists():
            raise FileNotFoundError(f".claude-plugin/plugin.json not found in {plugin_path}")

        try:
            manifest_data = json.loads(manifest_file.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid plugin.json: {e}") from e

        if "name" not in manifest_data:
            raise ValueError("plugin.json must define a 'name'")

        manifest = PluginManifest(
            name=manifest_data["name"],
            version=manifest_data.get("version"),
            description=manifest_data.get("description"),
            author=manifest_data.get("author"),
        )

        skills = self._scan_skills(plugin_path, source_url)
        skills.update(self._scan_commands(plugin_path, source_url))
        mcp_servers = self._parse_mcp_json(plugin_path, manifest.name)
        skipped = {
            "agents": self._count_entries(plugin_path / "agents"),
            "hooks": self._count_hooks(plugin_path),
        }

        return LoadedPlugin(
            manifest=manifest,
            plugin_path=plugin_path,
            skills=skills,
            mcp_servers=mcp_servers,
            skipped=skipped,
        )

    def _scan_skills(self, plugin_path: Path, source_url: str | None) -> dict[str, SkillContent]:
        """Scan skills/<name>/SKILL.md, returning {short_name: SkillContent}."""
        skills_dir = plugin_path / "skills"
        if not skills_dir.is_dir():
            return {}

        found: dict[str, SkillContent] = {}
        for skill_dir in sorted(skills_dir.iterdir()):
            skill_file = skill_dir / "SKILL.md"
            if not skill_dir.is_dir() or not skill_file.exists():
                continue
            try:
                # source_type "plugin" skips the dir-name==name check enforced for "local"
                metadata, body = self.skills_loader._parse_skill_file(skill_file, skill_dir, "plugin", source_url)
                found[metadata.name] = self.skills_loader._build_skill_content(skill_dir, metadata, body)
            except (ValueError, FileNotFoundError) as e:
                logger.warning("Skipping invalid skill in %s: %s", skill_dir, e)
        return found

    def _scan_commands(self, plugin_path: Path, source_url: str | None) -> dict[str, SkillContent]:
        """Scan commands/*.md (legacy slash commands) as skills."""
        commands_dir = plugin_path / "commands"
        if not commands_dir.is_dir():
            return {}

        found: dict[str, SkillContent] = {}
        for command_file in sorted(commands_dir.glob("*.md")):
            try:
                content = self._parse_command_file(command_file, source_url)
                found[content.metadata.name] = content
            except ValueError as e:
                logger.warning("Skipping invalid command %s: %s", command_file, e)
        return found

    def _parse_command_file(self, command_file: Path, source_url: str | None) -> SkillContent:
        """Parse a commands/*.md file into a SkillContent.

        Command files may have YAML frontmatter (description, argument-hint) or
        be plain markdown. The command name defaults to the file stem.
        """
        # Local import to avoid a module-level cycle with skills_loader internals
        from .skills_loader import SkillMetadata

        raw = command_file.read_text()
        frontmatter: dict = {}
        body = raw.strip()

        if raw.startswith("---\n"):
            parts = raw.split("---\n", 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1]) or {}
                except yaml.YAMLError as e:
                    raise ValueError(f"Invalid YAML frontmatter: {e}") from e
                body = parts[2].strip()

        name = frontmatter.get("name") or command_file.stem
        description = frontmatter.get("description") or f"{name} command"

        metadata = SkillMetadata(
            name=name,
            description=description,
            argument_hint=frontmatter.get("argument-hint"),
            allowed_tools=(frontmatter.get("allowed-tools", "").split() if frontmatter.get("allowed-tools") else []),
            skill_path=command_file.parent,
            source_type="plugin",
            source_url=source_url,
        )
        metadata.validate()

        for warning in validate_tool_description(f"command:{name}/body", body):
            logger.warning("Command content warning for %s: %s", name, warning)

        return SkillContent(metadata=metadata, body=body)

    def _parse_mcp_json(self, plugin_path: Path, plugin_name: str) -> dict[str, MCPServerConfig]:
        """Parse a bundled .mcp.json into namespaced MCPServerConfig entries."""
        mcp_file = plugin_path / ".mcp.json"
        if not mcp_file.exists():
            return {}

        try:
            data = json.loads(mcp_file.read_text())
        except json.JSONDecodeError as e:
            logger.warning("Invalid .mcp.json in %s: %s", plugin_path, e)
            return {}

        type_to_transport = {"http": "streamablehttp", "sse": "sse", "stdio": None}
        servers: dict[str, MCPServerConfig] = {}
        for name, cfg in data.get("mcpServers", {}).items():
            env = {key: os.path.expandvars(str(value)) for key, value in cfg.get("env", {}).items()}
            servers[f"{plugin_name}__{name}"] = MCPServerConfig(
                command=cfg.get("command", ""),
                args=cfg.get("args", []),
                env=env,
                url=cfg.get("url"),
                transport=type_to_transport.get(cfg.get("type"), cfg.get("type")),
            )
        return servers

    @staticmethod
    def _count_entries(directory: Path) -> int:
        """Count files in a directory (0 if it does not exist)."""
        if not directory.is_dir():
            return 0
        return sum(1 for entry in directory.iterdir() if entry.is_file())

    @staticmethod
    def _count_hooks(plugin_path: Path) -> int:
        """Return 1 if the plugin defines hooks, else 0."""
        hooks_json = plugin_path / "hooks" / "hooks.json"
        return 1 if hooks_json.exists() else 0
