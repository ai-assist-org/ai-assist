"""Config file watching for auto-reload

This module provides shared configuration file watching used by both
monitor mode and interactive mode.
"""

from pathlib import Path

from .config import get_config_dir
from .file_watchdog import FileWatchdog


class ConfigWatcher:
    """Watch config files and trigger callbacks on changes

    This class watches multiple configuration files and triggers
    appropriate reload callbacks when they change.

    Attributes:
        agent: The AiAssistAgent instance
        watchers: List of active FileWatchdog instances
    """

    def __init__(self, agent):
        """Initialize config watcher

        Args:
            agent: The AiAssistAgent instance to reload configs for
        """
        self.agent = agent
        self.watchers: list[FileWatchdog] = []
        self._skill_watchers: list[FileWatchdog] = []

    async def start(self):
        """Start watching all config files"""
        config_dir = get_config_dir()

        # Watch mcp_servers.yaml
        mcp_file = config_dir / "mcp_servers.yaml"
        if mcp_file.exists():
            watcher = FileWatchdog(mcp_file, self._on_mcp_change, debounce_seconds=1.0)
            await watcher.start()
            self.watchers.append(watcher)
            print(f"Watching {mcp_file} for changes")

        # Watch identity.yaml
        identity_file = config_dir / "identity.yaml"
        if identity_file.exists():
            watcher = FileWatchdog(identity_file, self._on_identity_change, debounce_seconds=1.0)
            await watcher.start()
            self.watchers.append(watcher)
            print(f"Watching {identity_file} for changes")

        # Watch installed-skills.json
        skills_file = config_dir / "installed-skills.json"
        if skills_file.exists():
            watcher = FileWatchdog(skills_file, self._on_skills_change, debounce_seconds=1.0)
            await watcher.start()
            self.watchers.append(watcher)
            print(f"Watching {skills_file} for changes")

        # Watch installed-plugins.json
        plugins_file = config_dir / "installed-plugins.json"
        if plugins_file.exists():
            watcher = FileWatchdog(plugins_file, self._on_plugins_change, debounce_seconds=1.0)
            await watcher.start()
            self.watchers.append(watcher)
            print(f"Watching {plugins_file} for changes")

        # Watch individual SKILL.md files
        await self._watch_skill_files()

    async def _watch_skill_files(self):
        """Watch SKILL.md files in installed skill directories."""
        for watcher in self._skill_watchers:
            await watcher.stop()
        self._skill_watchers = []

        if not hasattr(self.agent, "skills_manager"):
            return

        for skill in self.agent.skills_manager.installed_skills:
            if skill.source_type != "local":
                continue
            skill_md = Path(skill.cache_path) / "SKILL.md"
            if skill_md.exists():
                callback = self._make_skill_file_callback(skill.name)
                watcher = FileWatchdog(skill_md, callback, debounce_seconds=1.0)
                await watcher.start()
                self._skill_watchers.append(watcher)

    async def _on_mcp_change(self):
        """Callback when mcp_servers.yaml changes"""
        try:
            await self.agent.reload_mcp_servers()
        except Exception as e:
            print(f"❌ Failed to reload MCP servers: {e}")

    async def _on_identity_change(self):
        """Callback when identity.yaml changes"""
        try:
            from .identity import get_identity

            self.agent.identity = get_identity(reload=True)
            print("✅ Identity reloaded")
        except Exception as e:
            print(f"❌ Failed to reload identity: {e}")

    async def _on_skills_change(self):
        """Callback when installed-skills.json changes"""
        try:
            self.agent.skills_manager.load_installed_skills()
            # load_installed_skills rebuilds loaded_skills from scratch, dropping
            # plugin skills — re-apply them.
            if hasattr(self.agent, "plugins_manager"):
                self.agent.plugins_manager.reapply_to_loaded_skills()
            await self._watch_skill_files()
            print("✅ Skills reloaded")
        except Exception as e:
            print(f"❌ Failed to reload skills: {e}")

    async def _on_plugins_change(self):
        """Callback when installed-plugins.json changes"""
        try:
            self.agent.plugins_manager.load_installed_plugins()
            print("✅ Plugins reloaded")
        except Exception as e:
            print(f"❌ Failed to reload plugins: {e}")

    def _make_skill_file_callback(self, skill_name: str):
        """Create a callback for a specific skill's SKILL.md."""

        async def on_change():
            try:
                self.agent.skills_manager.load_installed_skills()
                print(f"✅ Skill '{skill_name}' reloaded")
            except Exception as e:
                print(f"❌ Failed to reload skill '{skill_name}': {e}")

        return on_change

    async def stop(self):
        """Stop all watchers"""
        for watcher in self._skill_watchers:
            await watcher.stop()
        for watcher in self.watchers:
            await watcher.stop()
