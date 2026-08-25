"""TUI components for ai-assist"""

import os

from prompt_toolkit.completion import Completer, Completion

# Public Claude Code plugin marketplaces that work out of the box.
WELL_KNOWN_MARKETPLACES = [
    ("anthropics/claude-plugins-official@main", "Anthropic-managed official plugin directory"),
    ("anthropics/claude-plugins-community@main", "Community-submitted plugins"),
    ("anthropics/life-sciences@main", "Claude for Life Sciences plugins"),
    ("openshift-eng/ai-helpers@main", "OpenShift/Red Hat developer plugins"),
]


def format_tool_display_name(tool_name: str) -> str:
    """Format a tool name for user-friendly display.

    Converts internal tool names like 'mcp__dci__search' to 'dci → search'.
    """
    return tool_name.replace("mcp__", "").replace("__", " → ").replace("_", " ")


def format_tool_args(input_dict: dict, max_len: int = 100) -> str:
    """Format tool arguments for display, truncating long values.

    Escapes Rich markup and collapses newlines so console.print
    renders the text literally instead of interpreting markdown.
    """
    from rich.markup import escape

    args_display = []
    for key, value in input_dict.items():
        value_str = str(value).replace("\n", " ").replace("\r", "")
        if len(value_str) > max_len:
            value_str = value_str[:max_len] + "..."
        args_display.append(f"{key}={escape(value_str)}")
    return ", ".join(args_display)


class AiAssistCompleter(Completer):
    """Command completer for ai-assist interactive mode"""

    def __init__(self, agent=None):
        from .commands import get_interactive_commands

        self.agent = agent
        self.commands = [c.name for c in get_interactive_commands()]

    @staticmethod
    def _is_path_prefix(word: str) -> bool:
        return word.startswith(("./", "../")) or (word.startswith("~") and len(word) >= 2)

    @staticmethod
    def _get_path_completions(word: str):
        expanded = os.path.expanduser(word)
        if expanded.endswith("/"):
            search_dir = expanded
            prefix = ""
        else:
            search_dir = os.path.dirname(expanded) or "."
            prefix = os.path.basename(expanded)
        try:
            entries = os.listdir(search_dir)
        except OSError:
            return
        for entry in sorted(entries):
            if not prefix and entry.startswith("."):
                continue
            if entry.lower().startswith(prefix.lower()):
                full_path = os.path.join(search_dir, entry)
                is_dir = os.path.isdir(full_path)
                suffix = "/" if is_dir else ""
                yield Completion(
                    entry[len(prefix) :] + suffix,
                    display=entry + suffix,
                )

    def _get_plugin_arg_completions(self, text):
        """Completions for /plugin/marketplace add|update, /plugin/install, /plugin/uninstall|update."""
        if text.startswith("/plugin/marketplace add "):
            yield from self._get_marketplace_add_completions(text)
            return

        if text.startswith("/plugin/marketplace update ") and self.agent:
            prefix = text.split("update ", 1)[1]
            for market in self.agent.plugins_manager.marketplaces:
                if market.name.startswith(prefix):
                    full_command = f"/plugin/marketplace update {market.name}"
                    yield Completion(
                        full_command,
                        start_position=-len(text),
                        display=full_command,
                        display_meta=f"{market.source}@{market.branch}",
                    )
            return

        prefix = text.split(" ", 1)[1] if " " in text else ""

        if text.startswith(("/plugin/uninstall ", "/plugin/update ")) and self.agent:
            base = "/plugin/uninstall" if text.startswith("/plugin/uninstall ") else "/plugin/update"
            for plugin in self.agent.plugins_manager.installed_plugins:
                if plugin.name.startswith(prefix.lower()):
                    full_command = f"{base} {plugin.name}"
                    yield Completion(
                        full_command,
                        start_position=-len(text),
                        display=full_command,
                        display_meta=f"{plugin.source}",
                    )
            return

        # /plugin/install: suggest plugin names from registered marketplaces
        if self.agent:
            for name, market_name, description in self._marketplace_plugin_names():
                if name.startswith(prefix.lower()):
                    full_command = f"/plugin/install {name}"
                    yield Completion(
                        full_command,
                        start_position=-len(text),
                        display=full_command,
                        display_meta=(description or f"from {market_name}")[:60],
                    )

        # /plugin/install: suggest example patterns
        examples = [
            ("owner/repo@main", "Install a plugin from a git repository"),
            ("/path/to/plugin@main", "Local plugin path example"),
        ]
        for example, description in examples:
            if example.startswith(prefix):
                full_command = f"/plugin/install {example}"
                yield Completion(
                    full_command,
                    start_position=-len(text),
                    display=full_command,
                    display_meta=description,
                )

    @staticmethod
    def _get_marketplace_add_completions(text):
        """Suggest well-known marketplaces after /plugin/marketplace add."""
        prefix = text.split("add ", 1)[1]
        for repo, description in WELL_KNOWN_MARKETPLACES:
            if repo.startswith(prefix):
                full_command = f"/plugin/marketplace add {repo}"
                yield Completion(
                    full_command,
                    start_position=-len(text),
                    display=full_command,
                    display_meta=description,
                )

    def _marketplace_plugin_names(self):
        """Yield (name, marketplace_name, description) for all marketplace plugins."""
        from pathlib import Path

        manager = self.agent.plugins_manager
        for market in manager.marketplaces:
            try:
                manifest = manager._read_marketplace_manifest(Path(market.cache_path))
            except FileNotFoundError, ValueError:
                continue
            for plugin in manifest.get("plugins", []):
                name = plugin.get("name")
                if name:
                    yield name, market.name, plugin.get("description", "")

    def get_completions(self, document, complete_event):
        """Get completions for the current input"""
        text = document.text_before_cursor

        words = text.split()
        if words:
            last_word = words[-1]
            if self._is_path_prefix(last_word):
                yield from self._get_path_completions(last_word)
                return

        # Only complete if line starts with /
        if text.startswith("/"):
            word = text  # Keep the full text including /

            # Special handling for skill commands with arguments (space-separated)
            if text.startswith(("/skill/uninstall ", "/skill/update ")) and self.agent:
                # Complete with installed skill names
                base = "/skill/uninstall" if text.startswith("/skill/uninstall ") else "/skill/update"
                prefix = text.split(" ", 1)[1] if " " in text else ""
                for skill in self.agent.skills_manager.installed_skills:
                    if skill.name.startswith(prefix.lower()):
                        full_command = f"{base} {skill.name}"
                        yield Completion(
                            full_command,
                            start_position=-len(text),
                            display=full_command,
                            display_meta=f"{skill.source}",
                        )
                return  # Don't continue to other completions

            if text.startswith("/mcp/restart ") and self.agent:
                # Complete with configured MCP server names
                prefix = text.split(" ", 1)[1] if " " in text else ""
                for server_name in self.agent.config.mcp_servers.keys():
                    if server_name.startswith(prefix):
                        full_command = f"/mcp/restart {server_name}"
                        yield Completion(
                            full_command,
                            start_position=-len(text),
                            display=full_command,
                            display_meta="MCP server",
                        )
                return  # Don't continue to other completions

            if text.startswith("/skill/install "):
                # Suggest example patterns
                prefix = text.split(" ", 1)[1] if " " in text else ""
                examples = [
                    ("clawhub:skill-slug", "Install from ClawHub registry"),
                    ("anthropics/skills/skills/pdf@main", "Official PDF skill from Anthropic"),
                    ("anthropics/skills/skills/docx@main", "Official DOCX skill from Anthropic"),
                    ("/path/to/skill@main", "Local skill path example"),
                ]
                for example, description in examples:
                    if example.startswith(prefix):
                        full_command = f"/skill/install {example}"
                        yield Completion(
                            full_command,
                            start_position=-len(text),
                            display=full_command,
                            display_meta=description,
                        )
                return  # Don't continue to other completions

            if text.startswith(
                (
                    "/plugin/marketplace add ",
                    "/plugin/marketplace update ",
                    "/plugin/install ",
                    "/plugin/uninstall ",
                    "/plugin/update ",
                )
            ):
                yield from self._get_plugin_arg_completions(text)
                return  # Don't continue to other completions

            # Check if this looks like a prompt command (has a slash in it)
            parts = word.lstrip("/").split("/")

            # Completing MCP prompts: /server/prompt
            if len(parts) == 2 and self.agent and parts[0] not in ("skill", "plugin"):
                server_name, prompt_prefix = parts

                # If we have prompts from this server
                if server_name in self.agent.available_prompts:
                    for prompt_name, prompt in self.agent.available_prompts[server_name].items():
                        if prompt_name.startswith(prompt_prefix.lower()):
                            full_command = f"/{server_name}/{prompt_name}"
                            yield Completion(
                                full_command,
                                start_position=-len(word),
                                display=full_command,
                                display_meta=prompt.description[:60] if prompt.description else "MCP prompt",
                            )

            # Completing server names: /server
            elif len(parts) == 1 and self.agent and self.agent.available_prompts:
                # Suggest server names that have prompts, plus all their prompts
                for server_name in self.agent.available_prompts.keys():
                    server_cmd = f"/{server_name}/"
                    if server_cmd.startswith(word.lower()):
                        yield Completion(
                            server_cmd,
                            start_position=-len(word),
                            display=server_cmd,
                            display_meta=f"MCP server ({len(self.agent.available_prompts[server_name])} prompts)",
                        )
                        # Also yield individual prompt completions for direct access
                        for prompt_name, prompt in self.agent.available_prompts[server_name].items():
                            full_command = f"/{server_name}/{prompt_name}"
                            yield Completion(
                                full_command,
                                start_position=-len(word),
                                display=full_command,
                                display_meta=prompt.description[:60] if prompt.description else "MCP prompt",
                            )

            # Standard command completion
            for cmd in self.commands:
                if cmd.startswith(word.lower()):
                    # Yield the remainder of the command
                    yield Completion(
                        cmd, start_position=-len(word), display=cmd, display_meta=self._get_command_description(cmd)
                    )

            # User-invocable skills: /<skill-name>
            if self.agent:
                builtins = set(self.commands)
                for name, skill in self.agent.skills_manager.loaded_skills.items():
                    if not skill.metadata.user_invocable:
                        continue
                    full_command = f"/{name}"
                    if full_command in builtins:
                        continue  # built-in command takes precedence
                    if full_command.startswith(word.lower()):
                        meta = skill.metadata.argument_hint or skill.metadata.description[:60]
                        yield Completion(
                            full_command,
                            start_position=-len(word),
                            display=full_command,
                            display_meta=meta,
                        )
        else:
            # Mid-sentence: check if cursor is on a /server/prompt token
            words = text.split()
            if not words:
                return
            last_word = words[-1]
            if not last_word.startswith("/"):
                return
            # Only complete MCP prompts mid-sentence, not built-in commands
            if not self.agent or not self.agent.available_prompts:
                return

            parts = last_word.lstrip("/").split("/")

            if len(parts) == 2 and parts[0] != "skill":
                server_name, prompt_prefix = parts
                if server_name in self.agent.available_prompts:
                    for prompt_name, prompt in self.agent.available_prompts[server_name].items():
                        if prompt_name.startswith(prompt_prefix.lower()):
                            full_token = f"/{server_name}/{prompt_name}"
                            yield Completion(
                                full_token,
                                start_position=-len(last_word),
                                display=full_token,
                                display_meta=prompt.description[:60] if prompt.description else "MCP prompt",
                            )
            elif len(parts) == 1:
                for server_name in self.agent.available_prompts.keys():
                    server_cmd = f"/{server_name}/"
                    if server_cmd.startswith(last_word.lower()):
                        yield Completion(
                            server_cmd,
                            start_position=-len(last_word),
                            display=server_cmd,
                            display_meta=f"MCP server ({len(self.agent.available_prompts[server_name])} prompts)",
                        )
                        # Also yield individual prompt completions
                        for prompt_name, prompt in self.agent.available_prompts[server_name].items():
                            full_token = f"/{server_name}/{prompt_name}"
                            yield Completion(
                                full_token,
                                start_position=-len(last_word),
                                display=full_token,
                                display_meta=prompt.description[:60] if prompt.description else "MCP prompt",
                            )

    def _get_command_description(self, command: str) -> str:
        """Get description for a command"""
        from .commands import get_command

        cmd = get_command(command)
        return cmd.description if cmd else ""
