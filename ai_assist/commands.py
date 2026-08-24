"""Command registry and validation utilities for ai-assist"""

from dataclasses import dataclass


@dataclass
class CommandDef:
    """Single source of truth for a slash command."""

    name: str
    description: str
    mode: str = "both"  # "interactive", "cli", or "both"
    args: str = ""
    hidden: bool = False


COMMAND_REGISTRY: dict[str, CommandDef] = {}


def register_command(cmd: CommandDef) -> None:
    """Register a command in the global registry."""
    COMMAND_REGISTRY[cmd.name] = cmd


def get_command(name: str) -> CommandDef | None:
    """Look up a command by exact name."""
    return COMMAND_REGISTRY.get(name)


def get_interactive_commands() -> list[CommandDef]:
    """All commands available in interactive mode."""
    return [c for c in COMMAND_REGISTRY.values() if c.mode in ("interactive", "both")]


def get_cli_commands() -> list[CommandDef]:
    """All commands available at CLI level."""
    return [c for c in COMMAND_REGISTRY.values() if c.mode in ("cli", "both")]


# --- Interactive-only commands ---

register_command(CommandDef("/exit", "Exit interactive mode", mode="interactive"))
register_command(CommandDef("/quit", "Exit interactive mode", mode="interactive"))
register_command(CommandDef("/history", "Show recent monitoring history", mode="interactive"))
register_command(CommandDef("/clear", "Clear conversation memory", mode="interactive"))
register_command(CommandDef("/kg-save", "Toggle knowledge graph auto-save", mode="interactive", args="[on|off]"))
register_command(CommandDef("/prompts", "List available MCP prompts", mode="interactive"))
register_command(CommandDef("/prompt-info", "Show detailed prompt info", mode="interactive", args="<server/prompt>"))
register_command(CommandDef("/bg", "List background tasks", mode="interactive", args="[id|cancel [id]]"))
register_command(CommandDef("/bg cancel", "Cancel background task(s)", mode="interactive", hidden=True))
register_command(CommandDef("/plan", "Plan a task before executing", mode="interactive", args="<task>"))
register_command(CommandDef("/search", "Search conversation history", mode="interactive"))

# --- Skill management (interactive-only) ---

register_command(
    CommandDef(
        "/skill/install",
        "Install an Agent Skill from git, local path, or ClawHub",
        mode="interactive",
        args="<source>@<branch>",
    )
)
register_command(
    CommandDef("/skill/uninstall", "Uninstall an installed Agent Skill", mode="interactive", args="<name>")
)
register_command(CommandDef("/skill/list", "List all installed Agent Skills", mode="interactive"))
register_command(
    CommandDef("/skill/search", "Search ClawHub and skills.sh for skills", mode="interactive", args="<query>")
)
register_command(
    CommandDef("/skill/add_env", "Allow an env var for a skill's scripts", mode="interactive", args="<skill> <VAR>")
)
register_command(
    CommandDef("/skill/remove_env", "Remove an allowed env var from a skill", mode="interactive", args="<skill> <VAR>")
)
register_command(CommandDef("/skill/list_env", "Show allowed env vars for skills", mode="interactive", args="[skill]"))

# --- Plugin management (interactive-only) ---

register_command(
    CommandDef(
        "/plugin/install",
        "Install a Claude Code plugin from git, local path, or marketplace name",
        mode="interactive",
        args="<source|name>@<branch>",
    )
)
register_command(CommandDef("/plugin/uninstall", "Uninstall an installed plugin", mode="interactive", args="<name>"))
register_command(CommandDef("/plugin/list", "List all installed plugins", mode="interactive"))
register_command(
    CommandDef("/plugin/search", "Search registered marketplaces for plugins", mode="interactive", args="<query>")
)
register_command(
    CommandDef(
        "/plugin/marketplace",
        "Manage plugin marketplaces (add <repo> | list)",
        mode="interactive",
        args="<add <repo>|list>",
    )
)

# --- MCP management (interactive-only) ---

register_command(CommandDef("/mcp/restart", "Restart an MCP server", mode="interactive", args="<server>"))

# --- Commands available in both modes ---

register_command(CommandDef("/status", "Show state statistics"))
register_command(CommandDef("/clear-cache", "Clear expired cache"))
register_command(CommandDef("/help", "Show help message"))
register_command(CommandDef("/kg-viz", "Visualize knowledge graph in browser"))
register_command(CommandDef("/awl-viz", "Visualize an AWL workflow in browser", args="[script.awl]"))
register_command(CommandDef("/eval-stats", "Show evaluation metrics from query traces"))
register_command(CommandDef("/cost", "Show token cost summary", args="[period]"))

# --- CLI-only commands ---

register_command(CommandDef("/monitor", "Start monitoring mode", mode="cli"))
register_command(CommandDef("/query", "Run a one-off query", mode="cli", args="<question>"))
register_command(CommandDef("/interactive", "Start interactive mode", mode="cli"))
register_command(CommandDef("/run", "Execute an AWL workflow", mode="cli", args="<script.awl>"))
register_command(CommandDef("/identity-show", "Show current identity", mode="cli"))
register_command(CommandDef("/identity-init", "Initialize identity", mode="cli"))
register_command(CommandDef("/kg-stats", "Show knowledge graph statistics", mode="cli"))
register_command(CommandDef("/kg-asof", "Show knowledge graph state at a point in time", mode="cli"))
register_command(CommandDef("/kg-late", "Show stale knowledge graph entities", mode="cli"))
register_command(CommandDef("/kg-changes", "Show recent knowledge graph changes", mode="cli"))
register_command(CommandDef("/kg-show", "Show knowledge graph entity details", mode="cli"))
register_command(CommandDef("/cleanup-actions", "Clean up old action records", mode="cli"))
register_command(CommandDef("/sandbox", "Manage sandbox environments", mode="cli", args="<subcommand>"))
register_command(CommandDef("/service", "Manage persistent service", mode="cli", args="<subcommand>"))


# Backward-compatible list for imports (computed from registry)
INTERACTIVE_COMMANDS = [c.name for c in COMMAND_REGISTRY.values() if c.mode in ("interactive", "both") and not c.hidden]
CLI_COMMANDS = [c.name for c in COMMAND_REGISTRY.values() if c.mode in ("cli", "both")]


def is_valid_interactive_command(user_input: str) -> bool:
    """Check if user input is a valid interactive command.

    Returns True if it's a valid command or not a command at all.
    Returns False if it starts with / but is not a valid interactive command.
    """
    if not user_input.startswith("/"):
        return True

    base_command = user_input.split(maxsplit=1)[0].lower()

    cmd = COMMAND_REGISTRY.get(base_command)
    if cmd and cmd.mode in ("interactive", "both"):
        return True

    # Check if it matches a prefix (e.g. "/skill/install foo" matches "/skill/install")
    for name, cmd_def in COMMAND_REGISTRY.items():
        if base_command.startswith(name) and cmd_def.mode in ("interactive", "both"):
            return True

    return False


def is_valid_cli_command(command: str) -> bool:
    """Check if a CLI command is valid.

    Args:
        command: Command string (without leading /)
    """
    cmd = COMMAND_REGISTRY.get(f"/{command}")
    return cmd is not None and cmd.mode in ("cli", "both")


def get_command_suggestion(user_input: str, is_interactive: bool = False) -> str:
    """Get a helpful error message for invalid commands."""
    msg = f"Unknown command '{user_input}'\n\n"

    if is_interactive:
        msg += "Available commands:\n"
        for cmd in sorted(INTERACTIVE_COMMANDS):
            msg += f"  {cmd}\n"
        msg += "\nOr type a question without the / prefix to ask the AI assistant."
    else:
        msg += "Run 'ai-assist /help' to see available commands"

    return msg
