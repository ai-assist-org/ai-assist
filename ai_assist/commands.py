"""Command validation utilities for ai-assist"""

# Valid commands in interactive mode
INTERACTIVE_COMMANDS = [
    "/exit",
    "/quit",
    "/status",
    "/history",
    "/clear-cache",
    "/help",
    "/clear",
    "/kg-save",
    "/kg-viz",
    "/awl-viz",
    "/eval-stats",
]

# Valid commands at CLI level
CLI_COMMANDS = [
    "/monitor",
    "/query",
    "/interactive",
    "/run",
    "/status",
    "/clear-cache",
    "/identity-show",
    "/identity-init",
    "/kg-stats",
    "/kg-asof",
    "/kg-late",
    "/kg-changes",
    "/kg-show",
    "/cleanup-actions",
    "/kg-viz",
    "/awl-viz",
    "/eval-stats",
    "/service",
    "/help",
]


def is_valid_interactive_command(user_input: str) -> bool:
    """Check if user input is a valid interactive command

    Args:
        user_input: User's input string

    Returns:
        True if it's a valid command or not a command at all
        False if it starts with / but is not a valid command
    """
    if not user_input.startswith("/"):
        # Not a command, let it through to the agent
        return True

    # Extract base command (before any space)
    base_command = user_input.split(maxsplit=1)[0].lower()

    return base_command in INTERACTIVE_COMMANDS


def is_valid_cli_command(command: str) -> bool:
    """Check if a CLI command is valid

    Args:
        command: Command string (without leading /)

    Returns:
        True if valid, False otherwise
    """
    return f"/{command}" in CLI_COMMANDS


def get_command_suggestion(user_input: str, is_interactive: bool = False) -> str:
    """Get a helpful error message for invalid commands

    Args:
        user_input: User's invalid input
        is_interactive: Whether in interactive mode

    Returns:
        Error message string
    """
    msg = f"Unknown command '{user_input}'\n\n"

    if is_interactive:
        msg += "Available commands:\n"
        for cmd in sorted(INTERACTIVE_COMMANDS):
            msg += f"  {cmd}\n"
        msg += "\nOr type a question without the / prefix to ask the AI assistant."
    else:
        msg += "Run 'ai-assist /help' to see available commands"

    return msg
