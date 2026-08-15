"""Tests for command registry and validation utilities"""

from ai_assist.commands import (
    COMMAND_REGISTRY,
    INTERACTIVE_COMMANDS,
    get_cli_commands,
    get_command,
    get_command_suggestion,
    get_interactive_commands,
    is_valid_cli_command,
    is_valid_interactive_command,
)


class TestCommandRegistry:
    """Tests for the command registry"""

    def test_registry_is_populated(self):
        """Test that commands are registered at module load time"""
        assert len(COMMAND_REGISTRY) > 0

    def test_get_command_returns_registered_command(self):
        """Test exact lookup by name"""
        cmd = get_command("/status")
        assert cmd is not None
        assert cmd.name == "/status"
        assert cmd.description != ""

    def test_get_command_returns_none_for_unknown(self):
        """Test that unknown commands return None"""
        assert get_command("/nonexistent") is None

    def test_get_interactive_commands_excludes_cli_only(self):
        """Test that CLI-only commands are excluded from interactive list"""
        interactive = get_interactive_commands()
        names = [c.name for c in interactive]
        assert "/status" in names
        assert "/help" in names
        assert "/monitor" not in names
        assert "/query" not in names

    def test_get_cli_commands_excludes_interactive_only(self):
        """Test that interactive-only commands are excluded from CLI list"""
        cli = get_cli_commands()
        names = [c.name for c in cli]
        assert "/monitor" in names
        assert "/help" in names
        assert "/exit" not in names
        assert "/quit" not in names

    def test_both_mode_commands_appear_in_both_lists(self):
        """Test that 'both' mode commands appear in both interactive and CLI"""
        interactive_names = [c.name for c in get_interactive_commands()]
        cli_names = [c.name for c in get_cli_commands()]
        assert "/status" in interactive_names
        assert "/status" in cli_names
        assert "/help" in interactive_names
        assert "/help" in cli_names

    def test_previously_missing_commands_are_now_registered(self):
        """Test that commands that were missing from the old lists are now included"""
        interactive_names = [c.name for c in get_interactive_commands()]
        assert "/prompts" in interactive_names
        assert "/prompt-info" in interactive_names
        assert "/skill/install" in interactive_names
        assert "/skill/uninstall" in interactive_names
        assert "/skill/list" in interactive_names
        assert "/mcp/restart" in interactive_names

    def test_interactive_commands_list_backward_compatible(self):
        """Test that the INTERACTIVE_COMMANDS list contains expected commands"""
        assert "/exit" in INTERACTIVE_COMMANDS
        assert "/quit" in INTERACTIVE_COMMANDS
        assert "/status" in INTERACTIVE_COMMANDS
        assert "/help" in INTERACTIVE_COMMANDS
        assert "/plan" in INTERACTIVE_COMMANDS

    def test_hidden_commands_excluded_from_interactive_list(self):
        """Test that hidden commands are excluded from the backward-compatible list"""
        assert "/bg cancel" not in INTERACTIVE_COMMANDS

    def test_all_commands_have_descriptions(self):
        """Test that every registered command has a non-empty description"""
        for name, cmd in COMMAND_REGISTRY.items():
            assert cmd.description, f"Command {name} has no description"

    def test_all_commands_have_valid_mode(self):
        """Test that every registered command has a valid mode"""
        for name, cmd in COMMAND_REGISTRY.items():
            assert cmd.mode in ("interactive", "cli", "both"), f"Command {name} has invalid mode: {cmd.mode}"


class TestInteractiveCommandValidation:
    """Tests for interactive command validation"""

    def test_valid_interactive_commands(self):
        """Test that valid interactive commands are recognized"""
        for cmd in INTERACTIVE_COMMANDS:
            assert is_valid_interactive_command(cmd), f"Command {cmd} should be valid"

    def test_invalid_interactive_command(self):
        """Test that invalid commands starting with / are rejected"""
        assert not is_valid_interactive_command("/invalid")
        assert not is_valid_interactive_command("/unknown-command")

    def test_non_command_input_is_valid(self):
        """Test that input without / is considered valid (for agent)"""
        assert is_valid_interactive_command("what is the weather?")
        assert is_valid_interactive_command("hello")
        assert is_valid_interactive_command("analyze DCI jobs")

    def test_command_with_arguments(self):
        """Test that commands with arguments are validated by base command"""
        assert is_valid_interactive_command("/kg-save on")
        assert is_valid_interactive_command("/kg-save off")
        assert not is_valid_interactive_command("/invalid argument")

    def test_skill_commands_are_valid(self):
        """Test that /skill/* commands pass validation"""
        assert is_valid_interactive_command("/skill/install foo@main")
        assert is_valid_interactive_command("/skill/uninstall bar")
        assert is_valid_interactive_command("/skill/list")

    def test_mcp_restart_is_valid(self):
        """Test that /mcp/restart passes validation"""
        assert is_valid_interactive_command("/mcp/restart dci")

    def test_prompts_is_valid(self):
        """Test that /prompts passes validation"""
        assert is_valid_interactive_command("/prompts")
        assert is_valid_interactive_command("/prompt-info dci/rca")


class TestCLICommandValidation:
    """Tests for CLI command validation"""

    def test_valid_cli_commands(self):
        """Test that valid CLI commands are recognized"""
        valid_commands = [
            "monitor",
            "query",
            "interactive",
            "status",
            "clear-cache",
            "identity-show",
            "identity-init",
            "kg-stats",
            "kg-asof",
            "kg-late",
            "kg-changes",
            "kg-show",
            "help",
        ]
        for cmd in valid_commands:
            assert is_valid_cli_command(cmd), f"Command {cmd} should be valid"

    def test_invalid_cli_command(self):
        """Test that invalid CLI commands are rejected"""
        assert not is_valid_cli_command("invalid")
        assert not is_valid_cli_command("unknown-command")


class TestCommandSuggestion:
    """Tests for command suggestion messages"""

    def test_interactive_suggestion_shows_available_commands(self):
        """Test that interactive suggestion shows available commands"""
        msg = get_command_suggestion("/invalid", is_interactive=True)
        assert "Unknown command '/invalid'" in msg
        assert "Available commands:" in msg
        assert "/help" in msg
        assert "/exit" in msg

    def test_interactive_suggestion_mentions_asking_without_slash(self):
        """Test that interactive suggestion mentions asking without slash"""
        msg = get_command_suggestion("/invalid", is_interactive=True)
        assert "type a question without the / prefix" in msg.lower()

    def test_cli_suggestion_directs_to_help(self):
        """Test that CLI suggestion directs to help command"""
        msg = get_command_suggestion("/invalid", is_interactive=False)
        assert "Unknown command '/invalid'" in msg
        assert "ai-assist /help" in msg

    def test_suggestion_formats_command_correctly(self):
        """Test that suggestions format commands correctly"""
        msg = get_command_suggestion("/test", is_interactive=True)
        assert "/test" in msg
