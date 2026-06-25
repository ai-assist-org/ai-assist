"""Filesystem tools for ai-assist agent"""

import json
import re
import shlex
import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .config import AiAssistConfig, get_config_dir

ALLOWED_COMMANDS_FILE = "allowed_commands.json"
ALLOWED_PATHS_FILE = "allowed_paths.json"

SHELL_BUILTINS = frozenset(
    {
        "cd",
        "echo",
        "export",
        "source",
        ".",
        "set",
        "unset",
        "pwd",
        "pushd",
        "popd",
        "dirs",
        "type",
        "hash",
        "alias",
        "unalias",
        "read",
        "true",
        "false",
        "test",
        "[",
        "printf",
        "eval",
        "exec",
        "builtin",
        "command",
        "declare",
        "local",
        "readonly",
        "shift",
        "trap",
        "return",
        "exit",
        "break",
        "continue",
        "wait",
        "bg",
        "fg",
        "jobs",
        "umask",
        "ulimit",
        "shopt",
        "enable",
        "help",
        "history",
        "let",
        "getopts",
        "mapfile",
        "readarray",
        "compgen",
        "complete",
        "logout",
        "times",
        # Shell flow-control keywords (not commands, but extracted by _split_shell_commands)
        "for",
        "do",
        "done",
        "while",
        "until",
        "if",
        "then",
        "else",
        "elif",
        "fi",
        "case",
        "esac",
        "in",
        "select",
        "function",
        "time",
        "coproc",
        "{",
        "}",
        "!",
        "[[",
        "]]",
    }
)

PYTHON_COMMANDS = frozenset({"python", "python3"})

TRANSPARENT_WRAPPERS = frozenset({"env", "nohup", "nice", "ionice", "timeout", "stdbuf", "script"})

PRIVILEGE_WRAPPERS = frozenset({"sudo", "su", "doas"})

PROTECTED_CONFIG_FILES = frozenset(
    {
        ALLOWED_COMMANDS_FILE,
        ALLOWED_PATHS_FILE,
        "skill_env.json",
    }
)

ENV_VAR_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _resolve_command_token(token: str) -> str:
    """Return the effective command name for allowlist matching.

    Bare names (e.g. "grep") are returned as-is — the shell resolves them via PATH.
    Fully qualified or home-relative paths (e.g. "/usr/bin/grep", "~/bin/tool")
    are kept as-is so that the allowlist must explicitly permit that exact path.
    This prevents a malicious binary at an arbitrary path from matching a
    basename-only allowlist entry.
    """
    if "/" in token or token.startswith("~"):
        return token
    return token


_NUMERIC_ARG = re.compile(r"^\d+(\.\d+)?[smhd]?$")


def _skip_wrapper_args(tokens: list[str], idx: int) -> int:
    """Skip flags and positional args belonging to a transparent wrapper.

    After the wrapper name has been consumed, this skips:
    - env var assignments (KEY=value)
    - flags (-x, --long) and their non-flag argument
    - bare numeric tokens (e.g. timeout's duration, nice's adjustment)
    Stops at the first token that looks like a command name.
    """
    while idx < len(tokens):
        tok = tokens[idx]
        if ENV_VAR_PATTERN.match(tok):
            idx += 1
        elif tok.startswith("-"):
            idx += 1
            if idx < len(tokens) and not tokens[idx].startswith("-") and not ENV_VAR_PATTERN.match(tokens[idx]):
                # Could be a flag value (e.g. -n 10) or the real command.
                # If it's numeric, consume it as a flag value.
                if _NUMERIC_ARG.match(tokens[idx]):
                    idx += 1
        elif _NUMERIC_ARG.match(tok):
            idx += 1
        else:
            break
    return idx


def _strip_shell_comments(command: str) -> str:
    """Strip shell comments (# to end) outside of quotes.

    Respects single quotes, double quotes, and backslash escapes.
    """
    result: list[str] = []
    in_single = False
    in_double = False
    i = 0
    while i < len(command):
        c = command[i]
        if c == "\\" and not in_single and i + 1 < len(command):
            result.append(c)
            result.append(command[i + 1])
            i += 2
            continue
        if c == "'" and not in_double:
            in_single = not in_single
        elif c == '"' and not in_single:
            in_double = not in_double
        elif c == "#" and not in_single and not in_double:
            break
        result.append(c)
        i += 1
    return "".join(result)


def _split_shell_commands(command: str) -> list[str]:
    """Split command on shell operators (&&, ||, ;, |, &) outside of quotes."""
    segments: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    i = 0
    while i < len(command):
        c = command[i]
        if c == "\\" and not in_single and i + 1 < len(command):
            current.append(c)
            current.append(command[i + 1])
            i += 2
            continue
        if c == "'" and not in_double:
            in_single = not in_single
            current.append(c)
        elif c == '"' and not in_single:
            in_double = not in_double
            current.append(c)
        elif not in_single and not in_double:
            if c == ";":
                segments.append("".join(current))
                current = []
            elif c == "|":
                if i + 1 < len(command) and command[i + 1] == "|":
                    segments.append("".join(current))
                    current = []
                    i += 1
                else:
                    segments.append("".join(current))
                    current = []
            elif c == "&":
                if i + 1 < len(command) and command[i + 1] == "&":
                    segments.append("".join(current))
                    current = []
                    i += 1
                elif i > 0 and command[i - 1] == ">":
                    current.append(c)
                else:
                    segments.append("".join(current))
                    current = []
            else:
                current.append(c)
        else:
            current.append(c)
        i += 1
    if current:
        segments.append("".join(current))
    return [s.strip() for s in segments if s.strip()]


def _is_safe_env_assignment(token: str) -> bool:
    """Check if a token is a safe env var assignment (no command substitution).

    Returns False for assignments containing $( ), backticks, or ${ } which
    can execute arbitrary commands during variable expansion.
    """
    if not ENV_VAR_PATTERN.match(token):
        return False
    value = token.split("=", 1)[1]
    temp = value.replace("$((", "")
    if "$(" in temp or "`" in value or "${" in value:
        return False
    return True


def extract_command_names(command: str) -> list[str]:
    """Extract all command names from a shell command string.

    Handles comments, env var prefixes, compound commands, pipes,
    and full paths. Detects unsafe env var assignments containing
    command substitution.
    """
    stripped = _strip_shell_comments(command)
    if not stripped.strip():
        return []

    segments = _split_shell_commands(stripped)
    commands: list[str] = []

    for segment in segments:
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()

        if not tokens:
            continue

        # Skip env var assignments, flag unsafe ones only when they prefix a command.
        # Standalone assignments like `result=$(cmd -i file)` are shell variable
        # captures — shlex mis-splits them but they execute no separate command.
        idx = 0
        while idx < len(tokens) and ENV_VAR_PATTERN.match(tokens[idx]):
            idx += 1

        if idx >= len(tokens):
            continue

        # Strip transparent wrappers and their arguments
        while idx < len(tokens) and _resolve_command_token(tokens[idx]) in TRANSPARENT_WRAPPERS:
            idx += 1
            idx = _skip_wrapper_args(tokens, idx)

        if idx >= len(tokens):
            continue

        cmd_token = tokens[idx].lstrip("(").rstrip(")")
        if not cmd_token or cmd_token.startswith("-"):
            continue
        cmd_name = _resolve_command_token(cmd_token)
        commands.append(cmd_name)

    return commands


def compute_allowlist_prefix(command: str) -> str | None:
    """Compute a meaningful allowlist prefix from a full command string.

    Strips transparent wrappers (env, nohup) and env var assignments.
    Keeps privilege wrappers (sudo, su) as part of the prefix.
    Returns command + first non-flag argument as the prefix.

    Returns None if the command is empty or only builtins/wrappers.
    """
    stripped = _strip_shell_comments(command)
    if not stripped.strip():
        return None

    segments = _split_shell_commands(stripped)
    if not segments:
        return None

    # Use only the first segment (before pipes/&&/||)
    try:
        tokens = shlex.split(segments[0])
    except ValueError:
        tokens = segments[0].split()

    if not tokens:
        return None

    # Skip env var assignments
    idx = 0
    while idx < len(tokens) and ENV_VAR_PATTERN.match(tokens[idx]):
        idx += 1

    if idx >= len(tokens):
        return None

    # Strip transparent wrappers and their arguments
    while idx < len(tokens) and _resolve_command_token(tokens[idx]) in TRANSPARENT_WRAPPERS:
        idx += 1
        idx = _skip_wrapper_args(tokens, idx)

    if idx >= len(tokens):
        return None

    prefix_parts: list[str] = []

    # Keep privilege wrappers as part of the prefix
    cmd_token = _resolve_command_token(tokens[idx])
    if cmd_token in PRIVILEGE_WRAPPERS:
        prefix_parts.append(cmd_token)
        idx += 1
        # Skip flags after sudo (e.g. sudo -u user)
        while idx < len(tokens) and tokens[idx].startswith("-"):
            idx += 1
            # Skip the flag's argument if it's a known value-flag
            if idx < len(tokens) and not tokens[idx].startswith("-"):
                idx += 1

    if idx >= len(tokens):
        return " ".join(prefix_parts) if prefix_parts else None

    # Add the command name (preserving full path if given)
    cmd_name = _resolve_command_token(tokens[idx])
    if cmd_name in SHELL_BUILTINS:
        return None
    prefix_parts.append(cmd_name)
    idx += 1

    # Add first non-flag argument, but skip if preceded by inline-code
    # flags like -c or -e (means next token is code, not a script/subcommand)
    saw_inline_flag = False
    while idx < len(tokens):
        token = tokens[idx]
        if token.startswith("-"):
            if token in ("-c", "-e"):
                saw_inline_flag = True
            idx += 1
        else:
            if not saw_inline_flag:
                prefix_parts.append(token)
            break

    return " ".join(prefix_parts)


def _extract_command_argument_paths(command: str) -> list[tuple[str, str]]:
    """Extract paths from command arguments that need validation.

    Returns list of (command_name, path_or_marker) tuples for commands whose
    path arguments should be checked against allowed directories:
    - cd <dir>: the target directory
    - find <paths...>: the search paths (before option flags)
    - python/python3 <script>: the script file path

    Special markers for python:
    - ("<inline-code>") for python -c
    - ("<stdin>") for python -
    - ("<interactive>") for python with no arguments
    """
    stripped = _strip_shell_comments(command)
    if not stripped.strip():
        return []

    segments = _split_shell_commands(stripped)
    results: list[tuple[str, str]] = []

    for segment in segments:
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()

        if not tokens:
            continue

        # Skip env var prefixes
        idx = 0
        while idx < len(tokens) and ENV_VAR_PATTERN.match(tokens[idx]):
            idx += 1

        if idx >= len(tokens):
            continue

        cmd_name = Path(tokens[idx]).name
        args = tokens[idx + 1 :]

        if cmd_name == "cd" and args:
            target = args[0]
            if target != "-":
                results.append(("cd", target))

        elif cmd_name == "find":
            for arg in args:
                if arg.startswith("-") or arg.startswith("(") or arg == "!":
                    break
                results.append(("find", arg))

        elif cmd_name in PYTHON_COMMANDS:
            i = 0
            found = False
            while i < len(args):
                arg = args[i]
                if arg == "-c":
                    results.append((cmd_name, "<inline-code>"))
                    found = True
                    break
                elif arg == "-m":
                    found = True
                    break
                elif arg == "-":
                    results.append((cmd_name, "<stdin>"))
                    found = True
                    break
                elif arg.startswith("-"):
                    i += 1
                    continue
                else:
                    results.append((cmd_name, arg))
                    found = True
                    break
                i += 1
            if not found and not args:
                results.append((cmd_name, "<interactive>"))

    return results


class FilesystemTools:
    """Internal filesystem tools for the agent"""

    def __init__(self, config: AiAssistConfig, load_user_config: bool = True):
        """Initialize filesystem tools

        Args:
            config: AiAssistConfig instance for security settings
            load_user_config: Whether to load user-specific config from ~/.ai-assist (default True)
        """
        self.allowed_commands = list(config.allowed_commands)
        if load_user_config:
            self._load_user_allowed_commands()
        self.allowed_paths = [Path(p).expanduser().resolve() for p in config.allowed_paths if p]
        self._path_restrictions_enabled = bool(self.allowed_paths)
        if load_user_config:
            self._load_user_allowed_paths()
        self.confirm_tools = config.confirm_tools
        self.confirmation_callback: Callable[[str], Awaitable[bool]] | None = None
        self.path_confirmation_callback: Callable[[str], Awaitable[bool]] | None = None
        self.awl_authorized_commands: set[str] = set()

    def _load_user_allowed_commands(self):
        """Load user-added allowed commands from persistent file.

        Skips entries that are shell builtins, single characters, or
        otherwise invalid (likely saved by accident from older versions).
        """
        path = get_config_dir() / ALLOWED_COMMANDS_FILE
        if not path.exists():
            return
        try:
            with open(path) as f:
                commands = json.load(f)
            for cmd in commands:
                if not isinstance(cmd, str) or len(cmd) <= 1:
                    continue
                first_word = cmd.split()[0]
                if first_word in SHELL_BUILTINS:
                    continue
                if cmd not in self.allowed_commands:
                    self.allowed_commands.append(cmd)
        except json.JSONDecodeError, OSError:
            pass

    def add_permanent_allowed_command(self, cmd_name: str):
        """Add a command to the persistent allowlist."""
        if cmd_name not in self.allowed_commands:
            self.allowed_commands.append(cmd_name)

        path = get_config_dir() / ALLOWED_COMMANDS_FILE
        try:
            existing: list[str] = []
            if path.exists():
                with open(path) as f:
                    existing = json.load(f)
        except json.JSONDecodeError, OSError:
            existing = []

        if cmd_name not in existing:
            existing.append(cmd_name)
            with open(path, "w") as f:
                json.dump(existing, f, indent=2)

    def _load_user_allowed_paths(self):
        """Load user-added allowed paths from persistent file."""
        path = get_config_dir() / ALLOWED_PATHS_FILE
        if not path.exists():
            return
        try:
            with open(path) as f:
                paths = json.load(f)
            for p in paths:
                resolved = Path(p).expanduser().resolve()
                if resolved not in self.allowed_paths:
                    self.allowed_paths.append(resolved)
        except json.JSONDecodeError, OSError:
            pass

    def add_permanent_allowed_path(self, path_str: str):
        """Add a path to the persistent allowed paths list."""
        resolved = Path(path_str).expanduser().resolve()
        if resolved not in self.allowed_paths:
            self.allowed_paths.append(resolved)

        persist_path = get_config_dir() / ALLOWED_PATHS_FILE
        try:
            existing: list[str] = []
            if persist_path.exists():
                with open(persist_path) as f:
                    existing = json.load(f)
        except json.JSONDecodeError, OSError:
            existing = []

        if path_str not in existing:
            existing.append(path_str)
            with open(persist_path, "w") as f:
                json.dump(existing, f, indent=2)

    @staticmethod
    def _is_protected_config(path_str: str) -> bool:
        """Check if a path points to a security-sensitive config file."""
        resolved = Path(path_str).expanduser().resolve()
        config_dir = get_config_dir().resolve()
        for protected in PROTECTED_CONFIG_FILES:
            if resolved == config_dir / protected:
                return True
        return False

    async def _validate_path(self, path_str: str) -> str | None:
        """Validate that a path is within allowed directories.
        Falls back to path_confirmation_callback if path is blocked.

        Returns:
            Error message if path is not allowed, None if validation passes
        """
        if not self._path_restrictions_enabled:
            return None

        resolved = Path(path_str).expanduser().resolve()

        for allowed in self.allowed_paths:
            try:
                resolved.relative_to(allowed)
                return None
            except ValueError:
                continue

        # Path not in allowlist — try interactive approval
        if self.path_confirmation_callback is not None:
            description = f"Access path: {resolved}"
            approved = await self.path_confirmation_callback(description)
            if approved:
                return None
            return f"Error: Path access rejected by user: {resolved}"

        allowed_list = ", ".join(str(p) for p in self.allowed_paths)
        return f"Error: Path '{resolved}' is outside allowed directories: {allowed_list}. Not allowed."

    def get_tool_definitions(self) -> list[dict]:
        """Get tool definitions for the agent

        Returns:
            List of tool definitions in Anthropic tool format
        """
        return [
            {
                "name": "internal__read_file",
                "description": "Read the contents of a file from the filesystem. Can read the entire file or specific line ranges. For large files, use line_start and line_end to read only relevant sections. Use search_in_file first to find line numbers of interest.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read (~ is expanded automatically)",
                        },
                        "line_start": {
                            "type": "integer",
                            "description": "Starting line number (1-indexed). If specified, reads from this line.",
                            "default": None,
                        },
                        "line_end": {
                            "type": "integer",
                            "description": "Ending line number (1-indexed). If specified with line_start, reads only this range.",
                            "default": None,
                        },
                        "max_lines": {
                            "type": "integer",
                            "description": "Maximum number of lines to read (default: unlimited)",
                            "default": None,
                        },
                    },
                    "required": ["path"],
                },
                "_server": "internal",
                "_original_name": "read_file",
            },
            {
                "name": "internal__search_in_file",
                "description": "Search for a regex pattern in a file. Returns matching lines with line numbers. Use this to find specific patterns, errors, or keywords in log files. Supports line ranges to search only a portion of the file (e.g., last 100 lines with line_start=-100).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to search (~ is expanded automatically)",
                        },
                        "pattern": {"type": "string", "description": "Regex pattern to search for"},
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of matching lines to return (default: 100)",
                            "default": 100,
                        },
                        "line_start": {
                            "type": "integer",
                            "description": "Start searching from this line number (1-based). Negative values count from the end (e.g., -100 = last 100 lines). Default: search entire file.",
                        },
                        "line_end": {
                            "type": "integer",
                            "description": "Stop searching at this line number (inclusive). Negative values count from the end. Default: search to end of file.",
                        },
                    },
                    "required": ["path", "pattern"],
                },
                "_server": "internal",
                "_original_name": "search_in_file",
            },
            {
                "name": "internal__create_directory",
                "description": "Create a directory (and parent directories if needed). Use this to create directories for storing reports, logs, or downloaded files.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the directory to create (~ is expanded automatically)",
                        }
                    },
                    "required": ["path"],
                },
                "_server": "internal",
                "_original_name": "create_directory",
            },
            {
                "name": "internal__list_directory",
                "description": "List files and directories in a directory with type and size details. Always prefer this over execute_command for listing directory contents.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the directory to list (~ is expanded automatically)",
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Optional glob pattern to filter results (e.g., '*.log')",
                            "default": "*",
                        },
                    },
                    "required": ["path"],
                },
                "_server": "internal",
                "_original_name": "list_directory",
            },
            {
                "name": "internal__write_file",
                "description": "Write content to a file. Creates the file if it doesn't exist, or replaces its content if it does. Parent directories are created automatically.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to write (~ is expanded automatically)",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file",
                        },
                    },
                    "required": ["path", "content"],
                },
                "_server": "internal",
                "_original_name": "write_file",
            },
            {
                "name": "internal__edit_file",
                "description": "Edit a file by replacing an exact string match with new content. The old_string must appear exactly once in the file (for safety). Read the file first to get the exact text to replace. Works with multi-line strings.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to edit (~ is expanded automatically)",
                        },
                        "old_string": {
                            "type": "string",
                            "description": "The exact text to find and replace (must appear exactly once in the file)",
                        },
                        "new_string": {
                            "type": "string",
                            "description": "The text to replace it with",
                        },
                    },
                    "required": ["path", "old_string", "new_string"],
                },
                "_server": "internal",
                "_original_name": "edit_file",
            },
            {
                "name": "internal__execute_command",
                "description": "Execute a shell command and return the output. Only use for commands that have no dedicated tool (prefer internal__read_file, internal__list_directory, internal__search_in_file for filesystem operations). Commands are checked against an allowlist.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The command to execute"},
                        "working_directory": {
                            "type": "string",
                            "description": "Optional working directory for the command (default: current directory)",
                            "default": None,
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds (default: 30, max: 300)",
                            "default": 30,
                        },
                    },
                    "required": ["command"],
                },
                "_server": "internal",
                "_original_name": "execute_command",
            },
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a filesystem tool

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool result as string
        """
        if tool_name == "read_file":
            return await self._read_file(arguments)
        elif tool_name == "search_in_file":
            return await self._search_in_file(arguments)
        elif tool_name == "create_directory":
            return await self._create_directory(arguments)
        elif tool_name == "list_directory":
            return await self._list_directory(arguments)
        elif tool_name == "write_file":
            return await self._write_file(arguments)
        elif tool_name == "edit_file":
            return await self._edit_file(arguments)
        elif tool_name == "execute_command":
            return await self._execute_command(arguments)
        else:
            return f"Error: Unknown filesystem tool: {tool_name}"

    async def _check_confirmation(self, tool_full_name: str, description: str) -> str | None:
        """Check if a tool requires confirmation and prompt the user.

        Args:
            tool_full_name: Full tool name (e.g. "internal__create_directory")
            description: Human-readable description of the action

        Returns:
            Error message if rejected/blocked, None if approved
        """
        if tool_full_name not in self.confirm_tools:
            return None

        if self.confirmation_callback is None:
            return None

        approved = await self.confirmation_callback(description)
        if not approved:
            return f"Error: Action rejected by user: {description}"

        return None

    async def _read_file(self, args: dict) -> str:
        """Read a file from the filesystem"""
        path = args.get("path")
        line_start = args.get("line_start")
        line_end = args.get("line_end")
        max_lines = args.get("max_lines")

        if not path:
            return "Error: path parameter is required"

        path_error = await self._validate_path(path)
        if path_error:
            return path_error

        try:
            path_obj = Path(path).expanduser()

            if not path_obj.exists():
                return f"Error: File not found: {path}"

            if not path_obj.is_file():
                return f"Error: Not a file: {path}"

            with open(path_obj, encoding="utf-8", errors="replace") as f:
                if line_start is not None or line_end is not None or max_lines is not None:
                    lines: list[str] = []
                    start = line_start if line_start else 1
                    end = line_end if line_end else float("inf")
                    limit = max_lines if max_lines else float("inf")

                    for line_num, line in enumerate(f, 1):
                        if line_num < start:
                            continue
                        if line_num > end:
                            break
                        if len(lines) >= limit:
                            break
                        lines.append(f"{line_num}\t{line.rstrip()}")

                    content = "\n".join(lines)

                    range_info = f"lines {start}"
                    if line_end:
                        range_info = f"lines {start}-{line_end}"
                    elif max_lines:
                        range_info += f" (max {max_lines} lines)"

                    return f"File contents ({path_obj.name}, {range_info}, {len(lines)} lines, {len(content)} chars):\n\n{content}"

                else:
                    content = f.read()
                    return f"File contents ({len(content)} chars):\n\n{content}"

        except Exception as e:
            return f"Error reading file: {str(e)}"

    async def _search_in_file(self, args: dict) -> str:
        """Search for a pattern in a file, optionally within a line range"""
        path = args.get("path")
        pattern = args.get("pattern")
        max_results = args.get("max_results", 100)
        line_start = args.get("line_start")
        line_end = args.get("line_end")

        if not path or not pattern:
            return "Error: path and pattern parameters are required"

        path_error = await self._validate_path(path)
        if path_error:
            return path_error

        try:
            path_obj = Path(path).expanduser()

            if not path_obj.exists():
                return f"Error: File not found: {path}"

            if not path_obj.is_file():
                return f"Error: Not a file: {path}"

            try:
                regex = re.compile(pattern)
            except re.error as e:
                return f"Error: Invalid regex pattern: {e}"

            # Read all lines to support negative indices
            with open(path_obj, encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()

            total_lines = len(all_lines)

            # Resolve line range (1-based, negative counts from end)
            if line_start is not None:
                if line_start < 0:
                    start_idx = max(0, total_lines + line_start)
                else:
                    start_idx = max(0, line_start - 1)
            else:
                start_idx = 0

            if line_end is not None:
                if line_end < 0:
                    end_idx = max(0, total_lines + line_end + 1)
                else:
                    end_idx = min(total_lines, line_end)
            else:
                end_idx = total_lines

            # Search within range
            matches = []
            for idx in range(start_idx, end_idx):
                line = all_lines[idx]
                if regex.search(line):
                    matches.append(f"{idx + 1}: {line.rstrip()}")
                    if len(matches) >= max_results:
                        break

            range_info = ""
            if line_start is not None or line_end is not None:
                range_info = f" (lines {start_idx + 1}-{end_idx} of {total_lines})"

            if not matches:
                return f"No matches found for pattern: {pattern}{range_info}"

            result = f"Found {len(matches)} match(es) for pattern '{pattern}'{range_info}:\n\n"
            result += "\n".join(matches)

            if len(matches) >= max_results:
                result += f"\n\n... (limited to {max_results} results)"

            return result

        except Exception as e:
            return f"Error searching file: {str(e)}"

    async def _create_directory(self, args: dict) -> str:
        """Create a directory"""
        path = args.get("path")
        if not path:
            return "Error: path parameter is required"

        path_error = await self._validate_path(path)
        if path_error:
            return path_error

        confirm_error = await self._check_confirmation("internal__create_directory", f"Create directory: {path}")
        if confirm_error:
            return confirm_error

        try:
            path_obj = Path(path).expanduser()

            path_obj.mkdir(parents=True, exist_ok=True)

            return f"Directory created: {path_obj}"

        except Exception as e:
            return f"Error creating directory: {str(e)}"

    async def _list_directory(self, args: dict) -> str:
        """List files and directories"""
        path = args.get("path")
        pattern = args.get("pattern", "*")

        if not path:
            return "Error: path parameter is required"

        path_error = await self._validate_path(path)
        if path_error:
            return path_error

        try:
            path_obj = Path(path).expanduser()

            if not path_obj.exists():
                return f"Error: Directory not found: {path}"

            if not path_obj.is_dir():
                return f"Error: Not a directory: {path}"

            items = list(path_obj.glob(pattern))
            items.sort()

            if not items:
                return f"No items found matching pattern '{pattern}' in {path}"

            result = f"Contents of {path_obj} (pattern: {pattern}):\n\n"

            for item in items:
                if item.is_dir():
                    result += f"  [DIR]  {item.name}/\n"
                else:
                    size = item.stat().st_size
                    result += f"  [FILE] {item.name} ({size} bytes)\n"

            return result

        except Exception as e:
            return f"Error listing directory: {str(e)}"

    def _is_command_prefix_allowed(self, full_command: str) -> list[str]:
        """Check if a command matches any allowlist prefix.

        For each pipeline segment, extracts tokens (stripping env vars and
        transparent wrappers), then checks if any allowlist entry matches
        as a prefix of the segment's tokens. Shell builtins are always allowed.

        Returns an empty list if all commands are allowed, or a list of
        rejected command names.
        """
        stripped = _strip_shell_comments(full_command)
        segments = _split_shell_commands(stripped)
        rejected: list[str] = []

        for segment in segments:
            try:
                tokens = shlex.split(segment)
            except ValueError:
                tokens = segment.split()

            if not tokens:
                continue

            # Skip env var assignments
            idx = 0
            while idx < len(tokens) and ENV_VAR_PATTERN.match(tokens[idx]):
                idx += 1

            if idx >= len(tokens):
                continue

            # Strip transparent wrappers and their arguments
            while idx < len(tokens) and _resolve_command_token(tokens[idx]) in TRANSPARENT_WRAPPERS:
                idx += 1
                idx = _skip_wrapper_args(tokens, idx)

            if idx >= len(tokens):
                continue

            cmd_name = _resolve_command_token(tokens[idx])
            if cmd_name in SHELL_BUILTINS:
                continue

            # Build the effective token list for matching (preserve full paths)
            effective: list[str] = []
            for t in tokens[idx:]:
                if effective:
                    effective.append(t)
                else:
                    effective.append(_resolve_command_token(t))

            # Check if any allowlist entry is a prefix of the effective tokens
            matched = False
            for entry in self.allowed_commands:
                entry_parts = entry.split()
                if effective[: len(entry_parts)] == entry_parts:
                    matched = True
                    break

            if not matched:
                rejected.append(cmd_name)

        return rejected

    async def _check_command_allowed(self, cmd_names: list[str], full_command: str) -> str | None:
        """Check if all commands in a command line are allowed.

        Uses prefix-based matching: each allowlist entry is matched as a
        prefix of the command's tokens. Shell builtins are always allowed.

        Returns:
            Error message if blocked, None if allowed
        """
        rejected = self._is_command_prefix_allowed(full_command)
        if not rejected:
            return None

        if self.confirmation_callback is not None:
            approved = await self.confirmation_callback(full_command)
            if approved:
                return None
            return f"Error: Command '{full_command}' was rejected by the user. Try a different approach or use only allowed commands: {', '.join(self.allowed_commands)}."

        allowed_list = ", ".join(self.allowed_commands)
        return (
            f"Error: Command '{', '.join(rejected)}' is not in the allowed commands list: {allowed_list}. Not allowed."
        )

    async def _validate_command_arguments(self, command: str, was_auto_allowed: bool) -> str | None:
        """Validate path arguments and parameters for specific commands.

        Checks:
        - cd, find: paths must be within allowed directories
        - python/python3: script paths must be within allowed directories;
          -c (inline code), stdin, and interactive mode require confirmation
          when the command was auto-allowed via allowlist

        Args:
            command: The full shell command string
            was_auto_allowed: True if all commands passed the allowlist/builtin
                check without user confirmation (skips python -c confirmation
                to avoid double-prompting)

        Returns:
            Error message if blocked, None if allowed
        """
        pairs = _extract_command_argument_paths(command)

        for cmd_name, path_or_marker in pairs:
            if path_or_marker in ("<inline-code>", "<stdin>", "<interactive>"):
                if not was_auto_allowed:
                    continue
                if self.confirmation_callback is not None:
                    desc = f"{cmd_name} {path_or_marker.strip('<>')} execution: {command}"
                    approved = await self.confirmation_callback(desc)
                    if not approved:
                        return f"Error: {cmd_name} {path_or_marker} execution rejected by user."
                elif path_or_marker == "<interactive>":
                    # Interactive REPL (no args) is never allowed in task mode
                    # as it would hang waiting for input.
                    return f"Error: {cmd_name} {path_or_marker} execution is not allowed in non-interactive mode."
                elif cmd_name not in self.allowed_commands:
                    # In non-interactive mode, only block inline-code/stdin if the
                    # command is not in the user's explicit allowlist. If python3
                    # is allowlisted, python3 -c should also be trusted (e.g.
                    # for piped commands in scheduled tasks).
                    return f"Error: {cmd_name} {path_or_marker} execution is not allowed in non-interactive mode."
            else:
                path_error = await self._validate_path(path_or_marker)
                if path_error:
                    return f"Error: {cmd_name} target path is not allowed. {path_error}"

        return None

    async def _write_file(self, args: dict) -> str:
        """Write content to a file"""
        path = args.get("path")
        content = args.get("content")

        if not path:
            return "Error: path parameter is required"
        if content is None:
            return "Error: content parameter is required"

        if self._is_protected_config(path):
            return f"Error: {Path(path).name} is a protected security config file and cannot be modified by the agent."

        path_error = await self._validate_path(path)
        if path_error:
            return path_error

        confirm_error = await self._check_confirmation("internal__write_file", f"Write file: {path}")
        if confirm_error:
            return confirm_error

        try:
            path_obj = Path(path).expanduser()
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            path_obj.write_text(content)
            return f"File written: {path_obj} ({len(content)} chars)"
        except Exception as e:
            return f"Error writing file: {e}"

    async def _edit_file(self, args: dict) -> str:
        """Edit a file by replacing an exact string match"""
        for param in ("path", "old_string", "new_string"):
            if not args.get(param) and args.get(param) != "":
                return f"Error: {param} parameter is required"

        path: str = args["path"]
        old_string: str = args["old_string"]
        new_string: str = args["new_string"]

        if old_string == new_string:
            return "Error: old_string and new_string are identical"

        if self._is_protected_config(path):
            return f"Error: {Path(path).name} is a protected security config file and cannot be modified by the agent."

        for check in (
            await self._validate_path(path),
            await self._check_confirmation("internal__edit_file", f"Edit file: {path}"),
        ):
            if check:
                return check

        try:
            path_obj = Path(path).expanduser()

            if not path_obj.is_file():
                kind = "File not found" if not path_obj.exists() else "Not a file"
                return f"Error: {kind}: {path}"

            content = path_obj.read_text()
            count = content.count(old_string)

            if count != 1:
                if count == 0:
                    return "Error: old_string not found in file"
                return f"Error: old_string appears {count} times in file (must be unique). Provide more surrounding context to make it unique."

            new_content = content.replace(old_string, new_string, 1)
            path_obj.write_text(new_content)
            return f"File edited: {path_obj} (replaced 1 occurrence)"
        except Exception as e:
            return f"Error editing file: {e}"

    async def _execute_command(self, args: dict) -> str:
        """Execute a command with allowlist enforcement"""
        command = args.get("command")
        working_dir = args.get("working_directory")
        timeout = args.get("timeout", 30)

        if not command:
            return "Error: command parameter is required"

        # Block commands that target protected security config files
        config_dir = str(get_config_dir().resolve())
        for protected in PROTECTED_CONFIG_FILES:
            protected_path = f"{config_dir}/{protected}"
            if protected_path in command or f"~/.ai-assist/{protected}" in command:
                return f"Error: {protected} is a protected security config file and cannot be modified by the agent."

        # In interactive mode (confirmation_callback set), no timeout -- the user
        # can press Escape to interrupt like any normal interaction.
        # In non-interactive mode, enforce a timeout to prevent runaway commands.
        if self.confirmation_callback is not None:
            timeout = None
        else:
            timeout = min(timeout, 300)  # Max 5 minutes

        cmd_names = extract_command_names(command)

        # Determine if all commands are auto-allowed (builtins or prefix-matched allowlist)
        was_auto_allowed = not self._is_command_prefix_allowed(command)

        # Commands authorized by AWL scripts are treated as user-confirmed
        # (skip extra confirmation for python -c, etc.)
        awl_confirmed = any(name in self.awl_authorized_commands for name in cmd_names)

        error = await self._check_command_allowed(cmd_names, command)
        if error:
            return error

        # Validate command arguments (paths for cd/find, parameters for python)
        # AWL-authorized commands skip the extra inline-code confirmation
        arg_error = await self._validate_command_arguments(command, was_auto_allowed and not awl_confirmed)
        if arg_error:
            return arg_error

        try:
            cwd = None
            if working_dir:
                cwd = Path(working_dir).expanduser()
                if not cwd.exists() or not cwd.is_dir():
                    return f"Error: Invalid working directory: {working_dir}"

            # shell=True is required for shell command execution with pipes, redirects, etc.
            # Commands are validated through allowlists and confirmation callbacks above
            result = subprocess.run(  # nosec B602
                command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
            )

            output = f"Command: {command}\n"
            if working_dir:
                output += f"Working directory: {cwd}\n"
            output += f"Exit code: {result.returncode}\n\n"

            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"

            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"

            if result.returncode != 0:
                output += f"\nCommand failed with exit code {result.returncode}"

            return output

        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds. Use interactive mode for long-running commands."
        except Exception as e:
            return f"Error executing command: {str(e)}"
