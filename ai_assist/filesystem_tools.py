"""Filesystem tools for ai-assist agent"""

import glob
import json
import logging
import os
import re
import shlex
import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .config import AiAssistConfig, get_config_dir
from .redaction import REDACT_CONFIG_FILE, filter_sensitive_env
from .security import MCP_TOOL_DEFINITIONS_FILE

logger = logging.getLogger(__name__)

ALLOWED_COMMANDS_FILE = "allowed_commands.json"
ALLOWED_PATHS_FILE = "allowed_paths.json"
# Sensitive env var names execute_command may pass through (e.g. ["GITHUB_TOKEN"])
COMMAND_ENV_FILE = "command_env.json"

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

# The general command parser recognizes all shell builtins.  Auto mode is more
# restrictive: wrappers such as eval, source, command, and exec can bypass
# command classification, so only inert builtins are safe to auto-approve.
AUTO_SAFE_SHELL_BUILTINS = frozenset({"cd", "echo", "false", "pwd", "printf", "test", "true", "["})

# python, python3, python3.12, ...
_PYTHON_NAME = re.compile(r"^python(\d+(\.\d+)*)?$")


def _python_invocation(args: list[str]) -> str | None:
    """What a python command line runs: a script path, or a marker.

    Markers: <inline-code> (-c), <module:NAME> (-m), <stdin> (-), and
    <interactive> (no arguments). Handles combined short flags (-Ic, -Im) and
    the options taking a value (-W, -X).
    """
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "-":
            return "<stdin>"
        if not arg.startswith("-"):
            return f"<script:{arg}>"
        if not arg.startswith("--"):
            for j, flag in enumerate(arg[1:], start=1):
                rest = arg[j + 1 :]
                if flag in "cm":
                    value = rest or (args[i + 1] if i + 1 < len(args) else "")
                    return "<inline-code>" if flag == "c" else f"<module:{value}>"
                if flag in "WX":
                    if not rest:
                        i += 1  # the value is the next argument
                    break
        i += 1
    return None if args else "<interactive>"


# find primaries that run arbitrary commands or write/delete files. These
# bypass the command allowlist (the executed program is find's argument, not a
# separate allowlisted command), so they must be rejected explicitly.
FIND_DANGEROUS_PRIMARIES = frozenset(
    {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprintf", "-fprint", "-fprint0", "-fls"}
)

TRANSPARENT_WRAPPERS = frozenset(
    {
        "env",
        "nohup",
        "nice",
        "ionice",
        "timeout",
        "stdbuf",
        "script",
        "xargs",
        # Builtins and keywords that run the command following them
        "command",
        "builtin",
        "exec",
        "time",
        "if",
        "then",
        "else",
        "elif",
        "do",
        "while",
        "until",
        "!",
        "{",
    }
)

# Builtins that run code given as arguments or redirect later command lookups.
# They need an allowlist entry or user confirmation like any external command.
CODE_RUNNING_SHELL_BUILTINS = frozenset(
    {"eval", "source", ".", "trap", "alias", "enable", "hash", "mapfile", "readarray", "coproc"}
)

# Builtins that run nothing by themselves: skipped by the allowlist check
INERT_SHELL_BUILTINS = SHELL_BUILTINS - CODE_RUNNING_SHELL_BUILTINS - TRANSPARENT_WRAPPERS

# Per-wrapper flags whose argument is a separate, possibly non-numeric token
# (e.g. `xargs -I {} grep {}`), which the generic numeric-value rule would miss.
# These are matched by exact flag token; combined forms (`-I{}`, `--replace=X`) are
# already a single token handled as a plain flag.
_WRAPPER_VALUE_FLAGS: dict[str, frozenset[str]] = {
    "xargs": frozenset({"-I", "-E", "-d", "-a", "--replace", "--delimiter", "--arg-file", "--eof"}),
}

# Shells whose real work is in their arguments: `bash -c "<payload>"` runs the
# payload and `bash <script>` runs the script, so validation must look at those.
SHELL_INTERPRETERS = frozenset({"bash", "sh", "zsh", "dash", "ksh"})
# Shell options taking a separate value (`bash -o pipefail -c ...`)
_SHELL_VALUE_FLAGS = frozenset({"-o", "+o", "-O", "+O", "--rcfile", "--init-file"})
# Marker for a shell reading its commands from stdin: never allowlisted
SHELL_STDIN_MARKER = "<shell-stdin>"
_MAX_SHELL_NESTING = 5

PRIVILEGE_WRAPPERS = frozenset({"sudo", "su", "doas"})

PROTECTED_CONFIG_FILES = frozenset(
    {
        ALLOWED_COMMANDS_FILE,
        ALLOWED_PATHS_FILE,
        "skill_env.json",
        # Files whose contents get executed or trusted at load time
        "mcp_servers.yaml",
        ".env",
        "schedules.json",
        "event-schedules.json",
        MCP_TOOL_DEFINITIONS_FILE,
        "installed-plugins.json",
        "installed-skills.json",
        "plugin-marketplaces.json",
        REDACT_CONFIG_FILE,
        COMMAND_ENV_FILE,
    }
)

ENV_VAR_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

_SHELL_REDIRECT = re.compile(r"^(\d*[<>]|&>|>>|>&|<&|\d+>&)")
AUTO_TEMP_ROOT = Path("/tmp/ai-assist")  # nosec B108 - configured ai-assist workspace


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


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


def _skip_wrapper_args(tokens: list[str], idx: int, wrapper: str | None = None) -> int:
    """Skip flags and positional args belonging to a transparent wrapper.

    After the wrapper name has been consumed, this skips:
    - env var assignments (KEY=value)
    - flags (-x, --long) and their non-flag argument
    - bare numeric tokens (e.g. timeout's duration, nice's adjustment)
    - the separate argument of a wrapper's value-taking flags (e.g. `xargs -I {}`)
    Stops at the first token that looks like a command name.
    """
    value_flags = _WRAPPER_VALUE_FLAGS.get(wrapper or "", frozenset())
    while idx < len(tokens):
        tok = tokens[idx]
        if ENV_VAR_PATTERN.match(tok):
            idx += 1
        elif tok.startswith("-"):
            idx += 1
            if tok in value_flags:
                # This flag takes a separate argument (possibly non-numeric); consume it.
                if idx < len(tokens):
                    idx += 1
            elif idx < len(tokens) and not tokens[idx].startswith("-") and not ENV_VAR_PATTERN.match(tokens[idx]):
                # Could be a flag value (e.g. -n 10) or the real command.
                # If it's numeric, consume it as a flag value.
                if _NUMERIC_ARG.match(tokens[idx]):
                    idx += 1
        elif _NUMERIC_ARG.match(tok):
            idx += 1
        else:
            break
    return idx


def _strip_transparent_wrappers(tokens: list[str], idx: int) -> int:
    """Skip leading transparent wrappers (env, timeout, xargs, ...) and their flags,
    returning the index of the actual command they invoke.

    A wrapper's invoked command is still validated by the caller — so `xargs grep`
    is allowed (grep is), while `xargs rm` is rejected (rm is not).
    """
    while idx < len(tokens):
        wrapper = _resolve_command_token(tokens[idx])
        if wrapper not in TRANSPARENT_WRAPPERS:
            break
        idx += 1
        idx = _skip_wrapper_args(tokens, idx, wrapper)
    return idx


# Stands in for $(...), `...`, <(...) and >(...) in split segments: the
# substituted commands are validated as separate segments, and the "$" marks
# the value as unknown (e.g. for redirect targets).
SUBSTITUTION_PLACEHOLDER = "$__substitution__"
_HEREDOC_WORD_END = frozenset(" \t\n;|&<>()")


def _find_closing_paren(command: str, start: int) -> int:
    """Index of the ')' matching an already-open '(' (len(command) if none)."""
    depth = 1
    in_single = in_double = False
    i = start
    while i < len(command):
        c = command[i]
        if in_single:
            in_single = c != "'"
        elif c == "\\":
            i += 1
        elif c == '"':
            in_double = not in_double
        elif not in_double and c == "'":
            in_single = True
        elif not in_double and c == "(":
            depth += 1
        elif not in_double and c == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(command)


def _find_backtick_end(command: str, start: int) -> int:
    """Index of the closing backtick (len(command) if none)."""
    i = start
    while i < len(command):
        if command[i] == "\\":
            i += 2
            continue
        if command[i] == "`":
            return i
        i += 1
    return len(command)


def _substitution_at(command: str, i: int, process: bool = False) -> tuple[list[str], int] | None:
    """If a $(...), $((...)) or `...` starts at i, return (bodies, end index).

    With process=True, <(...) and >(...) process substitutions are matched too.
    """
    if process and command[i] in "<>" and command.startswith("(", i + 1):
        end = _find_closing_paren(command, i + 2)
        return [command[i + 2 : end]], end + 1
    if command[i] == "`":
        end = _find_backtick_end(command, i + 1)
        return [command[i + 1 : end]], end + 1
    if command.startswith("$((", i):
        end = _find_closing_paren(command, i + 2)
        # Arithmetic itself runs nothing, but substitutions inside it do
        return _scan_shell(command[i + 2 : end])[1], end + 1
    if command.startswith("$(", i):
        end = _find_closing_paren(command, i + 2)
        return [command[i + 2 : end]], end + 1
    return None


def _heredoc_body_substitutions(body: str) -> list[str]:
    """Substitutions in an unquoted heredoc body (quotes are literal there)."""
    bodies: list[str] = []
    i = 0
    while i < len(body):
        if body[i] == "\\":
            i += 2
            continue
        found = _substitution_at(body, i)
        if found:
            bodies.extend(found[0])
            i = found[1]
        else:
            i += 1
    return bodies


def _skip_heredoc_bodies(command: str, i: int, heredocs: list[tuple[str, bool, bool]], bodies: list[str]) -> int:
    """Skip the bodies of pending heredocs starting at line index i.

    Substitutions in bodies that expand are appended to bodies. Returns the
    index of the first line after the last body.
    """
    for delimiter, expands, strip_tabs in heredocs:
        body: list[str] = []
        while i < len(command):
            newline = command.find("\n", i)
            line_end = len(command) if newline == -1 else newline
            line = command[i:line_end]
            i = line_end + 1
            if (line.lstrip("\t") if strip_tabs else line) == delimiter:
                break
            body.append(line)
        if expands:
            bodies.extend(_heredoc_body_substitutions("\n".join(body)))
    return i


def _parse_heredoc_operator(command: str, i: int) -> tuple[int, str, bool, bool]:
    """Parse `<<[-]WORD` at i. Returns (end, delimiter, body_expands, strip_tabs)."""
    j = i + 2
    strip_tabs = j < len(command) and command[j] == "-"
    if strip_tabs:
        j += 1
    while j < len(command) and command[j] in " \t":
        j += 1
    word_start = j
    in_quote = ""
    while j < len(command) and (in_quote or command[j] not in _HEREDOC_WORD_END):
        if in_quote:
            if command[j] == in_quote:
                in_quote = ""
        elif command[j] in "'\"":
            in_quote = command[j]
        j += 1
    word = command[word_start:j]
    quoted = any(q in word for q in "'\"\\")
    delimiter = word.replace("'", "").replace('"', "").replace("\\", "")
    return j, delimiter, not quoted, strip_tabs


def _scan_shell(command: str) -> tuple[list[str], list[str]]:
    """Split a command line into simple-command segments.

    Splits on ;, &&, ||, |, &, parentheses and newlines outside quotes; drops comments and
    heredoc bodies; joins backslash-newline continuations. Command
    substitutions are replaced by SUBSTITUTION_PLACEHOLDER and their bodies
    returned separately (including those inside double quotes, $((...)) and
    unquoted heredoc bodies), as (segments, substitution_bodies).
    """
    segments: list[str] = []
    bodies: list[str] = []
    current: list[str] = []
    heredocs: list[tuple[str, bool, bool]] = []
    in_single = in_double = False
    # Inside a case statement, text up to the next ")" after "in" or ";;" is a
    # pattern, not a command (outside a case, ";;" is a syntax error)
    in_case_pattern = False
    n = len(command)
    i = 0

    def flush() -> None:
        nonlocal in_case_pattern
        segment = "".join(current).strip()
        current.clear()
        if segment:
            segments.append(segment)
            words = segment.split()
            in_case_pattern = words[0] == "case" and words[-1] == "in"

    while i < n:
        c = command[i]
        if in_single:
            in_single = c != "'"
            current.append(c)
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            if command[i + 1] != "\n":
                current.append(command[i : i + 2])
            i += 2
            continue
        found = _substitution_at(command, i, process=not in_double)
        if found:
            bodies.extend(found[0])
            current.append(SUBSTITUTION_PLACEHOLDER)
            i = found[1]
            continue
        if in_double:
            in_double = c != '"'
            current.append(c)
            i += 1
            continue
        nxt = command[i + 1] if i + 1 < n else ""
        if c == "'":
            in_single = True
            current.append(c)
        elif c == '"':
            in_double = True
            current.append(c)
        elif c == "#" and (not current or current[-1][-1].isspace()):
            newline = command.find("\n", i)
            i = n if newline == -1 else newline
            continue
        elif command.startswith("<<", i) and not command.startswith("<<<", i):
            end, delimiter, expands, strip_tabs = _parse_heredoc_operator(command, i)
            heredocs.append((delimiter, expands, strip_tabs))
            current.append(command[i:end])
            i = end
            continue
        elif c == "\n":
            flush()
            i = _skip_heredoc_bodies(command, i + 1, heredocs, bodies)
            heredocs.clear()
            continue
        elif c == "(" and nxt == "(":
            # Arithmetic command ((...)): runs nothing, but substitutions inside it do
            end = _find_closing_paren(command, i + 2)
            bodies.extend(_scan_shell(command[i + 2 : end])[1])
            flush()
            i = end + 2
            continue
        elif c == ")" and in_case_pattern:
            current.clear()
            in_case_pattern = False
        elif c == "|" and in_case_pattern:
            current.append(c)
        elif c == ";" and nxt in ";&" and nxt:
            # ;; ;& ;;& end a case branch: a pattern or esac follows
            flush()
            in_case_pattern = True
            i += 3 if command.startswith(";;&", i) else 2
            continue
        elif c in ";()":
            # Parentheses end segments too: subshells and case patterns
            flush()
        elif c == "|":
            flush()
            if nxt in "|&" and nxt:
                i += 1
        elif c == "&":
            if nxt == "&":
                flush()
                i += 1
            elif nxt == ">" or (current and current[-1][-1] in "<>"):
                current.append(c)
            else:
                flush()
        else:
            current.append(c)
        i += 1
    flush()
    return segments, bodies


def _split_shell_commands(command: str) -> list[str]:
    """Split a command line into segments, including substituted commands.

    See _scan_shell. Bodies of command substitutions are split recursively
    and appended, so every command that would run is a segment.
    """
    segments, bodies = _scan_shell(command)
    for body in bodies:
        segments.extend(_split_shell_commands(body))
    return segments


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


def _expand_shell_invocations(command: str, depth: int = 0) -> str:
    """Rewrite shell invocations into the commands they actually run, for validation.

    - `bash -c "<payload>"` (also `-lc`, `-ec`, `-o pipefail -c`) becomes the payload,
      expanded recursively
    - `bash <script> args` becomes `<script> args`, so the script must be allowed
    - a shell reading commands from stdin (`... | bash`) becomes SHELL_STDIN_MARKER

    Execution is unaffected; only allowlist and builtin checks see the rewrite.
    """
    if depth > _MAX_SHELL_NESTING:
        return SHELL_STDIN_MARKER
    segments = _split_shell_commands(command)
    expanded: list[str] = []
    changed = False
    for segment in segments:
        try:
            tokens = shlex.split(segment)
        except ValueError:
            expanded.append(segment)
            continue
        idx = 0
        while idx < len(tokens) and ENV_VAR_PATTERN.match(tokens[idx]):
            idx += 1
        while idx < len(tokens) and _resolve_command_token(tokens[idx]) in TRANSPARENT_WRAPPERS:
            idx += 1
            idx = _skip_wrapper_args(tokens, idx)
        if idx >= len(tokens) or Path(tokens[idx]).name not in SHELL_INTERPRETERS:
            expanded.append(segment)
            continue
        changed = True
        args = tokens[idx + 1 :]
        replacement = SHELL_STDIN_MARKER
        i = 0
        while i < len(args):
            arg = args[i]
            if arg in _SHELL_VALUE_FLAGS:
                i += 2
                continue
            if arg.startswith("-") and not arg.startswith("--") and "c" in arg[1:]:
                if i + 1 < len(args):
                    replacement = _expand_shell_invocations(args[i + 1], depth + 1)
                break
            if arg.startswith(("-", "+")):
                i += 1
                continue
            replacement = shlex.join(args[i:])
            break
        expanded.append(replacement)
    return " ; ".join(expanded) if changed else command


def extract_command_names(command: str) -> list[str]:
    """Extract all command names from a shell command string.

    Handles comments, env var prefixes, compound commands, pipes,
    and full paths. Detects unsafe env var assignments containing
    command substitution. Looks inside `bash -c` payloads and
    `bash <script>` invocations (see _expand_shell_invocations).
    """
    segments = _split_shell_commands(_expand_shell_invocations(command))
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
        idx = _strip_transparent_wrappers(tokens, idx)

        if idx >= len(tokens):
            continue

        cmd_token = tokens[idx].lstrip("(").rstrip(")")
        if not cmd_token or cmd_token.startswith("-"):
            continue
        cmd_name = _resolve_command_token(cmd_token)
        commands.append(cmd_name)

    return commands


def _compute_segment_prefix(tokens: list[str]) -> str | None:
    """Compute an allowlist prefix from a single command segment's tokens.

    Returns None if the segment is empty, only builtins, or only wrappers.
    """
    if not tokens:
        return None

    idx = 0
    while idx < len(tokens) and ENV_VAR_PATTERN.match(tokens[idx]):
        idx += 1

    if idx >= len(tokens):
        return None

    idx = _strip_transparent_wrappers(tokens, idx)

    if idx >= len(tokens):
        return None

    prefix_parts: list[str] = []

    cmd_token = _resolve_command_token(tokens[idx])
    if cmd_token in PRIVILEGE_WRAPPERS:
        prefix_parts.append(cmd_token)
        idx += 1
        while idx < len(tokens) and tokens[idx].startswith("-"):
            idx += 1
            if idx < len(tokens) and not tokens[idx].startswith("-"):
                idx += 1

    if idx >= len(tokens):
        return " ".join(prefix_parts) if prefix_parts else None

    cmd_name = _resolve_command_token(tokens[idx])
    if cmd_name in INERT_SHELL_BUILTINS:
        return None
    prefix_parts.append(cmd_name)
    idx += 1

    saw_inline_flag = False
    while idx < len(tokens):
        token = tokens[idx]
        if token == "-m" and _PYTHON_NAME.match(Path(cmd_name).name) and idx + 1 < len(tokens):
            prefix_parts.extend(["-m", tokens[idx + 1]])
            break
        if _SHELL_REDIRECT.match(token):
            idx += 1
            if idx < len(tokens):
                idx += 1
            continue
        if token.startswith("-"):
            if token in ("-c", "-e"):
                saw_inline_flag = True
            idx += 1
        else:
            if not saw_inline_flag:
                prefix_parts.append(token)
            break

    return " ".join(prefix_parts)


def compute_allowlist_prefixes(command: str) -> list[str]:
    """Compute allowlist prefixes for all non-builtin segments of a command.

    For compound commands (&&, ||, ;, |), returns a prefix for each segment
    that contains a non-builtin command. Builtin-only segments (cd, echo, etc.)
    are skipped.
    """
    prefixes: list[str] = []
    for segment in _split_shell_commands(command):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()
        prefix = _compute_segment_prefix(tokens)
        if prefix:
            prefixes.append(prefix)
    return prefixes


def compute_allowlist_prefix(command: str) -> str | None:
    """Compute a meaningful allowlist prefix from a command string.

    For compound commands, returns the prefix of the first non-builtin segment.
    Returns None if only builtins/wrappers.
    """
    prefixes = compute_allowlist_prefixes(command)
    return prefixes[0] if prefixes else None


# Redirection operators that write to their target
_WRITE_REDIRECTS = frozenset({">", ">>", ">|", "&>", "&>>", "<>", ">&"})
# Redirect targets that are always fine to write
_HARMLESS_REDIRECT_TARGETS = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty"})
# Marks a path the shell computes at run time ($VAR, globs, substitutions)
_DYNAMIC_PATH_CHARS = frozenset("$`*?[")
# Env vars whose value changes which program (or code) a command runs
PROGRAM_HIJACKING_ENV_VARS = frozenset(
    {
        "PATH",
        "BASH_ENV",
        "ENV",
        "IFS",
        "PROMPT_COMMAND",
        "SHELLOPTS",
        "BASHOPTS",
        "PS4",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_EXTERNAL_DIFF",
        "GIT_PAGER",
        "GIT_EDITOR",
        "GIT_ASKPASS",
        "SSH_ASKPASS",
        "PAGER",
        "EDITOR",
        "VISUAL",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONHOME",
        "NODE_OPTIONS",
        "PERL5OPT",
        "PERL5LIB",
        "RUBYOPT",
    }
)
_PROGRAM_HIJACKING_ENV_PREFIXES = ("LD_", "DYLD_", "GIT_CONFIG")
_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\+?=")
_ASSIGNING_BUILTINS = frozenset({"export", "declare", "typeset", "readonly", "local"})


def _program_hijacking_assignments(command: str) -> set[str]:
    """Names of PROGRAM_HIJACKING_ENV_VARS a command line assigns."""
    names: set[str] = set()
    for segment in _split_shell_commands(command):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()
        idx = 0
        while idx < len(tokens) and _ASSIGNMENT.match(tokens[idx]):
            idx += 1
        idx = _strip_transparent_wrappers(tokens, idx)
        candidates = tokens[:idx]
        if idx < len(tokens) and tokens[idx] in _ASSIGNING_BUILTINS:
            candidates += tokens[idx + 1 :]
        for token in candidates:
            match = _ASSIGNMENT.match(token)
            if match and (
                match.group(1) in PROGRAM_HIJACKING_ENV_VARS
                or match.group(1).startswith(_PROGRAM_HIJACKING_ENV_PREFIXES)
            ):
                names.add(match.group(1))
    return names


# Credential and key locations, matched under any home (yours, /root, other users)
_CREDENTIAL_LOCATIONS = (
    (".ssh",),
    (".gnupg",),
    (".aws",),
    (".azure",),
    (".config", "gcloud"),
    (".config", "gh"),
    (".kube",),
    (".docker", "config.json"),
    (".netrc",),
    (".git-credentials",),
    (".pgpass",),
    (".password-store",),
    (".local", "share", "keyrings"),
)


def sensitive_locations() -> list[Path]:
    """ai-assist's own secrets; see also _CREDENTIAL_LOCATIONS, matched anywhere."""
    config_dir = get_config_dir()
    return [config_dir / ".env", config_dir / "mcp-oauth"]


def startup_locations() -> list[Path]:
    """Places whose contents run later, outside ai-assist (writes need confirmation)."""
    home = Path.home()
    rc_files = [
        ".bashrc",
        ".bash_profile",
        ".bash_login",
        ".bash_logout",
        ".profile",
        ".zshrc",
        ".zshenv",
        ".zprofile",
        ".zlogin",
        ".xprofile",
        ".xinitrc",
        ".xsession",
        ".gitconfig",
    ]
    directories = [
        ".local/bin",
        "bin",
        ".local/lib",
        ".config/fish",
        ".config/systemd/user",
        ".config/autostart",
        ".config/environment.d",
        ".config/git",
    ]
    return [home / name for name in rc_files + directories]


def is_startup_path(path: Path) -> bool:
    resolved = path.expanduser().resolve()
    return any(_is_relative_to(resolved, location.resolve()) for location in startup_locations())


# Kernel / virtual filesystems: writing them controls the host
# (/proc/sys/kernel/core_pattern, /proc/sysrq-trigger, /sys/kernel/uevent_helper,
# /dev/mem, raw block devices). Reads are fine; writes are never allowed.
_VIRTUAL_FS_ROOTS = frozenset({"proc", "sys", "dev"})


def is_unwritable_system_path(path: Path) -> bool:
    resolved = path.expanduser().resolve()
    parts = resolved.parts
    if len(parts) < 2 or parts[1] not in _VIRTUAL_FS_ROOTS:
        return False
    # The usual harmless device targets stay writable
    if str(resolved) in _HARMLESS_REDIRECT_TARGETS or str(resolved).startswith("/dev/fd/"):
        return False
    return True


def is_sensitive_path(path: Path) -> bool:
    """Whether a path is (inside) a credential location the agent may never access."""
    resolved = path.expanduser().resolve()
    parts = resolved.parts
    for location in _CREDENTIAL_LOCATIONS:
        size = len(location)
        if any(parts[i : i + size] == location for i in range(len(parts) - size + 1)):
            return True
    return any(_is_relative_to(resolved, location.resolve()) for location in sensitive_locations())


def _command_path_candidates(command: str, cwd: str | None = None) -> list[Path]:
    """Every word of a command line interpreted as a path (also NAME=value values).

    $VARS, ~ and globs are expanded as the shell would, and relative paths are
    resolved against cwd and earlier cd targets.
    """
    candidates: list[Path] = []
    effective_cwd = Path(cwd).expanduser() if cwd else Path.cwd()
    for segment in _split_shell_commands(command):
        lexer = shlex.shlex(segment, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        try:
            tokens = list(lexer)
        except ValueError:
            tokens = segment.split()
        for token in tokens:
            for value in (token, token.split("=", 1)[1] if "=" in token else ""):
                if not value:
                    continue
                expanded = os.path.expanduser(os.path.expandvars(value))
                if any(c in expanded for c in "*?["):
                    candidates.extend(effective_cwd / match for match in glob.glob(expanded, root_dir=effective_cwd))
                else:
                    candidates.append(effective_cwd / expanded)
        if len(tokens) > 1 and tokens[0] == "cd":
            effective_cwd = effective_cwd / os.path.expanduser(os.path.expandvars(tokens[1]))
    return candidates


# git options that write files or run configured programs, even for read commands
_GIT_UNSAFE_READ_OPTIONS = frozenset({"--output", "--ext-diff", "--textconv"})
# git branch options that only list (and those taking a value)
_GIT_BRANCH_LIST_OPTIONS = frozenset(
    {
        "-a",
        "--all",
        "-r",
        "--remotes",
        "-v",
        "-vv",
        "--verbose",
        "-l",
        "--list",
        "--show-current",
        "--no-color",
        "--color",
        "--column",
        "--no-column",
        "--sort",
        "--format",
        "--contains",
        "--no-contains",
        "--merged",
        "--no-merged",
        "--points-at",
        "--abbrev",
        "--no-abbrev",
        "-i",
        "--ignore-case",
    }
)
_GIT_BRANCH_FILTER_OPTIONS = frozenset({"--contains", "--no-contains", "--merged", "--no-merged", "--points-at"})
_GH_API_BODY_OPTIONS = frozenset({"-f", "-F", "--field", "--raw-field", "--input"})
# git global options and ssh options that make the command run another program
_GIT_PROGRAM_OPTIONS = frozenset({"-c", "--config-env", "--exec-path"})
_SSH_PROGRAM_OPTIONS = frozenset({"proxycommand", "localcommand", "permitlocalcommand", "knownhostscommand"})
# git config forms that only read
_GIT_CONFIG_READ_OPTIONS = frozenset(
    {"--get", "--get-all", "--get-regexp", "--list", "-l", "--show-origin", "get", "list"}
)


def _program_running_arguments(command: str) -> list[str]:
    """Describe git/ssh arguments in a command line that make it run other programs."""
    found: list[str] = []
    for segment in _split_shell_commands(command):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()
        idx = 0
        while idx < len(tokens) and _ASSIGNMENT.match(tokens[idx]):
            idx += 1
        idx = _strip_transparent_wrappers(tokens, idx)
        if idx >= len(tokens):
            continue
        name, args = Path(tokens[idx]).name, tokens[idx + 1 :]
        if name == "git":
            i = 0
            while i < len(args) and args[i].startswith("-"):
                if args[i].split("=", 1)[0] in _GIT_PROGRAM_OPTIONS:
                    found.append(f"git {args[i]}")
                i += 2 if args[i] in ("-c", "-C", "--git-dir", "--work-tree", "--namespace") else 1
            rest = args[i:]
            if rest[:1] == ["config"]:
                values = [arg for arg in rest[1:] if not arg.startswith("-")]
                if not _GIT_CONFIG_READ_OPTIONS.intersection(rest[1:]) and len(values) >= 2:
                    found.append("git config (set)")
        elif name == "crontab":
            # Anything but listing installs or removes scheduled jobs
            if not args or any(arg != "-l" and not arg.startswith("-u") for arg in args):
                found.append("crontab (install/remove jobs)")
        elif name == "ssh":
            for i, arg in enumerate(args):
                if arg.startswith("-F"):
                    found.append("ssh -F")
                if arg == "-o" and i + 1 < len(args):
                    option = args[i + 1]
                elif arg.startswith("-o"):
                    option = arg[2:]
                else:
                    continue
                key = re.split(r"[=\s]", option, maxsplit=1)[0].lower()
                if key in _SSH_PROGRAM_OPTIONS:
                    found.append(f"ssh -o {key}")
    return found


# Commands that write a file given as an argument (not via redirection)
_FILE_WRITE_COMMANDS = frozenset({"cp", "mv", "install", "tee", "dd", "ln"})

# Interpreters/runners whose arguments are code, a script, or a package to run.
# Allowlisting the bare name = allowlisting arbitrary code, so an invocation that
# runs code needs confirmation unless the full invocation is explicitly allowlisted.
_CODE_INTERPRETERS = frozenset({"awk", "gawk", "mawk", "nawk", "perl", "ruby", "uvx", "uv", "npx", "pipx"})
# argv that only prints info and runs no user code
_BENIGN_INTERPRETER_ARGS = frozenset({"--version", "-V", "--help", "-h", "version", "help"})


def _interpreter_runs_code(name: str, args: list[str]) -> bool:
    """Whether an interpreter/runner invocation executes arbitrary code or a package."""
    if not args or _BENIGN_INTERPRETER_ARGS.intersection(args):
        return False
    if name in ("uv", "pipx"):
        # only the sub-commands that execute a package/script
        return args[0] in ("run", "tool")
    return True


_TAR_FLAG_LETTERS = set("cxtruvzjJafkpPhlmoswC")


def _tar_write_targets(args: list[str]) -> list[str]:
    """Destinations a tar invocation writes: the -f archive on create, the -C dir on extract."""
    create = extract = False
    file_val: str | None = None
    cdir: str | None = None
    expect: str | None = None
    for arg in args:
        if expect:
            file_val, cdir = (arg, cdir) if expect == "f" else (file_val, arg)
            expect = None
            continue
        if arg.startswith("--"):
            if arg.startswith("--file="):
                file_val = arg.split("=", 1)[1]
            elif arg == "--file":
                expect = "f"
            elif arg.startswith("--directory="):
                cdir = arg.split("=", 1)[1]
            elif arg == "--directory":
                expect = "C"
            elif arg == "--create":
                create = True
            elif arg in ("--extract", "--get"):
                extract = True
            continue
        flags = arg[1:] if arg.startswith("-") else arg
        if flags and all(c in _TAR_FLAG_LETTERS for c in flags):
            create = create or "c" in flags
            extract = extract or "x" in flags
            if "f" in flags:
                expect = "f"
            elif "C" in flags:
                expect = "C"
    targets: list[str] = []
    if create and file_val:
        targets.append(file_val)
    if extract:
        targets.append(cdir or ".")
    return targets


def _unzip_write_targets(args: list[str]) -> list[str]:
    """Destination an unzip invocation extracts into (-d dir, else the cwd)."""
    for i, arg in enumerate(args):
        if arg == "-d" and i + 1 < len(args):
            return [args[i + 1]]
        if arg.startswith("-d") and len(arg) > 2:
            return [arg[2:]]
    return ["."]


def _code_interpreter_invocations(command: str) -> list[tuple[str, list[str]]]:
    """Find (name, args) for code-running interpreters in a command line (see _CODE_INTERPRETERS)."""
    found: list[tuple[str, list[str]]] = []
    for segment in _split_shell_commands(command):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()
        idx = 0
        while idx < len(tokens) and _ASSIGNMENT.match(tokens[idx]):
            idx += 1
        idx = _strip_transparent_wrappers(tokens, idx)
        if idx >= len(tokens):
            continue
        name = Path(tokens[idx]).name
        args = tokens[idx + 1 :]
        if name in _CODE_INTERPRETERS and _interpreter_runs_code(name, args):
            found.append((name, args))
    return found


def _file_write_destinations(name: str, args: list[str]) -> list[str]:
    """Destination paths a file-copying command writes (see _FILE_WRITE_COMMANDS)."""
    if name == "dd":
        return [a[3:] for a in args if a.startswith("of=") and len(a) > 3]
    if name == "tee":
        return [a for a in args if not a.startswith("-")]
    # cp, mv, install, ln: -t/--target-directory names the dest dir; else last positional
    positionals: list[str] = []
    target_dir: str | None = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("-t", "--target-directory") and i + 1 < len(args):
            target_dir = args[i + 1]
            i += 2
            continue
        if arg.startswith("--target-directory="):
            target_dir = arg.split("=", 1)[1]
        elif not arg.startswith("-"):
            positionals.append(arg)
        i += 1
    if target_dir is not None:
        return [target_dir]
    if name == "ln":
        return [positionals[-1]] if positionals else []
    return [positionals[-1]] if len(positionals) >= 2 else []


# Pseudo command names under which _extract_command_argument_paths reports written files
_WRITE_TARGET_SOURCES = frozenset({">", "sort", "yq", "write"})
# find options that come before the search paths
_FIND_LEADING_OPTIONS = frozenset({"-H", "-L", "-P"})


def _redirect_write_targets(segment: str) -> list[str]:
    """Files a segment's redirections write to (fd duplications excluded)."""
    lexer = shlex.shlex(segment, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return []
    targets: list[str] = []
    for op, target in zip(tokens, tokens[1:], strict=False):
        if op not in _WRITE_REDIRECTS:
            continue
        if op == ">&" and (target.isdigit() or target == "-"):
            continue
        if target in _HARMLESS_REDIRECT_TARGETS or target.startswith("/dev/fd/"):
            continue
        targets.append(target)
    return targets


def _sort_output_paths(args: list[str]) -> list[str]:
    paths: list[str] = []
    for i, arg in enumerate(args):
        if arg in ("-o", "--output") and i + 1 < len(args):
            paths.append(args[i + 1])
        elif arg.startswith("--output="):
            paths.append(arg.split("=", 1)[1])
        elif arg.startswith("-o") and not arg.startswith("--"):
            paths.append(arg[2:])
    return paths


def _yq_inplace_paths(args: list[str]) -> list[str]:
    inplace = any(
        arg in ("--inplace", "-i") or (arg.startswith("-") and not arg.startswith("--") and "i" in arg[1:])
        for arg in args
    )
    if not inplace:
        return []
    positional = [arg for arg in args if not arg.startswith("-")]
    # First positional is the expression, unless it is the only one
    return positional[1:] if len(positional) > 1 else positional


def _extract_command_argument_paths(command: str, cwd: str | None = None) -> list[tuple[str, str]]:
    """Extract paths from command arguments that need validation.

    Returns list of (command_name, path_or_marker) tuples for commands whose
    path arguments should be checked against allowed directories:
    - cd <dir>: the target directory
    - find <paths...>: the search paths (before option flags)
    - python/python3 <script>: the script file path
    - sort -o <file>, yq -i <files>: files written in place
    - (">", target): files written by output redirections, for every command

    Relative paths are resolved against cwd and earlier cd targets.

    Special markers for python (see _python_invocation):
    - ("<inline-code>") for python -c
    - ("<module:NAME>") for python -m NAME
    - ("<stdin>") for python -
    - ("<interactive>") for python with no arguments

    Shell invocations are expanded first (see _expand_shell_invocations).
    """
    segments = _split_shell_commands(_expand_shell_invocations(command))
    results: list[tuple[str, str]] = []
    # Track cd targets so relative paths in later segments resolve correctly
    effective_cwd: str | None = cwd

    def resolve(path: str) -> str:
        if effective_cwd and not Path(path).expanduser().is_absolute():
            return str(Path(effective_cwd) / path)
        return path

    for segment in segments:
        results.extend((">", resolve(target)) for target in _redirect_write_targets(segment))

        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()

        if not tokens:
            continue

        # Skip env var prefixes and wrappers (env find ..., nice find ...)
        idx = 0
        while idx < len(tokens) and ENV_VAR_PATTERN.match(tokens[idx]):
            idx += 1
        idx = _strip_transparent_wrappers(tokens, idx)

        if idx >= len(tokens):
            continue

        cmd_name = Path(tokens[idx]).name
        args = tokens[idx + 1 :]

        if cmd_name == "cd" and args:
            target = args[0]
            if target != "-":
                results.append(("cd", resolve(target)))
                effective_cwd = resolve(target)

        elif cmd_name == "find":
            if any(arg in FIND_DANGEROUS_PRIMARIES for arg in args):
                results.append(("find", "<dangerous-find>"))
            i = 0
            while i < len(args) and (args[i] in _FIND_LEADING_OPTIONS or args[i].startswith(("-D", "-O"))):
                i += 2 if args[i] == "-D" else 1
            for arg in args[i:]:
                if arg.startswith("-") or arg.startswith("(") or arg == "!":
                    break
                results.append(("find", resolve(arg)))

        elif cmd_name == "sort":
            results.extend(("sort", resolve(path)) for path in _sort_output_paths(args))

        elif cmd_name == "yq":
            results.extend(("yq", resolve(path)) for path in _yq_inplace_paths(args))

        elif cmd_name in _FILE_WRITE_COMMANDS:
            results.extend(("write", resolve(dest)) for dest in _file_write_destinations(cmd_name, args))

        elif cmd_name == "tar":
            results.extend(("write", resolve(dest)) for dest in _tar_write_targets(args))

        elif cmd_name == "unzip":
            results.extend(("write", resolve(dest)) for dest in _unzip_write_targets(args))

        elif _PYTHON_NAME.match(cmd_name):
            invocation = _python_invocation(args)
            if invocation:
                results.append((cmd_name, invocation))

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
        self.permission_mode = getattr(config, "permission_mode", "manual")
        self.default_permission_mode = self.permission_mode
        self.workspace_root = Path.cwd().resolve()
        self.confirmation_callback: Callable[[str], Awaitable[bool]] | None = None
        self.path_confirmation_callback: Callable[[str], Awaitable[bool]] | None = None
        self.awl_authorized_commands: set[str] = set()

    def set_permission_mode(self, mode: str) -> str | None:
        """Set a session-only permission mode, returning an error if invalid."""
        if mode not in {"manual", "auto", "autonomous"}:
            return "Permission mode must be manual, auto, or autonomous."
        if mode == "autonomous" and os.getenv("AI_ASSIST_SANDBOX", "").lower() != "true":
            return "Autonomous mode is available only inside an ai-assist sandbox."
        self.permission_mode = mode
        return None

    def _is_workspace_path(self, path_str: str) -> bool:
        resolved = Path(path_str).expanduser().resolve()
        roots = (self.workspace_root, AUTO_TEMP_ROOT.resolve())
        return any(_is_relative_to(resolved, root) for root in roots)

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
                # Keep code-running builtins: an allowlist entry is how users allow them
                if first_word in SHELL_BUILTINS - CODE_RUNNING_SHELL_BUILTINS:
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

    async def _validate_path(self, path_str: str, reason: str | None = None, for_write: bool = False) -> str | None:
        """Validate that a path is within allowed directories.
        Falls back to path_confirmation_callback if path is blocked.

        Args:
            path_str: The path to validate.
            reason: Optional context (e.g. the triggering command) shown to the
                user in the approval prompt to explain why the path is needed.
            for_write: True when the path will be written. Writes never use the
                "no allowlist = unrestricted" escape hatch that reads do: with
                restrictions disabled, a write outside the workspace still needs
                confirmation (denied non-interactively), so a hijacked model
                cannot silently write anywhere the OS permits.

        Returns:
            Error message if path is not allowed, None if validation passes
        """
        if self.permission_mode == "autonomous":
            return None

        if is_sensitive_path(Path(path_str)):
            return f"Error: {path_str} is a sensitive location (keys/credentials) the agent cannot access."

        if for_write and is_unwritable_system_path(Path(path_str)):
            return f"Error: {path_str} is a kernel/virtual filesystem (/proc, /sys, /dev) and cannot be written."

        if self.permission_mode == "auto" and self._is_workspace_path(path_str):
            return None

        if not self._path_restrictions_enabled:
            # Reads are unrestricted when no allowlist is configured (backwards
            # compat). Writes are not: only the workspace is allowed silently,
            # anything else falls through to confirmation / non-interactive deny.
            if not for_write:
                return None
            if self._is_workspace_path(path_str):
                return None
        else:
            resolved = Path(path_str).expanduser().resolve()
            for allowed in self.allowed_paths:
                try:
                    resolved.relative_to(allowed)
                    return None
                except ValueError:
                    continue

        # Path not in allowlist (or a write outside it with no allowlist) — try approval
        resolved = Path(path_str).expanduser().resolve()
        if self.path_confirmation_callback is not None:
            description = f"Access path: {resolved}"
            if reason:
                description += f"\nReason: {reason}"
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
                "_readonly": True,
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
                "_readonly": True,
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
                "_readonly": False,
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
                "_readonly": True,
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
                "_readonly": False,
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
                "_readonly": False,
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
                "_readonly": False,
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

    async def confirm(self, description: str) -> bool:
        """Ask the user to approve an action; refused when nobody can be asked."""
        if self.permission_mode == "autonomous":
            return True
        if self.confirmation_callback is None:
            return False
        return await self.confirmation_callback(description)

    async def _check_confirmation(self, tool_full_name: str, description: str) -> str | None:
        """Check if a tool requires confirmation and prompt the user.

        Args:
            tool_full_name: Full tool name (e.g. "internal__create_directory")
            description: Human-readable description of the action

        Returns:
            Error message if rejected/blocked, None if approved
        """
        if self.permission_mode != "manual" or tool_full_name not in self.confirm_tools:
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

        path_error = await self._validate_path(path, for_write=True)
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
        segments = _split_shell_commands(_expand_shell_invocations(full_command))
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
            idx = _strip_transparent_wrappers(tokens, idx)

            if idx >= len(tokens):
                continue

            cmd_name = _resolve_command_token(tokens[idx])
            if cmd_name in INERT_SHELL_BUILTINS:
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

        For compound commands (&&, ;, ||, |), validates each segment
        independently:
        - cd <path> segments: validates the path via path_confirmation_callback
        - Non-builtin command segments: checks allowlist, prompts per-segment

        Returns:
            Error message if blocked, None if allowed
        """
        rejected = self._is_command_prefix_allowed(full_command)
        if not rejected:
            # All non-builtin segments are already allowed; still validate
            # cd paths (builtins are skipped by _is_command_prefix_allowed).
            pairs = _extract_command_argument_paths(full_command)
            for cmd_name, path_or_marker in pairs:
                if cmd_name == "cd" and path_or_marker not in ("<inline-code>", "<stdin>", "<interactive>"):
                    path_error = await self._validate_path(path_or_marker, reason=full_command)
                    if path_error:
                        return f"Error: cd target path is not allowed. {path_error}"
            return None

        # Per-segment validation
        segments = _split_shell_commands(_expand_shell_invocations(full_command))

        for segment in segments:
            try:
                tokens = shlex.split(segment)
            except ValueError:
                tokens = segment.split()

            if not tokens:
                continue

            idx = 0
            while idx < len(tokens) and ENV_VAR_PATTERN.match(tokens[idx]):
                idx += 1
            if idx >= len(tokens):
                continue

            idx = _strip_transparent_wrappers(tokens, idx)
            if idx >= len(tokens):
                continue

            cmd_name = _resolve_command_token(tokens[idx])

            # cd segments: validate the target path
            if cmd_name == "cd":
                args = tokens[idx + 1 :]
                if args and args[0] != "-":
                    path_error = await self._validate_path(args[0], reason=full_command)
                    if path_error:
                        return f"Error: cd target path is not allowed. {path_error}"
                continue

            if cmd_name in INERT_SHELL_BUILTINS:
                continue

            # Check if this segment is prefix-allowed
            effective: list[str] = []
            for t in tokens[idx:]:
                if effective:
                    effective.append(t)
                else:
                    effective.append(_resolve_command_token(t))

            matched = any(effective[: len(e.split())] == e.split() for e in self.allowed_commands)
            if matched:
                continue

            # Prompt for this specific segment
            if self.confirmation_callback is not None:
                approved = await self.confirmation_callback(segment.strip())
                if not approved:
                    return f"Error: Command '{segment.strip()}' was rejected by the user. Try a different approach or use only allowed commands: {', '.join(self.allowed_commands)}."
            else:
                allowed_list = ", ".join(self.allowed_commands)
                return f"Error: Command '{cmd_name}' is not in the allowed commands list: {allowed_list}. Not allowed."

        return None

    def _is_auto_command_safe(self, command: str) -> bool:
        """Return whether a command belongs to the conservative auto policy.

        This intentionally accepts only command families whose effects can be
        understood locally.  Anything unfamiliar is sent through the normal
        confirmation path.
        """
        if any(marker in command for marker in ("$(`", "$(", "`", "<<")):
            return False
        if re.search(r"(^|[\s;|&])(sudo|su|doas)(?:\s|$)", command):
            return False
        if re.search(r"(?:^|\s)(?:curl|wget|ssh|scp|rsync)(?:\s|$)", command):
            return False

        segments = _split_shell_commands(command)
        if not segments:
            return False
        return all(self._is_auto_segment_safe(segment) for segment in segments)

    def _is_auto_segment_safe(self, segment: str) -> bool:
        """Classify one shell segment after compound commands are split."""
        if re.search(r"(?:^|\s)(?:\d*>>?|&>|\d*>&)(?:\s|$)", segment):
            return False
        try:
            tokens = shlex.split(segment)
        except ValueError:
            return False
        idx = 0
        while idx < len(tokens) and _is_safe_env_assignment(tokens[idx]):
            idx += 1
        idx = _strip_transparent_wrappers(tokens, idx)
        if idx >= len(tokens):
            return False
        name = _resolve_command_token(tokens[idx])
        args = tokens[idx + 1 :]
        if name in AUTO_SAFE_SHELL_BUILTINS or name in self.allowed_commands:
            return True
        return self._is_auto_known_command_safe(name, args)

    def _is_auto_known_command_safe(self, name: str, args: list[str]) -> bool:
        """Classify non-allowlisted command families supported by auto mode."""
        if name == "sleep":
            # It has no side effects beyond consuming time; non-interactive
            # command execution is already capped at five minutes.
            return True
        if name == "date":
            return self._is_auto_date_safe(args)
        if name == "sqlite3":
            return self._is_auto_sqlite_read_safe(args)
        if name == "git":
            return self._is_auto_git_safe(args)
        if name == "gh":
            return self._is_auto_gh_safe(args)
        if name in {"pytest", "ruff", "mypy"}:
            return not self._has_outside_workspace_path(args)
        if name == "black":
            return "--check" in args and not self._has_outside_workspace_path(args)
        return (
            name in {"python", "python3"}
            and args[:2] == ["-m", "pytest"]
            and not self._has_outside_workspace_path(args[2:])
        )

    @staticmethod
    def _is_auto_date_safe(args: list[str]) -> bool:
        """Allow date display formats, never clock-setting forms."""
        index = 0
        while index < len(args):
            arg = args[index]
            if arg in {"-u", "--utc"} or arg.startswith(("+", "-I", "--iso-8601")):
                index += 1
                continue
            if arg in {"-d", "--date"}:
                index += 2
                continue
            if arg.startswith("--date="):
                index += 1
                continue
            return False
        return index == len(args)

    def _is_auto_sqlite_read_safe(self, args: list[str]) -> bool:
        """Allow a single read-only SQLite query with display-only options."""
        display_options = {"-header", "-noheader", "-column", "-csv", "-json", "-line", "-list", "-tabs"}
        index = 0
        while index < len(args) and args[index] in display_options:
            index += 1
        if len(args) - index != 2:
            return False

        database = args[index]
        if database != ":memory:" and not self._is_workspace_path(database):
            return False

        query = args[index + 1].strip()
        if ";" in query or not re.match(r"(?:select\b|with\b|explain\s+select\b)", query, re.IGNORECASE):
            return False
        # writefile/readfile/edit are sqlite3 shell functions touching arbitrary files
        return not re.search(
            r"\b(?:alter|attach|create|delete|detach|drop|insert|load_extension|pragma|replace|update|vacuum"
            r"|writefile|readfile|edit|fts3_tokenizer)\b",
            query,
            re.IGNORECASE,
        )

    def _is_auto_git_safe(self, args: list[str]) -> bool:
        args = list(args)
        while args and args[0].startswith("-"):
            flag = args.pop(0)
            if flag == "-C" and args and self._is_workspace_path(args.pop(0)):
                continue
            if flag != "--no-pager":
                return False
        if not args or args[0] not in {"status", "diff", "log", "show", "branch", "rev-parse", "ls-files"}:
            return False
        # --output writes a file; --ext-diff/--textconv run configured programs
        if any(arg.split("=", 1)[0] in _GIT_UNSAFE_READ_OPTIONS for arg in args[1:]):
            return False
        return args[0] != "branch" or self._is_git_branch_listing(args[1:])

    @staticmethod
    def _is_git_branch_listing(args: list[str]) -> bool:
        """`git branch` only lists branches: no names to create, delete, or move."""
        index = 0
        while index < len(args):
            arg = args[index]
            if arg in _GIT_BRANCH_FILTER_OPTIONS:
                index += 2
                continue
            if not arg.startswith("-") or arg.split("=", 1)[0] not in _GIT_BRANCH_LIST_OPTIONS:
                return False
            index += 1
        return True

    @staticmethod
    def _is_auto_gh_safe(args: list[str]) -> bool:
        if not args:
            return False
        if args[0] == "api":
            for index, arg in enumerate(args):
                if arg in ("-X", "--method"):
                    if index + 1 >= len(args) or args[index + 1].upper() != "GET":
                        return False
                elif arg.startswith(("-X", "--method=")):
                    if arg.removeprefix("-X").removeprefix("--method=").upper() != "GET":
                        return False
                elif arg.split("=", 1)[0] in _GH_API_BODY_OPTIONS or arg.startswith(("-f", "-F")):
                    # Fields and request bodies make gh send a POST
                    return False
            return True
        return args[:2] in (["pr", "view"], ["pr", "list"], ["pr", "status"], ["run", "view"], ["run", "list"])

    def _has_outside_workspace_path(self, args: list[str]) -> bool:
        """Detect explicit filesystem paths in an auto-approved command."""
        for arg in args:
            candidate = arg.split("=", 1)[-1]
            if candidate.startswith(("/", "~", "./", "../")) and not self._is_workspace_path(candidate):
                return True
        return False

    async def _validate_command_arguments(
        self, command: str, was_auto_allowed: bool, working_dir: str | None = None
    ) -> str | None:
        """Validate path arguments and parameters for specific commands.

        Checks:
        - find: paths must be within allowed directories
        - python/python3: script paths must be within allowed directories;
          -c (inline code), stdin, and interactive mode require confirmation
          when the command was auto-allowed via allowlist
        - files written by redirections, sort -o and yq -i: must be within
          allowed directories; paths computed at run time need confirmation

        Note: cd path validation is handled by _check_command_allowed to
        avoid double-prompting.

        Args:
            command: The full shell command string
            was_auto_allowed: True if all commands passed the allowlist/builtin
                check without user confirmation (skips python -c confirmation
                to avoid double-prompting)

        Returns:
            Error message if blocked, None if allowed
        """
        for path in _command_path_candidates(command, working_dir):
            if is_sensitive_path(path):
                return f"Error: {path} is a sensitive location (keys/credentials) the agent cannot access."

        # These are extra confirmation gates for commands that were only *silently*
        # auto-allowed (allowlist/builtins). When the command was explicitly
        # user-approved or authorized by a user-started AWL script, was_auto_allowed
        # is False and they are skipped — the user already signed off on it. Hard
        # blocks (sensitive paths above, write-target checks below) always apply.
        if was_auto_allowed:
            program_args = ", ".join(sorted(set(_program_running_arguments(command))))
            if program_args:
                if self.confirmation_callback is None:
                    return f"Error: {program_args} can run arbitrary programs; not allowed in non-interactive mode."
                if not await self.confirmation_callback(
                    f"Allow {program_args} (can run arbitrary programs): {command}"
                ):
                    return f"Error: {program_args} rejected by user."

            hijacked = ", ".join(sorted(_program_hijacking_assignments(command)))
            if hijacked:
                if self.confirmation_callback is None:
                    return f"Error: Setting {hijacked} changes which program runs; not allowed in non-interactive mode."
                if not await self.confirmation_callback(f"Set {hijacked} (changes which program runs): {command}"):
                    return f"Error: Setting {hijacked} rejected by user."

        for cmd_name, path_or_marker in _extract_command_argument_paths(command, working_dir):
            pair_error = await self._validate_argument_pair(cmd_name, path_or_marker, command, was_auto_allowed)
            if pair_error:
                return pair_error

        if was_auto_allowed:
            interpreter_error = await self._check_code_interpreters(command)
            if interpreter_error:
                return interpreter_error

        return None

    async def _validate_argument_pair(
        self, cmd_name: str, path_or_marker: str, command: str, was_auto_allowed: bool
    ) -> str | None:
        """Validate one (command, path-or-marker) pair from _extract_command_argument_paths."""
        if cmd_name == "cd":
            return None
        is_write = cmd_name in _WRITE_TARGET_SOURCES
        if is_write and (git_error := await self._check_code_location_write(path_or_marker)):
            return git_error
        if is_write and _DYNAMIC_PATH_CHARS.intersection(path_or_marker):
            if self.confirmation_callback is None:
                return f"Error: Cannot verify write target {path_or_marker} in non-interactive mode."
            if not await self.confirmation_callback(
                f"Write to a path computed at run time ({path_or_marker}): {command}"
            ):
                return f"Error: Write to {path_or_marker} rejected by user."
            return None
        if path_or_marker == "<dangerous-find>":
            return (
                "Error: find with -exec/-execdir/-ok/-okdir/-delete/-fprintf is not allowed "
                "(these run arbitrary commands or modify files, bypassing the command allowlist)."
            )
        is_module = path_or_marker.startswith("<module:")
        is_script = path_or_marker.startswith("<script:")
        if is_script:
            read_error = await self._validate_path(path_or_marker[len("<script:") : -1], reason=command)
            if read_error:
                return f"Error: {cmd_name} target path is not allowed. {read_error}"
        if is_module or is_script or path_or_marker in ("<inline-code>", "<stdin>", "<interactive>"):
            return await self._gate_code_execution(cmd_name, path_or_marker, command, was_auto_allowed)
        path_error = await self._validate_path(path_or_marker, reason=command, for_write=is_write)
        if path_error:
            return f"Error: {cmd_name} target path is not allowed. {path_error}"
        return None

    async def _gate_code_execution(
        self, cmd_name: str, path_or_marker: str, command: str, was_auto_allowed: bool
    ) -> str | None:
        """Confirm-gate python -c / -m / script / stdin / REPL execution."""
        if not was_auto_allowed:
            return None
        if path_or_marker.startswith("<module:") and self._is_python_module_allowlisted(cmd_name, path_or_marker[8:-1]):
            return None
        if path_or_marker.startswith("<script:") and self._is_invocation_allowlisted(
            [cmd_name, path_or_marker[len("<script:") : -1]]
        ):
            return None
        if self.confirmation_callback is None:
            # A bare interpreter on the allowlist is not enough to run arbitrary
            # code non-interactively; the specific invocation must be allowlisted.
            return f"Error: {cmd_name} {path_or_marker} execution is not allowed in non-interactive mode."
        if not await self.confirmation_callback(f"{cmd_name} {path_or_marker.strip('<>')} execution: {command}"):
            return f"Error: {cmd_name} {path_or_marker} execution rejected by user."
        return None

    async def _check_code_interpreters(self, command: str) -> str | None:
        """Confirm-gate interpreters/runners that execute arbitrary code or packages."""
        for name, iargs in _code_interpreter_invocations(command):
            if self._is_invocation_allowlisted([name, *iargs]):
                continue
            if self.confirmation_callback is None:
                return f"Error: {name} runs arbitrary code and needs confirmation; not allowed non-interactively."
            if not await self.confirmation_callback(f"{name} runs arbitrary code/package: {command}"):
                return f"Error: {name} invocation rejected by user."
        return None

    def _is_invocation_allowlisted(self, tokens: list[str]) -> bool:
        """Whether a multi-token allowlist entry explicitly permits this invocation."""
        return any(
            len(parts) >= 2 and tokens[: len(parts)] == parts
            for parts in (entry.split() for entry in self.allowed_commands)
        )

    @staticmethod
    def _load_command_env_allowlist() -> set[str]:
        """Sensitive env var names the user allows execute_command to see."""
        path = get_config_dir() / COMMAND_ENV_FILE
        if not path.exists():
            return set()
        try:
            with open(path) as f:
                return {name for name in json.load(f) if isinstance(name, str)}
        except json.JSONDecodeError, OSError, TypeError:
            logger.exception("Ignoring unreadable %s", path)
            return set()

    def _is_python_module_allowlisted(self, python: str, module: str) -> bool:
        """Whether an allowlist entry names this module explicitly (e.g. "python3 -m pytest")."""
        invocation = [python, "-m", module]
        return any(
            len(parts) >= 2 and invocation[: len(parts)] == parts
            for parts in (entry.split() for entry in self.allowed_commands)
        )

    async def _check_code_location_write(self, path: str) -> str | None:
        """Writes where code runs later, outside ai-assist, need confirmation.

        That is inside a .git directory (hooks, config) and startup locations
        (shell rc files, ~/.local/bin, systemd user units, ...).
        """
        if self.permission_mode == "autonomous":
            return None
        resolved = Path(path).expanduser().resolve()
        if ".git" in resolved.parts:
            what = "git internals (git runs hooks and programs configured there)"
        elif is_startup_path(resolved):
            what = "a startup location (code there runs later, outside ai-assist)"
        else:
            return None
        if await self.confirm(f"Modify {what}: {path}"):
            return None
        return f"Error: Writing to {path} requires user confirmation: it is {what.split(' (')[0]}."

    async def validate_write_target(self, path: str) -> str | None:
        """Run the internal__write_file checks on a path the agent wants to write.

        Returns an error message if the write is not allowed, None otherwise.
        """
        if self._is_protected_config(path):
            return f"Error: {Path(path).name} is a protected security config file and cannot be modified by the agent."

        path_error = await self._validate_path(path, for_write=True)
        if path_error:
            return path_error

        git_error = await self._check_code_location_write(path)
        if git_error:
            return git_error

        return await self._check_confirmation("internal__write_file", f"Write file: {path}")

    async def _write_file(self, args: dict) -> str:
        """Write content to a file"""
        path = args.get("path")
        content = args.get("content")

        if not path:
            return "Error: path parameter is required"
        if content is None:
            return "Error: content parameter is required"

        write_error = await self.validate_write_target(path)
        if write_error:
            return write_error

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
            await self._validate_path(path, for_write=True),
            await self._check_code_location_write(path),
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
        for cmd_name, path in _extract_command_argument_paths(command, working_dir):
            if cmd_name in _WRITE_TARGET_SOURCES and self._is_protected_config(path):
                return (
                    f"Error: {Path(path).name} is a protected security config file and cannot be modified by the agent."
                )

        # In interactive mode (confirmation_callback set), no timeout -- the user
        # can press Escape to interrupt like any normal interaction.
        # In non-interactive mode, enforce a timeout to prevent runaway commands.
        if self.confirmation_callback is not None:
            timeout = None
        else:
            timeout = min(timeout, 300)  # Max 5 minutes

        cmd_names = extract_command_names(command)

        if working_dir and self.permission_mode == "auto" and not self._is_workspace_path(working_dir):
            return f"Error: Working directory is outside the auto-mode workspace: {working_dir}"

        if self.permission_mode == "auto":
            unsafe_builtins = set(cmd_names).intersection(SHELL_BUILTINS - AUTO_SAFE_SHELL_BUILTINS)
            if unsafe_builtins:
                names = ", ".join(sorted(unsafe_builtins))
                return (
                    f"Error: Shell builtin(s) {names} are not allowed in auto mode. Switch to manual mode to use them."
                )

        # Determine if all commands are auto-allowed (builtins or prefix-matched allowlist)
        was_auto_allowed = not self._is_command_prefix_allowed(command)

        # Commands authorized by AWL scripts are treated as user-confirmed
        # (skip extra confirmation for python -c, etc.)
        awl_confirmed = any(name in self.awl_authorized_commands for name in cmd_names)

        auto_safe = self.permission_mode == "auto" and self._is_auto_command_safe(command)
        if self.permission_mode == "autonomous" or auto_safe:
            error = None
        else:
            error = await self._check_command_allowed(cmd_names, command)
        if error:
            return error

        # Validate command arguments (paths for cd/find, parameters for python)
        # AWL-authorized commands skip the extra inline-code confirmation
        arg_error = None
        if self.permission_mode != "autonomous":
            arg_error = await self._validate_command_arguments(
                command, was_auto_allowed and not awl_confirmed, working_dir
            )
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
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=filter_sensitive_env(os.environ, allowed=self._load_command_env_allowlist()),
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
