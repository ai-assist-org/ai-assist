# Permission Modes

`AI_ASSIST_PERMISSION_MODE` controls how ai-assist handles filesystem and shell
actions. Valid values are `manual` (the default), `auto`, and `autonomous`.
Use `/mode` in interactive mode to display the effective mode or temporarily
switch between modes for the current session.

## Modes

| Mode | Behavior |
| --- | --- |
| `manual` | Uses the command/path allowlists and asks for approval when an interactive callback is available. |
| `auto` | Automatically permits a deliberately small set of local development actions; anything it cannot classify safely follows the manual path. |
| `autonomous` | Bypasses ai-assist command and path approval checks. It is accepted only in a compose-generated ai-assist sandbox. |

`autonomous` is not a credential boundary. An autonomous agent can use every
mount, credential, MCP server, and network capability enabled for its sandbox.
Keep sandbox features minimal. Security configuration files
(`allowed_commands.json`, `allowed_paths.json`, and `skill_env.json`) remain
protected in every mode.

## What auto mode permits

Auto mode permits these operations without an interactive approval prompt:

- Existing explicit command allowlist entries.
- Workspace-local built-in file reads, writes, edits, and directory creation.
- Safe shell builtins: `cd`, `echo`, `false`, `pwd`, `printf`, `test`, `true`,
  and `[`. The target of `cd` is still path-validated.
- Read-only Git subcommands: `status`, `diff`, `log`, `show`, `branch`,
  `rev-parse`, and `ls-files`. Only `git -C <workspace-path>` and
  `git --no-pager` are accepted as Git global options.
- Read-only GitHub CLI operations: `gh api` with the default method or explicit
  `--method GET`, plus `gh pr view|list|status` and `gh run view|list`.
- Workspace-local test and static-analysis commands: `pytest`,
  `python -m pytest`, `python3 -m pytest`, `ruff`, `mypy`, and
  `black --check`.

The workspace is the current working directory when ai-assist starts, plus
`/tmp/ai-assist`. Auto-approved commands with explicit filesystem paths must
stay inside one of those roots.

## What auto mode does not approve

Auto mode deliberately falls back to the normal approval path for:

- Unknown commands, Git mutations (including `push`, `commit`, `add`, and
  `fetch`), and GitHub mutations.
- Google Workspace commands, network tools such as `curl`, `wget`, `ssh`,
  `scp`, and `rsync`, and privilege wrappers (`sudo`, `su`, `doas`).
- Shell redirects, command substitutions, backticks, heredocs, malformed shell
  syntax, and evaluative wrappers such as `eval`, `source`, `command`, or
  `exec`. Unsafe shell builtins are rejected in auto mode; switch to manual
  mode for an intentional use.
- Paths outside the workspace and `/tmp/ai-assist`.

In a non-interactive or background run, an action that needs approval is denied
because no user can answer the prompt.

## Review and extend the policy

The implementation is intentionally code-reviewed rather than user-editable:

- `ai_assist/filesystem_tools.py`: `_is_auto_command_safe()` and its command
  family helpers define the policy.
- `tests/test_filesystem_security.py`: regression cases define the expected
  safe and rejected behaviors.

When adding a command family, keep the rule narrow, validate relevant paths and
arguments, and add both an allowed and a rejected test case. Unknown syntax
must continue to fail closed.
