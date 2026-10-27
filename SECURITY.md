# Security Model

This document describes the security controls in ai-assist for protecting against attacks on AI agents.

## Overview

ai-assist implements defense-in-depth with multiple security layers:

1. **Prompt injection detection** — scans MCP tool results for injection patterns
2. **Tool poisoning detection** — validates MCP tool descriptions for hidden instructions
3. **Rug-pull detection** — detects tool definition changes after initial connection
4. **Script execution sandboxing** — constrains Agent Skills scripts
5. **Tool argument validation** — validates inputs against JSON schemas
6. **Audit logging** — records all tool calls for forensic analysis
7. **Secret redaction** — redacts secrets from tool results and filters sensitive environment variables
8. **Command/path allowlists** — restricts filesystem and command access
9. **Command argument validation** — validates paths in `cd`/`find` and parameters in `python`/`python3`
10. **Plugin trust model** — untrusted plugin skills scanned; bundled MCP servers confirmed before connecting

## 1. Prompt Injection Detection

Untrusted content entering the agent can contain adversarial instructions designed to hijack behavior (indirect prompt injection). ai-assist scans multiple entry points for known injection patterns.

**Module:** `ai_assist/security.py` — `sanitize_tool_result()`

**Detected patterns:**
- `ignore_instructions` — "ignore/disregard/override previous instructions"
- `new_instructions` — "you are now / from now on / new instructions"
- `system_prompt_extraction` — "reveal/show your system prompt"
- `role_hijack` — "pretend to be / roleplay as"
- `output_manipulation` — "do not mention / hide this fact"
- `delimiter_injection` — `</system>`, `[SYSTEM]`, `[INST]`, `<<SYS>>`

**Behavior:**
- Suspicious results are wrapped in `[UNTRUSTED_TOOL_OUTPUT_START]` / `[UNTRUSTED_TOOL_OUTPUT_END]` sentinel markers
- The system prompt instructs the model to treat marked content as raw data only
- All detections are logged as warnings
- Results are not blocked — the agent sees the data but is warned

**Scope:** All untrusted entry points are scanned:
- MCP server tool results (`_execute_tool()`)
- MCP prompt messages (`execute_mcp_prompt()`)
- Agent Skill script output (`execute_skill_script`)

Internal tools (introspection, filesystem, reports, schedules, knowledge) are trusted and not scanned.

## 2. Tool & Content Poisoning Detection

Descriptions and instruction text from external sources can contain hidden instructions that manipulate the agent's behavior (tool poisoning). ai-assist validates these at load/connect time.

**Module:** `ai_assist/security.py` — `validate_tool_description()`

**Detected patterns:**
- `imperative_to_model` — "you must", "always include", "make sure to"
- `references_system_prompt` — "system prompt", "hidden instructions"
- `behavioral_override` — "ignore all other", "disregard previous"
- `data_exfiltration` — "send to", "include in every response"
- `hidden_instructions` — "when called, also always", "before responding, also"

**Additional checks:**
- Description length limit (default: 5000 chars) to detect excessively long descriptions that may hide instructions

**Scope:** All external descriptions and instruction text are validated:
- MCP tool descriptions (at server connection)
- MCP prompt descriptions (at server connection)
- Agent Skill descriptions (at skill load time, in `skills_loader.py`)
- Agent Skill instruction bodies (at load time and when injected into system prompt, in `skills_manager.py`)

**Behavior:** Warnings are logged but content is not blocked, allowing the operator to investigate.

## 3. Rug-Pull Detection

MCP servers can change their tool definitions after initial approval (rug-pull attack). ai-assist fingerprints tool definitions at connection time and checks for changes on reconnect.

**Module:** `ai_assist/security.py` — `ToolDefinitionRegistry`, `compute_tool_fingerprint()`

**How it works:**
1. When a server connects, each tool's `(name, description, input_schema)` is hashed (SHA-256)
2. On server restart or reload, new definitions are compared against stored fingerprints
3. Changes are classified as `modified`, `added`, or `removed`
4. Warnings are printed and logged

**Integration points:**
- `_run_server()` — registers fingerprints at initial connection
- `restart_mcp_server()` — checks for changes after reconnect
- `reload_mcp_servers()` — checks for changes after config reload

## 4. Script Execution Sandboxing

Agent Skills can include executable scripts. **Script execution is disabled by default.**

### Enabling

```bash
# In .env file
AI_ASSIST_ALLOW_SCRIPT_EXECUTION=true
```

### Controls

| Control | Description |
|---------|-------------|
| **Disabled by default** | Requires explicit opt-in via environment variable |
| **Permission system** | Skills must declare `allowed-tools: "internal__execute_skill_script"` |
| **Path validation** | Scripts must be in the skill's `scripts/` directory; traversal blocked |
| **Environment filtering** | API keys, tokens, secrets filtered before execution |
| **Timeout** | 30-second hard limit |
| **Output limit** | 20KB maximum output |
| **No shell injection** | `subprocess.run()` with `shell=False` |

### Filtered environment patterns

`*API_KEY*`, `*TOKEN*`, `*SECRET*`, `*PASSWORD*`, `ANTHROPIC_*`, `GOOGLE_*`, `AWS_*`, `AZURE_*`, `GITHUB_TOKEN`, `JIRA_*`

## 5. Tool Argument Validation

Before executing any tool, arguments are validated against the tool's JSON schema. Invalid arguments are rejected without execution.

**Implementation:** `_validate_tool_arguments()` in `agent.py`

## 6. Audit Logging

All tool calls are logged with:
- Tool name and arguments
- Result text
- Success/failure status
- Timestamp

**Implementation:** `ai_assist/audit.py` — `AuditLogger`

## 7. Source Citation / Grounding

The system prompt requires every factual claim to cite the tool call that provided it. This prevents hallucination and makes it easy to verify data provenance.

## 8. Command Argument Validation

### How command lines are split

Before any allowlist or argument check, `_split_shell_commands()` splits a
command line into every simple command that would run:

- separators: `;`, `&&`, `||`, `|`, `&`, newlines, parentheses (subshells,
  `case` patterns); comments end at the newline; `\`-newline continues a line
- `$(...)`, backticks, `<(...)` and `>(...)` (also inside double quotes,
  `$((...))` and unquoted heredoc bodies) are checked as separate commands
- quoted heredoc bodies (`<<'EOF'`) are skipped as data
- keywords and builtins that run the next word (`if`, `then`, `do`, `!`, `{`,
  `time`, `command`, `builtin`, `exec`, ...) are skipped, and the command after
  them is checked
- builtins that run code (`eval`, `source`/`.`, `trap`, `alias`, `enable`,
  `hash`, `mapfile`/`readarray`, `coproc`) need an allowlist entry or
  confirmation; other builtins (`echo`, `cd`, `export`, ...) are free

Even when a command passes the allowlist check, its arguments may target restricted paths or execute arbitrary code. ai-assist inspects arguments for specific commands before execution.

**Module:** `ai_assist/filesystem_tools.py` — `_extract_command_argument_paths()`, `_validate_command_arguments()`

### Path validation for `cd`, `find` and written files

When path restrictions are enabled (`allowed_paths` is non-empty), the path arguments of `cd` and `find`, and the files a command writes, are validated against the same allowed directories used by filesystem tools.

| Command | What is checked |
|---------|-----------------|
| `cd <dir>` | Target directory must be within allowed paths |
| `find <paths...> [options]` | All search paths (before option flags) must be within allowed paths |

| `find -H/-L/-P <paths...>` | Leading `find` options are skipped, then paths are checked |
| `> file`, `>> file`, `2> file`, `&> file`, ... | Every output redirection target, for any command |
| `sort -o file`, `yq -i ... file` | Files written in place |

Path traversal is blocked: `cd /allowed/../../etc` is resolved before validation.
Write targets the shell computes at run time (`> $HOME/x`, globs,
substitutions) cannot be checked statically: they require confirmation and are
rejected in non-interactive mode. Protected config files are refused as write
targets in every permission mode. `/dev/null`, `/dev/stdout`, `/dev/stderr`,
`/dev/tty` and `/dev/fd/*` are always allowed.

`cd -` (previous directory) and `cd` with no arguments are not checked since they cannot be resolved statically.

### Sensitive locations

Commands and the built-in file tools can never touch credential and key
locations, even inside allowed paths: `~/.ssh`, `~/.gnupg`, `~/.aws`,
`~/.azure`, `~/.config/gcloud`, `~/.config/gh`, `~/.kube`,
`~/.docker/config.json`, `~/.netrc`, `~/.git-credentials`, `~/.pgpass`,
`~/.password-store`, `~/.local/share/keyrings`, and `.env` / `mcp-oauth/` in
the config dir. Every word of a command is checked after expanding `$VARS`,
`~`, globs and earlier `cd` targets. This is best effort: a path the shell
builds at run time (`find ~ | xargs cat`, `grep -r x ~`) is not caught, which
is why tool results are also redacted (section 8b).

### Assignments that change which program runs

Setting `PATH`, `LD_*`, `BASH_ENV`, `GIT_SSH_COMMAND`, `PYTHONPATH`,
`NODE_OPTIONS` and similar variables (as a `VAR=value cmd` prefix, a bare
assignment, via `env`, or with `export`/`declare`) would let an allowlisted
name run another program. Such commands require confirmation and are rejected
in non-interactive mode.

### Parameter validation for `python` / `python3`

Python commands receive additional scrutiny beyond the command allowlist:

| Invocation | Behavior |
|------------|----------|
| `python3 script.py` | Script path validated against allowed directories |
| `python3 -c "code"` | Blocked in non-interactive mode; requires confirmation in interactive mode |
| `python3 -` (stdin) | Blocked in non-interactive mode; requires confirmation in interactive mode |
| `python3` (no args) | Blocked in non-interactive mode; requires confirmation in interactive mode |
| `python3 -m module` | Allowed (module execution, no path to validate) |

**Double-prompting avoidance:** When `python` is not in the command allowlist, the user already confirms the full command (including `-c` details) during the allowlist check. The parameter validation only adds a second prompt when `python` was auto-allowed via the allowlist — preventing commands silently added to the allowlist from executing arbitrary inline code without review.

### Result redirection and jq

`__save_to_file` goes through the same checks as `internal__write_file`
(protected config files, path allowlist, write confirmation) before the tool
runs. `__jq_filter` and `internal__json_query` run jq with an empty
environment, so `$ENV` / `env` cannot read API keys.

### Secrets in command environments

`execute_command` runs with the same filtered environment as skill scripts
(see the patterns in section 4). To let commands see a specific variable
(e.g. `GITHUB_TOKEN` for `gh`), list it in `~/.ai-assist/command_env.json`:

```json
["GITHUB_TOKEN"]
```

Sandboxes created with the `gh` feature get this file automatically.

### AWL scripts and scheduled actions

Commands written in backticks in an AWL script are auto-authorized only when
the user started the script (`ai-assist /run`, or an action whose AWL prompt
the user confirmed). Scripts the agent runs via
`introspection__execute_awl_script` get no auto-authorization. Commands come
from the script text, never from interpolated variable values (see
`docs/AWL_SPECIFICATIONS.md`).

Creating or changing an action (`internal__create_action` /
`internal__update_action`) so that it runs an AWL script requires user
confirmation, and is refused when nobody can be asked. `event-schedules.json`
is a protected config file, so the agent cannot add actions behind this check.

### Plan mode

`/plan` restricts the agent to read-only tools both in the tool list it sees
and when a tool is called (`_execute_tool_impl`), so a hijacked model cannot
call a hidden write tool. Saving results with `__save_to_file` or report
redirection is refused too. A tool counts as read-only when its `_readonly`
annotation says so; MCP tools without one are allowed only when their name
starts with a read-style verb (`get_`, `list_`, `search_`, `query_`, `find_`,
`count_`, `read_`, ...). Set `readonly_tools` (fnmatch patterns) on a server in
`mcp_servers.yaml` to classify its tools explicitly.

### Saved user preferences

Knowledge saved as `user_preference` is injected into every future system
prompt. When the model saves one (`internal__save_knowledge`), the user must
confirm the key and content; it is refused when nobody can be asked. Other
knowledge types, and facts extracted from the conversation at compaction, are
saved without confirmation.

## 8b. Secret Redaction

**Module:** `ai_assist/redaction.py` — `Redactor`

Every tool result (MCP, internal, introspection, goal tools) and every MCP
prompt/resource is redacted before the model, history, audit log, knowledge
graph, MLflow traces, or `__save_to_file` see it:

- **Environment values** — the value of any variable whose name contains
  `TOKEN`, `SECRET`, `PASSWORD`, `API_KEY`, or ends in `_KEY`/`_PAT` (8+
  characters) becomes `[REDACTED:VAR_NAME]`.
- **Known formats** — Anthropic/OpenAI `sk-`, GitHub, GitLab, Slack, Google
  `AIza`, AWS access key ids, JWTs, `Bearer` tokens, `user:password@` in URLs,
  PEM private keys.
- **Your own list** — `~/.ai-assist/redact.json`:

```json
{"strings": ["internal-codename"], "patterns": ["EMP-\\d{6}"]}
```

`redact.json` and `command_env.json` are protected config files. Redaction
catches accidental leaks but not a model that deliberately transforms a secret
(`rev`, `base64`) before printing it, which is why command environments are
filtered too.

## 9. Claude Code Plugin Trust Model

Claude Code plugins (`/plugin/install`) bundle skills and MCP servers from
third-party sources. They are treated as **untrusted content**:

| Control | Description |
|---------|-------------|
| **Skill scanning** | Plugin skill descriptions and bodies are validated for injection patterns at load time, identically to standalone Agent Skills |
| **MCP confirmation** | Installing a plugin that bundles MCP servers lists the commands and requires explicit confirmation before any server is connected |
| **Persistence** | Confirmed servers reconnect automatically on later launches, like servers in `mcp_servers.yaml`; uninstalling the plugin disconnects and removes them |
| **Namespacing** | Plugin skills (`plugin:skill`) and servers (`plugin__server`) are namespaced to prevent collisions with local skills/servers |
| **No agents/hooks** | `agents/` and `hooks/` are not loaded (no equivalent subsystem), so their code never runs |
| **Script execution** | Still disabled by default; plugin skill scripts require the same `AI_ASSIST_ALLOW_SCRIPT_EXECUTION` opt-in |

**Residual risk:** a confirmed MCP server runs an external process with the same
permissions as the ai-assist process. `${CLAUDE_PLUGIN_ROOT}`/`${CLAUDE_*}`
variables are not expanded, so bundled stdio servers relying on them will not
start. Only install plugins from sources you trust.

**Implementation:** `plugins_loader.py`, `plugins_manager.py`

## Threat Model

### What We Protect Against

| Threat | Defense |
|--------|---------|
| Indirect prompt injection via tool results | Pattern detection + sentinel markers |
| Indirect prompt injection via MCP prompts | Prompt message sanitization |
| Indirect prompt injection via script output | Script output sanitization |
| Tool poisoning via MCP tool descriptions | Description validation at connect time |
| Tool poisoning via MCP prompt descriptions | Description validation at connect time |
| Skill poisoning via description or body | Validation at load time and system prompt injection |
| Plugin skill poisoning | Same validation as standalone skills at load time |
| Unsolicited plugin MCP server processes | Explicit confirmation required before connecting |
| Rug-pull attacks (definition changes) | Fingerprint registry + change detection |
| Directory traversal in scripts | Path validation |
| Secret leakage in scripts | Environment variable filtering |
| Script DoS (infinite loops) | 30-second timeout |
| Script memory exhaustion via output | 20KB output limit |
| Shell injection in scripts | `shell=False` |
| Hallucinated data | Grounding nudge + source citation |
| `cd`/`find` to restricted directories | Command argument path validation |
| `python -c` arbitrary code execution | Blocked or confirmation-gated |
| `python` interactive/stdin execution | Blocked or confirmation-gated |

### Known Limitations

| Risk | Status |
|------|--------|
| Novel injection patterns not in regex set | Logged but not blocked |
| Malicious system calls within scripts | Mitigated by trust model |
| Network access from scripts | No restriction (relies on system firewall) |
| CPU/memory exhaustion in scripts | No cgroup limits (future) |
| Filesystem access from scripts | Same permissions as ai-assist process |

## Best Practices

### For Users

1. **Review MCP server sources** — only connect to trusted servers
2. **Check logs for warnings** — security detections are logged at WARNING level
3. **Keep script execution disabled** unless needed
4. **Run with limited permissions** — don't run as root
5. **Review Agent Skills** before installation
6. **Vet Claude Code plugins** before installing — review their skills and only
   confirm bundled MCP servers from trusted sources

### For Skill Authors

1. Declare required tools explicitly in `SKILL.md`
2. Validate all inputs in scripts
3. Avoid `eval()`, `exec()`, `shell=True`
4. Document dependencies clearly
5. Handle errors gracefully

## Future Enhancements

- Container sandboxing for scripts (Docker)
- Resource limits via cgroups (CPU, memory)
- Guardrail LLM for injection detection (complement regex patterns)
- Code signing for Agent Skills
- Network restrictions for scripts

## Reporting Security Issues

If you discover a security vulnerability:

1. **Do not** open a public issue
2. Report via GitHub Security Advisories or email maintainers
3. Include: description, reproduction steps, impact, and suggested fix

## References

- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Simon Willison: Prompt Injection](https://simonwillison.net/series/prompt-injection/)
- [Anthropic MCP Security Guidelines](https://modelcontextprotocol.io/specification/2025-03-26/basic/security)
- [agentskills.io specification](https://agentskills.io/specification)
- [Python subprocess security](https://docs.python.org/3/library/subprocess.html#security-considerations)
