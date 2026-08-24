# Using Claude Code Plugins

ai-assist is compatible with a subset of the
[Claude Code plugin](https://docs.claude.com/en/docs/claude-code/plugins) format.
A plugin is a directory (or git repository) containing a
`.claude-plugin/plugin.json` manifest plus, optionally:

```
my-plugin/
├── .claude-plugin/
│   └── plugin.json      # Required: name, version, description
├── skills/              # Agent Skills (one directory per skill)
│   └── <name>/SKILL.md
├── commands/            # Legacy slash commands (loaded as skills)
│   └── <name>.md
├── .mcp.json            # Bundled MCP servers
├── agents/              # NOT supported (skipped)
└── hooks/               # NOT supported (skipped)
```

## What is loaded

| Component | Supported | Notes |
|-----------|-----------|-------|
| `skills/*/SKILL.md` | ✅ | Registered as namespaced Agent Skills |
| `commands/*.md` | ✅ | Loaded as skills (frontmatter optional) |
| `.mcp.json` | ✅ | Merged into ai-assist's MCP config |
| `agents/` (subagents) | ❌ | No equivalent subsystem; skipped with a notice |
| `hooks/` | ❌ | No equivalent subsystem; skipped with a notice |

## Installing plugins

```bash
ai-assist /interactive

# Directly from a git repository (or a subdirectory of one):
/plugin/install owner/repo@main
/plugin/install owner/repo/path/to/plugin@main

# From a local directory:
/plugin/install /absolute/path/to/plugin@main

# Manage installed plugins:
/plugin/list
/plugin/uninstall <plugin-name>
```

## Marketplaces

A marketplace is a git repo (or local directory) with a
`.claude-plugin/marketplace.json` listing plugins by name. Register one, search
it, then install by name:

```bash
/plugin/marketplace add owner/marketplace-repo@main
/plugin/marketplace list
/plugin/search <query>
/plugin/install <plugin-name>          # resolved through registered marketplaces
```

## Running plugin skills

Plugin skills are **namespaced** to avoid collisions. Run them as:

```
/<plugin>:<skill> [args]     # e.g. /acme:review src/
/<skill> [args]              # short form, only when the name is unique
```

Arguments fill `$ARGUMENTS` (full string) and `$1`, `$2`, ... exactly as with
regular skills. Plugin skills also appear in the system prompt and in tab
completion.

## Security

- **MCP servers run external processes.** When you install a plugin that bundles
  MCP servers, ai-assist lists the commands and asks you to confirm before
  connecting them. Confirmed servers reconnect automatically on subsequent
  launches, just like servers defined in `mcp_servers.yaml`.
- **Plugin skills are untrusted content** injected into the system prompt. Their
  descriptions and bodies are scanned for prompt-injection patterns at load time,
  the same as regular skills.
- **Script execution stays disabled by default** (`AI_ASSIST_ALLOW_SCRIPT_EXECUTION`).

See [SECURITY.md](../SECURITY.md) for the full trust model.

## Limitations

- `${CLAUDE_PLUGIN_ROOT}` and other `${CLAUDE_*}` variables are **not** expanded in
  `.mcp.json`; stdio servers that rely on them may not start.
- Claude Code command features `` !`shell` ``, `@file`, and `${CLAUDE_*}` are not
  interpreted in `commands/*.md` bodies (only `$ARGUMENTS`/`$1` substitution).
