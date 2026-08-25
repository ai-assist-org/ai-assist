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
/plugin/update [plugin-name]     # reinstall from source; all plugins if name omitted
/plugin/uninstall <plugin-name>
```

`/plugin/update` re-fetches a plugin from its recorded source to pick up upstream
changes (there is no in-place upgrade — it reinstalls). Omit the name to update
every installed plugin. If the reinstall fails, the previous version is kept.
Bundled MCP servers are re-offered for connection just like a fresh install.

## Marketplaces

A marketplace is a git repo (or local directory) with a
`.claude-plugin/marketplace.json` listing plugins by name. Register one, search
it, then install by name:

```bash
/plugin/marketplace add owner/marketplace-repo@main [nickname]
/plugin/marketplace update [name]      # git pull the cache; all marketplaces if name omitted
/plugin/marketplace list
/plugin/search <query>
/plugin/install <plugin-name>          # resolved through registered marketplaces
```

Pass an optional **nickname** to override the manifest name — handy when two
marketplaces share a name, or to give one a short handle:

```bash
/plugin/marketplace add openshift-eng/ai-helpers@main redhat
/plugin/marketplace update redhat
```

Adding a marketplace whose name (manifest name or nickname) would **mask a
different** already-registered one is refused, with a suggestion to pick a
nickname. Re-adding the same source under the same name just refreshes it.

`update` refreshes the cached marketplace repo so `/plugin/search` and
`/plugin/install` see newly published plugins. Re-running `add` on the same
source has the same effect.

Real-world marketplaces (e.g. `anthropics/claude-plugins-official`) declare each
plugin's `source` as an object rather than a bare path. All the common forms are
resolved automatically:

| `source` form | Example | Resolves to |
| --- | --- | --- |
| `github` | `{"source": "github", "repo": "owner/repo", "path": "plugins/x"}` | `owner/repo/plugins/x` |
| `git` / `git-subdir` | `{"source": "git-subdir", "url": "https://github.com/owner/repo.git", "path": "plugins/x"}` | `owner/repo/plugins/x` |
| string | `"plugins/x"` (relative to the repo, honoring top-level `pluginRoot`) | in-repo path |

An optional `ref`/`branch` on the object selects a git ref. Non-GitHub git URLs
are not supported and are reported clearly when you try to install them. Plugin
names from registered marketplaces are also offered in tab completion after
`/plugin/install`.

### Well-known marketplaces

These public Claude Code marketplaces work out of the box:

| Marketplace | Add command | Contents |
| --- | --- | --- |
| Anthropic official | `/plugin/marketplace add anthropics/claude-plugins-official@main` | Anthropic-managed, high-quality plugins |
| Anthropic community | `/plugin/marketplace add anthropics/claude-plugins-community@main` | Community-submitted plugins |
| Anthropic life sciences | `/plugin/marketplace add anthropics/life-sciences@main` | Plugins for the Claude for Life Sciences launch |
| OpenShift Eng (Red Hat) | `/plugin/marketplace add openshift-eng/ai-helpers@main` | OpenShift/Red Hat developer plugins (jira, ci, git, openshift, must-gather, ...) |

After adding one, discover and install by name:

```bash
/plugin/marketplace add anthropics/claude-plugins-official@main
/plugin/search <query>
/plugin/install <plugin-name>
```

> Note: `anthropics/claude-code` is the Claude Code product repo, **not** a
> marketplace — it has no `.claude-plugin/marketplace.json` and cannot be added.

## Running plugin skills

Plugin skills are **namespaced** to avoid collisions. Run them as:

```
/<plugin>:<skill> [args]     # e.g. /acme:review src/
/<skill> [args]              # short form, only when the name is unique
```

Arguments fill `$ARGUMENTS` (full string) and `$1`, `$2`, ... exactly as with
regular skills. Plugin skills also appear in the system prompt and in tab
completion.

## Agent awareness

The assistant can **see** what is installed but cannot change it. Two read-only
introspection tools are always available:

- `introspection__list_skills` — installed Agent Skills (including plugin skills)
- `introspection__list_plugins` — installed plugins, their source, skills, and MCP servers

Installing, updating, and uninstalling stay human-driven through the `/skill/*`
and `/plugin/*` commands above; the agent has no write access to them.

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
