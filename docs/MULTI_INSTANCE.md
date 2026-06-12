# Running Multiple Instances of ai-assist

ai-assist supports running multiple independent instances on the same system by using different configuration directories.

## Quick Start

Set the `AI_ASSIST_CONFIG_DIR` environment variable to use a different configuration directory:

```bash
# Instance 1 (work)
export AI_ASSIST_CONFIG_DIR=~/.ai-assist-work
ai-assist /interactive

# Instance 2 (personal)
export AI_ASSIST_CONFIG_DIR=~/.ai-assist-personal
ai-assist /interactive
```

## Configuration Directories

Each instance uses its own isolated configuration directory containing:

- **mcp_servers.yaml** - MCP server configurations
- **identity.yaml** - User/assistant identity and preferences
- **event-schedules.json** - Action schedules and event triggers
- **knowledge_graph.db** - Bi-temporal knowledge graph database
- **installed-skills.json** - Installed Agent Skills registry
- **state/** - Monitor state and cache
- **logs/** - Condition action logs
- **skills-cache/** - Cached Agent Skills
- **interactive_history.txt** - Interactive mode command history

## Configuration Priority

The configuration directory is determined by (highest priority first):

1. **Override parameter** (programmatic use)
2. **`AI_ASSIST_CONFIG_DIR` environment variable**
3. **Default:** `~/.ai-assist`

## Use Cases

### Work and Personal Separation

```bash
# Work instance with work MCP servers
export AI_ASSIST_CONFIG_DIR=~/.ai-assist-work
ai-assist /setup-identity  # Configure work identity
# Edit ~/.ai-assist-work/mcp_servers.yaml with work servers

# Personal instance
export AI_ASSIST_CONFIG_DIR=~/.ai-assist-home
ai-assist /setup-identity  # Configure personal identity
```

### Multiple Projects

```bash
# Project A
export AI_ASSIST_CONFIG_DIR=~/projects/project-a/.ai-assist
ai-assist /monitor

# Project B
export AI_ASSIST_CONFIG_DIR=~/projects/project-b/.ai-assist
ai-assist /monitor
```

### Testing/Development

```bash
# Production instance
ai-assist /monitor  # Uses default ~/.ai-assist

# Test instance
export AI_ASSIST_CONFIG_DIR=/tmp/ai-assist-test
ai-assist /interactive  # Isolated test environment
```

## Shell Aliases

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# Work instance
alias ai-work='AI_ASSIST_CONFIG_DIR=~/.ai-assist-work ai-assist'

# Personal instance
alias ai-home='AI_ASSIST_CONFIG_DIR=~/.ai-assist-personal ai-assist'

# Project-specific
alias ai-projecta='AI_ASSIST_CONFIG_DIR=~/projects/project-a/.ai-assist ai-assist'
```

Usage:
```bash
ai-work /interactive
ai-home /monitor
ai-projecta /query "What's the status?"
```

## Reports Directory

Reports are handled separately via `AI_ASSIST_REPORTS_DIR`:

```bash
# Different config dirs can share the same reports dir
export AI_ASSIST_CONFIG_DIR=~/.ai-assist-work
export AI_ASSIST_REPORTS_DIR=~/shared-reports
ai-assist /interactive

# Or use separate report dirs
export AI_ASSIST_CONFIG_DIR=~/.ai-assist-home
export AI_ASSIST_REPORTS_DIR=~/home-reports
ai-assist /interactive
```

Default reports location: `~/ai-assist/reports/`

## Environment Variable Summary

| Variable | Purpose | Default |
|----------|---------|---------|
| `AI_ASSIST_CONFIG_DIR` | Configuration directory | `~/.ai-assist` |
| `AI_ASSIST_REPORTS_DIR` | Reports output directory | `~/ai-assist/reports` |
| `ANTHROPIC_API_KEY` | API authentication | (required) |
| `AI_ASSIST_MODEL` | Claude model to use | `claude-sonnet-4-5@20250929` |
| `AI_ASSIST_ALLOW_SCRIPT_EXECUTION` | Enable script execution | `false` |

## Systemd Services

Run instances as persistent user services that start on login and restart on failure.

```bash
ai-assist /service install              # default instance (~/.ai-assist)
ai-assist /service install ~/.iris      # named instance
ai-assist /service install ~/.iris ~/reports/iris  # with reports dir

ai-assist /service status               # status of default instance
ai-assist /service status ~/.iris       # status of specific instance
ai-assist /service restart              # restart after updating code
ai-assist /service restart ~/.iris      # restart specific instance
ai-assist /service logs ~/.iris         # follow logs (Ctrl+C to exit)
ai-assist /service stop ~/.iris
ai-assist /service start ~/.iris
ai-assist /service remove ~/.iris       # stop, disable, and remove
```

Config dir falls back to `AI_ASSIST_CONFIG_DIR` env var, then `~/.ai-assist`.
Reports dir falls back to `AI_ASSIST_REPORTS_DIR` env var.

### Boot persistence

```bash
loginctl enable-linger $USER
```

Without this, user services stop when you log out.

### Note on .env files

`.env` files are loaded automatically by the app in all modes (service, interactive, query, etc.):

1. **Project `.env`** — loaded first from the source/install directory
2. **Instance `.env`** (`~/.{instance}/.env`) — loaded second, overrides the project `.env`

## Important Notes

1. **Separate Knowledge Graphs**: Each instance maintains its own knowledge graph database
2. **Independent Schedules**: Monitors and tasks run independently per instance
3. **Different Identities**: Each can have its own user/assistant identity
4. **API Key Sharing**: All instances can share the same `ANTHROPIC_API_KEY`
5. **MCP Servers**: Each instance can connect to different MCP servers
6. **Port Conflicts**: None - ai-assist doesn't listen on ports

## Troubleshooting

### Check Current Config Directory

```python
from ai_assist.config import get_config_dir
print(get_config_dir())
```

### Verify Instance Isolation

```bash
# Instance 1
export AI_ASSIST_CONFIG_DIR=/tmp/instance1
ai-assist /setup-identity
ls /tmp/instance1/  # Should see identity.yaml

# Instance 2
export AI_ASSIST_CONFIG_DIR=/tmp/instance2
ai-assist /setup-identity
ls /tmp/instance2/  # Should see different identity.yaml
```

### Clean Start

```bash
# Remove all data for an instance
rm -rf ~/.ai-assist-work

# Restart fresh
export AI_ASSIST_CONFIG_DIR=~/.ai-assist-work
ai-assist /setup-identity
```

## Migration from Single Instance

If you have an existing `~/.ai-assist` and want to split into multiple instances:

```bash
# Backup original
cp -r ~/.ai-assist ~/.ai-assist.backup

# Create work instance (copy from original)
cp -r ~/.ai-assist ~/.ai-assist-work

# Create personal instance (fresh)
export AI_ASSIST_CONFIG_DIR=~/.ai-assist-personal
ai-assist /setup-identity

# Keep original as default or remove it
# mv ~/.ai-assist ~/.ai-assist-legacy
```

## See Also

- [Configuration Guide](../README.md#configuration)
- [MCP Server Tools](../README.md#mcp-server-tools)
- [Interactive Mode](../README.md#interactive-mode)
