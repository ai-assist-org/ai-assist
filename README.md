# ai-assist - AI Assistant for Knowledge Workers

An intelligent AI assistant powered by Claude, Skills and MCP (Model Context Protocol) that helps knowledge workers with periodic automated monitoring and interactive querying.

Works with MCP servers like:
- [DCI MCP Server](https://github.com/redhat-community-ai-tools/dci-mcp-server) for DCI and Jira
- [Second Brain](https://github.com/flepied/second-brain-agent) for personal notes
- Any other MCP-compatible server

Works with skills like:

- [pdf](https://github.com/anthropics/skills/blob/main/skills/pdf/SKILL.md)
- [markdown-converter](https://github.com/intellectronica/agent-skills/blob/main/skills/markdown-converter/SKILL.md)
- Any other [skills](https://skills.sh/)

## Features

- 🤖 **AI-Powered**: Claude Sonnet 4.5 for intelligent analysis
- 🔌 **MCP Integration**: Connect to any MCP server for tools and data
- 📊 **Monitoring**: Automated scheduled checks with smart notifications
- ⏰ **Scheduled Actions**: One-shot future actions with notifications
- 💬 **Interactive Mode**: Rich TUI with streaming responses and history
- 🧠 **Knowledge Graph**: Temporal database tracking entities and changes
- 📝 **Report Generation**: Create and manage markdown reports
- ⚡ **Hot Reload**: Schedule changes take effect immediately
- 🚀 **Agent Skills**: Install specialized skills following [agentskills.io](https://agentskills.io/) specification
- 🔄 **AWL Workflows**: Script multi-step agent workflows with conditionals, loops, and variable propagation

## Quick Start

### Prerequisites

Choose ONE authentication method:
- **Vertex AI** (Google Cloud): Enterprise/company Claude access
- **Direct API** (Anthropic): Personal use with free tier

### Installation

First, install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install ai-assist:

```bash
git clone https://github.com/fredericlepied/ai-assist
cd ai-assist
uv sync
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your credentials
```

**Vertex AI** (Google Cloud):
```bash
export ANTHROPIC_VERTEX_PROJECT_ID='your-gcp-project-id'
gcloud auth application-default login
```

**Direct API** (Anthropic - free tier available):
```bash
export ANTHROPIC_API_KEY='sk-ant-...'  # Get from console.anthropic.com
```

📖 **Vertex AI setup:** See [VERTEX_AI_SETUP.md](VERTEX_AI_SETUP.md)

**Personalization** (Optional):

Customize the assistant's personality and communication style:

```bash
uv run ai-assist /identity-init  # Create template
uv run ai-assist /identity-show  # View current settings
```

Edit `~/.ai-assist/identity.yaml` to configure:
- Your name, role, and organization
- Assistant nickname and personality
- Communication preferences (formality, verbosity, emoji usage)
- Work context for better assistance

📖 **Full identity guide:** See [docs/IDENTITY.md](docs/IDENTITY.md)

**Logging** (Optional):

Control logging verbosity for debugging and monitoring:

```bash
# File logging level (default: INFO)
export AI_ASSIST_LOG_LEVEL=DEBUG

# Console logging level (default: WARNING)
export AI_ASSIST_CONSOLE_LOG_LEVEL=INFO
```

Logs are written to: `~/.ai-assist/logs/ai-assist-YYYY-MM-DD.log`

📖 **Logging guide:** See [docs/LOGGING.md](docs/LOGGING.md)

## Usage

### Interactive Mode

Chat with the assistant using a rich terminal interface:

```bash
uv run ai-assist /interactive
# or simply
uv run ai-assist
```

**Features:**
- 🎨 Rich markdown formatting with syntax highlighting
- 📝 Multi-line editing (Esc-Enter for newlines)
- 📚 Command history with search (Ctrl-R)
- ⌨️ Tab completion for commands
- ⚡ Streaming responses in real-time
- 🛑 Cancel streaming with Escape key
- 🧠 Conversation memory with persistent storage in knowledge graph
- 💾 Nightly self-reflection extracts knowledge from conversations

**Built-in Commands:**
- `/status` - Show statistics
- `/history` - Recent monitoring history
- `/clear` - Clear conversation memory
- `/clear-cache` - Clear response cache
- `/kg-save [on|off]` - Toggle knowledge graph auto-save
- `@no-kg <query>` - Suppress KG context injection for a single query
- `/kg-viz` - Visualize knowledge graph in browser
- `/awl-viz [script.awl]` - Visualize AWL workflow as interactive flowchart in browser
- `/prompts` - List available MCP prompts with arguments
- `/prompt-info <server/prompt>` - Show detailed prompt info
- `/skill/install <source>@<branch>` - Install an Agent Skill
- `/skill/uninstall <name>` - Uninstall an Agent Skill
- `/skill/list` - List installed Agent Skills
- `/skill/search <query>` - Search ClawHub and skills.sh registries
- `/skill/add_env <skill> <VAR>` - Allow an environment variable for a skill's scripts
- `/skill/remove_env <skill> <VAR>` - Remove an allowed env var from a skill
- `/skill/list_env [skill]` - Show allowed env vars for skills
- `/mcp/restart <server>` - Restart an MCP server (picks up binary updates)
- `/help` - Show help
- `/exit` or `/quit` - Exit

**Security:**

When a command is not on the allowlist, the assistant prompts for confirmation:

```
Allow? [y/N/a(lways)]
```

- **y** - Allow once
- **N** - Deny (default)
- **a(lways)** - Allow and permanently add to the allowlist

**Example queries:**
```
What are the latest failing DCI jobs?
Show me critical Jira tickets in the CILAB project
Find OpenShift 4.19 jobs that failed in the last week
```

### MCP Prompts

Execute MCP server prompts directly with slash commands:

```bash
# In interactive mode
/dci/rca         # Run RCA prompt from dci server
/prompts         # List all available prompts
```

MCP prompts can request arguments interactively and automatically execute with streaming results.

### Agent Skills

Install specialized skills following the [agentskills.io](https://agentskills.io) specification:

```bash
# In interactive mode
/skill/install anthropics/skills/skills/pdf@main     # Install PDF skill from GitHub
/skill/install /path/to/my-skill@main                # Install local skill
/skill/list                                          # List installed skills
/skill/uninstall pdf                                 # Uninstall skill
/skill/search pdf                                    # Search ClawHub and skills.sh
```

**What are Agent Skills?**
- Specialized instructions for Claude following agentskills.io standard
- Automatically loaded into system prompt (no activation needed)
- Can include scripts, references, and assets

📖 **Creating personal skills:** See [docs/PERSONAL_SKILLS.md](docs/PERSONAL_SKILLS.md)
- Persistent across sessions

**Example skills:**
- PDF processing (extract text, fill forms, merge)
- DOCX manipulation
- Custom workflow automation
- Domain-specific expertise

**Creating your own skills:**
```bash
mkdir -p my-skill
cat > my-skill/SKILL.md << 'EOF'
---
name: my-skill
description: My custom skill
---

# My Skill Instructions

When the user asks for X, do Y...
EOF

uv run ai-assist
You> /skill/install /path/to/my-skill@main
```

📖 **Skill specification:** See [agentskills.io/specification](https://agentskills.io/specification)

#### Script Execution (Advanced)

Skills can include executable scripts in a `scripts/` directory. **Script execution is disabled by default for security.**

**Enable script execution:**
```bash
export AI_ASSIST_ALLOW_SCRIPT_EXECUTION=true
uv run ai-assist
```

**Security requirements:**
- Skills must declare permission in SKILL.md:
  ```yaml
  allowed-tools: "internal__execute_skill_script"
  ```
- Scripts run in a sandboxed environment:
  - No access to API keys or secrets (environment filtered)
  - Per-skill env var allowlist via `/skill/add_env` (persisted in `~/.ai-assist/skill_env.json`)
  - 30-second timeout limit
  - Output limited to 20KB
  - Directory traversal blocked
  - No shell injection possible

**Example skill with script:**
```markdown
---
name: pdf-tools
description: PDF processing utilities
compatibility: Requires python3, python3-pypdf
allowed-tools: "internal__execute_skill_script"
---

# PDF Tools

Run `scripts/check_fillable_fields.py` to detect form fields.
```

📋 **Security details:** See [SECURITY.md](SECURITY.md) for the complete security model

#### Context Window

Claude 4.6+ models (Opus 4.6, Opus 4.7, Sonnet 4.6) have native 1M token context windows — no opt-in or extra configuration needed.

#### Adaptive Context Limits

Truncation limits scale automatically with the context window. By default, each message is limited to 5% of the context window and total messages to 60%.

You can tune these percentages via environment variables:

```bash
export AI_ASSIST_MESSAGE_LIMIT_PCT=5      # Max % of context per message (1-20, default: 5)
export AI_ASSIST_TOTAL_MESSAGES_PCT=60    # Max % of context for all messages (20-80, default: 60)
export AI_ASSIST_RESERVE_PCT=25           # Reserve % for system prompt + output (10-40, default: 25)
```

### Monitoring Mode

Run periodic monitoring with automated checks:

```bash
uv run ai-assist /monitor
```

- Runs actions from `~/.ai-assist/event-event-schedules.json`
- Auto-saves findings to knowledge graph
- Hot-reloads when schedules or actions change (FileWatchdog)
- Sends notifications on important updates
- Handles laptop suspension gracefully (catches up missed runs)
- **Executes scheduled one-shot actions** (event-driven, no polling)

**Create monitors via interactive mode:**
```bash
uv run ai-assist
You: Create a monitor to check for failed DCI jobs every 5 minutes
```

**Enable notifications for periodic tasks:**

Add `notify` and `notification_channels` to any action in `event-event-schedules.json`:

```json
{
  "version": "2.0",
  "actions": [
    {
      "name": "critical-failures",
      "prompt": "Check for critical DCI failures",
      "trigger": {"type": "interval", "every": "1h"},
      "notify": true,
      "notification_channels": ["desktop", "file"]
    },
    {
      "name": "daily-summary",
      "prompt": "Summarize yesterday's DCI jobs",
      "trigger": {"type": "schedule", "at": "9:00", "days": "weekdays"},
      "notify": true,
      "notification_channels": ["console"]
    }
  ]
}
```

When `notify` is true, you'll receive notifications on task completion via:
- **desktop**: System notifications (notify-send on Linux)
- **file**: Append to `~/.ai-assist/notifications.log`
- **console**: Display in /monitor output
- Interactive mode automatically shows notifications from any channel in the TUI
- Recent notifications (last 15 min) are injected into the interactive agent's context, so you can ask follow-up questions

### Event-Driven Actions

React to external events via MQTT or D-Bus. Actions in `~/.ai-assist/event-event-schedules.json` use a unified **trigger + prompt** model. D-Bus triggers support per-trigger bus selection (`system` or `session`).

**Example: USB device monitoring**

```json
{
  "event_sources": { "dbus": {} },
  "actions": [
    {
      "name": "USB Filesystem Mounted",
      "trigger": {
        "type": "dbus",
        "interface": "org.freedesktop.DBus.Properties",
        "signal": "PropertiesChanged",
        "bus": "system",
        "payload_contains": "MountPoints"
      },
      "prompt": "A USB filesystem was just mounted. Describe the device.",
      "notify": true,
      "notification_channels": ["desktop", "console", "file"]
    }
  ]
}
```

**Example: "Welcome back" briefing on screen unlock**

```json
{
  "actions": [
    {
      "name": "Screen Locked",
      "trigger": {
        "type": "dbus",
        "interface": "org.gnome.ScreenSaver",
        "signal": "ActiveChanged",
        "bus": "session",
        "payload_contains": "True"
      },
      "prompt": "Save the current timestamp to the report screen-lock-time."
    },
    {
      "name": "Welcome Back Briefing",
      "trigger": {
        "type": "dbus",
        "interface": "org.gnome.ScreenSaver",
        "signal": "ActiveChanged",
        "bus": "session",
        "payload_contains": "False"
      },
      "prompt": "The user just unlocked their screen. Read the screen-lock-time report, check recent notifications, and give a brief summary of how long they were away and what happened.",
      "notify": true,
      "notification_channels": ["desktop", "console", "file"]
    }
  ]
}
```

**Trigger types**: `interval`, `schedule`, `interval_range`, `once`, `mqtt`, `dbus`

**Event filtering**: `payload_contains` and `payload_regex` filter events before execution. Burst signals are debounced (3s window).

**Optional dependencies**: `pip install ai-assist[mqtt]` for MQTT, `pip install ai-assist[dbus]` for D-Bus.

The agent can also create/update/delete actions at runtime via built-in tools (`internal__create_action`, `internal__list_actions`, etc.).

### Scheduled Actions

Schedule one-time future actions that execute automatically with notifications:

```bash
uv run ai-assist /interactive
You: Remind me in 2 hours to check the DCI job status for job-456
```

The agent will:
- Schedule the action for execution in 2 hours
- Execute it automatically when due (via `/monitor` process)
- Send desktop notification + file log when complete

**Supported Time Formats:**
- `in X hours/minutes/days` - Relative time
- `tomorrow at 9am` - Next day specific time
- `next monday 10:00` - Specific day and time

**Notification Channels:**
- **desktop**: System notifications (notify-send on Linux)
- **file**: Append to `~/.ai-assist/notifications.log`
- **console**: Display in /monitor output
- **TUI**: Automatically displayed in interactive mode (watches notification log)

**Agent Decision Making:**
The agent intelligently decides how to execute scheduled actions:
- "Remind me to watch TV" → Simple notification (no agent query)
- "Tell me what's in my Gmail inbox" → Query via agent, notify with results
- "Check failed jobs and save report" → Query via agent, save to report

**View scheduled actions:**
```bash
cat ~/.ai-assist/scheduled-actions.json
```

**View notification history:**
```bash
tail -f ~/.ai-assist/notifications.log
```

**Managing Scheduled Actions:**

Completed and failed actions older than 7 days are automatically archived to `~/.ai-assist/scheduled-actions-archive.jsonl`.

```bash
# Manual cleanup
uv run ai-assist /cleanup-actions

# View archive
cat ~/.ai-assist/scheduled-actions-archive.jsonl | jq
```

**File Structure:**
- `scheduled-actions.json` - Active and recent (≤7 days) actions
- `scheduled-actions-archive.jsonl` - Historical actions (>7 days old)

**⚠️ Important:** The `/monitor` process must be running for scheduled actions to execute:
```bash
uv run ai-assist /monitor  # Keep running in background

# Or install as a persistent systemd user service:
ai-assist /service install          # default instance (~/.ai-assist)
ai-assist /service install ~/.iris  # additional instance
```

**Note:** Systemd user services only run while you're logged in. To start services at boot:
```bash
sudo loginctl enable-linger $USER
```

See [docs/MULTI_INSTANCE.md](docs/MULTI_INSTANCE.md#systemd-services) for details.

#### Sandbox Instances (Container Isolation)

Run ai-assist in isolated podman-compose stacks with per-service credential isolation.
Each MCP server runs in its own container and only sees its own secrets:

```bash
# Build container images
make sandbox-build                                         # base + dci-mcp-server
make sandbox-build-dev                                     # dev profile (adds Go, Ansible, uv, shellcheck, yamllint)

# Create and configure an instance
ai-assist /sandbox init my-agent                          # all features, base image
ai-assist /sandbox init my-agent --features=ssh,git       # only ssh and git
ai-assist /sandbox init my-agent --image=ai-assist-dev    # use dev image with extra tools
cp ~/.ai-assist-instances/my-agent/.env.example ~/.ai-assist-instances/my-agent/.env
# Edit .env with credentials, edit sandbox/.ai-assist/identity.yaml

# Run queries, AWL scripts, or monitoring in the sandbox
ai-assist /sandbox run my-agent /query "What failed today?"
ai-assist /sandbox run my-agent /run workflow.awl
ai-assist /sandbox run my-agent /monitor

# Manage instances
ai-assist /sandbox list
ai-assist /sandbox stop my-agent
ai-assist /sandbox delete my-agent

# Install as a persistent systemd service (runs /monitor)
ai-assist /sandbox service my-agent install
ai-assist /sandbox service my-agent status
ai-assist /sandbox service my-agent logs -f
ai-assist /sandbox service my-agent stop
ai-assist /sandbox service my-agent remove
```

Available features: `ssh` (agent forwarding), `gpg` (commit signing), `git` (gitconfig), `gh` (GitHub CLI), `dci` (DCI MCP server), `dbus` (session bus).
Vertex AI (gcloud) is always included. Default: all features enabled.

Image profiles allow different toolchains per workload. The base image (`ai-assist-sandbox`) has only runtime essentials.
Custom profiles in `sandbox/profiles/` extend it — e.g. `ai-assist-dev` adds Go, Ansible, Python dev tools, shellcheck, and yamllint.

Instance directory layout:
```
~/.ai-assist-instances/my-agent/
  compose.yaml          # host-only: stack definition
  .env                  # host-only: credentials (never mounted)
  sandbox/              # bind-mounted into the ai-assist container
    .ai-assist/         # config, state, logs, KG
    reports/            # output reports
    workspace/          # agent working area
```

MCP servers connect via SSE transport instead of stdio. Configure with `url` in `mcp_servers.yaml`:
```yaml
servers:
  dci:
    url: "http://dci-mcp-server:8001/sse"
```

The instances root directory is configurable via `AI_ASSIST_INSTANCES_DIR` (default: `~/.ai-assist-instances`).

#### MCP Prompts in Tasks

Execute MCP prompts directly from periodic tasks for consistent, automated workflows:

**Natural Language (Traditional):**
```json
{
  "tasks": [
    {
      "name": "System Check",
      "prompt": "Find failures in the last 24 hours",
      "interval": "1h"
    }
  ]
}
```

**MCP Prompts (New):**
```json
{
  "tasks": [
    {
      "name": "Daily RCA Report",
      "prompt": "mcp://dci/rca",
      "prompt_arguments": {
        "days": "1",
        "status": "failure"
      },
      "interval": "8:00 on weekdays"
    }
  ]
}
```

**Format:** `mcp://server_name/prompt_name`

**Benefits:**
- Direct execution - no interpretation needed
- Consistent results - same prompt, same output
- Argument support - pass structured data
- Less token usage - no translation layer

**Creating via Interactive Mode:**
```bash
uv run ai-assist /interactive

# Natural language task
You: Create a task to check for failures every hour

# MCP prompt task (be specific about format)
You: Create a task named "Weekly Report" that runs mcp://tpci/weekly_report
     with argument "for" set to "Semih" at 10:45 on monday
```

**Discovery:**
```bash
uv run ai-assist /interactive

# List all prompts with their arguments
/prompts

# See detailed info about a specific prompt (arguments, descriptions, examples)
/prompt-info tpci/weekly_report
```

See `.ai-assist/event-schedules.json.example` for complete examples.

### One-off Queries

```bash
uv run ai-assist /query "What are the top 5 failing DCI jobs today?"
```

### AWL Workflows

Run multi-step agent workflows using AWL (Agent Workflow Language) scripts:

```bash
uv run ai-assist /run workflow.awl
uv run ai-assist /run workflow.awl project=CILAB days=7
```

Initial variables can be passed as `key=value` arguments and are available as `${key}` in the script.

AWL scripts define intent-driven workflows where the agent autonomously selects tools:

```
@start

@task find_handlers @no-kg
Goal: Find HTTP handlers defined in the repository.
Expose: handlers
@end

@if len(handlers) > 0

@loop handlers as handler limit=5 collect=summaries(handler_summary)

@task inspect_handler @no-history
Goal: Understand what ${handler} does.
Expose: handler_summary
@end

@end

@end

@end
```

Loops support `collect=<var>` to accumulate results across iterations (map-reduce pattern). Use `collect=<var>(<fields>)` to capture only specific fields.

See [docs/AWL_SPECIFICATIONS.md](docs/AWL_SPECIFICATIONS.md) for the full language reference.

Visualize any AWL workflow as an interactive flowchart diagram in the browser:

```bash
uv run ai-assist /awl-viz workflow.awl    # Visualize a specific script
uv run ai-assist /awl-viz                 # Discover and select from available scripts
```

The visualization shows task blocks connected by arrows with variable names on edges, if/else decision diamonds, loop clusters with back-arrows, and mouse-over tooltips with full task details.

### Knowledge Graph

Query temporal data and track changes:

```bash
uv run ai-assist /kg-stats              # View statistics
uv run ai-assist /kg-changes 24         # Changes in last 24 hours
uv run ai-assist /kg-asof '2026-02-04'  # Historical state
uv run ai-assist /kg-show <id>          # Entity details
uv run ai-assist /kg-viz                # Visualize graph in browser
uv run ai-assist /kg-late 60            # Late discoveries (>60 min lag)
```

The knowledge graph automatically stores:
- DCI jobs and components
- Jira tickets and status history
- Entity relationships
- Temporal changes (when created vs. when discovered)
- Conversation exchanges (user/assistant pairs)

**KG Synthesis**: A built-in scheduled task (`kg-synthesis`) runs at 22:00 on weekdays to review the day's conversations, extract structured knowledge (preferences, lessons, context, rationale), and discover connections between entities. The schedule is configurable in `event-schedules.json`.

**Learning Reinforcement**: Synthesized knowledge from the KG is automatically injected into the system prompt:
- **User preferences** are always present so the assistant remembers your communication style and choices
- **Lessons learned, project context, and decision rationale** are injected when semantically relevant to the current query (matched by meaning, not just keywords)

**Auto Context Injection**: When you ask a question, relevant KG entities are automatically surfaced in the system prompt using semantic similarity, giving the assistant immediate context without needing to search.

**Suppressing KG Context**: Prefix any query with `@no-kg` to suppress all KG injection (preferences, learnings, auto-context) for that single query. The model will answer without any prior KG knowledge. KG saving and synthesis remain active.

## Available Tools

### Built-in Tools

**Report Management:**
- `write_report` - Create/overwrite markdown report
- `append_to_report` - Add content to report
- `read_report` - Read report content
- `list_reports` - List all reports
- `delete_report` - Delete report

Reports are stored in `~/ai-reports/` (configurable via `AI_ASSIST_REPORTS_DIR`).

**Action Management:**
- `create_action` - Create periodic or event-driven action
- `list_actions` - List all actions
- `update_action` - Update action properties
- `delete_action` - Remove action
- `enable_action` - Enable/disable action
- `get_action_status` - Get action execution status

Actions stored in `~/.ai-assist/event-event-schedules.json` with hot-reload support.

**Planning:**
- `think` - Planning and reasoning scratchpad for complex multi-step tasks

**JSON Processing** (requires `jq` installed):
- `json_query` - Query or transform JSON files using jq filter expressions

**Filesystem Tools:**
- `read_file` - Read files with line-range support
- `write_file` - Write content to files (creates parent directories)
- `edit_file` - Surgical string replacement in files (exact match, must be unique)
- `search_in_file` - Regex search in files
- `create_directory` - Create directories
- `list_directory` - List directory contents
- `execute_command` - Execute bash commands

### MCP Server Tools

All tools from configured MCP servers are automatically available. Configure servers in `~/.ai-assist/mcp_servers.yaml`:

```yaml
servers:
  dci:
    command: "uvx"
    args: ["--from", "dci-mcp-server", "dci-mcp-server"]
    env:
      DCI_CLIENT_ID: "${DCI_CLIENT_ID}"
      DCI_API_SECRET: "${DCI_API_SECRET}"
```

See `.ai-assist/mcp_servers.yaml.example` for templates.

## State Management

Persistent state stored in `~/.ai-assist/`:

```
~/.ai-assist/
├── state/                   # Monitor states and cache
├── knowledge_graph.db       # Temporal database (SQLite)
├── event-schedules.json          # Monitor/task definitions
├── mcp_servers.yaml        # MCP server configuration
├── allowed_commands.json   # Permanently allowed commands
├── skill_env.json          # Per-skill env var allowlists
└── interactive_history.txt # Command history
```

## Project Structure

```
ai-assist/
├── ai_assist/              # Main package
│   ├── main.py            # CLI entry point
│   ├── agent.py           # MCP agent with tool execution
│   ├── think_tool.py      # Planning/reasoning scratchpad tool
│   ├── monitors.py        # Monitoring tasks
│   ├── state.py           # State management and caching
│   ├── knowledge_graph.py # Temporal knowledge graph
│   ├── awl_ast.py         # AWL AST node definitions
│   ├── awl_parser.py      # AWL workflow language parser
│   ├── awl_expressions.py # AWL expression evaluator
│   ├── awl_runtime.py     # AWL workflow execution engine
│   ├── awl_visualization.py # AWL workflow flowchart visualization
│   └── filesystem_tools.py # Filesystem operations
├── emacs/                  # Emacs major mode for AWL files
├── tests/                  # Test suite
├── .env.example           # Example environment variables
├── VERTEX_AI_SETUP.md     # Vertex AI troubleshooting
├── SECURITY.md            # Security model
├── CONTRIBUTING.md        # Development setup
└── AGENTS.md              # Development philosophy
```

## Documentation

- **[docs/AWL_SPECIFICATIONS.md](docs/AWL_SPECIFICATIONS.md)** - AWL (Agent Workflow Language) specification
- **[docs/PERSONAL_SKILLS.md](docs/PERSONAL_SKILLS.md)** - Creating and managing personal Agent Skills
- **[docs/IDENTITY.md](docs/IDENTITY.md)** - Complete guide to identity.yaml configuration
- **[docs/LOGGING.md](docs/LOGGING.md)** - Logging configuration and troubleshooting
- **[docs/MULTI_INSTANCE.md](docs/MULTI_INSTANCE.md)** - Running multiple ai-assist instances
- **[VERTEX_AI_SETUP.md](VERTEX_AI_SETUP.md)** - Vertex AI configuration and troubleshooting
- **[SECURITY.md](SECURITY.md)** - Security model for skill script execution
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development setup with pre-commit hooks
- **[AGENTS.md](AGENTS.md)** - Development philosophy (TDD/DRY/Tracer Bullet)
- **[presentation/](presentation/)** - Project presentation slides (LaTeX/Beamer)

## Development

### Auto-Reload

Configuration files are automatically watched and reloaded in both monitor and interactive modes:

- **`mcp_servers.yaml`** - Reconnects to changed MCP servers
- **`identity.yaml`** - Updates identity and system prompt
- **`installed-skills.json`** - Reloads Agent Skills
- **`event-schedules.json`** - Restarts monitors and tasks (monitor mode only)

Changes take effect immediately without manual restart.

### Code Auto-Reload (Dev Mode)

Use the `--dev` flag to enable automatic process restart when Python code changes:

```bash
# Monitor mode with code watching
uv run ai-assist --dev /monitor

# Interactive mode with code watching
uv run ai-assist --dev /interactive
```

This is useful during development to see code changes immediately.

## Requirements

- Python 3.12+
- Anthropic API access (Vertex AI or Direct API)
- MCP servers (optional but recommended)
- `jq` (optional, enables `internal__json_query` tool for efficient JSON processing)

## License

See LICENSE file for details.
