# AGENTS.md - Development Guide for AI Agents and Developers

This document provides guidance for AI agents and developers working on the ai-assist project. It outlines the project architecture, development principles, and best practices.

## Project Overview

**ai-assist** (AI Assistant for Managers) is an intelligent assistant that integrates with Claude, MCP (Model Context Protocol) servers, Agent Skills, and a temporal knowledge graph to provide intelligent monitoring, querying, and workflow automation.

### Core Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ai-assist CLI Interface                    │
│         (Interactive / Monitor / Query modes)            │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                    AiAssistAgent                             │
│  - Manages conversation with Claude                      │
│  - Routes tool calls to MCP servers                      │
│  - Handles multi-turn interactions                       │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼────────┐ ┌───▼────────┐ ┌───▼────────────┐
│ dci-mcp-server │ │   Future   │ │    Future      │
│                │ │   MCP      │ │     MCP        │
│ - DCI Tools    │ │  Servers   │ │   Servers      │
│ - Jira Tools   │ │            │ │                │
│ - Google Docs  │ │            │ │                │
└────────────────┘ └────────────┘ └────────────────┘
```

The system has five major subsystems:

1. **MCP Integration** (`agent.py`): Connects to MCP servers (DCI, Jira, etc.) and routes tool calls using the format `server_name__tool_name`
2. **AWL Runtime** (`awl_parser.py`, `awl_runtime.py`, `awl_ast.py`, `awl_expressions.py`): Agent Workflow Language interpreter for multi-step workflows
3. **Knowledge Graph** (`knowledge_graph.py`): Temporal SQLite database with vector embeddings for entity tracking, change detection, and conversation memory
4. **Agent Skills** (`skills_loader.py`, `skills_manager.py`): Loads agentskills.io-compliant skills with optional sandboxed script execution
5. **Event Sources** (`event_sources.py`, `event_source_mqtt.py`, `event_source_dbus.py`): Pluggable event system for reactive task triggering via MQTT, D-Bus, etc.

Key interaction modes:
- **Interactive** (`tui_interactive.py`): Rich TUI with streaming responses, history, tab completion
- **Monitor** (`monitors.py`, `task_runner.py`): Periodic scheduled tasks with hot-reload
- **Query** (`main.py`): One-off queries
- **AWL** (`awl_runtime.py`): Multi-step workflow execution with conditionals and loops

### MCP Agent Flow
```
User → CLI (main.py) → AiAssistAgent (agent.py) → Claude API
                                ↓
                        MCP Server Sessions
                        (DCI, Jira, GitHub, etc.)
                                ↓
                        Tool Results → Response
```

### AWL (Agent Workflow Language)

AWL scripts define intent-driven workflows where the agent autonomously selects tools to achieve goals. Key constructs:

- `@task <name> @flags`: Defines a goal for the agent to achieve, exposing variables
- `@if <condition>`: Conditional execution with Python-like expressions
- `@loop <var> as <item> limit=N collect=<var>`: Iteration with map-reduce pattern
- `${var}`: Variable interpolation in goals and text
- `@no-kg`, `@no-history`: Flags to suppress knowledge graph context or conversation history

Files: `awl_parser.py` (parser), `awl_runtime.py` (executor), `awl_ast.py` (AST nodes), `awl_expressions.py` (variable interpolation and expression evaluation)

See `docs/AWL_SPECIFICATIONS.md` for complete syntax.

### Knowledge Graph

Temporal database tracking:
- Entities (DCI jobs, Jira tickets, conversations)
- Relationships between entities
- Change history (created_at vs discovered_at for lag detection)
- Vector embeddings for semantic search (`embedding.py`, `context.py`)

**KG Synthesis**: Nightly task extracts structured knowledge (preferences, lessons, context) from conversations and injects relevant context into future queries via semantic similarity.

**Auto Context Injection**: Semantically relevant entities are automatically surfaced in the system prompt based on query similarity.

Files: `knowledge_graph.py`, `kg_queries.py`, `kg_query_tools.py`, `knowledge_tools.py`, `embedding.py`, `context.py`

### Agent Skills

Skills follow the [agentskills.io](https://agentskills.io) specification:
- SKILL.md with YAML frontmatter (name, description, allowed-tools)
- Optional scripts/ directory for sandboxed execution
- Loaded into system prompt automatically

**Script Execution Security** (disabled by default):
- Requires `AI_ASSIST_ALLOW_SCRIPT_EXECUTION=true`
- Sandboxed environment (no API keys, 30s timeout, 20KB output limit)
- Per-skill env var allowlist via `/skill/add_env` (persisted in `~/.ai-assist/skill_env.json`)

Files: `skills_loader.py`, `skills_manager.py`, `script_execution_tools.py`, `security.py`

### Service Management

Cross-platform background service installation for persistent monitoring:
- **Linux**: systemd user services with `journalctl` logging
- **macOS**: launchd user agents with file-based logging (`~/Library/Logs/`)
- Abstract `ServiceBackend` pattern for platform extensibility
- Preserves user environment (PATH, env vars) in service context
- Multiple instance support via config directory naming

Files: `service.py`

### Event-Driven Actions

Unified action system where everything is a **trigger + prompt**. Actions are stored in `~/.ai-assist/event-schedules.json` and managed via agent tools. Trigger types include time-based (interval, schedule, once) and event-based (MQTT, D-Bus).

- **ActionDefinition** (`action_model.py`): unified model replacing `TaskDefinition` + `ScheduledAction`
- **TriggerMatcher** (`action_model.py`): matches incoming events against event-based trigger configs
- **ActionEngine** (`action_engine.py`): executes actions (natural language, MCP prompts, AWL, builtins) with event context
- **ActionScheduler** (`action_scheduler.py`): unified scheduler for timer and event actions
- **ActionLoader** (`action_loader.py`): load/save `event-schedules.json`, auto-migration from old format
- **ActionTools** (`action_tools.py`): agent CRUD tools for actions
- **EventSource** ABC (`event_sources.py`): pluggable event sources (MQTT, D-Bus)
- **MQTT** (`event_source_mqtt.py`): optional dep `aiomqtt` — `pip install ai-assist[mqtt]`
- **D-Bus** (`event_source_dbus.py`): optional dep `dbus-next` — `pip install ai-assist[dbus]`

Configuration in `event-schedules.json`:
```json
{
  "event_sources": {"mqtt": {"broker": "localhost", "port": 1883}},
  "actions": [
    {"name": "DCI Check", "trigger": {"type": "interval", "every": "5m"}, "prompt": "Check DCI failures"},
    {"name": "Alert Handler", "trigger": {"type": "mqtt", "topic": "alerts/#"}, "prompt": "Analyze this alert."}
  ]
}
```

**Example: USB device monitoring via D-Bus**

React to USB filesystem mount/unmount events on Linux:
```json
{
  "event_sources": {
    "dbus": { "bus": "system" }
  },
  "actions": [
    {
      "name": "USB Filesystem Mounted",
      "trigger": {
        "type": "dbus",
        "interface": "org.freedesktop.DBus.Properties",
        "signal": "PropertiesChanged",
        "payload_contains": "MountPoints"
      },
      "prompt": "A USB filesystem was just mounted. Describe the mount point and device.",
      "notify": true,
      "notification_channels": ["desktop", "console", "file"]
    },
    {
      "name": "USB Device Removed",
      "trigger": {
        "type": "dbus",
        "interface": "org.freedesktop.DBus.ObjectManager",
        "signal": "InterfacesRemoved",
        "path": "/org/freedesktop/UDisks2",
        "payload_contains": "Filesystem"
      },
      "prompt": "A USB filesystem was just removed. Describe what happened.",
      "notify": true,
      "notification_channels": ["desktop", "console", "file"]
    }
  ]
}
```

**Example: "Welcome back" briefing on screen unlock (GNOME)**
```json
{
  "actions": [
    {
      "name": "Screen Locked",
      "trigger": {"type": "dbus", "interface": "org.gnome.ScreenSaver", "signal": "ActiveChanged", "bus": "session", "payload_contains": "True"},
      "prompt": "Save the current timestamp to the report screen-lock-time."
    },
    {
      "name": "Welcome Back Briefing",
      "trigger": {"type": "dbus", "interface": "org.gnome.ScreenSaver", "signal": "ActiveChanged", "bus": "session", "payload_contains": "False"},
      "prompt": "The user unlocked their screen. Read screen-lock-time, check recent notifications, summarize what happened.",
      "notify": true, "notification_channels": ["desktop", "console", "file"]
    }
  ]
}
```

D-Bus triggers support per-trigger `bus` field (`system` or `session`) — one D-Bus source connects to both buses as needed. Event-based actions support debouncing (3s window) to collapse burst signals. The `payload_contains` and `payload_regex` fields filter events by their content before execution.

**Context injection**: recent notifications from `/monitor` are automatically injected into the interactive agent's system prompt (last 15 minutes), so the user can ask follow-up questions about events.

Files: `action_model.py`, `action_engine.py`, `action_scheduler.py`, `action_loader.py`, `action_tools.py`, `event_sources.py`, `event_source_mqtt.py`, `event_source_dbus.py`

### Configuration & State

**Configuration sources** (precedence order):
1. Environment variables (`.env` file)
2. `~/.ai-assist/mcp_servers.yaml` (MCP server definitions)
3. `~/.ai-assist/identity.yaml` (personalization)
4. `~/.ai-assist/schedules.json` (monitors/tasks)

**Hot-reload** (`config_watcher.py`, `file_watchdog.py`):
- Changes to config files trigger automatic reload without restart
- Monitor mode restarts tasks when `schedules.json` changes

**State persistence** (`~/.ai-assist/`):
- `knowledge_graph.db` - Temporal database
- `state/` - Monitor states and cache
- `allowed_commands.json` - User-approved shell commands
- `skill_env.json` - Per-skill env var allowlists
- `scheduled-actions.json` - One-time future actions
- `events.jsonl` - Event bridge (monitor→interactive IPC)
- `interactive_history.txt` - Command history

Files: `config.py`, `state.py`, `config_watcher.py`, `file_watchdog.py`

### Tools Architecture

**Built-in tools** are organized by domain:
- `report_tools.py` - Markdown report management (`~/ai-reports/`)
- `schedule_tools.py` - Create/update/delete monitors and tasks
- `schedule_action_tools.py` - One-time scheduled actions with notifications
- `knowledge_tools.py` - KG synthesis and learning
- `kg_query_tools.py` - KG querying (stats, changes, historical state)
- `filesystem_tools.py` - File operations (read, search, execute)
- `think_tool.py` - Planning/reasoning scratchpad
- `json_tools.py` - JSON querying via jq (requires jq installed)
- `introspection_tools.py` - Context awareness (current date/time, working directory, user info)

**MCP tools** from configured servers are dynamically loaded and prefixed with server name.

**Large tool results**: All tools support `__save_to_file`, `__write_to_report`, `__append_to_report`, and `__collect_to_report` parameters. Add to any tool call to redirect raw results to a file or report. Agent receives a summary. `__collect_to_report` auto-paginates and collects all results in one call (requires server pagination config in `mcp_servers.yaml`).

## Documentation

Only create the minimum document for the end user.

No need for implementation docs and comments. The code should be self explanatory.

Use comments only to document something unusual.

## Development Principles

### 1. TDD - Test-Driven Development

**Philosophy**: Write tests before implementation to ensure correctness and maintainability.

#### Testing Strategy for ai-assist

```python
# Example test structure (to be implemented)

# tests/test_agent.py
async def test_agent_connects_to_mcp_servers():
    """Test that agent successfully connects to configured MCP servers"""
    config = AiAssistConfig(
        anthropic_api_key="test-key",
        mcp_servers={"test": MCPServerConfig(command="echo", args=["test"])}
    )
    agent = AiAssistAgent(config)
    await agent.connect_to_servers()
    assert "test" in agent.sessions

async def test_agent_executes_tool_call():
    """Test that agent can execute a tool call and return results"""
    # Mock MCP server response
    # Execute tool
    # Assert correct result format

# tests/test_monitors.py
async def test_jira_monitor_queries_projects():
    """Test that JiraMonitor queries configured projects"""
    # Mock agent
    # Create monitor with test projects
    # Run check
    # Assert correct projects were queried

async def test_dci_monitor_reports_failures():
    """Test that DCIMonitor identifies and reports failures"""
    # Mock agent with failing jobs
    # Run check
    # Assert failures are reported
```

#### TDD Workflow

1. **Red**: Write a failing test that defines desired behavior
2. **Green**: Write minimal code to make the test pass
3. **Refactor**: Clean up code while keeping tests green

#### Baseline Assumption

All tests are green before any change. If tests fail, your change broke them — fix it.

#### Testing Guidelines

- **Unit Tests**: Test individual functions and classes in isolation
- **Integration Tests**: Test MCP server connections and tool execution
- **End-to-End Tests**: Test complete workflows (query → response)
- **Mock External Services**: Use pytest-asyncio and mock for external APIs
- **Test Error Handling**: Ensure graceful degradation when services fail

```bash
# Running tests (when implemented)
pytest tests/
pytest tests/test_agent.py -v
pytest tests/ --cov=ai_assist --cov-report=html
```

Run `make test-cov` to verify coverage (≥71% required for build).

### 2. DRY - Don't Repeat Yourself

**Philosophy**: Avoid code duplication by creating reusable abstractions.

#### DRY Patterns in ai-assist

**✓ Good: Reusable Configuration**
```python
# ai_assist/config.py - Single source of truth for configuration
class AiAssistConfig(BaseModel):
    @classmethod
    def from_env(cls) -> "AiAssistConfig":
        """Load configuration from environment"""
        # Configuration logic in one place
```

**✗ Bad: Repeated Configuration Loading**
```python
# Don't do this in multiple files
api_key = os.getenv("ANTHROPIC_API_KEY")
dci_client = os.getenv("DCI_CLIENT_ID")
```

**✓ Good: Centralized Tool Execution**
```python
# ai_assist/agent.py - Single method for all tool execution
async def _execute_tool(self, tool_name: str, arguments: dict) -> str:
    """Execute any tool on any MCP server"""
    # Parse server and tool
    # Route to correct server
    # Return formatted result
```

**✗ Bad: Repeated Tool Execution Logic**
```python
# Don't implement tool execution separately for each server
async def execute_dci_tool(...):
async def execute_jira_tool(...):
```

#### DRY Checklist

- [ ] Is this logic used in multiple places? → Extract to a function
- [ ] Are these patterns similar? → Create a base class or utility
- [ ] Is this configuration repeated? → Centralize in config.py
- [ ] Are these error messages duplicated? → Use constants or templates
- [ ] Is this query pattern repeated? → Create a query builder

### 3. Tracer Bullet Development

**Philosophy**: Build end-to-end functionality early, then enhance iteratively.

#### Tracer Bullet Approach

A "tracer bullet" is a minimal end-to-end implementation that proves the architecture works. Like tracer ammunition that shows the bullet's path, this approach shows that data flows correctly through the system.

#### ai-assist Tracer Bullets

**Tracer Bullet 1: Basic Query Flow** ✓ Implemented
```
User Input → CLI → Agent → Claude → MCP Tool → Result → User
```

**Tracer Bullet 2: Monitoring Loop** ✓ Implemented
```
Scheduler → Monitor → Agent → Tool Call → Report → Console
```

**Tracer Bullet 3: Multi-Server Integration** (Future)
```
Agent → Server A → Tool 1 → Result
     → Server B → Tool 2 → Result
     → Combine → Response
```

#### Implementing New Features with Tracer Bullets

Example: Adding Slack Notifications

**Phase 1: Tracer Bullet (Minimal End-to-End)**
```python
# ai_assist/notifications.py
class SlackNotifier:
    async def send(self, message: str):
        print(f"[Slack] {message}")  # Console output first

# ai_assist/monitors.py
def _report_results(self, monitor_name: str, results: list[dict]):
    notifier = SlackNotifier()
    await notifier.send(f"{monitor_name}: {len(results)} updates")
```

**Phase 2: Enhancement**
```python
class SlackNotifier:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    async def send(self, message: str):
        async with httpx.AsyncClient() as client:
            await client.post(self.webhook_url, json={"text": message})
```

**Phase 3: Polish**
```python
class SlackNotifier:
    async def send(self, message: str, attachments: list[dict] = None):
        # Rich formatting
        # Error handling
        # Retry logic
        # Rate limiting
```

#### Tracer Bullet Benefits

1. **Early Validation**: Proves architecture works before investing in details
2. **Continuous Integration**: Always have a working system
3. **Risk Reduction**: Identifies integration issues early
4. **Stakeholder Feedback**: Get feedback on working features, not mockups
5. **Momentum**: Visible progress maintains motivation

### 4. Additional Best Practices

#### Minimize Code Branches

**Philosophy**: Keep code paths simple. Avoid fallbacks and conditional branches unless absolutely necessary.

**Why:**
- Fewer branches = fewer bugs
- Easier to test (no need to test multiple paths)
- Clearer code flow
- Faster debugging

**✓ Good: Fail fast with clear error**
```python
from google.cloud import aiplatform

def discover_models():
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)

    # Query for models
    models = aiplatform.Model.list(filter='...')

    return models
```

**✗ Bad: Fallback branches**
```python
def discover_models():
    try:
        from google.cloud import aiplatform
        # Try API approach
        models = aiplatform.Model.list(...)
        return models
    except ImportError:
        # Fallback to hardcoded list
        return test_known_models()
    except Exception:
        # Another fallback
        return default_models()
```

**When to have branches:**
- User-facing choices (interactive prompts, command flags)
- Data-driven logic (different entity types, different formats)
- NOT for: missing dependencies, API failures, "just in case" scenarios

**Handle missing dependencies at install time:**
```python
# In pyproject.toml
dependencies = [
    "anthropic",
    "google-cloud-aiplatform",  # Required, not optional
]
```

**Handle errors clearly:**
```python
try:
    models = aiplatform.Model.list(...)
except Exception as e:
    print(f"Error: Failed to discover models: {e}")
    print("Make sure google-cloud-aiplatform is installed:")
    print("  pip install google-cloud-aiplatform")
    sys.exit(1)
```

#### Path Handling — Always Expand `~`

Any string that could contain a `~` (user input, config files, CLI arguments, tool parameters) **must** have `.expanduser()` called when wrapping it in `Path()`. Without this, `~` is treated as a literal directory name.

```python
# Good
path = Path(user_path).expanduser()

# Bad — ~/foo becomes a literal child directory named "~"
path = Path(user_path)
```

This applies to all code that receives paths from outside the program (config files, tool arguments, CLI args). Internal paths constructed from `Path(__file__)`, `Path.cwd()`, or `get_config_dir()` don't need it.

#### Error Handling

**Graceful Degradation**
```python
async def connect_to_servers(self):
    """Connect to all configured MCP servers"""
    for server_name, server_config in self.config.mcp_servers.items():
        try:
            await self._connect_server(server_name, server_config)
            print(f"✓ Connected to {server_name} MCP server")
        except Exception as e:
            # Don't fail completely - log and continue
            print(f"✗ Failed to connect to {server_name}: {e}")
```

**Informative Error Messages**
```python
if not config.anthropic_api_key:
    print("Error: ANTHROPIC_API_KEY not set in environment")
    print("Please add it to your .env file or export it:")
    print("  export ANTHROPIC_API_KEY='your-key-here'")
    sys.exit(1)
```

**Logging Exceptions**

Always use `logger.exception()` in `except` blocks — never `logger.debug()` or
`logger.warning()` with `%s, e`. `logger.exception()` captures the full
traceback, which is essential for debugging.

```python
# Good
except Exception:
    logger.exception("AWL @notify dispatch failed")

# Bad — loses the traceback
except Exception as e:
    logger.debug("AWL @notify dispatch failed: %s", e)
```

#### Async/Await Patterns

```python
# Use asyncio.gather() for parallel operations
async def check_all_projects(self):
    tasks = [self.check_project(p) for p in self.projects]
    results = await asyncio.gather(*tasks, return_exceptions=True)

# Use async context managers for resources
async def _connect_server(self, name: str, config: MCPServerConfig):
    async with stdio_client(server_params) as (stdio, write):
        session = ClientSession(stdio, write)
        await session.initialize()
```

#### Type Hints

```python
# Always use type hints for better IDE support and documentation
from typing import Optional, List, Dict

async def query(self, prompt: str, max_turns: int = 10) -> str:
    messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
    tool_results: List[Dict] = []
```

#### Configuration Validation

Use Pydantic models (`config.py`) for validation and environment loading with `Field(default_factory=lambda: os.getenv(...))`.

```python
class MyConfig(BaseModel):
    setting: str = Field(default_factory=lambda: os.getenv("MY_SETTING", "default"))
    number: int = Field(default=42, ge=0, le=100)

    @validator("setting")
    def validate_setting(cls, v):
        if not v:
            raise ValueError("Setting cannot be empty")
        return v
```

## Important Patterns

### Tool Execution

All MCP tools use the format `server_name__tool_name`. The agent parses this and routes to the correct session.

### Monitor Pattern

Monitors (`monitors.py`) run periodic checks, cache results, and track seen items to avoid duplicate notifications. They support:
- Interval schedules (e.g., "5m", "1h", "9:00 on weekdays")
- Knowledge graph auto-save
- Notification channels (desktop, file, console)
- MCP prompts (e.g., `mcp://server/prompt`)

```python
async def check(self) -> list[dict]:
    """Standard monitor check pattern"""
    results = []

    for query_item in self.query_items:
        prompt = self._build_prompt(query_item)

        try:
            result = await self.agent.query(prompt)
            results.append({
                "item": query_item,
                "summary": result,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            logger.error(f"Error checking {query_item}: {e}")

    self.last_check = datetime.now()
    return results
```

### Scheduled Actions

One-time future actions (`scheduled_actions.py`) execute via the `/monitor` process with notifications. The agent decides whether to send a simple notification or query via the agent based on the prompt.

Actions older than 7 days are archived to `scheduled-actions-archive.jsonl`.

## Code Organization

```
ai_assist/
├── main.py                    # CLI entry point
├── agent.py                   # MCP agent with tool execution
├── config.py                  # Pydantic configuration models
├── service.py                 # Cross-platform service management (systemd/launchd)
├── awl_*.py                   # AWL parser, runtime, AST, expressions
├── knowledge_graph.py         # Temporal KG database
├── embedding.py, context.py   # Vector embeddings and semantic search
├── skills_*.py                # Agent Skills loader and manager
├── tui*.py                    # Terminal UI components
├── monitors.py, tasks.py      # Monitoring and task execution
├── action_model.py            # Unified action model (trigger + prompt)
├── action_engine.py           # Action execution engine
├── action_scheduler.py        # Unified scheduler (timer + event + one-shot)
├── action_loader.py           # Load/save event-schedules.json
├── action_tools.py            # Agent tools for action CRUD
├── awl_executor.py            # Shared AWL script execution
├── event_sources.py           # Pluggable event source system (ABC, manager)
├── event_source_mqtt.py       # MQTT event source (optional: aiomqtt)
├── event_source_dbus.py       # D-Bus event source (optional: dbus-next)
├── *_tools.py                 # Tool implementations (report, schedule, KG, filesystem, etc.)
├── scheduled_actions.py       # One-time future actions (legacy)
├── notification_*.py          # Notification channels and dispatcher
├── state.py                   # State management and caching
├── *_watcher.py               # File watching and hot-reload
└── security.py                # Sandboxed script execution

tests/                         # Test suite (mirrors ai_assist structure)
docs/                          # Detailed documentation
presentation/                  # LaTeX/Beamer slides (run `make -C presentation`)
emacs/                         # AWL major mode for Emacs
```

## Configuration Files

- `.env` - Environment variables (API keys, model, feature flags)
- `~/.ai-assist/mcp_servers.yaml` - MCP server definitions
- `~/.ai-assist/identity.yaml` - User/assistant personalization
- `~/.ai-assist/event-schedules.json` - Unified actions (trigger + prompt)
- `~/.ai-assist/schedules.json` - Monitors and periodic tasks (legacy, auto-migrated)
- `.pre-commit-config.yaml` - Git hooks configuration
- `pyproject.toml` - Python package config, tool settings, linting rules

## Documentation

Detailed documentation in `docs/`:
- `AWL_SPECIFICATIONS.md` - Complete AWL syntax and semantics
- `PERSONAL_SKILLS.md` - Creating custom Agent Skills
- `IDENTITY.md` - identity.yaml configuration guide
- `MULTI_INSTANCE.md` - Running multiple ai-assist instances
- `KNOWLEDGE_MANAGEMENT.md` - Knowledge graph usage and synthesis
- `SCHEDULED_ACTIONS.md` - One-time future actions

Also see:
- `CONTRIBUTING.md` - Pre-commit setup and workflow
- `SECURITY.md` - Security model for skill scripts
- `VERTEX_AI_SETUP.md` - Vertex AI configuration

## Development Workflow

### Adding a New Feature

1. **Define**: Write a clear specification in an issue or comment
2. **Test First**: Write failing tests that define the behavior (TDD)
3. **Tracer Bullet**: Implement minimal end-to-end functionality
4. **Iterate**: Enhance incrementally while keeping tests green
5. **Refactor**: Apply DRY principle to eliminate duplication
6. **Document**: Update README.md and AGENTS.md
7. **Presentation**: Update `presentation/ai-assist-presentation.tex` to reflect the new feature and verify it builds with `make -C presentation`
8. **Verify**: Run `pre-commit run -a` and ensure all checks pass before declaring the feature complete

### Adding a New Tool

1. Create tool class in appropriate `*_tools.py` file (follow `think_tool.py` pattern)
2. Register in `agent.py`: import, init, tool registration, dispatch chain
3. Write tests in `tests/test_*_tools.py`
4. Update README.md if user-facing
5. Update presentation if appropriate

### Adding a New MCP Server

1. Add server config to `~/.ai-assist/mcp_servers.yaml`
2. Test connection in interactive mode
3. Document available tools in README.md

```python
# Example: Adding GitHub MCP Server
# 1. Add to config
mcp_servers={
    "dci": MCPServerConfig(...),
    "github": MCPServerConfig(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_TOKEN": os.getenv("GITHUB_TOKEN", "")}
    ),
}
```

### Adding a New AWL Feature

1. Update AST nodes in `awl_ast.py` if needed
2. Update parser in `awl_parser.py`
3. Update runtime in `awl_runtime.py`
4. Add tests in `tests/test_awl_*.py`
5. Update `docs/AWL_SPECIFICATIONS.md`

### Adding Platform Support for Service Management

1. Create new class inheriting from `ServiceBackend` in `service.py`
2. Implement all abstract methods: `service_file_path`, `service_content`, `install`, `remove`, `action`, `status`, `logs`
3. Add platform detection in `_get_backend()` factory function
4. Write tests in `tests/test_service.py`
5. Document platform-specific requirements in README.md

## State Management Pattern

ai-assist uses persistent state to avoid redundant queries and maintain context:

```python
# Check cache before querying
cache_key = f"query_{unique_identifier}"
cached = state_manager.get_cached_query(cache_key)

if cached:
    return cached

# Perform query
result = await agent.query(prompt)

# Cache the result with TTL
state_manager.cache_query_result(cache_key, result, ttl_seconds=300)

# Track seen items
state_manager.update_monitor(
    monitor_name,
    results,
    seen_items=set(item_ids)
)

# Append to history
state_manager.append_history(monitor_name, result)
```

## Future Enhancements

### Potential Features (Apply Tracer Bullet Approach)

1. **Multi-Channel Notifications**
   - Tracer: Console output
   - Enhance: Email support
   - Polish: Slack, Teams, webhooks

2. **Report Generation**
   - Tracer: Print to console
   - Enhance: Save to markdown file
   - Polish: Generate Google Docs, PDFs

3. **Intelligent Alerting**
   - Tracer: Alert on any failure
   - Enhance: Alert on patterns (3+ failures)
   - Polish: ML-based anomaly detection

4. **Dashboard**
   - Tracer: CLI status command
   - Enhance: Static HTML page
   - Polish: Live web dashboard

## Resources

- **MCP Documentation**: https://modelcontextprotocol.io/
- **mcp-agent**: https://github.com/anthropics/mcp-agent-python
- **Anthropic API**: https://docs.anthropic.com/
- **DCI Documentation**: https://docs.distributed-ci.io/

## Contributing

When contributing to ai-assist:

1. Follow TDD: Write tests first
2. Apply DRY: Avoid duplication
3. Use Tracer Bullets: Start simple, enhance iteratively
4. Document: Update AGENTS.md and README.md
5. Type Hints: Always use type annotations
6. Error Handling: Graceful degradation, informative messages

## Questions for AI Agents

When working on ai-assist, ask yourself:

- [ ] Does this change have tests? (TDD)
- [ ] Am I repeating existing code? (DRY)
- [ ] Can I implement this in phases? (Tracer Bullet)
- [ ] Is the error handling graceful?
- [ ] Are types properly annotated?
- [ ] Is the configuration centralized?
- [ ] Does it follow existing patterns?
- [ ] Is it documented?
- [ ] Have I run `pre-commit run -a` and fixed all issues?

---

**Remember**: The goal is to build a reliable, maintainable AI assistant. Take the time to do it right, and the system will be easier to enhance and debug over time.
