# Architecture

How ai-assist is put together: the query loop, each subsystem's implementation
notes, and where things live. User-facing behaviour is documented in
[README](../README.md) and the guides in this directory; this page is the map
for people (and agents) changing the code.

## Components

```
┌─────────────────────────────────────────────────────────┐
│                    ai-assist CLI                        │
│         (interactive / monitor / query modes)           │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                    AiAssistAgent                        │
│  - conversation with the model                          │
│  - routes tool calls to MCP servers                     │
│  - multi-turn interactions                              │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼────────┐ ┌────▼───────┐ ┌─────▼──────────┐
│ MCP servers    │ │  internal  │ │   future MCP   │
│ (dci, jira, …) │ │   tools    │ │    servers     │
└────────────────┘ └────────────┘ └────────────────┘
```

Five subsystems:

1. **MCP integration** (`agent.py`) — connects to MCP servers and routes tool
   calls named `server_name__tool_name`.
2. **AWL runtime** (`awl_parser.py`, `awl_runtime.py`, `awl_ast.py`,
   `awl_expressions.py`) — the Agent Workflow Language interpreter.
3. **Knowledge graph** (`knowledge_graph.py`) — temporal SQLite database with
   vector embeddings for entities, change detection and memory.
4. **Agent Skills** (`skills_loader.py`, `skills_manager.py`) — agentskills.io
   skills with optional sandboxed scripts.
5. **Event sources** (`event_sources.py`, `event_source_mqtt.py`,
   `event_source_dbus.py`, `event_source_file.py`) — reactive task triggering.

Interaction modes: **interactive** (`tui_interactive.py`), **plan mode**
(`plan_mode.py`, `/plan`), **monitor** (`monitors.py`, `task_runner.py`),
**one-off query** (`main.py`), **AWL** (`awl_runtime.py`).

## MCP agent flow

```
User → CLI (main.py) → AiAssistAgent (agent.py) → model API
                                ↓
                        MCP server sessions
                        (dci, jira, github, …)
                                ↓
                        tool results → response
```

Both entry points use the streaming agent loop; `query()` collects the final
answer from `query_streaming()`, so execution logic lives in the shared
streaming path.

- **Time budget**: both entry points default to 600 s
  (`AI_ASSIST_QUERY_TIMEOUT`); `max_time_seconds=NO_TIME_LIMIT` (AWL tasks
  without `max_time=`, scheduled actions) removes it. Time spent waiting on an
  approval prompt (`pause_timeout()`) does not count.
- **Turn budget**: soft. At `max_turns` the model is told to wrap up and gets up
  to 50% more turns. When that is exhausted, or the time budget passes (checked
  between turns), the model gets one final turn, offered only
  `internal__finish_task` in an AWL task and no tools otherwise.
- **Hard cutoff**: a backstop at 1.5× the time budget
  (`_HARD_TIME_LIMIT_FACTOR`).
- **Model API failures**: `query()` returns error text unless the caller passes
  `raise_on_api_error=True` (AWL tasks and scheduled actions do, also via
  `execute_mcp_prompt`), which raises `QueryAPIError`. `raise_on_incomplete=True`
  (AWL tasks) raises `QueryIncompleteError` when a query stops without a final
  answer. Both live in `query_errors.py`.
- **Stream retry**: a turn whose stream drops mid-reply (transport/SSL/
  connection-reset) is retried up to twice (`_STREAM_RETRY_BACKOFF_SECONDS`)
  without consuming the turn budget, but only if the turn hasn't added to the
  conversation yet.
- **Context trimming**: when messages exceed `AI_ASSIST_TOTAL_MESSAGES_PCT` of
  the context window, `_trim_oldest_messages` drops whole oldest exchanges (the
  list keeps starting on a user text turn) and never the current query's
  question or its turns. Each drop is logged.
- **Enclosing queries**: an AWL script run from a tool call does not count
  against the enclosing query (`exempt_enclosing_queries_from_timeout()`); the
  script bounds itself per task, and its tasks keep their own budgets.

## AWL

`docs/AWL_SPECIFICATIONS.md` is the language reference. Implementation notes
that are not in the spec:

- **Task outcome**: while a task runs, `_build_api_tools` offers the agent
  `internal__finish_task` (`task_result_tool.py`). It reports
  `status="success"` with the exposed `values`, or `status="failed"` with a
  reason, which fails the task through the normal failure path. A turn that ends
  without calling it gets one nudge (`_finish_nudge_fired`); after that, values
  are extracted from the reply and missing ones stay `None` — the expose nudge
  must not guess. Limitation: a nested non-AWL query made during a task also
  sees the tool.
- **Model tiers**: `model=low|medium|high|default` resolves via
  `resolve_tier` (`awl_runtime.py`) using `AI_ASSIST_MODEL_LOW`/`_MEDIUM`/`_HIGH`,
  falling back to the session model. `default` and `high` fall back to each
  other. Effort resolves through `AiAssistAgent._model_for` / `_effort_for` as
  `effort=` override → the model var's `:effort` suffix → session effort;
  `no` sends none.
- **Auto-authorized commands**: commands written in backticks are authorized
  without a prompt only for scripts the user started: `/run`, an action whose
  prompt is the script, or — via `is_trusted_by_action` (`awl_executor.py`) — a
  script named by the confirmed prompt of the running action. When such a named
  script fails or cannot acquire its execution lock, the action is reported as
  failed even if the agent carries on.
- **Failure propagation**: both `@start` and `@goal` execution check
  `WorkflowResult.success`; a failed cycle cannot complete a goal.
- **Audit attribution**: the audit log attributes each tool call to its script
  and task (`AuditLogger.task_context`, entered by `AWLRuntime`).

## Knowledge graph

Bi-temporal SQLite database tracking entities (DCI jobs, Jira tickets,
conversations, knowledge), relationships, valid time (when a fact is true in
reality), transaction time (when the agent learned it), and vector embeddings
for semantic search (`embedding.py`, `context.py`).

- **Temporal tools**: `kg_snapshot` (point-in-time), `search_knowledge` with
  `since`, `expire_knowledge`.
- **KG synthesis**: a nightly task snapshots new/changed reports and discovers
  connections between entities. It does not extract knowledge from
  conversations — compaction-time extraction does (below).
- **Compaction-time extraction**: in interactive mode, durable facts the user
  stated are extracted when the conversation is compacted, cleared (`/clear`),
  or the session exits (synchronously on exit), as a background task.
  Extraction reuses an existing `project_context` key when a new fact is a
  near-duplicate (semantic similarity ≥ `AiAssistAgent.DEDUP_SIM_THRESHOLD`), so
  the KG's last-write-wins upsert supersedes it instead of accreting parallel
  entries. The threshold is deliberately conservative: a false merge overwrites
  a distinct fact irrecoverably.
- **Auto context injection**: semantically relevant entities are surfaced in
  the per-query context based on query similarity. Bookkeeping types in
  `KnowledgeGraph.NON_EMBEDDED_TYPES` (`tool_result`, `skill_performance`) are
  never embedded, so they stay out of recall.
- **Skill usage**: every skill invocation (model-invoked via
  `introspection__get_skill_help`, or manual `/<skill-name>`) is counted as a
  `skill_performance` entity (invocation count + last used); `/skill/stats`
  shows it. Recording is best-effort and never blocks the run. It is
  invocation-only (no outcome/confidence) by design; `skill_performance.py`
  keeps the confidence machinery for a future outcome-scoring step.

Files: `knowledge_graph.py`, `kg_queries.py`, `kg_query_tools.py`,
`knowledge_tools.py`, `embedding.py`, `context.py`.

## Agent Skills and plugins

**Skills** follow the agentskills.io spec. Loaded into the system prompt
(model-invoked) and user-invocable as `/<skill-name> [args]`; the SKILL.md body
runs as a query with `$ARGUMENTS`/`$1` substitution (quoted segments stay
grouped). Opt out with `user-invocable: false`. Built-in commands take
precedence over same-named skills. Implementation: `handle_skill_invocation`
(`tui_interactive.py`).

Script execution is disabled by default: it requires
`AI_ASSIST_ALLOW_SCRIPT_EXECUTION=true`, runs sandboxed (no API keys, 30 s
timeout, 20 KB output limit), and per-skill env vars need an allowlist via
`/skill/add_env` (persisted in `~/.ai-assist/skill_env.json`). Files:
`skills_loader.py`, `skills_manager.py`, `script_execution_tools.py`,
`security.py`, `skill_performance.py`.

**Plugins** load `.claude-plugin/plugin.json` + `skills/*/SKILL.md` + legacy
`commands/*.md` + `.mcp.json`.

- Plugin skills/commands are registered in the shared
  `skills_manager.loaded_skills` dict under **namespaced keys** (`plugin:skill`),
  so every consumer (system prompt, completion, `/help`, invocation) works
  unchanged. Namespacing lives at the registry-key layer only, never in
  `SkillMetadata.name`.
- Bundled MCP servers are namespaced (`plugin__server`) and merged into
  `config.mcp_servers`; kept in `plugins_manager.plugin_mcp_servers` and
  re-merged after any `reload_mcp_servers()` / skills reload.
- `agents/` (subagents) and `hooks/` are not supported — skipped with a notice.
- Commands: `/plugin/install|uninstall|update|list|search`,
  `/plugin/marketplace add [nickname]|update [name]|list`. `update` reinstalls
  from the recorded source (rollback on failure); marketplace `add` refuses to
  mask a different registration and accepts a nickname. A plugin offered by
  several marketplaces is ambiguous: install errors with the candidates;
  disambiguate with `<name>@<marketplace>`. Installing bundled MCP servers
  prompts before connecting.
- The agent gets read-only introspection (`introspection__list_skills` /
  `list_plugins`); install/update/uninstall stay human-driven slash commands.

Files: `plugins_loader.py`, `plugins_manager.py`, `skills_manager.py`.
User docs: `docs/PLUGINS.md`.

## Service management

Cross-platform background service installation for persistent monitoring:
systemd user services with `journalctl` on Linux, launchd user agents with
file logging (`~/Library/Logs/`) on macOS. `ServiceBackend` is the abstract
platform interface; `_get_backend()` detects the platform. Services preserve the
user environment (PATH, env vars) and support multiple instances via config
directory naming. File: `service.py`.

Adding a backend: subclass `ServiceBackend` in `service.py`, implement
`service_file_path`, `service_content`, `install`, `remove`, `action`, `status`
and `logs`, add platform detection in `_get_backend()`, and cover it in
`tests/test_service.py`.

## Sandbox instances

The experimental `egress-proxy` feature makes the shared Compose bridge
internal and drops all capabilities from the agent with no-new-privileges.
The generator adds a mitmproxy service on both the internal bridge and a
separate upstream network. HTTP clients use CONNECT, with internal MCP and
LiteLLM service names in the proxy exclusion list. The CONNECT authority,
SNI and HTTP authority must agree. Workloads receive the public CA only.
Allowlists, CA private keys and egress audit records live outside the workspace.
Each workload joins the namespace of a separate rootless network guard.
Only that guard has NET_ADMIN there. Its nftables queues inspect outbound
DNS and new connections. Direct packets to unauthorized destinations reach
a dummy interface route, are audited, and are rejected in the output hook.
If the queue consumer is unavailable the kernel drops queued packets.
Internal DNS aliases for allowed hosts point direct HTTPS clients to a
second mitmproxy listener. Both listeners enforce the same SNI and HTTP policy.

The identifier check was replayed against all 63 public dcibot issue comments
on redhatci/ansible-collection-redhatci-ocp from 25 November 2026 06:19 UTC
through 26 November 18:33 UTC. The 26 even-ID comments included the one
reported incident, detected as `internal-ticket`. The 37 odd-ID holdouts
produced zero alerts (zero false positives against the incident report).
Four held-back PR descriptions by the same author (#1143–1146) also passed.
Run `python tests/replay_egress_corpus.py` with GitHub CLI access to replay
the labels. The fixture stores comment IDs and hashes, so changed or missing
published bodies fail the replay.
Reversed and base64-transformed secrets are outside the detector's coverage.

Isolated instances via podman-compose with per-service credential isolation:
each MCP server runs in its own container and only receives its own secrets.
The ai-assist container never sees MCP server credentials.

- **SSE transport**: MCP servers connect via SSE (`MCPServerConfig.url`)
  instead of stdio; `_transport()` in `agent.py` selects it.
- **Instance directory**: `$AI_ASSIST_INSTANCES_DIR/<name>/` with host-only
  `compose.yaml`/`.env` and a bind-mounted `sandbox/` directory.
- **Readiness**: `_wait_for_mcp_servers()` polls TCP ports before launching.
- **CLI**: `ai-assist /sandbox init|run|stop|list|delete|service`.
- **Features**: `--features=ssh,gpg,git,gh,dci,dbus,litellm,web` (default
  `git,dci,vertex`; `web` enables `internal__web_search`/`web_fetch`, off
  otherwise so an autonomous agent cannot send private work to search
  providers). Vertex AI (gcloud) is always included.
- **`litellm` feature**: a sidecar fronting non-Anthropic models (Gemini,
  OpenAI) behind the Anthropic Messages dialect; auto-wires
  `ANTHROPIC_BASE_URL`/`AI_ASSIST_API_KEY` so the container only sees
  `LITELLM_MASTER_KEY`. Aliases live in
  `sandbox/.ai-assist/litellm_config.yaml` (copied from
  `sandbox_templates/litellm_config.yaml`); see `litellm-proxy/README.md`.
- **`dci` feature downloads**: both containers mount `sandbox/dci-files` at
  `/tmp/dci`, the server's user (1001) is mapped to the host user
  (`userns_mode: keep-id:uid=1001,gid=1001`), `/tmp/dci` is an allowed path,
  and `DCI_DOWNLOAD_DIR_SHARED=true` lets the server pre-download at render
  time in SSE mode.
- **Timezone**: `--timezone=` sets `TZ` on every container (default: the host's;
  `--timezone=` keeps the image's UTC). It decides when scheduled actions fire
  (the scheduler compares against a naive `datetime.now()` and ignores a
  trigger's own `timezone`) and keeps one clock across shared files and logs.
- **Image profiles**: `--image=ai-assist-dev`; profile Dockerfiles in
  `sandbox/profiles/` extend the base image and must not set `USER` (it breaks
  `userns_mode: keep-id`).
- **Service mode**: `ai-assist /sandbox service <name> install|remove|start|stop|status|logs`
  wraps the compose stack as a systemd user service running `/monitor`; boot
  startup needs `loginctl enable-linger $USER`.
- **No pods**: `--in-pod=false`, so each container has its own user namespace.
  The ai-assist container uses `userns_mode: keep-id`; MCP server containers
  run as their default user.
- **Read-only mounts**: `sandbox_mounts.py` implements
  `/sandbox mount add|remove|list`, binding existing host directories read-only
  and updating allowed paths. Sources with writable aliases are refused. Git,
  credentials, content updates and workflow locks belong to the orchestration
  layer. Changing mounts requires container recreation.

The DCI MCP server image is built from upstream `Containerfile.sse`; the
ai-assist image from `sandbox/ai-assist/Dockerfile`. The `--timezone` default
comes from the host's `/etc/localtime`.

Testing: `tests/test_sandbox.py` mocks subprocess (default suite).
`tests/test_sandbox_integration.py` starts real containers and is run with
`make test-integration` before merging sandbox changes.

Files: `sandbox.py`, `sandbox_templates/`, `config.py` (`MCPServerConfig.url`).

## Event-driven actions

Everything is a **trigger + prompt** stored in
`~/.ai-assist/event-schedules.json` and managed via agent tools.

- **ActionDefinition** (`action_model.py`) — unified model replacing
  `TaskDefinition` + `ScheduledAction`.
- **TriggerMatcher** (`action_model.py`) — matches incoming events.
- **ActionEngine** (`action_engine.py`) — executes actions (natural language,
  MCP prompts, AWL, builtins) with event context.
- **ActionScheduler** (`action_scheduler.py`) — unified scheduler for timer,
  event and one-shot actions. Edits hot-reload it: `FileWatchdog` collapses a
  burst of writes into one callback (debounce state on the loop thread; a
  running callback is never cancelled), and `reload()` is serialized by a lock,
  since overlapping reloads would leave untracked duplicate timers.
- **ActionLoader** (`action_loader.py`) — load/save, auto-migration.
- **ActionTools** (`action_tools.py`) — agent CRUD tools.
- **EventSource** ABC (`event_sources.py`) — MQTT (`ai-assist[mqtt]`), D-Bus
  (`ai-assist[dbus]`), file/glob (`watchdog`, no extra deps). MQTT pulls in
  `aiomqtt`, D-Bus in `dbus-next`.

Configuration shape and examples are in
[README § Event-Driven Actions](../README.md#event-driven-actions) and
[docs/SCHEDULED_ACTIONS.md](SCHEDULED_ACTIONS.md). Notes for implementers:

- One-time actions created with `internal__schedule_action` (or
  `internal__create_action`) run once, then move to
  `event-schedules-archive.jsonl` with `completed`/`failed` status, matched by
  name and time. Each (re)load also archives finished ones and those past due
  beyond the 24 h catch-up window (`expired`) — see
  `ActionLoader.archive_actions`.
- A `"reminder": true` action notifies with the prompt text and makes no model
  call.
- Event-based actions debounce bursts (3 s); `payload_contains` /
  `payload_regex` filter before execution.
- Recent notifications from `/monitor` (last 15 min) are added to the
  interactive agent's per-query context, so follow-up questions work.

## Background tasks (interactive mode)

Long-running queries run as background `asyncio` tasks in the same event loop;
the user keeps chatting. A finished task's result is handed back once
(`BackgroundTaskManager.take_finished_note()`): an idle, empty prompt is ended
so the agent reports at once (`wake_idle_prompt` in `tui_interactive.py`),
otherwise the result goes out with the next message. `NotificationWatcher` still
shows completion to the user.

- `background_tasks.py` — spawn/track/cancel, in-memory only.
- `background_task_tools.py` — `internal__run_background`,
  `internal__list_background_tasks`, `internal__get_background_task`,
  `internal__cancel_background_task`.
- **Isolation**: prompts are prefixed `@no-kg @no-history` plus a note that the
  answer goes back to the main conversation and nobody can answer questions.
  Shared MCP sessions are reused. A background task cannot start another.
- **Security**: `in_background_task()` (a `ContextVar` set in the task's own
  asyncio task) auto-denies the confirmation prompts of background tasks only;
  foreground prompts are still asked.
- **Not handed back**: cancelled tasks, and tasks spawned with
  `notify_agent=False` (MCP channel replies, which answer through their own
  tool).

## Configuration, state and cost

Configuration precedence: environment (`.env`) → `~/.ai-assist/mcp_servers.yaml`
→ `identity.yaml` → `variables.yaml` (AWL locals, auto-injected) →
`event-schedules.json`.

Hot-reload (`config_watcher.py`, `file_watchdog.py`): config changes reload
without restart; monitor mode restarts actions when `event-schedules.json`
changes.

`setup_logging()` logs the source commit of the ai-assist checkout (with
"(uncommitted changes)" when tracked files changed), or nothing when it does not
run from its own git checkout (`get_source_commit()` in `config.py`).

State in `~/.ai-assist/`: `knowledge_graph.db`, `state/`,
`allowed_commands.json`, `skill_env.json`, `audit/tool_calls.jsonl`,
`traces/query_traces.jsonl`, `traces/requests/`, `events.jsonl`,
`interactive_history.txt`. The audit trail is append-only, rotates at 10 MB, and
keeps rotated files 7 days; query traces are kept 100 days; captured requests
only exist with `AI_ASSIST_CAPTURE_MODEL_REQUESTS=true` (mode 0600, never
pruned).

**Cost tracking** (`pricing.py`, `eval.py`):

- Prices are not available from the API. `MODEL_PRICING` is hardcoded and must
  be updated when rates change
  (https://docs.anthropic.com/en/docs/about-claude/models).
- `_track_token_usage()` records cost per turn; `QueryTrace.total_cost_usd`
  aggregates per query. `/cost [period]` and `/eval-stats` recompute each
  query's cost from per-turn counts (`QueryTrace.cost_usd()`), so old traces are
  reported correctly; the stored `total_cost_usd` is used only when per-turn
  counts are missing or the model has no price.
- The API's `input_tokens` excludes cached tokens: `compute_turn_cost` bills
  uncached input, cache writes, cache reads and output separately. Anything
  measuring context fullness (observation masking, the budget warning,
  `get_context_usage`) uses `prompt_tokens()`, the sum of the three input
  counts.

**Prompt caching** (`AI_ASSIST_ENABLE_CACHE`, on by default): the static system
block and the last two user messages of each request are marked
(`_with_message_cache_breakpoints` in `agent.py`); the markers go on a copy, the
conversation is never modified. The system prompt must stay byte-identical
across queries: the clock, recent notifications and KG learnings/auto-context
are appended to the current user turn as a separate block
(`_build_query_context`, `_with_query_context`), and the breakpoint goes on the
user's own text before it, so the next query hits the cache at the same
position. Each turn's token entry records `prefix_sha` (tools + system);
`workflows/self-improvement/lib/audit-cache.py` reports misses.

**MLflow** (`mlflow_tracing.py`): optional (`pip install ai-assist[mlflow]`,
`mlflow-skinny` + `sqlalchemy`), enabled with `AI_ASSIST_ENABLE_MLFLOW=true`.
`MLFLOW_TRACKING_URI` selects the backend; unset means a local SQLite store at
`<config dir>/mlflow.db`. Emits a `query → tool` span tree and mirrors
`QueryTrace` metrics. No-op when disabled. The JSONL cost tracking is
unaffected.

## Tools architecture

**Built-in tools** by domain: `report_tools.py`, `action_tools.py`,
`schedule_action_tools.py`, `knowledge_tools.py`, `kg_query_tools.py`,
`filesystem_tools.py` (`execute_command` reports per-stage `PIPESTATUS` and saves
oversized output in the session tmpdir), `think_tool.py`, `json_tools.py`,
`transform_tools.py`, `introspection_tools.py`. MCP tools are prefixed with the
server name.

**Progressive disclosure** (`tool_disclosure.py`, `AI_ASSIST_DEFER_TOOLS`, on by
default): only `CORE_TOOLS` are sent in full; every other tool, MCP or internal,
is listed in the description of `introspection__call_tool`, read with
`introspection__get_tool_help` and called through that proxy. The tool list
never changes, so the prompt cache and thinking signatures stay valid. Choose
`CORE_TOOLS` from audit trails, not by guess. Skills work the same way: only
their front matter is in the system prompt. A proxied call is validated against
the tool's schema (`check_arguments`, strict) before it runs; an invalid call is
not executed and the error carries the tool's documentation. `last_tool_calls`
records what ran this query.

**Agent collaborators**: `tool_result_router.py` (`ToolResultRouter`, result
redirection and pagination), `synthesis_engine.py` (`SynthesisEngine`,
synthesis, `_save_insights`, KG connection discovery),
`system_prompt_builder.py` (`SystemPromptBuilder`, system prompt and per-query
context). Each holds an agent back-reference and reads collaborators live;
`AiAssistAgent` keeps thin delegating wrappers for the names tests and other
modules use.

**Result redirection**: every tool accepts `__save_to_file`,
`__write_to_report`, `__append_to_report`, `__collect_to_report` (auto-paginate
and collect, needs server pagination config) and `__jq_filter` (runs first).

**Dedup and loop detection** (`_execute_tools_concurrently` in `agent.py`):
identical calls (name + args) within a query are served from a result cache; a
signature repeated ≥3× in the last 5 triggers loop detection (first offense
nudges, a second aborts). **Poll-safe tools** — async status checks expected to
be called repeatedly — are exempt: list fnmatch patterns in
`AI_ASSIST_REPEATABLE_TOOLS` (whitespace/comma separated,
`config.repeatable_tools`), matched by `AiAssistAgent._is_repeatable_tool`. A
repeatable tool bypasses the cache and is never loop-flagged.

**Audit**: each `_execute_tool` call runs inside `audited_call()` (`audit.py`, a
`ContextVar`, so concurrent calls stay separate), which assigns the
`correlation_id` and collects authorization events (`record_authorization_event`
— approval prompts via `FilesystemTools._record_confirmation`, policy refusals
of paths, commands and protected configs, plan mode) and lets one record through
per call. See `SECURITY.md` §6 for the record format, rotation and
`AI_ASSIST_AUDIT_SINK`.

## Code map

```
ai_assist/
├── main.py                    # CLI entry point
├── agent.py                   # MCP agent with tool execution
├── tool_disclosure.py         # Core tools sent, others proxied
├── tool_result_router.py      # Result redirection/pagination
├── synthesis_engine.py        # Conversation synthesis + KG connections
├── system_prompt_builder.py   # System prompt + KG context assembly
├── config.py                  # Pydantic configuration models
├── service.py                 # systemd/launchd service management
├── awl_*.py                   # AWL parser, runtime, AST, expressions
├── knowledge_graph.py         # Temporal KG database
├── embedding.py, context.py   # Vector embeddings and semantic search
├── skills_*.py                # Agent Skills loader and manager
├── plugins_*.py               # Claude Code plugin loader and manager
├── plan_mode.py               # Plan mode (explore → approve → execute)
├── tui*.py                    # Terminal UI components
├── monitors.py, tasks.py      # Monitoring and task execution
├── action_*.py                # Action model, engine, scheduler, loader, tools
├── awl_executor.py            # Shared AWL script execution
├── event_sources.py           # Pluggable event source system
├── event_source_*.py          # MQTT, D-Bus, file sources
├── background_tasks.py        # Background task manager
├── background_task_tools.py   # Background task tools
├── *_tools.py                 # Tool implementations
├── transform_harness.py       # Child process of internal__transform
├── notification_*.py          # Notification channels and dispatcher
├── state.py                   # State management and caching
├── *_watcher.py               # File watching and hot-reload
└── security.py                # Sandboxed script execution

tests/                         # Test suite (mirrors ai_assist structure)
docs/                          # Detailed documentation
workflows/                     # Shipped AWL workflows; helpers in <workflow>/lib/
presentation/                  # LaTeX/Beamer slides (make -C presentation)
emacs/                         # AWL major mode for Emacs
```
