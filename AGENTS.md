# AGENTS.md — Working in ai-assist

Instructions for agents and developers changing this repository. Subsystem
internals live in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); user
documentation in [README.md](README.md). Keep this file short: invariants and
pointers, not prose.

## Commands

| Task | Command |
|---|---|
| All tests (no integration) | `make test` |
| Fast tests (skip slow) | `make test-fast` |
| Coverage (≥71% required) | `make test-cov` |
| One file / one test | `pytest tests/test_x.py::test_y` |
| Integration (podman + images) | `make test-integration` |
| Eval, check judges | `make test-eval` |
| Eval, LLM judges | `make test-eval-full` |
| Eval replay (offline, no key) | `make test-eval-replay` |
| Record eval cassettes | `make eval-record` |
| All hooks | `pre-commit run -a` |
| Sandbox images | `make sandbox-build` / `make sandbox-build-dev` |
| Slides | `make -C presentation` |

## Where to look

| Change | Start in |
|---|---|
| Query loop, tool dispatch, caching | `ai_assist/agent.py` |
| Tool result redirection / pagination | `ai_assist/tool_result_router.py` |
| System prompt and per-query context | `ai_assist/system_prompt_builder.py` |
| Tool visibility | `ai_assist/tool_disclosure.py` |
| AWL | `ai_assist/awl_*.py`, `ai_assist/awl_executor.py` |
| Knowledge graph | `ai_assist/knowledge_graph.py`, `kg_*.py`, `embedding.py` |
| Actions / triggers | `ai_assist/action_*.py`, `event_source*.py` |
| Sandbox | `ai_assist/sandbox.py`, `sandbox/` |
| Skills, plugins | `ai_assist/skills_*.py`, `plugins_*.py` |
| Services | `ai_assist/service.py` |
| CLI / TUI | `ai_assist/main.py`, `ai_assist/tui*.py` |

The full tree is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#code-map).

## Invariants

### Code

- All tests are green before your change. If they fail afterwards, your change
  broke them.
- Write the test first (TDD); see
  [CONTRIBUTING.md](CONTRIBUTING.md#development-philosophy).
- `logger.exception()` in every `except` block — never `logger.debug/warning`
  with the exception, which loses the traceback.
- Expand `~` on every externally supplied path: `Path(p).expanduser()`.
- Type hints everywhere; prefer builtins and `X | None`.
- Configuration is Pydantic in `config.py`; read the environment there, not
  ad hoc.
- No implementation docs or comments unless something is unusual — the code
  should explain itself.

### Prompt cache and cost

- The system prompt must be byte-identical across queries. The clock, recent
  notifications and KG context go on the **current user turn**; breakpoints go
  on a copy (`_with_message_cache_breakpoints`), never on the conversation.
- Context fullness uses `prompt_tokens()` (uncached input + cache write + cache
  read), not the API's `input_tokens`.
- `MODEL_PRICING` in `pricing.py` is manual: update it when prices change.
- Work limits wind a query down, never cut it off: soft turn budget, one final
  turn, hard cutoff at 1.5× the time budget. Context trimming never drops the
  current query's question. A retried stream does not spend turn budget.

### Tools

- MCP tools are named `server_name__tool_name`.
- A proxied `introspection__call_tool` call must be unwrapped before caching,
  loop detection, plan mode and audit — they must see the real tool.
- Every refusal path records an authorization event, or the call is logged as
  `autonomous` (`SECURITY.md` §6).
- Poll-safe tools go in `AI_ASSIST_REPEATABLE_TOOLS` (e.g.
  `apify__get-actor-run`): they bypass the result cache and are never
  loop-flagged.

### Portability and pinning

- CI actions and pre-commit hooks are pinned to a commit SHA with a `# vX.Y.Z`
  comment; container images to a digest. Enforced by
  `scripts/check_action_pins.py`.
- Shell and awk must work on macOS: bash 3.2 (no `mapfile`, associative arrays,
  `${var,,}`), POSIX awk (no `mktime`, `strftime`, `gensub`), BSD userland (no
  `sed -i`, `date -I`, `stat -c`, `grep -P`), and `sha1sum` only with a
  `shasum -a 1` fallback. Enforced by `scripts/check_portability.py`.
- Sandbox profile Dockerfiles must not set `USER` (it breaks
  `userns_mode: keep-id`).

### Security

- Skill script execution is off by default (needs
  `AI_ASSIST_ALLOW_SCRIPT_EXECUTION=true`); per-skill env vars need an
  allowlist.
- Background tasks cannot start other background tasks, and their confirmation
  prompts are auto-denied.
- The sandbox ai-assist container never sees MCP server credentials.

## Subsystems

| Subsystem | One line | Reference |
|---|---|---|
| MCP agent | One streaming loop for `query()` and interactive use; MCP servers via stdio or SSE | [ARCHITECTURE](docs/ARCHITECTURE.md#mcp-agent-flow) |
| AWL | Intent-driven tasks; `finish_task` reports values, `model=`/`effort=` pick tiers | [AWL spec](docs/AWL_SPECIFICATIONS.md) |
| Knowledge graph | Bi-temporal SQLite; synthesis, compaction-time extraction, semantic recall | [KG guide](docs/KNOWLEDGE_MANAGEMENT.md) |
| Skills | agentskills.io; model-invoked and slash commands; sandboxed scripts | [Skills](docs/PERSONAL_SKILLS.md) |
| Plugins | Claude Code bundle subset; namespaced skills and MCP servers | [Plugins](docs/PLUGINS.md) |
| Sandbox | podman-compose instance per service, one credential each | [ARCHITECTURE](docs/ARCHITECTURE.md#sandbox-instances) |
| Actions | trigger + prompt; interval, schedule, once, MQTT, D-Bus, file | [Scheduled actions](docs/SCHEDULED_ACTIONS.md) |
| Background tasks | Long queries in the same event loop; result handed back once | [ARCHITECTURE](docs/ARCHITECTURE.md#background-tasks-interactive-mode) |
| Services | systemd (Linux) / launchd (macOS) for `/monitor` | [ARCHITECTURE](docs/ARCHITECTURE.md#service-management) |
| Tools | Built-ins, MCP tools, progressive disclosure, audit | [ARCHITECTURE](docs/ARCHITECTURE.md#tools-architecture) |

## Adding things

### A feature

1. Write the failing test that defines the behaviour.
2. Implement the smallest end-to-end path, then iterate.
3. Update `README.md` and `AGENTS.md` when user-visible.
4. Update `presentation/ai-assist-presentation.tex`: `\date{...}` to the current
   month and year (`check-presentation-date` enforces it), and the Quick Start
   "Project Stats" box (lines in `ai_assist/`, tests and test files, core
   dependencies, hooks; `tests/test_presentation_stats.py` fails if a figure
   overstates or falls more than 20% behind). Build with `make -C presentation`
   without overfull warnings.
5. `pre-commit run -a`.

### A tool

1. Tool class in the matching `ai_assist/*_tools.py` (`think_tool.py` is the
   pattern).
2. Register in `agent.py`: import, init, tool registration, dispatch chain.
3. Tests in `tests/test_*_tools.py`.
4. Document it in `README.md` if user-facing.

### An MCP server

1. Add it to `~/.ai-assist/mcp_servers.yaml`.
2. Test the connection interactively.
3. Document its tools in `README.md`.

### An AWL feature

1. AST nodes (`awl_ast.py`) → parser (`awl_parser.py`) → runtime
   (`awl_runtime.py`).
2. Tests in `tests/test_awl_*.py`.
3. `docs/AWL_SPECIFICATIONS.md`.

## Documentation

Keep documentation for the end user only. Implementation notes belong in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); do not create new implementation
docs. The code should explain itself — comment only what is unusual.

Index: [README § Documentation](README.md#documentation) ·
[ARCHITECTURE](docs/ARCHITECTURE.md) · [CONTRIBUTING](CONTRIBUTING.md) ·
[SECURITY](SECURITY.md) · [AWL](docs/AWL_SPECIFICATIONS.md) ·
[Plugins](docs/PLUGINS.md) · [Skills](docs/PERSONAL_SKILLS.md) ·
[Knowledge](docs/KNOWLEDGE_MANAGEMENT.md) ·
[Actions](docs/SCHEDULED_ACTIONS.md) · [Identity](docs/IDENTITY.md) ·
[Instances](docs/MULTI_INSTANCE.md) · [Permissions](docs/PERMISSION_MODES.md)

## Before you declare it done

- [ ] The change has tests, and `make test` passes.
- [ ] No duplicated logic (DRY); existing patterns reused.
- [ ] Error handling is graceful and informative; types are annotated.
- [ ] Configuration is centralized.
- [ ] Docs updated where user-visible.
- [ ] `pre-commit run -a` passes.
