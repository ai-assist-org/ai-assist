# Agent Workflow Language (AWL) Specification

## Overview

The **Agent Workflow Language (AWL)** is a lightweight scripting language used to orchestrate AI agent workflows.

AWL focuses on **intent-driven orchestration** rather than explicit tool invocation.

Instead of specifying which tools to call, AWL scripts define:

- goals
- workflow structure
- conditions
- loops
- variable propagation

The **agent remains responsible for selecting tools and strategies** to accomplish each task.

This design keeps AWL:

- human readable
- LLM-friendly
- flexible
- safe

---

# Design Principles

## Intent-driven workflows

Scripts describe **what must be achieved**, not **how to implement it**.

## Agent autonomy

The agent determines:

- which tools to use
- how to search or analyze
- when a task is complete

## Minimal syntax

The language remains simple so that both humans and LLMs can easily produce and edit scripts.

## Deterministic structure

Although the agent reasons dynamically, the workflow structure is deterministic and validated by the runtime.

---

# Script Structure

An AWL script begins with `@start` and ends with `@end`.

Example:

@start

@task find_server
Goal: Find where the HTTP server is initialized.
Success: Identify the file and function responsible for startup.
Expose: server_file, server_function
@end

@end

---

# Directives

AWL defines the following directives.

| Directive | Description |
|---|---|
| @start | start workflow |
| @end | end block |
| @task | define agent task |
| @set | assign variable |
| @if | conditional execution |
| @else | alternate branch |
| @loop | iterate collection |
| @return | return workflow result |
| @goal | autonomous goal pursuit |

---

# Task Hints

Tasks may include **execution hints** placed after the task name.

Hints influence the **context available to the agent**, but do not prescribe specific tools.

Example:

@task fresh_search @no-history @no-kg
Goal: Re-evaluate where the HTTP server is initialized.
Success: Identify the startup path using only repository evidence.
Expose: server_file, server_function
@end

---

# Supported Hints

## @no-history

The agent must ignore prior conversation history and previous reasoning context.

This ensures that the task is evaluated with a **fresh reasoning context**.

Example:

@task fresh_analysis @no-history
Goal: Analyze the repository structure without using previous conclusions.
@end

---

## @no-kg

The agent must not consult the external knowledge graph or knowledge base.

Only live data sources such as repository inspection or runtime tools may be used.

Example:

@task repository_only @no-kg
Goal: Determine how the system initializes using only source code evidence.
@end

---

## max_tool_calls=N and max_time=N

`max_tool_calls` overrides the default tool call budget (100) for a single task.

`max_time` overrides the default wall-clock timeout (600 seconds) for a single task.
This is useful for tasks that involve long-running operations like RCA analysis.

Use this when a task may need many tool calls, such as running a detailed analysis before creating a ticket.

Example:

@task analyze_and_report max_tool_calls=200 max_time=1800
Goal: Run a root cause analysis on the job and create a Jira ticket.
Expose: jira_ticket
@end

---

## model=\<model_name\>

`model` overrides the global model for a single task. Use this when a task needs a
more capable model (e.g., Opus for deep analysis) or a cheaper/faster model
(e.g., Haiku for mechanical data gathering).

The model name is validated against known models at workflow load time. Unknown
models cause the workflow to fail before any task executes.

Example:

@task search_jobs model=claude-haiku-4-5-20251001 @no-kg
Goal: Search DCI for failed jobs in the last 24 hours.
Expose: failed_jobs
@end

@task root_cause model=claude-opus-4-6
Goal: Perform root cause analysis on ${failed_jobs}.
Expose: rca_result
@end

---

## @continue-on-failure

By default, if a top-level task (outside a loop) fails, the workflow **stops immediately** and returns `success: false`. This prevents cascading errors from missing variables.

Tasks inside loops are not affected — loop iterations that fail are skipped, and the loop continues to the next item.

The `@continue-on-failure` hint overrides the default fail-fast behavior, allowing the workflow to continue even if the task fails.

Example:

@task optional_cleanup @continue-on-failure
Goal: Clean up temporary files. Not critical if this fails.
@end

---

# Tasks

Tasks represent **objectives for the agent**.

Syntax:

@task <task_id> [hints...] [max_tool_calls=N] [max_time=N] [model=<model_name>]
...
@end

Example:

@task locate_config
Goal: Find where the server configuration is loaded.
Context: Focus on startup code and configuration parsing.
Success: Identify the entrypoint and parser.
Expose: config_entrypoint, config_parser
@end

---

# Task Fields

Inside a task block the following fields may appear.

| Field | Purpose |
|---|---|
| Goal | task objective |
| Context | additional context |
| Constraints | optional limitations |
| Success | completion criteria |
| Expose | variables produced |

## Shell Commands in Tasks

When a task requires the agent to run a specific shell command, enclose it in backticks:

@task update_repo
Goal: Fetch the latest changes using `git -C /path/to/repo fetch origin master`.
@end

Commands in backticks are **auto-authorized** by the runtime: they are temporarily added to the allowed commands list for the duration of the workflow, so the user is not prompted for confirmation each time.

Commands written without backticks will require manual authorization in interactive mode.

---

# Variables

Variables can be defined using `@set`.

Example:

@set target = "HTTP server"

Variables can be referenced using interpolation:

${target}

Example:

Goal: Find where ${target} is initialized.

## Initial Variables

Variables can be injected before the script runs, without needing `@set`.

**From the CLI:**

```bash
ai-assist /run workflow.awl target="HTTP server" days=7
```

**From an agent prompt:** ask the agent to run a `.awl` file and it will call the
`introspection__execute_awl_script` tool with the variables you specify.

Injected variables are available from the first node, just like `@set` variables.

## Variable Validation

AWL performs static analysis before execution to detect variables that are used
before they are defined.  This catches common mistakes such as:

- Typos in variable names (e.g., `@if failure_count > 0` when only `failed_jobs` was exposed)
- Referencing a `collect` variable inside the loop that produces it (the `collect` variable is only available **after** the loop)
- Using `${var}` in a task goal when no prior task exposed that variable

Validation warnings are reported in three places:

1. **`inspect_awl_script`** — included as `variable_warnings` in the result
2. **`execute_awl_script`** — blocks execution and returns an error
3. **Runtime** — printed to the console before execution starts

### Scoping rules

| Source | Available where |
|---|---|
| `@set` | All subsequent nodes |
| `Expose:` | All subsequent nodes |
| `@loop ... as item` | Inside the loop body |
| `collect=var` | **After** the loop (not inside the body) |
| Initial variables | Everywhere |

---

# Failure Handling

## @fail

Aborts the workflow immediately with an error message.

Syntax:

@fail <message>

Example — abort unconditionally:

@fail Jira server is in maintenance

Example — abort when a task did not produce the expected result:

@if not jira_report
  @fail Jira collection failed - aborting quarterly review
@end

The error is reported to the caller. Combine with `@if`/`@else` for conditional
abort vs. alternative path.

**Agent guidance**: When a task's upstream service is unavailable or returns an
error, do NOT expose the expected variables — leave them unset. The script can
then detect the failure with `@if not <variable>` and abort with `@fail`.

---

# Conditional Execution

Syntax:

@if <expression>
...
@else
...
@end

Example:

@if handlers

@task analyze_handlers
Goal: Understand the detected handlers.
Expose: handler_summaries
@end

@else

@task search_handlers
Goal: Search more broadly for request entry points.
Expose: handlers
@end

@end

---

# Expressions

Expressions are intentionally simple.

Supported operations:

| Expression | Example |
|---|---|
| variable | handlers |
| negation | not report_exists |
| property | config.entrypoint |
| index | handlers[0] |
| length | len(handlers) |
| comparison | len(handlers) > 0 |

---

# Loops

Loops iterate over collections.

Syntax:

@loop <collection> as <item> [limit=N] [collect=<var>[(<fields>)]]
...
@end

The optional `limit=N` parameter caps the number of iterations.

## Collecting Results

The optional `collect=<var>` parameter accumulates exposed variables from each successful iteration into a list.

Example:

@loop handlers as handler limit=5 collect=summaries

@task inspect_handler
Goal: Understand what ${handler} does.
Expose: handler_summary, handler_priority
@end

@end

After the loop, `summaries` contains:

[
  {"handler_summary": "...", "handler_priority": "high"},
  {"handler_summary": "...", "handler_priority": "low"}
]

To collect only specific fields, use parentheses:

@loop handlers as handler collect=summaries(handler_summary)

This produces:

[
  {"handler_summary": "..."},
  {"handler_summary": "..."}
]

Failed iterations are skipped and do not appear in the collected list.

**Important:** The `collect` variable is only available **after** the loop completes.
Referencing it inside the loop body (e.g., `${summaries}` in a task goal) will
trigger a validation error, because the variable has not been populated yet.

## Basic Loop

Example without collect:

@loop handlers as handler limit=5

@task inspect_handler
Goal: Understand what ${handler} does.
Expose: handler_summary
@end

@end

---

# Conditional Loop

`@while` repeats its body while an expression is truthy.

Syntax:

@while <expression> [max_iterations=N]
...
@end

The optional `max_iterations=N` parameter limits the number of iterations (default: 100)
as a safety guard against infinite loops.

Example:

@while running_count > 0 max_iterations=24

@task check_status @no-history @no-kg
Goal: Count running DCI jobs. Do NOT fetch details.
Expose: running_count
@end

@end

Tasks inside `@while` do **not** count against the workflow's `max_steps` limit,
since iteration count is already bounded by `max_iterations`.

---

# Wait

`@wait` pauses execution for a specified duration.

Syntax:

@wait <N>s|m|h

Examples:

@wait 30s
@wait 5m
@wait 1h

Useful for rate-limiting API calls or spacing out polling iterations.

---

# Notify

`@notify` emits a user-facing message with variable interpolation.
No agent call is made — this is a pure runtime operation.

Syntax:

@notify <message>

Example:

@notify ${running_count} jobs still running
@notify All daily DCI jobs have completed

---

# Return

Workflows can return a result.

@return handlers

---

# Polling Example

This workflow polls for running DCI jobs, notifies progress, and exits when done:

@start

@task check_jobs @no-history @no-kg
Goal: Search for daily DCI jobs still running. Count them.
Expose: running_count
@end

@while running_count > 0 max_iterations=24

@notify ${running_count} DCI jobs still running

@wait 1h

@task recheck_jobs @no-history @no-kg @continue-on-failure
Goal: Search for daily DCI jobs still running. Count them.
Expose: running_count
@end

@end

@notify All daily DCI jobs have completed

@end

---

# Example Workflow

@start

@task find_handlers @no-kg
Goal: Find HTTP handlers defined in the repository.
Expose: handlers
@end

@if len(handlers) > 0

@loop handlers as handler limit=5

@task inspect_handler @no-history
Goal: Understand what ${handler} does.
Expose: handler_summary
@end

@end

@else

@task fallback_search
Goal: Search more broadly for request entry points.
Expose: handlers
@end

@end

@return handlers

@end

---

# Runtime Architecture

The runtime contains four main components.

| Component | Responsibility |
|---|---|
| Workflow Engine | executes AWL scripts |
| Agent | performs reasoning |
| Tool Runtime | executes tools safely |
| Memory System | stores workflow state |

Execution pipeline:

AWL Script
↓
Parser
↓
Workflow AST
↓
Runtime Engine
↓
Agent reasoning loop
↓
Tool execution
↓
Memory updates

---

# Task Execution Model

Each task runs an **agent reasoning loop**.

Lifecycle:

task start
↓
build task context
↓
agent reasoning loop
↓
success detection
↓
store exposed variables
↓
task end

---

# Task Context

Before executing a task, the runtime builds a context object.

Example:

{
  "task_id": "find_server",
  "goal": "Find where the HTTP server is initialized.",
  "success": "Identify the file and function responsible for startup.",
  "expose": ["server_file","server_function"],
  "variables": {}
}

Hints such as `@no-history` and `@no-kg` are included in this context.

---

# Task Outcomes

A task may terminate with one of several outcomes.

| Status | Meaning |
|---|---|
| success | goal achieved |
| failed | task attempted but not successful |
| blocked | task cannot proceed due to missing dependency |
| aborted | execution intentionally stopped |

Example success result:

{
  "status": "success",
  "summary": "Server initialized in cmd/server/main.go",
  "exposed": {
    "server_file": "cmd/server/main.go"
  }
}

Example blocked result:

{
  "status": "blocked",
  "reason": "repository_not_loaded",
  "message": "Cannot analyze repository because none is loaded."
}

---

# Runtime Limits

Safety limits enforced by the runtime.

| Limit | Purpose |
|---|---|
| max_steps | maximum reasoning steps |
| max_tool_calls | maximum tool executions |
| timeout | maximum task duration |

Example configuration:

max_steps = 12
max_tool_calls = 20
timeout = 30s

---

# Goals

Goals enable **autonomous agent behavior** by scheduling periodic execution of an AWL body.

Syntax:

@goal <goal_id> [max_actions=<N>]
  Success: <natural language criterion>
  <body>
@end

Parameters:

| Parameter | Required | Description |
|---|---|---|
| max_actions | no | Max tool calls per cycle (default: 5) |

Scheduling is independent from the goal definition. Goals can be run:

- From the CLI: `ai-assist /run goal.awl`
- Via `event-schedules.json`: `{"prompt": "goals/track_failures.awl", "trigger": {"type": "interval", "every": "30m"}}`
- As a one-shot scheduled action

The `Success:` field is mandatory. After each cycle, Claude evaluates whether the criterion is met. When met, the goal status is set to "completed" and scheduling stops.

## Variable Persistence

Variables exposed by tasks inside the goal body persist between cycles via a JSON sidecar file. This allows the goal to track state across executions.

## Example

@start

@goal track_failures max_actions=5
  Success: Failure rate stays below 10%

  @task check_status @no-history
  Goal: Check DCI failure rate for OCP 4.19.
  Expose: failure_rate, failures
  @end

  @if failure_rate > 10
    @task create_alert
    Goal: Write a failure report about ${failures}.
    @end
  @end

@end

@end

## Goal Files

Goal AWL scripts live in `~/.ai-assist/goals/`. The monitor mode scans this directory and schedules active goals. State is persisted in `~/.ai-assist/state/goal_<id>.json`.

## Goal Management

The agent can create, list, and update goals via built-in tools:

- `goal__create` -- generates a `.awl` file
- `goal__list` -- scans the goals directory
- `goal__update` -- changes goal status (pause, resume, cancel)

---

# Summary

The Agent Workflow Language enables structured orchestration of AI agents.

Key properties:

- intent-driven workflows
- agent autonomy
- task hints controlling context
- deterministic execution
- extensible architecture
