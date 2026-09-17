---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

![ai-assist](assets/logo-badge.svg){ .hero-logo width="96" }

# ai-assist

### AI Assistant for Knowledge Workers

An intelligent AI assistant powered by an LLM through the Anthropic Messages
API, Skills and MCP (Model Context Protocol) that helps knowledge workers with
periodic automated monitoring and interactive querying.

[Get Started](getting-started.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/ai-assist-org/ai-assist){ .md-button }

</div>

## Why ai-assist?

<div class="grid cards" markdown>

-   🤖 **AI-Powered**

    ---

    Uses an LLM via the Anthropic Messages API for intelligent analysis, with
    abstract model tiers (low / medium / high) selectable per task.

-   🔌 **MCP Integration**

    ---

    Connect to any MCP server — DCI, Jira, GitHub, Google Docs, Second Brain —
    for tools and live data.

-   💬 **Interactive Mode**

    ---

    A rich terminal UI with streaming responses, markdown rendering, history
    search, and tab completion.

-   📊 **Monitoring**

    ---

    Automated scheduled checks with smart desktop, file, and console
    notifications, plus hot reload.

-   🧠 **Knowledge Graph**

    ---

    A bi-temporal database with vector embeddings tracking entities,
    relationships, and conversation memory.

-   🔄 **AWL Workflows**

    ---

    Script multi-step agent workflows with conditionals, loops, and variable
    propagation in the Agent Workflow Language.

-   🚀 **Agent Skills**

    ---

    Install specialized skills following the
    [agentskills.io](https://agentskills.io/) specification and run them as
    `/<skill> args`.

-   ⚡ **Event-Driven Actions**

    ---

    React to the world with triggers from MQTT, D-Bus, and filesystem changes.

</div>

## Quick start

```bash
# Install uv, then clone and sync
git clone https://github.com/ai-assist-org/ai-assist
cd ai-assist
uv sync

# Configure credentials
cp .env.example .env   # edit with your LLM/MCP credentials

# Run
uv run ai-assist                      # interactive mode (default)
uv run ai-assist /monitor             # monitoring mode
uv run ai-assist /query "..."         # one-off query
uv run ai-assist /run workflow.awl    # execute an AWL workflow
```

See the [Getting Started](getting-started.md) guide for authentication options
and next steps.
