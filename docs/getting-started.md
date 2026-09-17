# Getting Started

`ai-assist` integrates with LLMs via the Anthropic Messages API, MCP (Model
Context Protocol) servers, Agent Skills, and a temporal knowledge graph to
provide intelligent monitoring, querying, and workflow automation.

## Prerequisites

- **Python 3.14+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** for
  dependency management
- One authentication method (see [below](#configure-authentication))

## Install

Install `uv` if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install ai-assist:

```bash
git clone https://github.com/ai-assist-org/ai-assist
cd ai-assist
uv sync
```

## Configure authentication

Copy the example environment file and fill in credentials for **one** of the
methods below:

```bash
cp .env.example .env   # then edit .env
```

=== "Vertex AI"

    Enterprise / company Claude access via Google Cloud.

    ```bash
    export ANTHROPIC_VERTEX_PROJECT_ID='your-gcp-project-id'
    gcloud auth application-default login
    ```

    Full walkthrough: see `VERTEX_AI_SETUP.md` in the repository.

=== "Direct API"

    Personal use with a free tier from Anthropic.

    ```bash
    export ANTHROPIC_API_KEY='sk-ant-...'  # from console.anthropic.com
    ```

=== "Custom endpoint"

    OpenRouter or a self-hosted, Anthropic-Messages-compatible endpoint. This
    takes precedence over the other methods.

    ```bash
    # OpenRouter
    export ANTHROPIC_BASE_URL='https://openrouter.ai/api'
    export AI_ASSIST_API_KEY='sk-or-...'
    export AI_ASSIST_MODEL='anthropic/claude-sonnet-4.6'
    ```

## Run

```bash
uv run ai-assist                      # interactive mode (default)
uv run ai-assist /monitor             # start monitoring mode
uv run ai-assist /query "..."         # one-off query
uv run ai-assist /run workflow.awl    # execute an AWL workflow
```

## Next steps

- [Knowledge Quick Start](KNOWLEDGE_QUICK_START.md) — start capturing and
  querying knowledge.
- [Scheduled Actions](SCHEDULED_ACTIONS.md) — set up automated monitoring and
  one-shot future actions.
- [Personal Skills](PERSONAL_SKILLS.md) — install and build agent skills.
- [Plugins](PLUGINS.md) — bundle skills and MCP servers via marketplaces.
- [AWL Specifications](AWL_SPECIFICATIONS.md) — the Agent Workflow Language
  reference.
