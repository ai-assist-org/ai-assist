# LiteLLM proxy — reach non-Anthropic models from ai-assist

ai-assist only ever speaks the Anthropic Messages API (`self.anthropic.messages.stream()` —
no OpenAI SDK, no litellm library, no native Gemini client). Gateways that expose
non-Claude models exclusively through an OpenAI-compatible dialect (e.g. Gemini via
EnMaaS) are unreachable directly, regardless of `.env` settings.

This directory runs a small [LiteLLM proxy](https://github.com/BerriAI/litellm)
that exposes an **Anthropic-compatible `/v1/messages` endpoint** and translates
those requests to whatever backend LiteLLM actually supports — Gemini, OpenAI,
or anything else. ai-assist talks to it exactly like it talks to EnMaaS or
OpenRouter today (`ANTHROPIC_BASE_URL` + `AI_ASSIST_API_KEY`); no ai-assist code
changes needed.

Verified for real, end-to-end, through the actual ai-assist agent (not just
curl):
- **Gemini** (`gemini/` provider, direct to Google): request correctly
  forwarded to Google's Generative Language API.
- **GPT-5** (`openai/` provider, direct to OpenAI): a live ai-assist eval case
  ran GPT-5 through this proxy, correctly called `internal__think`, and
  produced a coherent answer — full round trip confirmed.
- **Gemini via EnMaaS as the backend does NOT work**: EnMaaS only exposes
  Gemini through the OpenAI Chat Completions dialect (`/v1/chat/completions`),
  but LiteLLM's Anthropic-adapter targets the newer Responses API
  (`/v1/responses`) for `openai/`-provider backends, which EnMaaS's Gemini
  models 404 on. Not fixable via this proxy's config (tried `mode: chat` in
  `litellm_params`, no effect) — route straight to the provider's own API
  instead (`gemini/`, `openai/`, etc.), not through another gateway's
  OpenAI-compatible layer.

## Setup

```bash
cd litellm-proxy
cp .env.example .env
# edit .env: set GEMINI_API_KEY (https://aistudio.google.com/apikey) and/or
# OPENAI_API_KEY, and pick your own LITELLM_MASTER_KEY (any string — it's the
# key *this* proxy expects from clients, not a provider credential). Only
# fill in the keys for the providers you actually use.

podman-compose up -d   # or: docker compose up -d
# If port 4000 is already taken (e.g. your own LiteLLM instance), override it:
# LITELLM_PORT=4010 podman-compose up -d
```

Add more models by editing `config.yaml`'s `model_list` and restarting
(`podman-compose restart`).

## Smoke test

```bash
export LITELLM_KEY='sk-choose-your-own-local-key'   # same as LITELLM_MASTER_KEY in .env

curl http://127.0.0.1:4000/v1/messages \
  -H "x-api-key: $LITELLM_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model": "gemini-flash", "max_tokens": 64, "messages": [{"role": "user", "content": "Reply with exactly: ok"}]}'
```

> **Rootless podman note:** if this hangs/resets with `localhost`, use
> `127.0.0.1` explicitly — this environment's rootless podman networking
> (pasta) had an IPv6 (`::1`) forwarding quirk; IPv4 worked immediately.

## Point ai-assist at it

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:4000   # or :4010, whatever LITELLM_PORT you used
AI_ASSIST_API_KEY=sk-choose-your-own-local-key   # your LITELLM_MASTER_KEY
AI_ASSIST_MODEL=gemini-flash                     # or gemini-pro, gpt-5, your own alias

# None of these models are in ai-assist's built-in capability tables, so pin
# real limits (check the provider's docs for the exact model) rather than
# falling back to the conservative unknown-model default (4096 max tokens):
AI_ASSIST_MODEL_MAX_TOKENS=65536
AI_ASSIST_MODEL_CONTEXT_WINDOW=1000000
```

**Reasoning models (GPT-5, etc.) need generous `max_tokens` headroom.** They
spend real output tokens on hidden reasoning (surfaced as a `redacted_thinking`
block) before any visible text — confirmed live: 64 max_tokens produced zero
visible output, 1024 was enough for a one-word reply. Size accordingly.

See `.env.litellm` / `.env.openai` at the repo root for ready-to-symlink
examples (`ln -sf .env.litellm .env`).
