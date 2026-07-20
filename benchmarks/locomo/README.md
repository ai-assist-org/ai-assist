# LoCoMo Benchmark for ai-assist Knowledge Graph

Evaluates ai-assist's KnowledgeGraph against the [LoCoMo benchmark](https://github.com/snap-research/locomo) — the standard evaluation for long-term conversational memory systems.

## What it measures

LoCoMo provides 10 long conversations (~300 turns each) with ~1,540 questions across 5 categories:

| Category | Tests |
|----------|-------|
| single-hop | Direct factual recall |
| multi-hop | Reasoning across multiple facts |
| temporal | Time-aware queries |
| open-domain | General knowledge mixed with conversation facts |
| adversarial | Questions designed to trip up retrieval |

Primary metric: **LLM Judge Score** — an LLM rates whether retrieved context supports the correct answer.

## How it works

The benchmark runs the same pipeline as production:

1. **Store conversations** — each dialogue turn becomes a `conversation` entity (like `tui_interactive.py`)
2. **Synthesize** — LLM extracts structured knowledge (`user_preference`, `lesson_learned`, `project_context`, `decision_rationale`) using the same prompt as `agent.py`
3. **Discover connections** — LLM identifies relationships between entities
4. **Retrieve** — for each question, uses the same multi-strategy retrieval as the agent:
   - `search_knowledge()` for user preferences (always loaded)
   - `semantic_search()` over knowledge types
   - `semantic_search()` for auto-context (conversation entities)
5. **Judge** — LLM scores whether retrieved context supports the correct answer

## Quick start

```bash
# Full pipeline on 1 conversation (~150 questions)
make run-quick

# Full benchmark (all 10 conversations)
make run

# Vector-only baseline (no synthesis, for comparison)
make run-baseline
```

## Options

```
python run_benchmark.py \
  --dataset locomo10.json \         # path to dataset (auto-downloaded)
  --db-dir /tmp/my-kg \             # custom KG location (default: temp dir)
  --model claude-sonnet-4-6 \       # LLM judge model
  --synthesis-model claude-sonnet-4-6 \  # model for synthesis/connections
  --limit 2 \                       # limit to N conversations
  --keep-db \                       # keep KG database after run
  --skip-synthesis \                # skip synthesis (vector baseline)
  --output results.md \             # write results to file
  -v                                # verbose logging
```

## Reference scores

| System | Overall Score |
|--------|-------------|
| Evermind | 93.05% |
| ai-assist (synthesis + hybrid) | 77.6% |
| Memobase | 75.78% |
| Mem0 | ~70% |
| ai-assist (vector baseline) | 32.0% |

## Isolation

The benchmark creates a temporary SQLite database. Your `~/.ai-assist/knowledge_graph.db` is never touched. The temp database is cleaned up unless `--keep-db` is passed.
