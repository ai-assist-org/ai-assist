# Agent Knowledge Management

The ai-assist agent can learn and remember information from conversations.

📖 **See also:** [INSPECTING_KNOWLEDGE.md](INSPECTING_KNOWLEDGE.md) for detailed instructions on viewing stored knowledge.

## What Gets Saved

The agent automatically saves:
- **Conversations**: Every user/assistant exchange is stored in the Knowledge Graph on-the-fly
- **User Preferences**: Code style, tools, workflows
- **Lessons Learned**: Bug patterns, best practices
- **Project Context**: Goals, constraints, background
- **Decision Rationale**: Why choices were made

## How It Works

1. Every conversation exchange is saved to the Knowledge Graph as it happens
2. A **KG synthesis** task (configurable in `event-schedules.json`) reviews the day's conversations
3. The synthesis extracts structured knowledge (preferences, lessons, context, rationale)
4. Extracted insights are saved to the Knowledge Graph automatically
5. Agent can recall saved knowledge in future conversations

## Example

```
You: I prefer pytest over unittest for Python testing
Agent: Got it! I'll remember to use pytest for Python tests.
[Agent saves: user_preference:python_test_framework]

# Next conversation:
You: Write tests for the login function
Agent: I'll write pytest tests for login...
[Agent recalled preference from knowledge base]
```

## Access Tracking

Every time the agent recalls knowledge — whether via system prompt injection or explicit search — an access event is recorded. This lets you measure which knowledge is actually useful:

```
You: How healthy is your knowledge graph?
Agent: [Calls internal__kg_knowledge_health]
Agent:
  Total accesses: 142
  Most accessed: user_preference:python_test_framework (28 times)
  Never accessed: 3 entries
  Stale (>7 days): 2 entries
```

## Storage

Knowledge is stored locally in `~/.ai-assist/knowledge_graph.db`.

## Checking Stored Knowledge

### Method 1: Ask the Agent (Recommended)

In interactive mode, simply ask:

```bash
ai-assist /interactive

You: What preferences do you know about me?
Agent: [Calls internal__search_knowledge(entity_type="user_preference")]
Agent: I know you prefer pytest over unittest for Python testing...

You: What lessons have you learned?
Agent: [Calls internal__search_knowledge(entity_type="lesson_learned")]
Agent: I've learned that DCI jobs fail more on Fridays...

You: Show me all your stored knowledge
Agent: [Calls introspection__get_kg_stats + searches all types]
```

### Method 2: Knowledge Graph Statistics

The agent has an introspection tool that shows counts:

```bash
ai-assist /interactive

You: What's in your knowledge graph?
Agent: [Calls introspection__get_kg_stats]
Agent:
  Total entities: 15
  By type:
    - user_preference: 3
    - lesson_learned: 2
    - project_context: 1
    - dci_job: 8
    - jira_ticket: 1
```

### Method 3: Direct Database Query (Advanced)

For advanced users, inspect the SQLite database directly:

```bash
# View the database
sqlite3 ~/.ai-assist/knowledge_graph.db

# List all entity types
SELECT entity_type, COUNT(*) FROM entities WHERE tx_to IS NULL GROUP BY entity_type;

# View knowledge entities
SELECT id, entity_type, json_extract(data, '$.content')
FROM entities
WHERE entity_type IN ('user_preference', 'lesson_learned', 'project_context', 'decision_rationale')
AND tx_to IS NULL;

# View a specific preference
SELECT * FROM entities WHERE id LIKE 'user_preference:%';
```

## Temporal Queries

The Knowledge Graph uses a bi-temporal model — every entry tracks both when it happened in reality (`valid_from`/`valid_to`) and when the agent learned about it (`tx_from`/`tx_to`). The agent can query knowledge across time:

### Point-in-Time Snapshots

```
You: What did you know yesterday at 3pm?
Agent: [Calls internal__kg_snapshot(time="2026-06-28T15:00:00", mode="known_at")]

You: What was happening on June 15?
Agent: [Calls internal__kg_snapshot(time="2026-06-15T00:00:00", mode="valid_at")]
```

Two modes:
- **`known_at`**: What the agent believed at that moment (transaction time)
- **`valid_at`**: What was true in reality at that moment (valid time)

### Time-Filtered Search

```
You: What have you learned in the past 2 days?
Agent: [Calls internal__search_knowledge(since="2026-06-27T00:00:00")]

You: Any new lessons since Monday?
Agent: [Calls internal__search_knowledge(entity_type="lesson_learned", since="2026-06-23T00:00:00")]
```

### Expiring Knowledge

The agent can mark entries as outdated or incorrect:

```
You: I switched from vim to neovim
Agent: [Calls internal__expire_knowledge(entity_id="user_preference:editor", reason="no_longer_valid")]
       [Calls internal__save_knowledge(..., content="User uses neovim")]

You: Actually, that lesson about Friday failures was wrong
Agent: [Calls internal__expire_knowledge(entity_id="lesson_learned:dci_friday_failures", reason="retract")]
```

Two reasons:
- **`no_longer_valid`**: The fact stopped being true (e.g., user changed preference)
- **`retract`**: The belief was incorrect (e.g., wrong assumption)

### Future Knowledge

The agent can store facts that will become true in the future:

```
You: I'll be at a conference in Berlin on July 10
Agent: [Calls internal__save_knowledge(..., valid_from="2026-07-10")]
```

Future-dated knowledge is excluded from normal searches and system prompt injection by default — it only surfaces when its `valid_from` date arrives. The agent can explicitly query future entries:

```
You: What do you know about my upcoming plans?
Agent: [Calls internal__search_knowledge(include_future=true)]
```

## Agent Tools (Internal)

The agent has these internal tools (automatically used, not directly callable):

**Knowledge Management:**
- `internal__save_knowledge`: Explicitly save facts
- `internal__search_knowledge`: Search for saved learnings (supports `since` filter)
- `internal__expire_knowledge`: Mark entries as no longer valid or retract incorrect beliefs

**Temporal Queries:**
- `internal__kg_snapshot`: Query the knowledge graph at a point in time
- `internal__kg_recent_changes`: Get changes in the last N hours
- `internal__kg_late_discoveries`: Find entities discovered late
- `internal__kg_discovery_lag_stats`: Analyze discovery lag statistics

**Health & Monitoring:**
- `internal__kg_knowledge_health`: Report access statistics — most used entries, never accessed, stale

**Introspection:**
- `introspection__get_kg_stats`: Get knowledge graph statistics
- `introspection__search_knowledge_graph`: Search for entities (DCI jobs, Jira tickets)
- `introspection__get_kg_entity`: Get specific entity details

These work automatically during conversations.
