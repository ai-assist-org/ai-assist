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

## Agent Tools (Internal)

The agent has these internal tools (automatically used, not directly callable):

**Knowledge Management:**
- `internal__search_knowledge`: Search for saved learnings
- `internal__save_knowledge`: Explicitly save facts

**Introspection:**
- `introspection__get_kg_stats`: Get knowledge graph statistics
- `introspection__search_knowledge_graph`: Search for entities (DCI jobs, Jira tickets)
- `introspection__get_kg_entity`: Get specific entity details

These work automatically during conversations.
