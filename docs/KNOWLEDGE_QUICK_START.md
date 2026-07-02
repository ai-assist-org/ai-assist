# Knowledge Management - Quick Start

## 5-Minute Guide

### What is it?

The agent learns and remembers information from conversations:
- Your preferences (coding style, tools, workflows)
- Lessons learned (patterns, gotchas, best practices)
- Project context (team info, goals, constraints)
- Decision rationale (why choices were made)

### How does it work?

**Automatic** - The agent decides when to learn and save knowledge. No user action required.

```
You: I prefer pytest over unittest for Python testing
Agent: Got it! I'll use pytest.
      [Saves: user_preference:python_test_framework]

# Next conversation...
You: Write tests for login
Agent: I'll write pytest tests...
      [Recalled from knowledge base]
```

### How to see what's stored?

**Just ask the agent:**

```bash
ai-assist /interactive

You: What do you know about me?
You: What have you learned?
You: Show me your knowledge graph
```

**Or run the inspector:**

```bash
python examples/inspect_knowledge_base.py
```

### Where is it stored?

```
~/.ai-assist/knowledge_graph.db
```

SQLite database with bi-temporal tracking (when learned, when valid).

### Common questions

**Q: Does the agent save everything I say?**
A: Yes. All conversation exchanges are stored in the Knowledge Graph as they happen. A KG synthesis task then reviews the day's conversations and extracts structured knowledge (preferences, lessons, context, decisions).

**Q: Can I delete stored knowledge?**
A: Yes, ask the agent to expire it ("that preference is outdated") or manually edit the SQLite database. The agent uses `internal__expire_knowledge` to mark entries as no longer valid or retract incorrect beliefs.

**Q: How much does it store?**
A: Only what you explicitly state or the agent learns from conversation. Typical: 10-50 entries.

**Q: Does it work offline?**
A: Knowledge retrieval works offline. Synthesis requires Claude API (uses Haiku model).

**Q: Can I see synthesis happen?**
A: Synthesis runs as a nightly scheduled task (configurable in `event-schedules.json`). You'll see logs in monitor mode when it runs.

### Example: Real knowledge base

```bash
$ python examples/inspect_knowledge_base.py

Total entities: 283
  • project_context: 2
  • user_preference: 0
  • lesson_learned: 0
  • dci_job: 114 (auto-captured)
  • jira_ticket: 156 (auto-captured)

Project Context (2):
  • sami_antilla_manager: Sami Antilla is Fred's manager...
  • franck_baudin_role: Franck Baudin is the Product Manager for DCI...
```

### Tips

1. **Be explicit** when stating preferences:
   - "I prefer pytest over unittest"
   - "I switched to neovim" (agent will expire old preference and save the new one)

2. **Ask the agent** to recall:
   - "What testing framework do I prefer?"
   - "What have you learned about DCI?"

3. **Query by time**:
   - "What did you know yesterday at 3pm?"
   - "What have you learned since Monday?"
   - "What was happening on June 15?"

4. **Store future plans**:
   - "I'll be at a conference in Berlin on July 10" (stored with future valid_from)
   - "What are my upcoming plans?" (agent searches with include_future=true)

5. **Correct the agent**:
   - "That lesson about Friday failures was wrong" (agent will retract it)
   - "I no longer use vim" (agent will expire the preference)

6. **Check periodically** what's stored:
   - "Show me all your stored knowledge"
   - Or run: `python examples/inspect_knowledge_base.py`

7. **Backup** your knowledge base:
   ```bash
   cp ~/.ai-assist/knowledge_graph.db ~/backup-$(date +%Y%m%d).db
   ```

### Learn more

- [KNOWLEDGE_MANAGEMENT.md](KNOWLEDGE_MANAGEMENT.md) - Full user guide
- [INSPECTING_KNOWLEDGE.md](INSPECTING_KNOWLEDGE.md) - Detailed inspection guide
- [examples/README.md](../examples/README.md) - Example scripts

### Quick commands

```bash
# Interactive mode
ai-assist /interactive

# Ask what agent knows
You: What preferences do you know?
You: What's in your knowledge graph?

# Inspect database
python examples/inspect_knowledge_base.py

# Direct SQL
sqlite3 ~/.ai-assist/knowledge_graph.db "SELECT entity_type, COUNT(*) FROM entities WHERE tx_to IS NULL GROUP BY entity_type;"
```

---

**That's it!** The agent learns automatically. Just use it normally and ask what it knows when curious.
