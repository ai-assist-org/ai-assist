"""Context management for intelligent conversations"""

import logging
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .knowledge_graph import KnowledgeGraph


class ConversationMemory:
    """Manages conversation history with context window management

    Keeps track of recent user-assistant exchanges to provide conversation
    context to Claude, enabling natural follow-up questions and maintaining
    conversation flow.
    """

    def __init__(self, max_exchanges: int = 10, compaction_threshold: int = 8):
        """Initialize conversation memory

        Args:
            max_exchanges: Maximum number of exchanges to keep in memory.
                          Older exchanges are dropped to manage context window.
            compaction_threshold: Number of exchanges that triggers compaction.
                                When reached, the oldest half is summarized into one exchange.
        """
        self.exchanges: list[dict[str, str]] = []  # List of {user, assistant, timestamp}
        self.max_exchanges = max_exchanges
        self.compaction_threshold = compaction_threshold

    def add_exchange(self, user_input: str, assistant_response: str):
        """Add a conversation exchange

        Args:
            user_input: The user's question/message
            assistant_response: The assistant's response
        """
        self.exchanges.append(
            {"user": user_input, "assistant": assistant_response, "timestamp": datetime.now().isoformat()}
        )

        # Keep only recent exchanges to fit context window
        if len(self.exchanges) > self.max_exchanges:
            self.exchanges = self.exchanges[-self.max_exchanges :]

    def to_messages(self) -> list[dict]:
        """Convert conversation history to Claude messages format

        Returns:
            List of message dicts in Claude API format:
            [
                {"role": "user", "content": "question 1"},
                {"role": "assistant", "content": "answer 1"},
                {"role": "user", "content": "question 2"},
                ...
            ]
        """
        messages = []
        for exchange in self.exchanges:
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["assistant"]})
        return messages

    def load_exchanges(self, exchanges: list[dict[str, str]]):
        """Load exchanges from saved data, applying max_exchanges limit"""
        for ex in exchanges:
            if "user" in ex and "assistant" in ex:
                self.exchanges.append(
                    {
                        "user": ex["user"],
                        "assistant": ex["assistant"],
                        "timestamp": ex.get("timestamp", ""),
                    }
                )
        if len(self.exchanges) > self.max_exchanges:
            self.exchanges = self.exchanges[-self.max_exchanges :]

    def needs_compaction(self) -> bool:
        """Check if conversation memory should be compacted.

        Returns:
            True if number of exchanges has reached the compaction threshold
        """
        return len(self.exchanges) >= self.compaction_threshold

    def compact(self, anthropic_client: Any, model: str, keep_recent: int = 4) -> bool:
        """Compact old exchanges into a summary using Claude.

        Takes the oldest exchanges (all except keep_recent most recent),
        summarizes them into a single synthetic exchange, and replaces them.

        Args:
            anthropic_client: Anthropic or AnthropicVertex client instance
            model: Model name to use for summarization
            keep_recent: Number of recent exchanges to keep verbatim

        Returns:
            True if compaction was performed, False if not needed
        """
        if len(self.exchanges) <= keep_recent:
            return False

        old_exchanges = self.exchanges[:-keep_recent]
        recent_exchanges = self.exchanges[-keep_recent:]

        # Build text from old exchanges
        history_lines = []
        for ex in old_exchanges:
            history_lines.append(f"User: {ex['user']}")
            history_lines.append(f"Assistant: {ex['assistant']}")
        history_text = "\n".join(history_lines)

        compaction_prompt = (
            "Summarize this conversation history into key facts, decisions, entities mentioned, "
            "and pending items. Be concise but preserve important details like ticket numbers, "
            "job IDs, error messages, and action items. Output a brief summary paragraph."
        )

        try:
            with anthropic_client.messages.stream(
                model=model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": f"{compaction_prompt}\n\n{history_text}",
                    }
                ],
            ) as stream:
                response = stream.get_final_message()

            summary_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    summary_text += block.text

            if summary_text.strip():
                # Replace old exchanges with a single synthetic summary exchange
                summary_exchange = {
                    "user": "[Conversation summary]",
                    "assistant": summary_text.strip(),
                    "timestamp": datetime.now().isoformat(),
                }
                self.exchanges = [summary_exchange] + recent_exchanges
                logging.info(
                    "Compacted %d old exchanges into summary (%d recent kept)",
                    len(old_exchanges),
                    len(recent_exchanges),
                )
                return True

        except Exception as e:
            logging.warning("Context compaction failed: %s", e)

        return False

    def clear(self):
        """Clear all conversation history"""
        self.exchanges = []

    def get_exchange_count(self) -> int:
        """Get the number of exchanges in memory"""
        return len(self.exchanges)

    def get_last_exchange(self) -> dict | None:
        """Get the most recent exchange

        Returns:
            Dictionary with 'user', 'assistant', and 'timestamp' keys,
            or None if no exchanges
        """
        if self.exchanges:
            return self.exchanges[-1]
        return None

    def __len__(self) -> int:
        """Return number of exchanges"""
        return len(self.exchanges)

    def __repr__(self) -> str:
        return f"ConversationMemory(exchanges={len(self.exchanges)}, max={self.max_exchanges})"


class KnowledgeGraphContext:
    """Enriches prompts with relevant context from the knowledge graph

    Automatically detects entity references in user queries (Jira tickets,
    DCI jobs, time references) and queries the knowledge graph to provide
    relevant historical context.
    """

    def __init__(self, knowledge_graph: KnowledgeGraph | None = None):
        """Initialize knowledge graph context

        Args:
            knowledge_graph: KnowledgeGraph instance, or None to disable enrichment
        """
        self.knowledge_graph = knowledge_graph
        self.last_context_used: list[str] = []  # Track what context was added

    def extract_entity_references(self, text: str) -> dict[str, list[str]]:
        """Extract entity references from user input

        Args:
            text: User's query text

        Returns:
            Dictionary with entity types as keys and lists of IDs as values:
            {
                "jira_tickets": ["CILAB-123", "CNF-456"],
                "dci_jobs": ["abc-123", "def-456"],
                "time_refs": ["yesterday", "last week"]
            }
        """
        refs: dict[str, list[str]] = {"jira_tickets": [], "dci_jobs": [], "time_refs": []}

        # Extract Jira ticket references (PROJECT-123 format)
        jira_pattern = r"\b([A-Z][A-Z0-9]+-\d+)\b"
        refs["jira_tickets"] = re.findall(jira_pattern, text)

        # Extract time references
        time_patterns = [
            r"\byesterday\b",
            r"\btoday\b",
            r"\blast\s+week\b",
            r"\blast\s+month\b",
            r"\brecent(?:ly)?\b",
            r"\bthis\s+week\b",
            r"\bthis\s+month\b",
        ]
        for pattern in time_patterns:
            time_match = re.search(pattern, text, re.IGNORECASE)
            if time_match:
                refs["time_refs"].append(time_match.group())

        return refs

    def parse_time_reference(self, time_ref: str) -> datetime:
        """Convert time reference to datetime

        Args:
            time_ref: Time reference like "yesterday", "last week"

        Returns:
            Corresponding datetime
        """
        now = datetime.now()
        time_ref_lower = time_ref.lower()

        if "yesterday" in time_ref_lower:
            return now - timedelta(days=1)
        elif "last week" in time_ref_lower:
            return now - timedelta(weeks=1)
        elif "last month" in time_ref_lower:
            return now - timedelta(days=30)
        elif "this week" in time_ref_lower:
            return now - timedelta(days=now.weekday())
        elif "this month" in time_ref_lower:
            return now.replace(day=1)
        elif "recent" in time_ref_lower:
            return now - timedelta(days=7)
        else:
            return now - timedelta(days=1)

    def enrich_prompt(self, prompt: str, max_entities: int = 5) -> tuple[str, list[str]]:
        """Enrich prompt with relevant context from knowledge graph

        Args:
            prompt: Original user prompt
            max_entities: Maximum number of entities to include as context

        Returns:
            Tuple of (enriched_prompt, context_summary)
            - enriched_prompt: Prompt with added context
            - context_summary: List of strings describing what context was added
        """
        if not self.knowledge_graph:
            return prompt, []

        refs = self.extract_entity_references(prompt)
        context_parts = []
        context_summary = []

        # Add Jira ticket context
        for ticket_key in refs["jira_tickets"][:max_entities]:
            entity = self.knowledge_graph.get_entity(ticket_key)
            if entity:
                ticket_data = entity.data
                context_parts.append(
                    f"**{ticket_key}**: {ticket_data.get('summary', 'N/A')} "
                    f"[Status: {ticket_data.get('status', 'Unknown')}]"
                )
                context_summary.append(f"Jira ticket {ticket_key}")

        # Add time-based context
        if refs["time_refs"]:
            # Get recent failures or issues
            time_ref = refs["time_refs"][0]
            since_time = self.parse_time_reference(time_ref)

            # Query current DCI jobs (what ai-assist knows now)
            all_current_jobs = self.knowledge_graph.query_as_of(
                datetime.now(), entity_type="dci_job", limit=None  # Get all, we'll filter
            )

            # Filter to only jobs that became valid after since_time
            recent_jobs = [j for j in all_current_jobs if j.valid_from >= since_time][:max_entities]

            if recent_jobs:
                failed_jobs = [j for j in recent_jobs if j.data.get("status") in ["failure", "error"]]
                if failed_jobs:
                    context_parts.append(
                        f"\n**Recent failures since {time_ref}**: " f"{len(failed_jobs)} DCI job(s) failed"
                    )
                    context_summary.append(f"{len(failed_jobs)} recent failures")

        # Build enriched prompt
        if context_parts:
            context_block = "\n\n## Relevant Context\n" + "\n".join(context_parts)
            enriched_prompt = prompt + context_block
            self.last_context_used = context_summary
            return enriched_prompt, context_summary

        self.last_context_used = []
        return prompt, []

    def get_last_context(self) -> list[str]:
        """Get summary of context used in last enrichment

        Returns:
            List of strings describing what context was added
        """
        return self.last_context_used.copy()
