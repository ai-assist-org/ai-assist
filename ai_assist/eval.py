"""Agent evaluation framework: query traces and metrics"""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from .config import get_config_dir


def _partition_by_age(lines: list[str], cutoff: datetime) -> tuple[list[str], int]:
    """Split JSONL lines into kept/removed based on timestamp age."""
    kept: list[str] = []
    removed = 0
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            entry = json.loads(stripped)
            if datetime.fromisoformat(entry["timestamp"]) >= cutoff:
                kept.append(stripped)
            else:
                removed += 1
        except json.JSONDecodeError, KeyError, ValueError:
            kept.append(stripped)
    return kept, removed


@dataclass
class QueryTrace:
    """Structured trace record for a single query execution."""

    # Input
    query_text: str
    timestamp: str  # ISO format

    # Execution
    tool_calls: list[dict] = field(default_factory=list)  # [{tool_name, arguments}] - no results
    turn_count: int = 0

    # Output
    response_text: str = ""

    # Token usage
    token_usage: list[dict] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_thinking_tokens: int = 0

    # Performance
    duration_seconds: float = 0.0

    # Metadata
    model: str = ""
    tools_available_count: int = 0
    pid: int = 0

    # Dedup
    duplicate_tool_calls: int = 0

    def to_jsonl_line(self) -> str:
        """Serialize to a single JSON line."""
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_json(cls, data: dict) -> QueryTrace:
        """Deserialize from JSON dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class TraceStore:
    """JSONL-based trace storage, following the audit.py pattern."""

    def __init__(self, trace_dir: Path | None = None) -> None:
        if trace_dir is None:
            trace_dir = get_config_dir() / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_file = trace_dir / "query_traces.jsonl"

    def append(self, trace: QueryTrace) -> None:
        """Append a trace record to the JSONL file."""
        with open(self.trace_file, "a") as f:
            f.write(trace.to_jsonl_line() + "\n")

    def read_all(self) -> list[QueryTrace]:
        """Read all traces from the JSONL file."""
        if not self.trace_file.exists():
            return []
        traces = []
        with open(self.trace_file) as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    try:
                        data = json.loads(stripped)
                        traces.append(QueryTrace.from_json(data))
                    except json.JSONDecodeError, TypeError:
                        pass  # Skip malformed lines
        return traces

    def cleanup(self, max_age_days: int = 30) -> int:
        """Remove traces older than max_age_days. Returns count removed."""
        if not self.trace_file.exists():
            return 0
        cutoff = datetime.now() - timedelta(days=max_age_days)
        lines = self.trace_file.read_text().splitlines()
        kept, removed = _partition_by_age(lines, cutoff)
        self.trace_file.write_text("".join(line + "\n" for line in kept))
        return removed


@dataclass
class EvalMetrics:
    """Aggregate evaluation metrics computed from traces."""

    total_queries: int

    # Groundedness
    avg_citation_ratio: float
    queries_with_citations: int

    # Tool usage
    tool_usage_rate: float
    avg_tools_per_query: float

    # Efficiency
    avg_turns: float
    avg_total_tokens: int
    avg_duration_seconds: float

    # Dedup
    avg_duplicate_tool_calls: float
    queries_with_duplicates: int


class QueryEvaluator:
    """Compute evaluation metrics from query traces.

    All metrics are computed locally from trace data.
    No extra API calls are made.
    """

    CITATION_PATTERN = re.compile(r"\(source:\s*\w+", re.IGNORECASE)

    @classmethod
    def citation_ratio(cls, response_text: str) -> float:
        """Compute the ratio of sentences containing source citations.

        Looks for patterns like (source: tool_name) in the response.
        Returns 0.0 if no sentences, otherwise fraction with citations.
        """
        sentences = re.split(r"[.!?]\s+", response_text.strip())
        sentences = [s for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        cited = sum(1 for s in sentences if cls.CITATION_PATTERN.search(s))
        return cited / len(sentences)

    @classmethod
    def evaluate_traces(cls, traces: list[QueryTrace]) -> EvalMetrics:
        """Compute aggregate metrics from a list of traces."""
        if not traces:
            return EvalMetrics(
                total_queries=0,
                avg_citation_ratio=0.0,
                queries_with_citations=0,
                tool_usage_rate=0.0,
                avg_tools_per_query=0.0,
                avg_turns=0.0,
                avg_total_tokens=0,
                avg_duration_seconds=0.0,
                avg_duplicate_tool_calls=0.0,
                queries_with_duplicates=0,
            )

        n = len(traces)

        citation_ratios = [cls.citation_ratio(t.response_text) for t in traces]
        queries_with_tools = sum(1 for t in traces if t.tool_calls)
        total_tool_calls = sum(len(t.tool_calls) for t in traces)
        total_duplicates = sum(t.duplicate_tool_calls for t in traces)
        queries_with_dupes = sum(1 for t in traces if t.duplicate_tool_calls > 0)

        return EvalMetrics(
            total_queries=n,
            avg_citation_ratio=sum(citation_ratios) / n,
            queries_with_citations=sum(1 for r in citation_ratios if r > 0),
            tool_usage_rate=queries_with_tools / n,
            avg_tools_per_query=total_tool_calls / n,
            avg_turns=sum(t.turn_count for t in traces) / n,
            avg_total_tokens=int(sum(t.total_input_tokens + t.total_output_tokens for t in traces) / n),
            avg_duration_seconds=round(sum(t.duration_seconds for t in traces) / n, 1),
            avg_duplicate_tool_calls=total_duplicates / n,
            queries_with_duplicates=queries_with_dupes,
        )
