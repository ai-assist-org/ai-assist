"""Agent evaluation framework: query traces and metrics"""

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from .config import get_config_dir
from .pricing import compute_turn_cost, is_known_model, safe_int

_HOME = Path.home()


def _normalize_script_path(path: str) -> str:
    """Collapse absolute home paths to ~ so aggregation keys are consistent."""
    if path.startswith(str(_HOME) + "/"):
        return "~/" + path[len(str(_HOME)) + 1 :]
    if path.startswith("~/"):
        return path
    return path


def _is_expired(stripped_line: str, cutoff: datetime) -> bool:
    """True if a JSONL line has a timestamp older than cutoff (unparseable lines are kept)."""
    try:
        return datetime.fromisoformat(json.loads(stripped_line)["timestamp"]) < cutoff
    except Exception:
        return False


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
    total_cost_usd: float = 0.0

    # Performance
    duration_seconds: float = 0.0

    # Metadata
    model: str = ""
    tools_available_count: int = 0
    pid: int = 0

    # Script context (top-level AWL script or MCP prompt that triggered this query)
    script_path: str = ""

    # Dedup
    duplicate_tool_calls: int = 0

    def cost_usd(self) -> float:
        """Cost recomputed from per-turn token counts with the current pricing.

        Traces written before a pricing fix keep a stale total_cost_usd, so this
        is what reports should show. Falls back to the stored total when there
        are no per-turn counts or the model has no pricing entry.
        """
        if not self.token_usage or not is_known_model(self.model):
            return self.total_cost_usd
        return sum(compute_turn_cost(self.model, entry) for entry in self.token_usage if isinstance(entry, dict))

    def cache_tokens(self) -> tuple[int, int]:
        """Total (cache write, cache read) tokens across this query's turns."""
        entries = [e for e in self.token_usage if isinstance(e, dict)]
        return (
            sum(safe_int(e.get("cache_creation_input_tokens")) for e in entries),
            sum(safe_int(e.get("cache_read_input_tokens")) for e in entries),
        )

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
                    except Exception:
                        pass  # Skip malformed lines
        return traces

    def cleanup(self, max_age_days: int = 100) -> int:
        """Remove traces older than max_age_days. Returns count removed."""
        if not self.trace_file.exists():
            return 0
        cutoff = datetime.now() - timedelta(days=max_age_days)
        # Stream line by line: the file can be very large and this runs on every startup.
        with open(self.trace_file) as f:
            removed = sum(1 for line in f if (stripped := line.strip()) and _is_expired(stripped, cutoff))
        if not removed:
            return 0

        tmp_file = self.trace_file.with_name(self.trace_file.name + ".tmp")
        with open(self.trace_file) as src, open(tmp_file, "w") as dst:
            for line in src:
                stripped = line.strip()
                if stripped and not _is_expired(stripped, cutoff):
                    dst.write(stripped + "\n")
        tmp_file.replace(self.trace_file)
        return removed


class QueryEventStore:
    """Append-only, redacted timeline for queries that may not finish."""

    def __init__(self, trace_dir: Path | None = None) -> None:
        if trace_dir is None:
            trace_dir = get_config_dir() / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        self.event_file = trace_dir / "query_events.jsonl"

    def append(
        self,
        query_id: str,
        phase: str,
        *,
        elapsed_seconds: float,
        turn: int | None = None,
        tool_name: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Record metadata only; prompts, tool arguments, and results stay out."""
        event: dict[str, object] = {
            "timestamp": datetime.now().isoformat(),
            "query_id": query_id,
            "phase": phase,
            "elapsed_seconds": round(elapsed_seconds, 2),
        }
        if turn is not None:
            event["turn"] = turn
        if tool_name is not None:
            event["tool_name"] = tool_name
        if detail is not None:
            event["detail"] = detail[:200]
        with open(self.event_file, "a") as f:
            f.write(json.dumps(event) + "\n")


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

    # Cost
    total_cost_usd: float = 0.0
    avg_cost_per_query_usd: float = 0.0

    # Dedup
    avg_duplicate_tool_calls: float = 0.0
    queries_with_duplicates: int = 0


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
        total_cost = sum(t.cost_usd() for t in traces)

        return EvalMetrics(
            total_queries=n,
            avg_citation_ratio=sum(citation_ratios) / n,
            queries_with_citations=sum(1 for r in citation_ratios if r > 0),
            tool_usage_rate=queries_with_tools / n,
            avg_tools_per_query=total_tool_calls / n,
            avg_turns=sum(t.turn_count for t in traces) / n,
            avg_total_tokens=int(sum(t.total_input_tokens + t.total_output_tokens for t in traces) / n),
            avg_duration_seconds=round(sum(t.duration_seconds for t in traces) / n, 1),
            total_cost_usd=round(total_cost, 4),
            avg_cost_per_query_usd=round(total_cost / n, 4),
            avg_duplicate_tool_calls=total_duplicates / n,
            queries_with_duplicates=queries_with_dupes,
        )


@dataclass
class CostSummary:
    """Aggregated cost data from traces."""

    label: str
    query_count: int
    total_cost: float
    avg_cost: float
    total_input_tokens: int  # uncached input only
    total_output_tokens: int
    cost_by_model: dict[str, float]
    queries_by_model: dict[str, int]
    cost_per_day: float = 0.0
    num_days: int = 0
    cost_by_script: dict[str, float] = field(default_factory=dict)
    queries_by_script: dict[str, int] = field(default_factory=dict)
    total_cache_write_tokens: int = 0
    total_cache_read_tokens: int = 0


def compute_cost_summary(period: str | None = None) -> CostSummary | str:
    """Compute cost summary from traces.

    Returns CostSummary on success, or an error message string.
    """
    store = TraceStore()
    traces = store.read_all()

    if period:
        match = re.match(r"^(\d+)d$", period)
        if not match:
            return f"Invalid period '{period}'. Use format: 7d, 30d, 90d"
        days = int(match.group(1))
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        traces = [t for t in traces if t.timestamp >= cutoff]

    if not traces:
        return "No query traces found for the selected period."

    costs = [t.cost_usd() for t in traces]
    total_cost = sum(costs)
    total_input = sum(t.total_input_tokens for t in traces)
    total_output = sum(t.total_output_tokens for t in traces)
    cache_totals = [t.cache_tokens() for t in traces]

    distinct_days = {t.timestamp[:10] for t in traces}
    num_days = len(distinct_days)
    cost_per_day = total_cost / num_days if num_days else 0.0

    cost_by_model: dict[str, float] = defaultdict(float)
    queries_by_model: dict[str, int] = defaultdict(int)
    cost_by_script: dict[str, float] = defaultdict(float)
    queries_by_script: dict[str, int] = defaultdict(int)
    for t, cost in zip(traces, costs, strict=True):
        cost_by_model[t.model] += cost
        queries_by_model[t.model] += 1
        script_key = _normalize_script_path(t.script_path) if t.script_path else "(interactive)"
        cost_by_script[script_key] += cost
        queries_by_script[script_key] += 1

    return CostSummary(
        label=f"last {period}" if period else "all time",
        query_count=len(traces),
        total_cost=total_cost,
        avg_cost=total_cost / len(traces),
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        cost_per_day=cost_per_day,
        num_days=num_days,
        cost_by_model=dict(cost_by_model),
        queries_by_model=dict(queries_by_model),
        cost_by_script=dict(cost_by_script),
        queries_by_script=dict(queries_by_script),
        total_cache_write_tokens=sum(w for w, _ in cache_totals),
        total_cache_read_tokens=sum(r for _, r in cache_totals),
    )
