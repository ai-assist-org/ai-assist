"""Tests for the agent evaluation framework (eval.py)"""

import json
import os
from datetime import datetime, timedelta

from ai_assist.eval import EvalMetrics, QueryEvaluator, QueryTrace, TraceStore


class TestQueryTrace:
    """Tests for QueryTrace dataclass"""

    def test_to_jsonl_line(self):
        """Serializes to valid JSON"""
        trace = QueryTrace(
            query_text="What jobs failed?",
            timestamp="2026-02-23T10:00:00",
            tool_calls=[{"tool_name": "search_dci_jobs", "arguments": {"query": "status=failure"}}],
            turn_count=3,
            response_text="Two jobs failed.",
            total_input_tokens=1000,
            total_output_tokens=200,
            duration_seconds=5.2,
            model="claude-sonnet-4-5-20250514",
        )
        line = trace.to_jsonl_line()
        data = json.loads(line)

        assert data["query_text"] == "What jobs failed?"
        assert data["turn_count"] == 3
        assert data["total_input_tokens"] == 1000
        assert len(data["tool_calls"]) == 1

    def test_from_json_roundtrip(self):
        """Roundtrip serialization preserves data"""
        original = QueryTrace(
            query_text="Show status",
            timestamp="2026-02-23T10:00:00",
            tool_calls=[{"tool_name": "today", "arguments": {}}],
            turn_count=1,
            response_text="Today is Monday.",
            total_input_tokens=500,
            total_output_tokens=100,
            duration_seconds=2.1,
            model="claude-sonnet-4-5-20250514",
            tools_available_count=15,
        )
        data = json.loads(original.to_jsonl_line())
        restored = QueryTrace.from_json(data)

        assert restored.query_text == original.query_text
        assert restored.turn_count == original.turn_count
        assert restored.tools_available_count == 15
        assert restored.duration_seconds == 2.1

    def test_from_json_ignores_unknown_fields(self):
        """Unknown fields in JSON are silently ignored"""
        data = {
            "query_text": "test",
            "timestamp": "2026-02-23T10:00:00",
            "unknown_field": "should be ignored",
        }
        trace = QueryTrace.from_json(data)
        assert trace.query_text == "test"

    def test_defaults(self):
        """Default values are sensible"""
        trace = QueryTrace(query_text="test", timestamp="2026-02-23T10:00:00")
        assert trace.tool_calls == []
        assert trace.turn_count == 0
        assert trace.response_text == ""
        assert trace.total_input_tokens == 0
        assert trace.model == ""
        assert trace.duplicate_tool_calls == 0
        assert trace.pid == 0


class TestTraceStore:
    """Tests for TraceStore JSONL storage"""

    def test_append_and_read(self, tmp_path):
        """Can append and read back traces"""
        store = TraceStore(trace_dir=tmp_path)
        trace = QueryTrace(
            query_text="What failed?",
            timestamp="2026-02-23T10:00:00",
            response_text="Nothing failed.",
        )
        store.append(trace)

        traces = store.read_all()
        assert len(traces) == 1
        assert traces[0].query_text == "What failed?"

    def test_read_empty_file(self, tmp_path):
        """Returns empty list if no traces"""
        store = TraceStore(trace_dir=tmp_path)
        assert store.read_all() == []

    def test_multiple_appends(self, tmp_path):
        """Multiple appends accumulate"""
        store = TraceStore(trace_dir=tmp_path)
        for i in range(5):
            store.append(QueryTrace(query_text=f"query {i}", timestamp="2026-02-23T10:00:00"))

        traces = store.read_all()
        assert len(traces) == 5
        assert traces[2].query_text == "query 2"

    def test_malformed_lines_skipped(self, tmp_path):
        """Malformed JSONL lines are silently skipped"""
        store = TraceStore(trace_dir=tmp_path)
        store.append(QueryTrace(query_text="good", timestamp="2026-02-23T10:00:00"))

        # Inject a malformed line
        with open(store.trace_file, "a") as f:
            f.write("this is not valid json\n")

        store.append(QueryTrace(query_text="also good", timestamp="2026-02-23T10:00:00"))

        traces = store.read_all()
        assert len(traces) == 2
        assert traces[0].query_text == "good"
        assert traces[1].query_text == "also good"

    def test_cleanup_removes_old(self, tmp_path):
        """Cleanup removes traces older than max_age_days"""
        store = TraceStore(trace_dir=tmp_path)
        old_ts = (datetime.now() - timedelta(days=60)).isoformat()
        recent_ts = datetime.now().isoformat()

        store.append(QueryTrace(query_text="old", timestamp=old_ts))
        store.append(QueryTrace(query_text="recent", timestamp=recent_ts))

        removed = store.cleanup(max_age_days=30)
        assert removed == 1

        traces = store.read_all()
        assert len(traces) == 1
        assert traces[0].query_text == "recent"

    def test_cleanup_no_file(self, tmp_path):
        """Cleanup returns 0 if no trace file exists"""
        store = TraceStore(trace_dir=tmp_path)
        assert store.cleanup() == 0

    def test_cleanup_keeps_unparseable(self, tmp_path):
        """Cleanup keeps lines it can't parse"""
        store = TraceStore(trace_dir=tmp_path)
        store.append(QueryTrace(query_text="good", timestamp=datetime.now().isoformat()))

        with open(store.trace_file, "a") as f:
            f.write("not json\n")

        removed = store.cleanup(max_age_days=30)
        assert removed == 0

        # Read raw lines to check unparseable line is kept
        with open(store.trace_file) as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 2


class TestCitationRatio:
    """Tests for QueryEvaluator.citation_ratio"""

    def test_no_citations(self):
        """Text with no citations returns 0.0"""
        ratio = QueryEvaluator.citation_ratio("Two jobs failed today. The cluster is healthy.")
        assert ratio == 0.0

    def test_all_cited(self):
        """Text with all sentences cited returns 1.0"""
        text = "Two jobs failed (source: search_dci_jobs). The ticket is open (source: get_jira_ticket)."
        ratio = QueryEvaluator.citation_ratio(text)
        assert ratio == 1.0

    def test_partial_citation(self):
        """Mixed text returns correct fraction"""
        text = (
            "Two jobs failed (source: search_dci_jobs). I think it might be a network issue. The cluster was restarted."
        )
        ratio = QueryEvaluator.citation_ratio(text)
        # 1 out of 3 sentences has citation
        assert abs(ratio - 1 / 3) < 0.01

    def test_empty_text(self):
        """Empty text returns 0.0"""
        assert QueryEvaluator.citation_ratio("") == 0.0

    def test_case_insensitive(self):
        """Citation pattern is case insensitive"""
        text = "Jobs found (Source: SEARCH_DCI_JOBS). Done."
        ratio = QueryEvaluator.citation_ratio(text)
        assert ratio > 0.0


class TestEvaluateTraces:
    """Tests for QueryEvaluator.evaluate_traces"""

    def test_empty_traces(self):
        """Empty trace list returns zero metrics"""
        metrics = QueryEvaluator.evaluate_traces([])
        assert metrics.total_queries == 0
        assert metrics.avg_citation_ratio == 0.0
        assert metrics.tool_usage_rate == 0.0

    def test_single_trace(self):
        """Single trace produces correct metrics"""
        trace = QueryTrace(
            query_text="test",
            timestamp="2026-02-23T10:00:00",
            tool_calls=[{"tool_name": "search", "arguments": {}}],
            turn_count=2,
            response_text="Result (source: search). Done.",
            total_input_tokens=1000,
            total_output_tokens=200,
            duration_seconds=3.5,
        )
        metrics = QueryEvaluator.evaluate_traces([trace])

        assert metrics.total_queries == 1
        assert metrics.tool_usage_rate == 1.0
        assert metrics.avg_tools_per_query == 1.0
        assert metrics.avg_turns == 2.0
        assert metrics.avg_total_tokens == 1200
        assert metrics.avg_duration_seconds == 3.5

    def test_multiple_traces(self):
        """Multiple traces produce correct averages"""
        traces = [
            QueryTrace(
                query_text="q1",
                timestamp="2026-02-23T10:00:00",
                tool_calls=[{"tool_name": "t1", "arguments": {}}],
                turn_count=2,
                total_input_tokens=1000,
                total_output_tokens=200,
                duration_seconds=3.0,
            ),
            QueryTrace(
                query_text="q2",
                timestamp="2026-02-23T10:01:00",
                tool_calls=[],
                turn_count=1,
                total_input_tokens=500,
                total_output_tokens=100,
                duration_seconds=1.0,
            ),
        ]
        metrics = QueryEvaluator.evaluate_traces(traces)

        assert metrics.total_queries == 2
        assert metrics.tool_usage_rate == 0.5  # 1 of 2 used tools
        assert metrics.avg_tools_per_query == 0.5  # 1 total / 2 queries
        assert metrics.avg_turns == 1.5  # (2+1)/2
        assert metrics.avg_total_tokens == 900  # (1200+600)/2
        assert metrics.avg_duration_seconds == 2.0  # (3+1)/2

    def test_metrics_is_dataclass(self):
        """EvalMetrics is a proper dataclass"""
        metrics = EvalMetrics(
            total_queries=10,
            avg_citation_ratio=0.5,
            queries_with_citations=5,
            tool_usage_rate=0.8,
            avg_tools_per_query=2.5,
            avg_turns=3.0,
            avg_total_tokens=5000,
            avg_duration_seconds=4.2,
            avg_duplicate_tool_calls=0.5,
            queries_with_duplicates=3,
        )
        assert metrics.total_queries == 10
        assert metrics.avg_duplicate_tool_calls == 0.5
        assert metrics.queries_with_duplicates == 3

    def test_duplicate_tool_calls_metric(self):
        """Duplicate tool call metrics are computed correctly"""
        traces = [
            QueryTrace(
                query_text="q1",
                timestamp="2026-02-23T10:00:00",
                duplicate_tool_calls=2,
            ),
            QueryTrace(
                query_text="q2",
                timestamp="2026-02-23T10:01:00",
                duplicate_tool_calls=0,
            ),
            QueryTrace(
                query_text="q3",
                timestamp="2026-02-23T10:02:00",
                duplicate_tool_calls=1,
            ),
        ]
        metrics = QueryEvaluator.evaluate_traces(traces)

        assert metrics.avg_duplicate_tool_calls == 1.0  # (2+0+1)/3
        assert metrics.queries_with_duplicates == 2  # q1 and q3

    def test_empty_traces_duplicate_metrics(self):
        """Empty traces returns zero duplicate metrics"""
        metrics = QueryEvaluator.evaluate_traces([])
        assert metrics.avg_duplicate_tool_calls == 0.0
        assert metrics.queries_with_duplicates == 0


class TestCaptureTrace:
    """Tests for agent.capture_trace() method"""

    def test_capture_trace_builds_trace(self):
        """capture_trace builds a QueryTrace from agent state"""
        from ai_assist.config import AiAssistConfig

        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={})
        agent = __import__("ai_assist.agent", fromlist=["AiAssistAgent"]).AiAssistAgent(config)

        # Simulate agent state after a query
        agent.last_tool_calls = [
            {"tool_name": "search_dci_jobs", "arguments": {"query": "status=failure"}, "result": "big result"}
        ]
        agent._turn_token_usage = [{"input_tokens": 500, "output_tokens": 100}]

        import time

        start = time.time() - 2.0  # 2 seconds ago

        trace = agent.capture_trace("What failed?", "Two jobs failed.", start, 3)

        assert trace.query_text == "What failed?"
        assert trace.response_text == "Two jobs failed."
        assert trace.turn_count == 3
        assert trace.total_input_tokens == 500
        assert trace.total_output_tokens == 100
        assert trace.duration_seconds >= 1.5  # At least 1.5s
        assert len(trace.tool_calls) == 1
        # Results should NOT be in the trace (keeps traces small)
        assert "result" not in trace.tool_calls[0]
        assert trace.pid == os.getpid()

    def test_capture_trace_uses_last_turn_count(self):
        """capture_trace reads _last_turn_count when turn_count not passed"""
        from ai_assist.config import AiAssistConfig

        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={})
        agent = __import__("ai_assist.agent", fromlist=["AiAssistAgent"]).AiAssistAgent(config)

        agent.last_tool_calls = []
        agent._turn_token_usage = []
        agent._last_turn_count = 15

        import time

        start = time.time() - 1.0

        trace = agent.capture_trace("test query", "test response", start)

        assert trace.turn_count == 15


class TestCostFields:
    """Tests for cost tracking in QueryTrace and EvalMetrics"""

    def test_query_trace_cost_field_default(self):
        trace = QueryTrace(query_text="test", timestamp="2026-01-01T00:00:00")
        assert trace.total_cost_usd == 0.0

    def test_query_trace_cost_roundtrip(self):
        trace = QueryTrace(
            query_text="test",
            timestamp="2026-01-01T00:00:00",
            total_cost_usd=0.0123,
        )
        line = trace.to_jsonl_line()
        data = json.loads(line)
        restored = QueryTrace.from_json(data)
        assert abs(restored.total_cost_usd - 0.0123) < 1e-10

    def test_backward_compat_no_cost_field(self):
        data = {
            "query_text": "old trace",
            "timestamp": "2025-01-01T00:00:00",
            "total_input_tokens": 100,
            "total_output_tokens": 50,
        }
        trace = QueryTrace.from_json(data)
        assert trace.total_cost_usd == 0.0

    def test_eval_metrics_cost_aggregation(self):
        traces = [
            QueryTrace(
                query_text="q1",
                timestamp="2026-01-01T00:00:00",
                total_cost_usd=0.01,
                total_input_tokens=100,
                total_output_tokens=50,
            ),
            QueryTrace(
                query_text="q2",
                timestamp="2026-01-01T01:00:00",
                total_cost_usd=0.03,
                total_input_tokens=200,
                total_output_tokens=100,
            ),
        ]
        metrics = QueryEvaluator.evaluate_traces(traces)
        assert abs(metrics.total_cost_usd - 0.04) < 1e-10
        assert abs(metrics.avg_cost_per_query_usd - 0.02) < 1e-10
