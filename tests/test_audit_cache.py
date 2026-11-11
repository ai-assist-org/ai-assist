"""Tests for workflows/self-improvement/lib/audit-cache.py"""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "workflows" / "self-improvement" / "lib" / "audit-cache.py"


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("audit_cache", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _turn(turn, read, write, sha=None, uncached=50):
    entry = {
        "turn": turn,
        "input_tokens": uncached,
        "output_tokens": 100,
        "cache_creation_input_tokens": write,
        "cache_read_input_tokens": read,
    }
    if sha:
        entry["prefix_sha"] = sha
    return entry


def test_hit_is_not_a_miss(audit):
    turns = [_turn(0, 0, 60000, "aaa"), _turn(1, 60050, 800, "aaa")]
    assert audit.find_misses(turns) == []


def test_full_rewrite_with_same_prefix_is_cache_absent(audit):
    turns = [_turn(0, 0, 60000, "aaa"), _turn(1, 0, 61000, "aaa"), _turn(2, 61050, 500, "aaa")]
    misses = audit.find_misses(turns)
    assert misses == [{"turn": 1, "kind": "cache-absent", "rewritten": 60050}]


def test_rewrite_after_prefix_change_is_prefix_changed(audit):
    turns = [_turn(0, 0, 60000, "aaa"), _turn(1, 0, 61000, "bbb")]
    assert audit.find_misses(turns)[0]["kind"] == "prefix-changed"


def test_rewrite_without_fingerprint_is_unclassified(audit):
    turns = [_turn(0, 0, 60000), _turn(1, 0, 61000)]
    assert audit.find_misses(turns)[0]["kind"] == "unclassified"


def test_small_prompt_below_cache_minimum_is_ignored(audit):
    turns = [_turn(0, 0, 0, uncached=1500), _turn(1, 0, 0, uncached=1700)]
    assert audit.find_misses(turns) == []


def test_load_queries_filters_model_and_date_and_parses_repr(audit, tmp_path):
    traces = tmp_path / "query_traces.jsonl"
    rows = [
        {"timestamp": "2026-11-01T10:00:00", "model": "claude-haiku-4-5", "token_usage": [_turn(0, 0, 1)]},
        {"timestamp": "2026-11-10T10:00:00", "model": "claude-haiku-4-5", "token_usage": str([_turn(0, 0, 2)])},
        {"timestamp": "2026-11-10T11:00:00", "model": "other/model", "token_usage": [_turn(0, 0, 3)]},
    ]
    traces.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")

    queries = audit.load_queries(traces, since="2026-11-05", model="claude")

    assert [q["token_usage"][0]["cache_creation_input_tokens"] for q in queries] == [2]
