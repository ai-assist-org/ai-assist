"""Tests for state persistence and caching"""

import json
import time
from datetime import datetime, timedelta

import pytest

from ai_assist.state import MonitorState, StateManager


@pytest.fixture
def manager(tmp_path):
    return StateManager(state_dir=tmp_path / "state")


def test_monitor_state_from_dict_full():
    data = {
        "last_check": "2026-01-01T12:00:00",
        "seen_items": ["a", "b"],
        "last_results": {"count": 3},
        "metadata": {"source": "test"},
    }
    state = MonitorState.from_dict(data)

    assert state.last_check == datetime(2026, 1, 1, 12, 0, 0)
    assert state.seen_items == {"a", "b"}
    assert state.last_results == {"count": 3}
    assert state.metadata == {"source": "test"}


def test_monitor_state_from_dict_empty():
    state = MonitorState.from_dict({})

    assert state.last_check is None
    assert state.seen_items == set()
    assert state.last_results == {}


def test_monitor_state_round_trip():
    original = MonitorState(
        last_check=datetime(2026, 1, 2, 8, 30, 0),
        seen_items={"x", "y", "z"},
        last_results={"failures": 2},
    )
    dumped = original.model_dump()

    assert dumped["last_check"] == "2026-01-02T08:30:00"
    assert sorted(dumped["seen_items"]) == ["x", "y", "z"]

    restored = MonitorState.from_dict(dumped)
    assert restored.last_check == original.last_check
    assert restored.seen_items == original.seen_items


def test_monitor_state_serialize_none_last_check():
    state = MonitorState()
    assert state.model_dump()["last_check"] is None


def test_get_monitor_state_default(manager):
    state = manager.get_monitor_state("fresh")

    assert isinstance(state, MonitorState)
    assert state.last_check is None
    assert state.seen_items == set()


def test_get_monitor_state_loads_from_file(manager):
    state = MonitorState(seen_items={"item-1"}, last_check=datetime(2026, 3, 1, 10, 0, 0))
    manager.save_monitor_state("loader", state)

    fresh_manager = StateManager(state_dir=manager.state_dir)
    loaded = fresh_manager.get_monitor_state("loader")

    assert loaded.seen_items == {"item-1"}
    assert loaded.last_check == datetime(2026, 3, 1, 10, 0, 0)


def test_get_monitor_state_cached_in_memory(manager):
    first = manager.get_monitor_state("mon")
    second = manager.get_monitor_state("mon")
    assert first is second


def test_save_monitor_state_persists(manager):
    state = MonitorState(last_results={"count": 5})
    manager.save_monitor_state("saver", state)

    state_file = manager.state_dir / "saver.json"
    assert state_file.exists()
    on_disk = json.loads(state_file.read_text())
    assert on_disk["last_results"] == {"count": 5}


def test_update_monitor_sets_check_and_results(manager):
    manager.update_monitor("upd", {"count": 7}, seen_items={"new-1", "new-2"})
    state = manager.get_monitor_state("upd")

    assert state.last_check is not None
    assert state.last_results == {"count": 7}
    assert state.seen_items == {"new-1", "new-2"}


def test_update_monitor_without_seen_items(manager):
    manager.update_monitor("upd2", {"count": 1})
    state = manager.get_monitor_state("upd2")

    assert state.last_results == {"count": 1}
    assert state.seen_items == set()


def test_get_new_items(manager):
    manager.update_monitor("nm", {}, seen_items={"a", "b"})
    new = manager.get_new_items("nm", {"a", "b", "c"})

    assert new == {"c"}


def test_cache_and_get_query(manager):
    manager.cache_query_result("my query", {"answer": 42})
    result = manager.get_cached_query("my query")

    assert result == {"answer": 42}


def test_get_cached_query_missing(manager):
    assert manager.get_cached_query("never cached") is None


def test_cache_expiry_monotonic(manager):
    manager.cache_query_result("expiring", {"data": 1}, ttl_seconds=300)
    cache_file = manager.cache_dir / f"{manager._sanitize_key('expiring')}.json"

    data = json.loads(cache_file.read_text())
    data["cached_at_mono"] = time.monotonic() - 10000
    cache_file.write_text(json.dumps(data))

    assert manager.get_cached_query("expiring") is None
    assert not cache_file.exists()


def test_get_cached_query_wall_clock_fallback(manager):
    cache_file = manager.cache_dir / f"{manager._sanitize_key('old fmt')}.json"
    old_time = (datetime.now() - timedelta(seconds=1000)).isoformat()
    cache_file.write_text(json.dumps({"result": {"v": 1}, "timestamp": old_time, "ttl_seconds": 300}))

    assert manager.get_cached_query("old fmt") is None
    assert not cache_file.exists()


def test_get_cached_query_wall_clock_valid(manager):
    cache_file = manager.cache_dir / f"{manager._sanitize_key('recent fmt')}.json"
    recent = datetime.now().isoformat()
    cache_file.write_text(json.dumps({"result": {"v": 2}, "timestamp": recent, "ttl_seconds": 300}))

    assert manager.get_cached_query("recent fmt") == {"v": 2}


def test_save_and_load_conversation_context(manager):
    manager.save_conversation_context("session", {"topic": "deployment"})
    loaded = manager.load_conversation_context("session")

    assert loaded == {"topic": "deployment"}


def test_load_conversation_context_missing(manager):
    assert manager.load_conversation_context("nope") is None


def test_sanitize_key():
    assert StateManager._sanitize_key("simple") == "simple"
    assert StateManager._sanitize_key("a b/c:d") == "a_b_c_d"
    assert StateManager._sanitize_key("keep-under_score") == "keep-under_score"


def test_get_stats(manager):
    manager.save_monitor_state("m1", MonitorState())
    manager.cache_query_result("q1", {"a": 1})
    manager.append_history("m1", {"count": 1})

    stats = manager.get_stats()

    assert stats["monitors"] == 1
    assert stats["cached_queries"] == 1
    assert stats["history_files"] == 1
    assert stats["state_dir"] == str(manager.state_dir)


def test_get_stats_no_history(manager):
    stats = manager.get_stats()
    assert stats["history_files"] == 0


def test_history_round_trip(manager):
    for i in range(3):
        manager.append_history("hist", {"index": i})

    history = manager.get_history("hist", limit=2)

    assert len(history) == 2
    assert history[0]["result"] == {"index": 1}
    assert history[1]["result"] == {"index": 2}


def test_get_history_missing(manager):
    assert manager.get_history("absent") == []


def test_cleanup_expired_cache(manager):
    manager.cache_query_result("keep", {"v": 1}, ttl_seconds=300)
    manager.cache_query_result("drop", {"v": 2}, ttl_seconds=300)

    drop_file = manager.cache_dir / f"{manager._sanitize_key('drop')}.json"
    data = json.loads(drop_file.read_text())
    data["cached_at_mono"] = time.monotonic() - 10000
    drop_file.write_text(json.dumps(data))

    removed = manager.cleanup_expired_cache()

    assert removed == 1
    assert not drop_file.exists()
    assert manager.get_cached_query("keep") == {"v": 1}


def test_cleanup_expired_cache_wall_clock(manager):
    cache_file = manager.cache_dir / f"{manager._sanitize_key('wc old')}.json"
    old_time = (datetime.now() - timedelta(seconds=1000)).isoformat()
    cache_file.write_text(json.dumps({"result": {}, "timestamp": old_time, "ttl_seconds": 300}))

    removed = manager.cleanup_expired_cache()

    assert removed == 1
    assert not cache_file.exists()


def test_cleanup_removes_corrupt_cache(manager):
    corrupt = manager.cache_dir / "corrupt.json"
    corrupt.write_text("{not valid json")

    removed = manager.cleanup_expired_cache()

    assert removed == 1
    assert not corrupt.exists()
