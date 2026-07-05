"""Tests for file event source"""

import asyncio

import pytest

from ai_assist.event_source_file import FileEventSource, _has_glob, _path_matches_pattern


@pytest.fixture
def file_source():
    return FileEventSource({})


class TestFileEventSource:
    def test_name(self, file_source):
        assert file_source.name == "file"

    def test_subscribe(self, file_source):
        file_source.subscribe("task1", {"type": "file", "path": "/tmp/test.txt"})
        assert "/tmp/test.txt" in file_source._subscriptions
        assert file_source._subscriptions["/tmp/test.txt"] == ["task1"]

    def test_subscribe_multiple_tasks_same_path(self, file_source):
        file_source.subscribe("task1", {"type": "file", "path": "/tmp/test.txt"})
        file_source.subscribe("task2", {"type": "file", "path": "/tmp/test.txt"})
        assert file_source._subscriptions["/tmp/test.txt"] == ["task1", "task2"]

    def test_unsubscribe_all(self, file_source):
        file_source.subscribe("task1", {"type": "file", "path": "/tmp/a"})
        file_source.subscribe("task2", {"type": "file", "path": "/tmp/b"})
        file_source.unsubscribe_all()
        assert len(file_source._subscriptions) == 0

    def test_config_defaults(self):
        source = FileEventSource({})
        assert source.debounce_seconds == 1.0

    def test_config_custom_debounce(self):
        source = FileEventSource({"debounce_seconds": 2.5})
        assert source.debounce_seconds == 2.5

    @pytest.mark.asyncio
    async def test_stop_without_start(self, file_source):
        await file_source.stop()

    @pytest.mark.asyncio
    async def test_file_change_dispatches_event(self, tmp_path):
        test_file = tmp_path / "watched.txt"
        test_file.write_text("initial")

        dispatched = []

        async def mock_dispatch(task_name, event):
            dispatched.append((task_name, event))

        source = FileEventSource({"debounce_seconds": 0.1})
        source.subscribe("handler", {"type": "file", "path": str(test_file)})
        await source.start(mock_dispatch)

        try:
            test_file.write_text("changed")
            await asyncio.sleep(0.5)
        finally:
            await source.stop()

        assert len(dispatched) >= 1
        task_name, event = dispatched[0]
        assert task_name == "handler"
        assert event.source_type == "file"
        assert event.metadata["path"] == str(test_file)
        assert event.metadata["filename"] == "watched.txt"

    @pytest.mark.asyncio
    async def test_dir_change_dispatches_event(self, tmp_path):
        dispatched = []

        async def mock_dispatch(task_name, event):
            dispatched.append((task_name, event))

        source = FileEventSource({"debounce_seconds": 0.1})
        source.subscribe("handler", {"type": "file", "path": str(tmp_path)})
        await source.start(mock_dispatch)

        try:
            new_file = tmp_path / "newfile.txt"
            new_file.write_text("hello")
            await asyncio.sleep(0.5)
        finally:
            await source.stop()

        assert len(dispatched) >= 1
        assert dispatched[0][1].metadata["filename"] == "newfile.txt"

    @pytest.mark.asyncio
    async def test_glob_pattern_filters(self, tmp_path):
        dispatched = []

        async def mock_dispatch(task_name, event):
            dispatched.append((task_name, event))

        source = FileEventSource({"debounce_seconds": 0.1})
        source.subscribe("handler", {"type": "file", "path": str(tmp_path / "*.txt")})
        await source.start(mock_dispatch)

        try:
            (tmp_path / "match.txt").write_text("yes")
            (tmp_path / "skip.log").write_text("no")
            await asyncio.sleep(0.5)
        finally:
            await source.stop()

        paths = [e.metadata["filename"] for _, e in dispatched]
        assert "match.txt" in paths
        assert "skip.log" not in paths


class TestPathMatching:
    def test_has_glob(self):
        assert _has_glob("/tmp/*.txt")
        assert _has_glob("/tmp/file?.log")
        assert _has_glob("/tmp/[ab].txt")
        assert not _has_glob("/tmp/file.txt")
        assert not _has_glob("/tmp/dir")

    def test_glob_match(self):
        assert _path_matches_pattern("/tmp/report.pdf", "/tmp/*.pdf")
        assert not _path_matches_pattern("/tmp/report.txt", "/tmp/*.pdf")

    def test_glob_question_mark(self):
        assert _path_matches_pattern("/tmp/file1.txt", "/tmp/file?.txt")
        assert not _path_matches_pattern("/tmp/file10.txt", "/tmp/file?.txt")

    def test_dir_match(self, tmp_path):
        child = tmp_path / "somefile.txt"
        child.write_text("")
        assert _path_matches_pattern(str(child), str(tmp_path))

    def test_dir_no_match_outside(self, tmp_path):
        assert not _path_matches_pattern("/other/file.txt", str(tmp_path))
