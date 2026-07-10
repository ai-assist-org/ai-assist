"""File/directory event source — watches filesystem paths for changes"""

import asyncio
import fnmatch
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler

from .event_sources import EventContext, EventSource

logger = logging.getLogger(__name__)


class FileEventSource(EventSource):
    """Event source that watches files and directories for changes"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.debounce_seconds = config.get("debounce_seconds", 1.0)
        self._subscriptions: dict[str, list[str]] = {}  # path_pattern -> [task_name, ...]
        self._dispatch: Callable[[str, EventContext], Awaitable[None]] | None = None
        self._handlers: list[_FileHandler] = []
        self._observer_refs: list[str] = []

    @property
    def name(self) -> str:
        return "file"

    def subscribe(self, task_name: str, trigger_config: dict[str, Any]) -> None:
        path = trigger_config["path"]
        self._subscriptions.setdefault(path, []).append(task_name)

    def unsubscribe_all(self) -> None:
        self._subscriptions.clear()

    async def start(self, dispatch: Callable[[str, EventContext], Awaitable[None]]) -> None:
        self._dispatch = dispatch
        loop = asyncio.get_running_loop()

        watch_dirs: dict[str, list[tuple[str, list[str]]]] = {}

        for path_pattern, task_names in self._subscriptions.items():
            expanded = str(Path(path_pattern).expanduser())
            p = Path(expanded)

            if p.is_dir() or (not p.exists() and not _has_glob(expanded)):
                watch_dir = expanded
            else:
                watch_dir = str(p.parent)

            watch_dirs.setdefault(watch_dir, []).append((expanded, task_names))

        for watch_dir, patterns in watch_dirs.items():
            handler = _FileHandler(
                patterns=patterns,
                dispatch=dispatch,
                debounce_seconds=self.debounce_seconds,
                loop=loop,
            )
            self._handlers.append(handler)

            from .file_watchdog import _get_shared_observer

            self._observer_refs.append(watch_dir)
            observer = _get_shared_observer(watch_dir)
            observer.schedule(handler, watch_dir, recursive=False)
            if not observer.is_alive():
                try:
                    observer.start()
                except OSError:
                    logger.warning(
                        "Failed to start file watcher for %s (inotify limit reached)",
                        watch_dir,
                    )

        if self._subscriptions:
            logger.info("File watcher started for %d path(s)", len(self._subscriptions))

    async def stop(self) -> None:
        from .file_watchdog import _release_shared_observer

        for handler in self._handlers:
            await handler.cancel_pending()
        self._handlers.clear()

        for watch_dir in self._observer_refs:
            _release_shared_observer(watch_dir)
        self._observer_refs.clear()


def _has_glob(path: str) -> bool:
    return any(c in path for c in ("*", "?", "["))


def _path_matches_pattern(file_path: str, pattern: str) -> bool:
    p = Path(pattern)
    if p.is_dir():
        return Path(file_path).parent == p or file_path.startswith(str(p) + "/")
    if _has_glob(pattern):
        return fnmatch.fnmatch(file_path, pattern)
    return str(Path(file_path).resolve()) == str(p.resolve()) if Path(file_path).exists() else file_path == pattern


class _FileHandler(FileSystemEventHandler):
    """Watchdog handler that dispatches file events with debouncing."""

    def __init__(
        self,
        patterns: list[tuple[str, list[str]]],
        dispatch: Callable[[str, EventContext], Awaitable[None]],
        debounce_seconds: float,
        loop: asyncio.AbstractEventLoop,
    ):
        super().__init__()
        self.patterns = patterns
        self._dispatch_fn = dispatch
        self.debounce_seconds = debounce_seconds
        self.loop = loop
        self._debounce_tasks: dict[str, asyncio.Task[None]] = {}

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._handle(event, "file_modified")

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._handle(event, "file_created")

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        if event.dest_path:
            self._handle_path(str(event.dest_path), "file_created")

    def _handle(self, event: FileSystemEvent, event_type: str) -> None:
        self._handle_path(str(event.src_path), event_type)

    def _handle_path(self, file_path: str, event_type: str) -> None:
        for pattern, task_names in self.patterns:
            if _path_matches_pattern(file_path, pattern):
                for task_name in task_names:
                    key = f"{task_name}:{file_path}"
                    self.loop.call_soon_threadsafe(self._schedule_dispatch, key, task_name, file_path, event_type)

    def _schedule_dispatch(self, key: str, task_name: str, file_path: str, event_type: str) -> None:
        if key in self._debounce_tasks and not self._debounce_tasks[key].done():
            self._debounce_tasks[key].cancel()
        self._debounce_tasks[key] = self.loop.create_task(
            self._debounced_dispatch(key, task_name, file_path, event_type)
        )

    async def _debounced_dispatch(self, key: str, task_name: str, file_path: str, event_type: str) -> None:
        try:
            await asyncio.sleep(self.debounce_seconds)
            p = Path(file_path)
            size = p.stat().st_size if p.exists() else 0
            event = EventContext(
                source_type="file",
                event_type=event_type,
                payload=f"File {event_type.split('_')[1]}: {file_path}",
                metadata={
                    "path": file_path,
                    "filename": p.name,
                    "size": size,
                },
                timestamp=datetime.now(),
            )
            await self._dispatch_fn(task_name, event)
        except asyncio.CancelledError:
            pass
        finally:
            self._debounce_tasks.pop(key, None)

    async def cancel_pending(self) -> None:
        for task in self._debounce_tasks.values():
            if not task.done():
                task.cancel()
        if self._debounce_tasks:
            await asyncio.gather(*self._debounce_tasks.values(), return_exceptions=True)
        self._debounce_tasks.clear()
