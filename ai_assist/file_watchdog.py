"""OS-level file watching using watchdog library.

This module provides efficient file watching using OS-level events
(inotify on Linux, FSEvents on macOS, ReadDirectoryChanges on Windows).

A single Observer is shared per watched directory to avoid FSEvents
errors on macOS where the same path cannot be scheduled twice.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

# Shared observers per directory to avoid macOS FSEvents "already scheduled" errors
_shared_observers: dict[str, tuple[Any, int]] = {}  # path -> (observer, refcount)


def _get_shared_observer(watch_path: str) -> Any:
    """Get or create a shared Observer for a directory."""
    if watch_path in _shared_observers:
        observer, refcount = _shared_observers[watch_path]
        _shared_observers[watch_path] = (observer, refcount + 1)
        return observer

    observer = Observer()
    _shared_observers[watch_path] = (observer, 1)
    return observer


def _release_shared_observer(watch_path: str) -> None:
    """Release a reference to a shared Observer, stopping it when no longer needed."""
    if watch_path not in _shared_observers:
        return

    observer, refcount = _shared_observers[watch_path]
    if refcount <= 1:
        observer.stop()
        observer.join(timeout=2.0)
        del _shared_observers[watch_path]
    else:
        _shared_observers[watch_path] = (observer, refcount - 1)


class FileWatchdog:
    """Watches a specific file for modifications using OS-level events.

    Uses the watchdog library for efficient, kernel-level file change
    detection. Includes debouncing to avoid triggering callbacks for
    rapid successive changes.

    Multiple FileWatchdog instances watching files in the same directory
    share a single Observer to avoid macOS FSEvents errors.

    Attributes:
        file_path: Path to the file to watch
        callback: Async function to call when file changes
        debounce_seconds: Wait time after last change before triggering callback
    """

    def __init__(
        self,
        file_path: Path,
        callback: Callable[[], Awaitable[None]],
        debounce_seconds: float = 0.5,
    ):
        """Initialize file watchdog.

        Args:
            file_path: Path to the file to watch
            callback: Async function to call when file changes
            debounce_seconds: Wait time after last change before triggering.
                Default 0.5s to avoid multiple triggers for atomic writes.
        """
        self.file_path = Path(file_path)
        self.callback = callback
        self.debounce_seconds = debounce_seconds

        self._watch_path: str | None = None
        self._handler: _DebounceHandler | None = None
        self._running = False

    async def start(self) -> None:
        """Start watching the file for changes."""
        if self._running:
            return

        self._running = True

        # Get current event loop to pass to handler
        loop = asyncio.get_running_loop()

        # Create event handler
        self._handler = _DebounceHandler(
            target_file=self.file_path,
            callback=self.callback,
            debounce_seconds=self.debounce_seconds,
            loop=loop,
        )

        # Get or create shared observer for this directory
        self._watch_path = str(self.file_path.parent)
        observer = _get_shared_observer(self._watch_path)
        observer.schedule(self._handler, self._watch_path, recursive=False)
        if not observer.is_alive():
            try:
                observer.start()
            except OSError:
                logger.warning(
                    "Failed to start file watcher for %s (inotify limit reached). "
                    "File change detection disabled for this path.",
                    self.file_path,
                )
                _release_shared_observer(self._watch_path)
                self._watch_path = None
                self._handler = None
                self._running = False

    async def stop(self) -> None:
        """Stop watching the file."""
        if not self._running:
            return

        self._running = False

        # Cancel pending debounce in handler
        if self._handler:
            await self._handler.cancel_pending()

        # Release shared observer
        if self._watch_path:
            _release_shared_observer(self._watch_path)
            self._watch_path = None
            self._handler = None


class _DebounceHandler(FileSystemEventHandler):
    """Internal handler that debounces file change events."""

    def __init__(
        self,
        target_file: Path,
        callback: Callable[[], Awaitable[None]],
        debounce_seconds: float,
        loop: asyncio.AbstractEventLoop,
    ):
        super().__init__()
        self.target_file = target_file
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self.loop = loop
        self._debounce_task: asyncio.Task | None = None
        self._last_mtime: float = target_file.stat().st_mtime if target_file.exists() else 0.0

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return

        # Check if this is our target file
        event_path = Path(str(event.src_path)).resolve()
        target_path = self.target_file.resolve()

        if event_path != target_path:
            return

        # macOS FSEvents emits directory-level events; guard with mtime to filter false positives.
        try:
            current_mtime = self.target_file.stat().st_mtime
        except OSError:
            return
        if current_mtime == self._last_mtime:
            return
        self._last_mtime = current_mtime

        print(f"Detected change in {self.target_file.name}", flush=True)
        # Trigger debounced callback
        self._trigger_debounced()

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        # Treat creation as modification for our purposes
        self.on_modified(event)

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move/rename events (atomic saves)."""
        if event.is_directory:
            return

        # Check if file was moved TO our target path (atomic save pattern)
        # dest_path is always present on FileSystemEvent but is empty string for non-move events
        if event.dest_path:
            dest_path = Path(str(event.dest_path)).resolve()
            target_path = self.target_file.resolve()

            if dest_path == target_path:
                print(f"Detected change in {self.target_file.name}", flush=True)
                self._trigger_debounced()

    def _trigger_debounced(self) -> None:
        """Trigger callback after debounce period."""
        # Cancel existing debounce task
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()

        # Schedule new debounce task in the event loop
        # Use call_soon_threadsafe since observer runs in separate thread
        self.loop.call_soon_threadsafe(self._schedule_callback)

    def _schedule_callback(self) -> None:
        """Schedule callback in event loop (must be called from loop thread)."""
        self._debounce_task = self.loop.create_task(self._debounced_callback())

    async def _debounced_callback(self) -> None:
        """Wait for debounce period then call callback."""
        try:
            await asyncio.sleep(self.debounce_seconds)
            await self.callback()
        except asyncio.CancelledError:
            pass

    async def cancel_pending(self) -> None:
        """Cancel any pending debounce tasks."""
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
            try:
                await self._debounce_task
            except asyncio.CancelledError:
                pass
