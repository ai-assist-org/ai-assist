"""Monitoring scheduler"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .agent import AiAssistAgent
from .config import get_config_dir
from .config_watcher import ConfigWatcher
from .event_sources import EventSourceManager
from .file_watchdog import FileWatchdog
from .knowledge_graph import KnowledgeGraph
from .schedule_loader import ScheduleLoader
from .schedule_recalculator import ScheduleRecalculator
from .state import StateManager
from .suspend_detector import SuspendDetector
from .task_runner import TaskRunner
from .tasks import TaskLoader

logger = logging.getLogger(__name__)


class MonitoringScheduler:
    """Schedule and run monitoring tasks"""

    def __init__(
        self,
        agent: AiAssistAgent,
        config,
        state_manager: StateManager,
        knowledge_graph: KnowledgeGraph | None = None,
        schedule_file: Path | None = None,
    ):
        self.agent = agent
        self.config = config
        self.state_manager = state_manager
        self.knowledge_graph = knowledge_graph

        # Use schedule_file or default location
        if schedule_file:
            self.schedule_file = schedule_file
        else:
            self.schedule_file = get_config_dir() / "schedules.json"

        self.loader = ScheduleLoader(self.schedule_file)
        self.monitors: list[TaskRunner] = []
        self.user_tasks: list[TaskRunner] = []
        self.monitor_handles: list[asyncio.Task] = []
        self.user_task_handles: list[asyncio.Task] = []
        self.running = False

        # Suspension detection and recovery
        self.suspend_detector: SuspendDetector | None = None
        self.schedule_recalculator = ScheduleRecalculator()
        self.file_watchdog: FileWatchdog | None = None
        self.config_watcher: ConfigWatcher | None = None

        # Event source manager (MQTT, DBUS, etc.)
        self.event_source_manager: EventSourceManager | None = None

        # Scheduled actions (one-shot future executions)
        self.action_manager: Any = None
        self.action_task_handle: asyncio.Task | None = None
        self.action_file_watchdog: FileWatchdog | None = None

        # Load initial schedules
        self.monitors = self._load_monitors()
        self.user_tasks = self._load_user_tasks()

    def _load_monitors(self) -> list[TaskRunner]:
        """Load monitors from JSON file as regular tasks"""
        if not self.loader:
            return []

        try:
            task_defs = self.loader.load_monitors()

            runners = []
            for task_def in task_defs:
                if task_def.enabled:
                    runner = TaskRunner(task_def, self.agent, self.state_manager)
                    runners.append(runner)
                    if task_def.is_event_triggered and task_def.trigger is not None:
                        print(f"Loaded monitor: {task_def.name} (trigger: {task_def.trigger.get('type', '?')})")
                    else:
                        print(f"Loaded monitor: {task_def.name} (interval: {task_def.interval})")
                else:
                    print(f"Skipping disabled monitor: {task_def.name}")

            return runners
        except Exception as e:
            logger.exception("Error loading monitors from %s: %s", self.schedule_file, e)
            return []

    def _load_user_tasks(self) -> list[TaskRunner]:
        """Load user-defined tasks from JSON file"""
        if not self.loader:
            return []

        try:
            # Ensure default tasks exist in the file
            self.loader.ensure_default_tasks()

            task_defs = self.loader.load_tasks()

            runners = []
            for task_def in task_defs:
                if task_def.enabled:
                    runner = TaskRunner(task_def, self.agent, self.state_manager)
                    runners.append(runner)
                    if task_def.is_event_triggered and task_def.trigger is not None:
                        print(f"Loaded task: {task_def.name} (trigger: {task_def.trigger.get('type', '?')})")
                    else:
                        print(f"Loaded task: {task_def.name} (interval: {task_def.interval})")
                else:
                    print(f"Skipping disabled task: {task_def.name}")

            return runners
        except Exception as e:
            logger.exception("Error loading tasks from %s: %s", self.schedule_file, e)
            return []

    async def reload_schedules(self):
        """Reload all schedules from JSON file (hot reload)"""
        print("\n" + "=" * 60)
        print("Reloading schedules...")
        print("=" * 60)

        try:
            # Stop event sources before reload
            await self._stop_event_sources()

            # Load new schedules
            new_monitors = self._load_monitors()
            new_tasks = self._load_user_tasks()

            # Cancel all existing tasks
            all_handles = self.monitor_handles + self.user_task_handles
            for handle in all_handles:
                handle.cancel()

            if all_handles:
                await asyncio.gather(*all_handles, return_exceptions=True)

            # Clear old handles
            self.monitor_handles.clear()
            self.user_task_handles.clear()

            # Update schedules
            self.monitors = new_monitors
            self.user_tasks = new_tasks

            # Restart timer-based monitor tasks
            for monitor in self.monitors:
                if monitor.task_def.is_event_triggered:
                    continue
                interval = (
                    0
                    if (monitor.task_def.is_time_based or monitor.task_def.is_interval_with_range)
                    else monitor.task_def.interval_seconds
                )
                task_handle = asyncio.create_task(
                    self._schedule_task(monitor.task_def.name, monitor.run, interval, task_def=monitor.task_def)
                )
                self.monitor_handles.append(task_handle)

            # Restart timer-based user tasks
            for task_runner in self.user_tasks:
                if task_runner.task_def.is_event_triggered:
                    continue
                interval = (
                    0
                    if (task_runner.task_def.is_time_based or task_runner.task_def.is_interval_with_range)
                    else task_runner.task_def.interval_seconds
                )
                task_handle = asyncio.create_task(
                    self._schedule_task(
                        task_runner.task_def.name, task_runner.run, interval, task_def=task_runner.task_def
                    )
                )
                self.user_task_handles.append(task_handle)

            # Restart event sources
            await self._start_event_sources()

            print(f"✓ Reloaded {len(new_monitors)} monitor(s) and {len(new_tasks)} task(s)")
            print("=" * 60 + "\n")

        except Exception:
            logger.exception("Failed to reload schedules; keeping existing schedules")
            print("=" * 60 + "\n")

    async def _wait_for_mcp_servers(self, timeout_seconds: float = 30.0) -> bool:
        """Wait until all configured MCP servers are connected.

        Returns True if all servers connected, False if timed out.
        """
        expected = set(self.agent.config.mcp_servers.keys())
        if not expected:
            return True

        deadline = asyncio.get_event_loop().time() + timeout_seconds
        while asyncio.get_event_loop().time() < deadline:
            connected = set(self.agent.sessions.keys())
            if expected <= connected:
                print(f"All {len(expected)} MCP servers connected")
                return True
            await asyncio.sleep(1.0)

        connected = set(self.agent.sessions.keys())
        missing = expected - connected
        logger.warning("Timed out waiting for MCP servers: %s not connected", ", ".join(missing))
        print(f"⚠️  {len(missing)} MCP server(s) not connected after {timeout_seconds}s: {', '.join(missing)}")
        return len(connected) > 0  # Proceed if at least some are connected

    async def start(self):
        """Start the monitoring loop"""
        self.running = True
        print("Starting monitoring scheduler...")
        print(f"State directory: {self.state_manager.state_dir}")

        removed = self.state_manager.cleanup_expired_cache()
        if removed:
            print(f"Cleaned up {removed} expired cache entries")

        # Wait for MCP servers before running any tasks
        await self._wait_for_mcp_servers()

        await self._run_missed_tasks_at_startup()

        tasks = []

        timer_monitor_count = 0
        for monitor in self.monitors:
            if monitor.task_def.is_event_triggered:
                continue
            interval = (
                0
                if (monitor.task_def.is_time_based or monitor.task_def.is_interval_with_range)
                else monitor.task_def.interval_seconds
            )
            task_handle = asyncio.create_task(
                self._schedule_task(monitor.task_def.name, monitor.run, interval, task_def=monitor.task_def)
            )
            tasks.append(task_handle)
            self.monitor_handles.append(task_handle)
            timer_monitor_count += 1

        if timer_monitor_count:
            print(f"Scheduled {timer_monitor_count} timer-based monitors")

        timer_task_count = 0
        for task_runner in self.user_tasks:
            if task_runner.task_def.is_event_triggered:
                continue
            interval = (
                0
                if (task_runner.task_def.is_time_based or task_runner.task_def.is_interval_with_range)
                else task_runner.task_def.interval_seconds
            )
            task_handle = asyncio.create_task(
                self._schedule_task(task_runner.task_def.name, task_runner.run, interval, task_def=task_runner.task_def)
            )
            tasks.append(task_handle)
            self.user_task_handles.append(task_handle)
            timer_task_count += 1

        if timer_task_count:
            print(f"Scheduled {timer_task_count} timer-based tasks")

        # Start event source manager for event-triggered tasks
        await self._start_event_sources()

        # Watch schedules.json for changes using OS-level file watching
        if self.schedule_file:
            self.file_watchdog = FileWatchdog(self.schedule_file, self.reload_schedules, debounce_seconds=0.5)
            await self.file_watchdog.start()
            print(f"Watching {self.schedule_file} for changes")

        # Start config watching (mcp_servers.yaml, identity.yaml, installed-skills.json)
        self.config_watcher = ConfigWatcher(self.agent)
        await self.config_watcher.start()

        # Start suspension detection
        self.suspend_detector = SuspendDetector(
            suspend_threshold_seconds=30.0,
            poll_interval_seconds=5.0,
        )
        suspend_task = asyncio.create_task(self.suspend_detector.watch(self._handle_wake_event))
        tasks.append(suspend_task)
        print("Suspension detection enabled")

        # Start scheduled action executor
        from ai_assist.scheduled_actions import ScheduledActionManager

        action_file = get_config_dir() / "scheduled-actions.json"
        self.action_manager = ScheduledActionManager(action_file, self.agent)
        await self.action_manager.load_actions()

        # Start executor (event-driven, no polling)
        action_task = asyncio.create_task(self.action_manager.start_executor())
        tasks.append(action_task)
        self.action_task_handle = action_task
        print("Scheduled action executor started (event-driven)")

        # Watch scheduled-actions.json for changes
        if action_file.exists() or True:  # Watch even if doesn't exist yet
            self.action_file_watchdog = FileWatchdog(
                action_file, self.action_manager.on_file_change, debounce_seconds=0.5
            )
            await self.action_file_watchdog.start()
            print(f"Watching {action_file} for changes...")

        await asyncio.gather(*tasks)

    async def _start_event_sources(self) -> None:
        """Configure and start event sources for event-triggered tasks"""
        event_configs = self.loader.load_event_source_configs()

        triggered_tasks: list[tuple[str, dict, TaskRunner]] = []
        for runner in self.monitors + self.user_tasks:
            if runner.task_def.is_event_triggered and runner.task_def.trigger is not None:
                triggered_tasks.append((runner.task_def.name, runner.task_def.trigger, runner))

        if not triggered_tasks:
            return

        if not event_configs:
            logger.warning(
                "%d event-triggered task(s) configured but no event_sources defined in schedules.json",
                len(triggered_tasks),
            )
            return

        self.event_source_manager = EventSourceManager()
        self.event_source_manager.register_available_sources(event_configs)
        self.event_source_manager.configure(triggered_tasks)
        await self.event_source_manager.start()
        print(f"Started {len(self.event_source_manager._sources)} event source(s) for {len(triggered_tasks)} task(s)")

    async def _stop_event_sources(self) -> None:
        """Stop event source manager"""
        if self.event_source_manager:
            await self.event_source_manager.stop()
            self.event_source_manager = None

    async def _schedule_task(self, name: str, task_func, interval: int, task_def=None):
        """Schedule a periodic task"""
        while self.running:
            try:
                if task_def and task_def.is_interval_with_range:
                    schedule = TaskLoader.parse_interval_with_range(task_def.interval)
                    next_run = TaskLoader.calculate_next_interval_run(schedule)
                    now = datetime.now()

                    wait_seconds = (next_run - now).total_seconds()
                    if wait_seconds > 0:
                        print(f"{name}: Next run at {next_run.strftime('%Y-%m-%d %H:%M')}")
                        await asyncio.sleep(wait_seconds)
                elif task_def and task_def.is_time_based:
                    schedule = TaskLoader.parse_time_schedule(task_def.interval)
                    next_run = TaskLoader.calculate_next_run(schedule)
                    now = datetime.now()

                    wait_seconds = (next_run - now).total_seconds()
                    if wait_seconds > 0:
                        print(f"{name}: Next run at {next_run.strftime('%Y-%m-%d %H:%M')}")
                        await asyncio.sleep(wait_seconds)

                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running {name}...")
                results = await task_func()

                from .task_runner import TaskResult

                if isinstance(results, TaskResult):
                    if results.success:
                        print(f"{name}: ✓ Completed")
                        print(f"\n{results.output}")
                    else:
                        print(f"{name}: ✗ Failed - {results.output}")
                elif results:
                    self._report_results(name, results)
                else:
                    print(f"{name}: No updates")

            except asyncio.CancelledError:
                # Task was cancelled (e.g., during reload) - exit gracefully
                break
            except Exception as e:
                logger.exception("Error in %s: %s", name, e)

            if not (task_def and (task_def.is_time_based or task_def.is_interval_with_range)):
                try:
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    # Cancelled during sleep - exit gracefully
                    break

    def _report_results(self, monitor_name: str, results: list[dict]):
        """Report monitoring results"""
        print(f"\n{'='*60}")
        print(f"{monitor_name} Report")
        print(f"{'='*60}")

        for result in results:
            if "monitor" in result:
                print(f"\nMonitor: {result['monitor']}")

            print(f"Time: {result['timestamp']}")
            print(f"\n{result['summary']}")
            print(f"{'-'*60}")

    async def _wait_for_network(self, timeout_seconds: float = 60.0, check_interval: float = 2.0) -> bool:
        """Wait until DNS resolution works. Returns True if network is available, False if timed out."""
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout_seconds
        while loop.time() < deadline:
            try:
                await loop.getaddrinfo("google.com", 443)
                return True
            except OSError:
                await asyncio.sleep(check_interval)
        return False

    async def _run_missed_tasks_at_startup(self, now: datetime | None = None) -> None:
        """Run any time-based tasks missed in the last 24 hours (startup/reboot catchup).

        Checks each time-based task to see if it should have run since its last
        recorded run. Skips tasks that already ran after their last scheduled time.
        """
        if now is None:
            now = datetime.now()

        lookback = now - timedelta(hours=24)
        missed = []

        for runner in self.monitors + self.user_tasks:
            if runner.task_def.is_event_triggered or not runner.task_def.is_time_based:
                continue
            assert runner.task_def.interval is not None
            try:
                schedule = TaskLoader.parse_time_schedule(runner.task_def.interval)
            except ValueError:
                continue

            last_scheduled = TaskLoader.calculate_next_run(schedule, from_time=lookback)
            if last_scheduled > now:
                continue  # Not due within the last 24h

            last_run = runner.get_last_run()
            if last_run and last_run >= last_scheduled:
                # Only skip if the last run actually succeeded
                state = runner.state_manager.get_monitor_state(runner.state_key)
                if state.last_results.get("last_success", True):
                    continue  # Already ran successfully since last scheduled time
                # Last run failed — retry

            missed.append((runner, last_scheduled))

        if not missed:
            return

        print(f"Catching up {len(missed)} missed task(s) at startup...")
        if not await self._wait_for_network():
            logger.warning("Network not available at startup, skipping missed task catchup")
            return

        for runner, last_scheduled in missed:
            print(
                f"Running missed task: {runner.task_def.name} (was due at {last_scheduled.strftime('%Y-%m-%d %H:%M')})"
            )
            try:
                await runner.run()
            except Exception as e:
                logger.exception("Error running missed task %s at startup: %s", runner.task_def.name, e)

    async def _handle_wake_event(self, wall_jump_seconds: float, now: datetime | None = None) -> None:
        """Handle system wake event after suspension.

        Args:
            wall_jump_seconds: How many seconds the wall clock jumped
            now: Current time (for testing, defaults to datetime.now())
        """
        if now is None:
            now = datetime.now()

        print("\n" + "=" * 60)
        print(f"⚠️  Suspension detected: {abs(wall_jump_seconds):.0f} seconds")
        print("Checking for missed scheduled runs...")
        print("=" * 60)

        print("Waiting for network connectivity...")
        if not await self._wait_for_network():
            logger.warning("Network not available after wake, skipping missed task recovery")
            print("⚠️  Network not available after timeout, skipping missed task recovery")
            return

        # Collect all tasks — recalculator filters to time-based ones internally
        all_tasks = [
            type("Adapter", (), {"schedule": runner.task_def.interval, "execute": runner.run})()
            for runner in self.monitors + self.user_tasks
        ]

        await self.schedule_recalculator.handle_wake_event(wall_jump_seconds, all_tasks, now=now)

        print("✓ Suspension recovery complete")
        print("=" * 60 + "\n")

    async def stop(self):
        """Stop the monitoring loop"""
        self.running = False
        print("Stopping monitoring scheduler...")

        # Stop event sources
        await self._stop_event_sources()

        # Stop scheduled action executor
        if self.action_task_handle:
            self.action_task_handle.cancel()
            try:
                await self.action_task_handle
            except asyncio.CancelledError:
                pass

        # Stop action file watchdog
        if self.action_file_watchdog:
            await self.action_file_watchdog.stop()

        # Stop file watchdog
        if self.file_watchdog:
            await self.file_watchdog.stop()

        # Stop config watcher
        if self.config_watcher:
            await self.config_watcher.stop()
