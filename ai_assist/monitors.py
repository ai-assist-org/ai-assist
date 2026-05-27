"""Monitoring scheduler"""

import asyncio
import logging
from pathlib import Path

from .action_scheduler import ActionScheduler
from .agent import AiAssistAgent
from .config import get_config_dir
from .config_watcher import ConfigWatcher
from .file_watchdog import FileWatchdog
from .knowledge_graph import KnowledgeGraph
from .state import StateManager
from .suspend_detector import SuspendDetector

logger = logging.getLogger(__name__)


class MonitoringScheduler:
    """Schedule and run monitoring tasks via unified ActionScheduler"""

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
        self.running = False

        # Suspension detection and recovery
        self.suspend_detector: SuspendDetector | None = None
        self.config_watcher: ConfigWatcher | None = None

        # Unified action scheduler (event-schedules.json)
        config_dir = get_config_dir() if not schedule_file else schedule_file.parent
        event_schedules_file = config_dir / "event-schedules.json"
        self.action_scheduler = ActionScheduler(agent, state_manager, event_schedules_file)
        self.action_scheduler_file_watchdog: FileWatchdog | None = None

    async def _wait_for_mcp_servers(self, timeout_seconds: float = 30.0) -> bool:
        """Wait until all configured MCP servers are connected."""
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
        return len(connected) > 0

    async def start(self):
        """Start the monitoring loop"""
        self.running = True
        print("Starting monitoring scheduler...")
        print(f"State directory: {self.state_manager.state_dir}")

        removed = self.state_manager.cleanup_expired_cache()
        if removed:
            print(f"Cleaned up {removed} expired cache entries")

        await self._wait_for_mcp_servers()

        await self.action_scheduler.run_missed_at_startup()

        tasks = []

        action_tasks = await self.action_scheduler.start()
        tasks.extend(action_tasks)

        # Watch event-schedules.json for changes
        self.action_scheduler_file_watchdog = FileWatchdog(
            self.action_scheduler.schedule_file, self.action_scheduler.reload, debounce_seconds=0.5
        )
        await self.action_scheduler_file_watchdog.start()

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

        await asyncio.gather(*tasks)

    async def _handle_wake_event(self, wall_jump_seconds: float, now=None) -> None:
        """Handle system wake event after suspension."""
        print("\n" + "=" * 60)
        print(f"⚠️  Suspension detected: {abs(wall_jump_seconds):.0f} seconds")
        print("Checking for missed scheduled runs...")
        print("=" * 60)

        self.action_scheduler.notify_resume()
        await self.action_scheduler.run_missed_at_startup(now=now)

        print("✓ Suspension recovery complete")
        print("=" * 60 + "\n")

    async def stop(self):
        """Stop the monitoring loop"""
        self.running = False
        print("Stopping monitoring scheduler...")

        await self.action_scheduler.stop()
        if self.action_scheduler_file_watchdog:
            await self.action_scheduler_file_watchdog.stop()

        if self.config_watcher:
            await self.config_watcher.stop()
