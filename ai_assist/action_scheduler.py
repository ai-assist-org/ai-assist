"""Unified action scheduler — handles timer, event, and one-shot actions"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

from .action_engine import ActionEngine
from .action_loader import ActionLoader
from .action_model import ActionDefinition, TriggerMatcher
from .agent import AiAssistAgent
from .event_sources import EventContext, EventSourceManager
from .state import StateManager
from .tasks import TaskLoader

logger = logging.getLogger(__name__)


class ActionScheduler:
    """Unified scheduler for all action types"""

    def __init__(
        self,
        agent: AiAssistAgent,
        state_manager: StateManager,
        schedule_file: Path,
    ) -> None:
        self.agent = agent
        self.state_manager = state_manager
        self.schedule_file = schedule_file
        self.loader = ActionLoader(schedule_file)
        self.engine = ActionEngine(agent, state_manager)
        self.matcher = TriggerMatcher()

        self.actions: list[ActionDefinition] = []
        self.timer_handles: list[asyncio.Task[None]] = []
        self.event_source_manager: EventSourceManager | None = None
        self.running = False
        self._debounce_tasks: dict[str, asyncio.Task[None]] = {}
        self._debounce_events: dict[str, list[EventContext]] = {}

    def load_actions(self) -> list[ActionDefinition]:
        self.loader.ensure_defaults()
        try:
            self.actions = self.loader.load_actions()
        except Exception:
            logger.exception("Error loading actions from %s", self.schedule_file)
            self.actions = []
        return self.actions

    async def start(self) -> list[asyncio.Task[None]]:
        self.running = True
        self.load_actions()

        tasks: list[asyncio.Task[None]] = []

        for action in self.actions:
            if not action.enabled:
                print(f"Skipping disabled action: {action.name}")
                continue

            if action.trigger_type == "once" and action.status in ("completed", "failed"):
                continue

            if action.is_time_based:
                handle = asyncio.create_task(self._schedule_timer_action(action))
                self.timer_handles.append(handle)
                tasks.append(handle)
                print(f"Scheduled action: {action.name} (trigger: {action.trigger_type})")
            elif action.is_event_based:
                print(f"Loaded event action: {action.name} (trigger: {action.trigger_type})")

        await self._start_event_sources()

        return tasks

    async def reload(self) -> None:
        print("\nReloading actions...")

        await self._stop_event_sources()

        for handle in self.timer_handles:
            handle.cancel()
        if self.timer_handles:
            await asyncio.gather(*self.timer_handles, return_exceptions=True)
        self.timer_handles.clear()

        self.load_actions()

        for action in self.actions:
            if not action.enabled:
                continue
            if action.is_time_based:
                handle = asyncio.create_task(self._schedule_timer_action(action))
                self.timer_handles.append(handle)

        await self._start_event_sources()
        print(f"Reloaded {len(self.actions)} action(s)")

    async def stop(self) -> None:
        self.running = False
        await self._stop_event_sources()
        for handle in self.timer_handles:
            handle.cancel()
        if self.timer_handles:
            await asyncio.gather(*self.timer_handles, return_exceptions=True)
        self.timer_handles.clear()

    async def _start_event_sources(self) -> None:
        event_configs = self.loader.load_event_source_configs()
        event_actions = [a for a in self.actions if a.is_event_based and a.enabled]

        if not event_actions:
            return
        if not event_configs:
            logger.warning(
                "%d event action(s) configured but no event_sources in %s",
                len(event_actions),
                self.schedule_file,
            )
            return

        self.event_source_manager = EventSourceManager()
        self.event_source_manager.register_available_sources(event_configs)

        for action in event_actions:
            source_type = action.trigger.get("type", "")
            source = self.event_source_manager.get_source(source_type)
            if source:
                source.subscribe(action.name, action.trigger)

        self.event_source_manager._event_handler = self._handle_event
        await self.event_source_manager.start()
        print(f"Started {len(self.event_source_manager._sources)} event source(s) for {len(event_actions)} action(s)")

    async def _stop_event_sources(self) -> None:
        if self.event_source_manager:
            await self.event_source_manager.stop()
            self.event_source_manager = None

    async def _handle_event(self, event: EventContext) -> None:
        debounce_seconds = 3.0

        for action in self.actions:
            if not action.enabled or not action.is_event_based:
                continue
            if self.matcher.matches(event, action.trigger):
                self._debounce_events.setdefault(action.name, []).append(event)

                if action.name in self._debounce_tasks:
                    self._debounce_tasks[action.name].cancel()

                self._debounce_tasks[action.name] = asyncio.create_task(
                    self._debounced_execute(action, debounce_seconds)
                )

    async def _debounced_execute(self, action: ActionDefinition, delay: float) -> None:
        await asyncio.sleep(delay)
        events = self._debounce_events.pop(action.name, [])
        self._debounce_tasks.pop(action.name, None)

        if not events:
            return

        combined = events[-1]
        if len(events) > 1:
            combined = EventContext(
                source_type=combined.source_type,
                event_type=combined.event_type,
                payload=f"{len(events)} events received:\n" + "\n".join(e.payload for e in events),
                metadata={**combined.metadata, "event_count": len(events)},
                timestamp=events[0].timestamp,
            )

        print(
            f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Event matched: {action.name} ({len(events)} signal(s))"
        )
        try:
            result = await self.engine.execute_action(action, event_context=combined)
            if result.success:
                print(f"{action.name}: completed")
            else:
                print(f"{action.name}: failed - {result.output[:200]}")
        except Exception:
            logger.exception("Error executing event action '%s'", action.name)

    async def _schedule_timer_action(self, action: ActionDefinition) -> None:
        trigger = action.trigger
        trigger_type = action.trigger_type

        while self.running:
            try:
                if trigger_type == "once":
                    if action.status in ("completed", "failed"):
                        break
                    target_time = datetime.fromisoformat(trigger["at"])
                    wait = (target_time - datetime.now()).total_seconds()
                    if wait > 0:
                        print(f"{action.name}: scheduled for {target_time.strftime('%Y-%m-%d %H:%M')}")
                        await asyncio.sleep(wait)
                    await self._execute_timer_action(action)
                    self._mark_once_completed(action)
                    break

                elif trigger_type == "schedule":
                    schedule_str = f"{trigger['at']} on {trigger['days']}"
                    schedule = TaskLoader.parse_time_schedule(schedule_str)
                    next_run = TaskLoader.calculate_next_run(schedule)
                    wait = (next_run - datetime.now()).total_seconds()
                    if wait > 0:
                        print(f"{action.name}: next run at {next_run.strftime('%Y-%m-%d %H:%M')}")
                        await asyncio.sleep(wait)

                elif trigger_type == "interval_range":
                    range_str = f"{trigger['every']} between {trigger['between']} and {trigger['and']}"
                    if "days" in trigger:
                        range_str += f" on {trigger['days']}"
                    schedule = TaskLoader.parse_interval_with_range(range_str)
                    next_run = TaskLoader.calculate_next_interval_run(schedule)
                    wait = (next_run - datetime.now()).total_seconds()
                    if wait > 0:
                        print(f"{action.name}: next run at {next_run.strftime('%Y-%m-%d %H:%M')}")
                        await asyncio.sleep(wait)

                elif trigger_type == "interval":
                    pass  # Execute immediately, then sleep after

                await self._execute_timer_action(action)

                if trigger_type == "interval":
                    interval_seconds = TaskLoader.parse_interval(trigger["every"])
                    await asyncio.sleep(interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in action '%s'", action.name)
                if trigger_type == "interval":
                    interval_seconds = TaskLoader.parse_interval(trigger["every"])
                    try:
                        await asyncio.sleep(interval_seconds)
                    except asyncio.CancelledError:
                        break

    async def _execute_timer_action(self, action: ActionDefinition) -> None:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running {action.name}...")
        result = await self.engine.execute_action(action)
        if result.success:
            print(f"{action.name}: completed")
            if result.output:
                print(f"\n{result.output}")
        else:
            print(f"{action.name}: failed - {result.output[:200]}")

    def _mark_once_completed(self, action: ActionDefinition) -> None:
        try:
            actions = self.loader.load_actions()
            for a in actions:
                if a.name == action.name and a.trigger_type == "once":
                    a.status = "completed"
                    a.executed_at = datetime.now()
                    break
            self.loader.save_actions(actions)
        except Exception:
            logger.exception("Failed to mark once-action '%s' as completed", action.name)

    async def run_missed_at_startup(self, now: datetime | None = None) -> None:
        if now is None:
            now = datetime.now()

        lookback = now - timedelta(hours=24)

        for action in self.actions:
            if not action.enabled:
                continue
            if action.trigger_type != "schedule":
                continue

            try:
                schedule_str = f"{action.trigger['at']} on {action.trigger['days']}"
                schedule = TaskLoader.parse_time_schedule(schedule_str)
            except (ValueError, KeyError):
                continue

            last_scheduled = TaskLoader.calculate_next_run(schedule, from_time=lookback)
            if last_scheduled > now:
                continue

            state_key = ActionEngine._state_key(action)
            last_run_state = self.state_manager.get_monitor_state(state_key)
            if last_run_state.last_check and last_run_state.last_check >= last_scheduled:
                if last_run_state.last_results.get("last_success", True):
                    continue

            print(f"Running missed action: {action.name} (was due at {last_scheduled.strftime('%Y-%m-%d %H:%M')})")
            try:
                await self.engine.execute_action(action)
            except Exception:
                logger.exception("Error running missed action '%s'", action.name)
