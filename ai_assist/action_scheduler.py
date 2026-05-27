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
        self._executing: set[str] = set()
        self._self_write_time: float = 0.0
        self._resume_event = asyncio.Event()

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
                handle = asyncio.create_task(self._schedule_timer_action(action), name=action.name)
                self.timer_handles.append(handle)
                tasks.append(handle)
                print(f"Scheduled action: {action.name} (trigger: {action.trigger_type})")
            elif action.is_event_based:
                print(f"Loaded event action: {action.name} (trigger: {action.trigger_type})")

        await self._start_event_sources()

        return tasks

    async def reload(self) -> None:
        import time

        if time.monotonic() - self._self_write_time < 2.0:
            return

        print("\nReloading actions...")

        await self._stop_event_sources()

        surviving: list[asyncio.Task[None]] = []
        to_cancel: list[asyncio.Task[None]] = []
        for handle in self.timer_handles:
            if handle.get_name() in self._executing:
                surviving.append(handle)
            else:
                to_cancel.append(handle)
                handle.cancel()
        if to_cancel:
            await asyncio.gather(*to_cancel, return_exceptions=True)

        self.timer_handles = list(surviving)

        self.load_actions()

        scheduled_names = {h.get_name() for h in self.timer_handles}
        for action in self.actions:
            if not action.enabled:
                continue
            if action.is_time_based and action.name not in scheduled_names:
                handle = asyncio.create_task(self._schedule_timer_action(action), name=action.name)
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

    def notify_resume(self) -> None:
        """Signal timer tasks to re-check wall-clock time after system resume."""
        self._resume_event.set()

    async def _sleep_until(self, target: datetime) -> None:
        """Sleep until target wall-clock time, resilient to system suspend."""
        while self.running:
            remaining = (target - datetime.now()).total_seconds()
            if remaining <= 0:
                return
            self._resume_event.clear()
            sleep_task = asyncio.ensure_future(asyncio.sleep(remaining))
            wake_task = asyncio.ensure_future(self._resume_event.wait())
            tasks = {sleep_task, wake_task}
            try:
                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
            except asyncio.CancelledError:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

    async def _schedule_timer_action(self, action: ActionDefinition) -> None:
        trigger = action.trigger
        trigger_type = action.trigger_type

        while self.running:
            try:
                if trigger_type == "once":
                    if action.status in ("completed", "failed"):
                        break
                    target_time = datetime.fromisoformat(trigger["at"])
                    if (target_time - datetime.now()).total_seconds() <= 0:
                        break
                    print(f"{action.name}: scheduled for {target_time.strftime('%Y-%m-%d %H:%M')}")
                    await self._sleep_until(target_time)
                    await self._execute_timer_action(action)
                    self._mark_once_completed(action)
                    break

                elif trigger_type == "schedule":
                    schedule_str = f"{trigger['at']} on {trigger['days']}"
                    schedule = TaskLoader.parse_time_schedule(schedule_str)
                    next_run = TaskLoader.calculate_next_run(schedule)
                    if (next_run - datetime.now()).total_seconds() > 0:
                        print(f"{action.name}: next run at {next_run.strftime('%Y-%m-%d %H:%M')}")
                        await self._sleep_until(next_run)

                elif trigger_type == "interval_range":
                    range_str = f"{trigger['every']} between {trigger['between']} and {trigger['and']}"
                    if "days" in trigger:
                        range_str += f" on {trigger['days']}"
                    schedule = TaskLoader.parse_interval_with_range(range_str)
                    next_run = TaskLoader.calculate_next_interval_run(schedule)
                    if (next_run - datetime.now()).total_seconds() > 0:
                        print(f"{action.name}: next run at {next_run.strftime('%Y-%m-%d %H:%M')}")
                        await self._sleep_until(next_run)

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
        self._executing.add(action.name)
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running {action.name}...")
            result = await self.engine.execute_action(action)
            if result.success:
                print(f"{action.name}: completed")
                if result.output:
                    print(f"\n{result.output}")
            else:
                print(f"{action.name}: failed - {result.output[:200]}")
        finally:
            self._executing.discard(action.name)

    def _mark_once_completed(self, action: ActionDefinition) -> None:
        import time

        try:
            actions = self.loader.load_actions()
            for a in actions:
                if a.name == action.name and a.trigger_type == "once":
                    a.status = "completed"
                    a.executed_at = datetime.now()
                    break
            self._self_write_time = time.monotonic()
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

            if action.trigger_type == "once":
                if action.status in ("completed", "failed"):
                    continue
                try:
                    target_time = datetime.fromisoformat(action.trigger["at"])
                except ValueError, KeyError:
                    continue
                if target_time > now or target_time < lookback:
                    continue
                print(
                    f"Running missed once-action: {action.name} (was due at {target_time.strftime('%Y-%m-%d %H:%M')})"
                )
                try:
                    await self.engine.execute_action(action)
                    self._mark_once_completed(action)
                except Exception:
                    logger.exception("Error running missed once-action '%s'", action.name)
                continue

            if action.trigger_type != "schedule":
                continue

            try:
                schedule_str = f"{action.trigger['at']} on {action.trigger['days']}"
                schedule = TaskLoader.parse_time_schedule(schedule_str)
            except ValueError, KeyError:
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
