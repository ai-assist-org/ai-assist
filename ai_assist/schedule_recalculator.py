"""Schedule recalculation and missed run detection.

This module handles detecting and executing missed scheduled runs
after system suspension or clock changes.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from .tasks import TaskLoader

logger = logging.getLogger(__name__)


class ScheduleRecalculator:
    """Detects and handles missed scheduled runs after suspension.

    For each time-based schedule, computes the next run that would have
    occurred after suspension started. If that time falls before wake time,
    the run was missed and is executed immediately.

    Interval-based schedules (e.g. "every 30m") are skipped — they continue
    from the current time naturally.
    """

    async def handle_wake_event(
        self,
        wall_jump_seconds: float,
        monitors: list[Any],
        now: datetime | None = None,
    ) -> None:
        """Handle system wake event and execute missed runs.

        Args:
            wall_jump_seconds: How many seconds the wall clock jumped
                (positive for suspension, negative for clock adjustment)
            monitors: List of objects with .schedule (str) and .execute() method
            now: Current time (defaults to datetime.now(), injectable for testing)
        """
        if now is None:
            now = datetime.now()

        before_suspend = now - timedelta(seconds=abs(wall_jump_seconds))

        for monitor in monitors:
            try:
                schedule = TaskLoader.parse_time_schedule(monitor.schedule)
            except ValueError:
                continue  # Not a time-based schedule — skip

            next_run = TaskLoader.calculate_next_run(schedule, from_time=before_suspend)
            if next_run <= now:
                try:
                    await monitor.execute()
                except Exception:
                    logger.exception("Error executing missed task")
