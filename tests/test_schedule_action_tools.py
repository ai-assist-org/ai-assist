"""Tests for schedule action tools and time parsing"""

from datetime import datetime, timedelta

import pytest

from ai_assist.action_loader import ActionLoader
from ai_assist.action_tools import ActionTools
from ai_assist.schedule_action_tools import parse_time_spec


class TestParseTimeSpec:
    """Test time specification parsing"""

    def test_parse_at_5pm(self):
        """Parse 'at 5pm' format"""
        result = parse_time_spec("at 5pm")
        assert result is not None
        assert result.hour == 17
        assert result.minute == 0

    def test_parse_5pm_without_at(self):
        """Parse '5pm' without 'at' prefix"""
        result = parse_time_spec("5pm")
        assert result is not None
        assert result.hour == 17
        assert result.minute == 0

    def test_parse_24hour_format(self):
        """Parse '17:00' 24-hour format"""
        result = parse_time_spec("17:00")
        assert result is not None
        assert result.hour == 17
        assert result.minute == 0

    def test_parse_at_24hour_format(self):
        """Parse 'at 17:00' with 'at' prefix"""
        result = parse_time_spec("at 17:00")
        assert result is not None
        assert result.hour == 17
        assert result.minute == 0

    def test_parse_today_at_5pm(self):
        """Parse 'today at 5pm'"""
        result = parse_time_spec("today at 5pm")
        assert result is not None
        assert result.hour == 17
        assert result.minute == 0

    def test_parse_tomorrow_at_9am(self):
        """Parse 'tomorrow at 9am'"""
        result = parse_time_spec("tomorrow at 9am")
        assert result is not None
        assert result.hour == 9
        assert result.minute == 0
        # Should be tomorrow
        assert result.date() == (datetime.now() + timedelta(days=1)).date()

    def test_parse_tomorrow_at_2pm(self):
        """Parse 'tomorrow at 2pm'"""
        result = parse_time_spec("tomorrow at 2pm")
        assert result is not None
        assert result.hour == 14
        assert result.minute == 0
        # Should be tomorrow
        assert result.date() == (datetime.now() + timedelta(days=1)).date()

    def test_parse_tomorrow_24hour(self):
        """Parse 'tomorrow at 14:30' 24-hour format"""
        result = parse_time_spec("tomorrow at 14:30")
        assert result is not None
        assert result.hour == 14
        assert result.minute == 30
        # Should be tomorrow
        assert result.date() == (datetime.now() + timedelta(days=1)).date()

    def test_parse_in_2_hours(self):
        """Parse 'in 2 hours' relative time"""
        now = datetime.now()
        result = parse_time_spec("in 2 hours")
        assert result is not None
        diff = (result - now).total_seconds()
        # Should be approximately 2 hours (within 1 second)
        assert 7199 <= diff <= 7201

    def test_parse_in_30_minutes(self):
        """Parse 'in 30 minutes' relative time"""
        now = datetime.now()
        result = parse_time_spec("in 30 minutes")
        assert result is not None
        diff = (result - now).total_seconds()
        # Should be approximately 30 minutes (within 1 second)
        assert 1799 <= diff <= 1801

    def test_parse_in_1_day(self):
        """Parse 'in 1 day' relative time"""
        now = datetime.now()
        result = parse_time_spec("in 1 day")
        assert result is not None
        diff = (result - now).total_seconds()
        # Should be approximately 1 day (within 1 second)
        assert 86399 <= diff <= 86401

    def test_parse_next_monday(self):
        """Parse 'next monday 10:00'"""
        result = parse_time_spec("next monday 10:00")
        assert result is not None
        assert result.hour == 10
        assert result.minute == 0
        assert result.weekday() == 0  # Monday

    def test_past_time_rolls_to_tomorrow(self):
        """Time that has already passed today should be scheduled for tomorrow"""
        now = datetime.now()
        # Schedule something for 1 hour ago
        past_hour = (now - timedelta(hours=1)).hour
        result = parse_time_spec(f"{past_hour}:00")
        assert result is not None
        assert result.hour == past_hour
        # Should be tomorrow since the time has passed
        assert result.date() >= now.date()
        if result.date() == now.date():
            # If same date, time must be in the future
            assert result > now

    def test_12am_midnight(self):
        """Parse '12am' as midnight"""
        result = parse_time_spec("12am")
        assert result is not None
        assert result.hour == 0
        assert result.minute == 0

    def test_12pm_noon(self):
        """Parse '12pm' as noon"""
        result = parse_time_spec("12pm")
        assert result is not None
        assert result.hour == 12
        assert result.minute == 0

    def test_invalid_spec_returns_none(self):
        """Invalid time specification returns None"""
        assert parse_time_spec("garbage") is None
        assert parse_time_spec("next year") is None
        assert parse_time_spec("") is None


class TestScheduleActionErrorMessage:
    """Test error messages from schedule_action"""

    @pytest.mark.asyncio
    async def test_invalid_format_returns_helpful_error(self, tmp_path):
        """Invalid time format returns error with supported formats"""
        from ai_assist.schedule_action_tools import ScheduleActionTools

        tools = ScheduleActionTools(ActionTools(schedules_file=tmp_path / "event-schedules.json"))

        result = await tools.schedule_action(
            prompt="Test reminder",
            time_spec="next year sometime",  # Invalid format
            description="Test",
        )

        # Should return error message
        assert "Error: Could not parse time specification" in result
        assert "next year sometime" in result

        # Should include all supported format examples
        assert "in 2 hours" in result
        assert "at 5pm" in result
        assert "tomorrow at 9am" in result
        assert "next monday 10:00" in result
        assert "Supported formats:" in result

    @pytest.mark.asyncio
    async def test_empty_time_spec_returns_error(self, tmp_path):
        """Empty time specification returns error"""
        from ai_assist.schedule_action_tools import ScheduleActionTools

        tools = ScheduleActionTools(ActionTools(schedules_file=tmp_path / "event-schedules.json"))

        result = await tools.schedule_action(
            prompt="Test reminder",
            time_spec="",  # Empty
            description="Test",
        )

        assert "Error: Could not parse time specification" in result
        assert "Supported formats:" in result


class TestScheduleActionCreatesActions:
    """internal__schedule_action creates one-time actions in event-schedules.json"""

    @pytest.fixture
    def schedules_file(self, tmp_path):
        return tmp_path / "event-schedules.json"

    @pytest.fixture
    def tools(self, schedules_file):
        from ai_assist.schedule_action_tools import ScheduleActionTools

        return ScheduleActionTools(ActionTools(schedules_file=schedules_file))

    @pytest.mark.asyncio
    async def test_query_action_written_to_event_schedules(self, tools, schedules_file):
        result = await tools.schedule_action(
            prompt="Check unread emails", time_spec="in 2 hours", description="Email check", execute_query=True
        )

        assert result.startswith("✓ Scheduled")
        [action] = ActionLoader(schedules_file).load_actions()
        assert action.trigger_type == "once"
        assert action.prompt == "Check unread emails"
        assert action.reminder is False
        assert action.notify is True
        assert action.notification_channels == ["desktop", "file"]
        assert action.created_by == "agent"
        assert action.status == "pending"
        at = datetime.fromisoformat(action.trigger["at"])
        assert abs((at - (datetime.now() + timedelta(hours=2))).total_seconds()) < 60
        assert action.name.startswith("Email check")

    @pytest.mark.asyncio
    async def test_simple_reminder_is_a_reminder_action(self, tools, schedules_file):
        await tools.schedule_action(prompt="Time to stretch", time_spec="in 30 minutes")

        [action] = ActionLoader(schedules_file).load_actions()
        assert action.reminder is True
        assert action.prompt == "Time to stretch"

    @pytest.mark.asyncio
    async def test_create_report_adds_report_instruction(self, tools, schedules_file):
        await tools.schedule_action(
            prompt="Summarize open PRs", time_spec="in 1 hour", execute_query=True, create_report=True
        )

        [action] = ActionLoader(schedules_file).load_actions()
        assert action.prompt.startswith("Summarize open PRs")
        assert "report" in action.prompt.lower()

    @pytest.mark.asyncio
    async def test_notify_false_is_silent(self, tools, schedules_file):
        await tools.schedule_action(prompt="Refresh cache", time_spec="in 1 hour", execute_query=True, notify=False)

        [action] = ActionLoader(schedules_file).load_actions()
        assert action.notify is False

    @pytest.mark.asyncio
    async def test_same_description_at_different_times_gets_unique_names(self, tools, schedules_file):
        await tools.schedule_action(prompt="Weekly report", time_spec="in 1 hour", description="Weekly report for X")
        await tools.schedule_action(prompt="Weekly report", time_spec="in 2 days", description="Weekly report for X")

        names = [a.name for a in ActionLoader(schedules_file).load_actions()]
        assert len(names) == 2
        assert len(set(names)) == 2
        assert all(n.startswith("Weekly report for X") for n in names)
