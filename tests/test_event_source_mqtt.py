"""Tests for MQTT event source"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

aiomqtt = pytest.importorskip("aiomqtt")

from ai_assist.event_source_mqtt import MqttEventSource  # noqa: E402


@pytest.fixture
def mqtt_source():
    return MqttEventSource({"broker": "localhost", "port": 1883, "client_id": "test"})


class TestMqttEventSource:
    def test_name(self, mqtt_source):
        assert mqtt_source.name == "mqtt"

    def test_subscribe(self, mqtt_source):
        mqtt_source.subscribe("task1", {"type": "mqtt", "topic": "alerts/#"})
        mqtt_source.subscribe("task2", {"type": "mqtt", "topic": "status/+"})
        assert "alerts/#" in mqtt_source._subscriptions
        assert mqtt_source._subscriptions["alerts/#"] == ["task1"]
        assert mqtt_source._subscriptions["status/+"] == ["task2"]

    def test_subscribe_multiple_tasks_same_topic(self, mqtt_source):
        mqtt_source.subscribe("task1", {"type": "mqtt", "topic": "alerts/#"})
        mqtt_source.subscribe("task2", {"type": "mqtt", "topic": "alerts/#"})
        assert mqtt_source._subscriptions["alerts/#"] == ["task1", "task2"]

    def test_unsubscribe_all(self, mqtt_source):
        mqtt_source.subscribe("task1", {"type": "mqtt", "topic": "a"})
        mqtt_source.unsubscribe_all()
        assert len(mqtt_source._subscriptions) == 0

    def test_config_parsing(self):
        source = MqttEventSource(
            {
                "broker": "mqtt.example.com",
                "port": 8883,
                "client_id": "my-client",
                "username": "user",
                "password": "pass",
            }
        )
        assert source.broker == "mqtt.example.com"
        assert source.port == 8883
        assert source.client_id == "my-client"
        assert source.username == "user"
        assert source.password == "pass"

    def test_config_defaults(self):
        source = MqttEventSource({})
        assert source.broker == "localhost"
        assert source.port == 1883
        assert source.client_id == "ai-assist"
        assert source.username is None

    @pytest.mark.asyncio
    async def test_stop_cancels_cleanly(self, mqtt_source):
        mqtt_source._running = True
        mqtt_source._task = asyncio.create_task(asyncio.sleep(999))

        await mqtt_source.stop()
        assert not mqtt_source._running

    @pytest.mark.asyncio
    async def test_listen_dispatches_matching_messages(self, mqtt_source):
        """Test that _listen dispatches events for matching topic patterns"""
        dispatched = []

        async def mock_dispatch(task_name, event):
            dispatched.append((task_name, event))
            # Stop after first dispatch to exit the reconnect loop
            mqtt_source._running = False

        mqtt_source.subscribe("alert-handler", {"type": "mqtt", "topic": "alerts/#"})
        mqtt_source._dispatch = mock_dispatch

        # Create a mock message
        mock_message = MagicMock()
        mock_message.payload = b"CPU at 95%"
        mock_message.qos = 0
        mock_message.retain = False
        mock_topic = MagicMock()
        mock_topic.matches.return_value = True
        mock_topic.__str__ = lambda self: "alerts/cpu"
        mock_message.topic = mock_topic

        # Mock aiomqtt.Client as async context manager with messages iterator
        async def mock_messages():
            yield mock_message

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.subscribe = AsyncMock()
        mock_client.messages = mock_messages()

        with patch("aiomqtt.Client", return_value=mock_client):
            mqtt_source._running = True
            task = asyncio.create_task(mqtt_source._listen())

            # Wait for the dispatch to happen and stop the loop
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except TimeoutError, asyncio.CancelledError:
                pass

        assert len(dispatched) == 1
        assert dispatched[0][0] == "alert-handler"
        assert dispatched[0][1].payload == "CPU at 95%"
        assert dispatched[0][1].source_type == "mqtt"
        assert dispatched[0][1].metadata["topic"] == "alerts/cpu"
