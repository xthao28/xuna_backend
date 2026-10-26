import uuid

import pytest

from xuna_backend.modules.transport.domain.entities import (
    MqttTopic,
    TransportChannel,
    TransportDirection,
)


def test_topic_formatting() -> None:
    dev_id = uuid.uuid4()
    t_status = MqttTopic(device_id=dev_id, channel=TransportChannel.STATUS)
    assert t_status.as_str() == f"xuna/v1/{dev_id}/status"

    t_events = MqttTopic(
        device_id=dev_id, channel=TransportChannel.EVENTS, direction=TransportDirection.UP
    )
    assert t_events.as_str() == f"xuna/v1/{dev_id}/up/events"

    t_suggest = MqttTopic(
        device_id=dev_id, channel=TransportChannel.SUGGEST, direction=TransportDirection.DOWN
    )
    assert t_suggest.as_str() == f"xuna/v1/{dev_id}/down/suggest"


def test_topic_parsing() -> None:
    dev_id = uuid.uuid4()
    parsed_status = MqttTopic.parse(f"xuna/v1/{dev_id}/status")
    assert parsed_status.device_id == dev_id
    assert parsed_status.channel == TransportChannel.STATUS
    assert parsed_status.direction is None

    parsed_up_events = MqttTopic.parse(f"xuna/v1/{dev_id}/up/events")
    assert parsed_up_events.device_id == dev_id
    assert parsed_up_events.channel == TransportChannel.EVENTS
    assert parsed_up_events.direction == TransportDirection.UP

    parsed_down_cmd = MqttTopic.parse(f"xuna/v1/{dev_id}/down/cmd")
    assert parsed_down_cmd.device_id == dev_id
    assert parsed_down_cmd.channel == TransportChannel.CMD
    assert parsed_down_cmd.direction == TransportDirection.DOWN


def test_topic_validation_failures() -> None:
    dev_id = uuid.uuid4()
    with pytest.raises(ValueError, match="STATUS topic must not have a direction"):
        MqttTopic(
            device_id=dev_id, channel=TransportChannel.STATUS, direction=TransportDirection.UP
        )

    with pytest.raises(ValueError, match="requires direction"):
        MqttTopic(device_id=dev_id, channel=TransportChannel.EVENTS, direction=None)

    with pytest.raises(ValueError, match="invalid MQTT topic"):
        MqttTopic.parse("invalid/topic/here")

    with pytest.raises(ValueError, match="invalid MQTT topic"):
        MqttTopic.parse(f"xuna/v1/{dev_id}/unknown/events")
