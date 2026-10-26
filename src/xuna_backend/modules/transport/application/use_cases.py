"""Application use cases for Direct Device Transport (CLD-40, CLD-41, E14, H1-H8)."""

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.audit.domain.event import AuditAction, AuditEvent
from xuna_backend.modules.identity.application.ports import IdentityRepository
from xuna_backend.modules.suggestions.domain.entities import CloudSuggestion
from xuna_backend.modules.transport.application.ports import (
    MqttPublisher,
    TransportStatusRepository,
)
from xuna_backend.modules.transport.domain.entities import (
    DeviceConnectionStatus,
    DeviceStatusRecord,
    DownlinkCommand,
    MqttTopic,
    TransportChannel,
    TransportDirection,
)


class TransportError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


async def handle_uplink_message(
    topic_str: str,
    payload_data: str | bytes | dict[str, Any],
    status_repo: TransportStatusRepository,
    identity_repo: IdentityRepository,
    audit: AuditLog,
    now: datetime,
    ip_address: str | None = None,
    client_id: str | None = None,
) -> dict[str, Any]:
    """Processes an incoming uplink MQTT message from a device (CLD-40, CLD-41)."""
    try:
        topic = MqttTopic.parse(topic_str)
    except ValueError as err:
        raise TransportError("invalid_topic") from err

    pet = await identity_repo.get_pet_by_device(topic.device_id)
    if pet is None:
        raise TransportError("device_not_found")

    if isinstance(payload_data, bytes):
        payload_data = payload_data.decode("utf-8")
    if isinstance(payload_data, str):
        try:
            payload: dict[str, Any] = json.loads(payload_data)
        except json.JSONDecodeError as err:
            raise TransportError("invalid_payload") from err
    else:
        payload = payload_data

    match topic.channel:
        case TransportChannel.STATUS:
            raw_status = payload.get("status", "offline")
            try:
                status = DeviceConnectionStatus(str(raw_status).lower())
            except ValueError as err:
                raise TransportError("invalid_status") from err

            record = DeviceStatusRecord(
                device_id=topic.device_id,
                pet_id=pet.pet_id.value,
                status=status,
                last_seen_at=now,
                ip_address=ip_address,
                client_id=client_id,
            )
            await status_repo.save_status(record)
            await audit.record(
                AuditEvent(
                    owner_id=pet.owner_id,
                    action=AuditAction.DEVICE_STATUS_CHANGED,
                    occurred_at=now,
                    subject=pet.pet_id.value,
                    metadata={"status": status.value},
                )
            )
            return {"status": "ok", "channel": "status", "deviceStatus": status.value}

        case TransportChannel.EVENTS:
            if topic.direction != TransportDirection.UP:
                raise TransportError("invalid_direction")

            event_type = str(payload.get("eventType", "interaction"))[:32]
            await audit.record(
                AuditEvent(
                    owner_id=pet.owner_id,
                    action=AuditAction.DEVICE_EVENT_INGESTED,
                    occurred_at=now,
                    subject=pet.pet_id.value,
                    metadata={"eventType": event_type},
                )
            )
            return {"status": "ok", "channel": "events", "eventType": event_type}

        case TransportChannel.STATE:
            if topic.direction != TransportDirection.UP:
                raise TransportError("invalid_direction")

            gen = int(payload.get("snapshotGeneration", 0))
            if gen > pet.last_generation:
                await identity_repo.update_generation(pet.pet_id, gen)

            await audit.record(
                AuditEvent(
                    owner_id=pet.owner_id,
                    action=AuditAction.PROJECTION_INGESTED,
                    occurred_at=now,
                    subject=pet.pet_id.value,
                    metadata={"generation": gen},
                )
            )
            return {"status": "ok", "channel": "state", "generation": gen}

        case _:
            raise TransportError("unsupported_channel")


async def send_downlink_suggestion(
    publisher: MqttPublisher,
    device_id: uuid.UUID,
    suggestion: CloudSuggestion,
) -> str:
    """Publishes a bounded suggestion to device topic xuna/v1/{device_id}/down/suggest."""
    topic = MqttTopic(
        device_id=device_id,
        channel=TransportChannel.SUGGEST,
        direction=TransportDirection.DOWN,
    ).as_str()

    payload = json.dumps(
        {
            "suggestionId": str(suggestion.suggestion_id),
            "kind": suggestion.kind.value,
            "target": suggestion.target,
            "delta": suggestion.delta,
            "bounds": suggestion.bounds,
            "provenance": suggestion.provenance.value,
            "expiresAt": suggestion.expires_at.astimezone(UTC).isoformat(),
        }
    )
    await publisher.publish(topic, payload, qos=1)
    return topic


async def send_downlink_command(
    publisher: MqttPublisher,
    identity_repo: IdentityRepository,
    audit: AuditLog,
    owner_id: uuid.UUID,
    device_id: uuid.UUID,
    action: str,
    args: dict[str, Any],
    now: datetime,
) -> DownlinkCommand:
    """Publishes an operational command to device topic xuna/v1/{device_id}/down/cmd."""
    pet = await identity_repo.get_pet_by_device(device_id)
    if pet is None or pet.owner_id != owner_id:
        raise TransportError("device_not_found")

    cmd = DownlinkCommand(
        command_id=uuid.uuid4(),
        device_id=device_id,
        action=action,
        args=args,
        issued_at=now,
    )

    topic = MqttTopic(
        device_id=device_id,
        channel=TransportChannel.CMD,
        direction=TransportDirection.DOWN,
    ).as_str()

    payload = json.dumps(
        {
            "commandId": str(cmd.command_id),
            "action": cmd.action,
            "args": cmd.args,
            "issuedAt": cmd.issued_at.astimezone(UTC).isoformat(),
        }
    )
    await publisher.publish(topic, payload, qos=1)

    await audit.record(
        AuditEvent(
            owner_id=owner_id,
            action=AuditAction.DEVICE_COMMAND_SENT,
            occurred_at=now,
            subject=pet.pet_id.value,
            metadata={"action": action[:32]},
        )
    )
    return cmd


async def get_device_status(
    status_repo: TransportStatusRepository,
    identity_repo: IdentityRepository,
    owner_id: uuid.UUID,
    device_id: uuid.UUID,
) -> DeviceStatusRecord:
    """Retrieves current transport connection status for a device."""
    pet = await identity_repo.get_pet_by_device(device_id)
    if pet is None or pet.owner_id != owner_id:
        raise TransportError("device_not_found")

    record = await status_repo.get_status(device_id)
    if record is None:
        return DeviceStatusRecord(
            device_id=device_id,
            pet_id=pet.pet_id.value,
            status=DeviceConnectionStatus.OFFLINE,
            last_seen_at=pet.bound_at,
        )
    return record
