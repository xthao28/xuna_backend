"""Direct Device Transport API routes (CLD-40, CLD-41, E14)."""

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from http import HTTPStatus
from typing import cast

from fastapi import APIRouter, Request

from xuna_backend.core.db import commit_or_conflict
from xuna_backend.core.errors import AppError
from xuna_backend.modules.accounts.api.dependencies import CurrentOwner, Session, state
from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.identity.application.ports import IdentityRepository
from xuna_backend.modules.transport.api.device_socket import router as device_router
from xuna_backend.modules.transport.application.ports import (
    MqttPublisher,
    TransportStatusRepository,
)
from xuna_backend.modules.transport.application.schemas import (
    DeviceTransportStatusResponse,
    MqttUplinkWebhookRequest,
    MqttUplinkWebhookResponse,
    SendCommandRequest,
    SendCommandResponse,
)
from xuna_backend.modules.transport.application.use_cases import (
    TransportError,
    get_device_status,
    handle_uplink_message,
    send_downlink_command,
)
from xuna_backend.modules.transport.domain.entities import (
    MqttTopic,
    TransportChannel,
    TransportDirection,
)

router = APIRouter(prefix="/v1", tags=["transport"])
router.include_router(device_router)

_STATUS: dict[str, HTTPStatus] = {
    "device_not_found": HTTPStatus.NOT_FOUND,
    "not_found": HTTPStatus.NOT_FOUND,
    "invalid_topic": HTTPStatus.BAD_REQUEST,
    "invalid_payload": HTTPStatus.BAD_REQUEST,
    "invalid_direction": HTTPStatus.BAD_REQUEST,
    "invalid_status": HTTPStatus.BAD_REQUEST,
    "unsupported_channel": HTTPStatus.BAD_REQUEST,
}


def _status_repo(request: Request, session: object) -> TransportStatusRepository:
    factory: Callable[[object], TransportStatusRepository] = state(
        request, "transport_status_repository"
    )
    return factory(session)


def _identity_repo(request: Request, session: object) -> IdentityRepository:
    factory: Callable[[object], IdentityRepository] = state(request, "identity_repository")
    return factory(session)


def _mqtt_publisher(request: Request) -> MqttPublisher:
    return cast(MqttPublisher, state(request, "mqtt_publisher"))


def _problem(error: TransportError) -> AppError:
    return AppError(error.code, _STATUS.get(error.code, HTTPStatus.BAD_REQUEST))


@router.post("/transport/mqtt/uplink")
async def post_mqtt_uplink(
    body: MqttUplinkWebhookRequest,
    session: Session,
    request: Request,
) -> MqttUplinkWebhookResponse:
    """Webhook bridge endpoint for MQTT broker uplink messages (CLD-40, CLD-41)."""
    now = datetime.now(UTC)
    status_repo = _status_repo(request, session)
    id_repo = _identity_repo(request, session)
    audit: AuditLog = state(request, "audit_log")(session)

    try:
        topic = MqttTopic.parse(body.topic)
        await handle_uplink_message(
            topic_str=body.topic,
            payload_data=body.payload,
            status_repo=status_repo,
            identity_repo=id_repo,
            audit=audit,
            now=now,
            ip_address=body.ipAddress,
            client_id=body.clientId,
        )
        await commit_or_conflict(session)
        return MqttUplinkWebhookResponse(
            status="ok",
            deviceId=str(topic.device_id),
            channel=topic.channel.value,
            handled=True,
        )
    except TransportError as error:
        raise _problem(error) from error


@router.get("/devices/{device_id}/transport/status")
async def get_device_transport_status(
    device_id: str,
    owner: CurrentOwner,
    session: Session,
    request: Request,
) -> DeviceTransportStatusResponse:
    """Retrieves current online/offline status for a device."""
    try:
        parsed_device_id = uuid.UUID(device_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error

    status_repo = _status_repo(request, session)
    id_repo = _identity_repo(request, session)

    try:
        record = await get_device_status(
            status_repo=status_repo,
            identity_repo=id_repo,
            owner_id=owner.owner_id,
            device_id=parsed_device_id,
        )
        return DeviceTransportStatusResponse(
            deviceId=str(record.device_id),
            petId=record.pet_id,
            status=record.status.value,
            lastSeenAt=record.last_seen_at.astimezone(UTC).isoformat(),
            ipAddress=record.ip_address,
            clientId=record.client_id,
            firmware=record.firmware,
            connectedAt=(
                record.connected_at.astimezone(UTC).isoformat() if record.connected_at else None
            ),
        )
    except TransportError as error:
        raise _problem(error) from error


@router.post("/devices/{device_id}/transport/command")
async def post_device_command(
    device_id: str,
    body: SendCommandRequest,
    owner: CurrentOwner,
    session: Session,
    request: Request,
) -> SendCommandResponse:
    """Sends an operational command down to a device over MQTT."""
    now = datetime.now(UTC)
    try:
        parsed_device_id = uuid.UUID(device_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error

    id_repo = _identity_repo(request, session)
    publisher = _mqtt_publisher(request)
    audit: AuditLog = state(request, "audit_log")(session)

    try:
        cmd = await send_downlink_command(
            publisher=publisher,
            identity_repo=id_repo,
            audit=audit,
            owner_id=owner.owner_id,
            device_id=parsed_device_id,
            action=body.action,
            args=body.args,
            now=now,
        )
        await commit_or_conflict(session)
        cmd_topic = MqttTopic(
            device_id=parsed_device_id,
            channel=TransportChannel.CMD,
            direction=TransportDirection.DOWN,
        ).as_str()
        return SendCommandResponse(
            commandId=str(cmd.command_id),
            deviceId=str(cmd.device_id),
            topic=cmd_topic,
            action=cmd.action,
            status="published",
            issuedAt=cmd.issued_at.astimezone(UTC).isoformat(),
        )
    except TransportError as error:
        raise _problem(error) from error
