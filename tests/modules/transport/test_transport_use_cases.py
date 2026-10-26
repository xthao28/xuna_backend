import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from xuna_backend.modules.audit.domain.event import AuditAction, AuditEvent
from xuna_backend.modules.identity.domain.entities import Challenge, Pet
from xuna_backend.modules.identity.domain.pet_id import PetId, PetOrigin
from xuna_backend.modules.suggestions.domain.entities import (
    CloudSuggestion,
    SuggestionKind,
    SuggestionProvenance,
)
from xuna_backend.modules.transport.application.use_cases import (
    TransportError,
    get_device_status,
    handle_uplink_message,
    send_downlink_command,
    send_downlink_suggestion,
)
from xuna_backend.modules.transport.domain.entities import (
    DeviceConnectionStatus,
    DeviceStatusRecord,
)
from xuna_backend.modules.transport.infrastructure.memory_broker import InMemoryMqttBroker


class _FakeIdentityRepo:
    def __init__(self, pet: Pet) -> None:
        self.pet = pet
        self.updated_generation: int | None = None

    async def add_challenge(self, challenge: Challenge) -> None:
        pass

    async def consume_challenge(self, challenge_id: uuid.UUID) -> Challenge | None:
        return None

    async def get_pet(self, pet_id: PetId) -> Pet | None:
        return self.pet if pet_id == self.pet.pet_id else None

    async def get_pet_by_device(self, device_id: uuid.UUID) -> Pet | None:
        return self.pet if device_id == self.pet.device_id else None

    async def list_pets(self, owner_id: uuid.UUID) -> list[Pet]:
        return [self.pet] if owner_id == self.pet.owner_id else []

    async def create_pet(self, **kwargs: Any) -> Pet:
        return self.pet

    async def update_generation(self, pet_id: PetId, generation: int) -> None:
        self.updated_generation = generation

    async def rebind_device(self, pet_id: PetId, public_key: bytes, now: datetime) -> Pet:
        return self.pet

    async def rotate_brain(self, pet_id: PetId, now: datetime) -> Pet:
        return self.pet


class _FakeStatusRepo:
    def __init__(self) -> None:
        self.records: dict[uuid.UUID, DeviceStatusRecord] = {}

    async def save_status(self, record: DeviceStatusRecord) -> None:
        self.records[record.device_id] = record

    async def get_status(self, device_id: uuid.UUID) -> DeviceStatusRecord | None:
        return self.records.get(device_id)

    async def list_statuses_for_pet(self, pet_id: str) -> list[DeviceStatusRecord]:
        return [r for r in self.records.values() if r.pet_id == pet_id]


class _FakeAuditLog:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def record(self, event: AuditEvent) -> None:
        self.events.append(event)


@pytest.fixture
def test_pet() -> Pet:
    return Pet(
        pet_id=PetId(uuid.uuid4().hex),
        owner_id=uuid.uuid4(),
        origin=PetOrigin.HATCHED_ON_DEVICE,
        active_brain_id=uuid.uuid4(),
        device_id=uuid.uuid4(),
        device_public_key=b"public_key_32_bytes_placeholder!",
        last_generation=10,
        bound_at=datetime.now(UTC),
    )


@pytest.mark.anyio
async def test_handle_uplink_status(test_pet: Pet) -> None:
    status_repo = _FakeStatusRepo()
    id_repo = _FakeIdentityRepo(test_pet)
    audit = _FakeAuditLog()
    now = datetime.now(UTC)

    topic = f"xuna/v1/{test_pet.device_id}/status"
    res = await handle_uplink_message(
        topic_str=topic,
        payload_data='{"status": "online"}',
        status_repo=status_repo,
        identity_repo=id_repo,
        audit=audit,
        now=now,
        ip_address="192.168.1.100",
        client_id="xuna-esp32-client",
    )
    assert res["status"] == "ok"
    assert res["deviceStatus"] == "online"

    saved = await status_repo.get_status(test_pet.device_id)
    assert saved is not None
    assert saved.status == DeviceConnectionStatus.ONLINE
    assert saved.ip_address == "192.168.1.100"
    assert saved.client_id == "xuna-esp32-client"

    assert len(audit.events) == 1
    assert audit.events[0].action == AuditAction.DEVICE_STATUS_CHANGED


@pytest.mark.anyio
async def test_handle_uplink_events_and_state(test_pet: Pet) -> None:
    status_repo = _FakeStatusRepo()
    id_repo = _FakeIdentityRepo(test_pet)
    audit = _FakeAuditLog()
    now = datetime.now(UTC)

    # 1. Event
    event_topic = f"xuna/v1/{test_pet.device_id}/up/events"
    ev_res = await handle_uplink_message(
        topic_str=event_topic,
        payload_data={"eventType": "button_press", "timestamp": now.isoformat()},
        status_repo=status_repo,
        identity_repo=id_repo,
        audit=audit,
        now=now,
    )
    assert ev_res["status"] == "ok"
    assert ev_res["eventType"] == "button_press"

    # 2. State
    state_topic = f"xuna/v1/{test_pet.device_id}/up/state"
    st_res = await handle_uplink_message(
        topic_str=state_topic,
        payload_data={"snapshotGeneration": 15, "life": {"activeSeconds": 200}},
        status_repo=status_repo,
        identity_repo=id_repo,
        audit=audit,
        now=now,
    )
    assert st_res["status"] == "ok"
    assert st_res["generation"] == 15
    assert id_repo.updated_generation == 15


@pytest.mark.anyio
async def test_send_downlink_suggestion(test_pet: Pet) -> None:
    broker = InMemoryMqttBroker()
    now = datetime.now(UTC)
    s = CloudSuggestion(
        suggestion_id=uuid.uuid4(),
        pet_id=test_pet.pet_id,
        brain_id=test_pet.active_brain_id,
        kind=SuggestionKind.LEARNER_PARAMETER,
        target="learner.gamma",
        delta=0.02,
        bounds={"min": 0.70, "max": 0.95},
        provenance=SuggestionProvenance.CLOUD_INFERRED,
        expires_at=now,
        created_at=now,
    )

    topic = await send_downlink_suggestion(broker, test_pet.device_id, s)
    assert topic == f"xuna/v1/{test_pet.device_id}/down/suggest"
    assert len(broker.published) == 1
    assert broker.published[0]["topic"] == topic
    assert '"target": "learner.gamma"' in broker.published[0]["payload"]


@pytest.mark.anyio
async def test_send_downlink_command(test_pet: Pet) -> None:
    broker = InMemoryMqttBroker()
    id_repo = _FakeIdentityRepo(test_pet)
    audit = _FakeAuditLog()
    now = datetime.now(UTC)

    cmd = await send_downlink_command(
        publisher=broker,
        identity_repo=id_repo,
        audit=audit,
        owner_id=test_pet.owner_id,
        device_id=test_pet.device_id,
        action="time_sync",
        args={"utc": 1729900000},
        now=now,
    )
    assert cmd.action == "time_sync"
    assert len(broker.published) == 1
    assert broker.published[0]["topic"] == f"xuna/v1/{test_pet.device_id}/down/cmd"
    assert len(audit.events) == 1
    assert audit.events[0].action == AuditAction.DEVICE_COMMAND_SENT


@pytest.mark.anyio
async def test_get_device_status_defaults(test_pet: Pet) -> None:
    status_repo = _FakeStatusRepo()
    id_repo = _FakeIdentityRepo(test_pet)

    rec = await get_device_status(
        status_repo=status_repo,
        identity_repo=id_repo,
        owner_id=test_pet.owner_id,
        device_id=test_pet.device_id,
    )
    assert rec.status == DeviceConnectionStatus.OFFLINE


@pytest.mark.anyio
async def test_transport_error_cases(test_pet: Pet) -> None:
    status_repo = _FakeStatusRepo()
    id_repo = _FakeIdentityRepo(test_pet)
    audit = _FakeAuditLog()
    now = datetime.now(UTC)

    # 1. Unknown device in topic
    unknown_id = uuid.uuid4()
    with pytest.raises(TransportError) as exc:
        await handle_uplink_message(
            topic_str=f"xuna/v1/{unknown_id}/status",
            payload_data={"status": "online"},
            status_repo=status_repo,
            identity_repo=id_repo,
            audit=audit,
            now=now,
        )
    assert exc.value.code == "device_not_found"

    # 2. Invalid payload json
    with pytest.raises(TransportError) as exc:
        await handle_uplink_message(
            topic_str=f"xuna/v1/{test_pet.device_id}/status",
            payload_data="not-valid-json{",
            status_repo=status_repo,
            identity_repo=id_repo,
            audit=audit,
            now=now,
        )
    assert exc.value.code == "invalid_payload"
