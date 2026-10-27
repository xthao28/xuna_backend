"""Device-direct session rules, frames and use cases (DC-3), without a database."""

import asyncio
import base64
import dataclasses
import uuid
from datetime import UTC, datetime

import pytest
from tests.helpers import MemoryAuditLog, SimulatedDevice

from xuna_backend.modules.audit.domain.event import AuditAction
from xuna_backend.modules.consent.domain.consent import ConsentState
from xuna_backend.modules.identity.domain.device_proof import ProofPurpose, proof_message
from xuna_backend.modules.identity.domain.entities import Pet
from xuna_backend.modules.identity.domain.pet_id import PetId, PetOrigin
from xuna_backend.modules.identity.infrastructure.p256 import P256SignatureVerifier
from xuna_backend.modules.transport.application.device_frames import (
    HelloFrame,
    PingFrame,
    ProofFrame,
    challenge_frame,
    decode_b64,
    parse_device_frame,
)
from xuna_backend.modules.transport.application.device_gateway import (
    DeviceSession,
    SessionProof,
    close_device_session,
    open_device_session,
    touch_device_session,
)
from xuna_backend.modules.transport.domain.device_session import (
    CONSENT_REQUIRED,
    NOT_BOUND,
    UNAUTHORIZED,
    UNSUPPORTED_PROTOCOL,
    GatewayLimits,
    SessionRejected,
    SlidingWindowLimiter,
)
from xuna_backend.modules.transport.domain.entities import (
    DeviceConnectionStatus,
    DeviceStatusRecord,
)
from xuna_backend.modules.transport.infrastructure.session_hub import InMemoryDeviceSessionHub

PET = PetId("d945c77026fd4f4a8a899bbff4aeed86")
NOW = datetime(2026, 10, 27, 12, tzinfo=UTC)
NONCE = bytes(range(32))


class FakeIdentity:
    def __init__(self, pet: Pet | None) -> None:
        self.pet = pet

    async def get_pet(self, pet_id: PetId) -> Pet | None:
        return self.pet if self.pet is not None and self.pet.pet_id == pet_id else None


class FakeConsent:
    def __init__(self, state: ConsentState) -> None:
        self.state = state

    async def current(self, owner_id: uuid.UUID) -> ConsentState:
        return self.state


class FakeStatuses:
    def __init__(self) -> None:
        self.saved: list[DeviceStatusRecord] = []

    async def save_status(self, record: DeviceStatusRecord) -> None:
        self.saved.append(record)


def make_pet(device: SimulatedDevice) -> Pet:
    return Pet(
        pet_id=PET,
        owner_id=uuid.uuid4(),
        origin=PetOrigin.MIGRATED_FROM_V1,
        active_brain_id=uuid.uuid4(),
        device_id=uuid.uuid4(),
        device_public_key=device.public_key,
        last_generation=38,
        bound_at=NOW,
    )


def session_proof(
    device: SimulatedDevice, *, purpose: ProofPurpose = ProofPurpose.SESSION, **overrides: object
) -> SessionProof:
    message = proof_message(NONCE, PET, purpose, 40, device.public_key)
    fields: dict[str, object] = {
        "pet_id": PET,
        "nonce": NONCE,
        "generation": 40,
        "public_key": None,
        "signature": device.sign(message),
        "firmware": "0.2.0-mvp",
    }
    fields.update(overrides)
    return SessionProof(**fields)


def open_with(
    pet: Pet | None,
    proof: SessionProof,
    consent: ConsentState | None = None,
) -> tuple[DeviceSession | SessionRejected, FakeStatuses, MemoryAuditLog]:
    statuses, audit = FakeStatuses(), MemoryAuditLog()

    async def run() -> DeviceSession | SessionRejected:
        try:
            return await open_device_session(
                FakeIdentity(pet),
                P256SignatureVerifier(),
                FakeConsent(consent or ConsentState(aggregate_metrics=True)),
                statuses,
                audit,
                proof,
                NOW,
            )
        except SessionRejected as error:
            return error

    return asyncio.run(run()), statuses, audit


def test_limiter_sliding_window_per_key() -> None:
    limiter = SlidingWindowLimiter(2, 10.0)
    assert limiter.allow("a", 0.0) and limiter.allow("a", 1.0)
    assert not limiter.allow("a", 2.0)
    assert limiter.allow("b", 2.0)  # keys are independent
    assert limiter.allow("a", 10.5)  # the first event left the window


def test_limits_reject_short_heartbeat() -> None:
    assert GatewayLimits().idle_timeout_s == 90.0
    with pytest.raises(ValueError):
        GatewayLimits(heartbeat_s=5)


def test_frames_parse_strictly() -> None:
    hello = parse_device_frame(
        '{"type":"hello","schemaVersion":1,"petId":"d945c77026fd4f4a8a899bbff4aeed86",'
        '"firmware":"0.2.0-mvp","capabilities":["projection.v1"]}'
    )
    assert isinstance(hello, HelloFrame) and hello.firmware == "0.2.0-mvp"
    assert isinstance(parse_device_frame('{"type":"ping","schemaVersion":1,"seq":7}'), PingFrame)
    proof = parse_device_frame(
        '{"type":"proof","schemaVersion":1,"generation":4,"signature":"AAAA"}'
    )
    assert isinstance(proof, ProofFrame) and proof.publicKey is None
    for bad in (
        "not json",
        '{"type":"ping","schemaVersion":2,"seq":1}',
        '{"type":"ping","schemaVersion":1,"seq":1,"extra":true}',
        '{"type":"ping","schemaVersion":1,"seq":"1"}',
        '{"type":"ping","schemaVersion":1,"seq":4294967296}',
        '{"type":"command","schemaVersion":1}',
        '{"type":"hello","schemaVersion":1,"petId":"D945","firmware":"x","capabilities":[]}',
        '{"type":"hello","schemaVersion":1,"petId":"d945c77026fd4f4a8a899bbff4aeed86",'
        '"firmware":"has space","capabilities":[]}',
    ):
        with pytest.raises(SessionRejected) as error:
            parse_device_frame(bad)
        assert error.value.close == UNSUPPORTED_PROTOCOL


def test_base64_sizes_are_exact() -> None:
    assert decode_b64(base64.b64encode(b"x" * 64).decode(), 64) == b"x" * 64
    for bad, size in (("AAAA", 64), ("!!!!", 3), (base64.b64encode(b"x" * 65).decode(), 64)):
        with pytest.raises(SessionRejected):
            decode_b64(bad, size)
    assert '"expiresIn":60' in challenge_frame(NONCE, 60)


def test_open_session_verifies_proof_and_records_online() -> None:
    device = SimulatedDevice()
    pet = make_pet(device)
    result, statuses, audit = open_with(pet, session_proof(device))
    assert isinstance(result, DeviceSession)
    assert result.device_id == pet.device_id and result.firmware == "0.2.0-mvp"
    assert statuses.saved[-1].status == DeviceConnectionStatus.ONLINE
    assert statuses.saved[-1].connected_at == NOW
    assert [e.action for e in audit.events] == [AuditAction.DEVICE_CONNECTED]
    assert audit.events[0].metadata == {"generation": 40, "firmware": "0.2.0-mvp"}


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("unknown_pet", NOT_BOUND),
        ("other_key", UNAUTHORIZED),
        ("bind_purpose", UNAUTHORIZED),
        ("other_nonce", UNAUTHORIZED),
        ("public_key_mismatch", UNAUTHORIZED),
        ("consent_off", CONSENT_REQUIRED),
    ],
)
def test_open_session_rejections(case: str, expected: object) -> None:
    device, stranger = SimulatedDevice(), SimulatedDevice()
    pet = make_pet(device)
    proof = session_proof(device)
    consent = ConsentState(aggregate_metrics=True)
    if case == "unknown_pet":
        pet = None
    elif case == "other_key":
        proof = session_proof(stranger)
    elif case == "bind_purpose":
        proof = session_proof(device, purpose=ProofPurpose.BIND)  # a bind proof is not a session
    elif case == "other_nonce":
        proof = session_proof(device, nonce=bytes(32))  # replay against a fresh challenge
    elif case == "public_key_mismatch":
        proof = session_proof(device, public_key=stranger.public_key)
    elif case == "consent_off":
        consent = ConsentState()  # U0
    result, statuses, audit = open_with(pet, proof, consent)
    assert isinstance(result, SessionRejected) and result.close == expected
    assert statuses.saved == []
    if pet is None:
        assert audit.events == []  # nobody to tell
    else:
        assert [e.action for e in audit.events] == [AuditAction.DEVICE_REJECTED]
        assert audit.events[0].metadata == {"code": result.close.reason}


def test_experience_events_consent_alone_is_enough() -> None:
    device = SimulatedDevice()
    result, _, _ = open_with(
        make_pet(device), session_proof(device), ConsentState(experience_events=True)
    )
    assert isinstance(result, DeviceSession)


def test_touch_and_close() -> None:
    device = SimulatedDevice()
    pet = make_pet(device)
    session = DeviceSession(pet.pet_id, pet.owner_id, pet.device_id, "0.2.0-mvp", NOW)
    statuses = FakeStatuses()

    async def touch(identity_pet: Pet | None, consent: ConsentState) -> object:
        return await touch_device_session(
            FakeIdentity(identity_pet),
            FakeConsent(consent),
            statuses,
            session,
            NOW,
        )

    assert asyncio.run(touch(pet, ConsentState(aggregate_metrics=True))) is None
    assert statuses.saved[-1].status == DeviceConnectionStatus.ONLINE
    rebound = dataclasses.replace(pet, device_id=uuid.uuid4())
    assert asyncio.run(touch(rebound, ConsentState(aggregate_metrics=True))) == NOT_BOUND
    assert asyncio.run(touch(None, ConsentState(aggregate_metrics=True))) == NOT_BOUND
    assert asyncio.run(touch(pet, ConsentState())) == CONSENT_REQUIRED
    asyncio.run(close_device_session(statuses, session, NOW))
    assert statuses.saved[-1].status == DeviceConnectionStatus.OFFLINE
    assert statuses.saved[-1].firmware == "0.2.0-mvp"


def test_hub_newest_session_wins() -> None:
    hub = InMemoryDeviceSessionHub()
    closed: list[str] = []
    first, second = object(), object()

    async def run() -> None:
        async def close_first() -> None:
            closed.append("first")

        async def close_second() -> None:
            closed.append("second")

        await hub.claim(PET.value, first, close_first)
        await hub.claim(PET.value, second, close_second)
        assert closed == ["first"]
        assert not hub.release(PET.value, first)  # the replaced session must not mark offline
        assert hub.release(PET.value, second)
        assert hub.live() == 0

    asyncio.run(run())
