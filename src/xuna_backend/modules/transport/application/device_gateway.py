"""Device-direct gateway use cases (DC-3, ADR-0010).

`open_device_session` verifies the session proof against the key pinned at bind
time and the owner's consent; `touch_device_session` re-checks both on every
heartbeat so an unbound pet or withdrawn consent ends a live session;
`close_device_session` records the board offline. The nonce lives in the
WebSocket session itself: one per connection, single-use, short-lived.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime

from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.audit.domain.event import AuditAction, AuditEvent
from xuna_backend.modules.consent.application.ports import ConsentRepository
from xuna_backend.modules.consent.domain.consent import ConsentClass
from xuna_backend.modules.identity.application.ports import (
    IdentityRepository,
    SignatureVerifier,
)
from xuna_backend.modules.identity.domain.device_proof import ProofPurpose, proof_message
from xuna_backend.modules.identity.domain.entities import Pet
from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.transport.application.ports import TransportStatusRepository
from xuna_backend.modules.transport.domain.device_session import (
    CONSENT_REQUIRED,
    NOT_BOUND,
    UNAUTHORIZED,
    SessionClose,
    SessionRejected,
)
from xuna_backend.modules.transport.domain.entities import (
    DeviceConnectionStatus,
    DeviceStatusRecord,
)

# Projections are this slice's only payload, so either backup class allows a session.
SESSION_CONSENTS = (ConsentClass.AGGREGATE_METRICS, ConsentClass.EXPERIENCE_EVENTS)


@dataclass(frozen=True, slots=True)
class SessionProof:
    pet_id: PetId
    nonce: bytes
    generation: int
    public_key: bytes | None
    signature: bytes
    firmware: str


@dataclass(frozen=True, slots=True)
class DeviceSession:
    pet_id: PetId
    owner_id: uuid.UUID
    device_id: uuid.UUID
    firmware: str
    connected_at: datetime


async def _consented(consent: ConsentRepository, owner_id: uuid.UUID) -> bool:
    state = await consent.current(owner_id)
    return any(state.allows(c) for c in SESSION_CONSENTS)


async def _reject(
    audit: AuditLog, pet: Pet, close: SessionClose, now: datetime, request_id: str
) -> SessionRejected:
    await audit.record(
        AuditEvent(
            owner_id=pet.owner_id,
            action=AuditAction.DEVICE_REJECTED,
            occurred_at=now,
            subject=pet.pet_id.value,
            request_id=request_id,
            metadata={"code": close.reason},
        )
    )
    return SessionRejected(close)


async def open_device_session(
    identity: IdentityRepository,
    verifier: SignatureVerifier,
    consent: ConsentRepository,
    statuses: TransportStatusRepository,
    audit: AuditLog,
    proof: SessionProof,
    now: datetime,
    request_id: str = "",
) -> DeviceSession:
    """Raises SessionRejected. A rejection for a bound pet is audited for its
    owner (caller commits); an unknown pet has no owner to tell."""
    pet = await identity.get_pet(proof.pet_id)
    if pet is None:
        raise SessionRejected(NOT_BOUND)
    if proof.public_key is not None and proof.public_key != pet.device_public_key:
        raise await _reject(audit, pet, UNAUTHORIZED, now, request_id)
    message = proof_message(
        proof.nonce, proof.pet_id, ProofPurpose.SESSION, proof.generation, pet.device_public_key
    )
    if not verifier.verify(pet.device_public_key, message, proof.signature):
        raise await _reject(audit, pet, UNAUTHORIZED, now, request_id)
    if not await _consented(consent, pet.owner_id):
        raise await _reject(audit, pet, CONSENT_REQUIRED, now, request_id)
    await statuses.save_status(
        DeviceStatusRecord(
            device_id=pet.device_id,
            pet_id=pet.pet_id.value,
            status=DeviceConnectionStatus.ONLINE,
            last_seen_at=now,
            firmware=proof.firmware,
            connected_at=now,
        )
    )
    await audit.record(
        AuditEvent(
            owner_id=pet.owner_id,
            action=AuditAction.DEVICE_CONNECTED,
            occurred_at=now,
            subject=pet.pet_id.value,
            request_id=request_id,
            metadata={"generation": proof.generation, "firmware": proof.firmware},
        )
    )
    return DeviceSession(pet.pet_id, pet.owner_id, pet.device_id, proof.firmware, now)


async def touch_device_session(
    identity: IdentityRepository,
    consent: ConsentRepository,
    statuses: TransportStatusRepository,
    session: DeviceSession,
    now: datetime,
) -> SessionClose | None:
    """Heartbeat: None keeps the session; otherwise the close to send."""
    pet = await identity.get_pet(session.pet_id)
    if pet is None or pet.device_id != session.device_id:
        return NOT_BOUND
    if not await _consented(consent, pet.owner_id):
        return CONSENT_REQUIRED
    await statuses.save_status(_status(session, DeviceConnectionStatus.ONLINE, now))
    return None


async def close_device_session(
    statuses: TransportStatusRepository, session: DeviceSession, now: datetime
) -> None:
    await statuses.save_status(_status(session, DeviceConnectionStatus.OFFLINE, now))


def _status(
    session: DeviceSession, status: DeviceConnectionStatus, now: datetime
) -> DeviceStatusRecord:
    return DeviceStatusRecord(
        device_id=session.device_id,
        pet_id=session.pet_id.value,
        status=status,
        last_seen_at=now,
        firmware=session.firmware,
        connected_at=session.connected_at,
    )
