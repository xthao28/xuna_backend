"""CLD-12 consent use cases and the enforcement check every data route uses."""

import uuid
from dataclasses import dataclass
from datetime import datetime

from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.audit.domain.event import AuditAction, AuditEvent
from xuna_backend.modules.consent.application.ports import ConsentRepository
from xuna_backend.modules.consent.domain.consent import ConsentClass, ConsentState


class ConsentError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class UpdateConsent:
    aggregate_metrics: bool
    experience_events: bool
    conversation_text: bool
    expected_version: int
    # None keeps the current value: clients that predate voice never revoke it.
    conversation_voice: bool | None = None


async def require_consent(
    repo: ConsentRepository, owner_id: uuid.UUID, consent: ConsentClass
) -> None:
    """Raises `consent_required` unless the owner opted in. Call before storing
    or forwarding any data of that class; U0 owners always fail."""
    if not (await repo.current(owner_id)).allows(consent):
        raise ConsentError("consent_required")


async def update_consent(
    repo: ConsentRepository,
    audit: AuditLog,
    owner_id: uuid.UUID,
    command: UpdateConsent,
    now: datetime,
    request_id: str,
) -> ConsentState:
    current = await repo.current(owner_id)
    if command.expected_version != current.version:
        raise ConsentError("consent_version_conflict")
    proposed = ConsentState(
        aggregate_metrics=command.aggregate_metrics,
        experience_events=command.experience_events,
        conversation_text=command.conversation_text,
        conversation_voice=(
            current.conversation_voice
            if command.conversation_voice is None
            else command.conversation_voice
        ),
        version=current.version + 1,
        updated_at=now,
    )
    changed = current.changed(proposed)
    if not changed:
        return current
    if not await repo.append(owner_id, proposed):
        raise ConsentError("consent_version_conflict")
    await audit.record(
        AuditEvent(
            owner_id=owner_id,
            action=AuditAction.CONSENT_CHANGED,
            occurred_at=now,
            request_id=request_id,
            metadata={
                "version": proposed.version,
                **{c.value.replace("_", "")[:31]: proposed.allows(c) for c in changed},
            },
        )
    )
    return proposed
