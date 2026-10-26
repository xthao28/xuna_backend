"""Application use cases for cloud suggestions and sleep learning (CLD-30, CLD-31, H1-H8)."""

import uuid
from datetime import datetime

from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.audit.domain.event import AuditAction, AuditEvent
from xuna_backend.modules.consent.domain.consent import ConsentClass, ConsentState
from xuna_backend.modules.identity.application.ports import IdentityRepository
from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.memory.application.ports import MemoryRepository
from xuna_backend.modules.suggestions.application.consolidation import (
    compute_consolidation_suggestions,
)
from xuna_backend.modules.suggestions.application.ports import SuggestionRepository
from xuna_backend.modules.suggestions.domain.entities import (
    CloudSuggestion,
    SuggestionKind,
    SuggestionStatus,
)


class SuggestionError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


async def consolidate_suggestions(
    repo: SuggestionRepository,
    identity_repo: IdentityRepository,
    memory_repo: MemoryRepository,
    audit: AuditLog,
    owner_id: uuid.UUID,
    pet_id: PetId,
    consent: ConsentState,
    now: datetime,
    request_id: str = "",
) -> list[CloudSuggestion]:
    """Runs sleep learning consolidation under U1/U2 consent (H6, O1, CLD-30)."""
    # H6: Opt-in data classes. Default U0 rejects with consent_required.
    if not (
        consent.allows(ConsentClass.AGGREGATE_METRICS)
        or consent.allows(ConsentClass.EXPERIENCE_EVENTS)
    ):
        raise SuggestionError("consent_required")

    pet = await identity_repo.get_pet(pet_id)
    if pet is None or pet.owner_id != owner_id:
        raise SuggestionError("not_found")

    latest_proj = await memory_repo.get_latest_projection(pet_id)
    suggestions = compute_consolidation_suggestions(
        pet_id=pet_id,
        brain_id=pet.active_brain_id,
        latest_projection=latest_proj,
        now=now,
    )

    for suggestion in suggestions:
        await repo.save(suggestion)

    await audit.record(
        AuditEvent(
            owner_id=owner_id,
            action=AuditAction.SUGGESTION_CONSOLIDATED,
            occurred_at=now,
            subject=pet_id.value,
            request_id=request_id,
            metadata={"count": len(suggestions)},
        )
    )

    return suggestions


async def list_suggestions(
    repo: SuggestionRepository,
    identity_repo: IdentityRepository,
    owner_id: uuid.UUID,
    pet_id: PetId,
    status: SuggestionStatus | None = None,
    kind: SuggestionKind | None = None,
) -> list[CloudSuggestion]:
    pet = await identity_repo.get_pet(pet_id)
    if pet is None or pet.owner_id != owner_id:
        raise SuggestionError("not_found")

    return await repo.list_for_pet(pet_id, status=status, kind=kind)


async def get_suggestion(
    repo: SuggestionRepository,
    identity_repo: IdentityRepository,
    owner_id: uuid.UUID,
    pet_id: PetId,
    suggestion_id: uuid.UUID,
) -> CloudSuggestion:
    pet = await identity_repo.get_pet(pet_id)
    if pet is None or pet.owner_id != owner_id:
        raise SuggestionError("not_found")

    suggestion = await repo.get_by_id(suggestion_id)
    if suggestion is None or suggestion.pet_id != pet_id:
        raise SuggestionError("not_found")

    return suggestion


async def update_suggestion_status(
    repo: SuggestionRepository,
    identity_repo: IdentityRepository,
    audit: AuditLog,
    owner_id: uuid.UUID,
    pet_id: PetId,
    suggestion_id: uuid.UUID,
    action: str,
    now: datetime,
    request_id: str = "",
) -> CloudSuggestion:
    pet = await identity_repo.get_pet(pet_id)
    if pet is None or pet.owner_id != owner_id:
        raise SuggestionError("not_found")

    suggestion = await repo.get_by_id(suggestion_id)
    if suggestion is None or suggestion.pet_id != pet_id:
        raise SuggestionError("not_found")

    applied_at: datetime | None = None
    target_status: SuggestionStatus
    audit_action: AuditAction

    match action:
        case "apply":
            if suggestion.is_expired(now):
                raise SuggestionError("suggestion_expired")
            target_status = SuggestionStatus.APPLIED
            applied_at = now
            audit_action = AuditAction.SUGGESTION_APPLIED
        case "reject":
            target_status = SuggestionStatus.REJECTED
            applied_at = None
            audit_action = AuditAction.SUGGESTION_REJECTED
        case "revert":
            target_status = SuggestionStatus.REVERTED
            applied_at = suggestion.applied_at
            audit_action = AuditAction.SUGGESTION_REVERTED
        case _:
            raise SuggestionError("invalid_action")

    if not suggestion.can_transition_to(target_status):
        raise SuggestionError("invalid_transition")

    updated = await repo.update_status(
        suggestion_id=suggestion_id,
        status=target_status,
        applied_at=applied_at,
    )
    if updated is None:
        raise SuggestionError("not_found")

    await audit.record(
        AuditEvent(
            owner_id=owner_id,
            action=audit_action,
            occurred_at=now,
            subject=pet_id.value,
            request_id=request_id,
            metadata={"status": target_status.value},
        )
    )

    return updated


async def delete_suggestion(
    repo: SuggestionRepository,
    identity_repo: IdentityRepository,
    owner_id: uuid.UUID,
    pet_id: PetId,
    suggestion_id: uuid.UUID,
) -> bool:
    pet = await identity_repo.get_pet(pet_id)
    if pet is None or pet.owner_id != owner_id:
        raise SuggestionError("not_found")

    suggestion = await repo.get_by_id(suggestion_id)
    if suggestion is None or suggestion.pet_id != pet_id:
        raise SuggestionError("not_found")

    return await repo.delete(suggestion_id)
