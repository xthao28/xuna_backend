"""Application use cases for Knowledge Domains (plan §6.2, §6.3)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.knowledge.application.ports import (
    KnowledgeCandidatePort,
    KnowledgeRepository,
)
from xuna_backend.modules.knowledge.application.schemas import (
    CreateDomainRequest,
    DecideSuggestionRequest,
    DomainResponse,
    KnowledgeSuggestionResponse,
    TeachKnowledgeRequest,
    TeachKnowledgeResponse,
    UpdateDomainRequest,
)
from xuna_backend.modules.knowledge.domain.entities import (
    DomainOrigin,
    DomainStatus,
    KnowledgeDomain,
    KnowledgeSuggestion,
    RelationKind,
    SuggestionStatus,
    compute_fact_key,
)
from xuna_backend.modules.knowledge.domain.lexical_matcher import (
    compute_lexical_similarity,
)
from xuna_backend.modules.knowledge.domain.packs import get_pack
from xuna_backend.modules.memory.domain.entities import (
    CandidateStatus,
    CandidateType,
    MemoryCandidate,
)

LEXICAL_ASSIGNMENT_THRESHOLD = 0.20
LEXICAL_REFINES_THRESHOLD = 0.60


class CreateDomainUseCase:
    def __init__(self, repo: KnowledgeRepository) -> None:
        self._repo = repo

    async def execute(
        self,
        pet_id: PetId,
        owner_id: uuid.UUID,
        request: CreateDomainRequest,
        now: datetime | None = None,
    ) -> DomainResponse:
        current_now = now or datetime.now(UTC)
        count = await self._repo.count_non_deleted_domains(pet_id)
        if count >= 5:
            raise ValueError("Maximum 5 non-deleted domains allowed per pet")

        existing_slot = await self._repo.get_domain_by_slot(pet_id, request.device_slot)
        if existing_slot is not None and existing_slot.status != DomainStatus.DELETED:
            raise ValueError(f"Device slot {request.device_slot} is already occupied")

        description = request.description.strip()
        pack_id: str | None = None
        expert_brief = ""
        safety_tier = "general"

        if request.pack_id is not None:
            pack = get_pack(request.pack_id)
            if pack is None:
                raise ValueError("Invalid packId")
            pack_id = pack.pack_id
            if not description:
                description = pack.description[:160]
            expert_brief = pack.brief
            safety_tier = pack.safety_tier.value

        domain = KnowledgeDomain(
            domain_id=uuid.uuid4(),
            owner_id=owner_id,
            pet_id=pet_id,
            name=request.name.strip(),
            description=description,
            status=DomainStatus.PENDING_DEVICE,
            origin=DomainOrigin.OWNER,
            created_at=current_now,
            updated_at=current_now,
            device_slot=request.device_slot,
            pack_id=pack_id,
            expert_brief=expert_brief,
            safety_tier=safety_tier,
        )
        await self._repo.save_domain(domain)
        return _to_domain_response(domain)


class ListDomainsUseCase:
    def __init__(self, repo: KnowledgeRepository) -> None:
        self._repo = repo

    async def execute(self, pet_id: PetId, include_deleted: bool = False) -> list[DomainResponse]:
        domains = await self._repo.list_domains(pet_id, include_deleted=include_deleted)
        return [_to_domain_response(d) for d in domains]


class GetDomainUseCase:
    def __init__(self, repo: KnowledgeRepository) -> None:
        self._repo = repo

    async def execute(self, domain_id: uuid.UUID) -> DomainResponse | None:
        domain = await self._repo.get_domain(domain_id)
        return _to_domain_response(domain) if domain else None


class UpdateDomainUseCase:
    def __init__(self, repo: KnowledgeRepository, candidate_port: KnowledgeCandidatePort) -> None:
        self._repo = repo
        self._candidates = candidate_port

    async def execute(
        self,
        domain_id: uuid.UUID,
        request: UpdateDomainRequest,
        now: datetime | None = None,
    ) -> DomainResponse:
        current_now = now or datetime.now(UTC)
        domain = await self._repo.get_domain(domain_id)
        if domain is None:
            raise KeyError(f"Domain {domain_id} not found")

        match request.action:
            case "device_ack":
                updated = KnowledgeDomain(
                    domain_id=domain.domain_id,
                    owner_id=domain.owner_id,
                    pet_id=domain.pet_id,
                    name=domain.name,
                    description=domain.description,
                    status=DomainStatus.ACTIVE,
                    origin=domain.origin,
                    created_at=domain.created_at,
                    updated_at=current_now,
                    device_slot=domain.device_slot,
                    embedding=domain.embedding,
                    pack_id=domain.pack_id,
                    expert_brief=domain.expert_brief,
                    safety_tier=domain.safety_tier,
                )
            case "rename":
                if not request.name:
                    raise ValueError("name is required for rename action")
                updated = KnowledgeDomain(
                    domain_id=domain.domain_id,
                    owner_id=domain.owner_id,
                    pet_id=domain.pet_id,
                    name=request.name.strip(),
                    description=domain.description,
                    status=domain.status,
                    origin=domain.origin,
                    created_at=domain.created_at,
                    updated_at=current_now,
                    device_slot=domain.device_slot,
                    embedding=domain.embedding,
                    pack_id=domain.pack_id,
                    expert_brief=domain.expert_brief,
                    safety_tier=domain.safety_tier,
                )
            case "describe":
                if request.description is None:
                    raise ValueError("description is required for describe action")
                updated = KnowledgeDomain(
                    domain_id=domain.domain_id,
                    owner_id=domain.owner_id,
                    pet_id=domain.pet_id,
                    name=domain.name,
                    description=request.description.strip(),
                    status=domain.status,
                    origin=domain.origin,
                    created_at=domain.created_at,
                    updated_at=current_now,
                    device_slot=domain.device_slot,
                    embedding=domain.embedding,
                    pack_id=domain.pack_id,
                    expert_brief=domain.expert_brief,
                    safety_tier=domain.safety_tier,
                )
            case "set_brief":
                updated = KnowledgeDomain(
                    domain_id=domain.domain_id,
                    owner_id=domain.owner_id,
                    pet_id=domain.pet_id,
                    name=domain.name,
                    description=domain.description,
                    status=domain.status,
                    origin=domain.origin,
                    created_at=domain.created_at,
                    updated_at=current_now,
                    device_slot=domain.device_slot,
                    embedding=domain.embedding,
                    pack_id=domain.pack_id,
                    expert_brief=(request.expert_brief or "").strip(),
                    safety_tier=domain.safety_tier,
                )
            case "apply_pack":
                if not request.pack_id:
                    raise ValueError("Invalid packId")
                pack = get_pack(request.pack_id)
                if pack is None:
                    raise ValueError("Invalid packId")
                updated = KnowledgeDomain(
                    domain_id=domain.domain_id,
                    owner_id=domain.owner_id,
                    pet_id=domain.pet_id,
                    name=domain.name,
                    description=domain.description or pack.description[:160],
                    status=domain.status,
                    origin=domain.origin,
                    created_at=domain.created_at,
                    updated_at=current_now,
                    device_slot=domain.device_slot,
                    embedding=domain.embedding,
                    pack_id=pack.pack_id,
                    expert_brief=pack.brief,
                    safety_tier=pack.safety_tier.value,
                )
            case "rest":
                updated = KnowledgeDomain(
                    domain_id=domain.domain_id,
                    owner_id=domain.owner_id,
                    pet_id=domain.pet_id,
                    name=domain.name,
                    description=domain.description,
                    status=DomainStatus.RESTING,
                    origin=domain.origin,
                    created_at=domain.created_at,
                    updated_at=current_now,
                    device_slot=None,
                    embedding=domain.embedding,
                    pack_id=domain.pack_id,
                    expert_brief=domain.expert_brief,
                    safety_tier=domain.safety_tier,
                )
            case "activate":
                updated = KnowledgeDomain(
                    domain_id=domain.domain_id,
                    owner_id=domain.owner_id,
                    pet_id=domain.pet_id,
                    name=domain.name,
                    description=domain.description,
                    status=DomainStatus.ACTIVE,
                    origin=domain.origin,
                    created_at=domain.created_at,
                    updated_at=current_now,
                    device_slot=domain.device_slot,
                    embedding=domain.embedding,
                    pack_id=domain.pack_id,
                    expert_brief=domain.expert_brief,
                    safety_tier=domain.safety_tier,
                )
            case "delete":
                updated = KnowledgeDomain(
                    domain_id=domain.domain_id,
                    owner_id=domain.owner_id,
                    pet_id=domain.pet_id,
                    name=domain.name,
                    description=domain.description,
                    status=DomainStatus.DELETED,
                    origin=domain.origin,
                    created_at=domain.created_at,
                    updated_at=current_now,
                    device_slot=None,
                    embedding=domain.embedding,
                    pack_id=domain.pack_id,
                    expert_brief=domain.expert_brief,
                    safety_tier=domain.safety_tier,
                )
                candidates = await self._candidates.list_candidates_by_domain(
                    domain_id, confirmed_only=False
                )
                if request.keep_knowledge:
                    # Move knowledge to unsorted
                    candidate_ids = [c.candidate_id for c in candidates]
                    await self._candidates.update_domain_for_candidates(
                        candidate_ids, new_domain_id=None
                    )
                else:
                    # Mark all as deleted
                    for c in candidates:
                        del_cand = MemoryCandidate(
                            candidate_id=c.candidate_id,
                            owner_id=c.owner_id,
                            pet_id=c.pet_id,
                            source=c.source,
                            candidate_type=c.candidate_type,
                            content=c.content,
                            status=CandidateStatus.DELETED,
                            created_at=c.created_at,
                            updated_at=current_now,
                            confirmed_at=c.confirmed_at,
                            committed_generation=c.committed_generation,
                            domain_id=None,
                            relation=c.relation,
                            related_candidate_id=c.related_candidate_id,
                            fact_key=c.fact_key,
                        )
                        await self._candidates.save_candidate(del_cand)
            case _:
                raise ValueError(f"Unknown action: {request.action}")

        await self._repo.update_domain(updated)
        return _to_domain_response(updated)


class TeachKnowledgeUseCase:
    def __init__(
        self,
        repo: KnowledgeRepository,
        candidate_port: KnowledgeCandidatePort,
    ) -> None:
        self._repo = repo
        self._candidates = candidate_port

    async def execute(
        self,
        pet_id: PetId,
        owner_id: uuid.UUID,
        request: TeachKnowledgeRequest,
        now: datetime | None = None,
    ) -> TeachKnowledgeResponse:
        current_now = now or datetime.now(UTC)
        text = request.text.strip()
        fact_key = compute_fact_key(text)

        proposed_domain_id: uuid.UUID | None = request.domain_id
        relation = RelationKind.NEW.value
        related_id: uuid.UUID | None = None

        if proposed_domain_id is None:
            # Lexical assignment against active domains
            domains = await self._repo.list_domains(pet_id, include_deleted=False)
            active_domains = [
                d for d in domains if d.status in (DomainStatus.ACTIVE, DomainStatus.RESTING)
            ]
            best_score = 0.0
            best_domain_id: uuid.UUID | None = None

            for d in active_domains:
                facts = await self._candidates.list_candidates_by_domain(
                    d.domain_id, confirmed_only=True
                )
                fact_texts = [
                    f.content.get("text", "")
                    for f in facts
                    if isinstance(f.content.get("text"), str)
                ]
                targets = [d.name, d.description, *fact_texts]
                score = compute_lexical_similarity(text, targets)
                if score > best_score:
                    best_score = score
                    best_domain_id = d.domain_id

            if best_score >= LEXICAL_ASSIGNMENT_THRESHOLD and best_domain_id is not None:
                proposed_domain_id = best_domain_id

        if proposed_domain_id is not None:
            # Check relation with existing facts in the chosen domain
            existing_facts = await self._candidates.list_candidates_by_domain(
                proposed_domain_id, confirmed_only=True
            )
            for f in existing_facts:
                f_text = f.content.get("text", "")
                if isinstance(f_text, str) and f_text:
                    sim = compute_lexical_similarity(text, [f_text])
                    if sim >= LEXICAL_REFINES_THRESHOLD:
                        relation = RelationKind.REFINES.value
                        related_id = f.candidate_id
                        break

        candidate = MemoryCandidate(
            candidate_id=uuid.uuid4(),
            owner_id=owner_id,
            pet_id=pet_id,
            source="owner",
            candidate_type=CandidateType.KNOWLEDGE,
            content={"text": text},
            status=CandidateStatus.AWAITING_OWNER_CONFIRMATION,
            created_at=current_now,
            updated_at=current_now,
            domain_id=proposed_domain_id,
            relation=relation,
            related_candidate_id=related_id,
            fact_key=fact_key,
        )
        await self._candidates.save_candidate(candidate)

        return TeachKnowledgeResponse(
            candidate_id=candidate.candidate_id,
            fact_key=fact_key,
            text=text,
            status=candidate.status.value,
            proposed_domain_id=proposed_domain_id,
            relation=relation,
            related_candidate_id=related_id,
        )


class ListSuggestionsUseCase:
    def __init__(self, repo: KnowledgeRepository) -> None:
        self._repo = repo

    async def execute(
        self, pet_id: PetId, status: SuggestionStatus | None = None
    ) -> list[KnowledgeSuggestionResponse]:
        target_status = status or SuggestionStatus.PENDING
        suggestions = await self._repo.list_suggestions(pet_id, status=target_status)
        return [_to_suggestion_response(s) for s in suggestions]


class DecideSuggestionUseCase:
    def __init__(
        self,
        repo: KnowledgeRepository,
        candidate_port: KnowledgeCandidatePort,
    ) -> None:
        self._repo = repo
        self._candidates = candidate_port

    async def execute(
        self,
        suggestion_id: uuid.UUID,
        request: DecideSuggestionRequest,
        now: datetime | None = None,
    ) -> KnowledgeSuggestionResponse:
        current_now = now or datetime.now(UTC)
        suggestion = await self._repo.get_suggestion(suggestion_id)
        if suggestion is None:
            raise KeyError(f"Suggestion {suggestion_id} not found")
        if suggestion.status != SuggestionStatus.PENDING:
            raise ValueError(f"Suggestion {suggestion_id} is not pending")

        if request.action == "dismiss":
            updated = KnowledgeSuggestion(
                suggestion_id=suggestion.suggestion_id,
                owner_id=suggestion.owner_id,
                pet_id=suggestion.pet_id,
                kind=suggestion.kind,
                proposed_name=suggestion.proposed_name,
                proposed_description=suggestion.proposed_description,
                member_candidate_ids=suggestion.member_candidate_ids,
                status=SuggestionStatus.DISMISSED,
                created_at=suggestion.created_at,
                target_domain_id=suggestion.target_domain_id,
                decided_at=current_now,
            )
            await self._repo.update_suggestion(updated)
            return _to_suggestion_response(updated)

        # action == "accept"
        updated = KnowledgeSuggestion(
            suggestion_id=suggestion.suggestion_id,
            owner_id=suggestion.owner_id,
            pet_id=suggestion.pet_id,
            kind=suggestion.kind,
            proposed_name=suggestion.proposed_name,
            proposed_description=suggestion.proposed_description,
            member_candidate_ids=suggestion.member_candidate_ids,
            status=SuggestionStatus.ACCEPTED,
            created_at=suggestion.created_at,
            target_domain_id=suggestion.target_domain_id,
            decided_at=current_now,
        )
        await self._repo.update_suggestion(updated)
        return _to_suggestion_response(updated)


def _to_domain_response(domain: KnowledgeDomain) -> DomainResponse:
    return DomainResponse(
        domain_id=domain.domain_id,
        pet_id=str(domain.pet_id),
        name=domain.name,
        description=domain.description,
        device_slot=domain.device_slot,
        status=domain.status.value,
        origin=domain.origin.value,
        pack_id=domain.pack_id,
        expert_brief=domain.expert_brief,
        safety_tier=domain.safety_tier,
        created_at=domain.created_at.astimezone(UTC).isoformat(),
        updated_at=domain.updated_at.astimezone(UTC).isoformat(),
    )


def _to_suggestion_response(
    suggestion: KnowledgeSuggestion,
) -> KnowledgeSuggestionResponse:
    return KnowledgeSuggestionResponse(
        suggestion_id=suggestion.suggestion_id,
        pet_id=str(suggestion.pet_id),
        kind=suggestion.kind.value,
        proposed_name=suggestion.proposed_name,
        proposed_description=suggestion.proposed_description,
        member_candidate_ids=suggestion.member_candidate_ids,
        target_domain_id=suggestion.target_domain_id,
        status=suggestion.status.value,
        created_at=suggestion.created_at.astimezone(UTC).isoformat(),
    )
