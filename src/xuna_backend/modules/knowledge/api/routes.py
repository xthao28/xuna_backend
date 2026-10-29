"""Knowledge Domains API endpoints (plan §6.2, §6.3, KD-3, KX-3)."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Query, Request

from xuna_backend.core.db import commit_or_conflict
from xuna_backend.core.errors import AppError
from xuna_backend.modules.accounts.api.dependencies import CurrentOwner, Session, state
from xuna_backend.modules.accounts.domain.owner import Owner
from xuna_backend.modules.cognition.application.ports import (
    ProviderUnavailableError,
    TextGenerator,
)
from xuna_backend.modules.consent.api.routes import RequireConsent
from xuna_backend.modules.consent.domain.consent import ConsentClass
from xuna_backend.modules.identity.application.use_cases import IdentityError, get_owned_pet
from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.knowledge.application.brief_draft import (
    DraftExpertBriefUseCase,
)
from xuna_backend.modules.knowledge.application.cases import (
    ProposeKnowledgeCaseUseCase,
)
from xuna_backend.modules.knowledge.application.ports import (
    KnowledgeCandidatePort,
    KnowledgeRepository,
)
from xuna_backend.modules.knowledge.application.schemas import (
    CreateDomainRequest,
    DecideSuggestionRequest,
    DomainResponse,
    DraftBriefRequest,
    DraftBriefResponse,
    KnowledgePackResponse,
    KnowledgeSuggestionResponse,
    ProposeKnowledgeCaseRequest,
    TeachKnowledgeRequest,
    TeachKnowledgeResponse,
    UpdateDomainRequest,
)
from xuna_backend.modules.knowledge.application.suggest import (
    GenerateKnowledgeSuggestionUseCase,
)
from xuna_backend.modules.knowledge.application.use_cases import (
    CreateDomainUseCase,
    DecideSuggestionUseCase,
    GetDomainUseCase,
    ListDomainsUseCase,
    ListSuggestionsUseCase,
    TeachKnowledgeUseCase,
    UpdateDomainUseCase,
)
from xuna_backend.modules.knowledge.domain.entities import (
    DomainStatus,
    SuggestionStatus,
)
from xuna_backend.modules.knowledge.domain.packs import PACKS

router = APIRouter(prefix="/v1", tags=["knowledge"])
U3Owner = Annotated[Owner, RequireConsent(ConsentClass.CONVERSATION_TEXT)]


def _repo(request: Request, session: object) -> KnowledgeRepository:
    factory: Callable[[object], KnowledgeRepository] = state(request, "knowledge_repository")
    return factory(session)


def _candidate_port(request: Request, session: object) -> KnowledgeCandidatePort:
    factory: Callable[[object], KnowledgeCandidatePort] = state(
        request, "knowledge_candidate_adapter"
    )
    return factory(session)


def _map_error(err: ValueError) -> AppError:
    msg = str(err)
    if "Domain limit reached" in msg or "Maximum 5 non-deleted domains" in msg:
        return AppError("domain_limit_reached", HTTPStatus.CONFLICT, msg)
    if "is already occupied" in msg:
        return AppError("slot_occupied", HTTPStatus.CONFLICT, msg)
    if "not found" in msg.lower():
        return AppError("not_found", HTTPStatus.NOT_FOUND, msg)
    if "Invalid domain state transition" in msg or "already been decided" in msg:
        return AppError("invalid_transition", HTTPStatus.CONFLICT, msg)
    if "slot must be provided" in msg or "device_slot is required" in msg:
        return AppError("slot_required", HTTPStatus.UNPROCESSABLE_ENTITY, msg)
    if "Invalid packId" in msg:
        return AppError("validation_failed", HTTPStatus.UNPROCESSABLE_ENTITY, "body.packId")
    return AppError("invalid_argument", HTTPStatus.BAD_REQUEST, msg)


@router.get("/knowledge-packs", response_model=list[KnowledgePackResponse])
async def list_knowledge_packs(
    _owner: CurrentOwner,
) -> list[KnowledgePackResponse]:
    return [
        KnowledgePackResponse(
            pack_id=pack.pack_id,
            name=pack.name,
            description=pack.description,
            safety_tier=pack.safety_tier.value,
        )
        for pack in PACKS
    ]


@router.get("/pets/{pet_id}/knowledge-domains", response_model=list[DomainResponse])
async def list_domains(
    pet_id: str,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    include_deleted: bool = Query(default=False, alias="includeDeleted"),
) -> list[DomainResponse]:
    try:
        identity_repo = state(request, "identity_repository")(session)
        await get_owned_pet(identity_repo, owner.owner_id, PetId(pet_id))
    except (ValueError, IdentityError) as err:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from err

    repo = _repo(request, session)
    use_case = ListDomainsUseCase(repo)
    return await use_case.execute(PetId(pet_id), include_deleted=include_deleted)


@router.post(
    "/pets/{pet_id}/knowledge-domains",
    response_model=DomainResponse,
    status_code=HTTPStatus.CREATED,
)
async def create_domain(
    pet_id: str,
    body: CreateDomainRequest,
    request: Request,
    owner: CurrentOwner,
    session: Session,
) -> DomainResponse:
    try:
        identity_repo = state(request, "identity_repository")(session)
        await get_owned_pet(identity_repo, owner.owner_id, PetId(pet_id))
    except (ValueError, IdentityError) as err:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from err

    repo = _repo(request, session)
    use_case = CreateDomainUseCase(repo)
    try:
        res = await use_case.execute(PetId(pet_id), owner.owner_id, body)
    except ValueError as err:
        raise _map_error(err) from err
    await commit_or_conflict(session)
    return res


@router.get("/knowledge-domains/{domain_id}", response_model=DomainResponse)
async def get_domain(
    domain_id: uuid.UUID,
    request: Request,
    owner: CurrentOwner,
    session: Session,
) -> DomainResponse:
    repo = _repo(request, session)
    domain = await repo.get_domain(domain_id)
    if domain is None or domain.owner_id != owner.owner_id:
        raise AppError("not_found", HTTPStatus.NOT_FOUND, "Domain not found")
    use_case = GetDomainUseCase(repo)
    res = await use_case.execute(domain_id)
    if res is None:
        raise AppError("not_found", HTTPStatus.NOT_FOUND, "Domain not found")
    return res


@router.patch("/knowledge-domains/{domain_id}", response_model=DomainResponse)
async def update_domain(
    domain_id: uuid.UUID,
    body: UpdateDomainRequest,
    request: Request,
    owner: CurrentOwner,
    session: Session,
) -> DomainResponse:
    repo = _repo(request, session)
    domain = await repo.get_domain(domain_id)
    if domain is None or domain.owner_id != owner.owner_id:
        raise AppError("not_found", HTTPStatus.NOT_FOUND, "Domain not found")
    candidate_port = _candidate_port(request, session)
    use_case = UpdateDomainUseCase(repo, candidate_port)
    try:
        res = await use_case.execute(domain_id, body)
    except (KeyError, ValueError) as err:
        if isinstance(err, KeyError):
            raise AppError("not_found", HTTPStatus.NOT_FOUND, str(err)) from err
        raise _map_error(err) from err
    await commit_or_conflict(session)
    return res


@router.post(
    "/knowledge-domains/{domain_id}/brief-drafts",
    response_model=DraftBriefResponse,
)
async def draft_domain_brief(
    domain_id: uuid.UUID,
    _body: DraftBriefRequest,
    request: Request,
    owner: U3Owner,
    session: Session,
) -> DraftBriefResponse:
    repo = _repo(request, session)
    domain = await repo.get_domain(domain_id)
    if domain is None or domain.owner_id != owner.owner_id or domain.status == DomainStatus.DELETED:
        raise AppError("not_found", HTTPStatus.NOT_FOUND, "Domain not found")

    expert_gen: TextGenerator | None = getattr(request.app.state, "expert_text_generator", None)
    gen: TextGenerator = expert_gen or state(request, "text_generator")
    use_case = DraftExpertBriefUseCase(gen)
    try:
        draft = await use_case.execute(domain.name, domain.description)
    except ProviderUnavailableError as err:
        raise AppError("provider_unavailable", HTTPStatus.SERVICE_UNAVAILABLE) from err
    return DraftBriefResponse(draft=draft)


@router.post(
    "/pets/{pet_id}/knowledge",
    response_model=TeachKnowledgeResponse,
    status_code=HTTPStatus.CREATED,
)
async def teach_knowledge(
    pet_id: str,
    body: TeachKnowledgeRequest,
    request: Request,
    owner: U3Owner,
    session: Session,
) -> TeachKnowledgeResponse:
    try:
        identity_repo = state(request, "identity_repository")(session)
        await get_owned_pet(identity_repo, owner.owner_id, PetId(pet_id))
    except (ValueError, IdentityError) as err:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from err

    repo = _repo(request, session)
    candidate_port = _candidate_port(request, session)
    use_case = TeachKnowledgeUseCase(repo, candidate_port)
    try:
        res = await use_case.execute(PetId(pet_id), owner.owner_id, body)
    except ValueError as err:
        raise _map_error(err) from err
    await commit_or_conflict(session)
    return res


@router.post(
    "/pets/{pet_id}/knowledge-cases",
    response_model=TeachKnowledgeResponse,
    status_code=HTTPStatus.CREATED,
)
async def propose_knowledge_case(
    pet_id: str,
    body: ProposeKnowledgeCaseRequest,
    request: Request,
    owner: U3Owner,
    session: Session,
) -> TeachKnowledgeResponse:
    try:
        identity_repo = state(request, "identity_repository")(session)
        await get_owned_pet(identity_repo, owner.owner_id, PetId(pet_id))
    except (ValueError, IdentityError) as err:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from err

    repo = _repo(request, session)
    candidate_port = _candidate_port(request, session)
    expert_gen: TextGenerator | None = getattr(request.app.state, "expert_text_generator", None)
    gen: TextGenerator = expert_gen or state(request, "text_generator")
    use_case = ProposeKnowledgeCaseUseCase(gen, repo, candidate_port)
    try:
        res = await use_case.execute(PetId(pet_id), owner.owner_id, body)
    except KeyError as err:
        raise AppError("not_found", HTTPStatus.NOT_FOUND, str(err)) from err
    except ValueError as err:
        raise _map_error(err) from err
    except ProviderUnavailableError as err:
        raise AppError("provider_unavailable", HTTPStatus.SERVICE_UNAVAILABLE) from err
    await commit_or_conflict(session)
    return res


@router.get(
    "/pets/{pet_id}/knowledge-suggestions",
    response_model=list[KnowledgeSuggestionResponse],
)
async def list_suggestions(
    pet_id: str,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    status: SuggestionStatus | None = None,
) -> list[KnowledgeSuggestionResponse]:
    try:
        identity_repo = state(request, "identity_repository")(session)
        await get_owned_pet(identity_repo, owner.owner_id, PetId(pet_id))
    except (ValueError, IdentityError) as err:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from err

    repo = _repo(request, session)
    use_case = ListSuggestionsUseCase(repo)
    return await use_case.execute(PetId(pet_id), status=status)


@router.patch(
    "/knowledge-suggestions/{suggestion_id}",
    response_model=KnowledgeSuggestionResponse,
)
async def decide_suggestion(
    suggestion_id: uuid.UUID,
    body: DecideSuggestionRequest,
    request: Request,
    owner: CurrentOwner,
    session: Session,
) -> KnowledgeSuggestionResponse:
    repo = _repo(request, session)
    candidate_port = _candidate_port(request, session)
    use_case = DecideSuggestionUseCase(repo, candidate_port)
    try:
        res = await use_case.execute(suggestion_id, body)
    except ValueError as err:
        raise _map_error(err) from err
    await commit_or_conflict(session)
    return res


@router.post(
    "/pets/{pet_id}/knowledge-suggestions/generate",
    response_model=KnowledgeSuggestionResponse | None,
)
async def generate_suggestion(
    pet_id: str,
    request: Request,
    owner: CurrentOwner,
    session: Session,
) -> KnowledgeSuggestionResponse | None:
    try:
        identity_repo = state(request, "identity_repository")(session)
        await get_owned_pet(identity_repo, owner.owner_id, PetId(pet_id))
    except (ValueError, IdentityError) as err:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from err

    repo = _repo(request, session)
    candidate_port = _candidate_port(request, session)
    embedder = getattr(request.app.state, "embedder", None)
    use_case = GenerateKnowledgeSuggestionUseCase(repo, candidate_port, embedder)
    suggestion = await use_case.execute(PetId(pet_id), owner.owner_id)
    if suggestion is None:
        return None
    await commit_or_conflict(session)
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
