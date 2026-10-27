"""Knowledge suggestion use cases (plan §6.3.4, KD-4).

Application layer rules: stdlib, domain, application only. No frameworks, no core.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.knowledge.domain.clustering import (
    check_cluster_duplicate,
    cluster_candidates_lexical,
    derive_cluster_name_and_description,
    is_suggestion_expired,
)
from xuna_backend.modules.knowledge.domain.entities import (
    DomainStatus,
    KnowledgeSuggestion,
    SuggestionKind,
    SuggestionStatus,
)

if TYPE_CHECKING:
    from xuna_backend.modules.cognition.application.ports import Embedder
    from xuna_backend.modules.knowledge.application.ports import (
        KnowledgeCandidatePort,
        KnowledgeRepository,
    )


class GenerateKnowledgeSuggestionUseCase:
    """Evaluates unsorted knowledge and proposes a new domain or add_to_domain (plan §6.3.4)."""

    def __init__(
        self,
        repo: KnowledgeRepository,
        candidate_port: KnowledgeCandidatePort,
        embedder: Embedder | None = None,
    ) -> None:
        self._repo = repo
        self._candidates = candidate_port
        self._embedder = embedder

    async def execute(
        self,
        pet_id: PetId,
        owner_id: uuid.UUID,
        now: datetime | None = None,
    ) -> KnowledgeSuggestion | None:
        current_now = now or datetime.now(UTC)

        # 1. Expire outdated pending suggestions (14 days)
        pending_list = await self._repo.list_suggestions(
            pet_id, status=SuggestionStatus.PENDING
        )
        for s in pending_list:
            if is_suggestion_expired(s.created_at, current_now, 14):
                expired_s = KnowledgeSuggestion(
                    suggestion_id=s.suggestion_id,
                    owner_id=s.owner_id,
                    pet_id=s.pet_id,
                    kind=s.kind,
                    proposed_name=s.proposed_name,
                    proposed_description=s.proposed_description,
                    member_candidate_ids=s.member_candidate_ids,
                    status=SuggestionStatus.EXPIRED,
                    created_at=s.created_at,
                    target_domain_id=s.target_domain_id,
                    decided_at=current_now,
                )
                await self._repo.update_suggestion(expired_s)

        # 2. Check if an active pending suggestion already exists
        active_pending = await self._repo.get_pending_suggestion(pet_id)
        if active_pending is not None:
            return None

        # 3. Fetch unsorted confirmed knowledge candidates
        unsorted = await self._candidates.list_unsorted_knowledge(
            pet_id, confirmed_only=True
        )
        if len(unsorted) < 4:
            return None

        # 4. Fetch non-deleted domains
        domains = await self._repo.list_domains(pet_id, include_deleted=False)
        active_domains = [d for d in domains if d.status != DomainStatus.DELETED]

        # 5. Cluster unsorted candidates
        clusters = cluster_candidates_lexical(
            unsorted, min_similarity=0.02, min_cluster_size=4
        )
        if not clusters:
            return None
        target_cluster = clusters[0]

        # 6. Check duplicate against existing domains (measured empirical threshold >= 0.10)
        dup_domain, sim = check_cluster_duplicate(
            target_cluster, active_domains, duplicate_threshold=0.10
        )
        if dup_domain is not None:
            suggestion = KnowledgeSuggestion(
                suggestion_id=uuid.uuid4(),
                owner_id=owner_id,
                pet_id=pet_id,
                kind=SuggestionKind.ADD_TO_DOMAIN,
                proposed_name=dup_domain.name,
                proposed_description=dup_domain.description,
                member_candidate_ids=[c.candidate_id for c in target_cluster],
                status=SuggestionStatus.PENDING,
                created_at=current_now,
                target_domain_id=dup_domain.domain_id,
            )
        else:
            if len(active_domains) >= 5:
                # Max 5 active slots reached, cannot create a new domain
                return None
            prop_name, prop_desc = derive_cluster_name_and_description(target_cluster)
            suggestion = KnowledgeSuggestion(
                suggestion_id=uuid.uuid4(),
                owner_id=owner_id,
                pet_id=pet_id,
                kind=SuggestionKind.NEW_DOMAIN,
                proposed_name=prop_name,
                proposed_description=prop_desc,
                member_candidate_ids=[c.candidate_id for c in target_cluster],
                status=SuggestionStatus.PENDING,
                created_at=current_now,
                target_domain_id=None,
            )

        await self._repo.save_suggestion(suggestion)
        return suggestion


class ExpireKnowledgeSuggestionsUseCase:
    """Marks pending suggestions older than 14 days as EXPIRED (plan §6.3.4)."""

    def __init__(self, repo: KnowledgeRepository) -> None:
        self._repo = repo

    async def execute(self, pet_id: PetId, now: datetime | None = None) -> int:
        current_now = now or datetime.now(UTC)
        pending_list = await self._repo.list_suggestions(
            pet_id, status=SuggestionStatus.PENDING
        )
        count = 0
        for s in pending_list:
            if is_suggestion_expired(s.created_at, current_now, 14):
                expired_s = KnowledgeSuggestion(
                    suggestion_id=s.suggestion_id,
                    owner_id=s.owner_id,
                    pet_id=s.pet_id,
                    kind=s.kind,
                    proposed_name=s.proposed_name,
                    proposed_description=s.proposed_description,
                    member_candidate_ids=s.member_candidate_ids,
                    status=SuggestionStatus.EXPIRED,
                    created_at=s.created_at,
                    target_domain_id=s.target_domain_id,
                    decided_at=current_now,
                )
                await self._repo.update_suggestion(expired_s)
                count += 1
        return count
