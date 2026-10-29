"""SQLAlchemy repository for Knowledge Domains (plan §6.2, §6.3, KD-3)."""

from __future__ import annotations

import uuid

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.knowledge.domain.documents import (
    DocumentMediaType,
    DocumentStatus,
    KnowledgeChunk,
    KnowledgeDocument,
)
from xuna_backend.modules.knowledge.domain.entities import (
    DomainOrigin,
    DomainStatus,
    KnowledgeDomain,
    KnowledgeSuggestion,
    KnowledgeSummary,
    SuggestionKind,
    SuggestionStatus,
)
from xuna_backend.modules.knowledge.infrastructure.models import (
    KnowledgeChunkRow,
    KnowledgeDocumentRow,
    KnowledgeDomainRow,
    KnowledgeSuggestionRow,
    KnowledgeSummaryRow,
)
from xuna_backend.modules.memory.domain.entities import (
    CandidateStatus,
    CandidateType,
    MemoryCandidate,
)
from xuna_backend.modules.memory.infrastructure.models import MemoryCandidateRow


def _to_domain(row: KnowledgeDomainRow) -> KnowledgeDomain:
    return KnowledgeDomain(
        domain_id=row.domain_id,
        pet_id=PetId(row.pet_id),
        owner_id=row.owner_id,
        name=row.name,
        description=row.description,
        status=DomainStatus(row.status),
        origin=DomainOrigin(row.origin),
        created_at=row.created_at,
        updated_at=row.updated_at,
        device_slot=row.device_slot,
        embedding=list(row.embedding) if row.embedding is not None else None,
        pack_id=row.pack_id,
        expert_brief=row.expert_brief,
        safety_tier=row.safety_tier,
    )


def _to_suggestion(row: KnowledgeSuggestionRow) -> KnowledgeSuggestion:
    return KnowledgeSuggestion(
        suggestion_id=row.suggestion_id,
        owner_id=row.owner_id,
        pet_id=PetId(row.pet_id),
        kind=SuggestionKind(row.kind),
        proposed_name=row.proposed_name,
        proposed_description=row.proposed_description,
        member_candidate_ids=[
            uuid.UUID(cid) if isinstance(cid, str) else cid for cid in row.member_candidate_ids
        ],
        status=SuggestionStatus(row.status),
        created_at=row.created_at,
        target_domain_id=row.target_domain_id,
        decided_at=row.decided_at,
    )


def _to_summary(row: KnowledgeSummaryRow) -> KnowledgeSummary:
    return KnowledgeSummary(
        domain_id=row.domain_id,
        text=row.text,
        source_candidate_ids=[
            uuid.UUID(cid) if isinstance(cid, str) else cid for cid in row.source_candidate_ids
        ],
        generated_at=row.generated_at,
        stale=row.stale,
    )


def _to_document(row: KnowledgeDocumentRow) -> KnowledgeDocument:
    return KnowledgeDocument(
        document_id=row.document_id,
        pet_id=PetId(row.pet_id),
        owner_id=row.owner_id,
        domain_id=row.domain_id,
        title=row.title,
        media_type=DocumentMediaType(row.media_type),
        byte_size=row.byte_size,
        sha256=row.sha256,
        status=DocumentStatus(row.status),
        chunk_count=row.chunk_count,
        candidate_id=row.candidate_id,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _to_chunk(row: KnowledgeChunkRow) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=row.chunk_id,
        document_id=row.document_id,
        ordinal=row.ordinal,
        text=row.text,
        embedding=list(row.embedding) if row.embedding is not None else None,
    )


def _to_candidate(row: MemoryCandidateRow) -> MemoryCandidate:
    return MemoryCandidate(
        candidate_id=row.candidate_id,
        owner_id=row.owner_id,
        pet_id=PetId(row.pet_id),
        source=row.source,
        candidate_type=CandidateType(row.candidate_type),
        content=dict(row.content),
        status=CandidateStatus(row.status),
        created_at=row.created_at,
        updated_at=row.updated_at,
        confirmed_at=row.confirmed_at,
        committed_generation=row.committed_generation,
        domain_id=row.domain_id,
        relation=row.relation,
        related_candidate_id=row.related_candidate_id,
        fact_key=row.fact_key,
        embedding=list(row.embedding) if row.embedding is not None else None,
    )


class SqlKnowledgeRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_domain(self, domain: KnowledgeDomain) -> None:
        row = KnowledgeDomainRow(
            domain_id=domain.domain_id,
            pet_id=domain.pet_id.value,
            owner_id=domain.owner_id,
            device_slot=domain.device_slot,
            name=domain.name,
            description=domain.description,
            status=domain.status.value,
            origin=domain.origin.value,
            created_at=domain.created_at,
            updated_at=domain.updated_at,
            embedding=domain.embedding,
            pack_id=domain.pack_id,
            expert_brief=domain.expert_brief,
            safety_tier=domain.safety_tier,
        )
        self._session.add(row)
        await self._session.flush()

    async def get_domain(self, domain_id: uuid.UUID) -> KnowledgeDomain | None:
        row = await self._session.scalar(
            select(KnowledgeDomainRow).where(KnowledgeDomainRow.domain_id == domain_id)
        )
        return _to_domain(row) if row is not None else None

    async def list_domains(
        self, pet_id: PetId, include_deleted: bool = False
    ) -> list[KnowledgeDomain]:
        stmt = select(KnowledgeDomainRow).where(KnowledgeDomainRow.pet_id == pet_id.value)
        if not include_deleted:
            stmt = stmt.where(KnowledgeDomainRow.status != DomainStatus.DELETED.value)
        stmt = stmt.order_by(
            KnowledgeDomainRow.device_slot.nulls_last(), KnowledgeDomainRow.created_at
        )
        result = await self._session.scalars(stmt)
        return [_to_domain(r) for r in result.all()]

    async def count_non_deleted_domains(self, pet_id: PetId) -> int:
        stmt = (
            select(func.count())
            .select_from(KnowledgeDomainRow)
            .where(
                (KnowledgeDomainRow.pet_id == pet_id.value)
                & (KnowledgeDomainRow.status != DomainStatus.DELETED.value)
            )
        )
        count = await self._session.scalar(stmt)
        return count or 0

    async def get_domain_by_slot(self, pet_id: PetId, slot: int) -> KnowledgeDomain | None:
        stmt = select(KnowledgeDomainRow).where(
            (KnowledgeDomainRow.pet_id == pet_id.value)
            & (KnowledgeDomainRow.device_slot == slot)
            & (KnowledgeDomainRow.status != DomainStatus.DELETED.value)
        )
        row = await self._session.scalar(stmt)
        return _to_domain(row) if row is not None else None

    async def update_domain(self, domain: KnowledgeDomain) -> None:
        await self._session.execute(
            update(KnowledgeDomainRow)
            .where(KnowledgeDomainRow.domain_id == domain.domain_id)
            .values(
                device_slot=domain.device_slot,
                name=domain.name,
                description=domain.description,
                status=domain.status.value,
                origin=domain.origin.value,
                updated_at=domain.updated_at,
                embedding=domain.embedding,
                pack_id=domain.pack_id,
                expert_brief=domain.expert_brief,
                safety_tier=domain.safety_tier,
            )
        )
        await self._session.flush()

    async def save_suggestion(self, suggestion: KnowledgeSuggestion) -> None:
        row = KnowledgeSuggestionRow(
            suggestion_id=suggestion.suggestion_id,
            owner_id=suggestion.owner_id,
            pet_id=suggestion.pet_id.value,
            kind=suggestion.kind.value,
            proposed_name=suggestion.proposed_name,
            proposed_description=suggestion.proposed_description,
            member_candidate_ids=[str(cid) for cid in suggestion.member_candidate_ids],
            status=suggestion.status.value,
            created_at=suggestion.created_at,
            target_domain_id=suggestion.target_domain_id,
            decided_at=suggestion.decided_at,
        )
        self._session.add(row)
        await self._session.flush()

    async def get_suggestion(self, suggestion_id: uuid.UUID) -> KnowledgeSuggestion | None:
        row = await self._session.scalar(
            select(KnowledgeSuggestionRow).where(
                KnowledgeSuggestionRow.suggestion_id == suggestion_id
            )
        )
        return _to_suggestion(row) if row is not None else None

    async def get_pending_suggestion(self, pet_id: PetId) -> KnowledgeSuggestion | None:
        stmt = (
            select(KnowledgeSuggestionRow)
            .where(
                (KnowledgeSuggestionRow.pet_id == pet_id.value)
                & (KnowledgeSuggestionRow.status == SuggestionStatus.PENDING.value)
            )
            .order_by(KnowledgeSuggestionRow.created_at.desc())
            .limit(1)
        )
        row = await self._session.scalar(stmt)
        return _to_suggestion(row) if row is not None else None

    async def list_suggestions(
        self, pet_id: PetId, status: SuggestionStatus | None = None
    ) -> list[KnowledgeSuggestion]:
        stmt = select(KnowledgeSuggestionRow).where(KnowledgeSuggestionRow.pet_id == pet_id.value)
        if status is not None:
            stmt = stmt.where(KnowledgeSuggestionRow.status == status.value)
        stmt = stmt.order_by(KnowledgeSuggestionRow.created_at.desc())
        result = await self._session.scalars(stmt)
        return [_to_suggestion(r) for r in result.all()]

    async def update_suggestion(self, suggestion: KnowledgeSuggestion) -> None:
        await self._session.execute(
            update(KnowledgeSuggestionRow)
            .where(KnowledgeSuggestionRow.suggestion_id == suggestion.suggestion_id)
            .values(
                status=suggestion.status.value,
                decided_at=suggestion.decided_at,
                target_domain_id=suggestion.target_domain_id,
            )
        )
        await self._session.flush()

    async def save_summary(self, summary: KnowledgeSummary) -> None:
        existing = await self._session.scalar(
            select(KnowledgeSummaryRow).where(KnowledgeSummaryRow.domain_id == summary.domain_id)
        )
        if existing:
            await self._session.execute(
                update(KnowledgeSummaryRow)
                .where(KnowledgeSummaryRow.domain_id == summary.domain_id)
                .values(
                    text=summary.text,
                    source_candidate_ids=[str(cid) for cid in summary.source_candidate_ids],
                    generated_at=summary.generated_at,
                    stale=summary.stale,
                )
            )
        else:
            row = KnowledgeSummaryRow(
                domain_id=summary.domain_id,
                text=summary.text,
                source_candidate_ids=[str(cid) for cid in summary.source_candidate_ids],
                generated_at=summary.generated_at,
                stale=summary.stale,
            )
            self._session.add(row)
        await self._session.flush()

    async def get_summary(self, domain_id: uuid.UUID) -> KnowledgeSummary | None:
        row = await self._session.scalar(
            select(KnowledgeSummaryRow).where(KnowledgeSummaryRow.domain_id == domain_id)
        )
        return _to_summary(row) if row is not None else None

    async def save_document(
        self, document: KnowledgeDocument, chunks: list[KnowledgeChunk]
    ) -> None:
        doc_row = KnowledgeDocumentRow(
            document_id=document.document_id,
            pet_id=document.pet_id.value,
            owner_id=document.owner_id,
            domain_id=document.domain_id,
            title=document.title,
            media_type=document.media_type.value,
            byte_size=document.byte_size,
            sha256=document.sha256,
            status=document.status.value,
            chunk_count=document.chunk_count,
            candidate_id=document.candidate_id,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )
        self._session.add(doc_row)
        await self._session.flush()
        for chunk in chunks:
            chunk_row = KnowledgeChunkRow(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                ordinal=chunk.ordinal,
                text=chunk.text,
                embedding=chunk.embedding,
            )
            self._session.add(chunk_row)
        await self._session.flush()

    async def get_document(self, document_id: uuid.UUID) -> KnowledgeDocument | None:
        row = await self._session.scalar(
            select(KnowledgeDocumentRow).where(
                (KnowledgeDocumentRow.document_id == document_id)
                & (KnowledgeDocumentRow.status != DocumentStatus.DELETED.value)
            )
        )
        return _to_document(row) if row is not None else None

    async def get_document_by_sha256(self, pet_id: PetId, sha256: str) -> KnowledgeDocument | None:
        row = await self._session.scalar(
            select(KnowledgeDocumentRow).where(
                (KnowledgeDocumentRow.pet_id == pet_id.value)
                & (KnowledgeDocumentRow.sha256 == sha256)
                & (KnowledgeDocumentRow.status != DocumentStatus.DELETED.value)
            )
        )
        return _to_document(row) if row is not None else None

    async def list_documents(self, pet_id: PetId) -> list[KnowledgeDocument]:
        stmt = (
            select(KnowledgeDocumentRow)
            .where(
                (KnowledgeDocumentRow.pet_id == pet_id.value)
                & (KnowledgeDocumentRow.status != DocumentStatus.DELETED.value)
            )
            .order_by(KnowledgeDocumentRow.created_at.desc())
        )
        result = await self._session.scalars(stmt)
        return [_to_document(r) for r in result.all()]

    async def count_documents(self, pet_id: PetId) -> int:
        stmt = (
            select(func.count())
            .select_from(KnowledgeDocumentRow)
            .where(
                (KnowledgeDocumentRow.pet_id == pet_id.value)
                & (KnowledgeDocumentRow.status != DocumentStatus.DELETED.value)
            )
        )
        count = await self._session.scalar(stmt)
        return count or 0

    async def count_chunks_for_pet(self, pet_id: PetId) -> int:
        stmt = (
            select(func.count())
            .select_from(KnowledgeChunkRow)
            .join(
                KnowledgeDocumentRow,
                KnowledgeChunkRow.document_id == KnowledgeDocumentRow.document_id,
            )
            .where(
                (KnowledgeDocumentRow.pet_id == pet_id.value)
                & (KnowledgeDocumentRow.status != DocumentStatus.DELETED.value)
            )
        )
        count = await self._session.scalar(stmt)
        return count or 0

    async def delete_document(self, document_id: uuid.UUID) -> None:
        await self._session.execute(
            delete(KnowledgeChunkRow).where(KnowledgeChunkRow.document_id == document_id)
        )
        await self._session.execute(
            delete(KnowledgeDocumentRow).where(KnowledgeDocumentRow.document_id == document_id)
        )
        await self._session.flush()

    async def list_chunks_for_pet(
        self, pet_id: PetId, domain_id: uuid.UUID | None = None
    ) -> list[tuple[KnowledgeDocument, KnowledgeChunk]]:
        stmt = (
            select(KnowledgeDocumentRow, KnowledgeChunkRow)
            .join(
                KnowledgeChunkRow,
                KnowledgeChunkRow.document_id == KnowledgeDocumentRow.document_id,
            )
            .where(
                (KnowledgeDocumentRow.pet_id == pet_id.value)
                & (KnowledgeDocumentRow.status == DocumentStatus.READY.value)
            )
        )
        if domain_id is not None:
            stmt = stmt.where(
                (KnowledgeDocumentRow.domain_id == domain_id)
                | (KnowledgeDocumentRow.domain_id.is_(None))
            )
        else:
            stmt = stmt.where(KnowledgeDocumentRow.domain_id.is_(None))
        stmt = stmt.order_by(
            KnowledgeDocumentRow.created_at.desc(), KnowledgeChunkRow.ordinal.asc()
        )
        result = await self._session.execute(stmt)
        return [(_to_document(doc_r), _to_chunk(chk_r)) for doc_r, chk_r in result.all()]


class SqlKnowledgeCandidateAdapter:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_candidate(self, candidate: MemoryCandidate) -> None:
        existing = await self._session.scalar(
            select(MemoryCandidateRow).where(
                MemoryCandidateRow.candidate_id == candidate.candidate_id
            )
        )
        if existing:
            await self._session.execute(
                update(MemoryCandidateRow)
                .where(MemoryCandidateRow.candidate_id == candidate.candidate_id)
                .values(
                    status=candidate.status.value,
                    content=candidate.content,
                    updated_at=candidate.updated_at,
                    confirmed_at=candidate.confirmed_at,
                    committed_generation=candidate.committed_generation,
                    domain_id=candidate.domain_id,
                    relation=candidate.relation,
                    related_candidate_id=candidate.related_candidate_id,
                    fact_key=candidate.fact_key,
                    embedding=candidate.embedding,
                )
            )
        else:
            self._session.add(
                MemoryCandidateRow(
                    candidate_id=candidate.candidate_id,
                    owner_id=candidate.owner_id,
                    pet_id=candidate.pet_id.value,
                    source=candidate.source,
                    candidate_type=candidate.candidate_type.value,
                    content=candidate.content,
                    status=candidate.status.value,
                    created_at=candidate.created_at,
                    updated_at=candidate.updated_at,
                    confirmed_at=candidate.confirmed_at,
                    committed_generation=candidate.committed_generation,
                    domain_id=candidate.domain_id,
                    relation=candidate.relation,
                    related_candidate_id=candidate.related_candidate_id,
                    fact_key=candidate.fact_key,
                    embedding=candidate.embedding,
                )
            )
        await self._session.flush()

    async def get_candidate(self, candidate_id: uuid.UUID) -> MemoryCandidate | None:
        row = await self._session.scalar(
            select(MemoryCandidateRow).where(MemoryCandidateRow.candidate_id == candidate_id)
        )
        return _to_candidate(row) if row is not None else None

    async def list_candidates_by_domain(
        self, domain_id: uuid.UUID, confirmed_only: bool = True
    ) -> list[MemoryCandidate]:
        stmt = select(MemoryCandidateRow).where(MemoryCandidateRow.domain_id == domain_id)
        if confirmed_only:
            stmt = stmt.where(
                MemoryCandidateRow.status.in_(
                    [
                        CandidateStatus.CONFIRMED.value,
                        CandidateStatus.AWAITING_DEVICE_COMMIT.value,
                    ]
                )
            )
        stmt = stmt.order_by(MemoryCandidateRow.created_at.desc())
        result = await self._session.scalars(stmt)
        return [_to_candidate(r) for r in result.all()]

    async def list_unsorted_knowledge(
        self, pet_id: PetId, confirmed_only: bool = True
    ) -> list[MemoryCandidate]:
        stmt = select(MemoryCandidateRow).where(
            (MemoryCandidateRow.pet_id == pet_id.value)
            & (MemoryCandidateRow.candidate_type == CandidateType.KNOWLEDGE.value)
            & (MemoryCandidateRow.domain_id.is_(None))
        )
        if confirmed_only:
            stmt = stmt.where(
                MemoryCandidateRow.status.in_(
                    [
                        CandidateStatus.CONFIRMED.value,
                        CandidateStatus.AWAITING_DEVICE_COMMIT.value,
                    ]
                )
            )
        stmt = stmt.order_by(MemoryCandidateRow.created_at.desc())
        result = await self._session.scalars(stmt)
        return [_to_candidate(r) for r in result.all()]

    async def update_domain_for_candidates(
        self, candidate_ids: list[uuid.UUID], new_domain_id: uuid.UUID | None
    ) -> None:
        if not candidate_ids:
            return
        await self._session.execute(
            update(MemoryCandidateRow)
            .where(MemoryCandidateRow.candidate_id.in_(candidate_ids))
            .values(domain_id=new_domain_id)
        )
        await self._session.flush()
