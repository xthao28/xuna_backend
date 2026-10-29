"""Knowledge document upload, listing and deletion use cases (KX-5, plan §6.10, ADR-0011).

Application rules: stdlib, domain, application only. Raw file bytes are never stored or logged.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
import uuid
import zlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from xuna_backend.modules.cognition.domain.types import EmbeddingRequest
from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.knowledge.application.schemas import (
    KnowledgeDocumentResponse,
    UploadDocumentRequest,
)
from xuna_backend.modules.knowledge.domain.documents import (
    MAX_CHUNKS_PER_DOCUMENT,
    MAX_CHUNKS_PER_PET,
    MAX_DOCUMENT_BYTES,
    MAX_DOCUMENTS_PER_PET,
    DocumentMediaType,
    DocumentStatus,
    KnowledgeChunk,
    KnowledgeDocument,
    chunk_document_text,
)
from xuna_backend.modules.knowledge.domain.entities import (
    DomainStatus,
    RelationKind,
    compute_fact_key,
)
from xuna_backend.modules.memory.domain.entities import (
    CandidateStatus,
    CandidateType,
    MemoryCandidate,
)

if TYPE_CHECKING:
    from xuna_backend.modules.cognition.application.ports import Embedder
    from xuna_backend.modules.knowledge.application.ports import (
        KnowledgeCandidatePort,
        KnowledgeRepository,
    )

_PDF_STREAM_RE = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.DOTALL)
_PDF_PAREN_TEXT_RE = re.compile(rb"\(([^()\\]*(?:\\.[^()\\]*)*)\)")


def _extract_pdf_text_stdlib(raw_bytes: bytes) -> str:
    """Extracts readable text from simple PDF streams using stdlib only."""
    pieces: list[str] = []
    # Check both raw and FlateDecode-decompressed streams
    streams = _PDF_STREAM_RE.findall(raw_bytes)
    buffers = [raw_bytes]
    for s in streams:
        try:
            buffers.append(zlib.decompress(s))
        except Exception:
            buffers.append(s)
    for buf in buffers:
        for match in _PDF_PAREN_TEXT_RE.finditer(buf):
            raw_token = match.group(1)
            token = (
                raw_token.replace(rb"\(", b"(")
                .replace(rb"\)", b")")
                .replace(rb"\\", b"\\")
                .replace(rb"\n", b"\n")
                .replace(rb"\r", b"\r")
                .decode("utf-8", errors="ignore")
                .strip()
            )
            if token:
                pieces.append(token)
    return "\n".join(pieces).strip()


def _extract_text(raw_bytes: bytes, media_type: DocumentMediaType) -> str:
    """Extracts UTF-8 text from raw bytes without retaining the raw file."""
    if media_type in (DocumentMediaType.MARKDOWN, DocumentMediaType.PLAIN):
        try:
            return raw_bytes.decode("utf-8")
        except UnicodeDecodeError as err:
            raise ValueError("Document must be valid UTF-8 text") from err
    if media_type == DocumentMediaType.PDF:
        extracted = _extract_pdf_text_stdlib(raw_bytes)
        if extracted:
            return extracted
        try:
            text = raw_bytes.decode("utf-8").strip()
            if text and not text.startswith("%PDF"):
                return text
        except UnicodeDecodeError:
            pass
        raise ValueError("Unable to extract text from PDF")
    raise ValueError(f"Unsupported mediaType: {media_type}")


def _to_response(doc: KnowledgeDocument, fact_key: int | None = None) -> KnowledgeDocumentResponse:
    computed_fk = fact_key if fact_key is not None else compute_fact_key(f"doc:{doc.sha256}")
    return KnowledgeDocumentResponse(
        document_id=doc.document_id,
        pet_id=str(doc.pet_id),
        domain_id=doc.domain_id,
        title=doc.title,
        media_type=doc.media_type.value,
        byte_size=doc.byte_size,
        sha256=doc.sha256,
        status=doc.status.value,
        chunk_count=doc.chunk_count,
        candidate_id=doc.candidate_id,
        fact_key=computed_fk,
        created_at=doc.created_at.astimezone(UTC).isoformat(),
        updated_at=doc.updated_at.astimezone(UTC).isoformat(),
    )


class UploadKnowledgeDocumentUseCase:
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
        request: UploadDocumentRequest,
        now: datetime | None = None,
    ) -> KnowledgeDocumentResponse:
        current_now = now or datetime.now(UTC)
        title = request.title.strip()
        if not title:
            raise ValueError("Document title must not be empty")

        try:
            raw_bytes = base64.b64decode(request.data, validate=True)
        except (binascii.Error, ValueError) as err:
            raise ValueError("Invalid base64 document data") from err

        byte_size = len(raw_bytes)
        if byte_size == 0:
            raise ValueError("Document data must not be empty")
        if byte_size > MAX_DOCUMENT_BYTES:
            raise ValueError("Document exceeds maximum size of 2 MB")

        if request.domain_id is not None:
            domain = await self._repo.get_domain(request.domain_id)
            if domain is None or domain.pet_id != pet_id or domain.status == DomainStatus.DELETED:
                raise KeyError("Domain not found")

        sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
        existing = await self._repo.get_document_by_sha256(pet_id, sha256_hex)
        if existing is not None:
            raise ValueError("Duplicate document sha256 for this pet")

        doc_count = await self._repo.count_documents(pet_id)
        if doc_count >= MAX_DOCUMENTS_PER_PET:
            raise ValueError(
                f"Document limit reached: maximum {MAX_DOCUMENTS_PER_PET} documents per pet"
            )

        media_type = DocumentMediaType(request.media_type)
        text = _extract_text(raw_bytes, media_type)
        # Drop raw_bytes reference immediately after text extraction
        del raw_bytes

        chunk_texts = chunk_document_text(text)
        if not chunk_texts:
            raise ValueError("Document contains no extractable text")
        if len(chunk_texts) > MAX_CHUNKS_PER_DOCUMENT:
            raise ValueError(
                f"Document exceeds maximum {MAX_CHUNKS_PER_DOCUMENT} chunks per document"
            )

        existing_chunks = await self._repo.count_chunks_for_pet(pet_id)
        if existing_chunks + len(chunk_texts) > MAX_CHUNKS_PER_PET:
            raise ValueError(f"Chunk limit reached: maximum {MAX_CHUNKS_PER_PET} chunks per pet")

        embeddings: list[list[float] | None] = [None] * len(chunk_texts)
        if self._embedder is not None:
            try:
                emb_res = await self._embedder.embed(EmbeddingRequest(texts=tuple(chunk_texts)))
                if len(emb_res.embeddings) == len(chunk_texts):
                    embeddings = [list(vec) for vec in emb_res.embeddings]
            except Exception:  # noqa: S110
                # Embedding is optional; lexical fallback remains functional
                pass

        doc_id = uuid.uuid4()
        candidate_id = uuid.uuid4()
        fact_key = compute_fact_key(f"doc:{sha256_hex}")

        candidate = MemoryCandidate(
            candidate_id=candidate_id,
            owner_id=owner_id,
            pet_id=pet_id,
            source="owner_document",
            candidate_type=CandidateType.KNOWLEDGE,
            content={
                "text": title,
                "summary": title,
                "kind": "document",
                "documentId": str(doc_id),
            },
            status=CandidateStatus.AWAITING_OWNER_CONFIRMATION,
            created_at=current_now,
            updated_at=current_now,
            domain_id=request.domain_id,
            relation=RelationKind.NEW.value,
            fact_key=fact_key,
        )
        await self._candidates.save_candidate(candidate)

        document = KnowledgeDocument(
            document_id=doc_id,
            pet_id=pet_id,
            owner_id=owner_id,
            domain_id=request.domain_id,
            title=title,
            media_type=media_type,
            byte_size=byte_size,
            sha256=sha256_hex,
            status=DocumentStatus.READY,
            chunk_count=len(chunk_texts),
            candidate_id=candidate_id,
            created_at=current_now,
            updated_at=current_now,
        )
        chunks = [
            KnowledgeChunk(
                chunk_id=uuid.uuid4(),
                document_id=doc_id,
                ordinal=idx,
                text=c_text,
                embedding=embeddings[idx],
            )
            for idx, c_text in enumerate(chunk_texts)
        ]
        await self._repo.save_document(document, chunks)
        return _to_response(document, fact_key=fact_key)


class ListKnowledgeDocumentsUseCase:
    def __init__(self, repo: KnowledgeRepository) -> None:
        self._repo = repo

    async def execute(self, pet_id: PetId) -> list[KnowledgeDocumentResponse]:
        docs = await self._repo.list_documents(pet_id)
        return [_to_response(d) for d in docs]


class DeleteKnowledgeDocumentUseCase:
    def __init__(self, repo: KnowledgeRepository, candidate_port: KnowledgeCandidatePort) -> None:
        self._repo = repo
        self._candidates = candidate_port

    async def execute(
        self,
        document_id: uuid.UUID,
        owner_id: uuid.UUID,
        now: datetime | None = None,
    ) -> None:
        current_now = now or datetime.now(UTC)
        doc = await self._repo.get_document(document_id)
        if doc is None or doc.owner_id != owner_id:
            raise KeyError("Document not found")

        if doc.candidate_id is not None:
            cand = await self._candidates.get_candidate(doc.candidate_id)
            if cand is not None and cand.status != CandidateStatus.REJECTED:
                rejected = MemoryCandidate(
                    candidate_id=cand.candidate_id,
                    owner_id=cand.owner_id,
                    pet_id=cand.pet_id,
                    source=cand.source,
                    candidate_type=cand.candidate_type,
                    content=cand.content,
                    status=CandidateStatus.REJECTED,
                    created_at=cand.created_at,
                    updated_at=current_now,
                    confirmed_at=cand.confirmed_at,
                    committed_generation=cand.committed_generation,
                    domain_id=cand.domain_id,
                    relation=cand.relation,
                    related_candidate_id=cand.related_candidate_id,
                    fact_key=cand.fact_key,
                    embedding=cand.embedding,
                )
                await self._candidates.save_candidate(rejected)

        await self._repo.delete_document(document_id)
