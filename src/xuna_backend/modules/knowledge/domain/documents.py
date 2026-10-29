"""Domain entities and chunker for Owner Knowledge Documents (KX-5, plan §6.10, ADR-0011).

Domain rules: stdlib and own/other domain only. No pydantic or frameworks.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from xuna_backend.modules.identity.domain.pet_id import PetId

MAX_DOCUMENTS_PER_PET = 20
MAX_DOCUMENT_BYTES = 2 * 1024 * 1024  # 2 MB
MAX_CHUNKS_PER_DOCUMENT = 400
MAX_CHUNKS_PER_PET = 2000
MAX_CHUNK_CHARS = 1200
TARGET_CHUNK_CHARS = 800
CHUNK_OVERLAP_CHARS = 100
MAX_DOCUMENT_TITLE_CHARS = 120


class DocumentMediaType(StrEnum):
    MARKDOWN = "text/markdown"
    PLAIN = "text/plain"
    PDF = "application/pdf"


class DocumentStatus(StrEnum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    DELETED = "deleted"


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    document_id: uuid.UUID
    pet_id: PetId
    owner_id: uuid.UUID
    domain_id: uuid.UUID | None
    title: str
    media_type: DocumentMediaType
    byte_size: int
    sha256: str
    status: DocumentStatus
    chunk_count: int
    candidate_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        if (
            not self.title
            or len(self.title.strip()) == 0
            or len(self.title) > MAX_DOCUMENT_TITLE_CHARS
        ):
            raise ValueError("title must be 1 to 120 characters")
        if not 1 <= self.byte_size <= MAX_DOCUMENT_BYTES:
            raise ValueError(f"byte_size must be 1..{MAX_DOCUMENT_BYTES}")
        if len(self.sha256) != 64:
            raise ValueError("sha256 must be 64 hex characters")
        if self.chunk_count < 0 or self.chunk_count > MAX_CHUNKS_PER_DOCUMENT:
            raise ValueError(f"chunk_count must be 0..{MAX_CHUNKS_PER_DOCUMENT}")
        if self.created_at.tzinfo is None or self.updated_at.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    ordinal: int
    text: str
    embedding: list[float] | None = None

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("ordinal must be >= 0")
        if not self.text or len(self.text.strip()) == 0 or len(self.text) > MAX_CHUNK_CHARS:
            raise ValueError(f"chunk text must be 1..{MAX_CHUNK_CHARS} characters")


_HEADING_RE = re.compile(r"(?m)^(?=#{1,6}\s+\S)")


def _split_long_block(
    block: str,
    *,
    target_chars: int = TARGET_CHUNK_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[str]:
    """Splits a text block around target_chars with overlap_chars, capped at max_chars."""
    cleaned = block.strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars and len(cleaned) <= target_chars + overlap_chars:
        return [cleaned]

    pieces: list[str] = []
    step = max(1, target_chars - overlap_chars)
    start = 0
    n = len(cleaned)
    while start < n:
        end = min(n, start + target_chars)
        if end < n:
            # Prefer breaking at newline or sentence/word boundary near target_chars
            window = cleaned[start : min(n, start + max_chars)]
            nl_pos = window.rfind("\n", target_chars // 2, target_chars)
            space_pos = window.rfind(" ", target_chars // 2, target_chars)
            if nl_pos != -1:
                end = start + nl_pos
            elif space_pos != -1:
                end = start + space_pos
        piece = cleaned[start:end].strip()
        if piece:
            if len(piece) > max_chars:
                piece = piece[:max_chars].rstrip()
            pieces.append(piece)
        if end >= n:
            break
        next_start = end - overlap_chars
        if next_start <= start:
            next_start = start + step
        start = next_start
    return pieces


def chunk_document_text(
    text: str,
    *,
    target_chars: int = TARGET_CHUNK_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[str]:
    """Splits document text by Markdown headings and blank lines into chunks (plan §6.10).

    Each chunk is `<= max_chars` (1200 chars), targeting `~target_chars` (800 chars)
    with `overlap_chars` (100 chars) when splitting long sections.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    raw_sections = [s.strip() for s in _HEADING_RE.split(normalized) if s.strip()]
    chunks: list[str] = []

    for section in raw_sections:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section) if p.strip()]
        current = ""
        for para in paragraphs:
            if not current:
                if len(para) <= target_chars:
                    current = para
                else:
                    chunks.extend(
                        _split_long_block(
                            para,
                            target_chars=target_chars,
                            overlap_chars=overlap_chars,
                            max_chars=max_chars,
                        )
                    )
            elif len(current) + 2 + len(para) <= target_chars:
                current = f"{current}\n\n{para}"
            else:
                chunks.extend(
                    _split_long_block(
                        current,
                        target_chars=target_chars,
                        overlap_chars=overlap_chars,
                        max_chars=max_chars,
                    )
                )
                if len(para) <= target_chars:
                    current = para
                else:
                    chunks.extend(
                        _split_long_block(
                            para,
                            target_chars=target_chars,
                            overlap_chars=overlap_chars,
                            max_chars=max_chars,
                        )
                    )
                    current = ""
        if current:
            chunks.extend(
                _split_long_block(
                    current,
                    target_chars=target_chars,
                    overlap_chars=overlap_chars,
                    max_chars=max_chars,
                )
            )

    return chunks


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Computes cosine similarity in [0.0, 1.0] between two embedding vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))
