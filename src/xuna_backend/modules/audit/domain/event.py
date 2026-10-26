"""Append-only security/audit event (CLD-12, D23).

Metadata is deliberately tiny and typed: short identifiers, codes, numbers
and booleans only, so conversation text or personal facts cannot be logged
here by accident.
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

MetadataValue = str | int | bool
_KEY = re.compile(r"^[a-z][a-zA-Z0-9]{0,31}$")
_VALUE = re.compile(r"^[A-Za-z0-9_.:-]{0,64}$")
MAX_METADATA = 8


class AuditAction(StrEnum):
    PET_BOUND = "pet.bound"
    PET_BIND_REJECTED = "pet.bind_rejected"
    CONSENT_CHANGED = "consent.changed"
    PROJECTION_INGESTED = "projection.ingested"
    PROJECTION_REJECTED = "projection.rejected"
    MEMORY_CANDIDATE_CONFIRMED = "memory.candidate_confirmed"
    MEMORY_CANDIDATE_REJECTED = "memory.candidate_rejected"
    BACKUP_CREATED = "backup.created"
    BACKUP_RESTORED = "backup.restored"
    TOMBSTONE_CREATED = "tombstone.created"
    PET_FORGOTTEN = "pet.forgotten"
    SUGGESTION_CONSOLIDATED = "suggestion.consolidated"
    SUGGESTION_APPLIED = "suggestion.applied"
    SUGGESTION_REJECTED = "suggestion.rejected"
    SUGGESTION_REVERTED = "suggestion.reverted"


@dataclass(frozen=True, slots=True)
class AuditEvent:
    owner_id: uuid.UUID
    action: AuditAction
    occurred_at: datetime
    subject: str = ""  # e.g. a PetId; never free text
    request_id: str = ""
    metadata: dict[str, MetadataValue] = field(default_factory=dict)
    event_id: uuid.UUID = field(default_factory=uuid.uuid4)

    def __post_init__(self) -> None:
        if self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        for text in (self.subject, self.request_id):
            if not _VALUE.match(text):
                raise ValueError("subject/request_id must be a short identifier")
        if len(self.metadata) > MAX_METADATA:
            raise ValueError("too many metadata entries")
        for key, value in self.metadata.items():
            if not _KEY.match(key):
                raise ValueError(f"bad metadata key {key!r}")
            if isinstance(value, str) and not _VALUE.match(value):
                raise ValueError(f"metadata {key!r} must be a short code, not free text")
