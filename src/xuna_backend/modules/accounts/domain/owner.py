"""Owner account. The identity provider owns credentials; we keep only the
stable `(issuer, subject)` pair and our own `owner_id` (`UserId`)."""

import uuid
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OwnerClaims:
    issuer: str
    subject: str

    def __post_init__(self) -> None:
        if not self.issuer or not self.subject or len(self.subject) > 255:
            raise ValueError("issuer and subject are required")


@dataclass(frozen=True, slots=True)
class Owner:
    owner_id: uuid.UUID
    issuer: str
    subject: str
