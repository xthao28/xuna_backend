import uuid
from typing import Protocol

from xuna_backend.modules.audit.domain.event import AuditEvent


class AuditLog(Protocol):
    async def record(self, event: AuditEvent) -> None:
        """Stored in the caller's transaction: audited iff the change commits."""
        ...

    async def recent(self, owner_id: uuid.UUID, limit: int) -> list[AuditEvent]: ...
