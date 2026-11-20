"""`Idempotency-Key` storage for mutating requests (contract draft §1).

Same owner + key + request hash → the stored response is replayed. Same key
with a different body → `409 idempotency_key_reused`. Keys expire after 24 h.
The stored response body must be the API response, which never contains
secrets or conversation text.
"""

import hashlib
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from http import HTTPStatus
from typing import Any

from sqlalchemy import DateTime, Integer, String, delete, event, select
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from xuna_backend.core.db import Base
from xuna_backend.core.errors import AppError

TTL = timedelta(hours=24)
_SWEEP_INTERVAL_S = 60.0
_last_sweep_monotonic: float = time.monotonic()


class IdempotencyRecord(Base):
    __tablename__ = "idempotency_keys"

    owner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    key: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    request_hash: Mapped[str] = mapped_column(String(64))
    status_code: Mapped[int] = mapped_column(Integer)
    body: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


@dataclass(frozen=True, slots=True)
class StoredResponse:
    status: HTTPStatus
    body: dict[str, Any]


def parse_key(raw: str | None) -> uuid.UUID:
    try:
        return uuid.UUID(raw or "")
    except ValueError as error:
        raise AppError("idempotency_key_required", HTTPStatus.BAD_REQUEST) from error


def request_hash(route: str, body: dict[str, Any]) -> str:
    canonical = json.dumps([route, body], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


async def run_once(
    session: AsyncSession,
    *,
    owner_id: uuid.UUID,
    key: uuid.UUID,
    fingerprint: str,
    now: datetime,
    action: Callable[[], Awaitable[StoredResponse]],
) -> StoredResponse:
    """Runs `action` at most once per (owner, key); caller commits the session."""
    global _last_sweep_monotonic
    cutoff = now - TTL
    existing = await session.scalar(
        select(IdempotencyRecord)
        .where(IdempotencyRecord.owner_id == owner_id, IdempotencyRecord.key == key)
        .with_for_update()
    )
    if existing is not None:
        if existing.created_at < cutoff:
            await session.delete(existing)
            await session.flush()
        else:
            if existing.request_hash != fingerprint:
                raise AppError("idempotency_key_reused", HTTPStatus.CONFLICT)
            return StoredResponse(HTTPStatus(existing.status_code), existing.body)
    response = await action()
    session.add(
        IdempotencyRecord(
            owner_id=owner_id,
            key=key,
            request_hash=fingerprint,
            status_code=response.status.value,
            body=response.body,
            created_at=now,
        )
    )
    mono = time.monotonic()
    if mono - _last_sweep_monotonic >= _SWEEP_INTERVAL_S:
        prev_mono = _last_sweep_monotonic
        await session.execute(
            delete(IdempotencyRecord).where(IdempotencyRecord.created_at < cutoff)
        )
        _last_sweep_monotonic = mono
        sync_session = getattr(session, "sync_session", None)
        if sync_session is not None:

            def _on_commit(_sess: Any) -> None:
                if event.contains(sync_session, "after_rollback", _on_rollback):
                    event.remove(sync_session, "after_rollback", _on_rollback)

            def _on_rollback(_sess: Any) -> None:
                global _last_sweep_monotonic
                if _last_sweep_monotonic == mono:
                    _last_sweep_monotonic = prev_mono
                if event.contains(sync_session, "after_commit", _on_commit):
                    event.remove(sync_session, "after_commit", _on_commit)

            event.listen(sync_session, "after_commit", _on_commit, once=True)
            event.listen(sync_session, "after_rollback", _on_rollback, once=True)
    return response
