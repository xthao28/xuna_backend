"""Owners can read their own audit trail (transparency, D23)."""

from collections.abc import Callable
from datetime import UTC
from typing import Any

from fastapi import APIRouter, Query, Request

from xuna_backend.modules.accounts.api.dependencies import CurrentOwner, Session, state
from xuna_backend.modules.audit.application.ports import AuditLog

router = APIRouter(prefix="/v1", tags=["audit"])


@router.get("/audit-events")
async def audit_events(
    request: Request,
    owner: CurrentOwner,
    session: Session,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    log_for: Callable[[object], AuditLog] = state(request, "audit_log")
    events = await log_for(session).recent(owner.owner_id, limit)
    return {
        "items": [
            {
                "eventId": str(e.event_id),
                "action": e.action.value,
                "subject": e.subject,
                "metadata": e.metadata,
                "occurredAt": e.occurred_at.astimezone(UTC).isoformat(),
            }
            for e in events
        ]
    }
