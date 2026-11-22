"""HTTP routes for `/v1/actions/habits` and `/v1/actions/focus` (Spatial Mesh §5.6.3-§5.6.4)."""

import contextlib
import inspect
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from http import HTTPStatus
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Query, Request
from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field

from xuna_backend.core.errors import AppError
from xuna_backend.modules.accounts.api.dependencies import CurrentOwner, Session, state
from xuna_backend.modules.actions.application.ports import HabitRepository
from xuna_backend.modules.actions.domain.habit_learner import explain_habit, habit_score
from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.audit.domain.event import AuditAction, AuditEvent
from xuna_backend.modules.hosts.application.ports import HostRepository, HostSessionHub
from xuna_backend.modules.hosts.domain.entities import (
    Host,
    HostId,
    HostStatus,
    validate_non_zero_uuid,
)
from xuna_backend.modules.hosts.domain.twin import HostFocusChangedNotice


def _require_int_schema_version(v: object) -> object:
    if not isinstance(v, int) or isinstance(v, bool):
        raise ValueError("schemaVersion must be an integer")
    return v


StrictSchemaVersion = Annotated[Literal[1], BeforeValidator(_require_int_schema_version)]
NonZeroHostUuid = Annotated[uuid.UUID, AfterValidator(validate_non_zero_uuid)]

router = APIRouter(prefix="/v1/actions", tags=["actions"])


class SetHostFocusBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schemaVersion: StrictSchemaVersion = 1
    hostId: NonZeroHostUuid | None = None
    followMe: bool | None = None
    petId: str | None = Field(default=None, min_length=1, max_length=64)


def _request_id(request: Request) -> str:
    rid = getattr(request.state, "request_id", "")
    return str(rid) if isinstance(rid, str) else ""


@router.get("/habits")
async def list_routing_habits(
    request: Request,
    owner: CurrentOwner,
    session: Session,
    petId: str | None = Query(default=None, min_length=1, max_length=64),
) -> dict[str, Any]:
    """Lists learned host routing habits with human-readable explanations (§5.6.3)."""
    habit_factory: Callable[[object], HabitRepository] = state(request, "habit_repository")
    host_factory: Callable[[object], HostRepository] = state(request, "host_repository")
    habits = await habit_factory(session).list_habits(owner.owner_id, pet_id=petId)
    hosts_by_id = {h.host_id: h for h in await host_factory(session).list_hosts(owner.owner_id)}
    items: list[dict[str, Any]] = []
    for habit in habits:
        host = hosts_by_id.get(habit.host_id)
        items.append(
            {
                "habitId": str(habit.habit_id),
                "petId": habit.pet_id,
                "contextKey": habit.context_key,
                "hostId": str(habit.host_id),
                "hostAlias": host.alias.value if host is not None else None,
                "hostPlatform": host.platform.value if host is not None else None,
                "positiveScore": round(habit.positive_score, 4),
                "negativeScore": round(habit.negative_score, 4),
                "netScore": round(habit_score(habit), 4),
                "sampleCount": habit.sample_count,
                "gamma": habit.decay_gamma,
                "lastReasonCode": habit.last_reason_code,
                "why": explain_habit(habit),
                "updatedAt": habit.updated_at.astimezone(UTC).isoformat(),
            }
        )
    return {"items": items}


@router.delete("/habits/{habit_id}")
async def delete_routing_habit(
    habit_id: str,
    request: Request,
    owner: CurrentOwner,
    session: Session,
) -> dict[str, Any]:
    """Deletes a single learned habit owned by the caller (§5.6.3)."""
    try:
        parsed_id = uuid.UUID(habit_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error

    habit_factory: Callable[[object], HabitRepository] = state(request, "habit_repository")
    deleted = await habit_factory(session).delete_habit(owner.owner_id, parsed_id)
    if not deleted:
        raise AppError("not_found", HTTPStatus.NOT_FOUND)

    audit_factory: Callable[[object], AuditLog] = state(request, "audit_log")
    await audit_factory(session).record(
        AuditEvent(
            owner_id=owner.owner_id,
            action=AuditAction.HOST_HABIT_DELETED,
            occurred_at=datetime.now(UTC),
            subject=str(parsed_id),
            request_id=_request_id(request),
            metadata={"habitId": str(parsed_id)},
        )
    )
    await session.commit()
    return {"habitId": str(parsed_id), "deleted": True}


@router.delete("/habits")
async def reset_routing_habits(
    request: Request,
    owner: CurrentOwner,
    session: Session,
    petId: str | None = Query(default=None, min_length=1, max_length=64),
) -> dict[str, Any]:
    """Resets all learned habits for the owner (optionally filtered by `petId`, §5.6.3)."""
    habit_factory: Callable[[object], HabitRepository] = state(request, "habit_repository")
    deleted_count = await habit_factory(session).delete_all_habits(owner.owner_id, pet_id=petId)

    audit_factory: Callable[[object], AuditLog] = state(request, "audit_log")
    await audit_factory(session).record(
        AuditEvent(
            owner_id=owner.owner_id,
            action=AuditAction.HOST_HABIT_DELETED,
            occurred_at=datetime.now(UTC),
            subject="all_habits",
            request_id=_request_id(request),
            metadata={"deletedCount": deleted_count},
        )
    )
    await session.commit()
    return {"deletedCount": deleted_count}


@router.post("/focus")
async def set_host_focus(
    body: SetHostFocusBody,
    request: Request,
    owner: CurrentOwner,
    session: Session,
) -> dict[str, Any]:
    """Pins or clears the owner-scoped live host focus and Follow Me mode (§5.6.4)."""
    host: Host | None = None
    target_host_id: HostId | None = None
    if body.hostId is not None:
        try:
            target_host_id = HostId(body.hostId)
        except ValueError as error:
            raise AppError(
                "validation_failed", HTTPStatus.UNPROCESSABLE_ENTITY, "hostId"
            ) from error
        host_factory: Callable[[object], HostRepository] = state(request, "host_repository")
        host = await host_factory(session).get_host(target_host_id)
        if host is None or host.owner_id != owner.owner_id or host.status != HostStatus.ACTIVE:
            raise AppError("not_found", HTTPStatus.NOT_FOUND)

    now = datetime.now(UTC)
    hub: HostSessionHub = state(request, "host_session_hub")
    focus_state = hub.set_focus_state(
        owner.owner_id,
        focused_host_id=target_host_id,
        keep_focused_host=False,
        follow_me=body.followMe,
        source="app",
        now=now,
    )
    notice = HostFocusChangedNotice(
        owner_id=owner.owner_id,
        host_id=host.host_id if host is not None else None,
        alias=host.alias.value if host is not None else None,
        platform=host.platform.value if host is not None else None,
        follow_me=focus_state.follow_me,
        source="app",
        updated_at=focus_state.updated_at,
    )
    await hub.publish_focus_changed(notice)
    on_focus = getattr(request.app.state, "on_host_focus_changed", None)
    if callable(on_focus):
        with contextlib.suppress(Exception):
            maybe = on_focus(notice)
            if inspect.isawaitable(maybe):
                await maybe

    return {
        "focusedHostId": (
            str(focus_state.focused_host_id) if focus_state.focused_host_id is not None else None
        ),
        "focusedAlias": host.alias.value if host is not None else None,
        "followMe": focus_state.follow_me,
        "source": focus_state.source,
        "updatedAt": focus_state.updated_at.astimezone(UTC).isoformat(),
    }
