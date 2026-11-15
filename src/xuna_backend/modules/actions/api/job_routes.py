"""HTTP API routes for autonomous jobs `/v1/actions/jobs` (Wave D M1).

Allows clients to create, observe, pause, resume, steer, and cancel autonomous tasks.
"""

import logging
import uuid
from http import HTTPStatus
from typing import Any, Literal

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from xuna_backend.core.errors import AppError
from xuna_backend.modules.accounts.api.dependencies import CurrentOwner
from xuna_backend.modules.actions.application.job_runner import AutonomousJobRunner
from xuna_backend.modules.actions.domain.job import (
    AutonomousJob,
    AutonomousJobState,
    SubGoal,
)
from xuna_backend.modules.hosts.domain.entities import HostId

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/actions/jobs", tags=["autonomous_jobs"])


class CreateAutonomousJobBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schemaVersion: Literal[1] = 1
    petId: str = Field(default="xuna_pet", min_length=1, max_length=64)
    goal: str = Field(min_length=1, max_length=2048)
    title: str | None = Field(default=None, max_length=128)
    hostId: uuid.UUID | None = None
    sessionId: str | None = Field(default=None, max_length=64)
    maxSteps: int = Field(default=50, ge=1, le=100)
    subGoals: list[dict[str, Any]] | None = None


class ResumeJobBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schemaVersion: Literal[1] = 1
    approvalId: uuid.UUID | None = None


class SteerJobBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schemaVersion: Literal[1] = 1
    instruction: str = Field(min_length=1, max_length=2048)


class CancelJobBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schemaVersion: Literal[1] = 1
    reason: str = Field(default="user_cancelled", max_length=64)


def _runner(request: Request) -> AutonomousJobRunner:
    runner: AutonomousJobRunner | None = getattr(request.app.state, "job_runner", None)
    if runner is None:
        raise AppError(
            "service_unavailable",
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Job runner not initialized",
        )
    return runner


async def _get_owner_job(
    runner: AutonomousJobRunner, job_id: uuid.UUID, owner_id: uuid.UUID
) -> AutonomousJob:
    job = await runner._store.get(job_id)
    if job is None or job.owner_id != owner_id:
        raise AppError("not_found", HTTPStatus.NOT_FOUND, f"Job {job_id} not found")
    return job


@router.post("", status_code=HTTPStatus.CREATED)
async def create_job(
    body: CreateAutonomousJobBody,
    request: Request,
    owner: CurrentOwner,
) -> dict[str, Any]:
    """Creates and starts an autonomous multi-step job."""
    runner = _runner(request)
    job_id = uuid.uuid4()
    host_id = HostId(body.hostId) if body.hostId is not None else None

    sub_goals: tuple[SubGoal, ...] = ()
    if body.subGoals:
        sub_goals = tuple(SubGoal.from_dict(sg) for sg in body.subGoals)

    job = await runner.start_job(
        job_id=job_id,
        owner_id=owner.owner_id,
        session_id=body.sessionId,
        pet_id=body.petId,
        host_id=host_id,
        title=body.title,
        goal=body.goal,
        max_steps=body.maxSteps,
        sub_goals=sub_goals,
    )
    return job.to_dict()


@router.get("")
async def list_jobs(
    request: Request,
    owner: CurrentOwner,
    state: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Lists autonomous jobs belonging to the authenticated owner."""
    runner = _runner(request)
    filter_states: tuple[AutonomousJobState, ...] | None = None
    if state:
        try:
            filter_states = (AutonomousJobState(state),)
        except ValueError as err:
            raise AppError(
                "invalid_argument", HTTPStatus.BAD_REQUEST, f"Unknown state: {state}"
            ) from err

    jobs = await runner._store.list_for_owner(
        owner_id=owner.owner_id,
        limit=limit,
        offset=offset,
        states=filter_states,
    )
    total = await runner._store.count_for_owner(
        owner_id=owner.owner_id,
        states=filter_states,
    )
    serialized = [j.to_dict() for j in jobs]
    return {
        "jobs": serialized,
        "items": serialized,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{job_id}")
async def get_job(
    job_id: uuid.UUID,
    request: Request,
    owner: CurrentOwner,
) -> dict[str, Any]:
    """Retrieves full details and progress of an autonomous job."""
    runner = _runner(request)
    job = await _get_owner_job(runner, job_id, owner.owner_id)
    return job.to_dict()


@router.post("/{job_id}/pause")
async def pause_job(
    job_id: uuid.UUID,
    request: Request,
    owner: CurrentOwner,
) -> dict[str, Any]:
    """Pauses an active autonomous job."""
    runner = _runner(request)
    await _get_owner_job(runner, job_id, owner.owner_id)
    try:
        updated = await runner.pause_job(job_id)
    except ValueError as err:
        raise AppError("conflict", HTTPStatus.CONFLICT, str(err)) from err
    return updated.to_dict()


@router.post("/{job_id}/resume")
async def resume_job(
    job_id: uuid.UUID,
    request: Request,
    owner: CurrentOwner,
    body: ResumeJobBody | None = None,
) -> dict[str, Any]:
    """Resumes a paused or R3-checkpointed autonomous job."""
    runner = _runner(request)
    await _get_owner_job(runner, job_id, owner.owner_id)
    approval_id = body.approvalId if body else None
    try:
        updated = await runner.resume_job(job_id, approval_id=approval_id)
    except ValueError as err:
        raise AppError("conflict", HTTPStatus.CONFLICT, str(err)) from err
    return updated.to_dict()


@router.post("/{job_id}/steer")
async def steer_job(
    job_id: uuid.UUID,
    body: SteerJobBody,
    request: Request,
    owner: CurrentOwner,
) -> dict[str, Any]:
    """Injects live human guidance into an ongoing autonomous job."""
    runner = _runner(request)
    await _get_owner_job(runner, job_id, owner.owner_id)
    try:
        updated = await runner.steer_job(job_id, steering_prompt=body.instruction)
    except ValueError as err:
        raise AppError("conflict", HTTPStatus.CONFLICT, str(err)) from err
    return updated.to_dict()


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: uuid.UUID,
    request: Request,
    owner: CurrentOwner,
    body: CancelJobBody | None = None,
) -> dict[str, Any]:
    """Cancels an autonomous job and halts its execution."""
    runner = _runner(request)
    await _get_owner_job(runner, job_id, owner.owner_id)
    reason = body.reason if body else "user_cancelled"
    try:
        updated = await runner.cancel_job(job_id, reason=reason)
    except ValueError as err:
        raise AppError("conflict", HTTPStatus.CONFLICT, str(err)) from err
    return updated.to_dict()
