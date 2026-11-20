"""HTTP routes for `/v1/hosts` enrollment, management and dev invocation (UDC-3 §5.4, UDC-7)."""

import base64
import logging
import secrets
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from http import HTTPStatus
from typing import Annotated, Any

from fastapi import APIRouter, Header, Request, Response
from fastapi.responses import JSONResponse

from xuna_backend.core.config import Settings
from xuna_backend.core.db import commit_or_conflict
from xuna_backend.core.errors import AppError
from xuna_backend.core.idempotency import StoredResponse, parse_key, request_hash, run_once
from xuna_backend.modules.accounts.api.dependencies import CurrentOwner, Session, state
from xuna_backend.modules.actions.application.computer_use import (
    MAX_SESSION_MINUTES,
    MAX_SESSION_STEPS,
    ComputerUseSessionState,
)
from xuna_backend.modules.actions.application.ports import ComputerUseSessionStore
from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.hosts.api.host_socket import router as socket_router
from xuna_backend.modules.hosts.api.schemas import (
    ClaimHostRequest,
    ConfirmHostRequest,
    DevInvokeToolRequest,
    DevPreflightRequest,
    SelfEnrollHostRequest,
    SetHostGrantsRequest,
    SetHostPolicyRequest,
    UpdateHostManifestRequest,
)
from xuna_backend.modules.hosts.application.ports import (
    EnrollmentRepository,
    HostRepository,
    HostSessionHub,
)
from xuna_backend.modules.hosts.application.use_cases import (
    ClaimEnrollmentCommand,
    DevApprovalInput,
    GrantUpdateSpec,
    HostError,
    HostSummaryView,
    SelfEnrollHostCommand,
    claim_enrollment,
    confirm_enrollment,
    get_host_policy,
    invoke_tool,
    list_hosts,
    owner_phone_public_keys_b64,
    preflight_tool,
    revoke_host,
    self_enroll_host,
    set_grants,
    set_host_policy,
    update_host_manifest,
)
from xuna_backend.modules.hosts.domain.crypto import (
    ENROLLMENT_CODE_SIZE,
    NONCE_SIZE,
    PUBLIC_KEY_SIZE,
    SIGNATURE_SIZE,
    public_key_fingerprint_b64url,
)
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    Host,
    HostId,
    HostPlatform,
    HostStatus,
    Manifest,
)
from xuna_backend.modules.hosts.domain.session import (
    HostRateLimiter,
    SelfEnrollmentChallengeStore,
)
from xuna_backend.modules.identity.application.ports import SignatureVerifier

router = APIRouter(prefix="/v1", tags=["hosts"])
router.include_router(socket_router)

IdempotencyKey = Annotated[str | None, Header(alias="Idempotency-Key")]

_ERROR_STATUS = {
    "not_found": HTTPStatus.NOT_FOUND,
    "conflict": HTTPStatus.CONFLICT,
    "rate_limited": HTTPStatus.TOO_MANY_REQUESTS,
    "unauthorized": HTTPStatus.FORBIDDEN,
    "validation_failed": HTTPStatus.UNPROCESSABLE_ENTITY,
    "invalid_argument": HTTPStatus.UNPROCESSABLE_ENTITY,
    "host_offline": HTTPStatus.CONFLICT,
    "tool_not_found": HTTPStatus.NOT_FOUND,
    "grant_denied": HTTPStatus.FORBIDDEN,
    "policy_blocked": HTTPStatus.UNPROCESSABLE_ENTITY,
}


def _host_repo(request: Request, session: object) -> HostRepository:
    factory: Callable[[object], HostRepository] = state(request, "host_repository")
    return factory(session)


def _enrollment_repo(request: Request, session: object) -> EnrollmentRepository:
    factory: Callable[[object], EnrollmentRepository] = state(request, "host_enrollment_repository")
    return factory(session)


def _audit(request: Request, session: object) -> AuditLog:
    factory: Callable[[object], AuditLog] = state(request, "audit_log")
    return factory(session)


def _hub(request: Request) -> HostSessionHub:
    hub: HostSessionHub = state(request, "host_session_hub")
    return hub


def _problem(error: HostError) -> AppError:
    return AppError(
        error.code,
        _ERROR_STATUS.get(error.code, HTTPStatus.BAD_REQUEST),
        error.detail,
    )


def _grant_body(grant: Grant) -> dict[str, Any]:
    return {
        "family": grant.family.value,
        "maxRisk": grant.max_risk.value,
        "enabled": grant.enabled,
        "updatedAt": grant.updated_at.astimezone(UTC).isoformat(),
        "expiresAt": (
            grant.expires_at.astimezone(UTC).isoformat() if grant.expires_at is not None else None
        ),
        "suspendedAt": (
            grant.suspended_at.astimezone(UTC).isoformat()
            if grant.suspended_at is not None
            else None
        ),
    }


def _host_body(
    host: Host,
    grants: list[Grant],
    *,
    online: bool = False,
    manifest: Manifest | None = None,
) -> dict[str, Any]:
    return {
        "hostId": str(host.host_id),
        "alias": host.alias.value,
        "platform": host.platform.value,
        "hostname": host.hostname,
        "osVersion": host.os_version,
        "agentVersion": host.agent_version,
        "status": host.status.value,
        "online": online,
        "keyFingerprint": public_key_fingerprint_b64url(host.public_key),
        "publicKey": base64.b64encode(host.public_key).decode("ascii"),
        "lanFingerprint": host.lan_fingerprint,
        "lanPort": host.lan_port,
        "createdAt": host.created_at.astimezone(UTC).isoformat(),
        "lastSeenAt": (
            host.last_seen_at.astimezone(UTC).isoformat() if host.last_seen_at is not None else None
        ),
        "grants": [_grant_body(g) for g in grants],
        "tools": [t.to_wire() for t in manifest.tools] if manifest is not None else [],
    }


def _summary_body(view: HostSummaryView) -> dict[str, Any]:
    return _host_body(
        view.host,
        view.grants,
        online=view.online,
        manifest=view.manifest,
    )


def _json(response: StoredResponse) -> JSONResponse:
    return JSONResponse(response.body, status_code=response.status.value)


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(encoded: str, expected_len: int) -> bytes:
    clean = encoded.strip()
    rem = len(clean) % 4
    if rem == 2:
        clean += "=="
    elif rem == 3:
        clean += "="
    elif rem != 0:
        raise ValueError("invalid base64url length")
    try:
        decoded = base64.urlsafe_b64decode(clean.encode("ascii"))
    except Exception as error:
        raise ValueError("invalid base64url encoding") from error
    if len(decoded) != expected_len:
        raise ValueError(f"expected {expected_len} decoded bytes")
    return decoded


@router.post("/hosts/self/challenge", status_code=HTTPStatus.OK)
async def self_enroll_challenge_endpoint(
    request: Request,
    owner: CurrentOwner,
) -> dict[str, Any]:
    """Issues a 32-byte server nonce and 16-byte code for phone self-enrollment (`UDC-5` §5.2)."""
    store: SelfEnrollmentChallengeStore = state(request, "host_self_challenge_store")
    nonce = secrets.token_bytes(NONCE_SIZE)
    code = secrets.token_bytes(ENROLLMENT_CODE_SIZE)
    store.issue(str(owner.owner_id), nonce, code, time.monotonic())
    return {
        "nonce": _b64url_encode(nonce),
        "enrollmentCode": _b64url_encode(code),
        "expiresIn": 60,
    }


@router.post("/hosts/self", status_code=HTTPStatus.OK)
async def self_enroll_host_endpoint(
    body: SelfEnrollHostRequest,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    idempotency_key: IdempotencyKey = None,
) -> JSONResponse:
    """Enrolls (or updates) the owner's phone as a host without QR (`UDC-5` §5.2)."""
    now = datetime.now(UTC)
    claim_limiter: HostRateLimiter = state(request, "host_claim_limiter")
    challenge_store: SelfEnrollmentChallengeStore = state(request, "host_self_challenge_store")
    verifier: SignatureVerifier = state(request, "signature_verifier")
    hub = _hub(request)

    try:
        pub_bytes = _b64url_decode(body.publicKey, PUBLIC_KEY_SIZE)
        sig_bytes = _b64url_decode(body.proof, SIGNATURE_SIZE)
    except ValueError as error:
        raise AppError("validation_failed", HTTPStatus.UNPROCESSABLE_ENTITY, str(error)) from error

    command = SelfEnrollHostCommand(
        platform=body.parsed_platform(),
        os_version=body.osVersion,
        agent_version=body.agentVersion,
        alias=body.alias,
        public_key=pub_bytes,
        signature=sig_bytes,
        hostname=body.hostname,
        tools=body.tools,
    )

    async def action() -> StoredResponse:
        repo = _host_repo(request, session)
        host, grants, manifest = await self_enroll_host(
            repo,
            verifier,
            _audit(request, session),
            claim_limiter,
            challenge_store,
            owner.owner_id,
            command,
            now,
            time.monotonic(),
            getattr(request.state, "request_id", ""),
        )
        phone_keys = await owner_phone_public_keys_b64(repo, owner.owner_id)
        all_hosts = await repo.list_hosts(owner.owner_id)
        for h in all_hosts:
            if h.status == HostStatus.ACTIVE and h.platform in (
                HostPlatform.WINDOWS,
                HostPlatform.MACOS,
            ):
                await hub.notify(
                    str(h.host_id),
                    "xuna/lanPeer",
                    {"phonePublicKeys": phone_keys},
                )
        return StoredResponse(
            HTTPStatus.OK,
            _host_body(
                host,
                grants,
                online=hub.is_online(str(host.host_id)),
                manifest=manifest,
            ),
        )

    try:
        if idempotency_key is not None:
            response = await run_once(
                session,
                owner_id=owner.owner_id,
                key=parse_key(idempotency_key),
                fingerprint=request_hash("self_enroll_host", body.model_dump(mode="json")),
                now=now,
                action=action,
            )
        else:
            response = await action()
    except HostError as error:
        await commit_or_conflict(session)
        raise _problem(error) from error
    await commit_or_conflict(session)
    return _json(response)


@router.put("/hosts/{host_id}/manifest", status_code=HTTPStatus.OK)
async def update_host_manifest_endpoint(
    host_id: str,
    body: UpdateHostManifestRequest,
    request: Request,
    owner: CurrentOwner,
    session: Session,
) -> dict[str, Any]:
    """Updates the saved tool manifest for a self-enrolled phone host."""
    try:
        parsed_id = HostId.from_str(host_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error
    now = datetime.now(UTC)
    try:
        manifest = await update_host_manifest(
            _host_repo(request, session),
            owner.owner_id,
            parsed_id,
            body.tools,
            now,
        )
    except HostError as error:
        raise _problem(error) from error
    await commit_or_conflict(session)
    return {
        "hostId": str(parsed_id),
        "tools": [t.to_wire() for t in manifest.tools],
        "updatedAt": manifest.updated_at.astimezone(UTC).isoformat(),
    }


@router.post("/hosts", status_code=HTTPStatus.ACCEPTED)
async def claim_host_endpoint(
    body: ClaimHostRequest,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    idempotency_key: IdempotencyKey = None,
) -> JSONResponse:
    now = datetime.now(UTC)
    claim_limiter: HostRateLimiter = state(request, "host_claim_limiter")
    command = ClaimEnrollmentCommand(
        enrollment_code=body.enrollmentCode,
        alias=body.alias,
        key_fingerprint=body.keyFingerprint,
    )

    async def action() -> StoredResponse:
        pending = await claim_enrollment(
            _enrollment_repo(request, session),
            _audit(request, session),
            claim_limiter,
            owner.owner_id,
            command,
            now,
            time.monotonic(),
            getattr(request.state, "request_id", ""),
        )
        pid = str(pending.pending_id)
        return StoredResponse(
            HTTPStatus.ACCEPTED,
            {
                "hostId": pid,
                "pendingId": pid,
                "sas": pending.sas,
                "platform": pending.platform.value,
                "hostname": pending.hostname,
                "expiresAt": pending.expires_at.astimezone(UTC).isoformat(),
            },
        )

    try:
        response = await run_once(
            session,
            owner_id=owner.owner_id,
            key=parse_key(idempotency_key),
            fingerprint=request_hash("claim_host", body.model_dump(mode="json")),
            now=now,
            action=action,
        )
    except HostError as error:
        await commit_or_conflict(session)
        raise _problem(error) from error
    await commit_or_conflict(session)
    return _json(response)


@router.post("/hosts/{host_id}/confirm", status_code=HTTPStatus.OK)
async def confirm_host_endpoint(
    host_id: str,
    body: ConfirmHostRequest,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    idempotency_key: IdempotencyKey = None,
) -> JSONResponse:
    try:
        parsed_id = HostId.from_str(host_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error
    now = datetime.now(UTC)
    hub = _hub(request)

    async def action() -> StoredResponse:
        host, grants = await confirm_enrollment(
            _enrollment_repo(request, session),
            _host_repo(request, session),
            hub,
            _audit(request, session),
            owner.owner_id,
            parsed_id,
            now,
            getattr(request.state, "request_id", ""),
        )
        return StoredResponse(
            HTTPStatus.OK,
            _host_body(host, grants, online=hub.is_online(str(host.host_id))),
        )

    try:
        response = await run_once(
            session,
            owner_id=owner.owner_id,
            key=parse_key(idempotency_key),
            fingerprint=request_hash(
                "confirm_host",
                {"hostId": str(parsed_id), **body.model_dump(mode="json")},
            ),
            now=now,
            action=action,
        )
    except HostError as error:
        raise _problem(error) from error
    await commit_or_conflict(session)
    return _json(response)


@router.get("/hosts")
async def list_hosts_endpoint(
    request: Request,
    owner: CurrentOwner,
    session: Session,
) -> dict[str, Any]:
    views = await list_hosts(_host_repo(request, session), _hub(request), owner.owner_id)
    return {"items": [_summary_body(v) for v in views]}


@router.delete("/hosts/{host_id}", status_code=HTTPStatus.NO_CONTENT)
async def revoke_host_endpoint(
    host_id: str,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    idempotency_key: IdempotencyKey = None,
) -> Response:
    try:
        parsed_id = HostId.from_str(host_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error
    now = datetime.now(UTC)

    async def action() -> StoredResponse:
        await revoke_host(
            _host_repo(request, session),
            _hub(request),
            _audit(request, session),
            owner.owner_id,
            parsed_id,
            now,
            getattr(request.state, "request_id", ""),
        )
        return StoredResponse(HTTPStatus.NO_CONTENT, {})

    try:
        await run_once(
            session,
            owner_id=owner.owner_id,
            key=parse_key(idempotency_key),
            fingerprint=request_hash("revoke_host", {"hostId": str(parsed_id)}),
            now=now,
            action=action,
        )
    except HostError as error:
        raise _problem(error) from error
    await commit_or_conflict(session)
    return Response(status_code=HTTPStatus.NO_CONTENT)


@router.put("/hosts/{host_id}/grants")
async def set_host_grants_endpoint(
    host_id: str,
    body: SetHostGrantsRequest,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    idempotency_key: IdempotencyKey = None,
) -> JSONResponse:
    try:
        parsed_id = HostId.from_str(host_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error
    now = datetime.now(UTC)

    specs: list[tuple[Any, Any, bool] | GrantUpdateSpec] = [
        GrantUpdateSpec(
            family=g.parsed_family(),
            max_risk=g.parsed_max_risk(),
            enabled=g.enabled,
            expires_at=g.expiresAt,
            suspended=g.suspended,
        )
        for g in body.grants
    ]

    async def action() -> StoredResponse:
        grants = await set_grants(
            _host_repo(request, session),
            _audit(request, session),
            owner.owner_id,
            parsed_id,
            specs,
            now,
            getattr(request.state, "request_id", ""),
            suspend_all=body.suspendAll,
        )
        return StoredResponse(
            HTTPStatus.OK,
            {"hostId": str(parsed_id), "grants": [_grant_body(g) for g in grants]},
        )

    try:
        response = await run_once(
            session,
            owner_id=owner.owner_id,
            key=parse_key(idempotency_key),
            fingerprint=request_hash(
                "set_host_grants",
                {"hostId": str(parsed_id), **body.model_dump(mode="json")},
            ),
            now=now,
            action=action,
        )
    except HostError as error:
        raise _problem(error) from error
    await commit_or_conflict(session)
    return _json(response)


@router.get("/hosts/{host_id}/policy")
async def get_host_policy_endpoint(
    host_id: str,
    request: Request,
    owner: CurrentOwner,
    session: Session,
) -> dict[str, Any]:
    """Retrieves active policy, apps, and allow/deny rules from an online host (UDC-18)."""
    try:
        parsed_id = HostId.from_str(host_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error

    try:
        policy = await get_host_policy(
            _host_repo(request, session),
            _hub(request),
            owner.owner_id,
            parsed_id,
        )
    except HostError as error:
        raise _problem(error) from error

    return policy


@router.put("/hosts/{host_id}/policy")
async def set_host_policy_endpoint(
    host_id: str,
    body: SetHostPolicyRequest,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    idempotency_key: IdempotencyKey = None,
) -> JSONResponse:
    """Updates the trust preset and app/domain allow/deny policies on an online host (UDC-18)."""
    try:
        parsed_id = HostId.from_str(host_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error
    now = datetime.now(UTC)

    xtp_payload = body.to_xtp_payload()

    async def action() -> StoredResponse:
        repo = _host_repo(request, session)
        hub = _hub(request)
        res = await set_host_policy(
            repo,
            hub,
            _audit(request, session),
            owner.owner_id,
            parsed_id,
            xtp_payload,
            now,
            request_id=getattr(request.state, "request_id", ""),
        )
        if "trustPreset" not in res:
            res = await get_host_policy(repo, hub, owner.owner_id, parsed_id)
        return StoredResponse(HTTPStatus.OK, res)

    try:
        response = await run_once(
            session,
            owner_id=owner.owner_id,
            key=parse_key(idempotency_key),
            fingerprint=request_hash(
                "set_host_policy",
                {"hostId": str(parsed_id), **xtp_payload},
            ),
            now=now,
            action=action,
        )
    except HostError as error:
        await commit_or_conflict(session)
        raise _problem(error) from error
    await commit_or_conflict(session)
    return _json(response)


@router.post("/hosts/{host_id}/tools:preflight")
async def dev_preflight_host_tool_endpoint(
    host_id: str,
    body: DevPreflightRequest,
    request: Request,
    owner: CurrentOwner,
    session: Session,
) -> JSONResponse:
    """Dev/test `xuna/preflight` endpoint enabled only when `XUNA_DEV_HOST_TOOLS=1`."""
    settings: Settings = state(request, "settings")
    if not settings.dev_host_tools:
        raise AppError("not_found", HTTPStatus.NOT_FOUND)
    try:
        parsed_id = HostId.from_str(host_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error
    try:
        result = await preflight_tool(
            _host_repo(request, session),
            _hub(request),
            owner.owner_id,
            parsed_id,
            body.tool,
            body.arguments,
            deadline_ms=body.deadlineMs,
        )
    except HostError as error:
        raise _problem(error) from error
    return JSONResponse(
        status_code=HTTPStatus.OK,
        content={
            "tool": result.tool_name,
            "status": result.status,
            "effectiveRisk": (
                result.effective_risk.value if result.effective_risk is not None else None
            ),
            "reasonCode": result.reason_code,
            "errorCode": result.error_code,
            "elementRole": result.element_role,
            "elementName": result.element_name,
            "durationMs": result.duration_ms,
        },
    )


@router.post("/hosts/{host_id}/tools:invoke")
async def dev_invoke_host_tool_endpoint(
    host_id: str,
    body: DevInvokeToolRequest,
    request: Request,
    owner: CurrentOwner,
    session: Session,
    idempotency_key: IdempotencyKey = None,
) -> JSONResponse:
    """Dev/test tool invocation endpoint enabled only when `XUNA_DEV_HOST_TOOLS=1`."""
    settings: Settings = state(request, "settings")
    if not settings.dev_host_tools:
        raise AppError("not_found", HTTPStatus.NOT_FOUND)
    try:
        parsed_id = HostId.from_str(host_id)
    except ValueError as error:
        raise AppError("not_found", HTTPStatus.NOT_FOUND) from error
    now = datetime.now(UTC)

    dev_approval: DevApprovalInput | None = None
    if any(
        val is not None
        for val in (
            body.approvalId,
            body.expiresAt,
            body.scope,
            body.signature,
            body.publicKey,
        )
    ):
        dev_approval = DevApprovalInput(
            approval_id=body.approvalId,
            expires_at=body.expiresAt,
            scope=body.scope,
            signature=body.signature,
            public_key=body.publicKey,
        )

    async def action() -> StoredResponse:
        repo = _host_repo(request, session)
        result = await invoke_tool(
            repo,
            _hub(request),
            _audit(request, session),
            owner.owner_id,
            parsed_id,
            body.tool,
            body.arguments,
            now,
            deadline_ms=body.deadlineMs,
            include_dev_approval=body.approve,
            dev_approval=dev_approval,
            request_id=getattr(request.state, "request_id", ""),
        )
        cu_store: ComputerUseSessionStore | None = getattr(
            request.app.state, "cu_session_store", None
        )
        if cu_store is not None and result.status == "ok":
            await _mirror_dev_cu_lifecycle(
                cu_store=cu_store,
                host_repo=repo,
                owner_id=owner.owner_id,
                host_id=parsed_id,
                tool_name=body.tool,
                arguments=body.arguments,
                structured_content=result.structured_content,
                now=now,
            )
        return StoredResponse(
            HTTPStatus.OK,
            {
                "invocationId": result.invocation_id,
                "tool": result.tool_name,
                "risk": result.risk.value,
                "status": result.status,
                "isError": result.is_error,
                "structuredContent": result.structured_content,
                "text": result.text,
                "durationMs": result.duration_ms,
                "errorCode": result.error_code,
            },
        )

    try:
        response = await run_once(
            session,
            owner_id=owner.owner_id,
            key=parse_key(idempotency_key),
            fingerprint=request_hash(
                "dev_invoke_host_tool",
                {"hostId": str(parsed_id), **body.model_dump(mode="json")},
            ),
            now=now,
            action=action,
        )
    except HostError as error:
        await commit_or_conflict(session)
        raise _problem(error) from error
    await commit_or_conflict(session)
    return _json(response)


async def _mirror_dev_cu_lifecycle(
    *,
    cu_store: ComputerUseSessionStore,
    host_repo: HostRepository,
    owner_id: uuid.UUID,
    host_id: HostId,
    tool_name: str,
    arguments: dict[str, object],
    structured_content: dict[str, object],
    now: datetime,
) -> None:
    """Mirrors dev `tools:invoke` CU start/end into `cu_session_store` for follow-up turns."""
    try:
        if tool_name == "cu.session_start":
            raw_sid = structured_content.get("sessionId")
            session_id = (
                raw_sid.strip()
                if isinstance(raw_sid, str) and raw_sid.strip()
                else str(uuid.uuid4())
            )
            raw_apps = arguments.get("apps")
            apps = (
                tuple(str(a).strip().lower() for a in raw_apps if str(a).strip())
                if isinstance(raw_apps, list)
                else ()
            )
            raw_domains = arguments.get("domains")
            domains = (
                tuple(str(d).strip().lower() for d in raw_domains if str(d).strip())
                if isinstance(raw_domains, list)
                else ()
            )
            raw_steps = arguments.get("maxSteps")
            max_steps = min(max(raw_steps if type(raw_steps) is int else 12, 1), MAX_SESSION_STEPS)
            raw_mins = arguments.get("maxMinutes")
            max_mins = min(max(raw_mins if type(raw_mins) is int else 5, 1), MAX_SESSION_MINUTES)
            host_obj = await host_repo.get_host(host_id)
            target_alias = host_obj.alias.value if host_obj is not None else ""
            new_state = ComputerUseSessionState(
                session_id=session_id,
                owner_id=owner_id,
                host_id=host_id,
                goal=str(arguments.get("goal", "")),
                allowed_apps=apps,
                allowed_domains=domains,
                max_steps=max_steps,
                max_minutes=max_mins,
                started_at_utc=now,
                last_heartbeat_at=now,
                target_alias=target_alias,
            )
            fg_app = structured_content.get("foregroundApp")
            if isinstance(fg_app, str) and fg_app.strip():
                new_state.last_app_id = fg_app
            win_title = structured_content.get("windowTitle")
            if isinstance(win_title, str) and win_title.strip():
                new_state.last_window_title = win_title
            await cu_store.start(new_state, now)
        elif tool_name == "cu.session_end":
            outcome = str(arguments.get("outcome") or "completed")
            raw_sid = arguments.get("sessionId")
            if isinstance(raw_sid, str) and raw_sid.strip():
                named = await cu_store.get(raw_sid.strip())
                if named is not None and named.owner_id == owner_id:
                    await cu_store.mark_ended(named.session_id, outcome, now)
            active = await cu_store.get_active_for_owner(owner_id)
            if active is not None and active.host_id == host_id:
                await cu_store.mark_ended(active.session_id, outcome, now)
    except Exception:
        logging.getLogger(__name__).debug("Failed to mirror dev CU lifecycle", exc_info=True)
