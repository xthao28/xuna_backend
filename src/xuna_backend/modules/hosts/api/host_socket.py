"""`GET /v1/hosts/connect`: Host Agent WebSocket over XTP v1 (UDC-3 §5.2, §5.4, UDC-7).

Supports:
- Mode A (Enrollment): `xuna/hello {enroll}` -> `xuna/challenge {purpose:1}` ->
  `xuna/proof {purpose:1}` -> `xuna/enrollPending` -> owner confirms ->
  `xuna/enrolled` -> `xuna/challenge {purpose:2}` -> `xuna/proof {purpose:2}` ->
  `xuna/ready` -> `xuna/lanPeer` -> `tools/list`.
- Mode B (Reconnect): `xuna/hello {hostId}` -> `xuna/challenge {purpose:2}` ->
  `xuna/proof {purpose:2}` -> `xuna/ready` -> `xuna/lanPeer` -> `tools/list`.
"""

import asyncio
import contextlib
import logging
import secrets
import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketState

from xuna_backend.modules.audit.application.ports import AuditLog
from xuna_backend.modules.hosts.application.ports import (
    EnrollmentRepository,
    HostRepository,
    HostSessionHub,
)
from xuna_backend.modules.hosts.application.use_cases import (
    OpenHostSessionCommand,
    StartEnrollmentCommand,
    close_host_session,
    open_host_session,
    owner_phone_public_keys_b64,
    record_duplicate_session_rejection,
    record_manifest,
    start_enrollment,
    touch_host_session,
    update_host_lan_peer,
)
from xuna_backend.modules.hosts.application.xtp_frames import (
    ParsedHelloEnroll,
    ParsedHelloReconnect,
    ParsedLanPeer,
    ParsedPing,
    ParsedProgress,
    ParsedProof,
    ParsedRpcError,
    ParsedRpcResponse,
    ParsedStop,
    ParsedToolsListChanged,
    challenge_notification_frame,
    enroll_pending_notification_frame,
    enrolled_notification_frame,
    hello_response_frame,
    lan_peer_notification_frame,
    parse_agent_frame,
    parse_tools_list_result,
    ping_response_frame,
    proof_response_frame,
    ready_notification_frame,
    rpc_request_frame,
)
from xuna_backend.modules.hosts.domain.crypto import NONCE_SIZE, HostProofPurpose
from xuna_backend.modules.hosts.domain.entities import (
    Host,
    HostAlias,
    HostId,
    HostPlatform,
)
from xuna_backend.modules.hosts.domain.session import (
    DUPLICATE_SESSION,
    ENROLLMENT_TTL_S,
    FRAME_TOO_LARGE,
    GOING_AWAY,
    IDLE_TIMEOUT,
    RATE_LIMITED,
    SUBPROTOCOL,
    UNAUTHORIZED,
    UNSUPPORTED_PROTOCOL,
    HostGatewayLimits,
    HostRateLimiter,
    HostSessionClose,
    HostSessionRejected,
)
from xuna_backend.modules.identity.application.ports import SignatureVerifier

logger = logging.getLogger(__name__)
router = APIRouter(tags=["hosts"])


class _ClientDisconnected(Exception):
    """The Host Agent closed the WebSocket."""


class _Repos:
    def __init__(self, app: Any, db: object) -> None:
        self.hosts: HostRepository = app.state.host_repository(db)
        self.enrollments: EnrollmentRepository = app.state.host_enrollment_repository(db)
        self.audit: AuditLog = app.state.audit_log(db)
        self.verifier: SignatureVerifier = app.state.signature_verifier


async def _receive_text(
    websocket: WebSocket,
    timeout_s: float,
    max_bytes: int,
    on_timeout: HostSessionClose,
) -> str:
    try:
        message = await asyncio.wait_for(websocket.receive(), max(timeout_s, 0.001))
    except TimeoutError as error:
        raise HostSessionRejected(on_timeout) from error
    if message["type"] == "websocket.disconnect":
        raise _ClientDisconnected
    if message.get("bytes") is not None:
        raise HostSessionRejected(UNSUPPORTED_PROTOCOL)
    text: str = message.get("text") or ""
    if len(text.encode("utf-8")) > max_bytes:
        raise HostSessionRejected(FRAME_TOO_LARGE)
    return text


async def _close_socket(websocket: WebSocket, close: HostSessionClose) -> None:
    if websocket.application_state == WebSocketState.CONNECTED:
        with contextlib.suppress(RuntimeError):
            await websocket.close(code=close.code, reason=close.reason)


async def _complete_session_proof(
    websocket: WebSocket,
    limits: HostGatewayLimits,
    host_id: HostId,
    platform: HostPlatform,
    os_version: str,
    agent_version: str,
) -> Host:
    app = websocket.app
    nonce = secrets.token_bytes(NONCE_SIZE)
    issued = time.monotonic()
    await websocket.send_text(
        challenge_notification_frame(HostProofPurpose.SESSION, nonce, int(limits.challenge_ttl_s))
    )
    raw = await _receive_text(
        websocket, limits.challenge_ttl_s, limits.max_frame_bytes, UNAUTHORIZED
    )
    proof_msg = parse_agent_frame(raw, max_bytes=limits.max_frame_bytes)
    if (
        not isinstance(proof_msg, ParsedProof)
        or proof_msg.params.parsed_purpose() != HostProofPurpose.SESSION
        or time.monotonic() - issued > limits.challenge_ttl_s
    ):
        raise HostSessionRejected(UNAUTHORIZED)

    command = OpenHostSessionCommand(
        host_id=host_id,
        nonce=nonce,
        signature=proof_msg.params.signature_bytes(),
        platform=platform,
        os_version=os_version,
        agent_version=agent_version,
    )
    async with app.state.database.sessions() as db:
        repos = _Repos(app, db)
        try:
            host = await open_host_session(
                repos.hosts,
                repos.verifier,
                repos.audit,
                command,
                datetime.now(UTC),
            )
        except HostSessionRejected:
            await db.commit()
            raise
        await db.commit()

    await websocket.send_text(proof_response_frame(proof_msg.request_id))
    return host


async def _handshake(websocket: WebSocket, limits: HostGatewayLimits) -> Host:
    app = websocket.app
    hub: HostSessionHub = app.state.host_session_hub
    raw = await _receive_text(
        websocket, limits.hello_timeout_s, limits.max_frame_bytes, UNAUTHORIZED
    )
    hello = parse_agent_frame(raw, max_bytes=limits.max_frame_bytes)

    if isinstance(hello, ParsedHelloReconnect):
        await websocket.send_text(hello_response_frame(hello.request_id))
        return await _complete_session_proof(
            websocket,
            limits,
            hello.params.parsed_host_id(),
            hello.params.parsed_platform(),
            hello.params.osVersion,
            hello.params.agentVersion,
        )

    if isinstance(hello, ParsedHelloEnroll):
        await websocket.send_text(hello_response_frame(hello.request_id))
        nonce = secrets.token_bytes(NONCE_SIZE)
        issued = time.monotonic()
        await websocket.send_text(
            challenge_notification_frame(
                HostProofPurpose.ENROLL, nonce, int(limits.challenge_ttl_s)
            )
        )
        raw_proof = await _receive_text(
            websocket, limits.challenge_ttl_s, limits.max_frame_bytes, UNAUTHORIZED
        )
        proof_msg = parse_agent_frame(raw_proof, max_bytes=limits.max_frame_bytes)
        if (
            not isinstance(proof_msg, ParsedProof)
            or proof_msg.params.parsed_purpose() != HostProofPurpose.ENROLL
            or time.monotonic() - issued > limits.challenge_ttl_s
        ):
            raise HostSessionRejected(UNAUTHORIZED)

        enroll_payload = hello.params.enroll
        enroll_cmd = StartEnrollmentCommand(
            code=enroll_payload.code_bytes(),
            public_key=enroll_payload.public_key_bytes(),
            nonce=nonce,
            signature=proof_msg.params.signature_bytes(),
            platform=enroll_payload.parsed_platform(),
            os_version=enroll_payload.osVersion,
            agent_version=enroll_payload.agentVersion,
            hostname=enroll_payload.hostname,
        )
        async with app.state.database.sessions() as db:
            repos = _Repos(app, db)
            pending = await start_enrollment(
                repos.enrollments,
                repos.verifier,
                enroll_cmd,
                datetime.now(UTC),
            )
            await db.commit()

        await websocket.send_text(proof_response_frame(proof_msg.request_id))
        await websocket.send_text(enroll_pending_notification_frame(ENROLLMENT_TTL_S))

        enrolled_future: asyncio.Future[tuple[HostId, HostAlias]] = (
            asyncio.get_running_loop().create_future()
        )

        async def _on_enrolled(host_id: HostId, alias: HostAlias) -> None:
            if not enrolled_future.done():
                enrolled_future.set_result((host_id, alias))

        pending_key = str(pending.pending_id)
        hub.register_pending(pending_key, _on_enrolled)
        try:
            enrolled_host_id, enrolled_alias = await asyncio.wait_for(
                enrolled_future, limits.enroll_wait_timeout_s
            )
        except TimeoutError as error:
            raise HostSessionRejected(UNAUTHORIZED) from error
        finally:
            hub.unregister_pending(pending_key)

        await websocket.send_text(
            enrolled_notification_frame(enrolled_host_id, enrolled_alias.value)
        )
        return await _complete_session_proof(
            websocket,
            limits,
            enrolled_host_id,
            enroll_payload.parsed_platform(),
            enroll_payload.osVersion,
            enroll_payload.agentVersion,
        )

    raise HostSessionRejected(UNAUTHORIZED)


async def _serve_session(
    websocket: WebSocket,
    host: Host,
    limits: HostGatewayLimits,
    hub: HostSessionHub,
) -> None:
    app = websocket.app
    next_manifest_req_id = -1
    pending_manifest_ids: set[int | str] = {"tools_list_initial"}

    await websocket.send_text(rpc_request_frame("tools_list_initial", "tools/list", {}))

    if host.platform in (HostPlatform.WINDOWS, HostPlatform.MACOS):
        async with app.state.database.sessions() as db:
            repos = _Repos(app, db)
            phone_keys = await owner_phone_public_keys_b64(repos.hosts, host.owner_id)
        if phone_keys:
            await websocket.send_text(lan_peer_notification_frame(phone_keys))

    while True:
        raw = await _receive_text(
            websocket, limits.idle_timeout_s, limits.max_frame_bytes, IDLE_TIMEOUT
        )
        msg = parse_agent_frame(raw, max_bytes=limits.max_frame_bytes)

        if isinstance(msg, ParsedPing):
            async with app.state.database.sessions() as db:
                repos = _Repos(app, db)
                close = await touch_host_session(repos.hosts, host.host_id, datetime.now(UTC))
                await db.commit()
            if close is not None:
                raise HostSessionRejected(close)
            await websocket.send_text(ping_response_frame(msg.request_id))
            continue

        if isinstance(msg, ParsedToolsListChanged):
            req_id = f"tools_list_{abs(next_manifest_req_id)}"
            next_manifest_req_id -= 1
            pending_manifest_ids.add(req_id)
            await websocket.send_text(rpc_request_frame(req_id, "tools/list", {}))
            continue

        if isinstance(msg, ParsedProgress):
            continue

        if isinstance(msg, ParsedLanPeer):
            async with app.state.database.sessions() as db:
                repos = _Repos(app, db)
                await update_host_lan_peer(
                    repos.hosts,
                    host.host_id,
                    msg.params.enabled,
                    msg.params.certFingerprint,
                    msg.params.port,
                    datetime.now(UTC),
                )
                await db.commit()
            continue

        if isinstance(msg, ParsedStop):
            logger.info("host stop notification reason=%s", msg.params.reason)
            if hasattr(app.state, "approval_broker"):
                from xuna_backend.modules.actions.api.routes import build_stream_orchestrator

                orch = build_stream_orchestrator(app.state)
                await orch.emergency_stop(
                    owner_id=host.owner_id,
                    reason=msg.params.reason,
                )
            continue

        if isinstance(msg, ParsedRpcResponse):
            if msg.request_id in pending_manifest_ids:
                pending_manifest_ids.discard(msg.request_id)
                try:
                    tools = parse_tools_list_result(msg.result)
                except ValueError as error:
                    raise HostSessionRejected(UNSUPPORTED_PROTOCOL) from error
                async with app.state.database.sessions() as db:
                    repos = _Repos(app, db)
                    await record_manifest(repos.hosts, host.host_id, tools, datetime.now(UTC))
                    await db.commit()
            hub.on_response(str(host.host_id), msg.request_id, msg.result)
            continue

        if isinstance(msg, ParsedRpcError):
            hub.on_error_response(str(host.host_id), msg.request_id, msg.code, msg.message)
            continue

        # Any handshake frame (`xuna/hello`, `xuna/proof`) received after `ready` is out of order.
        raise HostSessionRejected(UNAUTHORIZED)


@router.websocket("/hosts/connect")
async def host_connect(websocket: WebSocket) -> None:
    app = websocket.app
    limits: HostGatewayLimits = app.state.host_gateway_limits
    hub: HostSessionHub = app.state.host_session_hub
    offered = websocket.scope.get("subprotocols") or []
    await websocket.accept(subprotocol=SUBPROTOCOL if SUBPROTOCOL in offered else None)
    if SUBPROTOCOL not in offered:
        await _close_socket(websocket, UNSUPPORTED_PROTOCOL)
        return

    peer = websocket.client.host if websocket.client else ""
    connect_limiter: HostRateLimiter = app.state.host_connect_limiter
    if not connect_limiter.allow(peer, time.monotonic()):
        await _close_socket(websocket, RATE_LIMITED)
        return

    host: Host | None = None
    token = object()

    async def _send_text(text: str) -> None:
        await websocket.send_text(text)

    async def _close_duplicate() -> None:
        await _close_socket(websocket, DUPLICATE_SESSION)

    async def _close_revoked() -> None:
        await _close_socket(websocket, UNAUTHORIZED)

    try:
        host = await _handshake(websocket, limits)
        opened = await hub.open_session(
            str(host.host_id),
            token,
            _send_text,
            _close_duplicate,
            _close_revoked,
        )
        if not opened:
            async with app.state.database.sessions() as db:
                repos = _Repos(app, db)
                await record_duplicate_session_rejection(
                    repos.hosts, repos.audit, host.host_id, datetime.now(UTC)
                )
                await db.commit()
            raise HostSessionRejected(DUPLICATE_SESSION)

        await websocket.send_text(
            ready_notification_frame(datetime.now(UTC).isoformat(), limits.heartbeat_s)
        )
        await _serve_session(websocket, host, limits, hub)
    except HostSessionRejected as rejected:
        logger.info("host session closed code=%s", rejected.close.reason)
        await _close_socket(websocket, rejected.close)
    except (_ClientDisconnected, RuntimeError):
        pass
    except asyncio.CancelledError:
        await _close_socket(websocket, GOING_AWAY)
        raise
    finally:
        if host is not None and hub.close_session(str(host.host_id), token):
            await asyncio.shield(_mark_host_offline(app, host.host_id))


async def _mark_host_offline(app: Any, host_id: HostId) -> None:
    async with app.state.database.sessions() as db:
        await close_host_session(app.state.host_repository(db), host_id, datetime.now(UTC))
        await db.commit()
