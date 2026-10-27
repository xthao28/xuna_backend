"""`GET /v1/devices/connect`: the board's own WebSocket (DC-3, ADR-0010).

hello → challenge → proof → ready, then ping/pong. Logs carry close codes only:
no pet data, no keys, no peer address. The peer address is used in memory for
the connection rate limit and never stored.
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
from xuna_backend.modules.consent.application.ports import ConsentRepository
from xuna_backend.modules.identity.application.ports import (
    IdentityRepository,
    SignatureVerifier,
)
from xuna_backend.modules.identity.domain.device_proof import (
    NONCE_SIZE,
    PUBLIC_KEY_SIZE,
    SIGNATURE_SIZE,
)
from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.transport.application.device_frames import (
    HelloFrame,
    PingFrame,
    ProofFrame,
    challenge_frame,
    decode_b64,
    parse_device_frame,
    pong_frame,
    ready_frame,
)
from xuna_backend.modules.transport.application.device_gateway import (
    DeviceSession,
    SessionProof,
    close_device_session,
    open_device_session,
    touch_device_session,
)
from xuna_backend.modules.transport.application.ports import (
    DeviceSessionHub,
    TransportStatusRepository,
)
from xuna_backend.modules.transport.domain.device_session import (
    FRAME_TOO_LARGE,
    GOING_AWAY,
    IDLE_TIMEOUT,
    RATE_LIMITED,
    REPLACED,
    SUBPROTOCOL,
    UNAUTHORIZED,
    UNSUPPORTED_PROTOCOL,
    GatewayLimits,
    SessionClose,
    SessionRejected,
    SlidingWindowLimiter,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["transport"])  # included under /v1 by transport routes


class _Gone(Exception):
    """The device closed the connection."""


class _Repos:
    def __init__(self, app: Any, db: object) -> None:
        self.identity: IdentityRepository = app.state.identity_repository(db)
        self.consent: ConsentRepository = app.state.consent_repository(db)
        self.statuses: TransportStatusRepository = app.state.transport_status_repository(db)
        self.audit: AuditLog = app.state.audit_log(db)
        self.verifier: SignatureVerifier = app.state.signature_verifier


async def _receive(
    websocket: WebSocket, wait_s: float, limit: int, on_timeout: SessionClose
) -> str:
    try:
        message = await asyncio.wait_for(websocket.receive(), max(wait_s, 0.001))
    except TimeoutError as error:
        raise SessionRejected(on_timeout) from error
    if message["type"] == "websocket.disconnect":
        raise _Gone
    if message.get("bytes") is not None:
        raise SessionRejected(UNSUPPORTED_PROTOCOL)
    text: str = message.get("text") or ""
    if len(text.encode()) > limit:
        raise SessionRejected(FRAME_TOO_LARGE)
    return text


async def _close(websocket: WebSocket, close: SessionClose) -> None:
    if websocket.application_state == WebSocketState.CONNECTED:
        with contextlib.suppress(RuntimeError):  # already closing from the other side
            await websocket.close(code=close.code, reason=close.reason)


async def _handshake(websocket: WebSocket, limits: GatewayLimits) -> DeviceSession:
    app = websocket.app
    raw = await _receive(websocket, limits.hello_timeout_s, limits.max_frame_bytes, UNAUTHORIZED)
    hello = parse_device_frame(raw)
    if not isinstance(hello, HelloFrame):
        raise SessionRejected(UNAUTHORIZED)  # steps out of order
    challenge_limiter: SlidingWindowLimiter = app.state.device_challenge_limiter
    if not challenge_limiter.allow(hello.petId, time.monotonic()):
        raise SessionRejected(RATE_LIMITED)
    nonce = secrets.token_bytes(NONCE_SIZE)
    issued = time.monotonic()
    await websocket.send_text(challenge_frame(nonce, int(limits.challenge_ttl_s)))
    raw = await _receive(websocket, limits.challenge_ttl_s, limits.max_frame_bytes, UNAUTHORIZED)
    proof = parse_device_frame(raw)
    if not isinstance(proof, ProofFrame) or time.monotonic() - issued > limits.challenge_ttl_s:
        raise SessionRejected(UNAUTHORIZED)
    command = SessionProof(
        pet_id=PetId(hello.petId),
        nonce=nonce,
        generation=proof.generation,
        public_key=(
            decode_b64(proof.publicKey, PUBLIC_KEY_SIZE) if proof.publicKey is not None else None
        ),
        signature=decode_b64(proof.signature, SIGNATURE_SIZE),
        firmware=hello.firmware,
    )
    async with app.state.database.sessions() as db:
        repos = _Repos(app, db)
        try:
            session = await open_device_session(
                repos.identity,
                repos.verifier,
                repos.consent,
                repos.statuses,
                repos.audit,
                command,
                datetime.now(UTC),
            )
        except SessionRejected:
            await db.commit()  # keep the audited rejection
            raise
        await db.commit()
    return session


async def _serve(websocket: WebSocket, session: DeviceSession, limits: GatewayLimits) -> None:
    app = websocket.app
    while True:
        raw = await _receive(websocket, limits.idle_timeout_s, limits.max_frame_bytes, IDLE_TIMEOUT)
        frame = parse_device_frame(raw)
        if not isinstance(frame, PingFrame):
            # hello/proof again are out of order; projections arrive with DC-5.
            raise SessionRejected(
                UNAUTHORIZED if frame.type in ("hello", "proof") else UNSUPPORTED_PROTOCOL
            )
        async with app.state.database.sessions() as db:
            repos = _Repos(app, db)
            close = await touch_device_session(
                repos.identity, repos.consent, repos.statuses, session, datetime.now(UTC)
            )
            await db.commit()
        if close is not None:
            raise SessionRejected(close)
        await websocket.send_text(pong_frame(frame.seq))


@router.websocket("/devices/connect")
async def device_connect(websocket: WebSocket) -> None:
    app = websocket.app
    limits: GatewayLimits = app.state.device_gateway_limits
    hub: DeviceSessionHub = app.state.device_session_hub
    offered = websocket.scope.get("subprotocols") or []
    await websocket.accept(subprotocol=SUBPROTOCOL if SUBPROTOCOL in offered else None)
    if SUBPROTOCOL not in offered:
        await _close(websocket, UNSUPPORTED_PROTOCOL)
        return
    peer = websocket.client.host if websocket.client else ""
    connect_limiter: SlidingWindowLimiter = app.state.device_connect_limiter
    if not connect_limiter.allow(peer, time.monotonic()):
        await _close(websocket, RATE_LIMITED)
        return

    session: DeviceSession | None = None
    token = object()

    async def replace() -> None:
        await _close(websocket, REPLACED)

    try:
        session = await _handshake(websocket, limits)
        await hub.claim(session.pet_id.value, token, replace)
        await websocket.send_text(ready_frame(datetime.now(UTC).isoformat(), limits.heartbeat_s))
        await _serve(websocket, session, limits)
    except SessionRejected as rejected:
        logger.info("device session closed code=%s", rejected.close.reason)
        await _close(websocket, rejected.close)
    except (_Gone, RuntimeError):
        pass  # device went away, or a newer session closed this one
    except asyncio.CancelledError:
        await _close(websocket, GOING_AWAY)
        raise
    finally:
        if session is not None and hub.release(session.pet_id.value, token):
            # Shielded: a cancelled connection (shutdown, client gone) still ends offline.
            await asyncio.shield(_mark_offline(app, session))


async def _mark_offline(app: Any, session: DeviceSession) -> None:
    async with app.state.database.sessions() as db:
        await close_device_session(
            app.state.transport_status_repository(db), session, datetime.now(UTC)
        )
        await db.commit()
