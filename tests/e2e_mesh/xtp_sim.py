"""Wire-spec XTP v1 Host Agent simulator, Owner HTTP/SSE/WS client, and Cognition fakes.

Strictly opaque-box: implements ECDSA P-256 proofs, RFC 8785 JCS canonical JSON,
XTP v1 WebSocket framing, SSE parsing, and Voice WS framing solely from public
specifications (`XUNA_XTP_V1_CONTRACT.md`, `XUNA_SPATIAL_DEVICE_MESH.md`) and
verifies crypto parity against `backend/tests/fixtures/xtp/golden_vectors.json`.
"""

import asyncio
import base64
import contextlib
import hashlib
import json
import math
import secrets
import struct
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import websockets
from websockets.legacy.client import WebSocketClientProtocol
from websockets.legacy.client import connect as ws_legacy_connect

from tests.helpers import LocalIssuer, SimulatedDevice, b64, idem
from xuna_backend.modules.cognition.domain.types import (
    AudioChunk,
    ChatEvent,
    ChatFinished,
    ChatRequest,
    Locale,
    TextDelta,
    Transcript,
)

# ---------------------------------------------------------------------------
# Wire-Spec Cryptography & Canonical JSON (Public XTP v1 Contract)
# ---------------------------------------------------------------------------

_HOST_PROOF_DOMAIN = b"xuna.host.proof.v1\x00"
_SAS_DOMAIN = b"xuna.host.sas.v1"
_APPROVAL_DOMAIN = b"xuna.approval.v1\x00"
_PET_PROOF_DOMAIN = b"xuna.proof.v1\x00"


def b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def b64url_decode(encoded: str) -> bytes:
    clean = encoded.strip()
    rem = len(clean) % 4
    if rem == 2:
        clean += "=="
    elif rem == 3:
        clean += "="
    elif rem != 0:
        raise ValueError("invalid base64url length")
    return base64.urlsafe_b64decode(clean.encode("ascii"))


def public_key_fingerprint_b64url(public_key: bytes) -> str:
    digest = hashlib.sha256(public_key).digest()[:16]
    return b64url_encode(digest)


def sas(enrollment_code: bytes, public_key: bytes) -> str:
    digest = hashlib.sha256(_SAS_DOMAIN + enrollment_code + public_key).digest()
    return f"{int.from_bytes(digest[:4], 'big') % 1_000_000:06d}"


def host_proof_message(
    nonce: bytes,
    purpose: int,
    payload: bytes,
    host_id: uuid.UUID | None = None,
) -> bytes:
    host_bytes = bytes(16) if purpose == 1 else (host_id.bytes if host_id else bytes(16))
    return (
        _HOST_PROOF_DOMAIN
        + nonce
        + host_bytes
        + bytes([purpose])
        + hashlib.sha256(payload).digest()
    )


def pet_proof_message(
    nonce: bytes,
    pet_id_hex: str,
    purpose: int,
    generation: int,
    payload: bytes,
) -> bytes:
    return (
        _PET_PROOF_DOMAIN
        + nonce
        + bytes.fromhex(pet_id_hex)
        + struct.pack("<BI", purpose, generation)
        + hashlib.sha256(payload).digest()
    )


def _format_jcs_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("Non-finite float not allowed in canonical JSON")
    if value == 0.0:
        return "0"
    sign = "-" if value < 0.0 else ""
    s = repr(-value if value < 0.0 else value).lower()
    if "e" in s:
        mant, exp_str = s.split("e", 1)
        exp = int(exp_str)
    else:
        mant, exp = s, 0
    if "." in mant:
        int_part, frac_part = mant.split(".", 1)
        raw_digits = int_part + frac_part
        raw_exp = exp - len(frac_part)
    else:
        raw_digits = mant
        raw_exp = exp
    stripped_leading = raw_digits.lstrip("0")
    digits = stripped_leading.rstrip("0")
    tz = len(stripped_leading) - len(digits)
    actual_exp = raw_exp + tz
    k = len(digits)
    n = actual_exp + k
    if k <= n <= 21:
        body = digits + ("0" * (n - k))
    elif 0 < n <= 21:
        body = f"{digits[:n]}.{digits[n:]}"
    elif -6 < n <= 0:
        body = f"0.{'0' * (-n)}{digits}"
    else:
        mantissa = digits if k == 1 else f"{digits[0]}.{digits[1:]}"
        exp_val = n - 1
        exp_sign = "+" if exp_val >= 0 else "-"
        body = f"{mantissa}e{exp_sign}{abs(exp_val)}"
    return f"{sign}{body}"


def _write_canonical(parts: list[str], value: object) -> None:
    if value is None:
        parts.append("null")
    elif isinstance(value, bool):
        parts.append("true" if value else "false")
    elif isinstance(value, int):
        parts.append(str(value))
    elif isinstance(value, float):
        parts.append(_format_jcs_float(value))
    elif isinstance(value, str):
        parts.append(json.dumps(value, ensure_ascii=False))
    elif isinstance(value, list):
        parts.append("[")
        for idx, item in enumerate(value):
            if idx > 0:
                parts.append(",")
            _write_canonical(parts, item)
        parts.append("]")
    elif isinstance(value, dict):
        keys = sorted(str(k) for k in value)
        parts.append("{")
        for idx, k in enumerate(keys):
            if idx > 0:
                parts.append(",")
            parts.append(json.dumps(k, ensure_ascii=False))
            parts.append(":")
            _write_canonical(parts, value[k])
        parts.append("}")
    else:
        raise TypeError(f"Unsupported canonical JSON type: {type(value).__name__}")


def canonical_json(value: object) -> str:
    parts: list[str] = []
    _write_canonical(parts, value)
    return "".join(parts)


def args_sha256(arguments: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(arguments).encode("utf-8")).hexdigest()


def approval_proof_message(
    approval_id: str,
    args_sha256_hex: str,
    host_id: str,
    expires_at_iso: str,
) -> bytes:
    body = f"{approval_id}|{args_sha256_hex}|{host_id}|{expires_at_iso}".encode()
    return _APPROVAL_DOMAIN + body


def verify_golden_vectors(vectors_path: Path) -> None:
    """Self-verifies wire crypto helpers against `golden_vectors.json`."""
    data = json.loads(vectors_path.read_text(encoding="utf-8"))
    sas_vec = data["sas"]
    pub_bytes = bytes.fromhex(sas_vec["publicKeyHex"])
    code_bytes = bytes.fromhex(sas_vec["codeHex"])
    assert sas(code_bytes, pub_bytes) == sas_vec["sas"]
    assert public_key_fingerprint_b64url(pub_bytes) == sas_vec["keyFingerprintB64Url"]

    enroll = data["hostProof"]["enrollProof"]
    nonce_bytes = bytes.fromhex(data["hostProof"]["nonceHex"])
    enroll_msg = host_proof_message(nonce_bytes, 1, bytes.fromhex(enroll["payloadHex"]))
    assert enroll_msg.hex() == enroll["messageHex"]
    assert hashlib.sha256(enroll_msg).hexdigest() == enroll["messageSha256Hex"]

    session_vec = data["hostProof"]["sessionProof"]
    session_msg = host_proof_message(
        nonce_bytes,
        2,
        bytes.fromhex(session_vec["payloadHex"]),
        uuid.UUID(session_vec["hostIdUuid"]),
    )
    assert session_msg.hex() == session_vec["messageHex"]
    assert hashlib.sha256(session_msg).hexdigest() == session_vec["messageSha256Hex"]

    for vec in data["canonicalJson"]:
        assert canonical_json(vec["input"]) == vec["canonical"]
        assert args_sha256(vec["input"]) == vec["argsSha256"]


# ---------------------------------------------------------------------------
# Tool Descriptor Builders (Neutral Synthetic IDs — Zero Hardcoded App Names)
# ---------------------------------------------------------------------------


def make_tool_descriptor(
    name: str,
    *,
    family: str,
    risk: str = "R1",
    description: str = "Synthetic host tool for E2E verification.",
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    observes: bool = False,
    reversible: bool = True,
    permission: str = "granted",
    platform: str | None = None,
    idempotent: bool | None = None,
) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "version": 1,
        "family": family,
        "risk": risk,
        "observes": observes,
        "reversible": reversible,
        "permission": permission,
    }
    if platform is not None:
        policy["platform"] = platform
    if idempotent is not None:
        policy["idempotent"] = idempotent
    return {
        "name": name,
        "description": description,
        "inputSchema": input_schema
        or {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "outputSchema": output_schema
        or {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "additionalProperties": False,
        },
        "_meta": {"xuna/policy": policy},
    }


def make_standard_mesh_tools(
    *,
    platform: str | None = None,
    include_cu: bool = False,
    include_clipboard: bool = False,
    include_teleport: bool = False,
    include_vitals: bool = True,
    extra_tools: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = [
        make_tool_descriptor(
            "host.status",
            family="host",
            risk="R0",
            observes=False,
            reversible=True,
            platform=platform,
            output_schema={
                "type": "object",
                "properties": {"online": {"type": "boolean"}},
                "additionalProperties": False,
            },
        ),
        make_tool_descriptor(
            "media.play_pause",
            family="media",
            risk="R1",
            observes=False,
            reversible=True,
            platform=platform,
            idempotent=False,
            output_schema={
                "type": "object",
                "properties": {
                    "state": {"type": "string", "enum": ["playing", "paused", "unknown"]}
                },
                "required": ["state"],
                "additionalProperties": False,
            },
        ),
        make_tool_descriptor(
            "media.now_playing",
            family="media",
            risk="R0",
            observes=False,
            reversible=True,
            platform=platform,
            output_schema={
                "type": "object",
                "properties": {"playing": {"type": "boolean"}},
                "additionalProperties": False,
            },
        ),
        make_tool_descriptor(
            "app.list_allowed",
            family="app",
            risk="R0",
            observes=False,
            reversible=True,
            platform=platform,
            output_schema={
                "type": "object",
                "properties": {"apps": {"type": "array"}},
                "additionalProperties": False,
            },
        ),
        make_tool_descriptor(
            "app.launch",
            family="app",
            risk="R2",
            observes=False,
            reversible=False,
            platform=platform,
            input_schema={
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {"launched": {"type": "boolean"}, "id": {"type": "string"}},
                "additionalProperties": False,
            },
        ),
    ]
    if include_vitals:
        tools.append(
            make_tool_descriptor(
                "system.vitals",
                family="host",
                risk="R1",
                observes=True,
                reversible=True,
                platform=platform,
                output_schema={
                    "type": "object",
                    "properties": {"cpuLoad": {"type": "number"}},
                    "additionalProperties": False,
                },
            )
        )
    if include_clipboard:
        tools.extend(
            [
                make_tool_descriptor(
                    "selection.read",
                    family="app",
                    risk="R1",
                    observes=True,
                    reversible=True,
                    platform=platform,
                    input_schema={
                        "type": "object",
                        "properties": {
                            "maxChars": {"type": "integer", "minimum": 80, "maximum": 1200}
                        },
                        "additionalProperties": False,
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "maxLength": 1200},
                            "truncated": {"type": "boolean"},
                            "redactedCount": {"type": "integer"},
                        },
                        "additionalProperties": False,
                    },
                ),
                make_tool_descriptor(
                    "clipboard.write",
                    family="app",
                    risk="R2",
                    observes=False,
                    reversible=False,
                    platform=platform,
                    input_schema={
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "minLength": 1, "maxLength": 4000}
                        },
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                    output_schema={
                        "type": "object",
                        "properties": {"written": {"type": "boolean"}},
                        "additionalProperties": False,
                    },
                ),
            ]
        )
    if include_teleport:
        tools.extend(
            [
                make_tool_descriptor(
                    "browser.current_url",
                    family="url",
                    risk="R2",
                    observes=True,
                    reversible=True,
                    platform=platform,
                    output_schema={
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "additionalProperties": False,
                    },
                ),
                make_tool_descriptor(
                    "workspace.capture_capsule",
                    family="app",
                    risk="R1",
                    observes=True,
                    reversible=True,
                    platform=platform,
                    input_schema={
                        "type": "object",
                        "properties": {"project": {"type": "string"}},
                        "additionalProperties": False,
                    },
                    output_schema={
                        "type": "object",
                        "properties": {"project": {"type": "string"}},
                        "additionalProperties": False,
                    },
                ),
                make_tool_descriptor(
                    "workspace.git_summary",
                    family="app",
                    risk="R1",
                    observes=True,
                    reversible=True,
                    platform=platform,
                    input_schema={
                        "type": "object",
                        "properties": {"project": {"type": "string"}},
                        "additionalProperties": False,
                    },
                    output_schema={
                        "type": "object",
                        "properties": {"branch": {"type": "string"}, "dirty": {"type": "boolean"}},
                        "additionalProperties": False,
                    },
                ),
                make_tool_descriptor(
                    "url.open",
                    family="url",
                    risk="R2",
                    observes=False,
                    reversible=False,
                    platform=platform,
                    input_schema={
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                        "additionalProperties": False,
                    },
                    output_schema={
                        "type": "object",
                        "properties": {"opened": {"type": "boolean"}},
                        "additionalProperties": False,
                    },
                ),
            ]
        )
    if include_cu:
        tools.extend(
            [
                make_tool_descriptor(
                    "cu.session_start",
                    family="cu",
                    risk="R3",
                    observes=True,
                    reversible=False,
                    platform=platform,
                    input_schema={
                        "type": "object",
                        "properties": {
                            "goal": {"type": "string"},
                            "apps": {"type": "array", "items": {"type": "string"}},
                            "domains": {"type": "array", "items": {"type": "string"}},
                            "maxSteps": {"type": "integer"},
                            "maxMinutes": {"type": "integer"},
                        },
                        "required": ["goal"],
                        "additionalProperties": False,
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "sessionId": {"type": "string"},
                            "foregroundApp": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                ),
                make_tool_descriptor(
                    "cu.session_stop",
                    family="cu",
                    risk="R0",
                    observes=False,
                    reversible=True,
                    platform=platform,
                    input_schema={
                        "type": "object",
                        "properties": {"sessionId": {"type": "string"}},
                        "additionalProperties": False,
                    },
                    output_schema={
                        "type": "object",
                        "properties": {"stopped": {"type": "boolean"}},
                        "additionalProperties": False,
                    },
                ),
                make_tool_descriptor(
                    "input.click",
                    family="input",
                    risk="R3",
                    observes=False,
                    reversible=False,
                    platform=platform,
                    input_schema={
                        "type": "object",
                        "properties": {
                            "elementId": {"type": "string"},
                            "elementName": {"type": "string"},
                            "xNorm": {"type": "number"},
                            "yNorm": {"type": "number"},
                        },
                        "additionalProperties": False,
                    },
                    output_schema={
                        "type": "object",
                        "properties": {"clicked": {"type": "boolean"}},
                        "additionalProperties": False,
                    },
                ),
            ]
        )
    if extra_tools:
        tools.extend(extra_tools)
    return tools


# ---------------------------------------------------------------------------
# Cognition Provider Fakes
# ---------------------------------------------------------------------------


def done_event(latency_ms: int = 5) -> ChatFinished:
    return ChatFinished(
        provider="fake_mesh",
        model="scripted-v1",
        input_tokens=16,
        output_tokens=8,
        latency_ms=latency_ms,
    )


class ScriptedTextGenerator:
    """Deterministic `TextGenerator` recording every `ChatRequest` for opaque-box inspection."""

    def __init__(
        self,
        scripts: list[list[ChatEvent]] | None = None,
        *,
        responder: Callable[[ChatRequest, int], list[ChatEvent]] | None = None,
    ) -> None:
        self._scripts = list(scripts) if scripts is not None else []
        self._pushed_scripts: deque[list[ChatEvent]] = deque()
        self._responder = responder
        self.requests: list[ChatRequest] = []

    def push_script(self, events: list[ChatEvent]) -> None:
        if self._responder is not None:
            self._pushed_scripts.append(events)
        else:
            self._scripts.append(events)

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        idx = len(self.requests)
        self.requests.append(request)
        if self._pushed_scripts:
            events = self._pushed_scripts.popleft()
        elif self._responder is not None:
            events = self._responder(request, idx)
        elif idx < len(self._scripts):
            events = self._scripts[idx]
        else:
            events = [TextDelta("Done."), done_event()]
        for ev in events:
            yield ev


class FakeSpeechRecognizer:
    def __init__(self, transcript: str = "pause media") -> None:
        self.transcript = transcript

    async def transcribe(
        self, audio: AsyncIterator[AudioChunk], locale: Locale | None = None
    ) -> AsyncIterator[Transcript]:
        _ = [chunk async for chunk in audio]
        yield Transcript(self.transcript, final=True)


class FakeSpeechSynthesizer:
    async def synthesize(
        self,
        text: str,
        locale: Locale | None = None,
        voice: str | None = None,
    ) -> AsyncIterator[AudioChunk]:
        yield AudioChunk(data=b"FAKE_MP3_MESH", codec="mp3", sample_rate=24000)


# ---------------------------------------------------------------------------
# Simulated XTP v1 Host Agent (Real `websockets` over `xtp/1`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RpcErrorResponse:
    code: int
    message: str


class _XtpWebSocketClientProtocol(WebSocketClientProtocol):
    """Accepts the XTP v1 subprotocol (`xtp/1`) containing `/` without RFC token rejection."""

    def process_subprotocol(self, headers: Any, _available_subprotocols: Any) -> Any:
        return headers.get("Sec-WebSocket-Protocol")


class SimulatedXtpHost:
    """Real XTP v1 WebSocket host client speaking JSON-RPC 2.0 over `xtp/1`."""

    def __init__(
        self,
        *,
        platform: str = "macos",
        os_version: str = "15.0",
        agent_version: str = "0.2.0",
        hostname: str = "mesh-host",
        tools: list[dict[str, Any]] | None = None,
        tool_handler: Callable[[str, dict[str, Any], dict[str, Any]], Any] | None = None,
        preflight_handler: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.device = SimulatedDevice()
        self.platform = platform
        self.os_version = os_version
        self.agent_version = agent_version
        self.hostname = hostname
        self.tools: list[dict[str, Any]] = (
            list(tools) if tools is not None else make_standard_mesh_tools(platform=platform)
        )
        self._tool_handler = tool_handler
        self._preflight_handler = preflight_handler
        self.host_id: uuid.UUID | None = None
        self.alias: str | None = None
        self._ws: Any = None
        self._reader_task: asyncio.Task[None] | None = None
        self._next_id = 100
        self._pending_rpc: dict[int | str, asyncio.Future[dict[str, Any]]] = {}

        self.received_tool_calls: list[dict[str, Any]] = []
        self.received_stops: list[dict[str, Any]] = []
        self.received_preflights: list[dict[str, Any]] = []
        self.received_lan_peers: list[dict[str, Any]] = []
        self.received_frames: list[dict[str, Any]] = []
        self.close_code: int | None = None
        self.close_reason: str | None = None
        self.trust_preset: str = "standard"
        self._bg_tasks: set[asyncio.Task[None]] = set()

    def _allocate_id(self) -> int:
        self._next_id += 1
        return self._next_id

    async def connect_enroll(
        self,
        ws_url: str,
        owner_client: "SimulatedOwnerClient",
        alias: str,
        *,
        enrollment_code: bytes | None = None,
    ) -> uuid.UUID:
        code = enrollment_code or secrets.token_bytes(16)
        self.alias = alias
        ws = await ws_legacy_connect(
            f"{ws_url}/v1/hosts/connect",
            create_protocol=_XtpWebSocketClientProtocol,
            extra_headers={"Sec-WebSocket-Protocol": "xtp/1"},
        )
        self._ws = ws
        try:
            await ws.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "xuna/hello",
                        "params": {
                            "protocols": ["xtp/1"],
                            "enroll": {
                                "code": b64url_encode(code),
                                "publicKey": b64(self.device.public_key),
                                "platform": self.platform,
                                "osVersion": self.os_version,
                                "agentVersion": self.agent_version,
                                "hostname": self.hostname,
                            },
                        },
                    }
                )
            )
            hello_resp = json.loads(await ws.recv())
            assert hello_resp == {"jsonrpc": "2.0", "id": 1, "result": {"protocol": "xtp/1"}}

            challenge_1 = json.loads(await ws.recv())
            assert challenge_1["method"] == "xuna/challenge"
            assert challenge_1["params"]["purpose"] == 1
            nonce_1 = base64.b64decode(challenge_1["params"]["nonce"])
            sig_1 = self.device.sign(host_proof_message(nonce_1, 1, code))
            await ws.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "xuna/proof",
                        "params": {"purpose": 1, "signature": b64(sig_1)},
                    }
                )
            )
            proof_resp_1 = json.loads(await ws.recv())
            assert proof_resp_1 == {"jsonrpc": "2.0", "id": 2, "result": {"verified": True}}

            pending_note = json.loads(await ws.recv())
            assert pending_note["method"] == "xuna/enrollPending"

            claim = await owner_client.claim_host(
                b64url_encode(code),
                alias,
                key_fingerprint=public_key_fingerprint_b64url(self.device.public_key),
            )
            assert claim["sas"] == sas(code, self.device.public_key)
            await owner_client.confirm_host(claim["hostId"])

            enrolled_note = json.loads(await ws.recv())
            assert enrolled_note["method"] == "xuna/enrolled"
            self.host_id = uuid.UUID(enrolled_note["params"]["hostId"])

            await self._complete_session_handshake(ws, self.host_id)
            return self.host_id
        except BaseException:
            await self.disconnect()
            raise

    async def connect_reconnect(self, ws_url: str, *, host_id: uuid.UUID | None = None) -> None:
        target_id = host_id or self.host_id
        assert target_id is not None, "host_id required for reconnect"
        self.host_id = target_id
        ws = await ws_legacy_connect(
            f"{ws_url}/v1/hosts/connect",
            create_protocol=_XtpWebSocketClientProtocol,
            extra_headers={"Sec-WebSocket-Protocol": "xtp/1"},
        )
        self._ws = ws
        try:
            await ws.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "xuna/hello",
                        "params": {
                            "protocols": ["xtp/1"],
                            "hostId": str(target_id),
                            "platform": self.platform,
                            "osVersion": self.os_version,
                            "agentVersion": self.agent_version,
                        },
                    }
                )
            )
            hello_resp = json.loads(await ws.recv())
            assert hello_resp == {"jsonrpc": "2.0", "id": 1, "result": {"protocol": "xtp/1"}}
            await self._complete_session_handshake(ws, target_id)
        except BaseException:
            await self.disconnect()
            raise

    async def _complete_session_handshake(self, ws: Any, host_id: uuid.UUID) -> None:
        challenge_2 = json.loads(await ws.recv())
        assert challenge_2["method"] == "xuna/challenge"
        assert challenge_2["params"]["purpose"] == 2
        nonce_2 = base64.b64decode(challenge_2["params"]["nonce"])
        sig_2 = self.device.sign(host_proof_message(nonce_2, 2, self.device.public_key, host_id))
        await ws.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "xuna/proof",
                    "params": {"purpose": 2, "signature": b64(sig_2)},
                }
            )
        )
        proof_resp_2 = json.loads(await ws.recv())
        assert proof_resp_2 == {"jsonrpc": "2.0", "id": 3, "result": {"verified": True}}

        ready_msg = json.loads(await ws.recv())
        assert ready_msg["method"] == "xuna/ready"
        self.received_frames.append(ready_msg)

        # Serve initial tools/list (and optional xuna/lanPeer) before background reader starts.
        for _ in range(2):
            frame = json.loads(await ws.recv())
            self.received_frames.append(frame)
            if frame.get("method") == "tools/list":
                await ws.send(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": frame["id"],
                            "result": {"tools": self.tools},
                        }
                    )
                )
                break
            if frame.get("method") == "xuna/lanPeer":
                self.received_lan_peers.append(frame.get("params", {}))
            else:
                await self._dispatch_server_frame(frame)

        self._reader_task = asyncio.create_task(self._reader_loop())
        await self.ping()

    async def _reader_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    continue
                frame = json.loads(raw)
                self.received_frames.append(frame)
                await self._dispatch_server_frame(frame)
        except websockets.ConnectionClosed as closed:
            self.close_code = closed.code
            self.close_reason = closed.reason
        except asyncio.CancelledError:
            raise
        finally:
            for fut in self._pending_rpc.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("XTP WebSocket closed"))
            self._pending_rpc.clear()

    async def _dispatch_server_frame(self, frame: dict[str, Any]) -> None:
        assert self._ws is not None
        if "result" in frame or "error" in frame:
            req_id = frame.get("id")
            fut = self._pending_rpc.pop(req_id, None)
            if fut is not None and not fut.done():
                fut.set_result(frame)
            return

        method = frame.get("method")
        req_id = frame.get("id")
        params = frame.get("params") or {}

        if method == "tools/list":
            await self._ws.send(
                json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"tools": self.tools}})
            )
            return

        if method == "xuna/lanPeer":
            self.received_lan_peers.append(params)
            return

        if method == "xuna/stop":
            self.received_stops.append(params)
            return

        if method == "xuna/get_policy":
            await self._ws.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "schemaVersion": 1,
                            "trustPreset": self.trust_preset,
                            "allowApps": [],
                            "denyApps": [],
                            "allowDomains": [],
                            "denyDomains": [],
                            "discoveredApps": [],
                        },
                    }
                )
            )
            return

        if method == "xuna/preflight":
            tool = str(params.get("tool", ""))
            args = dict(params.get("arguments") or {})
            self.received_preflights.append({"tool": tool, "arguments": args})
            if self._preflight_handler is not None:
                preflight = self._preflight_handler(tool, args)
            else:
                risk = "R3"
                for td in self.tools:
                    if td["name"] == tool:
                        risk = td["_meta"]["xuna/policy"]["risk"]
                        break
                preflight = {
                    "status": "ok",
                    "effectiveRisk": risk,
                    "reasonCode": "ok",
                }
            await self._ws.send(
                json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"preflight": preflight}})
            )
            return

        if method == "tools/call":
            tool_name = str(params.get("name", ""))
            args = dict(params.get("arguments") or {})
            meta = dict(params.get("_meta") or {})
            self.received_tool_calls.append({"name": tool_name, "arguments": args, "meta": meta})
            # Run tool call handler in a task so concurrent calls/timeouts do not block reader.
            task = asyncio.create_task(self._execute_tool_call(req_id, tool_name, args, meta))
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)
            return

    async def _execute_tool_call(
        self,
        req_id: int | str,
        tool_name: str,
        args: dict[str, Any],
        meta: dict[str, Any],
    ) -> None:
        assert self._ws is not None
        try:
            if self._tool_handler is not None:
                res = self._tool_handler(tool_name, args, meta)
                if asyncio.iscoroutine(res):
                    res = await res
            else:
                res = self._default_tool_result(tool_name, args)
            if isinstance(res, RpcErrorResponse):
                payload: dict[str, Any] = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": res.code, "message": res.message},
                }
            elif "structuredContent" in res and "_meta" in res:
                payload = {"jsonrpc": "2.0", "id": req_id, "result": res}
            else:
                status = str(res.get("status", "ok"))
                structured = res.get("structuredContent", {"ok": status == "ok"})
                sc_text = json.dumps(structured, separators=(",", ":"))
                content_text = (
                    str(res["text"])
                    if "text" in res
                    else (status if len(sc_text.encode("utf-8")) > 16_384 else sc_text)
                )
                payload = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": content_text}],
                        "structuredContent": structured,
                        "isError": status != "ok",
                        "_meta": {
                            "xuna/status": status,
                            "xuna/durationMs": int(res.get("durationMs", 12)),
                            **(
                                {"xuna/errorCode": res["errorCode"]} if res.get("errorCode") else {}
                            ),
                        },
                    },
                }
            with contextlib.suppress(websockets.ConnectionClosed):
                await self._ws.send(json.dumps(payload, separators=(",", ":")))
        except Exception as exc:  # pragma: no cover - defensive simulator guard
            with contextlib.suppress(websockets.ConnectionClosed):
                await self._ws.send(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {"code": -32000, "message": type(exc).__name__},
                        }
                    )
                )

    @staticmethod
    def _default_tool_result(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "media.play_pause":
            structured: dict[str, Any] = {"state": "paused"}
        elif tool_name == "media.now_playing":
            structured = {"playing": False}
        elif tool_name == "app.launch":
            structured = {"launched": True, "id": str(args.get("id", "app_x"))}
        elif tool_name == "app.list_allowed":
            structured = {"apps": []}
        elif tool_name == "cu.session_start":
            structured = {"sessionId": f"cu-{uuid.uuid4().hex[:8]}"}
        elif tool_name == "cu.session_stop":
            structured = {"stopped": True}
        elif tool_name == "system.vitals":
            structured = {"cpuLoad": 0.25}
        elif tool_name == "url.open":
            structured = {"opened": True}
        elif tool_name == "clipboard.write":
            structured = {"written": True}
        else:
            structured = {"ok": True}
        return {
            "content": [{"type": "text", "text": json.dumps(structured)}],
            "structuredContent": structured,
            "isError": False,
            "_meta": {"xuna/status": "ok", "xuna/durationMs": 10},
        }

    async def ping(self) -> dict[str, Any]:
        assert self._ws is not None
        req_id = self._allocate_id()
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending_rpc[req_id] = fut
        await self._ws.send(
            json.dumps({"jsonrpc": "2.0", "id": req_id, "method": "ping", "params": {}})
        )
        return await asyncio.wait_for(fut, timeout=5.0)

    async def send_presence(
        self,
        *,
        idle_seconds: int = 0,
        screen_locked: bool = False,
        media_playing: bool = False,
        battery_percent: int | None = None,
        is_charging: bool | None = None,
        include_battery: bool = True,
        subnet_hash: str | None = None,
        foreground_app_id: str | None = None,
        updated_at: str | None = None,
        sync: bool = True,
    ) -> None:
        assert self._ws is not None
        params: dict[str, Any] = {
            "schemaVersion": 1,
            "idleSeconds": idle_seconds,
            "screenLocked": screen_locked,
            "mediaPlaying": media_playing,
            "updatedAt": updated_at or datetime.now(UTC).isoformat(),
        }
        if include_battery:
            if battery_percent is not None:
                params["batteryPercent"] = battery_percent
            if is_charging is not None:
                params["isCharging"] = is_charging
        if subnet_hash is not None:
            params["subnetHash"] = subnet_hash
        if foreground_app_id is not None:
            params["foregroundAppId"] = foreground_app_id
        await self._ws.send(
            json.dumps({"jsonrpc": "2.0", "method": "xuna/presence", "params": params})
        )
        if sync:
            await self.ping()

    async def send_resources(
        self,
        *,
        installed_apps: list[str],
        workspaces: list[str],
        capabilities_version: str | None = None,
        sync: bool = True,
    ) -> None:
        assert self._ws is not None
        params: dict[str, Any] = {
            "schemaVersion": 1,
            "installedApps": installed_apps,
            "workspaces": workspaces,
        }
        if capabilities_version is not None:
            params["capabilitiesVersion"] = capabilities_version
        await self._ws.send(
            json.dumps({"jsonrpc": "2.0", "method": "xuna/resources", "params": params})
        )
        if sync:
            await self.ping()

    async def send_focus_request(
        self,
        *,
        source: str = "hotkey",
        follow_me: bool | None = None,
        sync: bool = True,
    ) -> None:
        assert self._ws is not None
        params: dict[str, Any] = {"schemaVersion": 1, "source": source}
        if follow_me is not None:
            params["followMe"] = follow_me
        await self._ws.send(
            json.dumps({"jsonrpc": "2.0", "method": "xuna/focusRequest", "params": params})
        )
        if sync:
            await self.ping()

    async def update_tools(self, tools: list[dict[str, Any]]) -> None:
        assert self._ws is not None
        self.tools = list(tools)
        await self._ws.send(
            json.dumps(
                {"jsonrpc": "2.0", "method": "notifications/tools/list_changed", "params": {}}
            )
        )
        await self.ping()

    async def send_raw_json(self, payload: dict[str, Any], *, sync: bool = False) -> None:
        assert self._ws is not None
        await self._ws.send(json.dumps(payload))
        if sync:
            await self.ping()

    async def send_raw_text(self, text: str) -> None:
        assert self._ws is not None
        await self._ws.send(text)

    async def send_raw_bytes(self, data: bytes) -> None:
        assert self._ws is not None
        await self._ws.send(data)

    async def wait_closed(self, wait_s: float = 3.0) -> tuple[int | None, str | None]:
        if self._reader_task is not None:
            with contextlib.suppress(TimeoutError, websockets.ConnectionClosed):
                await asyncio.wait_for(self._reader_task, timeout=wait_s)
        if self._ws is not None and self.close_code is None:
            self.close_code = getattr(self._ws, "close_code", None)
            self.close_reason = getattr(self._ws, "close_reason", None)
        return self.close_code, self.close_reason

    async def disconnect(self) -> None:
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            with contextlib.suppress(BaseException):
                await self._reader_task
        self._reader_task = None
        self._ws = None
        # Allow server finally-block to finish marking host offline in PostgreSQL.
        await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# Simulated Owner HTTP + SSE + Voice WS Client
# ---------------------------------------------------------------------------


@dataclass
class VoiceWsSession:
    ws: Any

    async def send_turn_commit(
        self,
        text: str,
        *,
        target_host_id: str | None = None,
        follow_me: bool | None = None,
        client_subnet_hash: str | None = None,
        client_capabilities: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "type": "turn.commit",
            "text": text,
            "context": dict(context or {}),
        }
        if client_capabilities is not None:
            payload["clientCapabilities"] = client_capabilities
        if target_host_id is not None:
            payload["targetHostId"] = target_host_id
        if follow_me is not None:
            payload["followMe"] = follow_me
        if client_subnet_hash is not None:
            payload["clientSubnetHash"] = client_subnet_hash
        await self.ws.send(json.dumps(payload))

    async def send_stop(self, *, host_id: str | None = None, reason: str = "owner_app") -> None:
        payload: dict[str, Any] = {"type": "action.stop", "reason": reason}
        if host_id is not None:
            payload["hostId"] = host_id
        await self.ws.send(json.dumps(payload))

    async def send_raw_json(self, payload: dict[str, Any]) -> None:
        await self.ws.send(json.dumps(payload))

    async def receive_until_turn_done(
        self,
        *,
        on_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        wait_s: float = 15.0,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        deadline = asyncio.get_running_loop().time() + wait_s
        while True:
            remaining = max(0.1, deadline - asyncio.get_running_loop().time())
            raw = await asyncio.wait_for(self.ws.recv(), timeout=remaining)
            if isinstance(raw, bytes):
                continue
            ev = json.loads(raw)
            events.append(ev)
            if on_event is not None:
                await on_event(ev)
            if ev.get("type") in ("reply.finished", "error"):
                return events

    async def recv_event(self, wait_s: float = 5.0) -> dict[str, Any]:
        raw = await asyncio.wait_for(self.ws.recv(), timeout=wait_s)
        if isinstance(raw, bytes):
            return {"type": "__bytes__", "len": len(raw)}
        return dict(json.loads(raw))

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            await self.ws.send(json.dumps({"type": "bye"}))
        with contextlib.suppress(Exception):
            await self.ws.close()


@dataclass
class SseTurnOutcome:
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def of_type(self, event_name: str) -> list[dict[str, Any]]:
        return [data for name, data in self.events if name == event_name]

    def first(self, event_name: str) -> dict[str, Any] | None:
        for name, data in self.events:
            if name == event_name:
                return data
        return None

    def require(self, event_name: str) -> dict[str, Any]:
        ev = self.first(event_name)
        assert ev is not None, f"Expected SSE event {event_name!r}, got {self.events}"
        return ev


class SimulatedOwnerClient:
    """Real HTTP/SSE/Voice-WS owner client backed by `LocalIssuer` JWT & P-256 phone key."""

    def __init__(
        self,
        *,
        base_url: str,
        ws_url: str,
        issuer: LocalIssuer,
        subject: str | None = None,
    ) -> None:
        self.base_url = base_url
        self.ws_url = ws_url
        self.subject = subject or f"mesh-owner-{uuid.uuid4().hex[:12]}"
        self.jwt = issuer.token(self.subject)
        self.phone_device = SimulatedDevice()
        self.http = httpx.AsyncClient(base_url=base_url, timeout=20.0)

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.jwt}"}

    async def aclose(self) -> None:
        await self.http.aclose()

    async def bind_pet(self, pet_id: str | None = None, generation: int = 1) -> str:
        resolved_pet = pet_id or uuid.uuid4().hex
        board_device = SimulatedDevice()
        ch_resp = await self.http.post(
            "/v1/devices/challenges",
            headers={**self.headers, **idem()},
            json={"schemaVersion": 1, "petId": resolved_pet},
        )
        assert ch_resp.status_code == 201, ch_resp.text
        ch = ch_resp.json()
        msg = pet_proof_message(
            base64.b64decode(ch["nonce"]),
            resolved_pet,
            1,
            generation,
            board_device.public_key,
        )
        bind_resp = await self.http.post(
            "/v1/pets",
            headers={**self.headers, **idem()},
            json={
                "schemaVersion": 1,
                "petId": resolved_pet,
                "origin": "migrated_from_v1",
                "snapshotGeneration": generation,
                "deviceProof": {
                    "challengeId": ch["challengeId"],
                    "publicKey": b64(board_device.public_key),
                    "signature": b64(board_device.sign(msg)),
                },
            },
        )
        assert bind_resp.status_code in (200, 201), bind_resp.text
        return resolved_pet

    async def enable_consents(
        self,
        *,
        conversation_text: bool = True,
        conversation_voice: bool = True,
        host_screen: bool = True,
        host_notifications: bool = True,
    ) -> dict[str, Any]:
        cur = await self.http.get("/v1/consents", headers=self.headers)
        assert cur.status_code == 200, cur.text
        version = int(cur.json()["version"])
        put_resp = await self.http.put(
            "/v1/consents",
            headers={**self.headers, **idem()},
            json={
                "schemaVersion": 1,
                "aggregateMetrics": False,
                "experienceEvents": False,
                "conversationText": conversation_text,
                "conversationVoice": conversation_voice,
                "ownerDocuments": False,
                "hostScreen": host_screen,
                "hostNotifications": host_notifications,
                "expectedVersion": version,
            },
        )
        assert put_resp.status_code == 200, put_resp.text
        return dict(put_resp.json())

    async def self_enroll_mobile(
        self,
        *,
        alias: str = "Phone",
        platform: str = "android",
        tools: list[dict[str, Any]] | None = None,
    ) -> httpx.Response:
        ch = await self.http.post("/v1/hosts/self/challenge", headers=self.headers)
        assert ch.status_code == 200, ch.text
        ch_json = ch.json()
        nonce = b64url_decode(ch_json["nonce"])
        code = b64url_decode(ch_json["enrollmentCode"])
        sig = self.phone_device.sign(host_proof_message(nonce, 1, code))
        body: dict[str, Any] = {
            "schemaVersion": 1,
            "platform": platform,
            "osVersion": "15.0",
            "agentVersion": "0.2.0",
            "publicKey": b64url_encode(self.phone_device.public_key),
            "alias": alias,
            "proof": b64url_encode(sig),
            "hostname": "phone-device",
        }
        if tools is not None:
            body["tools"] = tools
        return await self.http.post(
            "/v1/hosts/self",
            headers={**self.headers, **idem()},
            json=body,
        )

    async def claim_host(
        self,
        enrollment_code: str,
        alias: str,
        *,
        key_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schemaVersion": 1,
            "enrollmentCode": enrollment_code,
            "alias": alias,
        }
        if key_fingerprint is not None:
            payload["keyFingerprint"] = key_fingerprint
        resp = await self.http.post("/v1/hosts", headers={**self.headers, **idem()}, json=payload)
        assert resp.status_code == 202, resp.text
        return dict(resp.json())

    async def confirm_host(self, host_id: str) -> dict[str, Any]:
        resp = await self.http.post(
            f"/v1/hosts/{host_id}/confirm",
            headers={**self.headers, **idem()},
            json={"schemaVersion": 1},
        )
        assert resp.status_code == 200, resp.text
        return dict(resp.json())

    async def patch_host(
        self,
        host_id: str,
        body: dict[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> httpx.Response:
        headers = dict(self.headers)
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        else:
            headers.update(idem())
        return await self.http.patch(f"/v1/hosts/{host_id}", headers=headers, json=body)

    async def list_hosts(self) -> list[dict[str, Any]]:
        resp = await self.http.get("/v1/hosts", headers=self.headers)
        assert resp.status_code == 200, resp.text
        return list(resp.json()["items"])

    async def revoke_host(self, host_id: str) -> httpx.Response:
        return await self.http.delete(
            f"/v1/hosts/{host_id}",
            headers={**self.headers, **idem()},
        )

    async def set_grants(self, host_id: str, grants: list[dict[str, Any]]) -> dict[str, Any]:
        resp = await self.http.put(
            f"/v1/hosts/{host_id}/grants",
            headers={**self.headers, **idem()},
            json={"schemaVersion": 1, "grants": grants},
        )
        assert resp.status_code == 200, resp.text
        return dict(resp.json())

    async def grant_all_standard(self, host_id: str, *, max_risk: str = "R3") -> dict[str, Any]:
        return await self.set_grants(
            host_id,
            [
                {"family": fam, "maxRisk": max_risk, "enabled": True}
                for fam in ("host", "media", "app", "url", "workflow", "cu", "ui", "input", "files")
            ],
        )

    async def set_focus(
        self,
        *,
        host_id: str | None = None,
        follow_me: bool | None = None,
        pet_id: str | None = None,
    ) -> httpx.Response:
        body: dict[str, Any] = {"schemaVersion": 1, "hostId": host_id}
        if follow_me is not None:
            body["followMe"] = follow_me
        if pet_id is not None:
            body["petId"] = pet_id
        return await self.http.post("/v1/actions/focus", headers=self.headers, json=body)

    async def list_habits(self, *, pet_id: str | None = None) -> httpx.Response:
        params = {"petId": pet_id} if pet_id is not None else None
        return await self.http.get("/v1/actions/habits", headers=self.headers, params=params)

    async def delete_habit(self, habit_id: str) -> httpx.Response:
        return await self.http.delete(f"/v1/actions/habits/{habit_id}", headers=self.headers)

    async def clear_habits(self, *, pet_id: str | None = None) -> httpx.Response:
        params = {"petId": pet_id} if pet_id is not None else None
        return await self.http.delete("/v1/actions/habits", headers=self.headers, params=params)

    async def emergency_stop(
        self,
        *,
        host_id: str | None = None,
        reason: str = "owner_app",
        include_host_id_key: bool = True,
    ) -> httpx.Response:
        body: dict[str, Any] = {"schemaVersion": 1, "reason": reason}
        if include_host_id_key:
            body["hostId"] = host_id
        return await self.http.post("/v1/actions/stop", headers=self.headers, json=body)

    async def decide_approval(
        self,
        approval_id: str,
        decision: str = "approve",
        *,
        scope: str = "once",
        host_id: str | None = None,
        args_sha256_hex: str | None = None,
        arguments: dict[str, Any] | None = None,
        expires_at: str | None = None,
        sign_r3: bool = False,
    ) -> httpx.Response:
        body: dict[str, Any] = {
            "schemaVersion": 1,
            "decision": decision,
            "scope": scope,
        }
        if sign_r3:
            assert host_id is not None and expires_at is not None
            resolved_hash = args_sha256_hex or args_sha256(arguments or {})
            msg = approval_proof_message(approval_id, resolved_hash, host_id, expires_at)
            sig = self.phone_device.sign(msg)
            body["signature"] = b64(sig)
            body["publicKey"] = b64(self.phone_device.public_key)
        return await self.http.post(
            f"/v1/actions/approvals/{approval_id}",
            headers=self.headers,
            json=body,
        )

    async def invoke_tool_dev(
        self,
        host_id: str,
        tool: str,
        arguments: dict[str, Any] | None = None,
        *,
        approve: bool = True,
    ) -> httpx.Response:
        return await self.http.post(
            f"/v1/hosts/{host_id}/tools:invoke",
            headers={**self.headers, **idem()},
            json={
                "schemaVersion": 1,
                "tool": tool,
                "arguments": dict(arguments or {}),
                "approve": approve,
            },
        )

    async def create_job(
        self,
        *,
        pet_id: str,
        goal: str,
        host_id: str | None = None,
        max_steps: int = 10,
        sub_goals: list[dict[str, Any]] | None = None,
    ) -> httpx.Response:
        body: dict[str, Any] = {
            "schemaVersion": 1,
            "petId": pet_id,
            "goal": goal,
            "maxSteps": max_steps,
        }
        if host_id is not None:
            body["hostId"] = host_id
        if sub_goals is not None:
            body["subGoals"] = sub_goals
        return await self.http.post("/v1/actions/jobs", headers=self.headers, json=body)

    async def get_job(self, job_id: str) -> httpx.Response:
        return await self.http.get(f"/v1/actions/jobs/{job_id}", headers=self.headers)

    async def audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
        resp = await self.http.get(
            "/v1/audit-events", headers=self.headers, params={"limit": limit}
        )
        assert resp.status_code == 200, resp.text
        return list(resp.json()["items"])

    async def post_turn_raw(self, pet_id: str, body: dict[str, Any]) -> httpx.Response:
        return await self.http.post(
            f"/v1/conversations/{pet_id}/turns",
            headers=self.headers,
            json=body,
        )

    async def stream_turn(
        self,
        pet_id: str,
        text: str,
        *,
        target_host_id: str | None = None,
        follow_me: bool | None = None,
        client_subnet_hash: str | None = None,
        client_capabilities: tuple[str, ...] = ("host_actions_v1",),
        self_host_id: str | None = None,
        locale: str | None = None,
        on_event: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> SseTurnOutcome:
        body: dict[str, Any] = {
            "schemaVersion": 1,
            "text": text,
            "clientCapabilities": list(client_capabilities),
        }
        if target_host_id is not None:
            body["targetHostId"] = target_host_id
        if follow_me is not None:
            body["followMe"] = follow_me
        if client_subnet_hash is not None:
            body["clientSubnetHash"] = client_subnet_hash
        if self_host_id is not None:
            body["selfHostId"] = self_host_id
        if locale is not None:
            body["locale"] = locale

        outcome = SseTurnOutcome()
        async with self.http.stream(
            "POST",
            f"/v1/conversations/{pet_id}/turns",
            headers=self.headers,
            json=body,
            timeout=30.0,
        ) as resp:
            assert resp.status_code == 200, await resp.aread()
            current_event = "message"
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event: "):
                    current_event = line[7:].strip()
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
                    outcome.events.append((current_event, data))
                    if on_event is not None:
                        await on_event(current_event, data)
        return outcome

    async def connect_voice_ws(
        self,
        pet_id: str,
        *,
        target_host_id: str | None = None,
        follow_me: bool | None = None,
        client_subnet_hash: str | None = None,
        client_capabilities: tuple[str, ...] = ("host_actions_v1",),
        speak: bool = False,
    ) -> VoiceWsSession:
        ws = await websockets.connect(f"{self.ws_url}/v1/voice")
        hello: dict[str, Any] = {
            "type": "hello",
            "schemaVersion": 1,
            "auth": {"token": self.jwt},
            "petId": pet_id,
            "speak": speak,
            "clientCapabilities": list(client_capabilities),
        }
        if target_host_id is not None:
            hello["targetHostId"] = target_host_id
        if follow_me is not None:
            hello["followMe"] = follow_me
        if client_subnet_hash is not None:
            hello["clientSubnetHash"] = client_subnet_hash
        await ws.send(json.dumps(hello))
        hello_ok = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
        assert hello_ok["type"] == "hello.ok", hello_ok
        return VoiceWsSession(ws=ws)


def load_mesh_frames_fixture() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "xtp" / "mesh_frames.json"
    return dict(json.loads(path.read_text(encoding="utf-8")))
