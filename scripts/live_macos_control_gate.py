"""Opaque-box live macOS control gate for UDC-20 M5 (`XUNA_DEV_HOST_TOOLS=1`, dev only).

Runs 6 mandatory verification gates against a live backend and signed `XUNA.app` host
(defaulting to `Xuans MacBook Pro`, `372e3cdc-4fcd-484d-9d07-0520a32558b0`), verifying
real OS end-state independently (`false_success == 0`) per XUNA Honest-Code Rules H1-H10
and ADR-0015.

Usage:
    XUNA_OWNER_TOKEN=<jwt> .venv/bin/python scripts/live_macos_control_gate.py --all --json
    .venv/bin/python scripts/live_macos_control_gate.py --dry-run --json
"""

import argparse
import asyncio
import base64
import json
import os
import statistics
import sys
import time
import unicodedata
import uuid
from collections.abc import Iterable, Iterator, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature,
    encode_dss_signature,
)

from xuna_backend.modules.audit.domain.event import AuditAction
from xuna_backend.modules.hosts.domain.crypto import (
    HostProofPurpose,
    approval_proof_message,
    args_sha256,
    host_proof_message,
)

DEFAULT_HOST_ID = "372e3cdc-4fcd-484d-9d07-0520a32558b0"
DEFAULT_HOST_ALIAS = "Xuans MacBook Pro"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
ALL_GATES: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
FORBIDDEN_TOOL_NAME = "ui.accessibility_tree"
GATE1_TOTAL_CALLS = 50
GATE1_BUDGET_MS = 500.0
CU_SESSION_START_KEYS: frozenset[str] = frozenset(
    {"goal", "apps", "domains", "maxSteps", "maxMinutes"}
)
HARD_DENIED_TERMINALS: frozenset[str] = frozenset(
    {"terminal", "iterm", "iterm2", "com.apple.terminal", "com.googlecode.iterm2"}
)
EDITABLE_DRAFT_ROLES: frozenset[str] = frozenset(
    {"textfield", "searchfield", "searchbox", "textarea"}
)
CONSEQUENTIAL_REF_WORDS: tuple[str, ...] = (
    "close",
    "quit",
    "delete",
    "remove",
    "send",
    "share",
    "lock",
    "zoom",
    "minimize",
    "full screen",
    "fullscreen",
    "thu nhỏ",
    "phóng to",
    "toàn màn hình",
    "xóa",
    "xoá",
    "gửi",
    "đóng",
)
PASSWORD_SECRET_LEAK_KEYS: tuple[str, ...] = (
    "text",
    "title",
    "label",
    "placeholder",
    "hint",
)
GATE_NAMES: dict[int, str] = {
    1: "stress_and_crash",
    2: "headless_first_and_continue_in_place",
    3: "os_project_launch",
    4: "complex_electron_search_and_play",
    5: "native_desktop_apps",
    6: "adr0015_security_and_trust_envelope",
}


class GateError(RuntimeError):
    """Raised when a live gate precondition or transport contract fails."""


@dataclass(frozen=True, slots=True)
class ToolCallOutcome:
    http_status: int
    tool: str
    status: str
    is_error: bool
    error_code: str | None
    structured_content: dict[str, Any]
    text: str
    duration_ms: float
    host_duration_ms: float | None
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PreflightOutcome:
    http_status: int
    tool: str
    status: str
    effective_risk: str
    reason_code: str | None
    error_code: str | None
    element_role: str | None
    element_name: str | None
    duration_ms: float
    raw: dict[str, Any]


@dataclass(slots=True)
class TurnOutcome:
    reply: str = ""
    owner_prompts: int = 0
    action_results: list[dict[str, Any]] = field(default_factory=list)
    cu_session_events: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GateCheck:
    name: str
    passed: bool
    detail: str
    false_success: bool = False


@dataclass(slots=True)
class GateResult:
    gate: int
    name: str
    passed: bool
    false_success: int = 0
    crashes: int = 0
    timeouts: int = 0
    duration_ms: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    checks: list[GateCheck] = field(default_factory=list)
    error: str | None = None


@dataclass(slots=True)
class PreflightCheckResult:
    ok: bool
    host_online: bool
    accessibility: str
    screen_recording: str
    accessibility_tree_exposed: bool
    dev_preflight_available: bool
    detail: str


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(encoded: str) -> bytes:
    clean = encoded.strip()
    rem = len(clean) % 4
    if rem == 2:
        clean += "=="
    elif rem == 3:
        clean += "="
    elif rem != 0:
        raise ValueError("invalid base64url length")
    return base64.urlsafe_b64decode(clean.encode("ascii"))


def normalize_vi(text: str) -> str:
    """Lowercases and strips Vietnamese diacritics for track/title comparison."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.replace("đ", "d").replace("Đ", "D").lower().strip()


def host_tool_names(host: dict[str, Any]) -> set[str]:
    """Extracts wire tool names advertised by a host in `GET /v1/hosts`."""
    names: set[str] = set()
    tools = host.get("tools")
    if isinstance(tools, list):
        for item in tools:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                names.add(item["name"])
    return names


def host_tool_permission(host: dict[str, Any], tool_names: Sequence[str]) -> str | None:
    """Extracts the declared `permission` state for the first matching tool in `GET /v1/hosts`."""
    tools = host.get("tools")
    if not isinstance(tools, list):
        return None
    by_name: dict[str, dict[str, Any]] = {}
    for item in tools:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            by_name[item["name"]] = item
    for name in tool_names:
        tool_obj = by_name.get(name)
        if tool_obj is None:
            continue
        direct = tool_obj.get("permission")
        if isinstance(direct, str) and direct:
            return direct
        meta = tool_obj.get("_meta")
        if isinstance(meta, dict):
            policy = meta.get("xuna/policy")
            if isinstance(policy, dict):
                nested_perm = policy.get("permission")
                if isinstance(nested_perm, str) and nested_perm:
                    return nested_perm
    return None


def validate_cu_session_start_args(arguments: dict[str, Any]) -> None:
    """Validates exact 5-key XTP contract for `cu.session_start`."""
    if set(arguments.keys()) != CU_SESSION_START_KEYS:
        raise ValueError(
            f"cu.session_start must contain exact keys {sorted(CU_SESSION_START_KEYS)}"
        )
    goal = arguments.get("goal")
    apps = arguments.get("apps")
    domains = arguments.get("domains")
    max_steps = arguments.get("maxSteps")
    max_minutes = arguments.get("maxMinutes")
    if not isinstance(goal, str) or not goal.strip() or len(goal) > 200:
        raise ValueError("goal must be a non-empty string <= 200 chars")
    if not isinstance(apps, list) or not all(isinstance(a, str) and a for a in apps):
        raise ValueError("apps must be a list of non-empty strings")
    if not isinstance(domains, list) or not all(isinstance(d, str) for d in domains):
        raise ValueError("domains must be a list of strings")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not (1 <= max_steps <= 40):
        raise ValueError("maxSteps must be an int in 1..40")
    if (
        isinstance(max_minutes, bool)
        or not isinstance(max_minutes, int)
        or not (1 <= max_minutes <= 20)
    ):
        raise ValueError("maxMinutes must be an int in 1..20")


def validate_jpeg_som_payload(
    structured: dict[str, Any],
    *,
    require_screenshot: bool = False,
) -> tuple[bool, str]:
    """Validates normalized `nodes[*].bounds` and JPEG screenshot header/trailer."""
    nodes = structured.get("nodes")
    if isinstance(nodes, list):
        for idx, node in enumerate(nodes):
            if not isinstance(node, dict):
                return False, f"node[{idx}] is not an object"
            bounds = node.get("bounds")
            if bounds is not None:
                if not isinstance(bounds, list) or len(bounds) != 4:
                    return False, f"node[{idx}].bounds must be a 4-element list"
                for val in bounds:
                    if isinstance(val, bool) or not isinstance(val, int | float):
                        return False, f"node[{idx}].bounds contains non-numeric entry"
                    if not (0.0 <= val <= 1.0):
                        return False, f"node[{idx}].bounds value {val} out of [0, 1]"

    shot = structured.get("screenshot")
    if not isinstance(shot, dict):
        shot = structured.get("image") if isinstance(structured.get("image"), dict) else None
    if shot is None:
        if require_screenshot:
            return False, "screenshot block missing"
        return True, "no screenshot block attached; bounds valid"

    mime = shot.get("mimeType")
    if mime != "image/jpeg":
        return False, f"expected mimeType='image/jpeg', got {mime!r}"
    width = shot.get("width")
    height = shot.get("height")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        return False, f"invalid screenshot dimensions width={width!r} height={height!r}"
    data_b64 = shot.get("dataBase64")
    if not isinstance(data_b64, str) or not data_b64:
        return False, "screenshot dataBase64 missing"
    try:
        raw = base64.b64decode(data_b64, validate=True)
    except Exception as error:
        return False, f"invalid base64 screenshot: {error}"
    if len(raw) < 4 or not raw.startswith(b"\xff\xd8") or not raw.endswith(b"\xff\xd9"):
        return False, "screenshot bytes are not a valid JPEG (missing SOI/EOI markers)"
    return True, f"valid JPEG ({len(raw)} bytes, {width}x{height})"


def iter_sse_lines(lines: Iterable[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Parses `event:` / `data:` SSE frames into `(event, payload)` pairs."""
    event = "message"
    data_parts: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r")
        if not line:
            if data_parts:
                try:
                    parsed = json.loads("\n".join(data_parts))
                except ValueError:
                    parsed = {}
                yield event, parsed if isinstance(parsed, dict) else {}
            event = "message"
            data_parts = []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_parts.append(line[5:].lstrip())
    if data_parts:
        try:
            parsed = json.loads("\n".join(data_parts))
        except ValueError:
            parsed = {}
        yield event, parsed if isinstance(parsed, dict) else {}


class DevPhoneSigner:
    """Ephemeral ECDSA P-256 phone key matching XTP v1 & `xuna_backend.hosts.domain.crypto`."""

    def __init__(self, private_key: ec.EllipticCurvePrivateKey | None = None) -> None:
        self._key = private_key or ec.generate_private_key(ec.SECP256R1())
        self.public_key_bytes = self._key.public_key().public_bytes(
            serialization.Encoding.X962,
            serialization.PublicFormat.UncompressedPoint,
        )

    @property
    def public_key_b64(self) -> str:
        return base64.b64encode(self.public_key_bytes).decode("ascii")

    @property
    def public_key_b64url(self) -> str:
        return _b64url_encode(self.public_key_bytes)

    def sign_raw(self, message: bytes) -> bytes:
        der = self._key.sign(message, ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(der)
        return r.to_bytes(32, "big") + s.to_bytes(32, "big")

    def verify_raw(self, message: bytes, raw_signature: bytes) -> bool:
        if len(raw_signature) != 64:
            return False
        r = int.from_bytes(raw_signature[:32], "big")
        s = int.from_bytes(raw_signature[32:], "big")
        try:
            self._key.public_key().verify(
                encode_dss_signature(r, s),
                message,
                ec.ECDSA(hashes.SHA256()),
            )
        except Exception:
            return False
        return True

    def sign_r3_approval_hash(
        self,
        approval_id: str,
        args_sha256_hex: str,
        host_id: str,
        expires_at_iso: str,
    ) -> tuple[str, str]:
        msg = approval_proof_message(approval_id, args_sha256_hex, host_id, expires_at_iso)
        sig_b64 = base64.b64encode(self.sign_raw(msg)).decode("ascii")
        return sig_b64, self.public_key_b64

    def sign_r3_approval(
        self,
        approval_id: str,
        arguments: dict[str, Any],
        host_id: str,
        expires_at_iso: str,
    ) -> dict[str, str]:
        digest = args_sha256(arguments)
        sig_b64, pub_b64 = self.sign_r3_approval_hash(approval_id, digest, host_id, expires_at_iso)
        return {
            "approvalId": approval_id,
            "expiresAt": expires_at_iso,
            "scope": "once",
            "signature": sig_b64,
            "publicKey": pub_b64,
        }

    def build_self_enroll_payload(
        self,
        nonce_b64url: str,
        enrollment_code_b64url: str,
        *,
        alias: str = "xuna-e2e-gate-phone",
    ) -> dict[str, Any]:
        nonce = _b64url_decode(nonce_b64url)
        code = _b64url_decode(enrollment_code_b64url)
        msg = host_proof_message(nonce=nonce, purpose=HostProofPurpose.ENROLL, payload=code)
        proof_b64url = _b64url_encode(self.sign_raw(msg))
        return {
            "schemaVersion": 1,
            "platform": "ios",
            "osVersion": "18.0",
            "agentVersion": "1.0.0",
            "publicKey": self.public_key_b64url,
            "alias": alias,
            "proof": proof_b64url,
        }

    def run_self_test(self) -> dict[str, Any]:
        sample_args: dict[str, Any] = {
            "goal": "Self-test",
            "apps": ["notes"],
            "domains": [],
            "maxSteps": 10,
            "maxMinutes": 5,
        }
        validate_cu_session_start_args(sample_args)
        approval_id = "00000000-0000-4000-8000-000000000001"
        expires_at = "2026-11-19T15:00:00+00:00"
        bundle = self.sign_r3_approval(approval_id, sample_args, DEFAULT_HOST_ID, expires_at)
        approval_msg = approval_proof_message(
            approval_id, args_sha256(sample_args), DEFAULT_HOST_ID, expires_at
        )
        approval_ok = self.verify_raw(approval_msg, base64.b64decode(bundle["signature"]))
        enroll_msg = host_proof_message(
            nonce=b"\x01" * 32,
            purpose=HostProofPurpose.ENROLL,
            payload=b"\x02" * 16,
        )
        enroll_ok = self.verify_raw(enroll_msg, self.sign_raw(enroll_msg))
        return {
            "publicKeyBytes": len(self.public_key_bytes),
            "approvalVerified": approval_ok,
            "enrollProofVerified": enroll_ok,
        }


def _sign_cu_session_start(
    signer: DevPhoneSigner, host_id: str, arguments: dict[str, Any]
) -> dict[str, str]:
    validate_cu_session_start_args(arguments)
    approval_id = str(uuid.uuid4())
    expires_at = (datetime.now(UTC) + timedelta(seconds=60)).isoformat()
    return signer.sign_r3_approval(approval_id, arguments, host_id, expires_at)


class GateClient:
    """Async HTTP/SSE client wrapping backend host-control and conversation endpoints."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_s: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}
        self._owns_client = http_client is None
        self.http = (
            http_client
            if http_client is not None
            else httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, read=max(timeout_s, 120.0)))
        )

    async def __aenter__(self) -> "GateClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self.http.aclose()

    async def resolve_host(self, host_id_or_alias: str) -> dict[str, Any]:
        resp = await self.http.get(f"{self.base_url}/v1/hosts", headers=self._headers)
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("items", []) if isinstance(payload, dict) else []
        for item in items:
            if isinstance(item, dict) and host_id_or_alias in (
                item.get("hostId"),
                item.get("alias"),
            ):
                return dict(item)
        target_lower = host_id_or_alias.strip().lower()
        for item in items:
            if isinstance(item, dict):
                alias = str(item.get("alias") or "").strip().lower()
                if alias == target_lower:
                    return dict(item)
        raise GateError(f"Host not found: {host_id_or_alias!r}")

    async def enroll_dev_phone(self, signer: DevPhoneSigner) -> str:
        ch_resp = await self.http.post(
            f"{self.base_url}/v1/hosts/self/challenge",
            headers=self._headers,
        )
        ch_resp.raise_for_status()
        ch = ch_resp.json()
        body = signer.build_self_enroll_payload(
            str(ch["nonce"]),
            str(ch["enrollmentCode"]),
        )
        enroll_resp = await self.http.post(
            f"{self.base_url}/v1/hosts/self",
            headers={**self._headers, "Idempotency-Key": str(uuid.uuid4())},
            json=body,
        )
        enroll_resp.raise_for_status()
        enrolled = enroll_resp.json()
        phone_id = str(enrolled.get("hostId") or "")
        if not phone_id:
            raise GateError("Self-enrollment response did not include hostId")
        return phone_id

    async def revoke_host(self, host_id: str) -> None:
        resp = await self.http.delete(
            f"{self.base_url}/v1/hosts/{host_id}",
            headers={**self._headers, "Idempotency-Key": str(uuid.uuid4())},
        )
        if resp.status_code not in (200, 204, 404):
            resp.raise_for_status()

    async def preflight_tool(
        self,
        host_id: str,
        tool: str,
        arguments: dict[str, Any],
        *,
        session_id: str | None = None,
    ) -> PreflightOutcome:
        if tool == FORBIDDEN_TOOL_NAME:
            raise ValueError(f"{FORBIDDEN_TOOL_NAME} is deprecated and forbidden")
        body: dict[str, Any] = {
            "schemaVersion": 1,
            "tool": tool,
            "arguments": arguments,
        }
        if session_id is not None:
            body["sessionId"] = session_id

        started = time.perf_counter()
        resp = await self.http.post(
            f"{self.base_url}/v1/hosts/{host_id}/tools:preflight",
            headers=self._headers,
            json=body,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        raw: dict[str, Any] = {}
        try:
            parsed = resp.json()
            if isinstance(parsed, dict):
                raw = parsed
        except ValueError:
            raw = {}

        nested = raw.get("preflight")
        block = dict(nested) if isinstance(nested, dict) else raw
        if resp.status_code >= 400:
            status = "error"
        else:
            raw_status = (
                block.get("status") if isinstance(block.get("status"), str) else raw.get("status")
            )
            status = str(raw_status) if isinstance(raw_status, str) and raw_status else "ok"
        error_code = (
            block.get("errorCode") or raw.get("errorCode") or raw.get("code") or raw.get("error")
        )
        element_role = block.get("elementRole")
        element_name = block.get("elementName")
        element_obj = block.get("element")
        if isinstance(element_obj, dict):
            element_role = element_role or element_obj.get("role")
            element_name = element_name or element_obj.get("name")
        return PreflightOutcome(
            http_status=resp.status_code,
            tool=tool,
            status=status,
            effective_risk=str(block.get("effectiveRisk") or ""),
            reason_code=str(block["reasonCode"]) if block.get("reasonCode") is not None else None,
            error_code=str(error_code) if error_code is not None else None,
            element_role=str(element_role) if element_role is not None else None,
            element_name=str(element_name) if element_name is not None else None,
            duration_ms=elapsed_ms,
            raw=raw,
        )

    async def invoke_tool(
        self,
        host_id: str,
        tool: str,
        arguments: dict[str, Any],
        *,
        deadline_ms: int = 10_000,
        approve: bool = True,
        approval: dict[str, str] | None = None,
    ) -> ToolCallOutcome:
        if tool == FORBIDDEN_TOOL_NAME:
            raise ValueError(f"{FORBIDDEN_TOOL_NAME} is deprecated and forbidden")
        body: dict[str, Any] = {
            "schemaVersion": 1,
            "tool": tool,
            "arguments": arguments,
            "deadlineMs": deadline_ms,
            "approve": approve,
        }
        if approval is not None:
            body.update(approval)

        started = time.perf_counter()
        resp = await self.http.post(
            f"{self.base_url}/v1/hosts/{host_id}/tools:invoke",
            headers={**self._headers, "Idempotency-Key": str(uuid.uuid4())},
            json=body,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        raw: dict[str, Any] = {}
        try:
            parsed = resp.json()
            if isinstance(parsed, dict):
                raw = parsed
        except ValueError:
            raw = {}

        sc_raw = raw.get("structuredContent")
        structured = dict(sc_raw) if isinstance(sc_raw, dict) else {}
        if resp.status_code >= 400:
            status = "error"
        else:
            raw_status = raw.get("status")
            status = str(raw_status) if isinstance(raw_status, str) and raw_status else "ok"
        is_error = resp.status_code >= 400 or bool(raw.get("isError", False))
        err_val = (
            raw.get("errorCode")
            or raw.get("code")
            or raw.get("error")
            or structured.get("errorCode")
        )
        host_dur = raw.get("durationMs")
        return ToolCallOutcome(
            http_status=resp.status_code,
            tool=tool,
            status=status,
            is_error=is_error,
            error_code=str(err_val) if err_val is not None else None,
            structured_content=structured,
            text=str(raw.get("text") or ""),
            duration_ms=elapsed_ms,
            host_duration_ms=float(host_dur) if isinstance(host_dur, int | float) else None,
            raw=raw,
        )

    async def decide_approval(
        self,
        approval_id: str,
        *,
        decision: str = "approve",
        scope: str = "once",
        signature: str | None = None,
        public_key: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "schemaVersion": 1,
            "decision": decision,
            "scope": scope,
        }
        if signature is not None:
            body["signature"] = signature
        if public_key is not None:
            body["publicKey"] = public_key
        resp = await self.http.post(
            f"{self.base_url}/v1/actions/approvals/{approval_id}",
            headers=self._headers,
            json=body,
        )
        resp.raise_for_status()
        parsed = resp.json()
        return dict(parsed) if isinstance(parsed, dict) else {}

    async def run_turn(
        self,
        pet_id: str,
        text: str,
        *,
        host_id: str,
        signer: DevPhoneSigner | None = None,
        auto_approve: bool = True,
    ) -> TurnOutcome:
        outcome = TurnOutcome()
        reply_parts: list[str] = []
        body = {
            "schemaVersion": 1,
            "text": text,
            "locale": "vi",
            "clientCapabilities": ["host_actions_v1"],
        }
        async with self.http.stream(
            "POST",
            f"{self.base_url}/v1/conversations/{pet_id}/turns",
            headers=self._headers,
            json=body,
        ) as resp:
            resp.raise_for_status()
            lines: list[str] = []
            async for raw_line in resp.aiter_lines():
                lines.append(raw_line)
                if raw_line.rstrip("\r") == "":
                    for event, payload in iter_sse_lines(lines):
                        await self._handle_sse_event(
                            event,
                            payload,
                            outcome,
                            reply_parts,
                            default_host_id=host_id,
                            signer=signer,
                            auto_approve=auto_approve,
                        )
                    lines.clear()
            if lines:
                for event, payload in iter_sse_lines(lines):
                    await self._handle_sse_event(
                        event,
                        payload,
                        outcome,
                        reply_parts,
                        default_host_id=host_id,
                        signer=signer,
                        auto_approve=auto_approve,
                    )
        outcome.reply = "".join(reply_parts)[:400]
        return outcome

    async def _handle_sse_event(
        self,
        event: str,
        payload: dict[str, Any],
        outcome: TurnOutcome,
        reply_parts: list[str],
        *,
        default_host_id: str,
        signer: DevPhoneSigner | None,
        auto_approve: bool,
    ) -> None:
        if event == "delta":
            reply_parts.append(str(payload.get("text", "")))
        elif event == "action_result":
            outcome.action_results.append(dict(payload))
        elif event == "cu_session":
            outcome.cu_session_events.append(dict(payload))
        elif event == "error":
            outcome.errors.append(str(payload.get("code") or payload.get("error") or "error"))
        elif event == "action_pending":
            outcome.owner_prompts += 1
            if auto_approve and isinstance(payload.get("approvalId"), str):
                approval_id = str(payload["approvalId"])
                sig: str | None = None
                pub: str | None = None
                args_hash = payload.get("argsSha256")
                expires_at = payload.get("expiresAt")
                target_host = str(payload.get("hostId") or default_host_id)
                if (
                    signer is not None
                    and isinstance(args_hash, str)
                    and args_hash
                    and isinstance(expires_at, str)
                    and expires_at
                ):
                    sig, pub = signer.sign_r3_approval_hash(
                        approval_id, args_hash, target_host, expires_at
                    )
                await self.decide_approval(
                    approval_id,
                    decision="approve",
                    scope="once",
                    signature=sig,
                    public_key=pub,
                )

    async def get_active_cu_session(self) -> dict[str, Any]:
        resp = await self.http.get(
            f"{self.base_url}/v1/actions/cu-session",
            headers=self._headers,
        )
        resp.raise_for_status()
        parsed = resp.json()
        return dict(parsed) if isinstance(parsed, dict) else {"active": False}

    async def stop_cu_session(self, session_id: str) -> dict[str, Any]:
        resp = await self.http.post(
            f"{self.base_url}/v1/actions/cu-session/{session_id}/stop",
            headers=self._headers,
        )
        if resp.status_code == 404:
            return {"sessionId": session_id, "stopped": False, "alreadyEnded": True}
        resp.raise_for_status()
        parsed = resp.json()
        return dict(parsed) if isinstance(parsed, dict) else {}

    async def list_audit_events(self, limit: int = 50) -> list[dict[str, Any]]:
        resp = await self.http.get(
            f"{self.base_url}/v1/audit-events",
            params={"limit": limit},
            headers=self._headers,
        )
        resp.raise_for_status()
        parsed = resp.json()
        if isinstance(parsed, list):
            return [dict(item) for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            items = parsed.get("items", [])
            if isinstance(items, list):
                return [dict(item) for item in items if isinstance(item, dict)]
        return []

    async def cleanup_dangling_cu_sessions(
        self, host_id: str, session_id: str | None = None
    ) -> None:
        if session_id:
            with suppress(httpx.HTTPError):
                await self.invoke_tool(
                    host_id,
                    "cu.session_end",
                    {"sessionId": session_id, "outcome": "completed"},
                )
        with suppress(httpx.HTTPError):
            active = await self.get_active_cu_session()
            if active.get("active"):
                session_obj = active.get("session")
                if isinstance(session_obj, dict):
                    active_id = session_obj.get("sessionId")
                    if isinstance(active_id, str) and active_id:
                        await self.stop_cu_session(active_id)


async def run_preflight_check(client: GateClient, host_id: str) -> PreflightCheckResult:
    """Verifies host online, TCC permissions, absence of `ui.accessibility_tree`, and preflight."""
    host = await client.resolve_host(host_id)
    online = bool(host.get("online"))
    tools = host_tool_names(host)
    ax_tree_exposed = FORBIDDEN_TOOL_NAME in tools

    status_res = await client.invoke_tool(host_id, "host.status", {})
    perms = status_res.structured_content.get("permissions")
    perms_dict = perms if isinstance(perms, dict) else {}
    accessibility = str(perms_dict.get("accessibility") or "unknown")
    raw_screen_recording = perms_dict.get("screenRecording")
    if isinstance(raw_screen_recording, str) and raw_screen_recording:
        screen_recording = raw_screen_recording
    else:
        screen_recording = (
            host_tool_permission(host, ("ui.observe_hybrid", "screen.capture")) or "unknown"
        )

    preflight_res = await client.preflight_tool(
        host_id,
        "input.click",
        {"xNorm": 0.5, "yNorm": 0.5},
    )
    dev_preflight_available = preflight_res.http_status != 404

    ok = (
        online
        and not ax_tree_exposed
        and status_res.status == "ok"
        and accessibility == "granted"
        and screen_recording == "granted"
        and dev_preflight_available
    )
    detail = (
        f"online={online} accessibility={accessibility} screenRecording={screen_recording} "
        f"ui.accessibility_tree_exposed={ax_tree_exposed} "
        f"dev_preflight_http={preflight_res.http_status}"
    )
    return PreflightCheckResult(
        ok=ok,
        host_online=online,
        accessibility=accessibility,
        screen_recording=screen_recording,
        accessibility_tree_exposed=ax_tree_exposed,
        dev_preflight_available=dev_preflight_available,
        detail=detail,
    )


def _extract_nodes(structured: dict[str, Any]) -> list[dict[str, Any]]:
    raw_nodes = structured.get("nodes")
    if not isinstance(raw_nodes, list):
        raw_nodes = structured.get("matches")
    if not isinstance(raw_nodes, list):
        return []
    return [dict(item) for item in raw_nodes if isinstance(item, dict)]


def _node_ref(node: dict[str, Any]) -> str:
    ref = node.get("ref")
    node_id = node.get("id")
    if isinstance(ref, str) and ref.startswith("@s"):
        return ref
    if isinstance(node_id, str) and node_id.startswith("@s"):
        return node_id
    if isinstance(ref, str) and ref:
        return ref
    if isinstance(node_id, str) and node_id:
        return node_id
    return ""


def _audit_events_delta(
    before: Sequence[dict[str, Any]],
    after: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if any(ev.get("eventId") is not None for ev in after) or any(
        ev.get("eventId") is not None for ev in before
    ):
        seen_ids = {str(ev["eventId"]) for ev in before if ev.get("eventId") is not None}
        return [
            ev
            for ev in after
            if ev.get("eventId") is not None and str(ev["eventId"]) not in seen_ids
        ]
    remaining = [json.dumps(ev, sort_keys=True) for ev in before]
    delta: list[dict[str, Any]] = []
    for ev in after:
        key = json.dumps(ev, sort_keys=True)
        if key in remaining:
            remaining.remove(key)
        else:
            delta.append(ev)
    return delta


def _audit_tool(ev: dict[str, Any]) -> str:
    meta = ev.get("metadata")
    if isinstance(meta, dict):
        tool = meta.get("tool")
        if isinstance(tool, str):
            return tool
    return ""


def _check_password_redaction(
    nodes: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    pw_nodes: list[dict[str, Any]] = []
    violations: list[str] = []
    for idx, node in enumerate(nodes):
        role = str(node.get("role") or "").lower()
        is_pw = bool(node.get("isPassword")) or role == "password"
        if not is_pw:
            continue
        pw_nodes.append(node)
        ref = _node_ref(node) or f"idx={idx}"
        if node.get("role") != "password":
            violations.append(f"{ref}:role={node.get('role')!r}")
        if node.get("name") != "•••":
            violations.append(f"{ref}:name={node.get('name')!r}")
        if "value" in node:
            violations.append(f"{ref}:value_present")
        for key in PASSWORD_SECRET_LEAK_KEYS:
            val = node.get(key)
            if isinstance(val, str) and val.strip() and val.strip() != "•••":
                violations.append(f"{ref}:leaked_{key}")
    return pw_nodes, violations


def _pick_editable_node(nodes: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    for node in nodes:
        role = str(node.get("role") or "").lower()
        is_pw = bool(node.get("isPassword"))
        if not is_pw and role in EDITABLE_DRAFT_ROLES and _node_ref(node):
            return node
    return None


def _pick_safe_ref_node(nodes: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    for node in nodes:
        role = str(node.get("role") or "").lower()
        name = unicodedata.normalize("NFC", str(node.get("name") or "")).lower()
        is_pw = bool(node.get("isPassword"))
        enabled = node.get("enabled")
        ref = _node_ref(node)
        bounds = node.get("bounds")
        top_y: float | None = None
        if isinstance(bounds, dict):
            raw_y = bounds.get("y")
            if isinstance(raw_y, int | float) and not isinstance(raw_y, bool):
                top_y = float(raw_y)
        if (
            not is_pw
            and enabled is not False
            and ref.startswith("@s")
            and role != "password"
            and (top_y is None or top_y >= 36.0)
            and not any(word in name for word in CONSEQUENTIAL_REF_WORDS)
        ):
            return node
    return None


def _node_norm_height(node: dict[str, Any]) -> float | None:
    bounds = node.get("bounds")
    if isinstance(bounds, list) and len(bounds) >= 4:
        y0, y1 = bounds[1], bounds[3]
        if (
            isinstance(y0, int | float)
            and isinstance(y1, int | float)
            and not isinstance(y0, bool)
            and not isinstance(y1, bool)
        ):
            return float(y1) - float(y0)
    return None


def _pick_track_node(nodes: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    matches = [
        node
        for node in nodes
        if "anh sai" in normalize_vi(str(node.get("name") or "")) and _node_ref(node)
    ]
    if not matches:
        return None
    row_matches = [
        node
        for node in matches
        if str(node.get("role") or "").lower() in {"list_item", "row", "button", "menuitem", "cell"}
    ]
    pool = row_matches or matches
    compact_rows = [
        node for node in pool if (h := _node_norm_height(node)) is not None and 0.0 < h <= 0.10
    ]
    if compact_rows:
        return compact_rows[0]
    return pool[0]


async def run_gate_1(
    client: GateClient,
    host_id: str,
    signer: DevPhoneSigner,
    *,
    concurrency: int = 10,
    total_calls: int = GATE1_TOTAL_CALLS,
) -> GateResult:
    """Gate 1: 50-call parallel stress & crash gate (<500ms, 0 crash, 0 timeout)."""
    t0 = time.perf_counter()
    result = GateResult(gate=1, name=GATE_NAMES[1], passed=False)
    session_id: str | None = None
    sem = asyncio.Semaphore(max(1, concurrency))

    try:
        launch_res = await client.invoke_tool(
            host_id,
            "app.launch",
            {"id": "visual-studio-code", "openTarget": "xuna_pet"},
        )
        launch_ok = launch_res.status == "ok" and not launch_res.is_error
        result.checks.append(
            GateCheck("launch_heavy_electron", launch_ok, f"status={launch_res.status}")
        )

        cu_args: dict[str, Any] = {
            "goal": "Gate 1 stress inspection",
            "apps": ["visual-studio-code"],
            "domains": [],
            "maxSteps": 40,
            "maxMinutes": 10,
        }
        cu_res = await client.invoke_tool(
            host_id,
            "cu.session_start",
            cu_args,
            approval=_sign_cu_session_start(signer, host_id, cu_args),
        )
        session_id = str(cu_res.structured_content.get("sessionId") or "") or None
        cu_ok = cu_res.status == "ok" and session_id is not None
        result.checks.append(
            GateCheck("start_cu_session", cu_ok, f"status={cu_res.status} sessionId={session_id}")
        )
        if not cu_ok:
            result.error = cu_res.error_code or "cu_session_start_failed"
            return result

        async def _burst_call(
            idx: int,
        ) -> tuple[str, float, float | None, bool, bool, bool, str]:
            async with sem:
                slot = idx % 4
                try:
                    if slot == 0:
                        res = await client.invoke_tool(
                            host_id,
                            "ui.inspect",
                            {"scope": "active_window", "maxNodes": 80},
                        )
                        is_timeout = res.status == "timeout" or res.error_code == "timeout"
                        is_crash = not is_timeout and (
                            res.http_status >= 400 or res.is_error or res.status != "ok"
                        )
                        return (
                            "ui.inspect",
                            res.duration_ms,
                            res.host_duration_ms,
                            is_crash,
                            is_timeout,
                            True,
                            res.status,
                        )
                    if slot == 1:
                        res = await client.invoke_tool(
                            host_id,
                            "ui.find",
                            {"query": "Explorer", "maxResults": 5},
                        )
                        is_timeout = res.status == "timeout" or res.error_code == "timeout"
                        is_crash = not is_timeout and (
                            res.http_status >= 400 or res.is_error or res.status != "ok"
                        )
                        return (
                            "ui.find",
                            res.duration_ms,
                            res.host_duration_ms,
                            is_crash,
                            is_timeout,
                            True,
                            res.status,
                        )
                    if slot == 2:
                        res = await client.invoke_tool(
                            host_id,
                            "ui.observe_hybrid",
                            {"scope": "active_window", "maxNodes": 60, "maxEdge": 960},
                        )
                        is_timeout = res.status == "timeout" or res.error_code == "timeout"
                        is_crash = not is_timeout and (
                            res.http_status >= 400 or res.is_error or res.status != "ok"
                        )
                        som_ok, som_msg = validate_jpeg_som_payload(
                            res.structured_content,
                            require_screenshot=True,
                        )
                        return (
                            "ui.observe_hybrid",
                            res.duration_ms,
                            res.host_duration_ms,
                            is_crash,
                            is_timeout,
                            som_ok,
                            som_msg,
                        )
                    pre = await client.preflight_tool(
                        host_id,
                        "input.click",
                        {"xNorm": 0.5, "yNorm": 0.5},
                        session_id=session_id,
                    )
                    is_timeout = pre.status == "timeout" or pre.error_code == "timeout"
                    is_crash = not is_timeout and (pre.http_status >= 400 or pre.status != "ok")
                    return (
                        "tools:preflight",
                        pre.duration_ms,
                        None,
                        is_crash,
                        is_timeout,
                        True,
                        pre.status,
                    )
                except httpx.TimeoutException:
                    return ("timeout", GATE1_BUDGET_MS, None, False, True, True, "http_timeout")
                except httpx.HTTPError as error:
                    return ("error", 0.0, None, True, False, False, type(error).__name__)

        burst_outcomes = await asyncio.gather(*(_burst_call(i) for i in range(total_calls)))
        latencies = [item[1] for item in burst_outcomes]
        host_latencies = [item[2] for item in burst_outcomes if item[2] is not None]
        crashes = sum(1 for item in burst_outcomes if item[3])
        timeouts = sum(1 for item in burst_outcomes if item[4])
        som_failures = [item[6] for item in burst_outcomes if not item[5]]
        over_budget = sum(
            1
            for item in burst_outcomes
            if item[1] >= GATE1_BUDGET_MS or (item[2] is not None and item[2] >= GATE1_BUDGET_MS)
        )

        host_alive = False
        health_detail = ""
        try:
            health_status = await client.invoke_tool(host_id, "host.status", {})
            health_ctx = await client.invoke_tool(host_id, "window.active_context", {})
            host_alive = (
                health_status.status == "ok"
                and not health_status.is_error
                and health_ctx.status == "ok"
                and not health_ctx.is_error
            )
            health_detail = f"host.status={health_status.status} active_context={health_ctx.status}"
        except httpx.TimeoutException:
            timeouts += 1
            health_detail = "trailing health check timeout"
        except httpx.HTTPError as error:
            crashes += 1
            health_detail = f"trailing health check {type(error).__name__}"

        sorted_lat = sorted(latencies)
        p50 = statistics.median(sorted_lat) if sorted_lat else 0.0
        p95_idx = min(len(sorted_lat) - 1, int(round(0.95 * (len(sorted_lat) - 1))))
        p95 = sorted_lat[p95_idx] if sorted_lat else 0.0
        max_lat = max(sorted_lat) if sorted_lat else 0.0

        result.crashes = crashes
        result.timeouts = timeouts
        result.metrics = {
            "totalCalls": total_calls,
            "concurrency": concurrency,
            "p50LatencyMs": round(p50, 2),
            "p95LatencyMs": round(p95, 2),
            "maxLatencyMs": round(max_lat, 2),
            "maxHostDurationMs": round(max(host_latencies), 2) if host_latencies else None,
            "overBudgetCount": over_budget,
            "somValidationErrors": som_failures[:5],
        }
        result.checks.extend(
            [
                GateCheck("zero_crashes", crashes == 0, f"crashes={crashes}"),
                GateCheck("zero_timeouts", timeouts == 0, f"timeouts={timeouts}"),
                GateCheck(
                    "under_500ms_budget",
                    over_budget == 0,
                    f"over_500ms={over_budget}/{total_calls} max={max_lat:.1f}ms",
                ),
                GateCheck(
                    "som_jpeg_valid",
                    len(som_failures) == 0,
                    f"som_failures={len(som_failures)}",
                ),
                GateCheck(
                    "trailing_host_alive",
                    host_alive,
                    health_detail,
                ),
            ]
        )
        result.passed = launch_ok and all(c.passed for c in result.checks)
        return result
    finally:
        await client.cleanup_dangling_cu_sessions(host_id, session_id)
        result.duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)


async def run_gate_2(
    client: GateClient,
    host_id: str,
    signer: DevPhoneSigner,
    *,
    pet_id: str | None = None,
) -> GateResult:
    """Gate 2: Headless-First (0 app.launch, 0 CU) & Continue-In-Place (targetAlreadyFocused)."""
    t0 = time.perf_counter()
    result = GateResult(gate=2, name=GATE_NAMES[2], passed=False)
    session_id: str | None = None
    part_a_app_launches = 0
    part_a_cu_starts = 0
    part_b_app_launches = 0

    try:
        # Part A: Headless / OS-native 1-step tools
        probe = uuid.uuid4().hex[:8]
        note_title = f"XUNA Gate2 {probe}"
        created = await client.invoke_tool(
            host_id,
            "notes.create",
            {"title": note_title, "body": f"headless-{probe}"},
        )
        searched = await client.invoke_tool(
            host_id,
            "notes.search",
            {"query": note_title, "maxResults": 5},
        )
        found_notes = searched.structured_content.get("notes")
        note_found = isinstance(found_notes, list) and any(
            isinstance(n, dict) and probe in str(n.get("title") or "") for n in found_notes
        )
        note_false_success = created.status == "ok" and not note_found
        if note_false_success:
            result.false_success += 1
        result.checks.append(
            GateCheck(
                "part_a_notes_headless_roundtrip",
                created.status == "ok" and note_found,
                f"created={created.status} found={note_found}",
                false_success=note_false_success,
            )
        )

        git_res = await client.invoke_tool(
            host_id,
            "workspace.git_summary",
            {"project": "xuna_pet"},
        )
        git_branch = str(git_res.structured_content.get("branch") or "")
        git_ok = git_res.status == "ok" and bool(git_branch)
        result.checks.append(GateCheck("part_a_git_summary", git_ok, f"branch={git_branch!r}"))

        media_res = await client.invoke_tool(host_id, "media.now_playing", {})
        media_state = str(media_res.structured_content.get("state") or "")
        media_ok = media_res.status == "ok" and media_state in {
            "playing",
            "paused",
            "stopped",
            "unknown",
        }
        result.checks.append(
            GateCheck("part_a_media_now_playing", media_ok, f"state={media_state!r}")
        )

        ctx_res = await client.invoke_tool(host_id, "window.active_context", {})
        running_apps = ctx_res.structured_content.get("runningApps")
        running_ok = (
            ctx_res.status == "ok"
            and isinstance(running_apps, list)
            and len(running_apps) > 0
            and all(isinstance(a, dict) and bool(a.get("name")) for a in running_apps)
        )
        result.checks.append(
            GateCheck(
                "part_a_running_apps_context",
                running_ok,
                f"runningApps_count={len(running_apps) if isinstance(running_apps, list) else 0}",
            )
        )

        if pet_id is not None:
            audits_before_a = await client.list_audit_events(limit=200)
            turn_a = await client.run_turn(
                pet_id,
                "Kiểm tra trạng thái git của dự án xuna_pet giúp mình",
                host_id=host_id,
                signer=signer,
                auto_approve=True,
            )
            audits_after_a = await client.list_audit_events(limit=200)
            delta_a = _audit_events_delta(audits_before_a, audits_after_a)
            part_a_app_launches += sum(
                1 for ev in delta_a if _audit_tool(ev) in {"app.launch", "app__launch"}
            )
            part_a_cu_starts += sum(
                1 for ev in delta_a if _audit_tool(ev) in {"cu.session_start", "cu__session_start"}
            )
            for ar in turn_a.action_results:
                target = str(ar.get("target") or "")
                if target in {"app.launch", "app__launch"}:
                    part_a_app_launches += 1
                if target in {"cu.session_start", "cu__session_start"}:
                    part_a_cu_starts += 1
            if any(
                ev.get("status") == "running" or bool(ev.get("sessionId"))
                for ev in turn_a.cu_session_events
            ):
                part_a_cu_starts = max(part_a_cu_starts, 1)
            turn_a_actions_ok = all(
                str(ar.get("status") or "ok") == "ok" for ar in turn_a.action_results
            )
            turn_a_ok = (
                part_a_app_launches == 0
                and part_a_cu_starts == 0
                and turn_a_actions_ok
                and not turn_a.errors
            )
            result.checks.append(
                GateCheck(
                    "part_a_turn_zero_launch_zero_cu",
                    turn_a_ok,
                    f"app_launch={part_a_app_launches} cu_start={part_a_cu_starts}",
                )
            )

        # Part B: Continue-In-Place when target app is already foregrounded
        setup_launch = await client.invoke_tool(host_id, "app.launch", {"id": "notes"})
        if setup_launch.status != "ok":
            result.checks.append(
                GateCheck("part_b_setup_foreground", False, f"status={setup_launch.status}")
            )
            return result

        cu_args: dict[str, Any] = {
            "goal": "Gate 2 continue in place",
            "apps": ["notes"],
            "domains": [],
            "maxSteps": 10,
            "maxMinutes": 5,
        }
        cu_res = await client.invoke_tool(
            host_id,
            "cu.session_start",
            cu_args,
            approval=_sign_cu_session_start(signer, host_id, cu_args),
        )
        session_id = str(cu_res.structured_content.get("sessionId") or "") or None
        focused = cu_res.structured_content.get("targetAlreadyFocused") is True
        snap_id = str(cu_res.structured_content.get("snapshotId") or "")
        nodes = _extract_nodes(cu_res.structured_content)
        valid_refs = bool(nodes) and all(_node_ref(n).startswith("@s") for n in nodes)
        cu_inplace_ok = (
            cu_res.status == "ok"
            and session_id is not None
            and focused
            and bool(snap_id)
            and valid_refs
        )
        result.checks.append(
            GateCheck(
                "part_b_cu_session_start_in_place",
                cu_inplace_ok,
                f"targetAlreadyFocused={focused} snapshotId={snap_id!r} nodes={len(nodes)}",
            )
        )

        find_res = await client.invoke_tool(
            host_id,
            "ui.find",
            {"role": "button", "maxResults": 5},
        )
        stale_refs = 1 if find_res.error_code == "stale_ref" else 0
        safe_node = _pick_safe_ref_node(nodes) or _pick_safe_ref_node(
            _extract_nodes(find_res.structured_content)
        )
        click_ok = False
        click_detail = "no_safe_ref"
        if safe_node is not None:
            safe_ref = _node_ref(safe_node)
            click_res = await client.invoke_tool(
                host_id,
                "input.click",
                {"elementId": safe_ref},
            )
            if click_res.error_code == "stale_ref":
                stale_refs += 1
            click_ok = click_res.status == "ok" and click_res.error_code != "stale_ref"
            click_detail = f"ref={safe_ref} status={click_res.status} err={click_res.error_code}"
        else:
            click_ok = find_res.status == "ok" and stale_refs == 0
            click_detail = f"ui.find_only status={find_res.status} stale_refs={stale_refs}"

        result.checks.append(GateCheck("part_b_in_place_step", click_ok, click_detail))

        if pet_id is not None:
            audits_before_b = await client.list_audit_events(limit=200)
            turn_b = await client.run_turn(
                pet_id,
                "Tìm nút tạo ghi chú trên cửa sổ Notes đang mở giúp mình",
                host_id=host_id,
                signer=signer,
                auto_approve=True,
            )
            audits_after_b = await client.list_audit_events(limit=200)
            delta_b = _audit_events_delta(audits_before_b, audits_after_b)
            part_b_app_launches += sum(
                1 for ev in delta_b if _audit_tool(ev) in {"app.launch", "app__launch"}
            )
            for ar in turn_b.action_results:
                if str(ar.get("target") or "") in {"app.launch", "app__launch"}:
                    part_b_app_launches += 1
            turn_b_actions_ok = all(
                str(ar.get("status") or "ok") == "ok" for ar in turn_b.action_results
            )
            result.checks.append(
                GateCheck(
                    "part_b_turn_zero_redundant_launch",
                    part_b_app_launches == 0 and turn_b_actions_ok and not turn_b.errors,
                    f"redundant_app_launch={part_b_app_launches}",
                )
            )

        result.metrics = {
            "partAAppLaunches": part_a_app_launches,
            "partACuSessions": part_a_cu_starts,
            "partBAppLaunches": part_b_app_launches,
            "staleRefs": stale_refs,
        }
        result.passed = all(c.passed for c in result.checks)
        return result
    finally:
        await client.cleanup_dangling_cu_sessions(host_id, session_id)
        result.duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)


async def run_gate_3(
    client: GateClient,
    host_id: str,
    *,
    settle_s: float = 0.5,
    poll_timeout_s: float = 5.0,
) -> GateResult:
    """Gate 3: OS Project Launch Gate (`app.launch(visual-studio-code, openTarget=xuna_pet)`)."""
    t0 = time.perf_counter()
    result = GateResult(gate=3, name=GATE_NAMES[3], passed=False)

    launch_res = await client.invoke_tool(
        host_id,
        "app.launch",
        {"id": "visual-studio-code", "openTarget": "xuna_pet"},
    )
    if settle_s > 0:
        await asyncio.sleep(settle_s)

    poll_deadline = time.monotonic() + (poll_timeout_s if settle_s > 0 else 0.0)
    poll_interval = min(settle_s, 0.5) if settle_s > 0 else 0.0
    app_name = ""
    window_title = ""
    verified = False
    while True:
        ctx_res = await client.invoke_tool(host_id, "window.active_context", {})
        app_name = str(ctx_res.structured_content.get("appName") or "")
        window_title = str(ctx_res.structured_content.get("windowTitle") or "")
        app_matches = "code" in app_name.lower() or "visual studio code" in app_name.lower()
        title_matches = "xuna_pet" in window_title.lower()
        verified = ctx_res.status == "ok" and app_matches and title_matches
        if verified or time.monotonic() >= poll_deadline:
            break
        await asyncio.sleep(poll_interval)

    is_false_success = launch_res.status == "ok" and not launch_res.is_error and not verified
    if is_false_success:
        result.false_success += 1

    warm_res = await client.invoke_tool(
        host_id,
        "app.launch",
        {"id": "visual-studio-code"},
    )
    already_running = warm_res.structured_content.get("alreadyRunning") is True

    result.checks = [
        GateCheck("app_launch_ok", launch_res.status == "ok", f"status={launch_res.status}"),
        GateCheck(
            "independent_window_title_xuna_pet",
            verified,
            f"appName={app_name!r} windowTitle={window_title!r}",
            false_success=is_false_success,
        ),
        GateCheck(
            "warm_activate_already_running",
            warm_res.status == "ok" and already_running,
            f"alreadyRunning={already_running}",
        ),
    ]
    result.metrics = {
        "appName": app_name,
        "windowTitle": window_title,
        "alreadyRunning": already_running,
    }
    result.passed = all(c.passed for c in result.checks)
    result.duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return result


async def _invoke_input_with_preflight(
    client: GateClient,
    host_id: str,
    signer: DevPhoneSigner,
    session_id: str,
    tool: str,
    arguments: dict[str, Any],
) -> ToolCallOutcome:
    pre = await client.preflight_tool(host_id, tool, arguments, session_id=session_id)
    if pre.http_status >= 400 or pre.status in {"denied", "error", "failed"}:
        return ToolCallOutcome(
            http_status=pre.http_status,
            tool=tool,
            status=pre.status,
            is_error=True,
            error_code=pre.error_code or pre.reason_code or "preflight_denied",
            structured_content={},
            text="",
            duration_ms=pre.duration_ms,
            host_duration_ms=None,
            raw=pre.raw,
        )
    approval: dict[str, str] | None = None
    approve = False
    eff_risk = pre.effective_risk.upper()
    if eff_risk == "R3":
        approval_id = str(uuid.uuid4())
        expires_at = (datetime.now(UTC) + timedelta(seconds=60)).isoformat()
        approval = signer.sign_r3_approval(approval_id, arguments, host_id, expires_at)
        approve = True
    elif eff_risk in {"R0", "R1", "R2"}:
        approve = True
    return await client.invoke_tool(host_id, tool, arguments, approve=approve, approval=approval)


async def run_gate_4(
    client: GateClient,
    host_id: str,
    signer: DevPhoneSigner,
    *,
    settle_s: float = 1.0,
) -> GateResult:
    """Gate 4: Complex Electron app in-window search & play verified by `media.now_playing`."""
    t0 = time.perf_counter()
    result = GateResult(gate=4, name=GATE_NAMES[4], passed=False)
    session_id: str | None = None

    try:
        before = await client.invoke_tool(host_id, "media.now_playing", {})
        before_state = str(before.structured_content.get("state") or "")
        before_title = normalize_vi(str(before.structured_content.get("title") or ""))
        if before_state == "playing" and "anh sai" in before_title:
            await client.invoke_tool(host_id, "media.play_pause", {})

        launch_res = await client.invoke_tool(host_id, "app.launch", {"id": "spotify"})
        if launch_res.status != "ok":
            result.checks.append(GateCheck("launch_spotify", False, f"status={launch_res.status}"))
            return result

        cu_args: dict[str, Any] = {
            "goal": "Search and play Anh sai roi in Spotify",
            "apps": ["spotify"],
            "domains": [],
            "maxSteps": 20,
            "maxMinutes": 5,
        }
        cu_res = await client.invoke_tool(
            host_id,
            "cu.session_start",
            cu_args,
            approval=_sign_cu_session_start(signer, host_id, cu_args),
        )
        session_id = str(cu_res.structured_content.get("sessionId") or "") or None
        if cu_res.status != "ok" or session_id is None:
            result.checks.append(
                GateCheck("start_spotify_cu_session", False, f"status={cu_res.status}")
            )
            return result

        is_cold_launch = launch_res.structured_content.get("alreadyRunning") is False
        search_node: dict[str, Any] | None = None
        max_find_attempts = 5 if is_cold_launch else 4
        for attempt in range(max_find_attempts):
            if attempt > 0 and settle_s > 0:
                await asyncio.sleep(0.75)
            find_search = await client.invoke_tool(
                host_id, "ui.find", {"role": "searchfield", "maxResults": 5}
            )
            search_node = _pick_editable_node(_extract_nodes(find_search.structured_content))
            if search_node is not None:
                break
            if attempt >= 1:
                inspect_search = await client.invoke_tool(
                    host_id, "ui.inspect", {"scope": "active_window", "maxNodes": 80}
                )
                search_node = _pick_editable_node(_extract_nodes(inspect_search.structured_content))
                if search_node is not None:
                    break
        if search_node is None:
            obs = await client.invoke_tool(
                host_id,
                "ui.observe_hybrid",
                {"scope": "active_window", "maxNodes": 80, "maxEdge": 960},
            )
            som_ok, som_msg = validate_jpeg_som_payload(
                obs.structured_content,
                require_screenshot=True,
            )
            result.checks.append(GateCheck("spotify_observe_hybrid_som", som_ok, som_msg))
            search_node = _pick_editable_node(_extract_nodes(obs.structured_content))

        if search_node is None:
            result.checks.append(
                GateCheck("locate_search_field", False, "no searchfield/textfield ref found")
            )
            return result

        search_ref = _node_ref(search_node)
        type_res = await _invoke_input_with_preflight(
            client,
            host_id,
            signer,
            session_id,
            "input.type_text",
            {"elementId": search_ref, "text": "Anh sai rồi"},
        )
        key_res = await _invoke_input_with_preflight(
            client,
            host_id,
            signer,
            session_id,
            "input.key",
            {"combo": "Return"},
        )
        if settle_s > 0:
            await asyncio.sleep(settle_s)

        track_node: dict[str, Any] | None = None
        for track_attempt in range(5):
            if track_attempt > 0 and settle_s > 0:
                await asyncio.sleep(0.75)
            if track_attempt == 0 or track_attempt >= 2:
                track_obs = await client.invoke_tool(
                    host_id,
                    "ui.observe_hybrid",
                    {"scope": "active_window", "maxNodes": 80, "maxEdge": 960},
                )
                if track_attempt == 0:
                    track_som_ok, track_som_msg = validate_jpeg_som_payload(
                        track_obs.structured_content,
                        require_screenshot=True,
                    )
                    result.checks.append(
                        GateCheck("spotify_track_observe_hybrid_som", track_som_ok, track_som_msg)
                    )
                track_node = _pick_track_node(_extract_nodes(track_obs.structured_content))
            if track_node is None:
                find_track = await client.invoke_tool(
                    host_id, "ui.find", {"query": "Anh", "maxResults": 5}
                )
                track_node = _pick_track_node(_extract_nodes(find_track.structured_content))
            if track_node is not None:
                break

        if track_node is None:
            result.checks.append(
                GateCheck("locate_track_result", False, "no result node matching 'anh sai'")
            )
            return result

        activate_res = await _invoke_input_with_preflight(
            client,
            host_id,
            signer,
            session_id,
            "input.click",
            {"elementId": _node_ref(track_node), "clickCount": 2},
        )
        if settle_s > 0:
            await asyncio.sleep(settle_s)

        now_playing = await client.invoke_tool(host_id, "media.now_playing", {})
        np_state = str(now_playing.structured_content.get("state") or "")
        np_title = str(now_playing.structured_content.get("title") or "")
        np_app = str(now_playing.structured_content.get("app") or "")
        playback_verified = (
            now_playing.status == "ok"
            and np_state == "playing"
            and "anh sai" in normalize_vi(np_title)
            and "spotify" in np_app.lower()
        )
        actions_ok = (
            type_res.status == "ok" and key_res.status == "ok" and activate_res.status == "ok"
        )
        is_false_success = actions_ok and not playback_verified
        if is_false_success:
            result.false_success += 1

        result.checks.extend(
            [
                GateCheck(
                    "search_and_activate_actions", actions_ok, f"activate={activate_res.status}"
                ),
                GateCheck(
                    "independent_media_now_playing",
                    playback_verified,
                    f"state={np_state!r} title={np_title!r} app={np_app!r}",
                    false_success=is_false_success,
                ),
            ]
        )
        result.metrics = {"state": np_state, "title": np_title, "app": np_app}
        result.passed = all(c.passed for c in result.checks)
        return result
    finally:
        await client.cleanup_dangling_cu_sessions(host_id, session_id)
        result.duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)


async def run_gate_5(
    client: GateClient,
    host_id: str,
    *,
    reveal_path: str = "/Users/xthao28/xuna_pet/AGENTS.md",
) -> GateResult:
    """Gate 5: Native macOS desktop apps (`Finder` + `Notes`) verified independently."""
    t0 = time.perf_counter()
    result = GateResult(gate=5, name=GATE_NAMES[5], passed=False)

    reveal_res = await client.invoke_tool(host_id, "finder.reveal", {"path": reveal_path})
    revealed_flag = reveal_res.structured_content.get("revealed") is True
    ctx_res = await client.invoke_tool(host_id, "window.active_context", {})
    front_app = str(ctx_res.structured_content.get("appName") or "")
    finder_verified = ctx_res.status == "ok" and "finder" in front_app.lower()
    finder_false_success = reveal_res.status == "ok" and revealed_flag and not finder_verified
    if finder_false_success:
        result.false_success += 1

    probe = uuid.uuid4().hex[:8]
    note_title = f"XUNA Gate5 {probe}"
    create_res = await client.invoke_tool(
        host_id,
        "notes.create",
        {"title": note_title, "body": f"verify-{probe}"},
    )
    search_res = await client.invoke_tool(
        host_id,
        "notes.search",
        {"query": note_title, "maxResults": 5},
    )
    notes_list = search_res.structured_content.get("notes")
    notes_found = (
        search_res.status == "ok"
        and isinstance(notes_list, list)
        and any(isinstance(n, dict) and probe in str(n.get("title") or "") for n in notes_list)
    )
    notes_false_success = create_res.status == "ok" and not notes_found
    if notes_false_success:
        result.false_success += 1

    result.checks = [
        GateCheck(
            "finder_reveal_frontmost",
            reveal_res.status == "ok" and revealed_flag and finder_verified,
            f"revealed={revealed_flag} appName={front_app!r}",
            false_success=finder_false_success,
        ),
        GateCheck(
            "notes_create_and_search_verify",
            create_res.status == "ok" and notes_found,
            f"create={create_res.status} found={notes_found}",
            false_success=notes_false_success,
        ),
    ]
    result.metrics = {"finderAppName": front_app, "noteProbe": probe}
    result.passed = all(c.passed for c in result.checks)
    result.duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return result


async def run_gate_6(
    client: GateClient,
    host_id: str,
    signer: DevPhoneSigner,
    *,
    pet_id: str | None = None,
    password_probe_url: str | None = None,
    settle_s: float = 0.0,
) -> GateResult:
    """Gate 6: ADR-0015 security envelope (`owner_prompts == 1`, Terminal/password blocked)."""
    t0 = time.perf_counter()
    result = GateResult(gate=6, name=GATE_NAMES[6], passed=False)
    opened_session_ids: list[str] = []
    owner_prompts = 0

    try:
        # 6a. Unsigned & tampered R3 rejection
        notes_cu_args: dict[str, Any] = {
            "goal": "Gate 6 trust envelope",
            "apps": ["notes"],
            "domains": [],
            "maxSteps": 15,
            "maxMinutes": 5,
        }
        unsigned_res = await client.invoke_tool(
            host_id, "cu.session_start", notes_cu_args, approve=False
        )
        unsigned_rejected = unsigned_res.status != "ok" and unsigned_res.error_code in {
            "approval_required",
            "approval_signature_invalid",
        }
        result.checks.append(
            GateCheck(
                "unsigned_r3_rejected",
                unsigned_rejected,
                f"status={unsigned_res.status} error={unsigned_res.error_code}",
            )
        )

        tampered_approval = _sign_cu_session_start(signer, host_id, notes_cu_args)
        raw_sig = bytearray(base64.b64decode(tampered_approval["signature"]))
        raw_sig[0] ^= 0xFF
        tampered_approval["signature"] = base64.b64encode(bytes(raw_sig)).decode("ascii")
        tampered_res = await client.invoke_tool(
            host_id,
            "cu.session_start",
            notes_cu_args,
            approve=True,
            approval=tampered_approval,
        )
        tampered_rejected = tampered_res.status != "ok" and tampered_res.error_code in {
            "approval_signature_invalid",
            "approval_required",
        }
        result.checks.append(
            GateCheck(
                "tampered_r3_rejected",
                tampered_rejected,
                f"status={tampered_res.status} error={tampered_res.error_code}",
            )
        )

        # 6b. Single-prompt signed CU session + R2 draft typing
        await client.invoke_tool(host_id, "app.launch", {"id": "notes"})
        signed_cu = await client.invoke_tool(
            host_id,
            "cu.session_start",
            notes_cu_args,
            approve=True,
            approval=_sign_cu_session_start(signer, host_id, notes_cu_args),
        )
        session_id = str(signed_cu.structured_content.get("sessionId") or "") or None
        if session_id is not None:
            opened_session_ids.append(session_id)
        if signed_cu.status == "ok" and session_id is not None:
            owner_prompts = 1
        result.checks.append(
            GateCheck(
                "signed_cu_session_start",
                signed_cu.status == "ok" and session_id is not None,
                f"status={signed_cu.status} sessionId={session_id}",
            )
        )

        if session_id is not None:
            inspect_res = await client.invoke_tool(
                host_id, "ui.inspect", {"scope": "active_window", "maxNodes": 60}
            )
            inspect_nodes = _extract_nodes(inspect_res.structured_content)
            inspect_pw_nodes, inspect_pw_violations = _check_password_redaction(inspect_nodes)
            result.checks.append(
                GateCheck(
                    "inspect_password_nodes_redacted",
                    len(inspect_pw_violations) == 0,
                    (
                        f"pw_nodes={len(inspect_pw_nodes)} "
                        f"violations={inspect_pw_violations[:3]}"
                    ),
                )
            )
            editable = _pick_editable_node(inspect_nodes)
            if editable is not None:
                editable_ref = _node_ref(editable)
                pre_draft = await client.preflight_tool(
                    host_id,
                    "input.type_text",
                    {"elementId": editable_ref, "text": "xuna-gate6-draft"},
                    session_id=session_id,
                )
                draft_res = await client.invoke_tool(
                    host_id,
                    "input.type_text",
                    {"elementId": editable_ref, "text": "xuna-gate6-draft"},
                    approve=True,
                    approval=None,
                )
                draft_ok = (
                    pre_draft.status == "ok"
                    and pre_draft.effective_risk.upper() == "R2"
                    and draft_res.status == "ok"
                )
                result.checks.append(
                    GateCheck(
                        "r2_unsigned_draft_inside_session",
                        draft_ok,
                        f"preflight={pre_draft.effective_risk} draft={draft_res.status}",
                    )
                )
            else:
                result.checks.append(
                    GateCheck(
                        "r2_unsigned_draft_inside_session",
                        False,
                        "no editable non-password field found in Notes",
                    )
                )

            # Password denial check using fresh ref from ui.inspect + explicit elementName
            if settle_s > 0:
                await asyncio.sleep(min(0.4, settle_s))
            pw_inspect_res = await client.invoke_tool(
                host_id, "ui.inspect", {"scope": "active_window", "maxNodes": 60}
            )
            pw_inspect_nodes = _extract_nodes(pw_inspect_res.structured_content)
            pw_editable = _pick_editable_node(pw_inspect_nodes)
            non_pw_nodes = [
                n
                for n in pw_inspect_nodes
                if not bool(n.get("isPassword"))
                and str(n.get("role") or "").lower() != "password"
                and _node_ref(n)
            ]
            probe_target = pw_editable or (non_pw_nodes[0] if non_pw_nodes else None)
            probe_ref = _node_ref(probe_target) if probe_target is not None else ""
            if not probe_ref:
                result.checks.append(
                    GateCheck(
                        "password_field_blocked",
                        False,
                        "no fresh snapshot ref from ui.inspect for password probe",
                    )
                )
            else:
                pw_pre = await client.preflight_tool(
                    host_id,
                    "input.type_text",
                    {
                        "elementId": probe_ref,
                        "elementName": "password",
                        "text": "probe",
                    },
                    session_id=session_id,
                )
                pw_inv = await client.invoke_tool(
                    host_id,
                    "input.type_text",
                    {
                        "elementId": probe_ref,
                        "elementName": "password",
                        "text": "probe",
                    },
                    approve=True,
                    approval=None,
                )
                pw_blocked = (
                    pw_pre.status == "denied"
                    and pw_pre.error_code == "password_field_forbidden"
                    and pw_inv.status == "denied"
                    and pw_inv.error_code == "password_field_forbidden"
                )
                result.checks.append(
                    GateCheck(
                        "password_field_blocked",
                        pw_blocked,
                        f"ref={probe_ref} preflight={pw_pre.status}/{pw_pre.error_code} "
                        f"invoke={pw_inv.status}/{pw_inv.error_code}",
                    )
                )

            await client.invoke_tool(
                host_id,
                "cu.session_end",
                {"sessionId": session_id, "outcome": "completed"},
            )
            if session_id in opened_session_ids:
                opened_session_ids.remove(session_id)

        # Optional live password field URL redaction + block check
        if password_probe_url is not None:
            await client.invoke_tool(host_id, "url.open", {"url": password_probe_url})
            browser_cu_args: dict[str, Any] = {
                "goal": "Gate 6 live password redaction probe",
                "apps": ["safari", "google-chrome"],
                "domains": [],
                "maxSteps": 10,
                "maxMinutes": 5,
            }
            b_cu = await client.invoke_tool(
                host_id,
                "cu.session_start",
                browser_cu_args,
                approval=_sign_cu_session_start(signer, host_id, browser_cu_args),
            )
            b_sid = str(b_cu.structured_content.get("sessionId") or "") or None
            if b_sid is not None:
                opened_session_ids.append(b_sid)
            b_inspect = await client.invoke_tool(
                host_id, "ui.inspect", {"scope": "active_window", "maxNodes": 80}
            )
            pw_nodes, b_violations = _check_password_redaction(
                _extract_nodes(b_inspect.structured_content)
            )
            redaction_ok = bool(pw_nodes) and len(b_violations) == 0
            live_pw_blocked = False
            pw_ref = _node_ref(pw_nodes[0]) if pw_nodes else ""
            if pw_ref:
                l_pre = await client.preflight_tool(
                    host_id,
                    "input.type_text",
                    {"elementId": pw_ref, "text": "probe"},
                    session_id=b_sid,
                )
                l_inv = await client.invoke_tool(
                    host_id,
                    "input.type_text",
                    {"elementId": pw_ref, "text": "probe"},
                    approve=True,
                    approval=None,
                )
                live_pw_blocked = (
                    l_pre.status == "denied"
                    and l_pre.error_code == "password_field_forbidden"
                    and l_inv.status == "denied"
                    and l_inv.error_code == "password_field_forbidden"
                )
            result.checks.append(
                GateCheck(
                    "live_password_redaction_and_block",
                    redaction_ok and live_pw_blocked,
                    f"pw_nodes={len(pw_nodes)} redacted={redaction_ok} "
                    f"violations={b_violations[:3]} blocked={live_pw_blocked}",
                )
            )
            if b_sid is not None:
                await client.invoke_tool(
                    host_id,
                    "cu.session_end",
                    {"sessionId": b_sid, "outcome": "completed"},
                )
                if b_sid in opened_session_ids:
                    opened_session_ids.remove(b_sid)

        # 6c. Terminal hard-denial
        term_launch = await client.invoke_tool(host_id, "app.launch", {"id": "terminal"})
        term_cu_args: dict[str, Any] = {
            "goal": "Open terminal",
            "apps": ["terminal"],
            "domains": [],
            "maxSteps": 5,
            "maxMinutes": 5,
        }
        term_cu = await client.invoke_tool(
            host_id,
            "cu.session_start",
            term_cu_args,
            approval=_sign_cu_session_start(signer, host_id, term_cu_args),
        )
        term_sid = str(term_cu.structured_content.get("sessionId") or "") or None
        if term_sid is not None:
            opened_session_ids.append(term_sid)
        ctx_after_term = await client.invoke_tool(host_id, "window.active_context", {})
        front_after_term = str(ctx_after_term.structured_content.get("appName") or "").lower()
        terminal_not_front = not any(tok in front_after_term for tok in HARD_DENIED_TERMINALS)
        terminal_blocked = (
            term_launch.status == "denied"
            and term_launch.error_code == "denylisted_target_forbidden"
            and term_cu.status == "denied"
            and term_cu.error_code == "denylisted_target_forbidden"
            and terminal_not_front
        )
        result.checks.append(
            GateCheck(
                "terminal_hard_denied",
                terminal_blocked,
                f"launch={term_launch.status}/{term_launch.error_code} "
                f"cu={term_cu.status}/{term_cu.error_code} front={front_after_term!r}",
            )
        )

        # 6d. Optional SSE /turns + /audit-events owner_prompts == 1 verification
        if pet_id is not None:
            if password_probe_url is not None:
                await client.invoke_tool(host_id, "app.launch", {"id": "notes"})
            audits_before_6 = await client.list_audit_events(limit=200)
            turn_6 = await client.run_turn(
                pet_id,
                "Trong cửa sổ Notes đang mở sẵn trên màn hình, hãy gõ nháp 'gate6-envelope' "
                "rồi kết thúc phiên điều khiển.",
                host_id=host_id,
                signer=signer,
                auto_approve=True,
            )
            owner_prompts = turn_6.owner_prompts
            audits_after_6 = await client.list_audit_events(limit=200)
            delta_6 = _audit_events_delta(audits_before_6, audits_after_6)
            ended_audits = [
                ev for ev in delta_6 if ev.get("action") == AuditAction.HOST_CU_SESSION_ENDED.value
            ]
            ended_prompt_values: list[int | None] = []
            for ev in ended_audits:
                meta = ev.get("metadata")
                prompts_val = meta.get("prompts") if isinstance(meta, dict) else None
                ended_prompt_values.append(prompts_val if type(prompts_val) is int else None)
            all_ended_prompts_valid = bool(ended_prompt_values) and all(
                val == 1 for val in ended_prompt_values
            )
            audit_prompts: int | None = None
            if all_ended_prompts_valid:
                audit_prompts = 1
            else:
                for val in ended_prompt_values:
                    if val is not None and val != 1:
                        audit_prompts = val
                        break
            turn_6_actions_ok = bool(turn_6.action_results) and all(
                str(ar.get("status") or "") == "ok" for ar in turn_6.action_results
            )
            turn_6_no_launch = not any(
                _audit_tool(ev) in {"app.launch", "app__launch"} for ev in delta_6
            )
            prompts_ok = (
                not turn_6.errors
                and turn_6_actions_ok
                and turn_6_no_launch
                and bool(turn_6.cu_session_events)
                and owner_prompts == 1
                and all_ended_prompts_valid
            )
            result.checks.append(
                GateCheck(
                    "turn_and_audit_owner_prompts_eq_1",
                    prompts_ok,
                    f"turn_prompts={owner_prompts} audit_prompts={audit_prompts} "
                    f"ended_audits={len(ended_audits)} actions={len(turn_6.action_results)} "
                    f"cu_events={len(turn_6.cu_session_events)} errors={turn_6.errors}",
                )
            )
        else:
            result.checks.append(
                GateCheck(
                    "single_owner_prompt_envelope",
                    owner_prompts == 1,
                    f"owner_prompts={owner_prompts}",
                )
            )

        result.metrics = {"ownerPrompts": owner_prompts}
        result.passed = all(c.passed for c in result.checks)
        return result
    finally:
        for sid in list(opened_session_ids):
            await client.cleanup_dangling_cu_sessions(host_id, sid)
        if not opened_session_ids:
            await client.cleanup_dangling_cu_sessions(host_id, None)
        result.duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)


def _parse_gates(raw_gates: Sequence[str] | None) -> list[int]:
    if not raw_gates:
        return list(ALL_GATES)
    selected: list[int] = []
    seen: set[int] = set()
    for item in raw_gates:
        for token in item.split(","):
            clean = token.strip()
            if not clean:
                continue
            try:
                num = int(clean)
            except ValueError as error:
                raise ValueError(f"invalid gate number: {clean!r}") from error
            if num not in GATE_NAMES:
                raise ValueError(f"gate must be in 1..6, got {num}")
            if num not in seen:
                seen.add(num)
                selected.append(num)
    if not selected:
        raise ValueError("at least one gate in 1..6 must be specified")
    return selected


def build_dry_run_report(
    *,
    gates: Sequence[int],
    host_id: str,
    base_url: str,
    pet_id: str | None,
    concurrency: int,
) -> dict[str, Any]:
    signer = DevPhoneSigner()
    signer_check = signer.run_self_test()
    now_iso = datetime.now(UTC).isoformat()
    return {
        "schemaVersion": 1,
        "mode": "dry-run",
        "hostId": host_id,
        "hostAlias": DEFAULT_HOST_ALIAS,
        "baseUrl": base_url,
        "petIdProvided": pet_id is not None,
        "startedAt": now_iso,
        "finishedAt": now_iso,
        "signerSelfTest": signer_check,
        "plan": [
            {
                "gate": g,
                "name": GATE_NAMES[g],
                "concurrency": concurrency if g == 1 else 1,
                "usesDeprecatedAccessibilityTree": False,
            }
            for g in gates
        ],
        "summary": {
            "passed": bool(
                signer_check["approvalVerified"] and signer_check["enrollProofVerified"]
            ),
            "gatesPlanned": list(gates),
            "falseSuccess": 0,
            "crashes": 0,
            "timeouts": 0,
        },
    }


def _aggregate_report(
    *,
    mode: str,
    host_id: str,
    host_alias: str,
    base_url: str,
    started_at: str,
    preflight: PreflightCheckResult | None,
    gate_results: Sequence[GateResult],
) -> dict[str, Any]:
    finished_at = datetime.now(UTC).isoformat()
    false_success = sum(g.false_success for g in gate_results)
    crashes = sum(g.crashes for g in gate_results)
    timeouts = sum(g.timeouts for g in gate_results)
    gates_passed = sum(1 for g in gate_results if g.passed)
    gates_failed = len(gate_results) - gates_passed

    p50_ms: float | None = None
    p95_ms: float | None = None
    max_ms: float | None = None
    owner_prompts: int | None = None
    for g in gate_results:
        if g.gate == 1 and g.metrics:
            p50_ms = g.metrics.get("p50LatencyMs")
            p95_ms = g.metrics.get("p95LatencyMs")
            max_ms = g.metrics.get("maxLatencyMs")
        if g.gate == 6 and g.metrics:
            owner_prompts = g.metrics.get("ownerPrompts")

    preflight_ok = preflight.ok if preflight is not None else True
    overall_passed = (
        preflight_ok and gates_failed == 0 and false_success == 0 and crashes == 0 and timeouts == 0
    )
    return {
        "schemaVersion": 1,
        "mode": mode,
        "hostId": host_id,
        "hostAlias": host_alias,
        "baseUrl": base_url,
        "startedAt": started_at,
        "finishedAt": finished_at,
        "summary": {
            "passed": overall_passed,
            "gatesRun": [g.gate for g in gate_results],
            "gatesPassed": gates_passed,
            "gatesFailed": gates_failed,
            "falseSuccess": false_success,
            "crashes": crashes,
            "timeouts": timeouts,
            "p50LatencyMs": p50_ms,
            "p95LatencyMs": p95_ms,
            "maxLatencyMs": max_ms,
            "ownerPrompts": owner_prompts,
        },
        "preflight": {
            "ok": preflight.ok,
            "hostOnline": preflight.host_online,
            "accessibility": preflight.accessibility,
            "screenRecording": preflight.screen_recording,
            "accessibilityTreeExposed": preflight.accessibility_tree_exposed,
            "devPreflightAvailable": preflight.dev_preflight_available,
            "detail": preflight.detail,
        }
        if preflight is not None
        else None,
        "gates": [asdict(g) for g in gate_results],
    }


def _emit_output(report: dict[str, Any], *, as_json: bool, out_path: Path | None) -> None:
    rendered_json = json.dumps(report, ensure_ascii=False, indent=2)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered_json + "\n", encoding="utf-8")

    if as_json:
        sys.stdout.write(rendered_json + "\n")
    else:
        summary = report.get("summary", {})
        mode = report.get("mode", "live")
        lines = [
            f"UDC-20 Live macOS Control Gate ({mode}) | host={report.get('hostId')} "
            f"| passed={summary.get('passed')}"
        ]
        preflight = report.get("preflight")
        if isinstance(preflight, dict):
            lines.append(
                f"  [PREFLIGHT] {'PASS' if preflight.get('ok') else 'FAIL'} "
                f"- {preflight.get('detail')}"
            )
        for gate in report.get("gates", []):
            if isinstance(gate, dict):
                status = "PASS" if gate.get("passed") else "FAIL"
                lines.append(
                    f"  [GATE {gate.get('gate')}] {status} {gate.get('name')} "
                    f"({gate.get('duration_ms')}ms, false_success={gate.get('false_success')})"
                )
        for plan_item in report.get("plan", []):
            if isinstance(plan_item, dict):
                lines.append(
                    f"  [PLAN {plan_item.get('gate')}] {plan_item.get('name')} "
                    f"(concurrency={plan_item.get('concurrency')})"
                )
        sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


async def execute_live(
    client: GateClient,
    *,
    host_arg: str,
    gates: Sequence[int],
    run_preflight: bool,
    run_gates: bool,
    pet_id: str | None,
    concurrency: int,
    password_probe_url: str | None,
    settle_s: float = 0.5,
) -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat()
    host = await client.resolve_host(host_arg)
    resolved_host_id = str(host["hostId"])
    resolved_alias = str(host.get("alias") or DEFAULT_HOST_ALIAS)

    preflight_res: PreflightCheckResult | None = None
    if run_preflight:
        preflight_res = await run_preflight_check(client, resolved_host_id)
        if not preflight_res.ok or not run_gates:
            return _aggregate_report(
                mode="preflight" if not run_gates else "live",
                host_id=resolved_host_id,
                host_alias=resolved_alias,
                base_url=client.base_url,
                started_at=started_at,
                preflight=preflight_res,
                gate_results=[],
            )

    signer = DevPhoneSigner()
    phone_host_id: str | None = None
    gate_results: list[GateResult] = []
    try:
        phone_host_id = await client.enroll_dev_phone(signer)
        for gate_num in gates:
            if gate_num == 1:
                gate_results.append(
                    await run_gate_1(client, resolved_host_id, signer, concurrency=concurrency)
                )
            elif gate_num == 2:
                gate_results.append(
                    await run_gate_2(client, resolved_host_id, signer, pet_id=pet_id)
                )
            elif gate_num == 3:
                gate_results.append(await run_gate_3(client, resolved_host_id, settle_s=settle_s))
            elif gate_num == 4:
                gate_results.append(
                    await run_gate_4(client, resolved_host_id, signer, settle_s=settle_s)
                )
            elif gate_num == 5:
                gate_results.append(await run_gate_5(client, resolved_host_id))
            elif gate_num == 6:
                gate_results.append(
                    await run_gate_6(
                        client,
                        resolved_host_id,
                        signer,
                        pet_id=pet_id,
                        password_probe_url=password_probe_url,
                        settle_s=settle_s,
                    )
                )
    finally:
        await client.cleanup_dangling_cu_sessions(resolved_host_id)
        if phone_host_id is not None:
            with suppress(httpx.HTTPError):
                await client.revoke_host(phone_host_id)

    return _aggregate_report(
        mode="live",
        host_id=resolved_host_id,
        host_alias=resolved_alias,
        base_url=client.base_url,
        started_at=started_at,
        preflight=preflight_res,
        gate_results=gate_results,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Opaque-box live macOS control gate runner for UDC-20 M5."
    )
    parser.add_argument(
        "--gate",
        action="append",
        default=None,
        help="Gate number 1..6 (repeatable or comma-separated).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 6 gates (default when --gate is omitted).",
    )
    parser.add_argument(
        "--host-id",
        "--host",
        dest="host_id",
        default=DEFAULT_HOST_ID,
        help=f"Target host ID or alias (default: {DEFAULT_HOST_ID}).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("XUNA_API_BASE_URL", DEFAULT_BASE_URL),
        help="Backend base URL (default: $XUNA_API_BASE_URL or http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("XUNA_OWNER_TOKEN", ""),
        help="Owner Bearer token (default: $XUNA_OWNER_TOKEN). Never printed.",
    )
    parser.add_argument(
        "--pet-id",
        default=None,
        help="Optional PetId enabling /v1/conversations/{pet_id}/turns SSE checks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate local signer/schema helpers and output gate plan with 0 network calls.",
    )
    parser.add_argument(
        "--preflight-check",
        action="store_true",
        help="Verify host online, TCC permissions, no ui.accessibility_tree, and tools:preflight.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional UTF-8 JSON report output path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write machine-readable JSON summary to stdout.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Concurrency for Gate 1 50-call stress burst (default: 10).",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=30.0,
        help="HTTP request timeout in seconds (default: 30.0).",
    )
    parser.add_argument(
        "--password-probe-url",
        default=None,
        help="Optional benign HTTPS/local URL with <input type=password> for live probe.",
    )
    return parser


async def async_main(
    argv: Sequence[str] | None = None,
    *,
    http_client: httpx.AsyncClient | None = None,
    settle_s: float = 0.5,
) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        gates = ALL_GATES if args.all and not args.gate else tuple(_parse_gates(args.gate))
    except ValueError as error:
        parser.error(str(error))

    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be > 0")

    if args.dry_run:
        report = build_dry_run_report(
            gates=gates,
            host_id=args.host_id,
            base_url=args.base_url,
            pet_id=args.pet_id,
            concurrency=args.concurrency,
        )
        _emit_output(report, as_json=args.json, out_path=args.out)
        return 0 if report["summary"]["passed"] else 1

    if not str(args.token).strip():
        parser.error("--token or XUNA_OWNER_TOKEN is required for live execution")

    run_gates = not args.preflight_check or bool(args.gate) or bool(args.all)
    async with GateClient(
        args.base_url,
        args.token,
        timeout_s=args.timeout_s,
        http_client=http_client,
    ) as client:
        report = await execute_live(
            client,
            host_arg=args.host_id,
            gates=gates,
            run_preflight=bool(args.preflight_check),
            run_gates=run_gates,
            pet_id=args.pet_id,
            concurrency=args.concurrency,
            password_probe_url=args.password_probe_url,
            settle_s=settle_s,
        )
    _emit_output(report, as_json=args.json, out_path=args.out)
    return 0 if report["summary"]["passed"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    sys.exit(main())
