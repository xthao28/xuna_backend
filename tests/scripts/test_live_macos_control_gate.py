"""Offline unit & MockTransport E2E tests for `scripts/live_macos_control_gate.py`."""

import asyncio
import base64
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from xuna_backend.modules.audit.domain.event import AuditAction
from xuna_backend.modules.hosts.domain.crypto import (
    HostProofPurpose,
    approval_proof_message,
    args_sha256,
    host_proof_message,
)

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "live_macos_control_gate.py"
_SPEC = importlib.util.spec_from_file_location("live_macos_control_gate", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
gate_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate_mod)

_SAMPLE_JPEG_B64 = base64.b64encode(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\xff\xd9").decode("ascii")


def _verify_p256(pub_bytes: bytes, msg: bytes, sig_raw: bytes) -> None:
    assert len(pub_bytes) == 65 and pub_bytes[0] == 0x04
    assert len(sig_raw) == 64
    pub_key = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), pub_bytes)
    r = int.from_bytes(sig_raw[:32], "big")
    s = int.from_bytes(sig_raw[32:], "big")
    pub_key.verify(encode_dss_signature(r, s), msg, ec.ECDSA(hashes.SHA256()))


def test_dev_phone_signer_matches_backend_crypto() -> None:
    signer = gate_mod.DevPhoneSigner()
    args: dict[str, Any] = {
        "goal": "Test gate",
        "apps": ["notes"],
        "domains": [],
        "maxSteps": 10,
        "maxMinutes": 5,
    }
    approval_id = "11111111-1111-4111-8111-111111111111"
    host_id = gate_mod.DEFAULT_HOST_ID
    expires_at = "2026-11-19T16:00:00+00:00"

    bundle = signer.sign_r3_approval(approval_id, args, host_id, expires_at)
    expected_msg = approval_proof_message(approval_id, args_sha256(args), host_id, expires_at)
    _verify_p256(
        base64.b64decode(bundle["publicKey"]),
        expected_msg,
        base64.b64decode(bundle["signature"]),
    )

    nonce_b64url = base64.urlsafe_b64encode(b"\xaa" * 32).rstrip(b"=").decode("ascii")
    code_b64url = base64.urlsafe_b64encode(b"\xbb" * 16).rstrip(b"=").decode("ascii")
    enroll = signer.build_self_enroll_payload(nonce_b64url, code_b64url)
    enroll_msg = host_proof_message(
        nonce=b"\xaa" * 32,
        purpose=HostProofPurpose.ENROLL,
        payload=b"\xbb" * 16,
    )
    _verify_p256(
        gate_mod._b64url_decode(enroll["publicKey"]),
        enroll_msg,
        gate_mod._b64url_decode(enroll["proof"]),
    )


def test_validation_helpers_and_forbidden_tool_guard() -> None:
    gate_mod.validate_cu_session_start_args(
        {
            "goal": "Open notes",
            "apps": ["notes"],
            "domains": [],
            "maxSteps": 10,
            "maxMinutes": 5,
        }
    )
    with pytest.raises(ValueError, match="exact keys"):
        gate_mod.validate_cu_session_start_args({"goal": "missing keys"})

    ok_som, _ = gate_mod.validate_jpeg_som_payload(
        {
            "nodes": [{"id": "ui_1", "ref": "@s00000001:e1", "bounds": [0.1, 0.2, 0.3, 0.4]}],
            "screenshot": {
                "mimeType": "image/jpeg",
                "width": 640,
                "height": 480,
                "dataBase64": _SAMPLE_JPEG_B64,
            },
        },
        require_screenshot=True,
    )
    assert ok_som is True

    missing_shot, missing_msg = gate_mod.validate_jpeg_som_payload(
        {"nodes": [{"id": "ui_1", "ref": "@s00000001:e1", "bounds": [0.1, 0.2, 0.3, 0.4]}]},
        require_screenshot=True,
    )
    assert missing_shot is False
    assert "screenshot block missing" in missing_msg

    bad_som, msg = gate_mod.validate_jpeg_som_payload(
        {
            "nodes": [{"id": "ui_1", "ref": "@s00000001:e1", "bounds": [0.1, 1.5, 0.3, 0.4]}],
        }
    )
    assert bad_som is False
    assert "out of [0, 1]" in msg

    assert gate_mod._node_ref({"id": "ui_1", "ref": "@s00000002:e3"}) == "@s00000002:e3"
    assert gate_mod._pick_safe_ref_node(
        [
            {
                "id": "ui_1",
                "ref": "@s00000001:e1",
                "role": "button",
                "name": "Disabled",
                "enabled": False,
            },
            {
                "id": "ui_2",
                "ref": "@s00000001:e2",
                "role": "button",
                "name": "Delete note",
                "enabled": True,
            },
            {
                "id": "ui_3",
                "ref": "@s00000001:e3",
                "role": "button",
                "name": "Xóa mục",
                "enabled": True,
            },
            {
                "id": "ui_4",
                "ref": "@s00000001:e4",
                "role": "button",
                "name": "Send email",
                "enabled": True,
            },
            {
                "id": "ui_5",
                "ref": "@s00000001:e5",
                "role": "button",
                "name": "Explorer",
                "enabled": True,
            },
        ]
    ) == {
        "id": "ui_5",
        "ref": "@s00000001:e5",
        "role": "button",
        "name": "Explorer",
        "enabled": True,
    }


_R2_PLUS_TOOLS: frozenset[str] = frozenset(
    {
        "app.launch",
        "cu.session_start",
        "cu.session_end",
        "input.click",
        "input.type_text",
        "input.key",
        "finder.reveal",
        "notes.create",
        "url.open",
    }
)


class _MockGateBackend:
    """Stateful mock backend simulating all 6 live gates and configurable failure modes."""

    def __init__(
        self,
        *,
        host_online: bool = True,
        accessibility: str = "granted",
        screen_recording: str = "granted",
        expose_ax_tree: bool = False,
        gate1_slow_ms: float | None = None,
        gate1_timeout: bool = False,
        gate1_preflight_404_rfc9457: bool = False,
        gate1_missing_screenshot: bool = False,
        gate1_trailing_connect_error: bool = False,
        gate2_redundant_launch_in_turn: bool = False,
        gate2_cu_in_turn: bool = False,
        gate2_part_b_launch_in_turn: bool = False,
        gate2_missing_focus: bool = False,
        gate2_stale_ref: bool = False,
        gate2_miss_note: bool = False,
        gate3_wrong_title: bool = False,
        gate3_warm_poll_flips: int = 0,
        gate4_false_success_paused: bool = False,
        gate4_wrong_app: str | None = None,
        gate4_non_empty_unrelated_observe: bool = False,
        gate4_track_missing_screenshot: bool = False,
        gate5_finder_not_front: bool = False,
        gate5_miss_note: bool = False,
        gate6_extra_prompt: bool = False,
        gate6_allow_terminal: bool = False,
        gate6_allow_password: bool = False,
        gate6_accept_tampered_r3: bool = False,
        gate6_unredacted_inspect_password: bool = False,
        gate6_leak_password_text: bool = False,
        gate6_sse_error: bool = False,
        gate6_denied_action_result: bool = False,
        gate6_omit_cu_sse: bool = False,
        gate6_omit_ended_audit: bool = False,
        gate6_bool_prompts_audit: bool = False,
        gate6_empty_action_results: bool = False,
        gate6_multiple_ended_audits_mismatch: bool = False,
    ) -> None:
        self.host_online = host_online
        self.accessibility = accessibility
        self.screen_recording = screen_recording
        self.expose_ax_tree = expose_ax_tree
        self.gate1_slow_ms = gate1_slow_ms
        self.gate1_timeout = gate1_timeout
        self.gate1_preflight_404_rfc9457 = gate1_preflight_404_rfc9457
        self.gate1_missing_screenshot = gate1_missing_screenshot
        self.gate1_trailing_connect_error = gate1_trailing_connect_error
        self.gate2_redundant_launch_in_turn = gate2_redundant_launch_in_turn
        self.gate2_cu_in_turn = gate2_cu_in_turn
        self.gate2_part_b_launch_in_turn = gate2_part_b_launch_in_turn
        self.gate2_missing_focus = gate2_missing_focus
        self.gate2_stale_ref = gate2_stale_ref
        self.gate2_miss_note = gate2_miss_note
        self.gate3_wrong_title = gate3_wrong_title
        self.gate3_warm_poll_flips = gate3_warm_poll_flips
        self.gate4_false_success_paused = gate4_false_success_paused
        self.gate4_wrong_app = gate4_wrong_app
        self.gate4_non_empty_unrelated_observe = gate4_non_empty_unrelated_observe
        self.gate4_track_missing_screenshot = gate4_track_missing_screenshot
        self.gate5_finder_not_front = gate5_finder_not_front
        self.gate5_miss_note = gate5_miss_note
        self.gate6_extra_prompt = gate6_extra_prompt
        self.gate6_allow_terminal = gate6_allow_terminal
        self.gate6_allow_password = gate6_allow_password
        self.gate6_accept_tampered_r3 = gate6_accept_tampered_r3
        self.gate6_unredacted_inspect_password = gate6_unredacted_inspect_password
        self.gate6_leak_password_text = gate6_leak_password_text
        self.gate6_sse_error = gate6_sse_error
        self.gate6_denied_action_result = gate6_denied_action_result
        self.gate6_omit_cu_sse = gate6_omit_cu_sse
        self.gate6_omit_ended_audit = gate6_omit_ended_audit
        self.gate6_bool_prompts_audit = gate6_bool_prompts_audit
        self.gate6_empty_action_results = gate6_empty_action_results
        self.gate6_multiple_ended_audits_mismatch = gate6_multiple_ended_audits_mismatch

        self.enrolled_pub_key: bytes | None = None
        self.enrolled_phone_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        self.deleted_hosts: list[str] = []
        self.calls: list[tuple[str, str]] = []
        self.invoked_tools: list[str] = []
        self.front_app = "Notes"
        self.front_title = "Notes"
        self.turn_front_apps: list[str] = []
        self.notes: list[dict[str, str]] = []
        self.spotify_playing = False
        self.spotify_track = ""
        self.active_cu_id: str | None = None
        self.snapshot_seq = 0
        self.latest_snapshot_id: str | None = None
        self.dropped_refs: set[str] = set()
        self.burst_counter = 0
        self.turn_counter = 0
        self.vscode_context_polls = 0
        self.audit_seq = 0
        self.audit_events: list[dict[str, Any]] = []

    def _record_audit(self, action: str, metadata: dict[str, Any]) -> None:
        self.audit_seq += 1
        self.audit_events.insert(
            0,
            {
                "eventId": f"ev-{self.audit_seq:06d}",
                "action": action,
                "subject": gate_mod.DEFAULT_HOST_ID,
                "metadata": metadata,
                "occurredAt": "2026-11-19T14:00:00+00:00",
            },
        )

    def _current_snapshot_id(self) -> str:
        return self.latest_snapshot_id or "s00000001"

    def _mint_snapshot_id(self) -> str:
        self.snapshot_seq += 1
        self.latest_snapshot_id = f"s{self.snapshot_seq:08x}"
        return self.latest_snapshot_id

    def _check_snapshot_ref(self, arguments: dict[str, Any]) -> httpx.Response | None:
        element_id = arguments.get("elementId")
        if isinstance(element_id, str) and element_id.startswith("@s"):
            if element_id in self.dropped_refs:
                return httpx.Response(
                    200,
                    json={
                        "status": "failed",
                        "isError": True,
                        "errorCode": "stale_ref",
                        "structuredContent": {},
                    },
                )
            snap_part = element_id.split(":", 1)[0].lstrip("@")
            if self.latest_snapshot_id is not None and snap_part != self.latest_snapshot_id:
                return httpx.Response(
                    200,
                    json={
                        "status": "failed",
                        "isError": True,
                        "errorCode": "stale_ref",
                        "structuredContent": {},
                    },
                )
        return None

    def _verify_signed_bundle(self, body: dict[str, Any], arguments: dict[str, Any]) -> bool:
        if self.enrolled_pub_key is None:
            return False
        approval_id = str(body.get("approvalId") or "")
        expires_at = str(body.get("expiresAt") or "")
        sig_b64 = str(body.get("signature") or "")
        pub_b64 = str(body.get("publicKey") or "")
        if not approval_id or not expires_at or not sig_b64 or not pub_b64:
            return False
        try:
            pub_bytes = base64.b64decode(pub_b64)
            sig_bytes = base64.b64decode(sig_b64)
            if pub_bytes != self.enrolled_pub_key:
                return False
            msg = approval_proof_message(
                approval_id,
                args_sha256(arguments),
                gate_mod.DEFAULT_HOST_ID,
                expires_at,
            )
            _verify_p256(pub_bytes, msg, sig_bytes)
            return True
        except Exception:
            return False

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append((request.method, path))

        if request.method == "GET" and path == "/v1/hosts":
            tools: list[dict[str, Any]] = [
                {"name": "host.status"},
                {"name": "window.active_context"},
                {"name": "app.launch"},
                {"name": "cu.session_start"},
                {"name": "cu.session_end"},
                {"name": "ui.inspect"},
                {"name": "ui.find"},
                {
                    "name": "ui.observe_hybrid",
                    "_meta": {
                        "xuna/policy": {
                            "version": 1,
                            "family": "ui",
                            "risk": "R1",
                            "observes": True,
                            "reversible": True,
                            "permission": self.screen_recording,
                        }
                    },
                },
                {
                    "name": "screen.capture",
                    "_meta": {
                        "xuna/policy": {
                            "version": 1,
                            "family": "screen",
                            "risk": "R1",
                            "observes": True,
                            "reversible": True,
                            "permission": self.screen_recording,
                        }
                    },
                },
                {"name": "input.click"},
                {"name": "input.type_text"},
                {"name": "input.key"},
                {"name": "notes.create"},
                {"name": "notes.search"},
                {"name": "finder.reveal"},
                {"name": "media.now_playing"},
                {"name": "workspace.git_summary"},
            ]
            if self.expose_ax_tree:
                tools.append({"name": "ui.accessibility_tree"})
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "hostId": gate_mod.DEFAULT_HOST_ID,
                            "alias": gate_mod.DEFAULT_HOST_ALIAS,
                            "platform": "macos",
                            "online": self.host_online,
                            "tools": tools,
                        }
                    ]
                },
            )

        if request.method == "POST" and path == "/v1/hosts/self/challenge":
            return httpx.Response(
                200,
                json={
                    "nonce": base64.urlsafe_b64encode(b"\x11" * 32).rstrip(b"=").decode("ascii"),
                    "enrollmentCode": (
                        base64.urlsafe_b64encode(b"\x22" * 16).rstrip(b"=").decode("ascii")
                    ),
                    "expiresIn": 60,
                },
            )

        if request.method == "POST" and path == "/v1/hosts/self":
            body = json.loads(request.read().decode())
            pub_bytes = gate_mod._b64url_decode(body["publicKey"])
            sig_bytes = gate_mod._b64url_decode(body["proof"])
            msg = host_proof_message(
                nonce=b"\x11" * 32,
                purpose=HostProofPurpose.ENROLL,
                payload=b"\x22" * 16,
            )
            _verify_p256(pub_bytes, msg, sig_bytes)
            self.enrolled_pub_key = pub_bytes
            return httpx.Response(200, json={"hostId": self.enrolled_phone_id})

        if request.method == "DELETE" and path.startswith("/v1/hosts/"):
            self.deleted_hosts.append(path.rsplit("/", 1)[-1])
            return httpx.Response(204)

        if request.method == "POST" and path.endswith("/tools:preflight"):
            if self.gate1_preflight_404_rfc9457 and self.front_app == "Visual Studio Code":
                return httpx.Response(
                    404,
                    json={
                        "type": "https://xuna.dev/problems/not_found",
                        "title": "Not Found",
                        "status": 404,
                        "code": "not_found",
                    },
                )
            body = json.loads(request.read().decode())
            tool = body["tool"]
            arguments = body.get("arguments", {})
            if tool in {"input.click", "input.type_text"}:
                stale_resp = self._check_snapshot_ref(arguments)
                if stale_resp is not None:
                    return stale_resp
            element_id = str(arguments.get("elementId") or "")
            if tool == "input.type_text" and (
                arguments.get("elementName") == "password" or element_id.endswith(":e9")
            ):
                if self.gate6_allow_password:
                    return httpx.Response(
                        200,
                        json={"status": "ok", "effectiveRisk": "R2", "reasonCode": "draft_text"},
                    )
                return httpx.Response(
                    200,
                    json={
                        "status": "denied",
                        "errorCode": "password_field_forbidden",
                        "effectiveRisk": "R3",
                    },
                )
            if tool == "input.type_text":
                return httpx.Response(
                    200,
                    json={"status": "ok", "effectiveRisk": "R2", "reasonCode": "draft_text"},
                )
            if tool == "input.key" and arguments.get("combo") == "Return":
                return httpx.Response(
                    200,
                    json={"status": "ok", "effectiveRisk": "R2", "reasonCode": "search_submit"},
                )
            return httpx.Response(
                200,
                json={"status": "ok", "effectiveRisk": "R2", "reasonCode": "safe"},
            )

        if request.method == "POST" and path.endswith("/tools:invoke"):
            body = json.loads(request.read().decode())
            tool = body["tool"]
            args = body.get("arguments", {})
            self.invoked_tools.append(tool)
            return self._handle_invoke(tool, args, body)

        if request.method == "GET" and path == "/v1/actions/cu-session":
            if self.active_cu_id:
                return httpx.Response(
                    200, json={"active": True, "session": {"sessionId": self.active_cu_id}}
                )
            return httpx.Response(200, json={"active": False})

        if request.method == "POST" and path.startswith("/v1/actions/cu-session/"):
            self.active_cu_id = None
            return httpx.Response(200, json={"stopped": True})

        if request.method == "POST" and path.endswith("/turns"):
            return self._handle_turn(json.loads(request.read().decode()))

        if request.method == "POST" and path.startswith("/v1/actions/approvals/"):
            return httpx.Response(200, json={"decision": "approve"})

        if request.method == "GET" and path == "/v1/audit-events":
            raw_limit = request.url.params.get("limit")
            if raw_limit is not None:
                try:
                    limit_val = max(1, min(200, int(raw_limit)))
                    return httpx.Response(200, json={"items": list(self.audit_events[:limit_val])})
                except ValueError:
                    pass
            return httpx.Response(200, json={"items": list(self.audit_events)})

        return httpx.Response(404)

    def _handle_invoke(
        self, tool: str, args: dict[str, Any], body: dict[str, Any]
    ) -> httpx.Response:
        # Mirror hosts/application/use_cases.py:879-906 backend R2+ approval gate
        if tool in _R2_PLUS_TOOLS and not body.get("approve"):
            return httpx.Response(
                200,
                json={
                    "status": "denied",
                    "isError": True,
                    "errorCode": "approval_required",
                    "structuredContent": {},
                },
            )

        if tool == "host.status":
            if self.gate1_trailing_connect_error and self.burst_counter > 0:
                raise httpx.ConnectError("host crashed and disconnected")
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "durationMs": 8.0,
                    "structuredContent": {
                        "permissions": {
                            "accessibility": self.accessibility,
                            "notifications": "granted",
                            "automation.notes": "granted",
                        }
                    },
                },
            )

        if tool == "window.active_context":
            title = self.front_title
            if (
                self.front_app == "Visual Studio Code"
                and self.gate3_warm_poll_flips > 0
                and self.vscode_context_polls < self.gate3_warm_poll_flips
            ):
                self.vscode_context_polls += 1
                title = "Welcome"
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "durationMs": 12.0,
                    "structuredContent": {
                        "appName": self.front_app,
                        "windowTitle": title,
                        "visibleSummary": "summary",
                        "redactedCount": 0,
                        "truncated": False,
                        "runningApps": [
                            {"name": "Notes", "bundleId": "com.apple.Notes"},
                            {"name": "Visual Studio Code", "bundleId": "com.microsoft.VSCode"},
                        ],
                    },
                },
            )

        if tool == "app.launch":
            app_id = str(args.get("id") or "")
            if app_id in {"terminal", "iterm2"} and not self.gate6_allow_terminal:
                return httpx.Response(
                    200,
                    json={
                        "status": "denied",
                        "isError": True,
                        "errorCode": "denylisted_target_forbidden",
                        "structuredContent": {},
                    },
                )
            if app_id == "visual-studio-code":
                self.front_app = "Visual Studio Code"
                if args.get("openTarget") == "xuna_pet":
                    self.front_title = "AGENTS.md — other" if self.gate3_wrong_title else "xuna_pet"
                return httpx.Response(
                    200,
                    json={
                        "status": "ok",
                        "durationMs": 45.0,
                        "structuredContent": {
                            "id": app_id,
                            "alreadyRunning": "openTarget" not in args,
                        },
                    },
                )
            if app_id == "spotify":
                self.front_app = "Spotify"
                self.front_title = "Spotify"
            elif app_id == "notes":
                self.front_app = "Notes"
                self.front_title = "Notes"
            elif app_id == "terminal":
                self.front_app = "Terminal"
                self.front_title = "zsh"
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "durationMs": 25.0,
                    "structuredContent": {"id": app_id, "alreadyRunning": True},
                },
            )

        if tool == "cu.session_start":
            apps = args.get("apps", [])
            if "terminal" in apps and not self.gate6_allow_terminal:
                return httpx.Response(
                    200,
                    json={
                        "status": "denied",
                        "isError": True,
                        "errorCode": "denylisted_target_forbidden",
                        "structuredContent": {},
                    },
                )
            if not self._verify_signed_bundle(body, args) and not self.gate6_accept_tampered_r3:
                return httpx.Response(
                    200,
                    json={
                        "status": "denied",
                        "isError": True,
                        "errorCode": "approval_signature_invalid",
                        "structuredContent": {},
                    },
                )
            self.active_cu_id = "cu-session-1"
            self.snapshot_seq = 0
            snap_id = self._mint_snapshot_id()
            focused = not self.gate2_missing_focus
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "durationMs": 30.0,
                    "structuredContent": {
                        "sessionId": self.active_cu_id,
                        "foregroundApp": self.front_app,
                        "windowTitle": self.front_title,
                        "targetAlreadyFocused": focused,
                        "snapshotId": snap_id,
                        "nodes": [
                            {
                                "id": "ui_1",
                                "ref": f"@{snap_id}:e1",
                                "role": "textfield",
                                "name": "New Note",
                                "enabled": True,
                                "bounds": [0.1, 0.1, 0.5, 0.2],
                            }
                        ],
                    },
                },
            )

        if tool == "cu.session_end":
            self.active_cu_id = None
            self.latest_snapshot_id = None
            return httpx.Response(
                200,
                json={"status": "ok", "structuredContent": {"ended": True}},
            )

        if tool in {"ui.inspect", "ui.find", "ui.observe_hybrid"}:
            self.burst_counter += 1
            if (
                self.gate1_timeout
                and self.front_app == "Visual Studio Code"
                and self.burst_counter == 3
            ):
                raise httpx.ReadTimeout("simulated burst timeout")
            dur = self.gate1_slow_ms if self.gate1_slow_ms is not None else 35.0
            if tool == "ui.inspect":
                snap_id = self._mint_snapshot_id()
                pw_node: dict[str, Any] = {
                    "id": "ui_9",
                    "ref": f"@{snap_id}:e9",
                    "role": "password",
                    "name": "•••",
                    "isPassword": True,
                }
                if self.gate6_unredacted_inspect_password:
                    pw_node = {
                        "id": "ui_9",
                        "ref": f"@{snap_id}:e9",
                        "role": "password",
                        "name": "hunter2",
                        "value": "hunter2",
                        "isPassword": True,
                    }
                elif self.gate6_leak_password_text and self.front_app == "Safari":
                    pw_node["text"] = "hunter2"
                nodes: list[dict[str, Any]] = [
                    {
                        "id": "ui_1",
                        "ref": f"@{snap_id}:e1",
                        "role": "textfield",
                        "name": "Body",
                        "enabled": True,
                    },
                    pw_node,
                ]
                return httpx.Response(
                    200,
                    json={
                        "status": "ok",
                        "durationMs": dur,
                        "structuredContent": {
                            "appId": self.front_app,
                            "windowTitle": self.front_title,
                            "snapshotId": snap_id,
                            "nodes": nodes,
                        },
                    },
                )
            if tool == "ui.find":
                snap_id = self._current_snapshot_id()
                matches: list[dict[str, Any]] = []
                if args.get("role") == "searchfield":
                    matches = [
                        {
                            "id": "ui_2",
                            "ref": f"@{snap_id}:e2",
                            "role": "searchfield",
                            "name": "Search",
                            "enabled": True,
                        }
                    ]
                elif args.get("query") == "Anh":
                    matches = [
                        {
                            "id": "ui_3",
                            "ref": f"@{snap_id}:e3",
                            "role": "row",
                            "name": "Anh Sai Rồi - Sơn Tùng",
                            "enabled": True,
                        }
                    ]
                else:
                    matches = [
                        {
                            "id": "ui_1",
                            "ref": f"@{snap_id}:e1",
                            "role": "button",
                            "name": "Explorer",
                            "enabled": True,
                        }
                    ]
                return httpx.Response(
                    200,
                    json={
                        "status": "ok",
                        "durationMs": dur,
                        "structuredContent": {
                            "appId": self.front_app,
                            "windowTitle": self.front_title,
                            "snapshotId": snap_id,
                            "searchedNodes": 10,
                            "total": len(matches),
                            "truncated": False,
                            "matches": matches,
                        },
                    },
                )
            snap_id = self._mint_snapshot_id()
            obs_nodes = [
                {
                    "id": "ui_3",
                    "ref": f"@{snap_id}:e3",
                    "role": "row",
                    "name": "Anh Sai Rồi",
                    "enabled": True,
                    "bounds": [0.1, 0.2, 0.8, 0.3],
                }
            ]
            if self.gate4_non_empty_unrelated_observe and self.front_app == "Spotify":
                obs_nodes = [
                    {
                        "id": "ui_4",
                        "ref": f"@{snap_id}:e4",
                        "role": "button",
                        "name": "Home",
                        "enabled": True,
                        "bounds": [0.05, 0.05, 0.15, 0.1],
                    }
                ]
            obs_structured: dict[str, Any] = {
                "appId": self.front_app,
                "windowTitle": self.front_title,
                "snapshotId": snap_id,
                "nodes": obs_nodes,
            }
            omit_shot = (
                self.gate1_missing_screenshot and self.front_app == "Visual Studio Code"
            ) or (self.gate4_track_missing_screenshot and self.front_app == "Spotify")
            if not omit_shot:
                obs_structured["screenshot"] = {
                    "mimeType": "image/jpeg",
                    "width": 640,
                    "height": 480,
                    "dataBase64": _SAMPLE_JPEG_B64,
                }
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "durationMs": dur,
                    "structuredContent": obs_structured,
                },
            )

        if tool == "input.click":
            stale_resp = self._check_snapshot_ref(args)
            if stale_resp is not None:
                return stale_resp
            if self.gate2_stale_ref and str(args.get("elementId") or "").endswith(":e1"):
                return httpx.Response(
                    200,
                    json={
                        "status": "failed",
                        "isError": True,
                        "errorCode": "stale_ref",
                        "structuredContent": {},
                    },
                )
            if args.get("clickCount") == 2 and not self.gate4_false_success_paused:
                self.spotify_playing = True
                self.spotify_track = "Anh Sai Rồi"
            return httpx.Response(
                200,
                json={"status": "ok", "structuredContent": {"effect": "changed"}},
            )

        if tool == "input.type_text":
            stale_resp = self._check_snapshot_ref(args)
            if stale_resp is not None:
                return stale_resp
            element_id = str(args.get("elementId") or "")
            if args.get("elementName") == "password" or element_id.endswith(":e9"):
                if self.gate6_allow_password:
                    return httpx.Response(
                        200,
                        json={"status": "ok", "structuredContent": {"effect": "changed"}},
                    )
                return httpx.Response(
                    200,
                    json={
                        "status": "denied",
                        "isError": True,
                        "errorCode": "password_field_forbidden",
                        "structuredContent": {},
                    },
                )
            if element_id:
                self.dropped_refs.add(element_id)
            return httpx.Response(
                200,
                json={"status": "ok", "structuredContent": {"effect": "changed"}},
            )

        if tool == "input.key":
            return httpx.Response(
                200,
                json={"status": "ok", "structuredContent": {"effect": "changed"}},
            )

        if tool == "url.open":
            self.front_app = "Safari"
            return httpx.Response(200, json={"status": "ok", "structuredContent": {"opened": True}})

        if tool == "notes.create":
            self.notes.append({"id": "n1", "title": str(args.get("title") or "")})
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "structuredContent": {"id": "n1", "title": args.get("title")},
                },
            )

        if tool == "notes.search":
            q = str(args.get("query") or "")
            if (self.gate2_miss_note and "Gate2" in q) or (self.gate5_miss_note and "Gate5" in q):
                matches = []
            else:
                matches = [n for n in self.notes if q in n["title"]]
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "structuredContent": {"notes": matches, "total": len(matches)},
                },
            )

        if tool == "workspace.git_summary":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "structuredContent": {
                        "workspace": "xuna_pet",
                        "branch": "main",
                        "clean": True,
                        "modifiedCount": 0,
                        "untrackedCount": 0,
                    },
                },
            )

        if tool == "media.now_playing":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "structuredContent": {
                        "state": "playing" if self.spotify_playing else "paused",
                        "title": self.spotify_track if self.spotify_playing else "Other Song",
                        "app": self.gate4_wrong_app or "Spotify",
                    },
                },
            )

        if tool == "media.play_pause":
            self.spotify_playing = not self.spotify_playing
            return httpx.Response(
                200,
                json={"status": "ok", "structuredContent": {"state": "paused"}},
            )

        if tool == "finder.reveal":
            if not self.gate5_finder_not_front:
                self.front_app = "Finder"
                self.front_title = "xuna_pet"
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "structuredContent": {"path": args.get("path"), "revealed": True},
                },
            )

        return httpx.Response(404)

    def _handle_turn(self, body: dict[str, Any]) -> httpx.Response:
        self.turn_counter += 1
        self.turn_front_apps.append(self.front_app)
        text = str(body.get("text") or "")
        events: list[str] = []
        alias = gate_mod.DEFAULT_HOST_ALIAS
        ar_ok_payload = json.dumps({"target": alias, "status": "ok"})
        ar_ok_event = f"event: action_result\ndata: {ar_ok_payload}\n\n"
        if "trạng thái git" in text or "git status" in text:
            if self.gate2_redundant_launch_in_turn:
                self._record_audit(
                    AuditAction.HOST_TOOL_INVOKED.value,
                    {"tool": "app.launch", "status": "ok"},
                )
                events.append(ar_ok_event)
            if self.gate2_cu_in_turn:
                self._record_audit(
                    AuditAction.HOST_TOOL_INVOKED.value,
                    {"tool": "cu.session_start", "status": "ok"},
                )
                events.append(
                    'event: cu_session\ndata: {"sessionId":"cu-bad","status":"running"}\n\n'
                )
                events.append(ar_ok_event)
            self._record_audit(
                AuditAction.HOST_TOOL_INVOKED.value,
                {"tool": "workspace.git_summary", "status": "ok"},
            )
            events.append(ar_ok_event)
            events.append('event: delta\ndata: {"text":"Branch main"}\n\n')
        elif "Tìm nút tạo ghi chú" in text:
            if self.gate2_part_b_launch_in_turn:
                self._record_audit(
                    AuditAction.HOST_TOOL_INVOKED.value,
                    {"tool": "app.launch", "status": "ok"},
                )
                events.append(ar_ok_event)
            self._record_audit(
                AuditAction.HOST_TOOL_INVOKED.value,
                {"tool": "ui.find", "status": "ok"},
            )
            events.append(ar_ok_event)
            events.append('event: delta\ndata: {"text":"Found"}\n\n')
        else:
            events.append(
                'event: action_pending\ndata: {"approvalId":"ap-1","risk":"r3",'
                '"argsSha256":"' + ("a" * 64) + '","expiresAt":"2026-11-19T16:00:00+00:00"}\n\n'
            )
            if self.gate6_extra_prompt:
                events.append(
                    'event: action_pending\ndata: {"approvalId":"ap-2","risk":"r3",'
                    '"argsSha256":"' + ("b" * 64) + '","expiresAt":"2026-11-19T16:00:00+00:00"}\n\n'
                )
            if not self.gate6_omit_cu_sse:
                events.append(
                    'event: cu_session\ndata: {"sessionId":"cu-turn-6","status":"running"}\n\n'
                )
            if self.gate6_denied_action_result:
                denied_payload = json.dumps(
                    {
                        "target": alias,
                        "status": "denied",
                        "errorCode": "denylisted_target_forbidden",
                    }
                )
                events.append(f"event: action_result\ndata: {denied_payload}\n\n")
            elif not self.gate6_empty_action_results:
                self._record_audit(
                    AuditAction.HOST_TOOL_INVOKED.value,
                    {"tool": "cu.session_start", "status": "ok"},
                )
                events.append(ar_ok_event)
            if not self.gate6_omit_cu_sse:
                events.append(
                    'event: cu_session\ndata: {"sessionId":"cu-turn-6","status":"ended"}\n\n'
                )
            if self.gate6_sse_error:
                events.append('event: error\ndata: {"code":"provider_failed"}\n\n')
            if not self.gate6_omit_ended_audit:
                if self.gate6_multiple_ended_audits_mismatch:
                    self._record_audit(AuditAction.HOST_CU_SESSION_ENDED.value, {"prompts": 3})
                    self._record_audit(AuditAction.HOST_CU_SESSION_ENDED.value, {"prompts": 1})
                else:
                    prompts_val: int | bool = (
                        True
                        if self.gate6_bool_prompts_audit
                        else (2 if self.gate6_extra_prompt else 1)
                    )
                    self._record_audit(
                        AuditAction.HOST_CU_SESSION_ENDED.value,
                        {"prompts": prompts_val},
                    )
            events.append('event: delta\ndata: {"text":"Done"}\n\n')
        events.append("event: finished\ndata: {}\n\n")
        return httpx.Response(
            200,
            text="".join(events),
            headers={"content-type": "text/event-stream"},
        )


def _make_async_client(backend: _MockGateBackend) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(backend.handler))


def test_all_six_gates_pass_with_self_enrollment_and_teardown(tmp_path: Path) -> None:
    backend = _MockGateBackend()
    out_file = tmp_path / "gate-report.json"
    code = asyncio.run(
        gate_mod.async_main(
            [
                "--all",
                "--preflight-check",
                "--token",
                "test-token",
                "--pet-id",
                "pet-123",
                "--password-probe-url",
                "https://example.test/login",
                "--out",
                str(out_file),
                "--json",
            ],
            http_client=_make_async_client(backend),
            settle_s=0.0,
        )
    )
    assert code == 0
    report = json.loads(out_file.read_text(encoding="utf-8"))
    assert report["summary"]["passed"] is True
    assert report["summary"]["gatesRun"] == [1, 2, 3, 4, 5, 6]
    assert report["summary"]["falseSuccess"] == 0
    assert report["summary"]["crashes"] == 0
    assert report["summary"]["timeouts"] == 0
    assert report["summary"]["ownerPrompts"] == 1
    assert backend.deleted_hosts == [backend.enrolled_phone_id]
    assert backend.turn_front_apps[-1] == "Notes"
    assert any(ev["action"] == "host.cu_session_ended" for ev in backend.audit_events)
    assert any(ev["action"] == "host.tool_invoked" for ev in backend.audit_events)


@pytest.mark.parametrize(
    ("setup_fn", "expected_false_success"),
    [
        (lambda: _MockGateBackend(gate1_slow_ms=650.0), False),
        (lambda: _MockGateBackend(gate1_timeout=True), False),
        (lambda: _MockGateBackend(gate1_preflight_404_rfc9457=True), False),
        (lambda: _MockGateBackend(gate1_missing_screenshot=True), False),
        (lambda: _MockGateBackend(gate1_trailing_connect_error=True), False),
        (lambda: _MockGateBackend(gate2_redundant_launch_in_turn=True), False),
        (lambda: _MockGateBackend(gate2_cu_in_turn=True), False),
        (lambda: _MockGateBackend(gate2_part_b_launch_in_turn=True), False),
        (lambda: _MockGateBackend(gate2_missing_focus=True), False),
        (lambda: _MockGateBackend(gate2_stale_ref=True), False),
        (lambda: _MockGateBackend(gate2_miss_note=True), True),
        (lambda: _MockGateBackend(gate3_wrong_title=True), True),
        (lambda: _MockGateBackend(gate4_false_success_paused=True), True),
        (lambda: _MockGateBackend(gate4_wrong_app="Safari"), True),
        (lambda: _MockGateBackend(gate4_track_missing_screenshot=True), False),
        (lambda: _MockGateBackend(gate5_finder_not_front=True), True),
        (lambda: _MockGateBackend(gate5_miss_note=True), True),
        (lambda: _MockGateBackend(gate6_extra_prompt=True), False),
        (lambda: _MockGateBackend(gate6_allow_terminal=True), False),
        (lambda: _MockGateBackend(gate6_allow_password=True), False),
        (lambda: _MockGateBackend(gate6_accept_tampered_r3=True), False),
        (lambda: _MockGateBackend(gate6_unredacted_inspect_password=True), False),
        (lambda: _MockGateBackend(gate6_sse_error=True), False),
        (lambda: _MockGateBackend(gate6_denied_action_result=True), False),
        (lambda: _MockGateBackend(gate6_omit_cu_sse=True), False),
        (lambda: _MockGateBackend(gate6_omit_ended_audit=True), False),
        (lambda: _MockGateBackend(gate6_bool_prompts_audit=True), False),
        (lambda: _MockGateBackend(gate6_empty_action_results=True), False),
        (lambda: _MockGateBackend(gate6_multiple_ended_audits_mismatch=True), False),
    ],
)
def test_fail_closed_and_false_success_scenarios(
    setup_fn: Callable[[], _MockGateBackend],
    expected_false_success: bool,
    tmp_path: Path,
) -> None:
    backend = setup_fn()
    out_file = tmp_path / "fail-report.json"
    code = asyncio.run(
        gate_mod.async_main(
            [
                "--all",
                "--token",
                "test-token",
                "--pet-id",
                "pet-123",
                "--out",
                str(out_file),
                "--json",
            ],
            http_client=_make_async_client(backend),
            settle_s=0.0,
        )
    )
    assert code == 1
    report = json.loads(out_file.read_text(encoding="utf-8"))
    assert report["summary"]["passed"] is False
    if expected_false_success:
        assert report["summary"]["falseSuccess"] >= 1


def test_preflight_check_flags_missing_permissions_and_exposed_ax_tree(
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad_perms = _MockGateBackend(accessibility="denied")
    code1 = asyncio.run(
        gate_mod.async_main(
            ["--preflight-check", "--token", "tok", "--json"],
            http_client=_make_async_client(bad_perms),
        )
    )
    assert code1 == 1
    _ = capsys.readouterr()

    denied_sr = _MockGateBackend(screen_recording="denied")
    code_sr = asyncio.run(
        gate_mod.async_main(
            ["--preflight-check", "--token", "tok", "--json"],
            http_client=_make_async_client(denied_sr),
        )
    )
    assert code_sr == 1
    sr_payload = json.loads(capsys.readouterr().out)
    assert sr_payload["preflight"]["screenRecording"] == "denied"

    exposed_ax = _MockGateBackend(expose_ax_tree=True)
    code2 = asyncio.run(
        gate_mod.async_main(
            ["--preflight-check", "--token", "tok", "--json"],
            http_client=_make_async_client(exposed_ax),
        )
    )
    assert code2 == 1
    _ = capsys.readouterr()

    ok_perms = _MockGateBackend()
    code_ok = asyncio.run(
        gate_mod.async_main(
            ["--preflight-check", "--token", "tok", "--json"],
            http_client=_make_async_client(ok_perms),
        )
    )
    assert code_ok == 0
    ok_payload = json.loads(capsys.readouterr().out)
    assert ok_payload["preflight"]["screenRecording"] == "granted"


def test_cli_gate_selection_and_missing_token(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("XUNA_OWNER_TOKEN", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(gate_mod.async_main(["--all"]))
    assert exc_info.value.code == 2

    dry_code = asyncio.run(gate_mod.async_main(["--dry-run", "--json"]))
    assert dry_code == 0
    dry_payload = json.loads(capsys.readouterr().out)
    assert dry_payload["mode"] == "dry-run"
    assert dry_payload["signerSelfTest"]["approvalVerified"] is True
    assert len(dry_payload["plan"]) == 6

    backend = _MockGateBackend()
    out_file = tmp_path / "subset.json"
    sub_code = asyncio.run(
        gate_mod.async_main(
            ["--gate", "1", "--gate", "6", "--token", "tok", "--out", str(out_file)],
            http_client=_make_async_client(backend),
            settle_s=0.0,
        )
    )
    assert sub_code == 0
    _ = capsys.readouterr()
    subset_report = json.loads(out_file.read_text(encoding="utf-8"))
    assert subset_report["summary"]["gatesRun"] == [1, 6]
    assert backend.deleted_hosts == [backend.enrolled_phone_id]

    backend_comma = _MockGateBackend()
    comma_code = asyncio.run(
        gate_mod.async_main(
            ["--gate", "3,5", "--token", "tok", "--json"],
            http_client=_make_async_client(backend_comma),
            settle_s=0.0,
        )
    )
    assert comma_code == 0
    comma_payload = json.loads(capsys.readouterr().out)
    assert comma_payload["summary"]["gatesRun"] == [3, 5]


def test_gate_client_rejects_deprecated_accessibility_tree() -> None:
    backend = _MockGateBackend()

    async def _run() -> None:
        async with gate_mod.GateClient(
            "http://127.0.0.1:8000",
            "tok",
            http_client=_make_async_client(backend),
        ) as client:
            with pytest.raises(ValueError, match="deprecated"):
                await client.invoke_tool(gate_mod.DEFAULT_HOST_ID, "ui.accessibility_tree", {})
            with pytest.raises(ValueError, match="deprecated"):
                await client.preflight_tool(gate_mod.DEFAULT_HOST_ID, "ui.accessibility_tree", {})

    asyncio.run(_run())


def test_rfc9457_404_and_422_client_parsing() -> None:
    def _rfc_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/tools:preflight"):
            return httpx.Response(
                404,
                json={
                    "type": "https://xuna.dev/problems/not_found",
                    "title": "Not Found",
                    "status": 404,
                    "code": "not_found",
                },
            )
        return httpx.Response(
            422,
            json={
                "type": "https://xuna.dev/problems/validation_failed",
                "title": "Unprocessable Entity",
                "status": 422,
                "code": "validation_failed",
            },
        )

    async def _run() -> None:
        http = httpx.AsyncClient(transport=httpx.MockTransport(_rfc_handler))
        async with gate_mod.GateClient("http://127.0.0.1:8000", "tok", http_client=http) as client:
            pre = await client.preflight_tool(gate_mod.DEFAULT_HOST_ID, "input.click", {})
            assert pre.http_status == 404
            assert pre.status == "error"
            assert pre.error_code == "not_found"

            inv = await client.invoke_tool(gate_mod.DEFAULT_HOST_ID, "input.click", {})
            assert inv.http_status == 422
            assert inv.status == "error"
            assert inv.is_error is True
            assert inv.error_code == "validation_failed"

    asyncio.run(_run())


def test_mock_backend_enforces_r2_approve_false_and_stale_snapshot_ref() -> None:
    backend = _MockGateBackend()
    signer = gate_mod.DevPhoneSigner()

    async def _run() -> None:
        async with gate_mod.GateClient(
            "http://127.0.0.1:8000",
            "tok",
            http_client=_make_async_client(backend),
        ) as client:
            await client.enroll_dev_phone(signer)
            r2_denied = await client.invoke_tool(
                gate_mod.DEFAULT_HOST_ID,
                "input.type_text",
                {"elementId": "@s00000001:e1", "text": "hello"},
                approve=False,
            )
            assert r2_denied.status == "denied"
            assert r2_denied.error_code == "approval_required"

            cu_args: dict[str, Any] = {
                "goal": "Check snapshot increment",
                "apps": ["notes"],
                "domains": [],
                "maxSteps": 10,
                "maxMinutes": 5,
            }
            cu_res = await client.invoke_tool(
                gate_mod.DEFAULT_HOST_ID,
                "cu.session_start",
                cu_args,
                approve=True,
                approval=gate_mod._sign_cu_session_start(signer, gate_mod.DEFAULT_HOST_ID, cu_args),
            )
            assert cu_res.structured_content["snapshotId"] == "s00000001"
            inspect_res = await client.invoke_tool(
                gate_mod.DEFAULT_HOST_ID,
                "ui.inspect",
                {"scope": "active_window", "maxNodes": 20},
            )
            assert inspect_res.structured_content["snapshotId"] == "s00000002"
            stale_click = await client.invoke_tool(
                gate_mod.DEFAULT_HOST_ID,
                "input.click",
                {"elementId": "@s00000001:e1"},
                approve=True,
            )
            assert stale_click.error_code == "stale_ref"

    asyncio.run(_run())


def test_gate3_cold_launch_polling_and_gate4_ui_find_fallback() -> None:
    backend_g3 = _MockGateBackend(gate3_warm_poll_flips=2)

    async def _run_g3() -> None:
        async with gate_mod.GateClient(
            "http://127.0.0.1:8000",
            "tok",
            http_client=_make_async_client(backend_g3),
        ) as client:
            res = await gate_mod.run_gate_3(
                client, gate_mod.DEFAULT_HOST_ID, settle_s=0.01, poll_timeout_s=1.0
            )
            assert res.passed is True
            assert backend_g3.vscode_context_polls == 2

    asyncio.run(_run_g3())

    backend_g4 = _MockGateBackend(gate4_non_empty_unrelated_observe=True)
    signer = gate_mod.DevPhoneSigner()

    async def _run_g4() -> None:
        async with gate_mod.GateClient(
            "http://127.0.0.1:8000",
            "tok",
            http_client=_make_async_client(backend_g4),
        ) as client:
            await client.enroll_dev_phone(signer)
            res = await gate_mod.run_gate_4(client, gate_mod.DEFAULT_HOST_ID, signer, settle_s=0.0)
            assert res.passed is True
            assert res.false_success == 0
            assert "ui.find" in backend_g4.invoked_tools

    asyncio.run(_run_g4())


def test_gate6_password_probe_url_catches_leaked_text_field() -> None:
    backend = _MockGateBackend(gate6_leak_password_text=True)
    signer = gate_mod.DevPhoneSigner()
    probe_url = "https://example.test/login"

    async def _run() -> None:
        async with gate_mod.GateClient(
            "http://127.0.0.1:8000",
            "tok",
            http_client=_make_async_client(backend),
        ) as client:
            await client.enroll_dev_phone(signer)
            res = await gate_mod.run_gate_6(
                client,
                gate_mod.DEFAULT_HOST_ID,
                signer,
                password_probe_url=probe_url,
            )
            assert res.passed is False
            check = next(c for c in res.checks if c.name == "live_password_redaction_and_block")
            assert check.passed is False
            assert "leaked_text" in check.detail

    asyncio.run(_run())


def test_r3_hardening_regressions() -> None:
    assert AuditAction.HOST_CU_SESSION_ENDED.value == "host.cu_session_ended"
    assert AuditAction.HOST_TOOL_INVOKED.value == "host.tool_invoked"

    nan_som, nan_msg = gate_mod.validate_jpeg_som_payload(
        {
            "nodes": [
                {"id": "ui_1", "ref": "@s00000001:e1", "bounds": [0.1, float("nan"), 0.3, 0.4]}
            ],
        }
    )
    assert nan_som is False
    assert "out of [0, 1]" in nan_msg

    bool_dim_som, bool_dim_msg = gate_mod.validate_jpeg_som_payload(
        {
            "nodes": [],
            "screenshot": {
                "mimeType": "image/jpeg",
                "width": True,
                "height": 480,
                "dataBase64": _SAMPLE_JPEG_B64,
            },
        },
        require_screenshot=True,
    )
    assert bool_dim_som is False
    assert "invalid screenshot dimensions" in bool_dim_msg

    assert gate_mod._pick_safe_ref_node(
        [
            {
                "id": "ui_1",
                "ref": "@s00000001:e1",
                "role": "button",
                "name": "Xo\u0301a ghi chu\u0301",
                "enabled": True,
            },
            {
                "id": "ui_2",
                "ref": "@s00000001:e2",
                "role": "button",
                "name": "Xoá mục",
                "enabled": True,
            },
            {
                "id": "ui_3",
                "ref": "@s00000001:e3",
                "role": "button",
                "name": "Gu\u031b\u0309i email",
                "enabled": True,
            },
            {
                "id": "ui_4",
                "ref": "@s00000001:e4",
                "role": "button",
                "name": "Ghi chu\u0301 m\u01a1\u0301i",
                "enabled": True,
            },
        ]
    ) == {
        "id": "ui_4",
        "ref": "@s00000001:e4",
        "role": "button",
        "name": "Ghi chu\u0301 m\u01a1\u0301i",
        "enabled": True,
    }

    assert (
        gate_mod._audit_events_delta(
            [{"eventId": None, "action": "x"}],
            [{"eventId": None, "action": "x"}],
        )
        == []
    )
    assert gate_mod._audit_events_delta(
        [{"eventId": None, "action": "x"}, {"eventId": "ev-1", "action": "y"}],
        [
            {"eventId": None, "action": "x"},
            {"eventId": "ev-1", "action": "y"},
            {"eventId": "ev-2", "action": "z"},
        ],
    ) == [{"eventId": "ev-2", "action": "z"}]

    def _preflight_denied_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/tools:preflight"):
            return httpx.Response(
                200,
                json={
                    "status": "denied",
                    "errorCode": "password_field_forbidden",
                    "effectiveRisk": "",
                },
            )
        raise AssertionError(f"tools:invoke should not be called after denied preflight: {request}")

    async def _run_preflight() -> None:
        http = httpx.AsyncClient(transport=httpx.MockTransport(_preflight_denied_handler))
        async with gate_mod.GateClient("http://127.0.0.1:8000", "tok", http_client=http) as client:
            outcome = await gate_mod._invoke_input_with_preflight(
                client,
                gate_mod.DEFAULT_HOST_ID,
                gate_mod.DevPhoneSigner(),
                "cu-session-1",
                "input.type_text",
                {"elementId": "@s00000001:e9", "text": "secret"},
            )
            assert outcome.status == "denied"
            assert outcome.is_error is True
            assert outcome.error_code == "password_field_forbidden"

    asyncio.run(_run_preflight())


def test_gate6_reinspects_after_draft_so_mutated_ref_is_not_stale() -> None:
    backend = _MockGateBackend()
    signer = gate_mod.DevPhoneSigner()
    backend.enrolled_pub_key = signer.public_key_bytes

    async def _run() -> Any:
        async with gate_mod.GateClient(
            "http://127.0.0.1:8000", "tok", http_client=_make_async_client(backend)
        ) as client:
            return await gate_mod.run_gate_6(
                client,
                gate_mod.DEFAULT_HOST_ID,
                signer,
                pet_id="pet-123",
            )

    res = asyncio.run(_run())
    assert res.passed is True, res.checks
    assert backend.dropped_refs  # draft type_text dropped pre-draft ref
    pw_check = next(c for c in res.checks if c.name == "password_field_blocked")
    assert pw_check.passed is True
    assert "password_field_forbidden" in pw_check.detail
    assert backend.invoked_tools.count("ui.inspect") >= 2


def test_pick_track_node_prefers_compact_song_row_over_link_and_top_result_card() -> None:
    picked = gate_mod._pick_track_node(
        [
            {
                "id": "ui_2",
                "ref": "@s00000001:e2",
                "role": "link",
                "name": "Anh Sai Rồi",
                "bounds": [0.37, 0.19, 0.48, 0.22],
            },
            {
                "id": "ui_21",
                "ref": "@s00000001:e21",
                "role": "list_item",
                "name": "Anh Sai Rồi",
                "bounds": [0.28, 0.16, 0.72, 0.29],
            },
            {
                "id": "ui_22",
                "ref": "@s00000001:e22",
                "role": "list_item",
                "name": "Anh Sai Rồi",
                "bounds": [0.28, 0.30, 0.72, 0.38],
            },
        ]
    )
    assert picked is not None
    assert picked["ref"] == "@s00000001:e22"
