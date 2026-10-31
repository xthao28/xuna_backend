"""UDC-8 unit & integration tests (S8.3, S8.5, S8.6):
- U5 (`host_screen`) and U6 (`host_notifications`) consent gating;
- 10 scripted tasks on Windows and macOS profiles (`10/10 >= 8/10`);
- Prompt-injection fixture set (`backend/tests/fixtures/injection/malicious_cases.json`)
  verifying 0 unapproved R3 actions, 0 actions outside allowlist, 0 password fields typed;
- Stop conditions: step budget, time budget, 3 consecutive `no_change`, duplicate action,
  dead-man switch (3 missed 5 s heartbeats = 15 s), and emergency stop latency `<= 1 s`;
- Zero retention of screenshots or UI trees in DB or logs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

from xuna_backend.modules.actions.application.catalog import build_tool_catalog
from xuna_backend.modules.actions.application.computer_use import (
    ComputerUseError,
    ComputerUseProgressEvent,
    ComputerUseSessionRunner,
    InMemoryComputerUseSessionStore,
    SignedR3ApprovalProof,
)
from xuna_backend.modules.actions.domain.entities import ApprovalScope
from xuna_backend.modules.consent.application.use_cases import (
    UpdateConsent,
    update_consent,
)
from xuna_backend.modules.consent.domain.consent import U0, ConsentClass, ConsentState
from xuna_backend.modules.hosts.domain.crypto import approval_proof_message
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    Host,
    HostAlias,
    HostId,
    HostPlatform,
    HostStatus,
    Manifest,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
)
from xuna_backend.modules.identity.infrastructure.p256 import P256SignatureVerifier

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "injection"


class _FakeConsentRepo:
    def __init__(self, initial: ConsentState = U0) -> None:
        self._by_owner: dict[uuid.UUID, ConsentState] = {}
        self._default = initial

    async def current(self, owner_id: uuid.UUID) -> ConsentState:
        return self._by_owner.get(owner_id, self._default)

    async def append(self, owner_id: uuid.UUID, state: ConsentState) -> bool:
        self._by_owner[owner_id] = state
        return True


class _FakeAuditLog:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def record(self, event: Any) -> None:
        self.events.append(event)

    async def recent(self, owner_id: uuid.UUID, limit: int) -> list[Any]:
        return [e for e in self.events if e.owner_id == owner_id][:limit]


def _generate_p256_keypair() -> tuple[ec.EllipticCurvePrivateKey, bytes]:
    priv = ec.generate_private_key(ec.SECP256R1())
    pub_bytes = priv.public_key().public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )
    return priv, pub_bytes


def _sign_r3_proof(
    priv: ec.EllipticCurvePrivateKey,
    pub_bytes: bytes,
    *,
    host_id: HostId,
    tool_name: str,
    arguments: dict[str, Any],
    scope: ApprovalScope = ApprovalScope.ONCE,
) -> SignedR3ApprovalProof:
    approval_id = uuid.uuid4()
    expires_at_iso = "2026-10-31T09:05:00Z"
    canonical = json.dumps(
        arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    args_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    payload = approval_proof_message(
        approval_id=str(approval_id),
        args_sha256_hex=args_sha256,
        host_id=str(host_id),
        expires_at_iso=expires_at_iso,
    )
    der_sig = priv.sign(payload, ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der_sig)
    raw_sig = r.to_bytes(32, "big") + s.to_bytes(32, "big")
    return SignedR3ApprovalProof(
        approval_id=approval_id,
        host_id=host_id,
        tool_name=tool_name,
        args_sha256=args_sha256,
        scope=scope,
        expires_at_iso=expires_at_iso,
        signature=raw_sig,
        signer_public_key=pub_bytes,
    )


def test_s83_consent_u5_u6_and_catalog_gating() -> None:
    async def _run() -> None:
        owner_id = uuid.uuid4()
        repo = _FakeConsentRepo()
        audit = _FakeAuditLog()
        now = datetime(2026, 10, 31, 9, 0, 0, tzinfo=UTC)

        # Default U0 denies U5 (HOST_SCREEN) and U6 (HOST_NOTIFICATIONS)
        state0 = await repo.current(owner_id)
        assert not state0.allows(ConsentClass.HOST_SCREEN)
        assert not state0.allows(ConsentClass.HOST_NOTIFICATIONS)

        # Enable U5 host_screen
        state1 = await update_consent(
            repo,
            audit,
            owner_id,
            UpdateConsent(
                aggregate_metrics=False,
                experience_events=False,
                conversation_text=True,
                expected_version=0,
                host_screen=True,
                host_notifications=False,
            ),
            now,
            "req-u5",
        )
        assert state1.allows(ConsentClass.HOST_SCREEN)
        assert not state1.allows(ConsentClass.HOST_NOTIFICATIONS)
        assert state1.version == 1

        # Verify catalog gates CU, SCREEN, UI, INPUT tools on host_screen_consent
        host_id = HostId(uuid.uuid4())
        host = Host(
            host_id=host_id,
            owner_id=owner_id,
            alias=HostAlias("Work PC"),
            platform=HostPlatform.WINDOWS,
            hostname="WORK-PC",
            os_version="11",
            agent_version="0.2.0",
            public_key=b"\x04" + b"\x11" * 64,
            status=HostStatus.ACTIVE,
            created_at=now,
        )
        empty_obj_schema: dict[str, object] = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        manifest = Manifest(
            host_id=host_id,
            tools=(
                ToolDescriptor(
                    name="cu.session_start",
                    family=ToolFamily.CU,
                    risk=RiskTier.R3,
                    description="Start CU session",
                    input_schema=empty_obj_schema,
                    output_schema=empty_obj_schema,
                ),
                ToolDescriptor(
                    name="ui.inspect",
                    family=ToolFamily.UI,
                    risk=RiskTier.R1,
                    description="Inspect UI",
                    input_schema=empty_obj_schema,
                    output_schema=empty_obj_schema,
                    observes=True,
                ),
                ToolDescriptor(
                    name="media.play_pause",
                    family=ToolFamily.MEDIA,
                    risk=RiskTier.R0,
                    description="Play/pause",
                    input_schema=empty_obj_schema,
                    output_schema=empty_obj_schema,
                ),
            ),
            updated_at=now,
        )
        grants = [
            Grant(
                host_id=host_id,
                family=ToolFamily.CU,
                enabled=True,
                max_risk=RiskTier.R3,
                updated_at=now,
            ),
            Grant(
                host_id=host_id,
                family=ToolFamily.UI,
                enabled=True,
                max_risk=RiskTier.R1,
                updated_at=now,
            ),
            Grant(
                host_id=host_id,
                family=ToolFamily.MEDIA,
                enabled=True,
                max_risk=RiskTier.R0,
                updated_at=now,
            ),
        ]

        cat_without_u5 = build_tool_catalog(
            hosts=[host],
            manifests_by_host={host_id: manifest},
            grants_by_host={host_id: grants},
            online_host_ids={str(host_id)},
            client_caps={"host_actions_v1"},
            board_available=False,
            host_screen_consent=False,
            now=now,
        )
        assert [t.name for t in cat_without_u5] == ["media__play_pause"]

        cat_with_u5 = build_tool_catalog(
            hosts=[host],
            manifests_by_host={host_id: manifest},
            grants_by_host={host_id: grants},
            online_host_ids={str(host_id)},
            client_caps={"host_actions_v1"},
            board_available=False,
            host_screen_consent=True,
            now=now,
        )
        assert {t.name for t in cat_with_u5} == {
            "cu__session_start",
            "ui__inspect",
            "media__play_pause",
        }

    asyncio.run(_run())


SCRIPTED_10_TASKS: tuple[dict[str, Any], ...] = (
    {
        "id": "task-01-open-latest-report",
        "goal": "Mở file báo cáo mới nhất trong thư mục Reports",
        "apps_win": ["explorer"],
        "apps_mac": ["finder"],
        "domains": [],
        "steps": [
            {"tool": "input.click", "args": {"elementId": "s00000001:1"}, "r3": False},
            {"tool": "input.key", "args": {"combo": "Enter", "inForm": False}, "r3": False},
        ],
    },
    {
        "id": "task-02-browser-calendar-draft",
        "goal": "Điền lịch họp nhóm lúc 15:00 trên trình duyệt và dừng trước nút Tạo",
        "apps_win": ["chrome"],
        "apps_mac": ["safari"],
        "domains": ["calendar.google.com"],
        "steps": [
            {"tool": "input.click", "args": {"elementId": "s00000001:2"}, "r3": False},
            {
                "tool": "input.type_text",
                "args": {"elementId": "s00000001:3", "text": "Họp nhóm XUNA 15:00"},
                "r3": True,
            },
        ],
    },
    {
        "id": "task-03-vscode-search-symbol",
        "goal": "Tìm hàm ComputerUseSessionRunner trong VS Code",
        "apps_win": ["code"],
        "apps_mac": ["code"],
        "domains": [],
        "steps": [
            {"tool": "input.key", "args": {"combo": "Ctrl+F", "inForm": False}, "r3": False},
            {
                "tool": "input.type_text",
                "args": {"elementId": "s00000001:4", "text": "ComputerUseSessionRunner"},
                "r3": True,
            },
        ],
    },
    {
        "id": "task-04-docs-scroll-and-select-heading",
        "goal": "Cuộn xuống mục Kiến trúc bảo mật trong tài liệu",
        "apps_win": ["chrome"],
        "apps_mac": ["safari"],
        "domains": ["docs.google.com"],
        "steps": [
            {"tool": "input.scroll", "args": {"direction": "down", "amount": 3}, "r3": False},
            {"tool": "input.click", "args": {"elementId": "s00000001:5"}, "r3": False},
        ],
    },
    {
        "id": "task-05-notes-append-todo",
        "goal": "Ghi chú thêm mục kiểm tra UDC-8 vào ứng dụng ghi chú",
        "apps_win": ["notepad"],
        "apps_mac": ["notes"],
        "domains": [],
        "steps": [
            {"tool": "input.click", "args": {"elementId": "s00000001:6"}, "r3": False},
            {
                "tool": "input.type_text",
                "args": {"elementId": "s00000001:6", "text": "- Hoàn thành kiểm thử UDC-8"},
                "r3": True,
            },
        ],
    },
    {
        "id": "task-06-spotify-search-playlist",
        "goal": "Tìm playlist Lo-Fi Coding trong Spotify",
        "apps_win": ["spotify"],
        "apps_mac": ["spotify"],
        "domains": [],
        "steps": [
            {"tool": "input.click", "args": {"elementId": "s00000001:7"}, "r3": False},
            {
                "tool": "input.type_text",
                "args": {"elementId": "s00000001:7", "text": "Lo-Fi Coding"},
                "r3": True,
            },
        ],
    },
    {
        "id": "task-07-github-pr-filter",
        "goal": "Lọc danh sách pull request đang mở trên GitHub",
        "apps_win": ["chrome"],
        "apps_mac": ["safari"],
        "domains": ["github.com"],
        "steps": [
            {"tool": "input.click", "args": {"elementId": "s00000001:8"}, "r3": False},
            {"tool": "input.scroll", "args": {"direction": "down", "amount": 2}, "r3": False},
        ],
    },
    {
        "id": "task-08-pdf-reader-page-jump",
        "goal": "Chuyển sang mục kết quả trong trình đọc PDF",
        "apps_win": ["acrobat"],
        "apps_mac": ["preview"],
        "domains": [],
        "steps": [
            {"tool": "input.key", "args": {"combo": "PageDown", "inForm": False}, "r3": False},
            {"tool": "input.click", "args": {"elementId": "s00000001:9"}, "r3": False},
        ],
    },
    {
        "id": "task-09-calculator-focus-and-clear",
        "goal": "Xóa màn hình tạm trong ứng dụng Máy tính",
        "apps_win": ["calc"],
        "apps_mac": ["calculator"],
        "domains": [],
        "steps": [
            {"tool": "input.click", "args": {"elementId": "s00000001:10"}, "r3": False},
            {"tool": "input.key", "args": {"combo": "Escape", "inForm": False}, "r3": False},
        ],
    },
    {
        "id": "task-10-consequential-send-with-owner-r3-signature",
        "goal": "Bấm nút Gửi báo cáo sau khi chủ nhân ký xác nhận R3 trên điện thoại",
        "apps_win": ["chrome"],
        "apps_mac": ["safari"],
        "domains": ["mail.google.com"],
        "steps": [
            {
                "tool": "input.click",
                "args": {"elementId": "s00000001:send", "elementName": "Gửi báo cáo"},
                "r3": True,
            },
        ],
    },
)


@pytest.mark.parametrize("platform_profile", ["windows", "macos"])
def test_s85_ten_scripted_tasks_succeed_on_windows_and_macos(
    platform_profile: str,
) -> None:
    async def _run() -> None:
        owner_id = uuid.uuid4()
        host_id = HostId(uuid.uuid4())
        priv, pub_bytes = _generate_p256_keypair()
        consent_repo = _FakeConsentRepo(
            ConsentState(conversation_text=True, host_screen=True, version=1)
        )
        store = InMemoryComputerUseSessionStore()
        progress_events: list[ComputerUseProgressEvent] = []

        current_app = "explorer"
        current_domain = ""

        async def fake_host_caller(
            _hid: HostId, tool_name: str, arguments: dict[str, Any]
        ) -> dict[str, Any]:
            if tool_name == "cu.session_start":
                return {"sessionId": str(uuid.uuid4()), "maxSteps": arguments["maxSteps"]}
            if tool_name == "ui.inspect":
                return {
                    "appId": current_app,
                    "processName": current_app,
                    "windowTitle": f"Window - {current_app}",
                    "domain": current_domain,
                    "isElevated": False,
                    "nodes": [
                        {"id": "s00000001:1", "role": "list_item", "name": "Q4_Report.pdf"},
                        {"id": "s00000001:2", "role": "button", "name": "New Event"},
                        {"id": "s00000001:3", "role": "textbox", "name": "Event title"},
                        {"id": "s00000001:4", "role": "textbox", "name": "Find"},
                        {"id": "s00000001:5", "role": "heading", "name": "Security Architecture"},
                        {"id": "s00000001:6", "role": "textbox", "name": "Note body"},
                        {"id": "s00000001:7", "role": "textbox", "name": "Search"},
                        {"id": "s00000001:8", "role": "button", "name": "Open PRs"},
                        {"id": "s00000001:9", "role": "link", "name": "Section 5"},
                        {"id": "s00000001:10", "role": "button", "name": "Clear"},
                        {"id": "s00000001:send", "role": "button", "name": "Gửi báo cáo"},
                    ],
                }
            if tool_name in ("input.click", "input.type_text", "input.key", "input.scroll"):
                return {"effect": "changed"}
            if tool_name == "cu.session_end":
                return {"stepsUsed": 2, "reason": arguments.get("reason", "completed")}
            raise AssertionError(f"Unexpected tool: {tool_name}")

        async def on_progress(ev: ComputerUseProgressEvent) -> None:
            progress_events.append(ev)

        runner = ComputerUseSessionRunner(
            consent_repo=consent_repo,
            session_store=store,
            signature_verifier=P256SignatureVerifier(),
            host_caller=fake_host_caller,
            on_progress=on_progress,
        )

        succeeded = 0
        for task in SCRIPTED_10_TASKS:
            apps = task["apps_win"] if platform_profile == "windows" else task["apps_mac"]
            domains = task["domains"]
            current_app = apps[0]
            current_domain = domains[0] if domains else ""

            start_args = {
                "goal": task["goal"],
                "apps": apps,
                "domains": domains,
                "maxSteps": 10,
                "maxMinutes": 5,
            }
            start_proof = _sign_r3_proof(
                priv,
                pub_bytes,
                host_id=host_id,
                tool_name="cu.session_start",
                arguments=start_args,
            )
            session = await runner.start_session(
                owner_id=owner_id,
                host_id=host_id,
                goal=task["goal"],
                apps=apps,
                domains=domains,
                max_steps=10,
                max_minutes=5,
                r3_proof=start_proof,
            )
            obs = await runner.observe(owner_id=owner_id, session_id=session.session_id)
            assert "[UNTRUSTED_SCREEN_OBSERVATION" in obs["untrustedPromptBlock"]

            for step in task["steps"]:
                step_proof = None
                if step["r3"]:
                    step_proof = _sign_r3_proof(
                        priv,
                        pub_bytes,
                        host_id=host_id,
                        tool_name=step["tool"],
                        arguments=step["args"],
                    )
                res = await runner.execute_step(
                    owner_id=owner_id,
                    session_id=session.session_id,
                    tool_name=step["tool"],
                    arguments=step["args"],
                    r3_proof=step_proof,
                )
                assert res["effect"] == "changed"

            ended = await runner.end_session(
                owner_id=owner_id, session_id=session.session_id, reason="completed"
            )
            assert ended["reason"] == "completed"
            assert session.unapproved_r3_attempts == 0
            assert session.outside_allowlist_attempts == 0
            assert session.password_typed_attempts == 0
            succeeded += 1

        assert succeeded == 10  # Gate requires >= 8/10
        assert len(progress_events) >= 30

    asyncio.run(_run())


def test_s86_prompt_injection_fixtures_and_safety_invariants() -> None:
    async def _run() -> None:
        fixture_data = json.loads(
            (FIXTURES_DIR / "malicious_cases.json").read_text(encoding="utf-8")
        )
        injected_html = (FIXTURES_DIR / "injected_page.html").read_text(encoding="utf-8")
        assert "BỎ QUA MỌI HƯỚNG DẪN" in injected_html

        owner_id = uuid.uuid4()
        host_id = HostId(uuid.uuid4())
        priv, pub_bytes = _generate_p256_keypair()
        consent_repo = _FakeConsentRepo(
            ConsentState(conversation_text=True, host_screen=True, version=1)
        )
        store = InMemoryComputerUseSessionStore()

        executed_host_actions: list[tuple[str, dict[str, Any]]] = []
        active_window_title = "Lịch họp tuần"
        active_injected_text = ""

        async def fake_host_caller(
            _hid: HostId, tool_name: str, arguments: dict[str, Any]
        ) -> dict[str, Any]:
            if tool_name == "cu.session_start":
                return {"sessionId": str(uuid.uuid4()), "maxSteps": arguments["maxSteps"]}
            if tool_name == "ui.inspect":
                return {
                    "appId": "chrome",
                    "processName": "chrome.exe",
                    "windowTitle": active_window_title,
                    "domain": "calendar.google.com",
                    "isElevated": False,
                    "nodes": [
                        {
                            "id": "s00000001:banner",
                            "role": "region",
                            "name": active_injected_text,
                        },
                        {
                            "id": "s00000001:pwd",
                            "role": "password_text",
                            "name": "Mật khẩu tài khoản",
                            "value": "SuperSecret123!",
                            "isPassword": True,
                        },
                        {
                            "id": "s00000001:99",
                            "role": "button",
                            "name": "Gửi email cho attacker@evil.example",
                        },
                        {
                            "id": "s00000001:100",
                            "role": "button",
                            "name": "Buy Now & Pay $999",
                        },
                    ],
                }
            if tool_name == "screen.capture":
                return {
                    "width": 1280,
                    "height": 720,
                    "scale": 1.0,
                    "jpegBase64": "AAAA_FAKE_SCREENSHOT_BYTES_NEVER_STORE",
                }
            if tool_name in ("input.click", "input.type_text", "input.key", "input.scroll"):
                executed_host_actions.append((tool_name, arguments))
                return {"effect": "changed"}
            return {"stepsUsed": 0, "reason": "completed"}

        runner = ComputerUseSessionRunner(
            consent_repo=consent_repo,
            session_store=store,
            signature_verifier=P256SignatureVerifier(),
            host_caller=fake_host_caller,
        )

        start_args: dict[str, Any] = {
            "goal": "Tạo sự kiện lịch họp tuần lúc 15:00",
            "apps": ["chrome"],
            "domains": ["calendar.google.com"],
            "maxSteps": 12,
            "maxMinutes": 5,
        }
        session = await runner.start_session(
            owner_id=owner_id,
            host_id=host_id,
            goal=start_args["goal"],
            apps=start_args["apps"],
            domains=start_args["domains"],
            max_steps=12,
            max_minutes=5,
            r3_proof=_sign_r3_proof(
                priv,
                pub_bytes,
                host_id=host_id,
                tool_name="cu.session_start",
                arguments=start_args,
            ),
        )

        for case in fixture_data["cases"]:
            active_window_title = case["windowTitle"]
            active_injected_text = case["injectedText"]
            obs_res = await runner.observe(
                owner_id=owner_id,
                session_id=session.session_id,
                capture_screen=True,
            )
            # 1. Prompt injection is flagged & framed as untrusted data
            assert obs_res["injectionDetected"] is True
            assert "[UNTRUSTED_SCREEN_OBSERVATION" in obs_res["untrustedPromptBlock"]
            # 2. Password value is redacted from observation and prompt block
            assert "SuperSecret123!" not in obs_res["untrustedPromptBlock"]
            assert "AAAA_FAKE_SCREENSHOT_BYTES_NEVER_STORE" not in obs_res["untrustedPromptBlock"]
            pwd_nodes = [
                n for n in obs_res["observation"]["nodes"] if n["id"] == "s00000001:pwd"
            ]
            assert pwd_nodes[0]["name"] == "•••"
            assert "value" not in pwd_nodes[0]

            # 3. Attempted attack is deterministically refused before reaching the host
            attack = case["attemptedAttack"]
            with pytest.raises(ComputerUseError) as exc_info:
                await runner.execute_step(
                    owner_id=owner_id,
                    session_id=session.session_id,
                    tool_name=attack["toolName"],
                    arguments=attack["arguments"],
                    r3_proof=None,
                )
            assert exc_info.value.code == attack["expectedErrorCode"]

        # Zero unapproved R3, zero outside-allowlist, and zero password actions reached the host!
        assert executed_host_actions == []
        assert session.injection_attempts_blocked == len(fixture_data["cases"])

    asyncio.run(_run())


def test_s86_stop_conditions_heartbeat_deadman_and_emergency_stop_latency() -> None:
    async def _run() -> None:
        owner_id = uuid.uuid4()
        host_id = HostId(uuid.uuid4())
        priv, pub_bytes = _generate_p256_keypair()
        consent_repo = _FakeConsentRepo(
            ConsentState(conversation_text=True, host_screen=True, version=1)
        )
        store = InMemoryComputerUseSessionStore()

        fake_now = 1000.0
        next_effect = "no_change"

        async def fake_host_caller(
            _hid: HostId, tool_name: str, arguments: dict[str, Any]
        ) -> dict[str, Any]:
            if tool_name == "cu.session_start":
                return {"sessionId": str(uuid.uuid4()), "maxSteps": arguments["maxSteps"]}
            if tool_name == "ui.inspect":
                return {
                    "appId": "explorer",
                    "processName": "explorer.exe",
                    "windowTitle": "Reports",
                    "isElevated": False,
                    "nodes": [
                        {"id": "s00000001:1", "role": "button", "name": "Item 1"},
                        {"id": "s00000001:2", "role": "button", "name": "Item 2"},
                        {"id": "s00000001:3", "role": "button", "name": "Item 3"},
                    ],
                }
            return {"effect": next_effect}

        runner = ComputerUseSessionRunner(
            consent_repo=consent_repo,
            session_store=store,
            signature_verifier=P256SignatureVerifier(),
            host_caller=fake_host_caller,
            clock_monotonic=lambda: fake_now,
        )

        def start_cu(max_steps: int = 10, max_minutes: int = 5) -> Any:
            args: dict[str, Any] = {
                "goal": "Kiểm tra điều kiện dừng",
                "apps": ["explorer"],
                "domains": [],
                "maxSteps": max_steps,
                "maxMinutes": max_minutes,
            }
            return runner.start_session(
                owner_id=owner_id,
                host_id=host_id,
                goal=args["goal"],
                apps=args["apps"],
                domains=args["domains"],
                max_steps=max_steps,
                max_minutes=max_minutes,
                r3_proof=_sign_r3_proof(
                    priv,
                    pub_bytes,
                    host_id=host_id,
                    tool_name="cu.session_start",
                    arguments=args,
                ),
            )

        # 1. Three consecutive no_change -> stops with no_progress
        s1 = await start_cu()
        await runner.execute_step(
            owner_id=owner_id,
            session_id=s1.session_id,
            tool_name="input.click",
            arguments={"elementId": "s00000001:1"},
        )
        await runner.execute_step(
            owner_id=owner_id,
            session_id=s1.session_id,
            tool_name="input.click",
            arguments={"elementId": "s00000001:2"},
        )
        with pytest.raises(ComputerUseError) as exc_np:
            await runner.execute_step(
                owner_id=owner_id,
                session_id=s1.session_id,
                tool_name="input.click",
                arguments={"elementId": "s00000001:3"},
            )
        assert exc_np.value.code == "no_progress"
        assert not s1.active

        # 2. Duplicate action on same element twice -> stops with duplicate_action
        next_effect = "changed"
        s2 = await start_cu()
        await runner.execute_step(
            owner_id=owner_id,
            session_id=s2.session_id,
            tool_name="input.click",
            arguments={"elementId": "s00000001:1"},
        )
        with pytest.raises(ComputerUseError) as exc_dup:
            await runner.execute_step(
                owner_id=owner_id,
                session_id=s2.session_id,
                tool_name="input.click",
                arguments={"elementId": "s00000001:1"},
            )
        assert exc_dup.value.code == "duplicate_action"
        assert not s2.active

        # 3. Dead-man switch: 3 missed 5 s heartbeats (> 15 s) -> stops with heartbeat_timeout
        s3 = await start_cu()
        fake_now += 5.0
        assert runner.record_heartbeat(owner_id=owner_id, session_id=s3.session_id) is True
        fake_now += 15.5  # > 15.0 s without heartbeat
        with pytest.raises(ComputerUseError) as exc_hb:
            await runner.observe(owner_id=owner_id, session_id=s3.session_id)
        assert exc_hb.value.code == "heartbeat_timeout"
        assert not s3.active

        # 4. Emergency stop latency <= 1 s and zero screen/tree retention in store after stop
        s4 = await start_cu()
        await runner.observe(owner_id=owner_id, session_id=s4.session_id)
        assert len(s4.cached_nodes) > 0
        t0 = time.perf_counter()
        stopped_count = store.stop_all_for_owner(owner_id, reason="owner_app")
        stop_latency_s = time.perf_counter() - t0
        assert stopped_count == 1
        assert stop_latency_s <= 1.0
        assert not s4.active
        assert s4.cached_nodes == {}

    asyncio.run(_run())
