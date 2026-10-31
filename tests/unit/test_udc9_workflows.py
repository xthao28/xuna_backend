"""UDC-9 unit and API integration tests (`S9.1` -> `S9.3`):

- `S9.1` (`browser.current_url` + `link_to_phone`):
  - 10/10 simulated cases across Edge & Chrome (Windows) and Safari & Chrome (macOS).
  - Single approval showing both hosts (`"{pc_alias} → {phone_alias}"`) and the domain only.
  - Zero-URL logging invariant: full URL is never written to `AuditLog`, `ApprovalRequest`,
    `ActionPendingEvent.summary`, or `ActionResultEvent.summary`.
  - Rejects `http://`, `file://`, `chrome://`, private IP (`192.168.1.1`), and `localhost`,
    stopping at Step 1 (`failedStep == 1`).
- `S9.2` (`continue_media` + `pause_everywhere`):
  - `continue_media`: 4-step handoff (`media.now_playing` on A -> `media.play_pause` on A ->
    `app.launch` on B -> `media.play_pause` on B), single approval listing both hosts,
    stops at first failure naming the step and host, and includes playback position sync note.
  - `pause_everywhere`: pauses all online hosts with `media` granted (skipping already-paused
    hosts) + board HID if connected, single approval listing all aliases, per-host results,
    and stops at first failure naming the host.
- `S9.3` (`PUT /v1/actions/workflow-slots/{slot}` + `POST /v1/actions/workflow-trigger`):
  - `R1` workflow assigned to slot 1 runs immediately (`status == "ok"`, deterministic
    gesture + board bubble `"Đang chạy: {workflow name}"`).
  - `R2` workflow assigned to slot 2 requires approval on the phone (`needs_approval` ->
    approve -> `"ok"`).
  - `R3` workflow cannot be assigned or triggered from board buttons (`UDC-T15`,
    HTTP 403 `workflow_risk_forbidden`).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi.testclient import TestClient
from tests.helpers import AUDIENCE, ISSUER, LocalIssuer, MemoryAuditLog

from xuna_backend.core.config import Environment, Settings
from xuna_backend.main import create_app
from xuna_backend.modules.accounts.domain.owner import Owner
from xuna_backend.modules.actions.application.catalog import (
    CAP_BOARD_HID_V1,
    CAP_HOST_ACTIONS_V1,
    CAP_WORKFLOW_TEMPLATES_V1,
    build_tool_catalog,
)
from xuna_backend.modules.actions.application.orchestrator import (
    ActionContext,
    Orchestrator,
)
from xuna_backend.modules.actions.domain.entities import (
    ActionPendingEvent,
    ActionStreamEvent,
    ApprovalDecision,
    BoardHidRequestEvent,
    LocalActionRequestEvent,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.repository import (
    InMemoryApprovalRepository,
)
from xuna_backend.modules.cognition.domain.types import Locale, Role, ToolCall
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

NOW = datetime(2026, 10, 31, 10, 0, 0, tzinfo=UTC)
FAKE_PUBKEY = b"\x04" + bytes(range(1, 65))


class _DummyDbSession:
    async def commit(self) -> None:
        pass

    async def close(self) -> None:
        pass


class _DummyDatabase:
    @asynccontextmanager
    async def sessions(self) -> AsyncIterator[_DummyDbSession]:
        yield _DummyDbSession()

    async def dispose(self) -> None:
        pass


class _FakeOwnerRepo:
    def __init__(self, owner: Owner) -> None:
        self._owner = owner

    async def get_or_create(self, claims: Any) -> Owner:
        return self._owner


class _FakeHostRepo:
    def __init__(self) -> None:
        self.hosts: dict[HostId, Host] = {}
        self.manifests: dict[HostId, Manifest] = {}
        self.grants: dict[HostId, list[Grant]] = {}

    async def get_host(self, host_id: HostId) -> Host | None:
        return self.hosts.get(host_id)

    async def save_host(self, host: Host) -> None:
        self.hosts[host.host_id] = host

    async def list_hosts(self, owner_id: uuid.UUID) -> list[Host]:
        return [h for h in self.hosts.values() if h.owner_id == owner_id]

    async def get_grants(self, host_id: HostId) -> list[Grant]:
        return list(self.grants.get(host_id, []))

    async def set_grants(self, host_id: HostId, grants: list[Grant]) -> None:
        self.grants[host_id] = list(grants)

    async def suspend_r2_plus_grants_for_owner(
        self, owner_id: uuid.UUID, now: datetime
    ) -> int:
        return 0

    async def get_manifest(self, host_id: HostId) -> Manifest | None:
        return self.manifests.get(host_id)

    async def save_manifest(self, manifest: Manifest) -> None:
        self.manifests[manifest.host_id] = manifest


class _FakeHub:
    def __init__(self) -> None:
        self.online_ids: set[str] = set()
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.handler: Any = None

    def is_online(self, host_id: str) -> bool:
        return host_id in self.online_ids

    async def send_request(
        self,
        host_id: str,
        method: str,
        params: dict[str, Any],
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        self.calls.append((host_id, method, params))
        if self.handler is not None:
            return await self.handler(host_id, method, params)
        return {
            "content": [{"type": "text", "text": "ok"}],
            "structuredContent": {},
            "isError": False,
            "_meta": {"xuna/status": "ok", "xuna/durationMs": 5},
        }

    async def stop_host(self, host_id: str, reason: str = "owner_app") -> bool:
        return host_id in self.online_ids


def _make_host(owner_id: uuid.UUID, alias: str, platform: HostPlatform) -> Host:
    return Host(
        host_id=HostId.generate(),
        owner_id=owner_id,
        alias=HostAlias(alias),
        platform=platform,
        hostname=f"{alias.lower()}-host",
        os_version="15.0",
        agent_version="0.1.0",
        public_key=FAKE_PUBKEY,
        status=HostStatus.ACTIVE,
        created_at=NOW,
        last_seen_at=NOW,
    )


def _desktop_tools() -> tuple[ToolDescriptor, ...]:
    empty_schema: dict[str, object] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    id_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "maxLength": 64, "pattern": r"^[a-z0-9._-]+$"}
        },
        "required": ["id"],
        "additionalProperties": False,
    }
    return (
        ToolDescriptor(
            name="browser.current_url",
            description="Read active https tab URL from Edge, Chrome, or Safari.",
            input_schema={
                "type": "object",
                "properties": {
                    "browser": {
                        "type": "string",
                        "enum": ["auto", "edge", "chrome", "safari"],
                    }
                },
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "maxLength": 2048},
                    "host": {"type": "string", "maxLength": 255},
                    "browser": {"type": "string", "enum": ["edge", "chrome", "safari"]},
                },
                "required": ["url", "host", "browser"],
                "additionalProperties": False,
            },
            family=ToolFamily.URL,
            risk=RiskTier.R2,
            observes=True,
            reversible=True,
        ),
        ToolDescriptor(
            name="media.now_playing",
            description="Read active media playback metadata.",
            input_schema=empty_schema,
            output_schema=empty_schema,
            family=ToolFamily.MEDIA,
            risk=RiskTier.R0,
        ),
        ToolDescriptor(
            name="media.play_pause",
            description="Toggle media playback.",
            input_schema=empty_schema,
            output_schema=empty_schema,
            family=ToolFamily.MEDIA,
            risk=RiskTier.R1,
        ),
        ToolDescriptor(
            name="app.launch",
            description="Launch allowlisted application by ID.",
            input_schema=id_schema,
            output_schema=id_schema,
            family=ToolFamily.APP,
            risk=RiskTier.R2,
            reversible=False,
        ),
        ToolDescriptor(
            name="workflow.run",
            description="Run allowlisted workflow by ID.",
            input_schema=id_schema,
            output_schema=id_schema,
            family=ToolFamily.WORKFLOW,
            risk=RiskTier.R1,
            reversible=False,
        ),
    )


def _phone_tools() -> tuple[ToolDescriptor, ...]:
    empty_schema: dict[str, object] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    return (
        ToolDescriptor(
            name="url.open",
            description="Open an https URL on the phone.",
            input_schema={
                "type": "object",
                "properties": {"url": {"type": "string", "maxLength": 2048}},
                "required": ["url"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {"host": {"type": "string", "maxLength": 255}},
                "required": ["host"],
                "additionalProperties": False,
            },
            family=ToolFamily.URL,
            risk=RiskTier.R2,
            reversible=False,
        ),
        ToolDescriptor(
            name="media.play_pause",
            description="Toggle phone media playback.",
            input_schema=empty_schema,
            output_schema=empty_schema,
            family=ToolFamily.MEDIA,
            risk=RiskTier.R1,
        ),
    )


def _all_grants(host_id: HostId) -> list[Grant]:
    return [
        Grant(
            host_id=host_id,
            family=fam,
            max_risk=RiskTier.R2,
            enabled=True,
            updated_at=NOW,
        )
        for fam in (
            ToolFamily.URL,
            ToolFamily.MEDIA,
            ToolFamily.APP,
            ToolFamily.WORKFLOW,
        )
    ]


def test_s9_1_link_to_phone_10_of_10_browsers_single_approval_and_zero_url_logging() -> None:
    """S9.1 acceptance:
    - 10/10 across Edge & Chrome (Windows) and Safari & Chrome (macOS).
    - Single approval per run showing both hosts (`"{pc_alias} → {phone_alias}"`) and domain only.
    - Full URL is never logged in `AuditLog`, `ApprovalRequest`, or stream events.
    - Non-https / local / private IP URLs are rejected at Step 1 (`failedStep == 1`)."""

    async def _run() -> None:
        owner_id = uuid.uuid4()
        win_pc = _make_host(owner_id, "PC", HostPlatform.WINDOWS)
        mac = _make_host(owner_id, "Mac", HostPlatform.MACOS)
        phone = _make_host(owner_id, "Điện thoại", HostPlatform.IOS)

        host_repo = _FakeHostRepo()
        for h, tools in (
            (win_pc, _desktop_tools()),
            (mac, _desktop_tools()),
            (phone, _phone_tools()),
        ):
            await host_repo.save_host(h)
            await host_repo.save_manifest(
                Manifest(host_id=h.host_id, tools=tools, updated_at=NOW)
            )
            await host_repo.set_grants(h.host_id, _all_grants(h.host_id))

        # Verify catalog includes workflow_template when CAP_WORKFLOW_TEMPLATES_V1 is enabled
        cat = build_tool_catalog(
            hosts=[win_pc, mac, phone],
            manifests_by_host=host_repo.manifests,
            grants_by_host=host_repo.grants,
            online_host_ids={str(win_pc.host_id), str(mac.host_id), str(phone.host_id)},
            client_caps=(CAP_HOST_ACTIONS_V1, CAP_WORKFLOW_TEMPLATES_V1),
            host_screen_consent=True,
            now=NOW,
        )
        assert any(t.name == "workflow_template" for t in cat)

        hub = _FakeHub()
        hub.online_ids = {str(win_pc.host_id), str(mac.host_id), str(phone.host_id)}
        approval_repo = InMemoryApprovalRepository()
        approval_broker = InMemoryApprovalBroker()
        local_broker = InMemoryLocalResultBroker()
        audit = MemoryAuditLog()

        orch = Orchestrator(
            host_repo=host_repo,
            hub=hub,
            approval_repo=approval_repo,
            approval_broker=approval_broker,
            local_result_broker=local_broker,
            last_target_store=InMemoryLastTargetStore(),
            audit=audit,
            clock=lambda: NOW,
        )

        ten_cases: list[tuple[str, str, str, str]] = [
            (
                "PC",
                "edge",
                "https://github.com/xthao28/xuna_pet?token=secret01",
                "github.com",
            ),
            (
                "PC",
                "edge",
                "https://docs.github.com/en/actions?token=secret02",
                "docs.github.com",
            ),
            (
                "PC",
                "edge",
                "https://learn.microsoft.com/en-us/windows?token=secret03",
                "learn.microsoft.com",
            ),
            (
                "PC",
                "chrome",
                "https://pub.dev/packages/flutter_test?token=secret04",
                "pub.dev",
            ),
            (
                "PC",
                "chrome",
                "https://jira.example.com/browse/XUNA-9?token=secret05",
                "jira.example.com",
            ),
            (
                "Mac",
                "safari",
                "https://developer.apple.com/documentation?token=secret06",
                "developer.apple.com",
            ),
            (
                "Mac",
                "safari",
                "https://news.ycombinator.com/item?id=99&token=secret07",
                "news.ycombinator.com",
            ),
            (
                "Mac",
                "safari",
                "https://sub.việt.vn/bai-viet?token=secret08",
                "sub.xn--vit-5kz.vn",
            ),
            (
                "Mac",
                "chrome",
                "https://arxiv.org/abs/2410.00001?token=secret09",
                "arxiv.org",
            ),
            (
                "Mac",
                "chrome",
                "https://openai.com/index/computer-use?token=secret10",
                "openai.com",
            ),
        ]

        active_tab_url = ""
        active_tab_browser = "edge"

        async def _fake_hub_handler(
            host_id: str, method: str, params: dict[str, Any]
        ) -> dict[str, Any]:
            assert method == "tools/call"
            assert params["name"] == "browser.current_url"
            return {
                "content": [{"type": "text", "text": "ok"}],
                "structuredContent": {
                    "url": active_tab_url,
                    "host": "ignored.example.com",
                    "browser": active_tab_browser,
                },
                "isError": False,
                "_meta": {"xuna/status": "ok", "xuna/durationMs": 6},
            }

        hub.handler = _fake_hub_handler

        for idx, (src_alias, browser_name, url, expected_domain) in enumerate(ten_cases):
            active_tab_url = url
            active_tab_browser = browser_name
            events: list[ActionStreamEvent] = []

            async def _on_event(
                ev: ActionStreamEvent,
                _ev_list: list[ActionStreamEvent] = events,
                _expected_src: str = src_alias,
                _expected_dom: str = expected_domain,
                _expected_url: str = url,
            ) -> None:
                _ev_list.append(ev)
                if isinstance(ev, ActionPendingEvent):
                    # Single approval shows both hosts and the domain only (never the full URL!)
                    assert ev.target == f"{_expected_src} → Điện thoại"
                    assert _expected_dom in ev.summary
                    assert "https://" not in ev.summary
                    assert "secret" not in ev.summary
                    await orch.decide_approval(
                        approval_id=uuid.UUID(ev.approval_id),
                        owner_id=owner_id,
                        decision=ApprovalDecision.APPROVE,
                    )
                elif isinstance(ev, LocalActionRequestEvent):
                    assert ev.tool == "url.open"
                    assert ev.arguments == {"url": _expected_url}
                    local_broker.report(
                        invocation_id=ev.invocation_id,
                        owner_id=owner_id,
                        status="ok",
                        output={"host": _expected_dom},
                    )

            ctx = ActionContext(
                owner_id=owner_id,
                pet_id="pet-1",
                locale=Locale.VI,
                client_caps=(CAP_HOST_ACTIONS_V1,),
                board_available=False,
                emit_event=_on_event,
                request_id=f"req-s91-{idx}",
                self_host_id=str(phone.host_id),
            )
            msg = await orch.handle_call(
                ToolCall(
                    id=f"call-s91-{idx}",
                    name="workflow_template",
                    arguments={
                        "template": "link_to_phone",
                        "source_target": src_alias,
                        "destination_target": "Điện thoại",
                        "browser": browser_name,
                    },
                ),
                ctx,
            )
            assert msg.role == Role.TOOL
            payload = json.loads(msg.content)
            assert payload["status"] == "ok"
            assert payload["output"]["domain"] == expected_domain
            assert payload["output"]["sourceHost"] == src_alias
            assert payload["output"]["destinationHost"] == "Điện thoại"
            assert payload["output"]["browser"] == browser_name
            assert "secret" not in msg.content

            pending_events = [e for e in events if isinstance(e, ActionPendingEvent)]
            assert len(pending_events) == 1

        # Rejects non-https / private IP / localhost URLs at Step 1 without asking approval
        for bad_url in (
            "http://github.com/insecure",
            "file:///C:/Users/Thao/secret.txt",
            "chrome://settings/passwords",
            "https://192.168.1.1/admin",
            "https://localhost:3000/dev",
        ):
            active_tab_url = bad_url
            bad_events: list[ActionStreamEvent] = []

            async def _on_bad_event(
                ev: ActionStreamEvent, _list: list[ActionStreamEvent] = bad_events
            ) -> None:
                _list.append(ev)

            ctx_bad = ActionContext(
                owner_id=owner_id,
                pet_id="pet-1",
                locale=Locale.VI,
                client_caps=(CAP_HOST_ACTIONS_V1,),
                board_available=False,
                emit_event=_on_bad_event,
                self_host_id=str(phone.host_id),
            )
            bad_msg = await orch.handle_call(
                ToolCall(
                    id="call-bad",
                    name="workflow_template",
                    arguments={
                        "template": "link_to_phone",
                        "source_target": "PC",
                        "destination_target": "Điện thoại",
                    },
                ),
                ctx_bad,
            )
            bad_payload = json.loads(bad_msg.content)
            assert bad_payload["status"] == "invalid_args"
            assert bad_payload["output"]["failedStep"] == 1
            assert bad_payload["output"]["failedHost"] == "PC"
            assert not any(isinstance(e, ActionPendingEvent) for e in bad_events)

        # Verify zero URLs or query tokens in AuditLog and ApprovalRepository
        for event in audit.events:
            serialized = json.dumps(event.metadata)
            assert "https://" not in serialized
            assert "secret" not in serialized
        for stored_req in approval_repo._items.values():
            assert "https://" not in stored_req.summary
            assert "secret" not in stored_req.summary

    asyncio.run(_run())


def test_s9_2_continue_media_and_pause_everywhere_plans_stop_on_first_failure() -> None:
    """S9.2 acceptance:
    - `continue_media`: executes 4 steps across Host A and Host B with one approval,
      includes position-sync note, and stops at first failure naming the host and step.
    - `pause_everywhere`: pauses all online hosts with `media` granted (skipping already-paused
      hosts) + board HID if connected, with one approval listing all aliases."""

    async def _run() -> None:
        owner_id = uuid.uuid4()
        pc = _make_host(owner_id, "PC", HostPlatform.WINDOWS)
        mac = _make_host(owner_id, "Mac", HostPlatform.MACOS)

        host_repo = _FakeHostRepo()
        for h in (pc, mac):
            await host_repo.save_host(h)
            await host_repo.save_manifest(
                Manifest(host_id=h.host_id, tools=_desktop_tools(), updated_at=NOW)
            )
            await host_repo.set_grants(h.host_id, _all_grants(h.host_id))

        hub = _FakeHub()
        hub.online_ids = {str(pc.host_id), str(mac.host_id)}
        approval_repo = InMemoryApprovalRepository()
        approval_broker = InMemoryApprovalBroker()
        local_broker = InMemoryLocalResultBroker()
        audit = MemoryAuditLog()

        orch = Orchestrator(
            host_repo=host_repo,
            hub=hub,
            approval_repo=approval_repo,
            approval_broker=approval_broker,
            local_result_broker=local_broker,
            last_target_store=InMemoryLastTargetStore(),
            audit=audit,
            clock=lambda: NOW,
        )

        # 1. Successful 4-step continue_media from PC to Mac
        async def _continue_ok_handler(
            host_id: str, method: str, params: dict[str, Any]
        ) -> dict[str, Any]:
            if params["name"] == "media.now_playing":
                return {
                    "content": [{"type": "text", "text": "ok"}],
                    "structuredContent": {
                        "state": "playing",
                        "title": "Lofi Beats",
                        "app": "spotify",
                    },
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 4},
                }
            return {
                "content": [{"type": "text", "text": "ok"}],
                "structuredContent": {},
                "isError": False,
                "_meta": {"xuna/status": "ok", "xuna/durationMs": 5},
            }

        hub.handler = _continue_ok_handler
        hub.calls.clear()
        cm_events: list[ActionStreamEvent] = []

        async def _on_cm_event(ev: ActionStreamEvent) -> None:
            cm_events.append(ev)
            if isinstance(ev, ActionPendingEvent):
                assert ev.target == "PC → Mac"
                await orch.decide_approval(
                    approval_id=uuid.UUID(ev.approval_id),
                    owner_id=owner_id,
                    decision=ApprovalDecision.APPROVE,
                )

        ctx = ActionContext(
            owner_id=owner_id,
            pet_id="pet-1",
            locale=Locale.VI,
            client_caps=(CAP_HOST_ACTIONS_V1, CAP_BOARD_HID_V1),
            board_available=True,
            emit_event=_on_cm_event,
        )
        cm_msg = await orch.handle_call(
            ToolCall(
                id="call-cm-1",
                name="workflow_template",
                arguments={
                    "template": "continue_media",
                    "source_target": "PC",
                    "destination_target": "Mac",
                },
            ),
            ctx,
        )
        cm_payload = json.loads(cm_msg.content)
        assert cm_payload["status"] == "ok"
        assert cm_payload["output"]["stepsCompleted"] == 4
        assert cm_payload["output"]["playerId"] == "spotify"
        assert "Spotify" in cm_payload["output"]["note"]
        assert len([e for e in cm_events if isinstance(e, ActionPendingEvent)]) == 1
        assert [c[2]["name"] for c in hub.calls] == [
            "media.now_playing",
            "media.play_pause",
            "app.launch",
            "media.play_pause",
        ]

        # 2. Failure at Step 3 (app.launch on Mac) stops immediately and names Mac & Step 3
        async def _continue_fail_step3(
            host_id: str, method: str, params: dict[str, Any]
        ) -> dict[str, Any]:
            if params["name"] == "media.now_playing":
                return {
                    "content": [{"type": "text", "text": "ok"}],
                    "structuredContent": {"state": "playing", "app": "spotify"},
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 4},
                }
            if params["name"] == "app.launch":
                return {
                    "content": [{"type": "text", "text": "denied"}],
                    "structuredContent": {},
                    "isError": True,
                    "_meta": {"xuna/status": "denied", "xuna/durationMs": 3},
                }
            return {
                "content": [{"type": "text", "text": "ok"}],
                "structuredContent": {},
                "isError": False,
                "_meta": {"xuna/status": "ok", "xuna/durationMs": 4},
            }

        hub.handler = _continue_fail_step3
        hub.calls.clear()
        fail_msg = await orch.handle_call(
            ToolCall(
                id="call-cm-fail",
                name="workflow_template",
                arguments={
                    "template": "continue_media",
                    "source_target": "PC",
                    "destination_target": "Mac",
                },
            ),
            ctx,
        )
        fail_payload = json.loads(fail_msg.content)
        assert fail_payload["status"] == "denied"
        assert fail_payload["output"]["failedStep"] == 3
        assert fail_payload["output"]["failedTool"] == "app.launch"
        assert fail_payload["output"]["failedHost"] == "Mac"
        # Step 4 (media.play_pause on Mac) was never called!
        assert [c[2]["name"] for c in hub.calls] == [
            "media.now_playing",
            "media.play_pause",
            "app.launch",
        ]

        # 3. pause_everywhere across Mac (already paused), PC (playing), and board HID
        async def _pause_handler(
            host_id: str, method: str, params: dict[str, Any]
        ) -> dict[str, Any]:
            if params["name"] == "media.now_playing":
                state_val = "paused" if host_id == str(mac.host_id) else "playing"
                return {
                    "content": [{"type": "text", "text": "ok"}],
                    "structuredContent": {"state": state_val},
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 3},
                }
            return {
                "content": [{"type": "text", "text": "ok"}],
                "structuredContent": {"state": "paused"},
                "isError": False,
                "_meta": {"xuna/status": "ok", "xuna/durationMs": 4},
            }

        hub.handler = _pause_handler
        hub.calls.clear()
        pe_events: list[ActionStreamEvent] = []

        async def _on_pe_event(ev: ActionStreamEvent) -> None:
            pe_events.append(ev)
            if isinstance(ev, ActionPendingEvent):
                assert "Mac" in ev.summary
                assert "PC" in ev.summary
                assert "Máy đang ghép" in ev.summary
                await orch.decide_approval(
                    approval_id=uuid.UUID(ev.approval_id),
                    owner_id=owner_id,
                    decision=ApprovalDecision.APPROVE,
                )
            elif isinstance(ev, BoardHidRequestEvent):
                assert ev.key == "stop"
                local_broker.report(
                    invocation_id=ev.invocation_id,
                    owner_id=owner_id,
                    status="ok",
                )

        ctx_pe = ActionContext(
            owner_id=owner_id,
            pet_id="pet-1",
            locale=Locale.VI,
            client_caps=(CAP_HOST_ACTIONS_V1, CAP_BOARD_HID_V1),
            board_available=True,
            emit_event=_on_pe_event,
        )
        pe_msg = await orch.handle_call(
            ToolCall(
                id="call-pe-1",
                name="workflow_template",
                arguments={"template": "pause_everywhere"},
            ),
            ctx_pe,
        )
        pe_payload = json.loads(pe_msg.content)
        assert pe_payload["status"] == "ok"
        results = pe_payload["output"]["results"]
        assert results == [
            {"host": "Mac", "status": "ok", "action": "already_paused"},
            {"host": "PC", "status": "ok", "action": "paused"},
            {"host": "Máy đang ghép", "status": "ok", "action": "paused"},
        ]

    asyncio.run(_run())


def test_s9_3_board_workflow_trigger_r1_immediate_r2_approval_and_r3_forbidden_api() -> None:
    """S9.3 acceptance (`PUT /v1/actions/workflow-slots/{slot}` &
    `POST /v1/actions/workflow-trigger`):
    - R1 workflow on slot 1 runs immediately (`status == "ok"`, bubble `"Đang chạy: ..."`).
    - R2 workflow on slot 2 returns `needs_approval`, and runs after phone approval.
    - R3 workflow cannot be assigned or triggered from board buttons (`UDC-T15`, 403)."""
    issuer = LocalIssuer()
    owner_id = uuid.uuid4()
    owner = Owner(
        owner_id=owner_id,
        issuer=ISSUER,
        subject="owner-udc9",
    )
    pc = _make_host(owner_id, "PC", HostPlatform.WINDOWS)

    host_repo = _FakeHostRepo()
    asyncio.run(host_repo.save_host(pc))
    asyncio.run(
        host_repo.save_manifest(
            Manifest(host_id=pc.host_id, tools=_desktop_tools(), updated_at=NOW)
        )
    )
    asyncio.run(host_repo.set_grants(pc.host_id, _all_grants(pc.host_id)))

    hub = _FakeHub()
    hub.online_ids = {str(pc.host_id)}
    approval_repo = InMemoryApprovalRepository()
    approval_broker = InMemoryApprovalBroker()
    local_broker = InMemoryLocalResultBroker()
    audit = MemoryAuditLog()

    app = create_app(
        Settings(
            environment=Environment.TEST,
            database_url="postgresql+asyncpg://unused/unused",
            oidc_issuer=ISSUER,
            oidc_audience=AUDIENCE,
        )
    )
    app.state.database = _DummyDatabase()
    app.state.token_verifier = issuer.verifier()
    app.state.owner_repository = lambda _s: _FakeOwnerRepo(owner)
    app.state.host_repository = lambda _s: host_repo
    app.state.host_session_hub = hub
    app.state.approval_repository = lambda _s: approval_repo
    app.state.approval_broker = approval_broker
    app.state.local_result_broker = local_broker
    app.state.last_target_store = InMemoryLastTargetStore()
    app.state.audit_log = lambda _s: audit

    headers = {"Authorization": f"Bearer {issuer.token(subject='owner-udc9')}"}
    client = TestClient(app)

    if True:
        # 1. Assigning an R3 workflow to slot 1 is rejected with 403 (UDC-T15)
        r3_assign = client.put(
            "/v1/actions/workflow-slots/1",
            headers=headers,
            json={
                "hostAlias": "PC",
                "workflowId": "deploy_prod",
                "workflowName": "Deploy Production",
                "risk": "R3",
            },
        )
        assert r3_assign.status_code == 403
        assert r3_assign.json()["code"] == "workflow_risk_forbidden"

        # 2. Direct trigger of an R3 workflow is also rejected with 403 workflow_risk_forbidden
        r3_trigger = client.post(
            "/v1/actions/workflow-trigger",
            headers=headers,
            json={
                "slot": 1,
                "hostAlias": "PC",
                "workflowId": "deploy_prod",
                "workflowName": "Deploy Production",
                "risk": "R3",
            },
        )
        assert r3_trigger.status_code == 403
        assert r3_trigger.json()["code"] == "workflow_risk_forbidden"

        # 3. Assign R1 workflow to slot 1 and trigger -> runs immediately!
        r1_assign = client.put(
            "/v1/actions/workflow-slots/1",
            headers=headers,
            json={
                "hostAlias": "PC",
                "workflowId": "focus",
                "workflowName": "Chế độ tập trung",
                "risk": "R1",
            },
        )
        assert r1_assign.status_code == 200

        r1_trig = client.post(
            "/v1/actions/workflow-trigger",
            headers=headers,
            json={"slot": 1},
        )
        assert r1_trig.status_code == 200
        r1_body = r1_trig.json()
        assert r1_body["status"] == "ok"
        assert r1_body["slot"] == 1
        assert r1_body["host"] == "PC"
        assert r1_body["workflowId"] == "focus"
        assert r1_body["bubble"] == "Đang chạy: Chế độ tập trung"
        assert r1_body["gesture"] == "bounce_play"

        # 4. Assign R2 workflow to slot 2 and trigger -> asks approval on phone first
        r2_assign = client.put(
            "/v1/actions/workflow-slots/2",
            headers=headers,
            json={
                "hostAlias": "PC",
                "workflowId": "clean_workspace",
                "workflowName": "Dọn không gian làm việc",
                "risk": "R2",
            },
        )
        assert r2_assign.status_code == 200

        r2_trig1 = client.post(
            "/v1/actions/workflow-trigger",
            headers=headers,
            json={"slot": 2},
        )
        assert r2_trig1.status_code == 200
        r2_pending = r2_trig1.json()
        assert r2_pending["status"] == "needs_approval"
        assert r2_pending["approvalId"] is not None
        assert r2_pending["bubble"] == "Đang chạy: Dọn không gian làm việc"

        approval_id = r2_pending["approvalId"]
        approve_resp = client.post(
            f"/v1/actions/approvals/{approval_id}",
            headers=headers,
            json={"decision": "approve", "scope": "once"},
        )
        assert approve_resp.status_code == 200

        r2_trig2 = client.post(
            "/v1/actions/workflow-trigger",
            headers=headers,
            json={"slot": 2, "approvalId": approval_id},
        )
        assert r2_trig2.status_code == 200
        r2_ok = r2_trig2.json()
        assert r2_ok["status"] == "ok"
        assert r2_ok["gesture"] == "bounce_play"
