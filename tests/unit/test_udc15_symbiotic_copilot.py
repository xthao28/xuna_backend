"""UDC-15 unit and integration tests (`S15.1` -> `S15.5`):

- `S15.1`-`S15.4`:
  - Validates tool name mapping (`validate_tool_name`) and deterministic summaries
    (`ActionSummary`) for `clipboard.write`, `meeting.status`, `meeting.control`,
    `dev.ports`, and `system.vitals`.
  - Validates conversational routine extraction (`extract_custom_workflow_candidate`)
    for `"tắt mic"` / `"bật mic"` (`meeting.control`) and `"kiểm tra pin"` (`system.vitals`).
- `S15.2`:
  - `CompositeWorkflowRunner` executes `meeting_ready` (`R1`): pauses local PC music if
    playing (`media.play_pause`), mutes Smart TV (`tv.control(mute_toggle)`), mutes
    meeting microphone (`meeting.control(mute_mic)`), and checks meeting status
    (`meeting.status`) without requiring an `R2` approval.
- `S15.3`:
  - `CompositeWorkflowRunner` executes `preview_on_phone` (`R2`): scans active localhost
    dev ports via `dev.ports`, requests **one** `R2` confirmation showing
    `"{pc} → {phone} · Mở bản xem thử cổng :{port}"`, and returns the LAN preview URL
    without logging it (`UDC-T38`).
- `S15.5`:
  - `CompositeWorkflowRunner` executes `daily_standup` (`R1`): reads Git branch & dirty
    file count (`workspace.git_summary`), active localhost dev ports (`dev.ports`), and
    host battery/RAM/disk vitals (`system.vitals`) without requiring an `R2` approval.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from tests.helpers import MemoryAuditLog

from xuna_backend.modules.actions.application.workflows import (
    CompositeWorkflowRunner,
    extract_custom_workflow_candidate,
)
from xuna_backend.modules.actions.domain.entities import (
    ActionPendingEvent,
    ActionStreamEvent,
    ActionSummary,
    ApprovalDecision,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.repository import (
    InMemoryApprovalRepository,
)
from xuna_backend.modules.cognition.domain.types import Locale, ToolCall
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
    validate_tool_name,
)

NOW = datetime(2026, 10, 31, 16, 0, 0, tzinfo=UTC)
FAKE_PUBKEY = b"\x04" + bytes(range(1, 65))


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
            "_meta": {"xuna/status": "ok", "xuna/durationMs": 4},
        }

    async def stop_host(self, host_id: str, reason: str = "owner_app") -> bool:
        return host_id in self.online_ids


def _build_symbiotic_manifest(host_id: HostId) -> Manifest:
    empty_schema: dict[str, object] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    any_obj_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "maxLength": 1600},
            "action": {"type": "string", "maxLength": 32},
            "device": {"type": "string", "maxLength": 64},
            "project": {"type": "string", "maxLength": 64},
            "ports": {
                "type": "array",
                "items": {"type": "integer"},
            },
        },
        "additionalProperties": False,
    }
    return Manifest(
        host_id=host_id,
        tools=(
            ToolDescriptor(
                name="clipboard.write",
                description="Copy text to OS clipboard.",
                input_schema=any_obj_schema,
                output_schema=empty_schema,
                family=ToolFamily.APP,
                risk=RiskTier.R2,
                observes=False,
                reversible=False,
            ),
            ToolDescriptor(
                name="meeting.status",
                description="Check meeting and microphone status.",
                input_schema=empty_schema,
                output_schema=empty_schema,
                family=ToolFamily.MEDIA,
                risk=RiskTier.R1,
                observes=True,
                reversible=True,
            ),
            ToolDescriptor(
                name="meeting.control",
                description="Control meeting microphone.",
                input_schema=any_obj_schema,
                output_schema=empty_schema,
                family=ToolFamily.MEDIA,
                risk=RiskTier.R1,
                observes=False,
                reversible=True,
            ),
            ToolDescriptor(
                name="dev.ports",
                description="Scan active localhost dev ports.",
                input_schema=any_obj_schema,
                output_schema=empty_schema,
                family=ToolFamily.APP,
                risk=RiskTier.R1,
                observes=True,
                reversible=True,
            ),
            ToolDescriptor(
                name="system.vitals",
                description="Check host battery, RAM, and disk vitals.",
                input_schema=empty_schema,
                output_schema=empty_schema,
                family=ToolFamily.HOST,
                risk=RiskTier.R1,
                observes=True,
                reversible=True,
            ),
            ToolDescriptor(
                name="media.now_playing",
                description="Read now playing.",
                input_schema=empty_schema,
                output_schema=empty_schema,
                family=ToolFamily.MEDIA,
                risk=RiskTier.R0,
                observes=True,
                reversible=True,
            ),
            ToolDescriptor(
                name="media.play_pause",
                description="Play or pause.",
                input_schema=empty_schema,
                output_schema=empty_schema,
                family=ToolFamily.MEDIA,
                risk=RiskTier.R1,
                observes=False,
                reversible=True,
            ),
            ToolDescriptor(
                name="tv.control",
                description="Control Smart TV.",
                input_schema=any_obj_schema,
                output_schema=empty_schema,
                family=ToolFamily.MEDIA,
                risk=RiskTier.R1,
                observes=False,
                reversible=True,
            ),
            ToolDescriptor(
                name="workspace.git_summary",
                description="Read Git summary.",
                input_schema=any_obj_schema,
                output_schema=empty_schema,
                family=ToolFamily.APP,
                risk=RiskTier.R1,
                observes=True,
                reversible=True,
            ),
        ),
        updated_at=NOW,
    )


def test_udc15_tool_validation_summaries_and_routine_clauses() -> None:
    assert validate_tool_name("clipboard.write") == ToolFamily.APP
    assert validate_tool_name("meeting.status") == ToolFamily.MEDIA
    assert validate_tool_name("meeting.control") == ToolFamily.MEDIA
    assert validate_tool_name("dev.ports") == ToolFamily.APP
    assert validate_tool_name("system.vitals") == ToolFamily.HOST

    vi_clip = ActionSummary.format_pending(
        "clipboard.write",
        {"text": "npm run dev -- --host"},
        "MacBook",
        Locale.VI,
    )
    assert "Chép 21 ký tự vào clipboard của MacBook" in vi_clip
    assert "npm run dev" not in vi_clip

    en_vitals = ActionSummary.format_pending(
        "system.vitals",
        {},
        "MacBook",
        Locale.EN,
    )
    assert "Check battery, memory, and disk vitals on MacBook" in en_vitals

    owner_id = uuid.uuid4()
    candidate = extract_custom_workflow_candidate(
        owner_id,
        "Khi mình nói 'vào họp gấp' thì tắt nhạc, tắt mic rồi kiểm tra pin",
        default_host_alias="MacBook",
    )
    assert candidate is not None
    assert [s.tool for s in candidate.steps] == [
        "media.play_pause",
        "meeting.control",
        "system.vitals",
    ]
    assert candidate.effective_risk == RiskTier.R1


def test_udc15_meeting_ready_preview_on_phone_and_daily_standup() -> None:
    async def _scenario() -> None:
        owner_id = uuid.uuid4()
        pc_id = HostId.generate()
        phone_id = HostId.generate()
        repo = _FakeHostRepo()
        hub = _FakeHub()
        approval_repo = InMemoryApprovalRepository()
        approval_broker = InMemoryApprovalBroker()
        local_broker = InMemoryLocalResultBroker()
        audit = MemoryAuditLog()

        pc_host = Host(
            host_id=pc_id,
            owner_id=owner_id,
            alias=HostAlias("MacBook Pro"),
            platform=HostPlatform.MACOS,
            hostname="macbook-pro",
            os_version="macOS 15.2",
            agent_version="0.1.0",
            public_key=FAKE_PUBKEY,
            status=HostStatus.ACTIVE,
            created_at=NOW,
        )
        phone_host = Host(
            host_id=phone_id,
            owner_id=owner_id,
            alias=HostAlias("iPhone 16"),
            platform=HostPlatform.IOS,
            hostname="iphone-16",
            os_version="iOS 18.2",
            agent_version="0.1.0",
            public_key=FAKE_PUBKEY,
            status=HostStatus.ACTIVE,
            created_at=NOW,
        )
        await repo.save_host(pc_host)
        await repo.save_host(phone_host)
        await repo.save_manifest(_build_symbiotic_manifest(pc_id))
        await repo.save_manifest(_build_symbiotic_manifest(phone_id))
        await repo.set_grants(
            pc_id,
            [
                Grant(
                    host_id=pc_id,
                    family=fam,
                    enabled=True,
                    max_risk=RiskTier.R2,
                    updated_at=NOW,
                )
                for fam in (
                    ToolFamily.MEDIA,
                    ToolFamily.APP,
                    ToolFamily.URL,
                    ToolFamily.WORKFLOW,
                    ToolFamily.HOST,
                )
            ],
        )
        await repo.set_grants(
            phone_id,
            [
                Grant(
                    host_id=phone_id,
                    family=ToolFamily.URL,
                    enabled=True,
                    max_risk=RiskTier.R2,
                    updated_at=NOW,
                )
            ],
        )
        hub.online_ids.add(str(pc_id))
        hub.online_ids.add(str(phone_id))

        async def _fake_handler(
            host_id: str, method: str, params: dict[str, Any]
        ) -> dict[str, Any]:
            name = params.get("name")
            if name == "media.now_playing":
                return {
                    "content": [{"type": "text", "text": "playing"}],
                    "structuredContent": {"state": "playing", "title": "Lo-Fi"},
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 4},
                }
            if name == "meeting.control":
                return {
                    "content": [{"type": "text", "text": "muted"}],
                    "structuredContent": {
                        "action": "mute_mic",
                        "micMuted": True,
                        "app": "zoom",
                    },
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 5},
                }
            if name == "meeting.status":
                return {
                    "content": [{"type": "text", "text": "in zoom"}],
                    "structuredContent": {
                        "inMeeting": True,
                        "app": "zoom",
                        "micMuted": True,
                    },
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 3},
                }
            if name == "dev.ports":
                return {
                    "content": [{"type": "text", "text": "ports"}],
                    "structuredContent": {
                        "lanIp": "192.168.1.42",
                        "activePorts": [
                            {
                                "port": 5173,
                                "process": "vite",
                                "url": "http://192.168.1.42:5173",
                            }
                        ],
                    },
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 8},
                }
            if name == "workspace.git_summary":
                return {
                    "content": [{"type": "text", "text": "git"}],
                    "structuredContent": {
                        "workspace": "xuna_pet",
                        "branch": "feat/udc-15-symbiotic",
                        "clean": False,
                        "modifiedCount": 3,
                        "untrackedCount": 1,
                    },
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 6},
                }
            if name == "system.vitals":
                return {
                    "content": [{"type": "text", "text": "vitals"}],
                    "structuredContent": {
                        "batteryPercent": 82,
                        "isCharging": True,
                        "memoryUsedPercent": 58,
                        "diskFreeGb": 110,
                        "thermalState": "nominal",
                        "healthStatus": "healthy",
                    },
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 4},
                }
            return {
                "content": [{"type": "text", "text": "ok"}],
                "structuredContent": {"state": "ok"},
                "isError": False,
                "_meta": {"xuna/status": "ok", "xuna/durationMs": 4},
            }

        hub.handler = _fake_handler
        runner = CompositeWorkflowRunner(
            host_repo=repo,
            hub=hub,
            approval_repo=approval_repo,
            approval_broker=approval_broker,
            local_result_broker=local_broker,
            audit=audit,
            clock=lambda: NOW,
        )

        # 1. S15.2: meeting_ready (R1, no approval required)
        meeting_events: list[ActionStreamEvent] = []

        async def _emit_meeting(ev: ActionStreamEvent) -> None:
            meeting_events.append(ev)

        meeting_msg = await runner.handle_template_call(
            ToolCall(
                id="call-meeting-1",
                name="workflow_template",
                arguments={"template": "meeting_ready"},
            ),
            owner_id=owner_id,
            locale=Locale.VI,
            client_caps=set(),
            board_available=False,
            emit_event=_emit_meeting,
            request_id="req-udc15-1",
        )
        meeting_payload = json.loads(meeting_msg.content)
        assert meeting_payload["status"] == "ok"
        assert meeting_payload["output"]["inMeeting"] is True
        assert meeting_payload["output"]["app"] == "zoom"
        assert meeting_payload["output"]["micMuted"] is True
        assert meeting_payload["output"]["mediaPaused"] is True
        assert meeting_payload["output"]["tvMuted"] is True
        assert not any(isinstance(e, ActionPendingEvent) for e in meeting_events)

        # 2. S15.3: preview_on_phone (R2, single approval, zero URL logging)
        preview_events: list[ActionStreamEvent] = []

        async def _emit_preview(ev: ActionStreamEvent) -> None:
            preview_events.append(ev)
            if isinstance(ev, ActionPendingEvent):
                await approval_broker.decide(
                    uuid.UUID(ev.approval_id),
                    owner_id,
                    ApprovalDecision.APPROVE,
                    NOW,
                )

        preview_msg = await runner.handle_template_call(
            ToolCall(
                id="call-preview-1",
                name="workflow_template",
                arguments={"template": "preview_on_phone"},
            ),
            owner_id=owner_id,
            locale=Locale.VI,
            client_caps=set(),
            board_available=False,
            emit_event=_emit_preview,
            request_id="req-udc15-2",
        )
        preview_payload = json.loads(preview_msg.content)
        assert preview_payload["status"] == "ok"
        assert preview_payload["output"]["port"] == 5173
        assert preview_payload["output"]["process"] == "vite"
        assert preview_payload["output"]["previewUrl"] == "http://192.168.1.42:5173"
        pending_cards = [e for e in preview_events if isinstance(e, ActionPendingEvent)]
        assert len(pending_cards) == 1
        assert (
            "MacBook Pro → iPhone 16 · Mở bản xem thử cổng :5173 (vite)"
            in pending_cards[0].summary
        )
        for audit_ev in audit.events:
            assert "192.168.1.42" not in json.dumps(audit_ev.metadata)

        # 3. S15.5: daily_standup (R1, no approval required)
        standup_events: list[ActionStreamEvent] = []

        async def _emit_standup(ev: ActionStreamEvent) -> None:
            standup_events.append(ev)

        standup_msg = await runner.handle_template_call(
            ToolCall(
                id="call-standup-1",
                name="workflow_template",
                arguments={"template": "daily_standup", "project": "xuna_pet"},
            ),
            owner_id=owner_id,
            locale=Locale.VI,
            client_caps=set(),
            board_available=False,
            emit_event=_emit_standup,
            request_id="req-udc15-3",
        )
        standup_payload = json.loads(standup_msg.content)
        assert standup_payload["status"] == "ok"
        assert standup_payload["output"]["workspace"] == "xuna_pet"
        assert standup_payload["output"]["branch"] == "feat/udc-15-symbiotic"
        assert standup_payload["output"]["uncommittedFiles"] == 4
        assert standup_payload["output"]["activePorts"] == [5173]
        assert standup_payload["output"]["batteryPercent"] == 82
        assert standup_payload["output"]["memoryUsedPercent"] == 58
        assert not any(isinstance(e, ActionPendingEvent) for e in standup_events)

    asyncio.run(_scenario())
