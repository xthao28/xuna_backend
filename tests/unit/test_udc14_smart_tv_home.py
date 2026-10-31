"""UDC-14 unit and integration tests (`S14.1` -> `S14.5`):

- `S14.1` & `S14.2`:
  - Validates tool name mapping (`validate_tool_name`) and deterministic summaries
    (`ActionSummary`) for `tv.list_devices`, `tv.control`, `tv.launch_app`,
    `tv.cast_media`, `smarthome.list_entities`, and `smarthome.control`.
- `S14.3`:
  - `CompositeWorkflowRunner` executes `cast_to_tv` (`R2`): reads active browser URL via
    `browser.current_url`, requests a single `R2` approval showing `{host} -> {tv} · {domain}`
    (never logging or showing the raw URL), pauses local PC playback (`media.play_pause`),
    and casts to the Smart TV (`tv.cast_media`).
- `S14.4`:
  - Enforces `UDC-T36` (`is_hard_denied_smarthome_entity`): physical locks (`lock.*`),
    alarms (`alarm_control_panel.*`), cameras (`camera.*`), garage doors (`garage_door.*`),
    and sirens (`siren.*`) are permanently hard-denied (`smarthome_domain_forbidden`).
- `S14.5`:
  - `CompositeWorkflowRunner` executes `cinema_mode` (`R2`: wakes Smart TV via `tv.control`,
    launches streaming app via `tv.launch_app`, dims room lights via `smarthome.control`,
    and pauses PC music via `media.play_pause` under one `R2` approval) and `goodnight_mode`
    (`R1`: checks Git status, pauses PC media, powers off TV, turns off lights, and locks
    PC screen).
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
    is_hard_denied_smarthome_entity,
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

NOW = datetime(2026, 10, 31, 15, 0, 0, tzinfo=UTC)
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


def _build_smart_tv_home_manifest(host_id: HostId) -> Manifest:
    empty_schema: dict[str, object] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    any_obj_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "device": {"type": "string", "maxLength": 64},
            "action": {"type": "string", "maxLength": 32},
            "level": {"type": "integer", "minimum": 0, "maximum": 100},
            "appId": {"type": "string", "maxLength": 32},
            "url": {"type": "string", "maxLength": 2048},
            "browser": {"type": "string", "maxLength": 24},
            "entityId": {"type": "string", "maxLength": 80},
            "value": {"type": "integer", "minimum": 0, "maximum": 100},
            "project": {"type": "string", "maxLength": 64},
        },
        "additionalProperties": False,
    }
    return Manifest(
        host_id=host_id,
        tools=(
            ToolDescriptor(
                name="tv.list_devices",
                description="Discover Smart TVs on LAN.",
                input_schema=empty_schema,
                output_schema=empty_schema,
                family=ToolFamily.MEDIA,
                risk=RiskTier.R1,
                observes=True,
            ),
            ToolDescriptor(
                name="tv.control",
                description="Control Smart TV power, volume, and playback.",
                input_schema=any_obj_schema,
                output_schema=empty_schema,
                family=ToolFamily.MEDIA,
                risk=RiskTier.R1,
                reversible=True,
            ),
            ToolDescriptor(
                name="tv.launch_app",
                description="Launch allowlisted streaming app on Smart TV.",
                input_schema=any_obj_schema,
                output_schema=empty_schema,
                family=ToolFamily.APP,
                risk=RiskTier.R2,
                reversible=False,
            ),
            ToolDescriptor(
                name="tv.cast_media",
                description="Cast https URL to Smart TV.",
                input_schema=any_obj_schema,
                output_schema=empty_schema,
                family=ToolFamily.URL,
                risk=RiskTier.R2,
                reversible=False,
            ),
            ToolDescriptor(
                name="smarthome.list_entities",
                description="List allowlisted smart-home entities.",
                input_schema=empty_schema,
                output_schema=empty_schema,
                family=ToolFamily.APP,
                risk=RiskTier.R1,
                observes=True,
            ),
            ToolDescriptor(
                name="smarthome.control",
                description="Control allowlisted smart-home appliance.",
                input_schema=any_obj_schema,
                output_schema=empty_schema,
                family=ToolFamily.APP,
                risk=RiskTier.R2,
                reversible=True,
            ),
            ToolDescriptor(
                name="browser.current_url",
                description="Read active https URL.",
                input_schema=any_obj_schema,
                output_schema=empty_schema,
                family=ToolFamily.URL,
                risk=RiskTier.R2,
                reversible=False,
            ),
            ToolDescriptor(
                name="media.now_playing",
                description="Read media state.",
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
                name="workspace.git_summary",
                description="Read workspace Git summary.",
                input_schema=any_obj_schema,
                output_schema=empty_schema,
                family=ToolFamily.APP,
                risk=RiskTier.R1,
            ),
            ToolDescriptor(
                name="host.lock_screen",
                description="Lock screen immediately.",
                input_schema=empty_schema,
                output_schema=empty_schema,
                family=ToolFamily.HOST,
                risk=RiskTier.R1,
            ),
        ),
        updated_at=NOW,
    )


def test_udc14_tool_names_summaries_and_hard_denylist() -> None:
    assert (
        validate_tool_name("tv.list_devices", expected_family=ToolFamily.MEDIA)
        == ToolFamily.MEDIA
    )
    assert (
        validate_tool_name("tv.control", expected_family=ToolFamily.MEDIA)
        == ToolFamily.MEDIA
    )
    assert (
        validate_tool_name("tv.launch_app", expected_family=ToolFamily.APP)
        == ToolFamily.APP
    )
    assert (
        validate_tool_name("tv.cast_media", expected_family=ToolFamily.URL)
        == ToolFamily.URL
    )
    assert (
        validate_tool_name("smarthome.list_entities", expected_family=ToolFamily.APP)
        == ToolFamily.APP
    )
    assert (
        validate_tool_name("smarthome.control", expected_family=ToolFamily.APP)
        == ToolFamily.APP
    )

    assert "Smart TV" in ActionSummary.format_pending(
        "tv.list_devices", {}, "MacBook", Locale.VI
    )
    assert "TV phòng khách" in ActionSummary.format_pending(
        "tv.control",
        {"device": "TV phòng khách", "action": "power_wake"},
        "MacBook",
        Locale.VI,
    )
    assert "netflix" in ActionSummary.format_pending(
        "tv.launch_app",
        {"device": "TV phòng khách", "appId": "netflix"},
        "MacBook",
        Locale.VI,
    )
    cast_summary = ActionSummary.format_pending(
        "tv.cast_media",
        {
            "device": "TV phòng khách",
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        },
        "MacBook",
        Locale.VI,
    )
    assert "www.youtube.com" in cast_summary
    assert "watch?v=" not in cast_summary

    assert "light.living_room" in ActionSummary.format_pending(
        "smarthome.control",
        {"entityId": "light.living_room", "action": "turn_off"},
        "MacBook",
        Locale.VI,
    )

    # UDC-T36 hard-denylist
    assert is_hard_denied_smarthome_entity("lock.front_door") is True
    assert is_hard_denied_smarthome_entity("alarm_control_panel.home") is True
    assert is_hard_denied_smarthome_entity("camera.living_room") is True
    assert is_hard_denied_smarthome_entity("garage_door.main") is True
    assert is_hard_denied_smarthome_entity("siren.hallway") is True
    assert is_hard_denied_smarthome_entity("light.living_room") is False
    assert is_hard_denied_smarthome_entity("climate.bedroom_ac") is False

    # Conversational routine extraction for TV & lights
    owner_id = uuid.uuid4()
    wf = extract_custom_workflow_candidate(
        owner_id,
        "Khi mình nói 'xem phim tối' thì bật tv, tắt đèn và dừng nhạc",
        default_host_alias="MacBook",
    )
    assert wf is not None
    assert [s.tool for s in wf.steps] == [
        "tv.control",
        "smarthome.control",
        "media.play_pause",
    ]


def test_udc14_cast_to_tv_cinema_mode_and_goodnight_mode() -> None:
    async def _scenario() -> None:
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        host = Host(
            host_id=host_id,
            owner_id=owner_id,
            alias=HostAlias("MacBook"),
            platform=HostPlatform.MACOS,
            hostname="macbook-pro",
            os_version="macOS 15.1",
            agent_version="0.1.0",
            public_key=FAKE_PUBKEY,
            status=HostStatus.ACTIVE,
            created_at=NOW,
        )

        repo = _FakeHostRepo()
        await repo.save_host(host)
        await repo.save_manifest(_build_smart_tv_home_manifest(host_id))
        await repo.set_grants(
            host_id,
            [
                Grant(
                    host_id=host_id,
                    family=ToolFamily.MEDIA,
                    max_risk=RiskTier.R1,
                    enabled=True,
                    updated_at=NOW,
                ),
                Grant(
                    host_id=host_id,
                    family=ToolFamily.APP,
                    max_risk=RiskTier.R2,
                    enabled=True,
                    updated_at=NOW,
                ),
                Grant(
                    host_id=host_id,
                    family=ToolFamily.URL,
                    max_risk=RiskTier.R2,
                    enabled=True,
                    updated_at=NOW,
                ),
                Grant(
                    host_id=host_id,
                    family=ToolFamily.HOST,
                    max_risk=RiskTier.R1,
                    enabled=True,
                    updated_at=NOW,
                ),
            ],
        )

        hub = _FakeHub()
        hub.online_ids.add(str(host_id))

        async def _fake_handler(
            _hid: str, _method: str, params: dict[str, Any]
        ) -> dict[str, Any]:
            tool_name = params.get("name")
            if tool_name == "browser.current_url":
                return {
                    "content": [{"type": "text", "text": "ok"}],
                    "structuredContent": {
                        "browser": "safari",
                        "url": "https://www.youtube.com/watch?v=secret_token_999",
                    },
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 5},
                }
            if tool_name == "media.now_playing":
                return {
                    "content": [{"type": "text", "text": "playing"}],
                    "structuredContent": {"state": "playing", "player": "spotify"},
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 3},
                }
            if tool_name == "tv.cast_media":
                return {
                    "content": [{"type": "text", "text": "cast"}],
                    "structuredContent": {
                        "deviceId": "lg_oled",
                        "deviceName": "LG OLED Phòng khách",
                        "appId": "youtube",
                    },
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 12},
                }
            if tool_name == "tv.launch_app":
                return {
                    "content": [{"type": "text", "text": "launched"}],
                    "structuredContent": {
                        "deviceId": "lg_oled",
                        "deviceName": "LG OLED Phòng khách",
                        "appId": "netflix",
                    },
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 10},
                }
            if tool_name == "workspace.git_summary":
                return {
                    "content": [{"type": "text", "text": "clean"}],
                    "structuredContent": {
                        "workspace": "xuna_pet",
                        "branch": "main",
                        "dirtyCount": 0,
                    },
                    "isError": False,
                    "_meta": {"xuna/status": "ok", "xuna/durationMs": 4},
                }
            return {
                "content": [{"type": "text", "text": "ok"}],
                "structuredContent": {},
                "isError": False,
                "_meta": {"xuna/status": "ok", "xuna/durationMs": 4},
            }

        hub.handler = _fake_handler
        approval_repo = InMemoryApprovalRepository()
        approval_broker = InMemoryApprovalBroker()
        local_broker = InMemoryLocalResultBroker()
        audit = MemoryAuditLog()

        runner = CompositeWorkflowRunner(
            host_repo=repo,
            hub=hub,
            approval_repo=approval_repo,
            approval_broker=approval_broker,
            local_result_broker=local_broker,
            audit=audit,
            clock=lambda: NOW,
        )

        events: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)
            if isinstance(ev, ActionPendingEvent):
                await approval_broker.decide(
                    uuid.UUID(ev.approval_id),
                    owner_id,
                    ApprovalDecision.APPROVE,
                    NOW,
                )

        # 1. S14.3: cast_to_tv (one R2 approval, pauses PC media, casts to TV)
        cast_msg = await runner.handle_template_call(
            ToolCall(
                id="call_cast_1",
                name="workflow_template",
                arguments={
                    "template": "cast_to_tv",
                    "source_target": "MacBook",
                    "tv_device": "LG OLED Phòng khách",
                },
            ),
            owner_id=owner_id,
            locale=Locale.VI,
            client_caps=set(),
            board_available=False,
            emit_event=_emit,
            request_id="req_cast_1",
        )
        cast_payload = json.loads(cast_msg.content)
        assert cast_payload["status"] == "ok"
        assert cast_payload["output"]["tvDevice"] == "LG OLED Phòng khách"
        assert cast_payload["output"]["appId"] == "youtube"
        assert cast_payload["output"]["domain"] == "www.youtube.com"
        assert cast_payload["output"]["mediaPaused"] is True
        pending_events = [e for e in events if isinstance(e, ActionPendingEvent)]
        assert len(pending_events) == 1
        assert "www.youtube.com" in pending_events[0].summary
        assert "secret_token_999" not in pending_events[0].summary

        # 2. S14.5: cinema_mode (wakes TV, launches Netflix, dims lights, pauses PC music)
        events.clear()
        cinema_msg = await runner.handle_template_call(
            ToolCall(
                id="call_cinema_1",
                name="workflow_template",
                arguments={
                    "template": "cinema_mode",
                    "source_target": "MacBook",
                    "tv_device": "LG OLED Phòng khách",
                    "tv_app_id": "netflix",
                    "light_entity_id": "light.living_room",
                },
            ),
            owner_id=owner_id,
            locale=Locale.VI,
            client_caps=set(),
            board_available=False,
            emit_event=_emit,
            request_id="req_cinema_1",
        )
        cinema_payload = json.loads(cinema_msg.content)
        assert cinema_payload["status"] == "ok"
        assert cinema_payload["output"]["tvAppId"] == "netflix"
        assert cinema_payload["output"]["lightsDimmed"] is True
        assert cinema_payload["output"]["mediaPaused"] is True

        # 3. UDC-T36: cinema_mode rejects hard-denied lock.* entityId before any host call
        denied_msg = await runner.handle_template_call(
            ToolCall(
                id="call_cinema_denied",
                name="workflow_template",
                arguments={
                    "template": "cinema_mode",
                    "source_target": "MacBook",
                    "light_entity_id": "lock.front_door",
                },
            ),
            owner_id=owner_id,
            locale=Locale.VI,
            client_caps=set(),
            board_available=False,
            emit_event=_emit,
            request_id="req_cinema_denied",
        )
        denied_payload = json.loads(denied_msg.content)
        assert denied_payload["status"] == "denied"
        assert denied_payload["output"]["reason"] == "smarthome_domain_forbidden"

        # 4. S14.5: goodnight_mode (R1: checks Git, pauses PC music, turns off TV & lights)
        events.clear()
        gn_msg = await runner.handle_template_call(
            ToolCall(
                id="call_gn_1",
                name="workflow_template",
                arguments={
                    "template": "goodnight_mode",
                    "source_target": "MacBook",
                    "tv_device": "LG OLED Phòng khách",
                    "light_entity_id": "light.living_room",
                },
            ),
            owner_id=owner_id,
            locale=Locale.VI,
            client_caps=set(),
            board_available=False,
            emit_event=_emit,
            request_id="req_gn_1",
        )
        gn_payload = json.loads(gn_msg.content)
        assert gn_payload["status"] == "ok"
        assert gn_payload["output"]["tvPoweredOff"] is True
        assert gn_payload["output"]["lightsOff"] is True
        assert gn_payload["output"]["mediaPaused"] is True
        assert gn_payload["output"]["locked"] is True
        assert not [e for e in events if isinstance(e, ActionPendingEvent)]

    asyncio.run(_scenario())
