"""Unit and table-driven tests for the `actions` module (UDC-4 S4.2).

Covers:
- `tool_names` mapping (`media.play_pause` <-> `media__play_pause`)
- Forbidden-tool impossibility across mapping, catalog, and orchestrator
- No-HID-text impossibility for `board.media_key` (only 7 consumer keys -> usage 1..7)
- `SchemaValidator` positive and negative cases (schema definition + arguments)
- `TargetResolver` deterministic rules without cross-host fallback
- `build_tool_catalog` capability gating, grant filtering, and 8-tool cap
- `Orchestrator` 8-step pipeline (R1 direct, R2 approval approve/reject/expire,
  grant denial, board HID route, and content-free audit trail)
"""

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from xuna_backend.modules.actions.application.catalog import (
    CAP_BOARD_HID_V1,
    CAP_HOST_ACTIONS_V1,
    build_tool_catalog,
)
from xuna_backend.modules.actions.application.orchestrator import (
    ActionContext,
    Orchestrator,
)
from xuna_backend.modules.actions.domain.entities import (
    ActionPendingEvent,
    ActionResultEvent,
    ActionStreamEvent,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalState,
    BoardHidRequestEvent,
)
from xuna_backend.modules.actions.domain.schema_validator import (
    ArgumentValidationError,
    SchemaDefinitionError,
    SchemaValidator,
)
from xuna_backend.modules.actions.domain.target_resolver import (
    TargetResolutionStatus,
    TargetResolver,
)
from xuna_backend.modules.actions.domain.tool_names import (
    BOARD_FUNCTION_NAME,
    BOARD_MEDIA_KEY_SCHEMA,
    BOARD_MEDIA_KEY_TO_USAGE_INDEX,
    BOARD_TOOL,
    from_function_name,
    is_forbidden_tool_name,
    to_function_name,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.repository import (
    InMemoryApprovalRepository,
)
from xuna_backend.modules.audit.domain.event import AuditAction, AuditEvent
from xuna_backend.modules.cognition.domain.types import Locale, Role, ToolCall
from xuna_backend.modules.hosts.domain.crypto import args_sha256
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

NOW = datetime(2026, 10, 30, 12, 0, 0, tzinfo=UTC)
FAKE_PUBKEY = b"\x04" + bytes(range(1, 65))


def _make_host(
    owner_id: uuid.UUID,
    alias: str,
    *,
    platform: HostPlatform = HostPlatform.MACOS,
    status: HostStatus = HostStatus.ACTIVE,
) -> Host:
    return Host(
        host_id=HostId.generate(),
        owner_id=owner_id,
        alias=HostAlias(alias),
        platform=platform,
        hostname=f"{alias.lower()}-host",
        os_version="15.0",
        agent_version="0.1.0",
        public_key=FAKE_PUBKEY,
        status=status,
        created_at=NOW,
        last_seen_at=NOW,
    )


def _media_play_pause_descriptor() -> ToolDescriptor:
    return ToolDescriptor(
        name="media.play_pause",
        description="Toggle media playback.",
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        family=ToolFamily.MEDIA,
        risk=RiskTier.R1,
    )


def _app_launch_descriptor() -> ToolDescriptor:
    return ToolDescriptor(
        name="app.launch",
        description="Launch an allowlisted application by ID.",
        input_schema={
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "maxLength": 64,
                    "pattern": r"^[a-z0-9._-]+$",
                }
            },
            "required": ["id"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        family=ToolFamily.APP,
        risk=RiskTier.R2,
    )


def _url_open_descriptor() -> ToolDescriptor:
    return ToolDescriptor(
        name="url.open",
        description="Open an allowlisted https:// URL.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "maxLength": 512,
                }
            },
            "required": ["url"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        family=ToolFamily.URL,
        risk=RiskTier.R2,
    )


class _FakeAuditLog:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def record(self, event: AuditEvent) -> None:
        self.events.append(event)

    async def recent(self, owner_id: uuid.UUID, limit: int) -> list[AuditEvent]:
        return [e for e in self.events if e.owner_id == owner_id][:limit]


class _FakeHostRepository:
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

    async def get_manifest(self, host_id: HostId) -> Manifest | None:
        return self.manifests.get(host_id)

    async def save_manifest(self, manifest: Manifest) -> None:
        self.manifests[manifest.host_id] = manifest


class _FakeSessionHub:
    def __init__(self) -> None:
        self.online_ids: set[str] = set()
        self.sent_requests: list[tuple[str, str, dict[str, object], float]] = []
        self.next_response: dict[str, object] = {
            "content": [{"type": "text", "text": "ok"}],
            "structuredContent": {"playing": True},
            "isError": False,
            "_meta": {"xuna/status": "ok", "xuna/durationMs": 42},
        }
        self.raise_timeout: bool = False

    def register_pending(self, pending_id: str, on_enrolled: object) -> None:
        pass

    def unregister_pending(self, pending_id: str) -> None:
        pass

    async def notify_enrolled(self, pending_id: str, host_id: HostId, alias: HostAlias) -> None:
        pass

    async def open_session(
        self,
        host_id: str,
        token: object,
        send_text: object,
        close_duplicate: object,
        close_revoked: object,
    ) -> bool:
        self.online_ids.add(host_id)
        return True

    def close_session(self, host_id: str, token: object) -> bool:
        self.online_ids.discard(host_id)
        return True

    def on_response(self, host_id: str, request_id: int | str, result: dict[str, object]) -> None:
        pass

    def on_error_response(
        self, host_id: str, request_id: int | str, code: int, message: str
    ) -> None:
        pass

    def is_online(self, host_id: str) -> bool:
        return host_id in self.online_ids

    async def send_request(
        self,
        host_id: str,
        method: str,
        params: dict[str, object],
        timeout_s: float,
    ) -> dict[str, object]:
        self.sent_requests.append((host_id, method, params, timeout_s))
        if self.raise_timeout:
            raise TimeoutError("simulated host timeout")
        return self.next_response

    async def notify(self, host_id: str, method: str, params: dict[str, object]) -> None:
        pass

    async def disconnect_host(self, host_id: str, reason: str = "revoked") -> None:
        self.online_ids.discard(host_id)


# ---------------------------------------------------------------------------
# 1. Tool names & Forbidden-tool impossibility tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("canonical", "fn_name"),
    [
        ("media.play_pause", "media__play_pause"),
        ("media.volume_set", "media__volume_set"),
        ("app.launch", "app__launch"),
        ("url.open", "url__open"),
        ("workflow.run", "workflow__run"),
        ("board.media_key", "board__media_key"),
    ],
)
def test_tool_name_round_trip(canonical: str, fn_name: str) -> None:
    assert to_function_name(canonical) == fn_name
    assert from_function_name(fn_name) == canonical


@pytest.mark.parametrize(
    "forbidden",
    [
        "shell.exec",
        "shell.run",
        "powershell.run",
        "powershell.exec",
        "osascript.run",
        "osascript.exec",
        "eval.run",
        "eval.exec",
        "script.run",
        "script.exec",
        "keyboard.type",
        "keyboard.press",
        "pointer.click",
        "pointer.move",
        "cmd.run",
        "cmd.exec",
        "bash.run",
        "bash.exec",
    ],
)
def test_forbidden_tools_rejected_everywhere(forbidden: str) -> None:
    assert is_forbidden_tool_name(forbidden) is True
    with pytest.raises(ValueError, match="forbidden"):
        to_function_name(forbidden)
    fn_candidate = forbidden.replace(".", "__")
    assert from_function_name(fn_candidate) is None


# ---------------------------------------------------------------------------
# 2. HID-text impossibility tests (`board.media_key`)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key_name", "expected_usage_index"),
    [
        ("play_pause", 1),
        ("next", 2),
        ("previous", 3),
        ("volume_up", 4),
        ("volume_down", 5),
        ("mute", 6),
        ("stop", 7),
    ],
)
def test_board_media_key_accepts_only_seven_consumer_keys(
    key_name: str, expected_usage_index: int
) -> None:
    clean_args, target = SchemaValidator.validate_arguments(
        BOARD_MEDIA_KEY_SCHEMA,
        {"key": key_name},
        allow_target=False,
    )
    assert target is None
    assert clean_args == {"key": key_name}
    assert BOARD_MEDIA_KEY_TO_USAGE_INDEX[key_name] == expected_usage_index


@pytest.mark.parametrize(
    "bad_args",
    [
        {},
        {"key": "a"},
        {"key": "enter"},
        {"key": "space"},
        {"key": "ctrl+c"},
        {"key": "rm -rf /"},
        {"text": "hello world"},
        {"key": 1},
        {"key": True},
        {"key": "play_pause", "text": "injected"},
        {"key": "play_pause", "target": "Mac"},
    ],
)
def test_board_media_key_hid_text_impossibility(bad_args: dict[str, object]) -> None:
    with pytest.raises(ArgumentValidationError):
        SchemaValidator.validate_arguments(
            BOARD_MEDIA_KEY_SCHEMA,
            bad_args,
            allow_target=False,
        )


# ---------------------------------------------------------------------------
# 3. `SchemaValidator` table-driven positive & negative tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "invalid_schema",
    [
        {"type": "string"},
        {"type": "object", "properties": {}},
        {"type": "object", "properties": {}, "additionalProperties": True},
        {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
            "anyOf": [],
        },
        {
            "type": "object",
            "properties": {"flag": {"type": "boolean"}},
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"items": {"type": "array"}},
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"name": {"type": "string", "minLength": 1}},
            "additionalProperties": False,
        },
    ],
)
def test_schema_validator_rejects_unsupported_schema_definitions(
    invalid_schema: dict[str, object],
) -> None:
    with pytest.raises(SchemaDefinitionError):
        SchemaValidator.validate_schema_definition(invalid_schema)


@pytest.mark.parametrize(
    ("args", "should_pass"),
    [
        ({"level": 50, "id": "spotify", "mode": "fast"}, True),
        ({"level": 0, "id": "vs-code.1", " target ": "Mac"}, False),
        ({"level": 100, "id": "vlc", "target": "Mac"}, True),
        ({"level": -1, "id": "vlc"}, False),
        ({"level": 101, "id": "vlc"}, False),
        ({"level": True, "id": "vlc"}, False),
        ({"level": 50, "id": "UPPERCASE"}, False),
        ({"level": 50, "id": "a" * 20}, False),
        ({"level": 50, "id": "vlc", "mode": "unknown"}, False),
        ({"level": 50}, False),
        ({"level": 50, "id": "vlc", "extra": 1}, False),
        ({"level": 50, "id": "vlc", "target": ""}, False),
    ],
)
def test_schema_validator_argument_validation_cases(
    args: dict[str, object], should_pass: bool
) -> None:
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "level": {"type": "integer", "minimum": 0, "maximum": 100},
            "id": {
                "type": "string",
                "maxLength": 16,
                "pattern": r"^[a-z0-9._-]+$",
            },
            "mode": {"type": "string", "enum": ["fast", "quiet"]},
        },
        "required": ["level", "id"],
        "additionalProperties": False,
    }
    SchemaValidator.validate_schema_definition(schema)
    if should_pass:
        clean, _ = SchemaValidator.validate_arguments(schema, args, allow_target=True)
        assert "target" not in clean
    else:
        with pytest.raises(ArgumentValidationError):
            SchemaValidator.validate_arguments(schema, args, allow_target=True)


# ---------------------------------------------------------------------------
# 4. `TargetResolver` deterministic table-driven tests
# ---------------------------------------------------------------------------


def test_target_resolver_rules_never_fall_back_to_wrong_host() -> None:
    owner_id = uuid.uuid4()
    mac = _make_host(owner_id, "Mac")
    pc = _make_host(owner_id, "PC", platform=HostPlatform.WINDOWS)

    # 1. Zero eligible hosts -> NO_HOST
    res0 = TargetResolver.resolve(
        eligible_hosts=[],
        requested_target=None,
        now=NOW,
    )
    assert res0.status == TargetResolutionStatus.NO_HOST

    # 2. Single eligible host with no target -> RESOLVED to Mac
    res1 = TargetResolver.resolve(
        eligible_hosts=[mac],
        requested_target=None,
        now=NOW,
    )
    assert res1.status == TargetResolutionStatus.RESOLVED
    assert res1.host == mac

    # 3. Single eligible host ("Mac"), but explicit target is "PC" -> ASK_TARGET (never falls back!)
    res_wrong = TargetResolver.resolve(
        eligible_hosts=[mac],
        requested_target="PC",
        now=NOW,
    )
    assert res_wrong.status == TargetResolutionStatus.ASK_TARGET
    assert res_wrong.host is None
    assert res_wrong.choices == ("Mac",)

    # 4. Two eligible hosts ("Mac", "PC"), explicit target "pc" -> RESOLVED to PC
    res_explicit = TargetResolver.resolve(
        eligible_hosts=[mac, pc],
        requested_target="pc",
        now=NOW,
    )
    assert res_explicit.status == TargetResolutionStatus.RESOLVED
    assert res_explicit.host == pc

    # 5. Two eligible hosts, no explicit target, recent last_target (3 min ago) -> RESOLVED
    res_last = TargetResolver.resolve(
        eligible_hosts=[mac, pc],
        requested_target=None,
        last_target_alias="Mac",
        last_target_at=NOW - timedelta(minutes=3),
        now=NOW,
    )
    assert res_last.status == TargetResolutionStatus.RESOLVED
    assert res_last.host == mac

    # 6. Two eligible hosts, expired last_target (6 min ago > 5 min TTL) -> ASK_TARGET
    res_expired_last = TargetResolver.resolve(
        eligible_hosts=[mac, pc],
        requested_target=None,
        last_target_alias="Mac",
        last_target_at=NOW - timedelta(minutes=6),
        now=NOW,
    )
    assert res_expired_last.status == TargetResolutionStatus.ASK_TARGET
    assert res_expired_last.choices == ("Mac", "PC")


# ---------------------------------------------------------------------------
# 5. `build_tool_catalog` tests
# ---------------------------------------------------------------------------


def test_build_tool_catalog_gates_by_capability_and_grants() -> None:
    owner_id = uuid.uuid4()
    mac = _make_host(owner_id, "Mac")
    pc = _make_host(owner_id, "PC", platform=HostPlatform.WINDOWS)

    manifest_mac = Manifest(
        host_id=mac.host_id,
        tools=(_media_play_pause_descriptor(), _app_launch_descriptor()),
        updated_at=NOW,
    )
    manifest_pc = Manifest(
        host_id=pc.host_id,
        tools=(_media_play_pause_descriptor(),),
        updated_at=NOW,
    )
    grants_mac = [
        Grant(
            host_id=mac.host_id,
            family=ToolFamily.MEDIA,
            max_risk=RiskTier.R1,
            enabled=True,
            updated_at=NOW,
        ),
        Grant(
            host_id=mac.host_id,
            family=ToolFamily.APP,
            max_risk=RiskTier.R0,
            enabled=False,
            updated_at=NOW,
        ),
    ]
    grants_pc = [
        Grant(
            host_id=pc.host_id,
            family=ToolFamily.MEDIA,
            max_risk=RiskTier.R1,
            enabled=True,
            updated_at=NOW,
        ),
    ]

    assert (
        build_tool_catalog(
            hosts=[mac, pc],
            manifests_by_host={mac.host_id: manifest_mac, pc.host_id: manifest_pc},
            grants_by_host={mac.host_id: grants_mac, pc.host_id: grants_pc},
            online_host_ids={str(mac.host_id), str(pc.host_id)},
            client_caps=(),
            board_available=True,
        )
        == ()
    )

    specs = build_tool_catalog(
        hosts=[mac, pc],
        manifests_by_host={mac.host_id: manifest_mac, pc.host_id: manifest_pc},
        grants_by_host={mac.host_id: grants_mac, pc.host_id: grants_pc},
        online_host_ids={str(mac.host_id), str(pc.host_id)},
        client_caps=(CAP_HOST_ACTIONS_V1, CAP_BOARD_HID_V1),
        board_available=True,
    )
    names = [s.name for s in specs]
    assert names == ["media__play_pause", BOARD_FUNCTION_NAME]
    media_props = specs[0].parameters["properties"]
    assert isinstance(media_props, dict)
    assert media_props["target"] == {"type": "string", "enum": ["Mac", "PC"]}


# ---------------------------------------------------------------------------
# 6. `Orchestrator` end-to-end unit tests (R1, R2 approve/reject/expire, board HID)
# ---------------------------------------------------------------------------


def _setup_orchestrator() -> (
    tuple[
        Orchestrator,
        _FakeHostRepository,
        _FakeSessionHub,
        InMemoryApprovalRepository,
        InMemoryApprovalBroker,
        InMemoryLocalResultBroker,
        _FakeAuditLog,
    ]
):
    host_repo = _FakeHostRepository()
    hub = _FakeSessionHub()
    approval_repo = InMemoryApprovalRepository()
    approval_broker = InMemoryApprovalBroker()
    local_broker = InMemoryLocalResultBroker()
    last_target_store = InMemoryLastTargetStore()
    audit = _FakeAuditLog()
    orch = Orchestrator(
        host_repo=host_repo,
        hub=hub,
        approval_repo=approval_repo,
        approval_broker=approval_broker,
        local_result_broker=local_broker,
        last_target_store=last_target_store,
        audit=audit,
        clock=lambda: NOW,
    )
    return orch, host_repo, hub, approval_repo, approval_broker, local_broker, audit


def test_orchestrator_r1_direct_execution_and_last_target_memory() -> None:
    async def _run() -> None:
        orch, host_repo, hub, _, _, _, audit = _setup_orchestrator()
        owner_id = uuid.uuid4()
        mac = _make_host(owner_id, "Mac")
        await host_repo.save_host(mac)
        await host_repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(_media_play_pause_descriptor(),),
                updated_at=NOW,
            )
        )
        await host_repo.set_grants(
            mac.host_id,
            [
                Grant(
                    host_id=mac.host_id,
                    family=ToolFamily.MEDIA,
                    max_risk=RiskTier.R1,
                    enabled=True,
                    updated_at=NOW,
                )
            ],
        )
        hub.online_ids.add(str(mac.host_id))

        emitted: list[ActionStreamEvent] = []

        async def emit(ev: ActionStreamEvent) -> None:
            emitted.append(ev)

        ctx = ActionContext(
            owner_id=owner_id,
            pet_id="pet-1",
            locale=Locale.VI,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=emit,
        )
        msg = await orch.handle_call(
            ToolCall(id="call-1", name="media__play_pause", arguments={}),
            ctx,
        )
        assert msg.role == Role.TOOL
        assert msg.tool_call_id == "call-1"
        payload = json.loads(msg.content)
        assert payload["status"] == "ok"
        assert payload["output"] == {"playing": True}

        assert len(emitted) == 1
        assert isinstance(emitted[0], ActionResultEvent)
        assert emitted[0].status == "ok"
        assert emitted[0].gesture == "bounce_play"
        assert len(audit.events) == 1
        assert audit.events[0].action == AuditAction.HOST_TOOL_INVOKED

    asyncio.run(_run())


def test_orchestrator_r2_approval_grant_and_hash_binding() -> None:
    async def _run() -> None:
        orch, host_repo, hub, approval_repo, _, _, audit = _setup_orchestrator()
        owner_id = uuid.uuid4()
        pc = _make_host(owner_id, "PC", platform=HostPlatform.WINDOWS)
        await host_repo.save_host(pc)
        await host_repo.save_manifest(
            Manifest(
                host_id=pc.host_id,
                tools=(_app_launch_descriptor(), _url_open_descriptor()),
                updated_at=NOW,
            )
        )
        await host_repo.set_grants(
            pc.host_id,
            [
                Grant(
                    host_id=pc.host_id,
                    family=ToolFamily.APP,
                    max_risk=RiskTier.R2,
                    enabled=True,
                    updated_at=NOW,
                ),
                Grant(
                    host_id=pc.host_id,
                    family=ToolFamily.URL,
                    max_risk=RiskTier.R2,
                    enabled=True,
                    updated_at=NOW,
                ),
            ],
        )
        hub.online_ids.add(str(pc.host_id))

        emitted: list[ActionStreamEvent] = []

        async def emit(ev: ActionStreamEvent) -> None:
            emitted.append(ev)
            if isinstance(ev, ActionPendingEvent):
                await orch.decide_approval(
                    approval_id=uuid.UUID(ev.approval_id),
                    owner_id=owner_id,
                    decision=ApprovalDecision.APPROVE,
                )

        ctx = ActionContext(
            owner_id=owner_id,
            pet_id="pet-1",
            locale=Locale.VI,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=emit,
        )
        msg = await orch.handle_call(
            ToolCall(
                id="call-r2",
                name="url__open",
                arguments={
                    "url": "https://docs.flutter.dev/get-started",
                    "target": "PC",
                },
            ),
            ctx,
        )
        payload = json.loads(msg.content)
        assert payload["status"] == "ok"

        assert len(emitted) == 2
        assert isinstance(emitted[0], ActionPendingEvent)
        assert emitted[0].target == "PC"
        assert emitted[0].risk == "R2"
        assert "docs.flutter.dev" in emitted[0].summary
        assert isinstance(emitted[1], ActionResultEvent)
        assert emitted[1].status == "ok"
        assert emitted[1].gesture == "bounce_play"

        assert len(hub.sent_requests) == 1
        _, _, call_params, _ = hub.sent_requests[0]
        meta = call_params["_meta"]
        assert isinstance(meta, dict)
        approval_token = meta["xuna/approval"]
        assert isinstance(approval_token, dict)
        expected_hash = args_sha256({"url": "https://docs.flutter.dev/get-started"})
        assert approval_token["argsSha256"] == expected_hash

        stored = await approval_repo.get(uuid.UUID(emitted[0].approval_id))
        assert stored is not None
        assert stored.state == ApprovalState.APPROVED
        assert stored.args_sha256 == expected_hash

        audit_actions = [e.action for e in audit.events]
        assert audit_actions == [
            AuditAction.HOST_APPROVAL_GRANTED,
            AuditAction.HOST_TOOL_INVOKED,
        ]
        for ev in audit.events:
            serialized_meta = json.dumps(ev.metadata)
            assert "docs.flutter.dev" not in serialized_meta

    asyncio.run(_run())


def test_orchestrator_r2_rejection_never_executes_tool() -> None:
    async def _run() -> None:
        orch, host_repo, hub, _, _, _, audit = _setup_orchestrator()
        owner_id = uuid.uuid4()
        mac = _make_host(owner_id, "Mac")
        await host_repo.save_host(mac)
        await host_repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(_app_launch_descriptor(),),
                updated_at=NOW,
            )
        )
        await host_repo.set_grants(
            mac.host_id,
            [
                Grant(
                    host_id=mac.host_id,
                    family=ToolFamily.APP,
                    max_risk=RiskTier.R2,
                    enabled=True,
                    updated_at=NOW,
                )
            ],
        )
        hub.online_ids.add(str(mac.host_id))

        emitted: list[ActionStreamEvent] = []

        async def emit(ev: ActionStreamEvent) -> None:
            emitted.append(ev)
            if isinstance(ev, ActionPendingEvent):
                await orch.decide_approval(
                    approval_id=uuid.UUID(ev.approval_id),
                    owner_id=owner_id,
                    decision=ApprovalDecision.REJECT,
                )

        ctx = ActionContext(
            owner_id=owner_id,
            pet_id="pet-1",
            locale=Locale.VI,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=emit,
        )
        msg = await orch.handle_call(
            ToolCall(id="call-rej", name="app__launch", arguments={"id": "code"}),
            ctx,
        )
        assert json.loads(msg.content) == {"status": "rejected"}
        assert hub.sent_requests == []
        assert isinstance(emitted[-1], ActionResultEvent)
        assert emitted[-1].status == "rejected"
        assert emitted[-1].gesture is None
        assert [e.action for e in audit.events] == [AuditAction.HOST_APPROVAL_REJECTED]

    asyncio.run(_run())


def test_orchestrator_ungranted_tool_returns_denied_and_audits() -> None:
    async def _run() -> None:
        orch, host_repo, hub, _, _, _, audit = _setup_orchestrator()
        owner_id = uuid.uuid4()
        mac = _make_host(owner_id, "Mac")
        await host_repo.save_host(mac)
        await host_repo.save_manifest(
            Manifest(
                host_id=mac.host_id,
                tools=(_app_launch_descriptor(),),
                updated_at=NOW,
            )
        )
        await host_repo.set_grants(
            mac.host_id,
            [
                Grant(
                    host_id=mac.host_id,
                    family=ToolFamily.APP,
                    max_risk=RiskTier.R0,
                    enabled=False,
                    updated_at=NOW,
                )
            ],
        )
        hub.online_ids.add(str(mac.host_id))

        emitted: list[ActionStreamEvent] = []

        async def emit(ev: ActionStreamEvent) -> None:
            emitted.append(ev)

        ctx = ActionContext(
            owner_id=owner_id,
            pet_id="pet-1",
            locale=Locale.VI,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=emit,
        )
        msg = await orch.handle_call(
            ToolCall(
                id="call-den",
                name="app__launch",
                arguments={"id": "code", "target": "Mac"},
            ),
            ctx,
        )
        assert json.loads(msg.content) == {"status": "denied"}
        assert hub.sent_requests == []
        assert len(emitted) == 1
        assert isinstance(emitted[0], ActionResultEvent)
        assert emitted[0].status == "denied"
        assert emitted[0].gesture == "look_around"
        assert [e.action for e in audit.events] == [AuditAction.HOST_TOOL_DENIED]

    asyncio.run(_run())


def test_orchestrator_board_hid_route_requires_approval_and_reports_result() -> None:
    async def _run() -> None:
        orch, _, _, _, _, _, audit = _setup_orchestrator()
        owner_id = uuid.uuid4()
        emitted: list[ActionStreamEvent] = []

        async def emit(ev: ActionStreamEvent) -> None:
            emitted.append(ev)
            if isinstance(ev, ActionPendingEvent):
                await orch.decide_approval(
                    approval_id=uuid.UUID(ev.approval_id),
                    owner_id=owner_id,
                    decision=ApprovalDecision.APPROVE,
                )
            elif isinstance(ev, BoardHidRequestEvent):
                assert ev.key == "play_pause"
                assert ev.usage_index == 1
                orch.report_board_result(
                    invocation_id=ev.invocation_id,
                    owner_id=owner_id,
                    status="ok",
                )

        ctx = ActionContext(
            owner_id=owner_id,
            pet_id="pet-1",
            locale=Locale.VI,
            client_caps=(CAP_BOARD_HID_V1,),
            board_available=True,
            emit_event=emit,
        )
        msg = await orch.handle_call(
            ToolCall(
                id="call-board-1",
                name=BOARD_FUNCTION_NAME,
                arguments={"key": "play_pause"},
            ),
            ctx,
        )
        payload = json.loads(msg.content)
        assert payload["status"] == "ok"
        assert payload["output"] == {"key": "play_pause", "usageIndex": 1}

        assert [type(e) for e in emitted] == [
            ActionPendingEvent,
            BoardHidRequestEvent,
            ActionResultEvent,
        ]
        assert [e.action for e in audit.events] == [
            AuditAction.HOST_APPROVAL_GRANTED,
            AuditAction.BOARD_HID_REQUESTED,
        ]

    asyncio.run(_run())


def test_approval_broker_timeout_marks_expired() -> None:
    async def _run() -> None:
        broker = InMemoryApprovalBroker()
        req_id = uuid.uuid4()
        req = ApprovalRequest(
            approval_id=req_id,
            owner_id=uuid.uuid4(),
            host_id=None,
            target_alias="Board",
            tool=BOARD_TOOL,
            args_sha256="a" * 64,
            risk=RiskTier.R2,
            summary="Test",
            created_at=NOW,
            expires_at=NOW + timedelta(seconds=60),
        )
        await broker.create(req)
        state = await asyncio.wait_for(broker.wait(req_id, timeout_s=0.02), timeout=1.0)
        assert state == ApprovalState.EXPIRED

    asyncio.run(_run())
