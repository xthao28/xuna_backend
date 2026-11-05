"""Deferred host-tool discovery (`tool_search` / `tool_describe` / `tool_invoke`), the
deterministic catalog, the bounded tool payload `hint`, and the chat-turn loop guard."""

import asyncio
import json
import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from xuna_backend.modules.actions.application.catalog import (
    CAP_HOST_ACTIONS_V1,
    MAX_CATALOG_TOOLS,
    build_tool_catalog,
    collect_eligible_host_tools,
    load_tool_catalog_for_owner,
    tool_usage_counts,
)
from xuna_backend.modules.actions.application.orchestrator import (
    ActionContext,
    Orchestrator,
    is_repeat_safe,
)
from xuna_backend.modules.actions.application.prompts import (
    encode_tool_payload,
    tool_system_prompt,
)
from xuna_backend.modules.actions.application.tool_discovery import (
    TOOL_DESCRIBE_FUNCTION,
    TOOL_INVOKE_FUNCTION,
    TOOL_SEARCH_FUNCTION,
    MetaToolArgumentError,
    describe_tool,
    parse_invoke,
    search_tools,
)
from xuna_backend.modules.actions.domain.entities import (
    MAX_TOOL_OUTPUT_BYTES,
    MAX_TOOL_ROUNDS,
    ActionStreamEvent,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.repository import InMemoryApprovalRepository
from xuna_backend.modules.audit.domain.event import AuditAction, AuditEvent
from xuna_backend.modules.cognition.domain.types import (
    ChatEvent,
    ChatFinished,
    ChatRequest,
    Locale,
    Message,
    Role,
    TextDelta,
    ToolCall,
    ToolCallRequested,
    ToolSpec,
)
from xuna_backend.modules.conversation.application.chat_turn import build_request, chat_turn
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

NOW = datetime(2026, 11, 5, 9, 0, 0, tzinfo=UTC)

# 24 real tool names: more than the 20 catalog slots, so the catalog must defer.
_TOOL_NAMES: tuple[str, ...] = (
    "media.play_pause",
    "media.next",
    "media.previous",
    "media.volume_set",
    "media.now_playing",
    "media.play_search",
    "app.launch",
    "app.list_allowed",
    "app.open_scheme",
    "url.open",
    "ui.accessibility_tree",
    "notifications.list",
    "notifications.action",
    "clipboard.read",
    "clipboard.write",
    "window.active_context",
    "selection.read",
    "dev.ports",
    "system.vitals",
    "tv.control",
    "tv.list_devices",
    "smarthome.control",
    "smarthome.list_entities",
    "display.wake",
)

_DESCRIPTIONS: dict[str, str] = {
    "tv.control": "Control a paired smart TV: power, volume, input and channel.",
    "smarthome.control": "Turn smart home lights, fans or switches on or off.",
    "clipboard.read": "Read the current clipboard text.",
}


def _descriptor(name: str, *, risk: RiskTier = RiskTier.R1) -> ToolDescriptor:
    props: dict[str, object] = {}
    required: list[str] = []
    if name == "tv.control":
        props = {
            "command": {"type": "string", "enum": ["power", "volume_up", "volume_down"]},
            "level": {"type": "integer", "minimum": 0, "maximum": 100},
        }
        required = ["command"]
    schema: dict[str, object] = {
        "type": "object",
        "properties": props,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return ToolDescriptor(
        name=name,
        description=_DESCRIPTIONS.get(name, f"Host tool {name}."),
        input_schema=schema,
        output_schema={"type": "object", "properties": {}, "additionalProperties": False},
        family=validate_tool_name(name),
        risk=risk,
    )


def _host(owner_id: uuid.UUID, alias: str = "Mac") -> Host:
    return Host(
        host_id=HostId.generate(),
        owner_id=owner_id,
        alias=HostAlias(alias),
        platform=HostPlatform.MACOS,
        hostname=f"{alias.lower()}-host",
        os_version="15.0",
        agent_version="0.1.0",
        public_key=b"\x04" + bytes(range(1, 65)),
        status=HostStatus.ACTIVE,
        created_at=NOW,
        last_seen_at=NOW,
    )


def _grants(host_id: HostId, families: set[ToolFamily]) -> list[Grant]:
    return [
        Grant(host_id=host_id, family=f, max_risk=RiskTier.R1, enabled=True, updated_at=NOW)
        for f in sorted(families, key=lambda f: f.value)
    ]


def _catalog_inputs(
    names: tuple[str, ...] = _TOOL_NAMES,
    *,
    granted: set[ToolFamily] | None = None,
) -> tuple[Host, Manifest, list[Grant]]:
    host = _host(uuid.uuid4())
    tools = tuple(_descriptor(n) for n in names)
    manifest = Manifest(host_id=host.host_id, tools=tools, updated_at=NOW)
    families = granted if granted is not None else {t.family for t in tools}
    return host, manifest, _grants(host.host_id, families)


def _build(names: tuple[str, ...] = _TOOL_NAMES) -> tuple[ToolSpec, ...]:
    host, manifest, grants = _catalog_inputs(names)
    return build_tool_catalog(
        hosts=[host],
        manifests_by_host={host.host_id: manifest},
        grants_by_host={host.host_id: grants},
        online_host_ids={str(host.host_id)},
        client_caps=(CAP_HOST_ACTIONS_V1,),
        host_screen_consent=True,
        now=NOW,
    )


# ---------------------------------------------------------------------------
# Catalog: deterministic, core-first, defers instead of dropping
# ---------------------------------------------------------------------------


def test_catalog_defers_overflow_behind_meta_tools() -> None:
    specs = _build()
    names = [s.name for s in specs]
    assert len(specs) <= MAX_CATALOG_TOOLS
    assert names[:4] == ["app__launch", "app__list_allowed", "ui__accessibility_tree", "url__open"]
    assert {TOOL_SEARCH_FUNCTION, TOOL_DESCRIBE_FUNCTION, TOOL_INVOKE_FUNCTION} <= set(names)
    direct = {n.replace("__", ".") for n in names if "__" in n}
    search_spec = next(s for s in specs if s.name == TOOL_SEARCH_FUNCTION)
    for tool in _TOOL_NAMES:
        # Every granted tool is either sent directly or named in the search directory.
        assert tool in direct or tool in search_spec.description


def test_catalog_sends_all_tools_directly_when_they_fit() -> None:
    specs = _build(_TOOL_NAMES[:6])
    names = {s.name for s in specs}
    assert TOOL_SEARCH_FUNCTION not in names
    assert names == {n.replace(".", "__") for n in _TOOL_NAMES[:6]}


def test_catalog_order_does_not_depend_on_the_utterance() -> None:
    assert [s.name for s in _build()] == [s.name for s in _build()]


def test_prompt_adds_discovery_module_only_with_meta_tools() -> None:
    with_meta = tool_system_prompt(Locale.VI, tools=_build())
    without_meta = tool_system_prompt(Locale.VI, tools=_build(_TOOL_NAMES[:6]))
    assert "tool_search" in with_meta
    assert "tool_search" not in without_meta


# ---------------------------------------------------------------------------
# tool_search / tool_describe / tool_invoke parsing
# ---------------------------------------------------------------------------


def _eligible(granted: set[ToolFamily] | None = None) -> dict[str, Any]:
    host, manifest, grants = _catalog_inputs(granted=granted)
    return collect_eligible_host_tools(
        hosts=[host],
        manifests_by_host={host.host_id: manifest},
        grants_by_host={host.host_id: grants},
        online_host_ids={str(host.host_id)},
        host_screen_consent=True,
        now=NOW,
    )


@pytest.mark.parametrize(
    ("query", "expected_first"),
    [
        ("tv volume", "tv.control"),
        ("smart home light", "smarthome.control"),
        ("clipboard", "clipboard.read"),
        ("tv.list_devices", "tv.list_devices"),
    ],
)
def test_search_ranks_name_and_description_matches(query: str, expected_first: str) -> None:
    result = search_tools(_eligible(), query)
    matches = result["matches"]
    assert isinstance(matches, list)
    assert matches[0]["name"] == expected_first
    assert matches[0]["hosts"] == ["Mac"]


def test_search_without_match_returns_tool_names_instead_of_nothing() -> None:
    result = search_tools(_eligible(), "zzzz qqqq")
    assert result["matches"] == []
    assert "tv.control" in result["allTools"]  # type: ignore[operator]


@pytest.mark.parametrize("bad", [None, "", "   ", 7, "x" * 121])
def test_search_rejects_bad_query(bad: object) -> None:
    with pytest.raises(MetaToolArgumentError) as err:
        search_tools(_eligible(), bad)
    assert err.value.code == "invalid_args"


def test_describe_renders_compact_signature() -> None:
    payload = json.loads(describe_tool(_eligible(), "tv__control"))
    args = payload["output"]["args"]
    assert payload["status"] == "ok"
    assert "power|volume_up|volume_down" in args["command"]
    assert "REQUIRED" in args["command"]
    assert "range 0..100" in args["level"]


@pytest.mark.parametrize("name", ["shell.exec", "nope.tool", "", None])
def test_describe_rejects_unknown_or_forbidden(name: object) -> None:
    with pytest.raises(MetaToolArgumentError) as err:
        describe_tool(_eligible(), name)
    assert err.value.code == "unknown_tool"


def test_parse_invoke_merges_target_and_validates_json() -> None:
    req = parse_invoke(
        _eligible(),
        {"name": "tv.control", "argumentsJson": '{"command": "power"}', "target": "Mac"},
    )
    assert req.tool_name == "tv.control"
    assert req.function_name == "tv__control"
    assert req.arguments == {"command": "power", "target": "Mac"}


@pytest.mark.parametrize(
    ("arguments", "code"),
    [
        ({"name": "tv.control", "argumentsJson": "{not json"}, "invalid_args"),
        ({"name": "tv.control", "argumentsJson": "[1, 2]"}, "invalid_args"),
        ({"name": "shell.exec", "argumentsJson": "{}"}, "unknown_tool"),
        ({"name": "bash__run"}, "unknown_tool"),
    ],
)
def test_parse_invoke_rejects_bad_calls(arguments: dict[str, object], code: str) -> None:
    with pytest.raises(MetaToolArgumentError) as err:
        parse_invoke(_eligible(), arguments)
    assert err.value.code == code


def test_ungranted_tool_is_not_reachable_through_discovery() -> None:
    # `url.open` is the URL family, which is not granted here.
    eligible = _eligible(granted={ToolFamily.MEDIA, ToolFamily.APP})
    assert "url.open" not in eligible
    with pytest.raises(MetaToolArgumentError):
        parse_invoke(eligible, {"name": "url.open"})


# ---------------------------------------------------------------------------
# Orchestrator: tool_invoke re-enters the full pipeline
# ---------------------------------------------------------------------------


class _Audit:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def record(self, event: AuditEvent) -> None:
        self.events.append(event)

    async def recent(self, owner_id: uuid.UUID, limit: int) -> list[AuditEvent]:
        return self.events[:limit]


class _Repo:
    def __init__(self, host: Host, manifest: Manifest, grants: list[Grant]) -> None:
        self.host = host
        self.manifest = manifest
        self.grants = grants

    async def list_hosts(self, owner_id: uuid.UUID) -> list[Host]:
        return [self.host] if self.host.owner_id == owner_id else []

    async def get_manifest(self, host_id: HostId) -> Manifest | None:
        return self.manifest if host_id == self.host.host_id else None

    async def get_grants(self, host_id: HostId) -> list[Grant]:
        return list(self.grants) if host_id == self.host.host_id else []


class _Hub:
    def __init__(self, online: str, response: dict[str, object]) -> None:
        self.online = online
        self.response = response
        self.sent: list[dict[str, object]] = []

    def is_online(self, host_id: str) -> bool:
        return host_id == self.online

    async def send_request(
        self, host_id: str, method: str, params: dict[str, object], timeout_s: float
    ) -> dict[str, object]:
        self.sent.append(params)
        return self.response


_OK_RESPONSE: dict[str, object] = {
    "structuredContent": {"done": True},
    "_meta": {"xuna/status": "ok", "xuna/durationMs": 5},
}


def _orchestrator(
    response: dict[str, object] = _OK_RESPONSE,
    granted: set[ToolFamily] | None = None,
) -> tuple[Orchestrator, _Hub, ActionContext]:
    host, manifest, grants = _catalog_inputs(granted=granted)
    hub = _Hub(str(host.host_id), response)
    orch = Orchestrator(
        host_repo=_Repo(host, manifest, grants),  # type: ignore[arg-type]
        hub=hub,  # type: ignore[arg-type]
        approval_repo=InMemoryApprovalRepository(),
        approval_broker=InMemoryApprovalBroker(),
        local_result_broker=InMemoryLocalResultBroker(),
        last_target_store=InMemoryLastTargetStore(),
        audit=_Audit(),
        clock=lambda: NOW,
    )

    async def emit(ev: ActionStreamEvent) -> None:
        return None

    ctx = ActionContext(
        owner_id=host.owner_id,
        pet_id="pet-1",
        locale=Locale.VI,
        client_caps=(CAP_HOST_ACTIONS_V1,),
        board_available=False,
        emit_event=emit,
    )
    return orch, hub, ctx


def test_tool_invoke_dispatches_deferred_tool_through_pipeline() -> None:
    orch, hub, ctx = _orchestrator()
    call = ToolCall(
        id="c1",
        name=TOOL_INVOKE_FUNCTION,
        arguments={"name": "tv.control", "argumentsJson": '{"command": "volume_up"}'},
    )
    msg = asyncio.run(orch.handle_call(call, ctx))
    assert msg.tool_call_id == "c1"
    assert json.loads(msg.content)["status"] == "ok"
    assert hub.sent[0]["name"] == "tv.control"
    assert hub.sent[0]["arguments"] == {"command": "volume_up"}


def test_tool_invoke_still_validates_inner_arguments() -> None:
    orch, hub, ctx = _orchestrator()
    call = ToolCall(
        id="c2",
        name=TOOL_INVOKE_FUNCTION,
        arguments={"name": "tv.control", "argumentsJson": '{"command": "self_destruct"}'},
    )
    msg = asyncio.run(orch.handle_call(call, ctx))
    assert json.loads(msg.content)["status"] == "invalid_args"
    assert hub.sent == []


def test_tool_invoke_cannot_reach_ungranted_tool() -> None:
    orch, hub, ctx = _orchestrator(granted={ToolFamily.MEDIA})
    call = ToolCall(id="c3", name=TOOL_INVOKE_FUNCTION, arguments={"name": "url.open"})
    payload = json.loads(asyncio.run(orch.handle_call(call, ctx)).content)
    assert payload["status"] == "unknown_tool"
    assert "tool_search" in payload["hint"]
    assert hub.sent == []


def test_meta_tools_require_host_actions_capability() -> None:
    orch, hub, ctx = _orchestrator()
    no_caps = ActionContext(
        owner_id=ctx.owner_id,
        pet_id=ctx.pet_id,
        locale=ctx.locale,
        client_caps=(),
        board_available=False,
        emit_event=ctx.emit_event,
    )
    call = ToolCall(id="c4", name=TOOL_SEARCH_FUNCTION, arguments={"query": "tv"})
    assert json.loads(asyncio.run(orch.handle_call(call, no_caps)).content) == {
        "status": "unknown_tool"
    }


def test_tool_search_via_orchestrator_returns_matches() -> None:
    orch, _, ctx = _orchestrator()
    call = ToolCall(id="c5", name=TOOL_SEARCH_FUNCTION, arguments={"query": "tv volume"})
    payload = json.loads(asyncio.run(orch.handle_call(call, ctx)).content)
    assert payload["output"]["matches"][0]["name"] == "tv.control"


def test_remote_error_code_reaches_model_with_hint() -> None:
    response: dict[str, object] = {
        "structuredContent": {},
        "_meta": {"xuna/status": "failed", "xuna/errorCode": "stale_ref"},
    }
    orch, _, ctx = _orchestrator(response=response)
    call = ToolCall(id="c6", name="ui__accessibility_tree", arguments={})
    payload = json.loads(asyncio.run(orch.handle_call(call, ctx)).content)
    assert payload["status"] == "failed"
    assert payload["output"]["errorCode"] == "stale_ref"
    assert "re-inspect" in payload["hint"]


# ---------------------------------------------------------------------------
# Tool payload: hint outside output, UI tree shrinks instead of vanishing
# ---------------------------------------------------------------------------


def test_hint_is_next_to_output_not_inside() -> None:
    payload = json.loads(encode_tool_payload("failed", {"errorCode": "missing_ref"}))
    assert "hint" not in payload["output"]
    assert "refId" in payload["hint"]


def test_large_vietnamese_ui_tree_is_shrunk_not_dropped() -> None:
    # Diacritic-heavy labels are ~3 bytes/char, so the first pass overflows 6 KiB.
    tree = "\n".join(f"[@s1:e{i}] AXButton 'Ưỡng ữ ặ ỗ ộ ễ ậ ợ {i}'" for i in range(300))
    output: dict[str, object] = {
        "compactUiTree": tree,
        "summaryVi": "ữ" * 1100,
        "nodes": [{"label": "ữ" * 300}] * 16,
    }
    encoded = encode_tool_payload("ok", output)
    assert len(encoded.encode("utf-8")) <= MAX_TOOL_OUTPUT_BYTES
    payload = json.loads(encoded)
    kept_tree = payload["output"]["compactUiTree"]
    assert kept_tree.startswith("[@s1:e0]")
    assert len(kept_tree) < len(tree)
    assert "nodes" not in payload["output"]
    assert "truncated" not in payload["output"]


# ---------------------------------------------------------------------------
# chat_turn: multi-round loop with guard
# ---------------------------------------------------------------------------


class _ScriptedLLM:
    def __init__(self, calls_per_round: list[list[ToolCall]], final_text: str) -> None:
        self.calls_per_round = calls_per_round
        self.final_text = final_text
        self.requests: list[ChatRequest] = []

    async def stream(self, request: ChatRequest) -> Any:
        idx = len(self.requests)
        self.requests.append(request)
        if request.tools and idx < len(self.calls_per_round):
            for call in self.calls_per_round[idx]:
                yield ToolCallRequested(call)
        else:
            yield TextDelta(self.final_text)
        yield ChatFinished(
            provider="fake", model="fake", input_tokens=1, output_tokens=1, latency_ms=1
        )


_SPEC = ToolSpec(
    name="ui__accessibility_tree",
    description="UI",
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
)


def _run_turn(
    llm: _ScriptedLLM,
    runner: Any,
    tools: tuple[ToolSpec, ...] = (_SPEC,),
) -> list[Any]:
    request = build_request("làm giúp mình", tools=tools)

    async def collect() -> list[Any]:
        return [ev async for ev in chat_turn(llm, request, tool_runner=runner)]  # type: ignore[arg-type]

    return asyncio.run(collect())


def test_turn_allows_long_observe_act_verify_chains() -> None:
    steps = 8  # more than the old 3-round cap
    calls = [
        [ToolCall(id=f"s{i}", name="ui__accessibility_tree", arguments={"step": i})]
        for i in range(steps)
    ]
    llm = _ScriptedLLM(calls, "Xong rồi nha.")
    executed: list[str] = []

    async def runner(call: ToolCall, emit: Any) -> Message:
        executed.append(call.id)
        return Message(role=Role.TOOL, content='{"status":"ok"}', tool_call_id=call.id)

    events = _run_turn(llm, runner)
    assert executed == [f"s{i}" for i in range(steps)]
    assert any(isinstance(e, TextDelta) and "Xong" in e.text for e in events)
    assert steps <= MAX_TOOL_ROUNDS


def test_turn_rounds_are_capped_and_last_round_is_tool_free() -> None:
    calls = [
        [ToolCall(id=f"r{i}", name="ui__accessibility_tree", arguments={"step": i})]
        for i in range(MAX_TOOL_ROUNDS + 5)
    ]
    llm = _ScriptedLLM(calls, "Mình dừng ở đây.")
    executed: list[str] = []

    async def runner(call: ToolCall, emit: Any) -> Message:
        executed.append(call.id)
        return Message(role=Role.TOOL, content='{"status":"ok"}', tool_call_id=call.id)

    _run_turn(llm, runner)
    assert len(executed) == MAX_TOOL_ROUNDS
    assert llm.requests[-1].tools == ()


def test_identical_failing_call_is_blocked_on_third_attempt() -> None:
    same = {"action": "invoke_ref", "refId": "ui_9"}
    calls = [
        [ToolCall(id=f"f{i}", name="ui__accessibility_tree", arguments=same)] for i in range(4)
    ]
    llm = _ScriptedLLM(calls, "Mình không bấm được nút đó.")
    executed: list[str] = []
    tool_contents: list[str] = []

    async def runner(call: ToolCall, emit: Any) -> Message:
        executed.append(call.id)
        return Message(
            role=Role.TOOL,
            content='{"status":"failed","output":{"errorCode":"stale_ref"}}',
            tool_call_id=call.id,
        )

    _run_turn(llm, runner)
    assert executed == ["f0", "f1"]
    for request in llm.requests[3:]:
        tool_contents.extend(m.content for m in request.messages if m.role == Role.TOOL)
    assert any("blocked_repeat" in c for c in tool_contents)


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ('{"status":"ok","repeatSafe":false}', ["t0"]),  # non-idempotent: not re-run
        ('{"status":"ok"}', ["t0", "t1"]),  # idempotent / undeclared-safe: allowed
    ],
)
def test_guard_follows_repeat_safe_flag_not_tool_names(result: str, expected: list[str]) -> None:
    spec = ToolSpec(
        name="custom__anything",
        description="Any tool",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )
    calls = [
        [ToolCall(id="t0", name="custom__anything", arguments={})],
        [ToolCall(id="t1", name="custom__anything", arguments={})],
    ]
    llm = _ScriptedLLM(calls, "Xong.")
    executed: list[str] = []

    async def runner(call: ToolCall, emit: Any) -> Message:
        executed.append(call.id)
        return Message(role=Role.TOOL, content=result, tool_call_id=call.id)

    _run_turn(llm, runner, tools=(spec,))
    assert executed == expected


# ---------------------------------------------------------------------------
# Host-declared idempotency and owner-usage ordering (no tool names in the agent loop)
# ---------------------------------------------------------------------------


def _wire_descriptor(name: str, idempotent: object = None) -> dict[str, object]:
    policy: dict[str, object] = {
        "version": 1,
        "family": validate_tool_name(name).value,
        "risk": "R1",
        "observes": False,
        "reversible": True,
        "permission": "granted",
    }
    if idempotent is not None:
        policy["idempotent"] = idempotent
    return {
        "name": name,
        "description": "Tool.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "outputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "_meta": {"xuna/policy": policy},
    }


def test_idempotent_policy_round_trips_and_is_optional() -> None:
    declared = ToolDescriptor.from_wire(_wire_descriptor("media.play_pause", False))
    assert declared.idempotent is False
    assert declared.to_wire()["_meta"]["xuna/policy"]["idempotent"] is False  # type: ignore[index]
    legacy = ToolDescriptor.from_wire(_wire_descriptor("media.play_pause"))
    assert legacy.idempotent is None
    assert "idempotent" not in legacy.to_wire()["_meta"]["xuna/policy"]  # type: ignore[index,operator]
    with pytest.raises(ValueError):
        ToolDescriptor.from_wire(_wire_descriptor("media.play_pause", "no"))


@pytest.mark.parametrize(
    ("name", "idempotent", "repeat_safe"),
    [
        ("media.play_pause", False, False),  # host declares a toggle
        ("media.play_pause", True, True),  # host declaration wins over legacy fallback
        ("media.play_pause", None, False),  # old host, undeclared: legacy fallback
        ("media.next", None, True),
        ("tv.control", False, False),  # any tool can declare itself non-idempotent
    ],
)
def test_repeat_safe_comes_from_host_declaration(
    name: str, idempotent: bool | None, repeat_safe: bool
) -> None:
    descriptor = ToolDescriptor.from_wire(_wire_descriptor(name, idempotent))
    assert is_repeat_safe(descriptor) is repeat_safe


def test_orchestrator_marks_non_idempotent_success() -> None:
    owner = uuid.uuid4()
    host = _host(owner)
    descriptor = ToolDescriptor.from_wire(_wire_descriptor("media.next", False))
    manifest = Manifest(host_id=host.host_id, tools=(descriptor,), updated_at=NOW)
    hub = _Hub(str(host.host_id), _OK_RESPONSE)
    orch = Orchestrator(
        host_repo=_Repo(host, manifest, _grants(host.host_id, {ToolFamily.MEDIA})),  # type: ignore[arg-type]
        hub=hub,  # type: ignore[arg-type]
        approval_repo=InMemoryApprovalRepository(),
        approval_broker=InMemoryApprovalBroker(),
        local_result_broker=InMemoryLocalResultBroker(),
        last_target_store=InMemoryLastTargetStore(),
        audit=_Audit(),
        clock=lambda: NOW,
    )

    async def emit(ev: ActionStreamEvent) -> None:
        return None

    ctx = ActionContext(
        owner_id=owner,
        pet_id="pet-1",
        locale=Locale.VI,
        client_caps=(CAP_HOST_ACTIONS_V1,),
        board_available=False,
        emit_event=emit,
    )
    msg = asyncio.run(orch.handle_call(ToolCall(id="n1", name="media__next", arguments={}), ctx))
    assert json.loads(msg.content)["repeatSafe"] is False


def _invoked(tool: str, status: str = "ok") -> AuditEvent:
    return AuditEvent(
        owner_id=uuid.uuid4(),
        action=AuditAction.HOST_TOOL_INVOKED,
        occurred_at=NOW,
        metadata={"tool": tool, "status": status},
    )


def test_usage_counts_only_successful_host_tool_calls() -> None:
    events = [
        _invoked("tv.control"),
        _invoked("tv.control"),
        _invoked("tv.control", status="failed"),
        _invoked("clipboard.read"),
        AuditEvent(
            owner_id=uuid.uuid4(),
            action=AuditAction.HOST_TOOL_DENIED,
            occurred_at=NOW,
            metadata={"tool": "url.open", "status": "denied"},
        ),
    ]
    assert tool_usage_counts(events) == {"tv.control": 2, "clipboard.read": 1}


def test_owner_usage_promotes_deferred_tool_into_direct_catalog() -> None:
    host, manifest, grants = _catalog_inputs()
    args: dict[str, Any] = {
        "hosts": [host],
        "manifests_by_host": {host.host_id: manifest},
        "grants_by_host": {host.host_id: grants},
        "online_host_ids": {str(host.host_id)},
        "client_caps": (CAP_HOST_ACTIONS_V1,),
        "host_screen_consent": True,
        "now": NOW,
    }
    cold = [s.name for s in build_tool_catalog(**args)]
    assert "display__wake" not in cold  # deferred by default
    warm = [s.name for s in build_tool_catalog(**args, usage={"display.wake": 5})]
    assert warm[0] == "display__wake"  # the owner's most used tool comes first
    assert TOOL_SEARCH_FUNCTION in warm


def test_catalog_loader_survives_audit_failure() -> None:
    host, manifest, grants = _catalog_inputs(_TOOL_NAMES[:3])

    class _BrokenAudit:
        async def recent(self, owner_id: uuid.UUID, limit: int) -> list[AuditEvent]:
            raise RuntimeError("db down")

    specs = asyncio.run(
        load_tool_catalog_for_owner(
            owner_id=host.owner_id,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            host_repo=_Repo(host, manifest, grants),  # type: ignore[arg-type]
            hub=_Hub(str(host.host_id), _OK_RESPONSE),  # type: ignore[arg-type]
            audit=_BrokenAudit(),  # type: ignore[arg-type]
            now=NOW,
        )
    )
    assert [s.name for s in specs] == [n.replace(".", "__") for n in _TOOL_NAMES[:3]]


def test_scripted_llm_events_are_chat_events() -> None:
    # Guards the fake: every yielded item must be a ChatEvent the turn understands.
    llm = _ScriptedLLM([], "hi")

    async def first() -> ChatEvent:
        async for ev in llm.stream(build_request("x")):
            return ev  # type: ignore[no-any-return]
        raise AssertionError("no event")

    assert isinstance(asyncio.run(first()), TextDelta)
