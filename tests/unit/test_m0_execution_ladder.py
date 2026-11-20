"""M0 Smart 4-Tier Execution Ladder unit tests (Features F1, F2, F3)."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from tests.modules.actions.test_tool_discovery import (
    _Audit,
    _catalog_inputs,
    _descriptor,
    _grants,
    _host,
    _Hub,
    _Repo,
)
from tests.unit.test_udc17_preflight import _build as _build_cu_world
from tests.unit.test_udc17_preflight import _descriptor as _cu_descriptor

from xuna_backend.modules.actions.application.catalog import (
    _PRIORITY_ORDER,
    CAP_HOST_ACTIONS_V1,
    MAX_CATALOG_TOOLS,
    build_tool_catalog,
    collect_eligible_host_tools,
)
from xuna_backend.modules.actions.application.computer_use import InMemoryComputerUseSessionStore
from xuna_backend.modules.actions.application.orchestrator import ActionContext, Orchestrator
from xuna_backend.modules.actions.application.prompts import (
    SYSTEM_PROMPT_EN,
    SYSTEM_PROMPT_VI,
    tool_system_prompt,
    truncate_tool_value,
)
from xuna_backend.modules.actions.application.tool_discovery import TOOL_SEARCH_FUNCTION
from xuna_backend.modules.actions.domain.entities import ActionStreamEvent
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.repository import InMemoryApprovalRepository
from xuna_backend.modules.cognition.domain.types import Locale, ToolCall, ToolSpec
from xuna_backend.modules.hosts.domain.entities import (
    Manifest,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
    validate_tool_name,
)

NOW = datetime(2026, 11, 19, 10, 0, 0, tzinfo=UTC)


def test_f1_catalog_36_slots_expose_all_tier_0_1_2_3_direct_tools_at_worst_case_reserved_3() -> (
    None
):
    """With 75+ tools and reserved=3 (board + workflow_template + autonomous_job),
    the 30 guaranteed direct slots hold window.active_context, Tier 1, primary Tier 0,
    and core Tier 2/3 CU/UI/Input tools while deferring overflow behind tool_search."""
    unique_priority = tuple(dict.fromkeys(_PRIORITY_ORDER))
    assert len(unique_priority) >= 75

    host = _host(uuid.uuid4())
    descriptors: list[ToolDescriptor] = []
    for name in unique_priority:
        base = _descriptor(name)
        observes = name in ("window.active_context", "notes.search", "calendar.list_events")
        descriptors.append(
            ToolDescriptor(
                name=base.name,
                description=base.description,
                input_schema=base.input_schema,
                output_schema=base.output_schema,
                family=base.family,
                risk=base.risk,
                observes=observes,
            )
        )
    manifest = Manifest(host_id=host.host_id, tools=tuple(descriptors), updated_at=NOW)
    families = {d.family for d in descriptors}
    grants = _grants(host.host_id, families)

    specs = build_tool_catalog(
        hosts=[host],
        manifests_by_host={host.host_id: manifest},
        grants_by_host={host.host_id: grants},
        online_host_ids={str(host.host_id)},
        client_caps=(CAP_HOST_ACTIONS_V1, "board_hid_v1", "workflow_templates_v1"),
        board_available=True,
        host_screen_consent=True,
        include_workflow_template=True,
        include_autonomous_job=True,
        now=NOW,
    )
    names = [s.name for s in specs]
    assert len(specs) == MAX_CATALOG_TOOLS == 36
    assert TOOL_SEARCH_FUNCTION in names

    expected_direct = {n.replace(".", "__") for n in _PRIORITY_ORDER[:30]}
    assert expected_direct <= set(names)
    assert names[0] == "window__active_context"

    # Fail-closed U5 (host_screen_consent=False) filters CU/UI/INPUT and observes=True tools
    no_u5_specs = build_tool_catalog(
        hosts=[host],
        manifests_by_host={host.host_id: manifest},
        grants_by_host={host.host_id: grants},
        online_host_ids={str(host.host_id)},
        client_caps=(CAP_HOST_ACTIONS_V1,),
        host_screen_consent=False,
        now=NOW,
    )
    no_u5_names = {s.name for s in no_u5_specs}
    assert "window__active_context" not in no_u5_names
    assert "notes__search" not in no_u5_names
    assert "calendar__list_events" not in no_u5_names
    assert "cu__session_start" not in no_u5_names
    assert "ui__inspect" not in no_u5_names
    assert "input__click" not in no_u5_names
    assert "app__launch" in no_u5_names


def test_f2_ui_accessibility_tree_suppressed_when_ui_inspect_present_and_legacy_compat_kept() -> (
    None
):
    """When modern ui.inspect is advertised, ui.accessibility_tree is suppressed;
    when only legacy ui.accessibility_tree exists, validate_tool_name & collection still work."""
    assert (
        validate_tool_name("ui.accessibility_tree", expected_family=ToolFamily.APP)
        == ToolFamily.APP
    )
    assert "ui.accessibility_tree" not in _PRIORITY_ORDER

    host, manifest, grants = _catalog_inputs(("app.launch", "ui.accessibility_tree", "ui.inspect"))
    eligible_modern = collect_eligible_host_tools(
        hosts=[host],
        manifests_by_host={host.host_id: manifest},
        grants_by_host={host.host_id: grants},
        online_host_ids={str(host.host_id)},
        host_screen_consent=True,
        now=NOW,
    )
    assert "ui.inspect" in eligible_modern
    assert "ui.accessibility_tree" not in eligible_modern

    host_old, manifest_old, grants_old = _catalog_inputs(("app.launch", "ui.accessibility_tree"))
    eligible_legacy = collect_eligible_host_tools(
        hosts=[host_old],
        manifests_by_host={host_old.host_id: manifest_old},
        grants_by_host={host_old.host_id: grants_old},
        online_host_ids={str(host_old.host_id)},
        host_screen_consent=True,
        now=NOW,
    )
    assert "ui.accessibility_tree" in eligible_legacy


def test_f3_prompts_4_tier_execution_ladder_and_rule_h5_compliance() -> None:
    """Prompts teach the 4-tier ladder & SoM [1]..[N], drop unconditional app__launch,
    never leak Observe -> Act -> Verify into non-UI prompts or tool_search into non-deferred,
    and respect Rule H5."""
    for locale in (Locale.VI, Locale.EN):
        full_prompt = SYSTEM_PROMPT_EN if locale == Locale.EN else SYSTEM_PROMPT_VI
        assert "ngay sau đó hãy gọi `app__launch`" not in full_prompt
        assert "immediately call `app__launch`" not in full_prompt
        assert "targetAlreadyFocused" in full_prompt
        assert "[1]..[N]" in full_prompt
        assert "0.2, 0.4, 0.6, 0.8" in full_prompt
        for forbidden in (
            "Spotify",
            "Chrome",
            "Safari",
            "Notes",
            "Reminders",
            "Finder",
            "Terminal",
        ):
            assert forbidden not in full_prompt

    # Host non-board tool triggers 4-tier ladder without Observe -> Act -> Verify or tool_search
    vi_media = tool_system_prompt(
        Locale.VI,
        tools=(ToolSpec(name="media__play_pause", description="Media", parameters={}),),
    )
    assert "Tầng 0" in vi_media
    assert "Tầng 1" in vi_media
    assert "Tầng 2" in vi_media
    assert "Tầng 3" in vi_media
    assert "Observe -> Act -> Verify" not in vi_media
    assert "tool_search" not in vi_media

    en_media = tool_system_prompt(
        Locale.EN,
        tools=(ToolSpec(name="media__play_pause", description="Media", parameters={}),),
    )
    assert "Tier 0" in en_media
    assert "Tier 3" in en_media
    assert "Observe -> Act -> Verify" not in en_media
    assert "tool_search" not in en_media

    # Board-only tool should NOT inject the desktop 4-tier ladder
    board_only = tool_system_prompt(
        Locale.VI,
        tools=(ToolSpec(name="board__media_key", description="Board", parameters={}),),
    )
    assert "Tầng 0" not in board_only

    # truncate_tool_value allows up to 32 runningApps entries
    running = [{"name": f"App{i}", "bundleId": f"com.test.app{i}"} for i in range(40)]
    truncated = truncate_tool_value({"runningApps": running})
    assert isinstance(truncated, dict)
    assert isinstance(truncated["runningApps"], list)
    assert len(truncated["runningApps"]) == 32


def test_f3_orchestrator_cu_session_start_continue_in_place_caches_and_redacts_nodes() -> None:
    """When cu.session_start returns targetAlreadyFocused=True + nodes:
    1) Password nodes are redacted in output and cached_nodes.
    2) new_state.cached_nodes is populated BEFORE cu_session_store.start().
    3) Hint explicitly forbids app__launch and directs acting on returned snapshot/nodes.
    4) Unfocused / missing nodes / failed calls emit matching fail-closed hints."""
    cu_start_desc = _cu_descriptor(
        "cu.session_start",
        RiskTier.R3,
        {
            "goal": {"type": "string"},
            "apps": {"type": "array", "items": {"type": "string"}},
            "domains": {"type": "array", "items": {"type": "string"}},
            "maxSteps": {"type": "integer"},
            "maxMinutes": {"type": "integer"},
        },
        ToolFamily.CU,
    )
    leaky_password_node = {
        "id": "ui_2",
        "ref": "@s00000001:e2",
        "role": "AXSecureTextField",
        "name": "Secret PIN",
        "value": "999999",
    }
    btn_node = {
        "id": "ui_1",
        "ref": "@s00000001:e1",
        "role": "AXButton",
        "name": "Continue",
        "bounds": [0.1, 0.2, 0.3, 0.25],
    }

    async def run_with_world() -> None:
        responses: dict[str, dict[str, Any]] = {
            "focused_with_nodes": {
                "sessionId": "cu-m0-focused",
                "foregroundApp": "explorer",
                "windowTitle": "Workspace",
                "targetAlreadyFocused": True,
                "snapshotId": "s00000001",
                "nodes": [dict(btn_node), dict(leaky_password_node)],
            },
            "focused_no_nodes": {
                "sessionId": "cu-m0-nonodes",
                "foregroundApp": "explorer",
                "windowTitle": "Workspace",
                "targetAlreadyFocused": True,
            },
            "unfocused": {
                "sessionId": "cu-m0-unfocused",
                "foregroundApp": "other",
                "windowTitle": "Other App",
                "targetAlreadyFocused": False,
            },
        }
        mode = ["focused_with_nodes"]

        def host_fn(name: str, args: dict[str, Any]) -> dict[str, Any] | None:
            if name == "cu.session_start":
                return dict(responses[mode[0]])
            return None

        world = await _build_cu_world([], extra_tools=(cu_start_desc,), structured_for=host_fn)

        # Case 1: targetAlreadyFocused=True with nodes
        await world.cu_store.stop_all_for_owner(world.owner_id, "test_reset", NOW)
        body1 = await world.call(
            "cu__session_start",
            {"goal": "Click continue", "apps": ["explorer"]},
        )
        assert body1["status"] == "ok"
        assert "Do NOT call app__launch" in body1["hint"]
        assert "snapshotId" in body1["hint"]
        nodes_out = body1["output"]["nodes"]
        assert nodes_out[0]["name"] == "Continue"
        assert nodes_out[1]["name"] == "•••"
        assert "value" not in nodes_out[1]
        assert "999999" not in json.dumps(body1)

        state1 = await world.cu_store.get("cu-m0-focused")
        assert state1 is not None
        assert "@s00000001:e1" in state1.cached_nodes
        assert state1.cached_nodes["@s00000001:e1"]["name"] == "Continue"
        assert state1.cached_nodes["@s00000001:e2"]["role"] == "AXSecureTextField"
        assert "s00000001:e1" in state1.cached_nodes
        assert "s00000001:e2" in state1.cached_nodes

        # Case 2: targetAlreadyFocused=True without nodes
        await world.cu_store.stop_all_for_owner(world.owner_id, "test_reset", NOW)
        mode[0] = "focused_no_nodes"
        body2 = await world.call(
            "cu__session_start",
            {"goal": "Inspect focused window", "apps": ["explorer"]},
        )
        assert body2["status"] == "ok"
        assert "Do NOT call app__launch" in body2["hint"]
        assert "ui__inspect" in body2["hint"]

        # Case 3: targetAlreadyFocused=False (different foreground app)
        await world.cu_store.stop_all_for_owner(world.owner_id, "test_reset", NOW)
        mode[0] = "unfocused"
        body3 = await world.call(
            "cu__session_start",
            {"goal": "Open explorer", "apps": ["explorer"]},
        )
        assert body3["status"] == "ok"
        assert "Do NOT call app__launch" not in body3["hint"]
        assert "app__launch" in body3["hint"]

    asyncio.run(run_with_world())

    # Case 4: Failed cu.session_start does not create a session or emit focused hint
    async def run_failed() -> None:
        host = _host(uuid.uuid4())
        desc = _descriptor("cu.session_start", risk=RiskTier.R1)
        manifest = Manifest(host_id=host.host_id, tools=(desc,), updated_at=NOW)
        grants = _grants(host.host_id, {ToolFamily.CU})
        cu_store = InMemoryComputerUseSessionStore()
        hub = _Hub(
            str(host.host_id),
            {
                "structuredContent": {"targetAlreadyFocused": True},
                "_meta": {"xuna/status": "failed", "xuna/errorCode": "permission_denied"},
            },
        )
        orch = Orchestrator(
            host_repo=_Repo(host, manifest, grants),  # type: ignore[arg-type]
            hub=hub,  # type: ignore[arg-type]
            approval_repo=InMemoryApprovalRepository(),
            approval_broker=InMemoryApprovalBroker(),
            local_result_broker=InMemoryLocalResultBroker(),
            last_target_store=InMemoryLastTargetStore(),
            audit=_Audit(),
            cu_session_store=cu_store,
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
        msg = await orch.handle_call(
            ToolCall(id="cf1", name="cu__session_start", arguments={}), ctx
        )
        payload = json.loads(msg.content)
        assert payload["status"] == "failed"
        assert "Do NOT call app__launch" not in payload.get("hint", "")
        assert await cu_store.get_active_for_owner(host.owner_id) is None

    asyncio.run(run_failed())
