"""Application tests for `MeshPreRouter`, `PendingIntentStore` & `format_host_roster`."""

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from tests.helpers import MemoryAuditLog

from xuna_backend.modules.actions.application.catalog import CAP_HOST_ACTIONS_V1
from xuna_backend.modules.actions.application.computer_use import (
    InMemoryComputerUseSessionStore,
)
from xuna_backend.modules.actions.application.mesh_router import (
    PENDING_INTENT_TTL,
    MeshPreRouter,
    PendingIntent,
    PendingIntentStore,
    format_host_roster,
    record_execution_habit,
    resolve_host_target_for_call,
)
from xuna_backend.modules.actions.application.orchestrator import ActionContext, Orchestrator
from xuna_backend.modules.actions.domain.personalization import InMemoryHostDefaultStore
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.habit_repository import InMemoryHabitRepository
from xuna_backend.modules.actions.infrastructure.repository import InMemoryApprovalRepository
from xuna_backend.modules.cognition.domain.types import Locale, ToolCall
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    Host,
    HostAlias,
    HostId,
    HostKind,
    HostPlatform,
    HostStatus,
    Manifest,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
)
from xuna_backend.modules.hosts.domain.twin import HostPresence, HostResources
from xuna_backend.modules.hosts.infrastructure.session_hub import InMemoryHostSessionHub

NOW = datetime(2026, 11, 20, 19, 0, 0, tzinfo=UTC)
PUB = b"\x04" + b"\x22" * 64


def _host(
    owner_id: uuid.UUID,
    alias: str,
    *,
    platform: HostPlatform = HostPlatform.MACOS,
    kind: HostKind = HostKind.DESKTOP,
    place: str | None = None,
    nicknames: tuple[str, ...] = (),
    wol_mac: str | None = None,
) -> Host:
    return Host(
        host_id=HostId.generate(),
        owner_id=owner_id,
        alias=HostAlias(alias),
        platform=platform,
        hostname=alias.lower().replace(" ", "-"),
        os_version="15.0",
        agent_version="0.2.0",
        public_key=PUB,
        status=HostStatus.ACTIVE,
        created_at=NOW,
        host_kind=kind,
        place=place,
        nicknames=nicknames,
        wol_mac=wol_mac,
    )


async def _open_online(hub: InMemoryHostSessionHub, owner_id: uuid.UUID, host: Host) -> None:
    async def _send(_txt: str) -> None:
        return None

    async def _close() -> None:
        return None

    await hub.open_session(
        str(host.host_id),
        object(),
        _send,
        _close,
        _close,
        owner_id=owner_id,
    )


def test_pre_router_target_host_id_slash_commands_and_at_tags() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        hub = InMemoryHostSessionHub()
        pending = PendingIntentStore()
        habits = InMemoryHabitRepository()
        router = MeshPreRouter(pending_store=pending, habit_repo=habits, clock=lambda: NOW)

        mac = _host(owner, "AlphaMac", place="office", nicknames=("Desk Mac",))
        win = _host(
            owner,
            "BetaWin",
            platform=HostPlatform.WINDOWS,
            place="office",
            nicknames=("PC Nha",),
            wol_mac="AA:BB:CC:DD:EE:01",
        )
        srv = _host(
            owner,
            "HomeServer",
            platform=HostPlatform.LINUX,
            kind=HostKind.SERVER,
            place="home",
        )
        await _open_online(hub, owner, mac)
        # win is offline; srv is online
        await _open_online(hub, owner, srv)

        hosts = [mac, win, srv]

        # 1. Foreign targetHostId -> not_found
        r_404 = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="hello",
            hosts=hosts,
            hub=hub,
            target_host_id=HostId.generate(),
            now=NOW,
        )
        assert r_404.kind == "not_found"

        # 2. Offline targetHostId -> host_offline with wolAvailable, never fallback to sibling
        r_off = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="pause",
            hosts=hosts,
            hub=hub,
            target_host_id=win.host_id,
            now=NOW,
        )
        assert r_off.kind == "target_required"
        assert r_off.reason_code == "host_offline"
        assert r_off.candidates[0].host == win
        assert r_off.candidates[0].wol_available is True

        # 3. /may AlphaMac -> pins focus & emits HostFocusChangedNotice
        r_may = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="/may AlphaMac",
            hosts=hosts,
            hub=hub,
            now=NOW,
        )
        assert r_may.kind == "slash_handled"
        assert r_may.pinned_host_id == mac.host_id
        assert r_may.focus_notice is not None and r_may.focus_notice.source == "slash_cmd"

        # Priority 5 picks up active focus pin
        r_pin = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="check status",
            hosts=hosts,
            hub=hub,
            now=NOW,
        )
        assert r_pin.kind == "pin"
        assert r_pin.pinned_host_id == mac.host_id

        # 4. /follow on & /may auto
        r_fol = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="/follow on",
            hosts=hosts,
            hub=hub,
            now=NOW,
        )
        assert r_fol.kind == "slash_handled"
        assert r_fol.focus_notice is not None and r_fol.focus_notice.follow_me is True

        r_auto = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="/may auto",
            hosts=hosts,
            hub=hub,
            now=NOW,
        )
        assert r_auto.kind == "slash_handled"
        assert r_auto.focus_notice is not None and r_auto.focus_notice.host_id is None

        # 5. Inline @"Desk Mac", trailing @AlphaMac,, ambiguous @office, email ignored
        r_nick = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text='open @"Desk Mac"',
            hosts=hosts,
            hub=hub,
            now=NOW,
        )
        assert r_nick.kind == "pin"
        assert r_nick.pinned_host_id == mac.host_id

        r_punct = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="pause @AlphaMac,",
            hosts=hosts,
            hub=hub,
            now=NOW,
        )
        assert r_punct.kind == "pin"
        assert r_punct.pinned_host_id == mac.host_id

        r_place = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="@office mute",
            hosts=hosts,
            hub=hub,
            now=NOW,
        )
        assert r_place.kind == "target_required"
        assert r_place.reason_code == "ask_target"
        assert {c.host for c in r_place.candidates} == {mac, win}

        r_email = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="send mail to user@domain.com",
            hosts=hosts,
            hub=hub,
            now=NOW,
        )
        assert r_email.kind == "pass"

    asyncio.run(_run())


def test_pending_intent_1_word_repair_ttl_and_habit_updates() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        hub = InMemoryHostSessionHub()
        pending = PendingIntentStore()
        habits = InMemoryHabitRepository()
        router = MeshPreRouter(pending_store=pending, habit_repo=habits, clock=lambda: NOW)

        mac = _host(owner, "NodeA")
        win = _host(owner, "NodeB", platform=HostPlatform.WINDOWS)
        await _open_online(hub, owner, mac)
        await _open_online(hub, owner, win)
        hosts = [mac, win]

        # Seed PendingIntent where NodeA was rank-1 suggested host
        pending.save(
            PendingIntent(
                owner_id=owner,
                pet_id="pet-1",
                original_tool="media.play_pause",
                original_arguments={},
                original_text="pause",
                candidate_host_ids=(mac.host_id, win.host_id),
                created_at=NOW,
                suggested_host_id=mac.host_id,
                resource_id=None,
            )
        )

        # Repair within TTL picking NodeB -> positive on NodeB, negative redirect on NodeA
        rep = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="NodeB",
            hosts=hosts,
            hub=hub,
            now=NOW + timedelta(seconds=30),
        )
        assert rep.kind == "repair_dispatch"
        assert rep.pinned_host_id == win.host_id
        assert rep.replay_tool == "media.play_pause"
        assert pending.get(owner, "pet-1", NOW + timedelta(seconds=31)) is None

        all_habits = await habits.list_habits(owner, pet_id="pet-1")
        by_host = {h.host_id: h for h in all_habits}
        assert by_host[win.host_id].positive_score == 1.0
        assert by_host[win.host_id].last_reason_code == "confirmed_pick"
        assert by_host[mac.host_id].negative_score == 1.0
        assert by_host[mac.host_id].last_reason_code == "redirect_penalty"

        # Repeating "NodeB" after consumed does NOT replay
        rep2 = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="NodeB",
            hosts=hosts,
            hub=hub,
            now=NOW + timedelta(seconds=35),
        )
        assert rep2.kind == "pass"

        # TTL expired (> 120s) and > 4 tokens both ignore & clear PendingIntent
        pending.save(
            PendingIntent(
                owner_id=owner,
                pet_id="pet-1",
                original_tool="media.play_pause",
                original_arguments={},
                original_text="pause",
                candidate_host_ids=(mac.host_id, win.host_id),
                created_at=NOW,
            )
        )
        assert pending.get(owner, "pet-1", NOW + PENDING_INTENT_TTL + timedelta(seconds=1)) is None

        pending.save(
            PendingIntent(
                owner_id=owner,
                pet_id="pet-1",
                original_tool="media.play_pause",
                original_arguments={},
                original_text="pause",
                candidate_host_ids=(mac.host_id, win.host_id),
                created_at=NOW,
            )
        )
        r_long = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="never mind just tell me a story",
            hosts=hosts,
            hub=hub,
            now=NOW + timedelta(seconds=10),
        )
        assert r_long.kind == "pass"
        assert pending.get(owner, "pet-1", NOW + timedelta(seconds=11)) is None

    asyncio.run(_run())


def test_format_host_roster_live_twin_and_no_stale_offline_twin() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        hub = InMemoryHostSessionHub()
        mac = _host(owner, "MacHost", place="office", nicknames=("m1",))
        srv = _host(owner, "RackNode", platform=HostPlatform.LINUX, kind=HostKind.SERVER)
        await _open_online(hub, owner, mac)
        await _open_online(hub, owner, srv)

        hub.update_presence(
            str(mac.host_id),
            HostPresence(
                idle_seconds=12,
                screen_locked=False,
                media_playing=True,
                battery_percent=78,
                is_charging=True,
                updated_at=NOW,
            ),
        )
        hub.update_resources(
            str(mac.host_id),
            HostResources(installed_apps=("app_a",), workspaces=("ws_1",), updated_at=NOW),
        )
        # RackNode has no battery
        hub.update_presence(
            str(srv.host_id),
            HostPresence(
                idle_seconds=0,
                screen_locked=False,
                media_playing=False,
                battery_percent=None,
                updated_at=NOW,
            ),
        )
        # Disconnect MacHost -> twin evicted immediately (H2)
        await hub.disconnect_host(str(mac.host_id), reason="offline")

        roster = format_host_roster(
            hosts=[mac, srv],
            hub=hub,
            owner_id=owner,
            pinned_host_id=srv.host_id,
        )
        assert "[HOST_ROSTER]" in roster
        assert "focusMode: pinned(RackNode)" in roster
        assert "MacHost" in roster and "state=offline" in roster
        assert "battery=unknown, appsCount=0, workspacesCount=0" in roster
        assert "RackNode (platform=linux, kind=server" in roster
        assert "100%" not in roster

    asyncio.run(_run())


def test_resolve_host_target_for_call_and_record_execution_habit() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        hub = InMemoryHostSessionHub()
        habits = InMemoryHabitRepository()
        pending = PendingIntentStore()
        mac = _host(owner, "Mac")
        await _open_online(hub, owner, mac)
        desc = ToolDescriptor(
            name="app.launch",
            description="Launch app",
            input_schema={
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
                "additionalProperties": False,
            },
            output_schema={"type": "object", "properties": {}, "additionalProperties": False},
            family=ToolFamily.APP,
            risk=RiskTier.R2,
        )
        manifest = Manifest(host_id=mac.host_id, tools=(desc,), updated_at=NOW)
        grants = (Grant(mac.host_id, ToolFamily.APP, RiskTier.R2, True, NOW),)

        async def _sink(_ev: object) -> None:
            return None

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.VI,
            client_caps=("host_actions_v1",),
            board_available=False,
            emit_event=_sink,
            pinned_host_id=mac.host_id,
        )
        res = await resolve_host_target_for_call(
            tool_name="app.launch",
            clean_args={"id": "audio_player_x"},
            requested_target=None,
            ctx=ctx,
            active_hosts=[mac],
            manifests_by_host={mac.host_id: manifest},
            grants_by_host={mac.host_id: grants},
            online_hosts_with_tool=[mac],
            reference_descriptor=desc,
            hub=hub,
            last_target_store=InMemoryLastTargetStore(),
            host_default_store=InMemoryHostDefaultStore(),
            habit_repo=habits,
            pending_store=pending,
            cu_session_store=InMemoryComputerUseSessionStore(),
            now=NOW,
        )
        assert res.resolved is True
        assert res.host == mac

        for step in range(3):
            await record_execution_habit(
                habits,
                ctx=ctx,
                descriptor=desc,
                clean_args={"id": "audio_player_x"},
                host_id=mac.host_id,
                now=NOW + timedelta(minutes=step),
            )
        rows = await habits.list_habits(owner, pet_id="pet-1")
        assert len(rows) == 2  # specific + '-'
        for r in rows:
            assert r.positive_score == 1.0 + 0.85 + 0.85**2
            assert r.sample_count == 3

    asyncio.run(_run())


@pytest.mark.parametrize(
    ("tool_name", "expected_family", "args"),
    [
        ("workspace.open", ToolFamily.APP, {"project": "apollo"}),
        ("clipboard.write", ToolFamily.APP, {}),
        ("browser.current_url", ToolFamily.URL, {}),
        ("finder.reveal", ToolFamily.FILES, {}),
        ("system.vitals", ToolFamily.HOST, {}),
    ],
)
def test_pending_intent_and_redirect_write_canonical_tool_family(
    tool_name: str,
    expected_family: ToolFamily,
    args: dict[str, object],
) -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        hub = InMemoryHostSessionHub()
        pending = PendingIntentStore()
        habits = InMemoryHabitRepository()
        router = MeshPreRouter(pending_store=pending, habit_repo=habits, clock=lambda: NOW)

        mac = _host(owner, "MacNode")
        win = _host(owner, "WinNode", platform=HostPlatform.WINDOWS)
        await _open_online(hub, owner, mac)
        await _open_online(hub, owner, win)

        desc = ToolDescriptor(
            name=tool_name,
            description=f"Tool {tool_name}",
            input_schema={
                "type": "object",
                "properties": {"project": {"type": "string"}},
                "additionalProperties": False,
            },
            output_schema={"type": "object", "properties": {}, "additionalProperties": False},
            family=expected_family,
            risk=RiskTier.R1,
        )
        manifests = {
            mac.host_id: Manifest(host_id=mac.host_id, tools=(desc,), updated_at=NOW),
            win.host_id: Manifest(host_id=win.host_id, tools=(desc,), updated_at=NOW),
        }
        grants = {
            mac.host_id: (Grant(mac.host_id, expected_family, RiskTier.R1, True, NOW),),
            win.host_id: (Grant(win.host_id, expected_family, RiskTier.R1, True, NOW),),
        }

        async def _sink(_ev: object) -> None:
            return None

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.VI,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_sink,
        )

        # Trigger ask_target to store PendingIntent with canonical family
        first = await resolve_host_target_for_call(
            tool_name=tool_name,
            clean_args=args,
            requested_target=None,
            ctx=ctx,
            active_hosts=[mac, win],
            manifests_by_host=manifests,
            grants_by_host=grants,
            online_hosts_with_tool=[mac, win],
            reference_descriptor=desc,
            hub=hub,
            last_target_store=InMemoryLastTargetStore(),
            host_default_store=InMemoryHostDefaultStore(),
            habit_repo=habits,
            pending_store=pending,
            cu_session_store=InMemoryComputerUseSessionStore(),
            now=NOW,
        )
        assert first.resolved is False
        assert first.error_status == "ask_target"
        saved_pending = pending.get(owner, "pet-1", NOW)
        assert saved_pending is not None
        assert saved_pending.family == expected_family.value

        # Repair to the non-suggested host -> writes canonical family + redirect penalty
        chosen_id = win.host_id if saved_pending.suggested_host_id == mac.host_id else mac.host_id
        chosen_alias = "WinNode" if chosen_id == win.host_id else "MacNode"
        rep = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text=chosen_alias,
            hosts=[mac, win],
            hub=hub,
            now=NOW + timedelta(seconds=5),
        )
        assert rep.kind == "repair_dispatch"
        assert rep.habit_recorded_in_preroute is True

        rows = await habits.list_habits(owner, pet_id="pet-1")
        assert rows
        assert all(r.context_key.startswith(f"{expected_family.value}:") for r in rows)

        # Subsequent resolve_host_target_for_call observes m_hab > 0.5 for chosen host
        after = await resolve_host_target_for_call(
            tool_name=tool_name,
            clean_args=args,
            requested_target=None,
            ctx=ctx,
            active_hosts=[mac, win],
            manifests_by_host=manifests,
            grants_by_host=grants,
            online_hosts_with_tool=[mac, win],
            reference_descriptor=desc,
            hub=hub,
            last_target_store=InMemoryLastTargetStore(),
            host_default_store=InMemoryHostDefaultStore(),
            habit_repo=habits,
            pending_store=PendingIntentStore(),
            cu_session_store=InMemoryComputerUseSessionStore(),
            now=NOW + timedelta(seconds=10),
        )
        assert after.scoring_decision is not None
        by_id = {c.host.host_id: c for c in after.scoring_decision.candidates}
        assert by_id[chosen_id].m_hab > 0.5

        # Also test Priority 1 explicit targetHostId redirect penalty without pending.family set
        p1_habits = InMemoryHabitRepository()
        p1_router = MeshPreRouter(pending_store=pending, habit_repo=p1_habits, clock=lambda: NOW)
        pending.save(
            PendingIntent(
                owner_id=owner,
                pet_id="pet-1",
                original_tool=tool_name,
                original_arguments=dict(args),
                original_text="do it",
                candidate_host_ids=(mac.host_id, win.host_id),
                created_at=NOW + timedelta(seconds=20),
                suggested_host_id=mac.host_id,
                resource_id="apollo" if "project" in args else None,
            )
        )
        pin_out = await p1_router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="on win",
            hosts=[mac, win],
            hub=hub,
            target_host_id=win.host_id,
            now=NOW + timedelta(seconds=25),
        )
        assert pin_out.kind == "pin"
        assert pending.get(owner, "pet-1", NOW + timedelta(seconds=26)) is None
        rows_after_p1 = await p1_habits.list_habits(owner, pet_id="pet-1")
        assert rows_after_p1
        assert all(
            r.context_key.startswith(f"{expected_family.value}:")
            and r.host_id == mac.host_id
            and r.negative_score == 1.0
            and r.last_reason_code == "redirect_penalty"
            for r in rows_after_p1
        )

    asyncio.run(_run())


def test_target_resolver_pronoun_fallback_and_single_host_unrecognized_target() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        hub = InMemoryHostSessionHub()
        mac = _host(owner, "Mac")
        win = _host(owner, "Win", platform=HostPlatform.WINDOWS)
        await _open_online(hub, owner, mac)
        await _open_online(hub, owner, win)
        desc = ToolDescriptor(
            name="media.play_pause",
            description="Toggle playback",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            output_schema={"type": "object", "properties": {}, "additionalProperties": False},
            family=ToolFamily.MEDIA,
            risk=RiskTier.R1,
        )
        manifests = {
            mac.host_id: Manifest(host_id=mac.host_id, tools=(desc,), updated_at=NOW),
            win.host_id: Manifest(host_id=win.host_id, tools=(desc,), updated_at=NOW),
        }
        grants = {
            mac.host_id: (Grant(mac.host_id, ToolFamily.MEDIA, RiskTier.R1, True, NOW),),
            win.host_id: (Grant(win.host_id, ToolFamily.MEDIA, RiskTier.R1, True, NOW),),
        }
        last_store = InMemoryLastTargetStore()
        last_store.set_last_target(owner, "pet-1", "Mac", NOW - timedelta(seconds=20))

        async def _sink(_ev: object) -> None:
            return None

        ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.VI,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_sink,
        )

        # "it" -> TargetResolver resolves to last_target ("Mac")
        res_it = await resolve_host_target_for_call(
            tool_name="media.play_pause",
            clean_args={},
            requested_target="it",
            ctx=ctx,
            active_hosts=[mac, win],
            manifests_by_host=manifests,
            grants_by_host=grants,
            online_hosts_with_tool=[mac, win],
            reference_descriptor=desc,
            hub=hub,
            last_target_store=last_store,
            host_default_store=InMemoryHostDefaultStore(),
            habit_repo=None,
            pending_store=PendingIntentStore(),
            cu_session_store=InMemoryComputerUseSessionStore(),
            now=NOW,
        )
        assert res_it.resolved is True
        assert res_it.host == mac

        # "máy kia" -> TargetResolver switches to "Win"
        res_other = await resolve_host_target_for_call(
            tool_name="media.play_pause",
            clean_args={},
            requested_target="máy kia",
            ctx=ctx,
            active_hosts=[mac, win],
            manifests_by_host=manifests,
            grants_by_host=grants,
            online_hosts_with_tool=[mac, win],
            reference_descriptor=desc,
            hub=hub,
            last_target_store=last_store,
            host_default_store=InMemoryHostDefaultStore(),
            habit_repo=None,
            pending_store=PendingIntentStore(),
            cu_session_store=InMemoryComputerUseSessionStore(),
            now=NOW,
        )
        assert res_other.resolved is True
        assert res_other.host == win

        # Expired last target + pronoun -> ASK_TARGET with full error_extra fields
        res_expired = await resolve_host_target_for_call(
            tool_name="media.play_pause",
            clean_args={},
            requested_target="it",
            ctx=ctx,
            active_hosts=[mac, win],
            manifests_by_host=manifests,
            grants_by_host=grants,
            online_hosts_with_tool=[mac, win],
            reference_descriptor=desc,
            hub=hub,
            last_target_store=InMemoryLastTargetStore(),
            host_default_store=InMemoryHostDefaultStore(),
            habit_repo=None,
            pending_store=PendingIntentStore(),
            cu_session_store=InMemoryComputerUseSessionStore(),
            now=NOW,
        )
        assert res_expired.resolved is False
        assert res_expired.error_status == "ask_target"
        assert res_expired.error_extra is not None
        assert res_expired.error_extra["reasonCode"] == "ask_target"
        assert "targetReason" in res_expired.error_extra
        assert len(res_expired.error_extra["candidates"]) == 2  # type: ignore[arg-type]

        # Single active host with unrecognized target must NOT say "Multiple hosts match..."
        res_single_unknown = await resolve_host_target_for_call(
            tool_name="media.play_pause",
            clean_args={},
            requested_target="UnknownBox",
            ctx=ctx,
            active_hosts=[mac],
            manifests_by_host={mac.host_id: manifests[mac.host_id]},
            grants_by_host={mac.host_id: grants[mac.host_id]},
            online_hosts_with_tool=[mac],
            reference_descriptor=desc,
            hub=hub,
            last_target_store=InMemoryLastTargetStore(),
            host_default_store=InMemoryHostDefaultStore(),
            habit_repo=None,
            pending_store=PendingIntentStore(),
            cu_session_store=InMemoryComputerUseSessionStore(),
            now=NOW,
        )
        assert res_single_unknown.resolved is False
        assert res_single_unknown.error_status == "ask_target"
        assert res_single_unknown.error_extra is not None
        assert "Multiple hosts match" not in str(res_single_unknown.error_extra["targetReason"])

    asyncio.run(_run())


def test_at_tag_preserves_dots_and_apostrophes_and_invalid_target_never_mutates_follow_me() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        hub = InMemoryHostSessionHub()
        router = MeshPreRouter(pending_store=PendingIntentStore(), clock=lambda: NOW)
        macbook = _host(owner, "MacBook", nicknames=("mac",))
        mac_mini = _host(owner, "Mac.Mini")
        oneil = _host(owner, "O'Neil")
        for h in (macbook, mac_mini, oneil):
            await _open_online(hub, owner, h)

        hosts = [macbook, mac_mini, oneil]

        r_dot = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="status @Mac.Mini",
            hosts=hosts,
            hub=hub,
            now=NOW,
        )
        assert r_dot.kind == "pin"
        assert r_dot.pinned_host_id == mac_mini.host_id

        r_dot_punct = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="status @Mac.Mini.",
            hosts=hosts,
            hub=hub,
            now=NOW,
        )
        assert r_dot_punct.kind == "pin"
        assert r_dot_punct.pinned_host_id == mac_mini.host_id

        r_apos = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="check @O'Neil,",
            hosts=hosts,
            hub=hub,
            now=NOW,
        )
        assert r_apos.kind == "pin"
        assert r_apos.pinned_host_id == oneil.host_id
        assert r_apos.cleaned_text == "check"

        r_unknown = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="status @nonexistent",
            hosts=hosts,
            hub=hub,
            now=NOW,
        )
        assert r_unknown.kind == "target_required"
        assert r_unknown.target_reason is not None
        assert r_unknown.target_reason.summary == "No host matched @nonexistent"

        # Invalid targetHostId must NOT mutate follow_me (both when unset and when False)
        r_not_found = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="hello",
            hosts=hosts,
            hub=hub,
            target_host_id=HostId.generate(),
            follow_me=True,
            now=NOW,
        )
        assert r_not_found.kind == "not_found"
        assert hub.get_focus_state(owner) is None

        hub.set_focus_state(owner, follow_me=False, source="app", now=NOW)
        r_not_found_2 = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="hello",
            hosts=hosts,
            hub=hub,
            target_host_id=HostId.generate(),
            follow_me=True,
            now=NOW,
        )
        assert r_not_found_2.kind == "not_found"
        focus_preserved = hub.get_focus_state(owner)
        assert focus_preserved is not None and focus_preserved.follow_me is False

    asyncio.run(_run())


def test_client_tz_offset_minutes_consistency_and_no_double_habit_on_repair_replay() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        hub = InMemoryHostSessionHub()
        pending = PendingIntentStore()
        habits = InMemoryHabitRepository()
        router = MeshPreRouter(pending_store=pending, habit_repo=habits, clock=lambda: NOW)

        # 15:00 UTC + 420 min (UTC+7) = 22:00 local -> night
        utc_15 = datetime(2026, 11, 20, 15, 0, 0, tzinfo=UTC)
        mac = _host(owner, "Mac")
        win = _host(owner, "Win", platform=HostPlatform.WINDOWS)
        await _open_online(hub, owner, mac)

        async def _win_send(raw: str) -> None:
            frame = json.loads(raw)
            if frame.get("method") == "tools/call":
                hub.on_response(
                    str(win.host_id),
                    frame["id"],
                    {"_meta": {"xuna/status": "ok"}, "structuredContent": {"playing": True}},
                )

        async def _close() -> None:
            return None

        await hub.open_session(
            str(win.host_id), object(), _win_send, _close, _close, owner_id=owner
        )

        desc = ToolDescriptor(
            name="media.play_pause",
            description="Toggle playback",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            output_schema={"type": "object", "properties": {}, "additionalProperties": False},
            family=ToolFamily.MEDIA,
            risk=RiskTier.R1,
        )
        manifests = {
            mac.host_id: Manifest(host_id=mac.host_id, tools=(desc,), updated_at=utc_15),
            win.host_id: Manifest(host_id=win.host_id, tools=(desc,), updated_at=utc_15),
        }
        grants = {
            mac.host_id: (Grant(mac.host_id, ToolFamily.MEDIA, RiskTier.R1, True, utc_15),),
            win.host_id: (Grant(win.host_id, ToolFamily.MEDIA, RiskTier.R1, True, utc_15),),
        }

        pending.save(
            PendingIntent(
                owner_id=owner,
                pet_id="pet-1",
                original_tool="media.play_pause",
                original_arguments={},
                original_text="pause",
                candidate_host_ids=(mac.host_id, win.host_id),
                created_at=utc_15,
                suggested_host_id=None,
                family="media",
            )
        )
        rep = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="Win",
            hosts=[mac, win],
            hub=hub,
            now=utc_15 + timedelta(seconds=10),
            client_tz_offset_minutes=420,
        )
        assert rep.kind == "repair_dispatch"
        assert rep.habit_recorded_in_preroute is True

        rows = await habits.list_habits(owner, pet_id="pet-1")
        assert len(rows) == 1
        assert rows[0].context_key == "media:-:night"
        assert rows[0].sample_count == 1
        assert rows[0].positive_score == 1.0

        # Replaying through Orchestrator with habit_recorded_in_preroute=True must NOT double-record
        class _MemRepo:
            async def get_host(self, host_id: HostId) -> Host | None:
                return {mac.host_id: mac, win.host_id: win}.get(host_id)

            async def list_hosts(self, _owner_id: uuid.UUID) -> list[Host]:
                return [mac, win]

            async def get_manifest(self, host_id: HostId) -> Manifest | None:
                return manifests.get(host_id)

            async def get_grants(self, host_id: HostId) -> list[Grant]:
                return list(grants.get(host_id, ()))

        orch = Orchestrator(
            host_repo=_MemRepo(),
            hub=hub,
            approval_repo=InMemoryApprovalRepository(),
            approval_broker=InMemoryApprovalBroker(),
            local_result_broker=InMemoryLocalResultBroker(),
            last_target_store=InMemoryLastTargetStore(),
            audit=MemoryAuditLog(),
            habit_repo=habits,
            pending_intent_store=pending,
            clock=lambda: utc_15 + timedelta(seconds=10),
        )

        async def _sink(_ev: object) -> None:
            return None

        replay_ctx = ActionContext(
            owner_id=owner,
            pet_id="pet-1",
            locale=Locale.VI,
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            emit_event=_sink,
            pinned_host_id=rep.pinned_host_id,
            client_tz_offset_minutes=420,
            habit_recorded_in_preroute=rep.habit_recorded_in_preroute,
        )
        msg = await orch.handle_call(
            ToolCall(id="rep-1", name="media__play_pause", arguments={}),
            replay_ctx,
        )
        assert json.loads(msg.content)["status"] == "ok"
        rows_after_replay = await habits.list_habits(owner, pet_id="pet-1")
        assert len(rows_after_replay) == 1
        assert rows_after_replay[0].sample_count == 1
        assert rows_after_replay[0].positive_score == 1.0

        # resolve_host_target_for_call with tz=420 uses night bucket (m_hab > 0.5)
        res_tz = await resolve_host_target_for_call(
            tool_name="media.play_pause",
            clean_args={},
            requested_target=None,
            ctx=ActionContext(
                owner_id=owner,
                pet_id="pet-1",
                locale=Locale.VI,
                client_caps=(CAP_HOST_ACTIONS_V1,),
                board_available=False,
                emit_event=_sink,
                client_tz_offset_minutes=420,
            ),
            active_hosts=[mac, win],
            manifests_by_host=manifests,
            grants_by_host=grants,
            online_hosts_with_tool=[mac, win],
            reference_descriptor=desc,
            hub=hub,
            last_target_store=InMemoryLastTargetStore(),
            host_default_store=InMemoryHostDefaultStore(),
            habit_repo=habits,
            pending_store=pending,
            cu_session_store=InMemoryComputerUseSessionStore(),
            now=utc_15 + timedelta(seconds=20),
        )
        assert res_tz.scoring_decision is not None
        by_id = {c.host.host_id: c for c in res_tz.scoring_decision.candidates}
        assert by_id[win.host_id].m_hab > 0.5

        # Explicit-target ambiguity must set suggested_host_id=None
        max_host = _host(owner, "Max")
        await _open_online(hub, owner, max_host)
        amb_manifests = {
            **manifests,
            max_host.host_id: Manifest(host_id=max_host.host_id, tools=(desc,), updated_at=utc_15),
        }
        amb_grants = {
            **grants,
            max_host.host_id: (
                Grant(max_host.host_id, ToolFamily.MEDIA, RiskTier.R1, True, utc_15),
            ),
        }
        amb_pending = PendingIntentStore()
        amb_res = await resolve_host_target_for_call(
            tool_name="media.play_pause",
            clean_args={},
            requested_target="Ma",
            ctx=ActionContext(
                owner_id=owner,
                pet_id="pet-1",
                locale=Locale.VI,
                client_caps=(CAP_HOST_ACTIONS_V1,),
                board_available=False,
                emit_event=_sink,
                pinned_host_id=None,
                client_tz_offset_minutes=420,
            ),
            active_hosts=[mac, max_host],
            manifests_by_host=amb_manifests,
            grants_by_host=amb_grants,
            online_hosts_with_tool=[mac, max_host],
            reference_descriptor=desc,
            hub=hub,
            last_target_store=InMemoryLastTargetStore(),
            host_default_store=InMemoryHostDefaultStore(),
            habit_repo=habits,
            pending_store=amb_pending,
            cu_session_store=InMemoryComputerUseSessionStore(),
            now=utc_15 + timedelta(seconds=30),
        )
        assert amb_res.resolved is False
        assert amb_res.error_status == "ask_target"
        saved = amb_pending.get(owner, "pet-1", utc_15 + timedelta(seconds=30))
        assert saved is not None
        assert saved.suggested_host_id is None
        assert set(saved.candidate_host_ids) == {mac.host_id, max_host.host_id}

        # record_execution_habit uses descriptor.family directly (e.g. shortcuts.run -> workflow)
        wf_desc = ToolDescriptor(
            name="shortcuts.run",
            description="Run shortcut",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            output_schema={"type": "object", "properties": {}, "additionalProperties": False},
            family=ToolFamily.WORKFLOW,
            risk=RiskTier.R1,
        )
        await record_execution_habit(
            habits,
            ctx=ActionContext(
                owner_id=owner,
                pet_id="pet-1",
                locale=Locale.VI,
                client_caps=(CAP_HOST_ACTIONS_V1,),
                board_available=False,
                emit_event=_sink,
                client_tz_offset_minutes=420,
            ),
            descriptor=wf_desc,
            clean_args={},
            host_id=win.host_id,
            now=utc_15 + timedelta(seconds=40),
        )
        wf_rows = await habits.list_habits(owner, pet_id="pet-1", context_key="workflow:-:night")
        assert len(wf_rows) == 1
        assert wf_rows[0].host_id == win.host_id

    asyncio.run(_run())


def test_prerouter_colliding_alias_pronoun_guard_and_explicit_it_tag() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        hub = InMemoryHostSessionHub()
        pending = PendingIntentStore()
        habits = InMemoryHabitRepository()
        router = MeshPreRouter(pending_store=pending, habit_repo=habits, clock=lambda: NOW)

        other_pc_1 = _host(owner, "Other PC 1", platform=HostPlatform.WINDOWS)
        bit_host = _host(owner, "Bit", platform=HostPlatform.WINDOWS)
        macbook = _host(owner, "MacBook")
        it_host = _host(owner, "IT")
        for h in (other_pc_1, bit_host, macbook, it_host):
            await _open_online(hub, owner, h)

        # Priority 4 PendingIntent reply "other pc" must NOT hijack Other PC 1
        pending.save(
            PendingIntent(
                owner_id=owner,
                pet_id="pet-1",
                original_tool="media.play_pause",
                original_arguments={},
                original_text="pause",
                candidate_host_ids=(other_pc_1.host_id, macbook.host_id),
                created_at=NOW,
                suggested_host_id=other_pc_1.host_id,
                family="media",
            )
        )
        rep_other = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="other pc",
            hosts=[other_pc_1, macbook],
            hub=hub,
            now=NOW + timedelta(seconds=5),
        )
        assert rep_other.kind == "pass"
        assert pending.get(owner, "pet-1", NOW + timedelta(seconds=6)) is None
        assert await habits.list_habits(owner, pet_id="pet-1") == []

        # /may other pc & @it against colliding hosts -> target_required (ask_target)
        r_may_other = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="/may other pc",
            hosts=[other_pc_1, macbook],
            hub=hub,
            now=NOW + timedelta(seconds=10),
        )
        assert r_may_other.kind == "target_required"
        assert r_may_other.reason_code == "ask_target"

        r_at_it_collide = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="pause @it",
            hosts=[bit_host, macbook],
            hub=hub,
            now=NOW + timedelta(seconds=15),
        )
        assert r_at_it_collide.kind == "target_required"
        assert r_at_it_collide.reason_code == "ask_target"

        # Explicit @IT & /may IT against Host("IT") still resolve to it_host (§5.4.4 step 2)
        r_at_it_exact = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="check @IT",
            hosts=[it_host, macbook],
            hub=hub,
            now=NOW + timedelta(seconds=20),
        )
        assert r_at_it_exact.kind == "pin"
        assert r_at_it_exact.pinned_host_id == it_host.host_id

        r_may_it_exact = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="/may IT",
            hosts=[it_host, macbook],
            hub=hub,
            now=NOW + timedelta(seconds=25),
        )
        assert r_may_it_exact.kind == "slash_handled"
        assert r_may_it_exact.pinned_host_id == it_host.host_id

        # resolve_host_target_for_call with requested_target="it" delegates to TargetResolver
        desc = ToolDescriptor(
            name="media.play_pause",
            description="Toggle playback",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            output_schema={"type": "object", "properties": {}, "additionalProperties": False},
            family=ToolFamily.MEDIA,
            risk=RiskTier.R1,
        )
        manifests = {
            it_host.host_id: Manifest(host_id=it_host.host_id, tools=(desc,), updated_at=NOW),
            macbook.host_id: Manifest(host_id=macbook.host_id, tools=(desc,), updated_at=NOW),
        }
        grants = {
            it_host.host_id: (Grant(it_host.host_id, ToolFamily.MEDIA, RiskTier.R1, True, NOW),),
            macbook.host_id: (Grant(macbook.host_id, ToolFamily.MEDIA, RiskTier.R1, True, NOW),),
        }
        last_store = InMemoryLastTargetStore()
        last_store.set_last_target(owner, "pet-1", "MacBook", NOW - timedelta(seconds=15))

        async def _sink(_ev: object) -> None:
            return None

        res_pronoun = await resolve_host_target_for_call(
            tool_name="media.play_pause",
            clean_args={},
            requested_target="it",
            ctx=ActionContext(
                owner_id=owner,
                pet_id="pet-1",
                locale=Locale.VI,
                client_caps=(CAP_HOST_ACTIONS_V1,),
                board_available=False,
                emit_event=_sink,
            ),
            active_hosts=[it_host, macbook],
            manifests_by_host=manifests,
            grants_by_host=grants,
            online_hosts_with_tool=[it_host, macbook],
            reference_descriptor=desc,
            hub=hub,
            last_target_store=last_store,
            host_default_store=InMemoryHostDefaultStore(),
            habit_repo=None,
            pending_store=PendingIntentStore(),
            cu_session_store=InMemoryComputerUseSessionStore(),
            now=NOW,
        )
        assert res_pronoun.resolved is True
        assert res_pronoun.host == macbook

    asyncio.run(_run())


def test_strip_single_tag_punctuation_and_priority1_clears_pending_before_habit_error() -> None:
    from xuna_backend.modules.actions.application.mesh_router import (
        HostCallRoutingResult,
        build_action_target_required_event,
    )
    from xuna_backend.modules.actions.domain.mesh_scoring import (
        ActionTargetRequiredReason,
        TargetReason,
        TargetReasonCode,
    )

    class _FailingHabitRepo(InMemoryHabitRepository):
        async def get_habit(
            self, owner_id: uuid.UUID, pet_id: str, context_key: str, host_id: HostId
        ) -> None:
            raise RuntimeError("db unavailable")

    async def _run() -> None:
        owner = uuid.uuid4()
        hub = InMemoryHostSessionHub()
        pending = PendingIntentStore()
        oneil = _host(owner, "O'Neil")
        other = _host(owner, "OtherHost")
        await _open_online(hub, owner, oneil)
        await _open_online(hub, owner, other)

        router = MeshPreRouter(pending_store=pending, habit_repo=_FailingHabitRepo())

        # Punctuation spacing after @tag removal: no stray space before '?'
        tag_res = await router.route(
            owner_id=owner,
            pet_id="pet-1",
            text="pause on @O'Neil?",
            hosts=[oneil, other],
            hub=hub,
            now=NOW,
        )
        assert tag_res.kind == "pin"
        assert tag_res.cleaned_text == "pause on?"

        # Priority 1 must clear PendingIntent BEFORE habit write raises
        pending.save(
            PendingIntent(
                owner_id=owner,
                pet_id="pet-1",
                original_tool="media.play_pause",
                original_arguments={},
                original_text="pause",
                candidate_host_ids=(oneil.host_id, other.host_id),
                created_at=NOW,
                suggested_host_id=oneil.host_id,
                family="media",
            )
        )
        import contextlib

        with contextlib.suppress(RuntimeError):
            await router.route(
                owner_id=owner,
                pet_id="pet-1",
                text="pause",
                hosts=[oneil, other],
                hub=hub,
                target_host_id=other.host_id,
                now=NOW,
            )
        assert pending.get(owner, "pet-1", NOW) is None

        # build_action_target_required_event maps ambiguous_candidates -> ask_target
        routing = HostCallRoutingResult(
            resolved=False,
            error_status=TargetReasonCode.AMBIGUOUS_CANDIDATES.value,
            target_reason=TargetReason(
                reason_code=TargetReasonCode.AMBIGUOUS_CANDIDATES,
                summary="Locked host needs confirmation",
            ),
        )
        ev = build_action_target_required_event("inv-1", "app.launch", routing)
        assert ev.reason_code == ActionTargetRequiredReason.ASK_TARGET.value
        assert ev.target_reason == "Locked host needs confirmation"

    asyncio.run(_run())


def test_tier5_focused_offline_host_triggers_wol_and_multi_tag_skips_focus_pin() -> None:
    """Tier 5 (Gaps 1 & 5): offline focused host emits host_offline+WoL; 2-tag skips focus pin."""
    from xuna_backend.modules.actions.application.catalog import is_tool_granted  # noqa: F401
    from xuna_backend.modules.actions.domain.entities import HostWakeRequestEvent
    from xuna_backend.modules.actions.domain.wake_queue import InMemoryWakeQueueStore
    from xuna_backend.modules.conversation.application.mesh_turn_helper import prepare_mesh_turn

    class _Repo:
        def __init__(
            self,
            hosts: list[Host],
            manifests: dict[HostId, Manifest],
            grants: dict[HostId, list[Grant]],
        ) -> None:
            self._hosts = hosts
            self._manifests = manifests
            self._grants = grants

        async def list_hosts(self, owner_id: uuid.UUID) -> list[Host]:
            return [h for h in self._hosts if h.owner_id == owner_id]

        async def get_manifest(self, host_id: HostId) -> Manifest | None:
            return self._manifests.get(host_id)

        async def get_grants(self, host_id: HostId) -> list[Grant]:
            return list(self._grants.get(host_id, []))

    async def _run() -> None:
        owner = uuid.uuid4()
        hub = InMemoryHostSessionHub()
        wake_queue = InMemoryWakeQueueStore()
        router = MeshPreRouter(
            pending_store=PendingIntentStore(),
            habit_repo=InMemoryHabitRepository(),
            wake_queue=wake_queue,
            clock=lambda: NOW,
        )

        mac = _host(owner, "AlphaMac", wol_mac="AA:BB:CC:DD:EE:99")
        win = _host(owner, "BetaWin", platform=HostPlatform.WINDOWS)
        tool = ToolDescriptor(
            name="media.play_pause",
            description="Toggle media",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            output_schema={"type": "object", "properties": {}, "additionalProperties": False},
            family=ToolFamily.MEDIA,
            risk=RiskTier.R1,
        )
        m_mac = Manifest(host_id=mac.host_id, tools=(tool,), updated_at=NOW)
        m_win = Manifest(host_id=win.host_id, tools=(tool,), updated_at=NOW)
        grants = {
            mac.host_id: [Grant(mac.host_id, ToolFamily.MEDIA, RiskTier.R1, True, NOW)],
            win.host_id: [Grant(win.host_id, ToolFamily.MEDIA, RiskTier.R1, True, NOW)],
        }
        repo = _Repo([mac, win], {mac.host_id: m_mac, win.host_id: m_win}, grants)

        # Open only Win; Mac is focused (follow_me=False) and offline
        await _open_online(hub, owner, win)
        hub.set_focus_state(
            owner, focused_host_id=mac.host_id, follow_me=False, source="app", now=NOW
        )

        prep_off = await prepare_mesh_turn(
            owner_id=owner,
            pet_id="pet-1",
            text="pause music",
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            self_host_id=None,
            target_host_id=None,
            follow_me=None,
            tz_offset_minutes=0,
            include_autonomous_job=False,
            consent_repo=None,
            host_repo=repo,  # type: ignore[arg-type]
            hub=hub,
            audit=MemoryAuditLog(),
            pre_router=router,
        )
        assert prep_off.outcome.kind == "target_required"
        assert prep_off.outcome.reason_code == "host_offline"
        assert prep_off.short_circuit is True
        assert prep_off.target_required_event is not None
        assert prep_off.target_required_event.reason_code == "host_offline"
        assert isinstance(prep_off.wake_request_event, HostWakeRequestEvent)
        assert prep_off.wake_request_event.mac_address == "AA:BB:CC:DD:EE:99"

        # With follow_me=True and Win online, offline pin is bypassed cleanly
        hub.set_focus_state(
            owner, focused_host_id=mac.host_id, follow_me=True, source="app", now=NOW
        )
        prep_fol = await prepare_mesh_turn(
            owner_id=owner,
            pet_id="pet-1",
            text="pause music",
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            self_host_id=None,
            target_host_id=None,
            follow_me=None,
            tz_offset_minutes=0,
            include_autonomous_job=False,
            consent_repo=None,
            host_repo=repo,  # type: ignore[arg-type]
            hub=hub,
            audit=MemoryAuditLog(),
            pre_router=router,
        )
        assert prep_fol.outcome.kind == "pass"
        assert prep_fol.pinned_host_id is None
        assert any(t.name == "media__play_pause" for t in prep_fol.tools)

        # Gap 5: 2-@tag cross-host utterance skips Priority 5 focus pin and keeps mesh__status
        await _open_online(hub, owner, mac)
        hub.set_focus_state(
            owner, focused_host_id=mac.host_id, follow_me=False, source="app", now=NOW
        )
        prep_2tag = await prepare_mesh_turn(
            owner_id=owner,
            pet_id="pet-1",
            text="Copy clipboard from @AlphaMac to @BetaWin",
            client_caps=(CAP_HOST_ACTIONS_V1,),
            board_available=False,
            self_host_id=None,
            target_host_id=None,
            follow_me=None,
            tz_offset_minutes=0,
            include_autonomous_job=False,
            consent_repo=None,
            host_repo=repo,  # type: ignore[arg-type]
            hub=hub,
            audit=MemoryAuditLog(),
            pre_router=router,
        )
        assert prep_2tag.outcome.kind == "pass"
        assert prep_2tag.outcome.skip_focus_pin is True
        assert prep_2tag.pinned_host_id is None
        tool_names = {t.name for t in prep_2tag.tools}
        assert "mesh__status" in tool_names
        assert "mesh__find_resource" in tool_names

    asyncio.run(_run())
