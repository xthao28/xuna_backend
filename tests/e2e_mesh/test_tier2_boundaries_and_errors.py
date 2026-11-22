"""Tier 2 Backend E2E Mesh Tests: Boundary & Corner Cases (F3-F9, F12).

Opaque-box boundary value analysis and error-path tests covering:
- F3: Migration 0022 collision suffixing, active folded 409 conflicts, tombstone reuse,
  single-lane constraint, downgrade restoration
- F4: Zero HostTwin DB/audit leakage, clean reconnect twin, empty-tools roster omission,
  focus clear on revoke, server/bridge unknown battery
- F5: 404 foreign/revoked targetHostId, offline explicit target host_offline (no fallback),
  ambiguous @place filtering, PendingIntent clearing/multi-token guard, punctuation/quotes
- F6: <2-char fuzzy safety, fuzzy near-tie H3 guard, no_host vs host_offline vs grant_denied
  precedence, R0..R3 threshold & screenLocked boundaries, habit gamma=0.85 math & dual-bucket,
  M_res 0.25 vs 0.50
- F7: Prober & meta-tool input validation, cross-owner hostIds isolation, all-timeout resilience,
  U5 consent denial, dynamic bridge tool discovery
- F8: Strict schema validation (extra="forbid", whitespace, MAC regex, subnetHash regex),
  404 isolation, AuditEvent contextKey privacy
- F9: XTP >512 KiB close 1009, binary frame close 1002, path/count/extraKey rejection
- F12: WoL missing MAC & >3 queued cap, battery 15% boundary, R3 MissionEnvelope P-256 enforcement.
"""

import asyncio
import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest
import websockets

from tests.e2e_mesh.conftest import fetch_scalar, pg_dsn, run_alembic
from tests.e2e_mesh.xtp_sim import (
    FakeSpeechRecognizer,
    FakeSpeechSynthesizer,
    RpcErrorResponse,
    ScriptedTextGenerator,
    SimulatedOwnerClient,
    SimulatedXtpHost,
    done_event,
    make_standard_mesh_tools,
    make_tool_descriptor,
)
from xuna_backend.modules.cognition.domain.types import (
    ChatRequest,
    Role,
    TextDelta,
    ToolCall,
    ToolCallRequested,
)

pytestmark = [pytest.mark.postgres, pytest.mark.anyio]


# =============================================================================
# Feature 3 Boundaries — Migration & Schema Constraints
# =============================================================================


async def test_t2_f3_01_migration_0022_alias_folded_collision_suffixing(
    clean_mesh_db: str,
) -> None:
    """[M1/M2] T2-F3-01: Pre-0022 duplicate folded aliases backfill with deterministic _2/_3."""
    dsn = pg_dsn(clean_mesh_db)
    owner_uuid = uuid.uuid4()
    h1, h2, h3, h_rev = uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    t1 = datetime(2026, 11, 20, 9, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 11, 20, 9, 1, 0, tzinfo=UTC)
    t3 = datetime(2026, 11, 20, 9, 2, 0, tzinfo=UTC)
    t0 = datetime(2026, 11, 20, 8, 0, 0, tzinfo=UTC)
    pubkey = b"\x04" + bytes(range(1, 65))

    try:
        run_alembic(clean_mesh_db, "downgrade", "0021")
        with psycopg.connect(dsn) as conn:
            conn.execute(
                "INSERT INTO owners (owner_id, issuer, subject, created_at) VALUES (%s,%s,%s,%s)",
                (owner_uuid, "https://auth.test.xuna", f"coll-{owner_uuid}", t0),
            )
            for hid, alias, status, ts in (
                (h_rev, "Mac", "revoked", t0),
                (h1, "Mac", "active", t1),
                (h2, "mac", "active", t2),
                (h3, "Mặc", "active", t3),
            ):
                conn.execute(
                    """
                    INSERT INTO hosts (
                        host_id, owner_id, alias, platform, hostname, os_version,
                        agent_version, public_key, status, created_at
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (hid, owner_uuid, alias, "macos", "h", "15.0", "0.1.0", pubkey, status, ts),
                )
            conn.commit()

        run_alembic(clean_mesh_db, "upgrade", "head")
        with psycopg.connect(dsn) as conn:
            rows = dict(
                conn.execute(
                    "SELECT host_id, alias_folded FROM hosts WHERE owner_id = %s",
                    (owner_uuid,),
                ).fetchall()
            )
            assert rows[h1] == "mac"
            assert rows[h2] == "mac_2"
            assert rows[h3] == "mac_3"
            assert rows[h_rev] == "mac"
    finally:
        run_alembic(clean_mesh_db, "upgrade", "head")


async def test_t2_f3_02_enrollment_active_alias_folded_collision_returns_409(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T2-F3-02: Enrolling active host with colliding folded alias returns 409."""
    async with mesh_server_factory() as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        h1 = SimulatedXtpHost(platform="macos")
        try:
            await h1.connect_enroll(srv.ws_url, owner, "PC Nhà")
            resp = await owner.self_enroll_mobile(alias="pc nha", platform="ios")
            assert resp.status_code == 409, resp.text
            assert resp.json()["code"] == "conflict"
        finally:
            await h1.disconnect()
            await owner.aclose()


async def test_t2_f3_03_revoked_host_alias_reuse_and_self_enroll_idempotent(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T2-F3-03: Revoked alias is reusable; self-enroll with same key & alias succeeds."""
    async with mesh_server_factory() as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        h1 = SimulatedXtpHost(platform="macos")
        h2 = SimulatedXtpHost(platform="windows")
        try:
            id1 = await h1.connect_enroll(srv.ws_url, owner, "SharedName")
            del_r = await owner.revoke_host(str(id1))
            assert del_r.status_code == 204

            # New host reuses revoked host's alias
            id2 = await h2.connect_enroll(srv.ws_url, owner, "sharedname")
            assert id2 != id1

            # Self-enroll phone twice with same public_key and same alias
            r1 = await owner.self_enroll_mobile(alias="MyPhone", platform="android")
            assert r1.status_code == 200, r1.text
            r2 = await owner.self_enroll_mobile(alias="MyPhone", platform="android")
            assert r2.status_code == 200, r2.text
        finally:
            await h1.disconnect()
            await h2.disconnect()
            await owner.aclose()


async def test_t2_f3_04_cu_sessions_blocks_two_active_lanes_same_owner_and_host(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T2-F3-04: Unique index blocks two active CU lanes on same (owner_id, host_id)."""
    async with mesh_server_factory() as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        host_a = SimulatedXtpHost(
            platform="macos",
            tools=make_standard_mesh_tools(platform="macos", include_cu=True),
        )
        try:
            id_a = await host_a.connect_enroll(srv.ws_url, owner, "SingleHost")
            await owner.grant_all_standard(str(id_a), max_risk="R3")

            r1 = await owner.invoke_tool_dev(str(id_a), "cu.session_start", {"goal": "Lane 1"})
            assert r1.status_code == 200, r1.text

            with psycopg.connect(pg_dsn(srv.database_url)) as conn:
                owner_uuid = fetch_scalar(
                    conn, "SELECT owner_id FROM hosts WHERE host_id = %s", (id_a,)
                )
                now = datetime.now(UTC)
                with pytest.raises(psycopg.errors.UniqueViolation):
                    conn.execute(
                        """
                        INSERT INTO cu_sessions (
                            session_id, owner_id, host_id, target_alias, goal,
                            apps, domains, max_steps, max_minutes, steps_used,
                            active, started_at, last_heartbeat_at
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        (
                            f"cu-dup-{uuid.uuid4().hex[:8]}",
                            owner_uuid,
                            id_a,
                            "SingleHost",
                            "Lane 2",
                            "{}",
                            "{}",
                            10,
                            5,
                            0,
                            True,
                            now,
                            now,
                        ),
                    )
                    conn.commit()
                conn.rollback()
        finally:
            await host_a.disconnect()
            await owner.aclose()


async def test_t2_f3_05_migration_0022_downgrade_restores_0021_schema(
    clean_mesh_db: str,
) -> None:
    """[M1/M2] T2-F3-05: alembic downgrade 0021 drops 0022 artifacts & restores old CU index."""
    dsn = pg_dsn(clean_mesh_db)
    try:
        run_alembic(clean_mesh_db, "downgrade", "0021")
        with psycopg.connect(dsn) as conn:
            rev = fetch_scalar(conn, "SELECT version_num FROM alembic_version")
            assert rev == "0021"
            tables = {
                r[0]
                for r in conn.execute(
                    """
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                    """
                ).fetchall()
            }
            assert "host_routing_habits" not in tables
            assert "host_defaults" not in tables
            indexes = {
                r[0]
                for r in conn.execute(
                    "SELECT indexname FROM pg_indexes WHERE tablename = 'cu_sessions'"
                ).fetchall()
            }
            assert "uq_cu_sessions_owner_active" in indexes
            assert "uq_cu_sessions_owner_host_active" not in indexes
    finally:
        run_alembic(clean_mesh_db, "upgrade", "head")


# =============================================================================
# Feature 4 Boundaries — HostTwin & Host Roster
# =============================================================================


async def test_t2_f4_01_host_twin_zero_leakage_into_postgres_and_audit(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T2-F4-01: Live presence/resources never appear in SQL tables or audit_events."""
    async with mesh_server_factory() as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        h1 = SimulatedXtpHost(platform="macos")
        try:
            id1 = await h1.connect_enroll(srv.ws_url, owner, "TwinCheck")
            sentinel_app = "synthetic_app_987"
            sentinel_ws = "synthetic_ws_987"
            await h1.send_presence(
                idle_seconds=7,
                screen_locked=False,
                media_playing=True,
                foreground_app_id=sentinel_app,
            )
            await h1.send_resources(installed_apps=[sentinel_app], workspaces=[sentinel_ws])

            events = await owner.audit_events(limit=100)
            assert sentinel_app not in json.dumps(events)
            assert sentinel_ws not in json.dumps(events)

            with psycopg.connect(pg_dsn(srv.database_url)) as conn:
                row = fetch_scalar(
                    conn,
                    "SELECT row_to_json(hosts) FROM hosts WHERE host_id = %s",
                    (id1,),
                )
                assert sentinel_app not in json.dumps(row)
                assert sentinel_ws not in json.dumps(row)
        finally:
            await h1.disconnect()
            await owner.aclose()


async def test_t2_f4_02_reconnect_starts_with_clean_host_twin_until_presence_arrives(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T2-F4-02: Reconnect does NOT resurrect stale presence before xuna/presence."""
    async with mesh_server_factory() as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        h1 = SimulatedXtpHost(platform="macos")
        try:
            id1 = await h1.connect_enroll(srv.ws_url, owner, "CleanRecon")
            await h1.send_presence(idle_seconds=1, screen_locked=False, media_playing=True)
            await h1.disconnect()

            await h1.connect_reconnect(srv.ws_url, host_id=id1)
            items = await owner.list_hosts()
            assert items[0]["online"] is True
            assert items[0]["presence"] is None
        finally:
            await h1.disconnect()
            await owner.aclose()


async def test_t2_f4_03_host_roster_omitted_when_tools_empty(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T2-F4-03: Text-only turn (no clientCapabilities) omits [HOST_ROSTER]."""
    llm = ScriptedTextGenerator(scripts=[[TextDelta("Hello."), done_event()]])
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h1 = SimulatedXtpHost(platform="macos")
        try:
            await h1.connect_enroll(srv.ws_url, owner, "SilentHost")
            await owner.stream_turn(pet_id, "hello pet", client_capabilities=())
            assert len(llm.requests) == 1
            assert llm.requests[0].tools == ()
            sys_text = "\n".join(
                m.content for m in llm.requests[0].messages if m.role == Role.SYSTEM
            )
            assert "[HOST_ROSTER]" not in sys_text
        finally:
            await h1.disconnect()
            await owner.aclose()


async def test_t2_f4_04_focus_state_cleared_when_focused_host_revoked(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T2-F4-04: Revoking focused host resets focus_state to auto (None)."""
    async with mesh_server_factory() as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        h1 = SimulatedXtpHost(platform="macos")
        h2 = SimulatedXtpHost(platform="windows")
        try:
            id1 = await h1.connect_enroll(srv.ws_url, owner, "FocusDrop")
            await h2.connect_enroll(srv.ws_url, owner, "RemainingHost")
            r = await owner.set_focus(host_id=str(id1), follow_me=False)
            assert r.status_code == 200

            del_r = await owner.revoke_host(str(id1))
            assert del_r.status_code == 204
            items = await owner.list_hosts()
            assert len(items) == 1
            assert items[0]["focused"] is False
        finally:
            await h1.disconnect()
            await h2.disconnect()
            await owner.aclose()


async def test_t2_f4_05_server_and_bridge_rendered_with_battery_unknown(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T2-F4-05: Linux server host renders kind=server, battery=unknown (never 100%)."""
    llm = ScriptedTextGenerator(scripts=[[TextDelta("Done."), done_event()]])
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h_srv = SimulatedXtpHost(platform="linux")
        try:
            id_srv = await h_srv.connect_enroll(srv.ws_url, owner, "RackNode")
            await owner.patch_host(str(id_srv), {"schemaVersion": 1, "hostKind": "server"})
            await h_srv.send_presence(
                idle_seconds=0,
                screen_locked=False,
                media_playing=False,
                include_battery=False,
            )
            await owner.stream_turn(pet_id, "check server")
            sys_text = "\n".join(
                m.content for m in llm.requests[0].messages if m.role == Role.SYSTEM
            )
            assert "RackNode" in sys_text
            assert "kind=server" in sys_text
            assert "battery=unknown" in sys_text
        finally:
            await h_srv.disconnect()
            await owner.aclose()


# =============================================================================
# Feature 5 Boundaries — Pre-Router & PendingIntent
# =============================================================================


async def test_t2_f5_01_target_host_id_foreign_or_revoked_returns_404(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T2-F5-01: TurnBody.targetHostId belonging to another owner or revoked returns 404."""
    async with mesh_server_factory() as srv:
        owner_a = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        owner_b = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_a = await owner_a.bind_pet()
        await owner_a.enable_consents()
        h_a = SimulatedXtpHost(platform="macos")
        h_b = SimulatedXtpHost(platform="windows")
        try:
            id_a = await h_a.connect_enroll(srv.ws_url, owner_a, "OwnerAHost")
            id_b = await h_b.connect_enroll(srv.ws_url, owner_b, "OwnerBHost")

            r_foreign = await owner_a.post_turn_raw(
                pet_a,
                {
                    "schemaVersion": 1,
                    "text": "hello",
                    "clientCapabilities": ["host_actions_v1"],
                    "targetHostId": str(id_b),
                },
            )
            assert r_foreign.status_code == 404
            assert r_foreign.json()["code"] == "not_found"

            await owner_a.revoke_host(str(id_a))
            r_revoked = await owner_a.post_turn_raw(
                pet_a,
                {
                    "schemaVersion": 1,
                    "text": "hello",
                    "clientCapabilities": ["host_actions_v1"],
                    "targetHostId": str(id_a),
                },
            )
            assert r_revoked.status_code == 404
        finally:
            await h_a.disconnect()
            await h_b.disconnect()
            await owner_a.aclose()
            await owner_b.aclose()


async def test_t2_f5_02_offline_explicit_target_emits_host_offline_never_falls_back(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T2-F5-02: Offline targetHostId or @alias emits host_offline, never routes sibling."""
    llm = ScriptedTextGenerator(
        responder=lambda _req, _idx: [
            ToolCallRequested(ToolCall(id="off1", name="media__play_pause", arguments={})),
            done_event(),
        ]
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h_off = SimulatedXtpHost(platform="macos")
        h_on = SimulatedXtpHost(platform="windows")
        try:
            id_off = await h_off.connect_enroll(srv.ws_url, owner, "OfflineMac")
            id_on = await h_on.connect_enroll(srv.ws_url, owner, "OnlineWin")
            await owner.grant_all_standard(str(id_off), max_risk="R1")
            await owner.grant_all_standard(str(id_on), max_risk="R1")
            await owner.patch_host(
                str(id_off),
                {"schemaVersion": 1, "wolMac": "AA:BB:CC:DD:EE:01"},
            )
            await h_off.disconnect()

            t1 = await owner.stream_turn(pet_id, "pause", target_host_id=str(id_off))
            req1 = t1.first("action_target_required")
            assert req1 is not None and req1["reasonCode"] == "host_offline"
            assert req1["candidates"][0]["hostId"] == str(id_off)
            assert req1["candidates"][0]["wolAvailable"] is True
            assert len(h_on.received_tool_calls) == 0

            t2 = await owner.stream_turn(pet_id, "@OfflineMac pause")
            req2 = t2.first("action_target_required")
            assert req2 is not None and req2["reasonCode"] == "host_offline"
            assert len(h_on.received_tool_calls) == 0
        finally:
            await h_off.disconnect()
            await h_on.disconnect()
            await owner.aclose()


async def test_t2_f5_03_ambiguous_at_place_returns_filtered_ask_target(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T2-F5-03: @office matching 2 hosts emits ask_target excluding 3rd host in home."""
    llm = ScriptedTextGenerator(
        responder=lambda _req, _idx: [
            ToolCallRequested(ToolCall(id="pl1", name="media__play_pause", arguments={})),
            done_event(),
        ]
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h1, h2, h3 = (
            SimulatedXtpHost(platform="macos"),
            SimulatedXtpHost(platform="windows"),
            SimulatedXtpHost(platform="linux"),
        )
        try:
            id1 = await h1.connect_enroll(srv.ws_url, owner, "OfficeMac")
            id2 = await h2.connect_enroll(srv.ws_url, owner, "OfficeWin")
            id3 = await h3.connect_enroll(srv.ws_url, owner, "HomeLinux")
            for hid in (id1, id2):
                await owner.patch_host(str(hid), {"schemaVersion": 1, "place": "office"})
                await owner.grant_all_standard(str(hid), max_risk="R1")
            await owner.patch_host(str(id3), {"schemaVersion": 1, "place": "home"})
            await owner.grant_all_standard(str(id3), max_risk="R1")

            turn = await owner.stream_turn(pet_id, "@office mute")
            req = turn.first("action_target_required")
            assert req is not None and req["reasonCode"] == "ask_target"
            cand_ids = {c["hostId"] for c in req["candidates"]}
            assert cand_ids == {str(id1), str(id2)}
            assert str(id3) not in cand_ids
        finally:
            await h1.disconnect()
            await h2.disconnect()
            await h3.disconnect()
            await owner.aclose()


async def test_t2_f5_04_and_05_pending_intent_consumed_once_and_ignored_on_multi_word(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T2-F5-04/05: Consumed/overwritten PendingIntent & >4-token turns do not replay.

    Note: 120s TTL expiry is not fast-forwarded over HTTP (public Settings has no TTL override).
    """
    llm = ScriptedTextGenerator(
        scripts=[
            [
                ToolCallRequested(ToolCall(id="p1", name="media__play_pause", arguments={})),
                done_event(),
            ],
            [TextDelta("Which?"), done_event()],
            [TextDelta("Replayed once."), done_event()],
            [TextDelta("Plain chat."), done_event()],
            [
                ToolCallRequested(ToolCall(id="p2", name="media__play_pause", arguments={})),
                done_event(),
            ],
            [TextDelta("Which again?"), done_event()],
            [TextDelta("Normal chat reply."), done_event()],
        ]
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h1, h2 = SimulatedXtpHost(platform="macos"), SimulatedXtpHost(platform="windows")
        try:
            id1 = await h1.connect_enroll(srv.ws_url, owner, "NodeAlpha")
            id2 = await h2.connect_enroll(srv.ws_url, owner, "NodeBeta")
            await owner.grant_all_standard(str(id1), max_risk="R1")
            await owner.grant_all_standard(str(id2), max_risk="R1")
            subnet = "abcdefabcdefabcdefabcdefabcdef01"
            for h in (h1, h2):
                await h.send_presence(
                    idle_seconds=10, screen_locked=False, media_playing=False, subnet_hash=subnet
                )

            t1 = await owner.stream_turn(pet_id, "pause", client_subnet_hash=subnet)
            assert t1.require("action_target_required")["reasonCode"] == "ask_target"

            # Repair once
            t2 = await owner.stream_turn(pet_id, "NodeAlpha", client_subnet_hash=subnet)
            assert t2.first("action_result") is not None
            assert len(h1.received_tool_calls) == 1

            # Repeating "NodeAlpha" after PendingIntent was cleared does NOT replay tool
            t3 = await owner.stream_turn(pet_id, "NodeAlpha", client_subnet_hash=subnet)
            assert t3.first("action_result") is None
            assert len(h1.received_tool_calls) == 1

            # Trigger ask_target again, then send >4 token new turn -> must bypass PendingIntent
            t4 = await owner.stream_turn(pet_id, "pause", client_subnet_hash=subnet)
            assert t4.first("action_target_required") is not None
            t5 = await owner.stream_turn(
                pet_id,
                "never mind just tell me a short story today",
                client_subnet_hash=subnet,
            )
            assert t5.first("action_result") is None
            assert len(h1.received_tool_calls) == 1
            assert len(h2.received_tool_calls) == 0
        finally:
            await h1.disconnect()
            await h2.disconnect()
            await owner.aclose()


async def test_t2_f5_06_quoted_multi_word_nickname_and_trailing_punctuation(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T2-F5-06: @"PC Nha" and trailing punctuation @Alpha, resolve cleanly."""
    llm = ScriptedTextGenerator(
        responder=lambda req, idx: (
            [
                ToolCallRequested(ToolCall(id=f"q{idx}", name="media__play_pause", arguments={})),
                done_event(),
            ]
            if not any(m.role == Role.TOOL for m in req.messages)
            else [TextDelta("OK."), done_event()]
        )
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h1, h2 = SimulatedXtpHost(platform="macos"), SimulatedXtpHost(platform="windows")
        try:
            id1 = await h1.connect_enroll(srv.ws_url, owner, "Alpha")
            id2 = await h2.connect_enroll(srv.ws_url, owner, "Beta")
            await owner.grant_all_standard(str(id1), max_risk="R1")
            await owner.grant_all_standard(str(id2), max_risk="R1")
            await owner.patch_host(str(id2), {"schemaVersion": 1, "nicknames": ["PC Nha"]})

            t_quote = await owner.stream_turn(pet_id, 'pause @"PC Nha"')
            assert t_quote.require("action_result")["hostId"] == str(id2)

            t_punct = await owner.stream_turn(pet_id, "pause @Alpha,")
            assert t_punct.require("action_result")["hostId"] == str(id1)
        finally:
            await h1.disconnect()
            await h2.disconnect()
            await owner.aclose()


# =============================================================================
# Feature 6 Boundaries — Fuzzy H3, Classifications, Risk Tiers, Habits
# =============================================================================


async def test_t2_f6_01_fuzzy_single_char_and_empty_query_no_zero_division(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T2-F6-01: Single-char @m and /may "" never ZeroDivisionError or guess blindly."""
    llm = ScriptedTextGenerator(
        responder=lambda _req, _idx: [
            ToolCallRequested(ToolCall(id="fz1", name="media__play_pause", arguments={})),
            done_event(),
        ]
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h1, h2 = SimulatedXtpHost(platform="macos"), SimulatedXtpHost(platform="windows")
        try:
            id1 = await h1.connect_enroll(srv.ws_url, owner, "Mac")
            id2 = await h2.connect_enroll(srv.ws_url, owner, "Win")
            await owner.grant_all_standard(str(id1), max_risk="R1")
            await owner.grant_all_standard(str(id2), max_risk="R1")

            t1 = await owner.stream_turn(pet_id, "@m pause")
            assert t1.first("error") is None
            req1 = t1.first("action_target_required")
            assert req1 is not None and req1["reasonCode"] == "ask_target"
            assert len(h1.received_tool_calls) == 0
            assert len(h2.received_tool_calls) == 0

            t2 = await owner.stream_turn(pet_id, "/may a")
            assert t2.first("error") is None
            req2 = t2.first("action_target_required")
            assert req2 is not None and req2["reasonCode"] == "ask_target"
        finally:
            await h1.disconnect()
            await h2.disconnect()
            await owner.aclose()


async def test_t2_f6_02_fuzzy_near_tie_and_low_similarity_never_guesses_h3(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T2-F6-02: @"Mac3" against ["Mac1", "Mac2"] and @"xyz" never guess a host (H3)."""
    llm = ScriptedTextGenerator(
        responder=lambda _req, _idx: [
            ToolCallRequested(ToolCall(id="ft1", name="media__play_pause", arguments={})),
            done_event(),
        ]
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h1, h2 = SimulatedXtpHost(platform="macos"), SimulatedXtpHost(platform="windows")
        try:
            id1 = await h1.connect_enroll(srv.ws_url, owner, "Mac1")
            id2 = await h2.connect_enroll(srv.ws_url, owner, "Mac2")
            await owner.grant_all_standard(str(id1), max_risk="R1")
            await owner.grant_all_standard(str(id2), max_risk="R1")

            t_tie = await owner.stream_turn(pet_id, "@Mac3 pause")
            req_tie = t_tie.first("action_target_required")
            assert req_tie is not None and req_tie["reasonCode"] == "ask_target"
            assert len(h1.received_tool_calls) == 0
            assert len(h2.received_tool_calls) == 0

            t_low = await owner.stream_turn(pet_id, "@xyz_unknown pause")
            req_low = t_low.first("action_target_required")
            assert req_low is not None and req_low["reasonCode"] == "ask_target"
            assert len(h1.received_tool_calls) == 0
        finally:
            await h1.disconnect()
            await h2.disconnect()
            await owner.aclose()


async def test_t2_f6_03_classification_priority_no_host_vs_offline_vs_grant_denied(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T2-F6-03: Strict classification order: no_host -> host_offline -> grant_denied."""
    llm = ScriptedTextGenerator(
        responder=lambda req, idx: (
            [
                ToolCallRequested(
                    ToolCall(id=f"cl{idx}", name="app__launch", arguments={"id": "app_x"})
                ),
                done_event(),
            ]
            if not any(m.role == Role.TOOL for m in req.messages)
            else [TextDelta("Handled."), done_event()]
        )
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        # Start with host_screen=False so U5 check triggers on observing tools
        await owner.enable_consents(host_screen=False)

        h_limited = SimulatedXtpHost(
            platform="linux",
            tools=[make_tool_descriptor("host.status", family="host", risk="R0")],
        )
        h_capable_a = SimulatedXtpHost(
            platform="macos",
            tools=make_standard_mesh_tools(platform="macos", include_teleport=True),
        )
        h_capable_b = SimulatedXtpHost(
            platform="windows",
            tools=make_standard_mesh_tools(platform="windows", include_teleport=True),
        )
        try:
            # (a) 0 hosts advertise app.launch -> reasonCode == "no_host"
            await h_limited.connect_enroll(srv.ws_url, owner, "LimitedOnly")
            t_no = await owner.stream_turn(pet_id, "launch app_x")
            req_no = t_no.first("action_target_required")
            assert req_no is not None, f"Expected no_host, got {t_no.events}"
            assert req_no["reasonCode"] == "no_host"

            # (b) 2 capable hosts both offline -> host_offline
            id_a = await h_capable_a.connect_enroll(srv.ws_url, owner, "CapA")
            id_b = await h_capable_b.connect_enroll(srv.ws_url, owner, "CapB")
            await h_capable_a.disconnect()
            await h_capable_b.disconnect()

            t_off = await owner.stream_turn(pet_id, "launch app_x")
            req_off = t_off.first("action_target_required")
            assert req_off is not None and req_off["reasonCode"] == "host_offline"

            # (c) Reconnect CapA online WITHOUT app R2 grant -> grant_denied (CapB still offline)
            await h_capable_a.connect_reconnect(srv.ws_url, host_id=id_a)
            t_gd = await owner.stream_turn(pet_id, "launch app_x")
            req_gd = t_gd.first("action_target_required")
            assert req_gd is not None and req_gd["reasonCode"] == "grant_denied"
            reasons = {c["hostId"]: c["unavailableReason"] for c in req_gd["candidates"]}
            assert reasons[str(id_a)] == "grant_denied"
            assert reasons[str(id_b)] == "host_offline"

            # Also check U5 denial on observing tool workspace.capture_capsule unconditionally
            await owner.grant_all_standard(str(id_a), max_risk="R2")
            llm.push_script(
                [
                    ToolCallRequested(
                        ToolCall(
                            id="u5c",
                            name="workspace__capture_capsule",
                            arguments={"project": "workspace_alpha"},
                        )
                    ),
                    done_event(),
                ]
            )
            llm.push_script([TextDelta("U5 required."), done_event()])
            t_u5 = await owner.stream_turn(pet_id, "capture workspace")
            req_u5 = t_u5.first("action_target_required")
            assert req_u5 is not None, f"Expected grant_denied, got {t_u5.events}"
            assert req_u5["reasonCode"] == "grant_denied"
            u5_reasons = {c["hostId"]: c["unavailableReason"] for c in req_u5["candidates"]}
            assert u5_reasons[str(id_a)] == "u5_consent_required"
        finally:
            await h_limited.disconnect()
            await h_capable_a.disconnect()
            await h_capable_b.disconnect()
            await owner.aclose()


async def test_t2_f6_04_risk_tiered_thresholds_near_tie_asks_target_r0_r1_vs_r3(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T2-F6-04: Risk-tiered thresholds (R0 theta=0.45/0.08, R1 0.58/0.12, R3 0.85/0.25)."""

    def _responder(req: ChatRequest, idx: int) -> list[Any]:
        if any(m.role == Role.TOOL for m in req.messages):
            return [TextDelta("Done."), done_event()]
        last = req.messages[-1].content
        if "r0_status" in last:
            return [
                ToolCallRequested(ToolCall(id=f"r0_{idx}", name="host__status", arguments={})),
                done_event(),
            ]
        if "r1_media" in last:
            return [
                ToolCallRequested(ToolCall(id=f"r1_{idx}", name="media__play_pause", arguments={})),
                done_event(),
            ]
        return [
            ToolCallRequested(
                ToolCall(id=f"r3_{idx}", name="input__click", arguments={"elementId": "btn_1"})
            ),
            done_event(),
        ]

    llm = ScriptedTextGenerator(responder=_responder)
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h1 = SimulatedXtpHost(
            platform="macos",
            tools=make_standard_mesh_tools(platform="macos", include_cu=True),
        )
        h2 = SimulatedXtpHost(
            platform="windows",
            tools=make_standard_mesh_tools(platform="windows", include_cu=True),
        )
        try:
            id1 = await h1.connect_enroll(srv.ws_url, owner, "Node1")
            id2 = await h2.connect_enroll(srv.ws_url, owner, "Node2")
            await owner.grant_all_standard(str(id1), max_risk="R3")
            await owner.grant_all_standard(str(id2), max_risk="R3")

            subnet = "11223344556677889900aabbccddeeff"
            # Near-tie (< 0.08 delta): R0, R1, and R3 all emit ask_target
            await h1.send_presence(
                idle_seconds=5, screen_locked=False, media_playing=False, subnet_hash=subnet
            )
            await h2.send_presence(
                idle_seconds=45, screen_locked=False, media_playing=False, subnet_hash=subnet
            )

            t_r0_tie = await owner.stream_turn(pet_id, "r0_status", client_subnet_hash=subnet)
            req_r0 = t_r0_tie.first("action_target_required")
            assert req_r0 is not None and req_r0["reasonCode"] == "ask_target"

            t_r1_tie = await owner.stream_turn(pet_id, "r1_media", client_subnet_hash=subnet)
            req_r1 = t_r1_tie.first("action_target_required")
            assert req_r1 is not None and req_r1["reasonCode"] == "ask_target"

            t_r3_tie = await owner.stream_turn(pet_id, "click btn_1", client_subnet_hash=subnet)
            req_r3 = t_r3_tie.first("action_target_required")
            assert req_r3 is not None and req_r3["reasonCode"] == "ask_target"
            assert len(h1.received_tool_calls) == 0

            # Moderate margin (~0.15): Node1 has 2 explicit host picks (last_target + habit)
            # enough to clear R0 (0.45/0.08) and R1 (0.58/0.12), but NOT R3 (0.85/0.25).
            for _ in range(2):
                await owner.stream_turn(
                    pet_id, "r0_status", target_host_id=str(id1), client_subnet_hash=subnet
                )
            await h1.send_presence(
                idle_seconds=5, screen_locked=False, media_playing=True, subnet_hash=subnet
            )
            await h2.send_presence(
                idle_seconds=120, screen_locked=False, media_playing=False, subnet_hash=subnet
            )

            t_r0_win = await owner.stream_turn(pet_id, "r0_status", client_subnet_hash=subnet)
            assert t_r0_win.first("action_target_required") is None
            assert t_r0_win.require("action_result")["hostId"] == str(id1)

            t_r1_win = await owner.stream_turn(pet_id, "r1_media", client_subnet_hash=subnet)
            assert t_r1_win.first("action_target_required") is None
            assert t_r1_win.require("action_result")["hostId"] == str(id1)

            before_r3_calls = len(h1.received_tool_calls)
            t_r3_ask = await owner.stream_turn(pet_id, "click btn_1", client_subnet_hash=subnet)
            req_r3_mid = t_r3_ask.first("action_target_required")
            assert req_r3_mid is not None and req_r3_mid["reasonCode"] == "ask_target"
            assert len(h1.received_tool_calls) == before_r3_calls
        finally:
            await h1.disconnect()
            await h2.disconnect()
            await owner.aclose()


async def test_t2_f6_05_single_eligible_r2_screen_locked_desktop_vs_server_vs_r1(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T2-F6-05: Locked desktop R2 asks target; server R2 & desktop R1 auto-resolve."""
    llm_a = ScriptedTextGenerator(
        scripts=[
            [
                ToolCallRequested(
                    ToolCall(id="lk1", name="app__launch", arguments={"id": "editor_x"})
                ),
                done_event(),
            ],
            [TextDelta("Unlock first."), done_event()],
            [
                ToolCallRequested(ToolCall(id="lk2", name="media__play_pause", arguments={})),
                done_event(),
            ],
            [TextDelta("Paused."), done_event()],
            [
                ToolCallRequested(ToolCall(id="lk3", name="workflow__run", arguments={})),
                done_event(),
            ],
            [TextDelta("Workflow ran."), done_event()],
        ]
    )
    async with mesh_server_factory(generator=llm_a) as srv_a:
        owner_a = SimulatedOwnerClient(
            base_url=srv_a.base_url, ws_url=srv_a.ws_url, issuer=srv_a.issuer
        )
        pet_a = await owner_a.bind_pet()
        await owner_a.enable_consents()
        h_desk = SimulatedXtpHost(platform="macos")
        h_srv = SimulatedXtpHost(
            platform="linux",
            tools=[
                make_tool_descriptor("host.status", family="host", risk="R0", platform="linux"),
                make_tool_descriptor(
                    "workflow.run", family="workflow", risk="R2", platform="linux"
                ),
            ],
        )
        try:
            id_d = await h_desk.connect_enroll(srv_a.ws_url, owner_a, "LockedMac")
            await owner_a.grant_all_standard(str(id_d), max_risk="R2")
            await h_desk.send_presence(idle_seconds=300, screen_locked=True, media_playing=False)

            # Case (a): single locked desktop R2 -> 1-candidate ask_target
            t_r2 = await owner_a.stream_turn(pet_a, "launch editor_x")
            req_r2 = t_r2.first("action_target_required")
            assert req_r2 is not None and req_r2["reasonCode"] == "ask_target"
            assert len(req_r2["candidates"]) == 1
            assert req_r2["candidates"][0]["hostId"] == str(id_d)
            assert req_r2["candidates"][0]["eligible"] is True

            # Case (c): same locked desktop on R1 -> auto-resolves
            t_r1 = await owner_a.stream_turn(pet_a, "pause")
            assert t_r1.first("action_target_required") is None
            assert t_r1.require("action_result")["status"] == "ok"

            # Case (b): HostKind.server R2 omits desktop screen-lock ask_target & auto-resolves
            id_s = await h_srv.connect_enroll(srv_a.ws_url, owner_a, "ServerNode")
            await owner_a.patch_host(str(id_s), {"schemaVersion": 1, "hostKind": "server"})
            await owner_a.grant_all_standard(str(id_s), max_risk="R2")
            await h_srv.send_presence(
                idle_seconds=0, screen_locked=True, media_playing=False, include_battery=False
            )

            async def _approve(ev: str, data: dict[str, Any]) -> None:
                if ev == "action_pending":
                    await owner_a.decide_approval(data["approvalId"], "approve")

            t_srv = await owner_a.stream_turn(pet_a, "run workflow", on_event=_approve)
            assert t_srv.first("action_target_required") is None
            pending_srv = t_srv.first("action_pending")
            res_srv = t_srv.first("action_result")
            assert pending_srv is not None and pending_srv["hostId"] == str(id_s)
            assert res_srv is not None and res_srv["status"] == "ok"
            assert res_srv["hostId"] == str(id_s)
        finally:
            await h_desk.disconnect()
            await h_srv.disconnect()
            await owner_a.aclose()


async def test_t2_f6_06_and_07_habit_gamma_085_override_and_dual_bucket_fallback(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T2-F6-06/07: 4 negative steps drop <0.50 & reroute; unseen q uses :-:."""
    llm = ScriptedTextGenerator(
        responder=lambda req, idx: (
            [
                ToolCallRequested(
                    ToolCall(
                        id=f"hb{idx}",
                        name="app__launch",
                        arguments={
                            "id": "tool_beta"
                            if "tool_beta" in req.messages[-1].content
                            else "tool_alpha"
                        },
                    )
                ),
                done_event(),
            ]
            if not any(m.role == Role.TOOL for m in req.messages)
            else [TextDelta("Done."), done_event()]
        )
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h1, h2 = SimulatedXtpHost(platform="macos"), SimulatedXtpHost(platform="windows")
        try:
            id1 = await h1.connect_enroll(srv.ws_url, owner, "HostOne")
            id2 = await h2.connect_enroll(srv.ws_url, owner, "HostTwo")
            await owner.grant_all_standard(str(id1), max_risk="R2")
            await owner.grant_all_standard(str(id2), max_risk="R2")

            async def _approve(ev: str, data: dict[str, Any]) -> None:
                if ev == "action_pending":
                    await owner.decide_approval(data["approvalId"], "approve")

            # Train positive habit on HostOne for tool_alpha (3 picks)
            for _ in range(3):
                await owner.stream_turn(
                    pet_id, "launch tool_alpha", target_host_id=str(id1), on_event=_approve
                )

            # Dual-bucket check: unseen tool_beta has no specific habit, falls back to app:-:daypart
            habits_before = await owner.list_habits(pet_id=pet_id)
            dash_rows = [
                h
                for h in habits_before.json()["items"]
                if h["hostId"] == str(id1) and ":-:" in h["contextKey"]
            ]
            assert len(dash_rows) == 1
            assert dash_rows[0]["netScore"] > 0.50
            assert "*" not in dash_rows[0]["contextKey"]

            subnet = "44556677889900112233445566778899"
            # Both advertise tool_beta (equal M_res=1.0); HostOne wins via app:-:day_part fallback
            await h1.send_presence(
                idle_seconds=5, screen_locked=False, media_playing=False, subnet_hash=subnet
            )
            await h2.send_presence(
                idle_seconds=180, screen_locked=False, media_playing=False, subnet_hash=subnet
            )
            await h1.send_resources(installed_apps=["tool_alpha", "tool_beta"], workspaces=[])
            await h2.send_resources(installed_apps=["other_app", "tool_beta"], workspaces=[])
            t_beta = await owner.stream_turn(
                pet_id, "launch tool_beta", client_subnet_hash=subnet, on_event=_approve
            )
            res_beta = t_beta.first("action_result")
            assert res_beta is not None and res_beta["hostId"] == str(id1)

            # Apply negative redirects away from HostOne onto HostTwo via ask_target -> "HostTwo".
            # HostOne keeps M_res=1.0 vs HostTwo M_res=0.25 with fresher HostTwo presence (and
            # locked screen on round 0) so ask_target fires every round with HostOne candidates[0].
            # Check <0.50 after 4 negative steps, then complete 5th step to clear R2 delta=0.18.
            for idx in range(5):
                await h1.send_presence(
                    idle_seconds=120,
                    screen_locked=(idx == 0),
                    media_playing=False,
                    subnet_hash=subnet,
                )
                await h2.send_presence(
                    idle_seconds=5, screen_locked=False, media_playing=False, subnet_hash=subnet
                )
                t_ask = await owner.stream_turn(
                    pet_id, "launch tool_alpha", client_subnet_hash=subnet
                )
                req_ask = t_ask.first("action_target_required")
                assert req_ask is not None and req_ask["reasonCode"] == "ask_target", t_ask.events
                assert req_ask["candidates"][0]["hostId"] == str(id1)
                t_rep = await owner.stream_turn(
                    pet_id, "HostTwo", client_subnet_hash=subnet, on_event=_approve
                )
                rep_res = t_rep.first("action_result")
                assert rep_res is not None and rep_res["hostId"] == str(id2)
                if idx == 3:
                    habits_after = await owner.list_habits(pet_id=pet_id)
                    h1_alpha_rows = [
                        h
                        for h in habits_after.json()["items"]
                        if h["hostId"] == str(id1)
                        and (":tool_alpha:" in h["contextKey"] or ":-:" in h["contextKey"])
                    ]
                    assert h1_alpha_rows
                    assert all(
                        row["negativeScore"] > 0 and row["netScore"] < 0.50 for row in h1_alpha_rows
                    )

            # Equalize resources & presence: subsequent unpinned turn routes to HostTwo
            await h2.send_resources(installed_apps=["tool_alpha", "tool_beta"], workspaces=[])
            for h in (h1, h2):
                await h.send_presence(
                    idle_seconds=10, screen_locked=False, media_playing=False, subnet_hash=subnet
                )
            t_final = await owner.stream_turn(
                pet_id, "launch tool_alpha", client_subnet_hash=subnet, on_event=_approve
            )
            res_final = t_final.first("action_result")
            assert res_final is not None and res_final["hostId"] == str(id2)
        finally:
            await h1.disconnect()
            await h2.disconnect()
            await owner.aclose()


async def test_t2_f6_08_resource_match_penalty_without_query_vs_unknown(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T2-F6-08: Host reporting resources without q (0.25) loses to host with q (1.00)."""
    llm = ScriptedTextGenerator(
        scripts=[
            [
                ToolCallRequested(
                    ToolCall(id="rq1", name="app__launch", arguments={"id": "app_target_b"})
                ),
                done_event(),
            ],
            [TextDelta("Launched."), done_event()],
        ]
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h_without, h_with = SimulatedXtpHost(platform="macos"), SimulatedXtpHost(platform="windows")
        try:
            id_wo = await h_without.connect_enroll(srv.ws_url, owner, "WithoutApp")
            id_w = await h_with.connect_enroll(srv.ws_url, owner, "WithApp")
            await owner.grant_all_standard(str(id_wo), max_risk="R2")
            await owner.grant_all_standard(str(id_w), max_risk="R2")

            subnet = "99887766554433221100ffeeddccbbaa"
            await h_without.send_presence(
                idle_seconds=5, screen_locked=False, media_playing=False, subnet_hash=subnet
            )
            await h_without.send_resources(installed_apps=["unrelated_app_a"], workspaces=[])
            await h_with.send_presence(
                idle_seconds=5, screen_locked=False, media_playing=False, subnet_hash=subnet
            )
            await h_with.send_resources(installed_apps=["app_target_b"], workspaces=[])

            async def _approve(ev: str, data: dict[str, Any]) -> None:
                if ev == "action_pending":
                    await owner.decide_approval(data["approvalId"], "approve")

            turn = await owner.stream_turn(
                pet_id, "launch app_target_b", client_subnet_hash=subnet, on_event=_approve
            )
            res = turn.first("action_result")
            assert res is not None and res["hostId"] == str(id_w)
            assert len(h_with.received_tool_calls) == 1
            assert len(h_without.received_tool_calls) == 0
        finally:
            await h_without.disconnect()
            await h_with.disconnect()
            await owner.aclose()


# =============================================================================
# Feature 7 Boundaries — Prober & Meta-Tools
# =============================================================================


async def test_t2_f7_01_to_05_prober_validation_foreign_isolation_timeout_u5_bridge(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T2-F7-01..05: Prober query bounds, foreign hostId isolation, error/U5/bridge."""
    tool_outputs: list[str] = []
    seen_tool_calls: set[tuple[str | None, str]] = set()

    def _responder(req: ChatRequest, idx: int) -> list[Any]:
        for m in req.messages:
            if m.role == Role.TOOL:
                key = (m.tool_call_id, m.content)
                if key not in seen_tool_calls:
                    seen_tool_calls.add(key)
                    tool_outputs.append(m.content)
        if idx == 0:
            return [
                ToolCallRequested(
                    ToolCall(id="b1", name="mesh__find_resource", arguments={"query": ""})
                ),
                done_event(),
            ]
        if idx == 1:
            return [
                ToolCallRequested(
                    ToolCall(id="b2", name="mesh__find_resource", arguments={"query": "q" * 65})
                ),
                done_event(),
            ]
        if idx == 2:
            return [
                ToolCallRequested(
                    ToolCall(
                        id="b3",
                        name="mesh__find_resource",
                        arguments={"query": "smarthome", "kind": "tool"},
                    )
                ),
                done_event(),
            ]
        if idx == 3:
            return [
                ToolCallRequested(
                    ToolCall(id="b4", name="mesh__status", arguments={"includeVitals": True})
                ),
                done_event(),
            ]
        return [TextDelta("Checked."), done_event()]

    def _err_handler(name: str, _args: dict[str, Any], _meta: dict[str, Any]) -> Any:
        if name == "host.status":
            return RpcErrorResponse(code=-32001, message="probe_failed")
        return {"status": "ok", "structuredContent": {"ok": True}}

    llm = ScriptedTextGenerator(responder=_responder)
    async with mesh_server_factory(generator=llm) as srv:
        owner_a = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        owner_b = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_a = await owner_a.bind_pet()
        # Turn U5 host_screen OFF to test T2-F7-04
        await owner_a.enable_consents(host_screen=False)

        bridge_tools = [
            make_tool_descriptor("host.status", family="host", risk="R0"),
            make_tool_descriptor("system.vitals", family="host", risk="R1", observes=True),
            make_tool_descriptor("smarthome.dim", family="app", risk="R1"),
        ]
        h_bridge = SimulatedXtpHost(platform="linux", tools=bridge_tools, tool_handler=_err_handler)
        h_foreign = SimulatedXtpHost(platform="macos")
        try:
            id_bridge = await h_bridge.connect_enroll(srv.ws_url, owner_a, "BridgeNode")
            id_foreign = await h_foreign.connect_enroll(srv.ws_url, owner_b, "ForeignNode")
            await owner_a.patch_host(str(id_bridge), {"schemaVersion": 1, "hostKind": "bridge"})
            await owner_a.grant_all_standard(str(id_bridge), max_risk="R1")

            await owner_a.stream_turn(pet_a, "find resources and status")
            assert len(tool_outputs) >= 4
            assert "invalid_args" in tool_outputs[0]
            assert "invalid_args" in tool_outputs[1]
            bridge_res = json.loads(tool_outputs[2])
            assert any(m["hostId"] == str(id_bridge) for m in bridge_res["matches"])

            status_res = json.loads(tool_outputs[3])
            row = next((h for h in status_res["hosts"] if h["hostId"] == str(id_bridge)), None)
            assert row is not None, f"Missing bridge host {id_bridge} in mesh__status response"
            assert row["probeStatus"] == "error"
            assert row["vitalsStatus"] == "denied"
            assert not any(c["name"] == "system.vitals" for c in h_bridge.received_tool_calls)

            # T2-F7-02: passing foreign hostId to mesh__status must not leak foreign telemetry
            llm.push_script(
                [
                    ToolCallRequested(
                        ToolCall(
                            id="b5",
                            name="mesh__status",
                            arguments={"hostIds": [str(id_foreign)], "includeVitals": False},
                        )
                    ),
                    done_event(),
                ]
            )
            llm.push_script([TextDelta("Done."), done_event()])
            await owner_a.stream_turn(pet_a, "status foreign")
            foreign_res = json.loads(tool_outputs[-1])
            assert all(h["hostId"] != str(id_foreign) for h in foreign_res.get("hosts", []))
        finally:
            await h_bridge.disconnect()
            await h_foreign.disconnect()
            await owner_a.aclose()
            await owner_b.aclose()


# =============================================================================
# Feature 8 Boundaries — Strict Validation, 404 Isolation & Audit Privacy
# =============================================================================


async def test_t2_f8_01_to_06_http_validation_404_isolation_and_audit_privacy(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T2-F8-01..06: PATCH/Turn/WS validation, 404 owner isolation, habit audit privacy."""
    llm = ScriptedTextGenerator(
        scripts=[
            [
                ToolCallRequested(ToolCall(id="au1", name="media__play_pause", arguments={})),
                done_event(),
            ],
            [TextDelta("Done."), done_event()],
        ]
    )
    async with mesh_server_factory(
        generator=llm,
        recognizer=FakeSpeechRecognizer(),
        synthesizer=FakeSpeechSynthesizer(),
    ) as srv:
        owner_a = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        owner_b = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_a = await owner_a.bind_pet()
        await owner_a.enable_consents()
        h1 = SimulatedXtpHost(platform="macos")
        h2 = SimulatedXtpHost(platform="windows")
        try:
            id1 = await h1.connect_enroll(srv.ws_url, owner_a, "Mac Nhà")
            id2 = await h2.connect_enroll(srv.ws_url, owner_a, "WinDesk")
            await owner_a.grant_all_standard(str(id1), max_risk="R1")

            # T2-F8-01: whitespace-only & >10 nicknames -> 422; "" clears place/wolMac to null
            for bad_body in (
                {"schemaVersion": 1, "alias": "   "},
                {"schemaVersion": 1, "place": "   "},
                {"schemaVersion": 1, "nicknames": ["ok", "   "]},
                {"schemaVersion": 1, "nicknames": [f"n{i}" for i in range(11)]},
                {"schemaVersion": 1, "wolMac": "ZZ:BB:CC:DD:EE:FF"},
                {"schemaVersion": 1, "hostKind": "robot"},
                {"schemaVersion": 1, "unknownField": 123},
            ):
                r_bad = await owner_a.patch_host(str(id1), bad_body)
                assert r_bad.status_code == 422, (bad_body, r_bad.text)

            # T2-F8-02: missing Idempotency-Key rejected
            r_no_idem = await owner_a.http.patch(
                f"/v1/hosts/{id1}",
                headers=owner_a.headers,
                json={"schemaVersion": 1, "place": "office"},
            )
            assert r_no_idem.status_code in (400, 422)

            # Set then clear place & wolMac via ""
            r_set = await owner_a.patch_host(
                str(id1),
                {"schemaVersion": 1, "place": "office", "wolMac": "AA:BB:CC:DD:EE:FF"},
            )
            assert r_set.status_code == 200
            r_clr = await owner_a.patch_host(
                str(id1),
                {"schemaVersion": 1, "place": "", "wolMac": ""},
            )
            assert r_clr.status_code == 200
            assert r_clr.json()["place"] is None
            assert r_clr.json()["wolMac"] is None

            # T2-F8-03: 409 folded collision on PATCH & 404 foreign owner
            r_409 = await owner_a.patch_host(str(id2), {"schemaVersion": 1, "alias": "mac nha"})
            assert r_409.status_code == 409
            assert r_409.json()["code"] == "conflict"

            r_404_patch = await owner_b.patch_host(str(id1), {"schemaVersion": 1, "alias": "Hack"})
            assert r_404_patch.status_code == 404

            # T2-F8-04: TurnBody & Voice WS hello/turn.commit reject bad clientSubnetHash & extras
            for bad_turn in (
                {"schemaVersion": 1, "text": "hi", "clientSubnetHash": "not-hex!"},
                {"schemaVersion": 1, "text": "hi", "clientSubnetHash": "abcd"},
                {"schemaVersion": 1, "text": "hi", "extraField": 1},
            ):
                r_t = await owner_a.post_turn_raw(pet_a, bad_turn)
                assert r_t.status_code == 422

            for bad_hello_extra in (
                {"clientSubnetHash": "not-hex!"},
                {"extraField": 1},
            ):
                async with websockets.connect(f"{srv.ws_url}/v1/voice") as bad_ws:
                    await bad_ws.send(
                        json.dumps(
                            {
                                "type": "hello",
                                "schemaVersion": 1,
                                "auth": {"token": owner_a.jwt},
                                "petId": pet_a,
                                "speak": False,
                                **bad_hello_extra,
                            }
                        )
                    )
                    hello_err = json.loads(await asyncio.wait_for(bad_ws.recv(), timeout=5.0))
                    assert hello_err["type"] == "error", (bad_hello_extra, hello_err)

            voice = await owner_a.connect_voice_ws(pet_a)
            try:
                for bad_commit in (
                    {"type": "turn.commit", "text": "hi", "clientSubnetHash": "not-hex!"},
                    {"type": "turn.commit", "text": "hi", "extraField": 1},
                ):
                    await voice.send_raw_json(bad_commit)
                    ws_err = await voice.recv_event(wait_s=5.0)
                    assert ws_err["type"] == "error", (bad_commit, ws_err)
            finally:
                await voice.close()

            # T2-F8-05: 404 on foreign stop & focus
            r_stop_404 = await owner_b.emergency_stop(host_id=str(id1))
            assert r_stop_404.status_code == 404
            r_focus_404 = await owner_b.set_focus(host_id=str(id1))
            assert r_focus_404.status_code == 404

            # T2-F8-06: habit deletion audit never logs contextKey
            await owner_a.stream_turn(pet_a, "pause", target_host_id=str(id1))
            habits = (await owner_a.list_habits(pet_id=pet_a)).json()["items"]
            assert len(habits) >= 1
            ctx_key = habits[0]["contextKey"]
            assert (await owner_b.delete_habit(habits[0]["habitId"])).status_code == 404
            assert (await owner_a.delete_habit(habits[0]["habitId"])).status_code == 200
            assert (await owner_a.clear_habits(pet_id=pet_a)).status_code == 200

            audits = await owner_a.audit_events(limit=50)
            del_audits = [e for e in audits if e["action"] == "host.habit_deleted"]
            assert len(del_audits) >= 2
            assert ctx_key not in json.dumps(del_audits)
        finally:
            await h1.disconnect()
            await h2.disconnect()
            await owner_a.aclose()
            await owner_b.aclose()


# =============================================================================
# Feature 9 Boundaries — XTP Gateway Rejection Codes
# =============================================================================


async def test_t2_f9_xtp_gateway_rejects_oversized_binary_paths_and_invalid_notifications(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T2-F9-01..05: XTP gateway enforces 512 KiB, 16/448 KiB SC, and strict schemas."""
    async with mesh_server_factory() as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        await owner.enable_consents(host_screen=True)

        # 1. Oversized > 512 KiB (524,289 bytes) -> 1009 frame_too_large
        h_big = SimulatedXtpHost(platform="linux")
        try:
            await h_big.connect_enroll(srv.ws_url, owner, "BigFrameHost")
            prefix = '{"jsonrpc":"2.0","id":888,"result":{"blob":"'
            suffix = '"}}'
            pad = 524_289 - len(prefix.encode("utf-8")) - len(suffix.encode("utf-8"))
            await h_big.send_raw_text(f"{prefix}{'a' * pad}{suffix}")
            code, reason = await h_big.wait_closed(wait_s=3.0)
            assert code == 1009
            assert reason == "frame_too_large"
        finally:
            await h_big.disconnect()

        # 2. Binary frame -> 1002 unsupported_protocol
        h_bin = SimulatedXtpHost(platform="linux")
        try:
            await h_bin.connect_enroll(srv.ws_url, owner, "BinHost")
            await h_bin.send_raw_bytes(b"\x00\x01\x02")
            code, _ = await h_bin.wait_closed(wait_s=3.0)
            assert code == 1002
        finally:
            await h_bin.disconnect()

        # 3. structuredContent ceilings: 16,384 B ordinary vs 458,752 B media (cap vs +1 B)
        sc_mode = "ordinary_ok"

        def _sc_handler(name: str, _args: dict[str, Any], _meta: dict[str, Any]) -> dict[str, Any]:
            if name == "host.status":
                pad_len = 16_384 - len(b'{"blob":""}')
                if sc_mode == "ordinary_over":
                    pad_len += 1
                return {"status": "ok", "structuredContent": {"blob": "x" * pad_len}}
            if name == "screen.capture":
                pad_len = 458_752 - len(b'{"dataBase64":""}')
                if sc_mode == "media_over":
                    pad_len += 1
                return {"status": "ok", "structuredContent": {"dataBase64": "A" * pad_len}}
            return {"status": "ok", "structuredContent": {"ok": True}}

        h_sc = SimulatedXtpHost(
            platform="linux",
            tools=[
                make_tool_descriptor("host.status", family="host", risk="R0", platform="linux"),
                make_tool_descriptor(
                    "screen.capture", family="screen", risk="R1", observes=True, platform="linux"
                ),
            ],
            tool_handler=_sc_handler,
        )
        try:
            id_sc = await h_sc.connect_enroll(srv.ws_url, owner, "ScLimitsHost")
            await owner.set_grants(
                str(id_sc),
                [
                    {"family": "host", "maxRisk": "R1", "enabled": True},
                    {"family": "screen", "maxRisk": "R1", "enabled": True},
                ],
            )
            sc_mode = "ordinary_ok"
            r_ord_ok = await owner.invoke_tool_dev(str(id_sc), "host.status", {})
            assert r_ord_ok.status_code == 200 and r_ord_ok.json()["status"] == "ok"

            sc_mode = "ordinary_over"
            r_ord_over = await owner.invoke_tool_dev(str(id_sc), "host.status", {})
            assert r_ord_over.status_code != 200 or r_ord_over.json()["status"] != "ok"

            sc_mode = "media_ok"
            r_med_ok = await owner.invoke_tool_dev(str(id_sc), "screen.capture", {})
            assert r_med_ok.status_code == 200 and r_med_ok.json()["status"] == "ok"

            sc_mode = "media_over"
            r_med_over = await owner.invoke_tool_dev(str(id_sc), "screen.capture", {})
            assert r_med_over.status_code != 200 or r_med_over.json()["status"] != "ok"
        finally:
            await h_sc.disconnect()

        # 4. Invalid mesh notifications -> 1002 unsupported_protocol
        for bad_notif in (
            {
                "jsonrpc": "2.0",
                "method": "xuna/resources",
                "params": {
                    "schemaVersion": 1,
                    "installedApps": [],
                    "workspaces": ["/Users/alice/repo"],
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "xuna/resources",
                "params": {
                    "schemaVersion": 1,
                    "installedApps": [],
                    "workspaces": ["dir\\repo"],
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "xuna/resources",
                "params": {
                    "schemaVersion": 1,
                    "installedApps": [f"app_{i:03d}" for i in range(257)],
                    "workspaces": [],
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "xuna/presence",
                "params": {
                    "schemaVersion": 1,
                    "idleSeconds": -1,
                    "screenLocked": False,
                    "mediaPlaying": False,
                    "updatedAt": "2026-11-20T16:00:00Z",
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "xuna/presence",
                "params": {
                    "schemaVersion": 1,
                    "idleSeconds": 0,
                    "screenLocked": False,
                    "mediaPlaying": False,
                    "batteryPercent": -1,
                    "updatedAt": "2026-11-20T16:00:00Z",
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "xuna/presence",
                "params": {
                    "schemaVersion": 1,
                    "idleSeconds": 0,
                    "screenLocked": False,
                    "mediaPlaying": False,
                    "batteryPercent": 101,
                    "updatedAt": "2026-11-20T16:00:00Z",
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "xuna/presence",
                "params": {
                    "schemaVersion": 1,
                    "idleSeconds": 0,
                    "screenLocked": False,
                    "mediaPlaying": False,
                    "subnetHash": "not-hex!",
                    "updatedAt": "2026-11-20T16:00:00Z",
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "xuna/presence",
                "params": {
                    "schemaVersion": 1,
                    "idleSeconds": 0,
                    "screenLocked": False,
                    "mediaPlaying": False,
                    "extraKey": 1,
                    "updatedAt": "2026-11-20T16:00:00Z",
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "xuna/focusRequest",
                "params": {"schemaVersion": 1, "source": "cli"},
            },
            {
                "jsonrpc": "2.0",
                "method": "xuna/focusRequest",
                "params": {"source": "hotkey"},
            },
        ):
            h_bad = SimulatedXtpHost(platform="linux")
            try:
                await h_bad.connect_enroll(srv.ws_url, owner, f"Bad-{uuid.uuid4().hex[:6]}")
                await h_bad.send_raw_json(bad_notif)
                code, _ = await h_bad.wait_closed(wait_s=3.0)
                assert code == 1002, f"Expected 1002 for {bad_notif}, got {code}"
            finally:
                await h_bad.disconnect()

        # Verify websockets raw connect without subprotocol also closes 1002
        async with websockets.connect(f"{srv.ws_url}/v1/hosts/connect") as raw_ws:
            await asyncio.sleep(0.05)
            try:
                await raw_ws.recv()
            except websockets.ConnectionClosed as closed:
                assert closed.code == 1002
        await owner.aclose()


# =============================================================================
# Feature 12 Boundaries — WoL, Battery Advisor, R3 MissionEnvelope
# =============================================================================


async def test_t2_f12_04_wol_missing_mac_fails_closed_and_max_3_steps_cap(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M4] T2-F12-04: Missing wol_mac fails closed; deferred WoL queue enforces max-3-steps cap.

    Note: 180s DeferredWakeStep TTL expiry is not fast-forwarded over HTTP because public Settings
    exposes no TTL override.
    """
    llm = ScriptedTextGenerator(
        responder=lambda req, idx: (
            [
                ToolCallRequested(
                    ToolCall(id=f"w{idx}", name="app__launch", arguments={"id": f"app_{idx}"})
                ),
                done_event(),
            ]
            if not any(m.role == Role.TOOL for m in req.messages)
            else [TextDelta("Handled."), done_event()]
        )
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        h_off = SimulatedXtpHost(platform="windows")
        h_wol = SimulatedXtpHost(platform="windows")
        try:
            id_off = await h_off.connect_enroll(srv.ws_url, owner, "NoWolWin")
            await owner.grant_all_standard(str(id_off), max_risk="R2")
            await h_off.disconnect()

            turn = await owner.stream_turn(pet_id, "launch app_x", target_host_id=str(id_off))
            assert turn.first("host_wake_request") is None
            req = turn.first("action_target_required")
            assert req is not None and req["reasonCode"] == "host_offline"
            assert req["candidates"][0]["wolAvailable"] is False
            assert req["candidates"][0]["wolMac"] is None

            # Max 3 queued deferred wake steps per owner: 4th queued step is bounded/rejected
            id_wol = await h_wol.connect_enroll(srv.ws_url, owner, "WolCapWin")
            await owner.grant_all_standard(str(id_wol), max_risk="R2")
            await owner.patch_host(str(id_wol), {"schemaVersion": 1, "wolMac": "AA:BB:CC:DD:EE:22"})
            await h_wol.disconnect()

            for step_idx in range(1, 4):
                t_q = await owner.stream_turn(
                    pet_id, f"launch app_{step_idx}", target_host_id=str(id_wol)
                )
                wake_ev = t_q.first("host_wake_request")
                assert wake_ev is not None, f"Step {step_idx} expected host_wake_request"
                assert wake_ev["macAddress"] == "AA:BB:CC:DD:EE:22"

            t_4th = await owner.stream_turn(pet_id, "launch app_4", target_host_id=str(id_wol))
            req_4th = t_4th.first("action_target_required")
            assert req_4th is not None and req_4th["reasonCode"] == "host_offline"
            # 4th step must exceed the 3-step queue cap (must not enqueue a 4th wake step)
            assert t_4th.first("host_wake_request") is None
        finally:
            await h_off.disconnect()
            await h_wol.disconnect()
            await owner.aclose()


async def test_t2_f12_05_and_06_battery_advisor_15pct_boundary_and_r3_mission_signature(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M4] T2-F12-05/06: 15% unplugged advised (plugged/null omitted); R1+R3 needs P-256."""

    def _responder(req: ChatRequest, idx: int) -> list[Any]:
        if any(m.role == Role.TOOL for m in req.messages):
            return [TextDelta("Finished."), done_event()]
        last = req.messages[-1].content
        if "check battery" in last:
            return [
                ToolCallRequested(
                    ToolCall(
                        id=f"bat_{idx}",
                        name="mesh__status",
                        arguments={"includeVitals": True},
                    )
                ),
                done_event(),
            ]
        tool_names = {t.name for t in req.tools}
        if "mesh__mission_r1_r3" in tool_names:
            return [
                ToolCallRequested(
                    ToolCall(
                        id=f"mis_{idx}",
                        name="mesh__mission_r1_r3",
                        arguments={"targetHost": "LowLaptop"},
                    )
                ),
                done_event(),
            ]
        return [
            ToolCallRequested(
                ToolCall(
                    id=f"cu_{idx}",
                    name="cu__session_start",
                    arguments={"goal": "Heavy desktop workflow"},
                )
            ),
            done_event(),
        ]

    llm = ScriptedTextGenerator(responder=_responder)
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        phone_resp = await owner.self_enroll_mobile(alias="OwnerPhone", platform="ios")
        assert phone_resp.status_code == 200, phone_resp.text

        h_low = SimulatedXtpHost(
            platform="macos",
            tools=make_standard_mesh_tools(platform="macos", include_cu=True),
        )
        h_ac = SimulatedXtpHost(
            platform="windows",
            tools=make_standard_mesh_tools(platform="windows", include_cu=True),
        )
        h_nobatt = SimulatedXtpHost(
            platform="linux",
            tools=make_standard_mesh_tools(platform="linux", include_cu=True),
        )
        try:
            id_low = await h_low.connect_enroll(srv.ws_url, owner, "LowLaptop")
            id_ac = await h_ac.connect_enroll(srv.ws_url, owner, "PluggedPC")
            id_nobatt = await h_nobatt.connect_enroll(srv.ws_url, owner, "NoBattNode")
            await owner.patch_host(str(id_nobatt), {"schemaVersion": 1, "hostKind": "server"})
            await owner.grant_all_standard(str(id_low), max_risk="R3")
            await owner.grant_all_standard(str(id_ac), max_risk="R3")
            await owner.grant_all_standard(str(id_nobatt), max_risk="R3")

            await h_low.send_presence(
                idle_seconds=5,
                screen_locked=False,
                media_playing=False,
                battery_percent=15,
                is_charging=False,
            )
            await h_ac.send_presence(
                idle_seconds=10,
                screen_locked=False,
                media_playing=False,
                battery_percent=15,
                is_charging=True,
            )
            await h_nobatt.send_presence(
                idle_seconds=0,
                screen_locked=False,
                media_playing=False,
                include_battery=False,
            )

            # T2-F12-05: 15% unplugged triggers advisory; 15% charging & null battery do NOT
            await owner.stream_turn(pet_id, "check battery")
            tool_msgs = [m.content for r in llm.requests for m in r.messages if m.role == Role.TOOL]
            assert tool_msgs
            status_payload = json.loads(tool_msgs[-1])
            by_id = {h["hostId"]: h for h in status_payload["hosts"]}
            assert by_id[str(id_low)].get("lowBattery") is True
            assert by_id[str(id_ac)].get("lowBattery") is not True
            assert by_id[str(id_nobatt)].get("lowBattery") is not True
            assert by_id[str(id_nobatt)]["presence"]["batteryPercent"] is None
            advisories = status_payload.get("batteryAdvisories") or []
            assert advisories and any("LowLaptop" in a for a in advisories)
            assert not any("PluggedPC" in a.split(";")[0] for a in advisories)

            # T2-F12-06: Multi-step R1+R3 MissionEnvelope / CU task requires phone P-256 signature
            approval_statuses: list[int] = []

            async def _test_r3_sig(ev: str, data: dict[str, Any]) -> None:
                if ev == "action_pending":
                    bad = await owner.decide_approval(data["approvalId"], "approve", sign_r3=False)
                    approval_statuses.append(bad.status_code)
                    good = await owner.decide_approval(
                        data["approvalId"],
                        "approve",
                        host_id=data["hostId"],
                        args_sha256_hex=data["argsSha256"],
                        expires_at=data["expiresAt"],
                        sign_r3=True,
                    )
                    assert good.status_code == 200, good.text

            outcome = await owner.stream_turn(
                pet_id,
                "run heavy task",
                target_host_id=str(id_low),
                on_event=_test_r3_sig,
            )
            assert approval_statuses == [400]
            pending = outcome.first("action_pending")
            assert pending is not None and pending["risk"] == "r3"
            env = pending.get("missionEnvelope")
            assert env is not None and env.get("goal")
            # Expect Battery Advisor warning/alternative on LowLaptop (15% unplugged), not PluggedPC
            events_json = json.dumps(outcome.events)
            assert "LowLaptop" in events_json
            assert "PluggedPC" in events_json or "NoBattNode" in events_json
        finally:
            await h_low.disconnect()
            await h_ac.disconnect()
            await h_nobatt.disconnect()
            await owner.aclose()


def start_mesh_server_with_gen(database_url: str, generator: Any) -> Any:
    from tests.e2e_mesh.conftest import start_mesh_server

    return start_mesh_server(database_url, generator=generator)
