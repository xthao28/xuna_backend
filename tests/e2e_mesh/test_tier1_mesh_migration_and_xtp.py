"""Tier 1 Backend E2E Mesh Tests: Migration 0022, XTP v1 Ingress, HostTwin & Mesh APIs.

Opaque-box tests exercising:
- Feature 3: Migration 0022 & schema updates (`hosts` spatial columns, `cu_sessions` lanes,
  `host_routing_habits`, `host_defaults`)
- Feature 4: Ephemeral RAM-only `HostTwin` & dynamic `[HOST_ROSTER]` prompt injection
- Feature 8: HTTP / SSE / WS Mesh APIs (`PATCH /v1/hosts/{id}`, `GET /v1/hosts`,
  `/v1/actions/habits`, `/v1/actions/focus`, scoped `/v1/actions/stop`, spatial events)
- Feature 9 (Backend Ingress): `golden_vectors.json`, `mesh_frames.json` (`linux`,
  `xuna/presence`, `xuna/resources`, `xuna/focusRequest`), `512 KiB` frame ceiling.
"""

import asyncio
import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from tests.e2e_mesh.conftest import FIXTURES_DIR, fetch_scalar, pg_dsn, run_alembic
from tests.e2e_mesh.xtp_sim import (
    ScriptedTextGenerator,
    SimulatedOwnerClient,
    SimulatedXtpHost,
    done_event,
    load_mesh_frames_fixture,
    make_standard_mesh_tools,
    verify_golden_vectors,
)
from xuna_backend.modules.cognition.domain.types import (
    Role,
    TextDelta,
    ToolCall,
    ToolCallRequested,
)

pytestmark = [pytest.mark.postgres, pytest.mark.anyio]


# =============================================================================
# Feature 3 — Migration 0022 & Schema Updates
# =============================================================================


async def test_t1_f3_01_migration_0022_schema_indexes_and_roundtrip(clean_mesh_db: str) -> None:
    """[M1/M2] T1-F3-01: Single head 0022, schema tables/indexes, downgrade->upgrade round-trip."""
    dsn = pg_dsn(clean_mesh_db)
    try:
        with psycopg.connect(dsn) as conn:
            rev = fetch_scalar(conn, "SELECT version_num FROM alembic_version")
            assert (
                rev is not None and int(str(rev)) >= 22
            ), f"Expected Alembic head >= '0022', got {rev!r}"

            host_cols = {
                r[0]
                for r in conn.execute(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = 'hosts'
                    """
                ).fetchall()
            }
            assert {"host_kind", "place", "nicknames", "alias_folded", "wol_mac"} <= host_cols

            indexes = {
                r[0]
                for r in conn.execute(
                    """
                    SELECT indexname FROM pg_indexes
                    WHERE schemaname = 'public'
                      AND tablename IN ('hosts', 'cu_sessions', 'host_routing_habits',
                                        'host_defaults')
                    """
                ).fetchall()
            }
            assert "uq_hosts_owner_alias_folded_active" in indexes
            assert "uq_cu_sessions_owner_host_active" in indexes
            assert "ix_cu_sessions_owner_host" in indexes
            assert "uq_cu_sessions_owner_active" not in indexes

            tables = {
                r[0]
                for r in conn.execute(
                    """
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                    """
                ).fetchall()
            }
            assert {"host_routing_habits", "host_defaults"} <= tables

        run_alembic(clean_mesh_db, "downgrade", "base")
        run_alembic(clean_mesh_db, "upgrade", "head")

        with psycopg.connect(dsn) as conn:
            rev_after = fetch_scalar(conn, "SELECT version_num FROM alembic_version")
            assert rev_after is not None and int(str(rev_after)) >= 22
    finally:
        run_alembic(clean_mesh_db, "upgrade", "head")


async def test_t1_f3_02_migration_0022_hosts_spatial_backfill_from_0021(
    clean_mesh_db: str,
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F3-02: Pre-0022 hosts backfill host_kind, alias_folded, nicknames on upgrade."""
    dsn = pg_dsn(clean_mesh_db)
    owner_uuid = uuid.uuid4()
    mac_host_id = uuid.uuid4()
    ios_host_id = uuid.uuid4()
    now = datetime(2026, 11, 20, 10, 0, 0, tzinfo=UTC)
    pubkey = b"\x04" + bytes(range(1, 65))

    try:
        run_alembic(clean_mesh_db, "downgrade", "0021")
        with psycopg.connect(dsn) as conn:
            conn.execute(
                """
                INSERT INTO owners (owner_id, issuer, subject, created_at)
                VALUES (%s, %s, %s, %s)
                """,
                (owner_uuid, "https://auth.test.xuna", f"backfill-{owner_uuid}", now),
            )
            conn.execute(
                """
                INSERT INTO hosts (
                    host_id, owner_id, alias, platform, hostname, os_version,
                    agent_version, public_key, status, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s),
                         (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    mac_host_id,
                    owner_uuid,
                    "Mac Nhà",
                    "macos",
                    "mac-desk",
                    "15.0",
                    "0.1.0",
                    pubkey,
                    "active",
                    now,
                    ios_host_id,
                    owner_uuid,
                    "Phone",
                    "ios",
                    "phone-1",
                    "18.0",
                    "0.1.0",
                    pubkey,
                    "active",
                    now,
                ),
            )
            conn.commit()

        run_alembic(clean_mesh_db, "upgrade", "head")

        with psycopg.connect(dsn) as conn:
            rev = fetch_scalar(conn, "SELECT version_num FROM alembic_version")
            assert rev is not None and int(str(rev)) >= 22
            rows = conn.execute(
                """
                SELECT host_id, host_kind, alias_folded, nicknames, place, wol_mac
                FROM hosts WHERE owner_id = %s
                """,
                (owner_uuid,),
            ).fetchall()
            by_id = {r[0]: r for r in rows}
            assert by_id[mac_host_id][1] == "desktop"
            assert by_id[mac_host_id][2] == "mac nha"
            assert by_id[mac_host_id][3] == []
            assert by_id[mac_host_id][4] is None
            assert by_id[mac_host_id][5] is None

            assert by_id[ios_host_id][1] == "mobile"
            assert by_id[ios_host_id][2] == "phone"

        async with mesh_server_factory() as srv:
            owner = SimulatedOwnerClient(
                base_url=srv.base_url,
                ws_url=srv.ws_url,
                issuer=srv.issuer,
                subject=f"backfill-{owner_uuid}",
            )
            try:
                items = await owner.list_hosts()
                by_alias = {item["alias"]: item for item in items}
                assert by_alias["Mac Nhà"]["hostKind"] == "desktop"
                assert by_alias["Phone"]["hostKind"] == "mobile"
                assert by_alias["Mac Nhà"]["nicknames"] == []
                assert by_alias["Mac Nhà"]["place"] is None
                assert by_alias["Mac Nhà"]["wolMac"] is None
            finally:
                await owner.aclose()
    finally:
        run_alembic(clean_mesh_db, "upgrade", "head")


async def test_t1_f3_03_cu_sessions_concurrent_per_host_lanes(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F3-03: Two hosts maintain concurrent active CU lanes under one owner."""
    async with mesh_server_factory() as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        await owner.enable_consents()
        host_a = SimulatedXtpHost(
            platform="macos",
            tools=make_standard_mesh_tools(platform="macos", include_cu=True),
        )
        host_b = SimulatedXtpHost(
            platform="windows",
            tools=make_standard_mesh_tools(platform="windows", include_cu=True),
        )
        try:
            id_a = await host_a.connect_enroll(srv.ws_url, owner, "AlphaMac")
            id_b = await host_b.connect_enroll(srv.ws_url, owner, "BetaWin")
            await owner.grant_all_standard(str(id_a), max_risk="R3")
            await owner.grant_all_standard(str(id_b), max_risk="R3")

            for hid, goal in ((id_a, "Task A"), (id_b, "Task B")):
                inv_resp = await owner.invoke_tool_dev(str(hid), "cu.session_start", {"goal": goal})
                assert inv_resp.status_code == 200, inv_resp.text
                assert inv_resp.json()["status"] == "ok"

            with psycopg.connect(pg_dsn(srv.database_url)) as conn:
                active_hosts = {
                    str(r[0])
                    for r in conn.execute(
                        "SELECT host_id FROM cu_sessions WHERE active = true"
                    ).fetchall()
                }
            assert active_hosts == {str(id_a), str(id_b)}

            stop_resp = await owner.emergency_stop(host_id=str(id_a))
            assert stop_resp.status_code == 200, stop_resp.text
            assert stop_resp.json().get("hostId") == str(id_a)

            with psycopg.connect(pg_dsn(srv.database_url)) as conn:
                remaining_hosts = {
                    str(r[0])
                    for r in conn.execute(
                        "SELECT host_id FROM cu_sessions WHERE active = true"
                    ).fetchall()
                }
            assert remaining_hosts == {str(id_b)}
        finally:
            await host_a.disconnect()
            await host_b.disconnect()
            await owner.aclose()


async def test_t1_f3_04_host_routing_habits_persistence_and_cascade_delete(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F3-04: Habits persist in host_routing_habits and cascade-delete with host."""
    llm = ScriptedTextGenerator(
        scripts=[
            [
                ToolCallRequested(
                    ToolCall(id="h1", name="media__play_pause", arguments={}),
                ),
                done_event(),
            ],
            [TextDelta("Paused."), done_event()],
        ]
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        host_a = SimulatedXtpHost(platform="macos")
        try:
            id_a = await host_a.connect_enroll(srv.ws_url, owner, "WorkstationA")
            await owner.grant_all_standard(str(id_a), max_risk="R2")

            outcome = await owner.stream_turn(
                pet_id,
                "pause media",
                target_host_id=str(id_a),
            )
            res = outcome.first("action_result")
            assert res is not None and res["status"] == "ok"

            habits_resp = await owner.list_habits(pet_id=pet_id)
            assert habits_resp.status_code == 200, habits_resp.text
            items = habits_resp.json()["items"]
            assert len(items) >= 1
            assert items[0]["gamma"] == pytest.approx(0.85)
            assert items[0]["hostId"] == str(id_a)

            # Verify DB ON DELETE CASCADE when host row is removed
            with psycopg.connect(pg_dsn(srv.database_url)) as conn:
                conn.execute("DELETE FROM hosts WHERE host_id = %s", (id_a,))
                conn.commit()
                count = fetch_scalar(
                    conn,
                    "SELECT COUNT(*) FROM host_routing_habits WHERE host_id = %s",
                    (id_a,),
                )
                assert count == 0
        finally:
            await host_a.disconnect()
            await owner.aclose()


async def test_t1_f3_05_host_defaults_table_and_unique_scope_key(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F3-05: host_defaults persists scoped defaults with unique (owner, pet, scope)."""
    async with mesh_server_factory() as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        host_a = SimulatedXtpHost(platform="macos")
        host_b = SimulatedXtpHost(platform="windows")
        try:
            id_a = await host_a.connect_enroll(srv.ws_url, owner, "NodeA")
            id_b = await host_b.connect_enroll(srv.ws_url, owner, "NodeB")
            now = datetime.now(UTC)

            with psycopg.connect(pg_dsn(srv.database_url)) as conn:
                owner_uuid = fetch_scalar(
                    conn, "SELECT owner_id FROM hosts WHERE host_id = %s", (id_a,)
                )
                assert owner_uuid is not None
                conn.execute(
                    """
                    INSERT INTO host_defaults (
                        default_id, owner_id, pet_id, scope_key, host_id, source,
                        created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (uuid.uuid4(), owner_uuid, pet_id, "media", id_a, "confirmed", now, now),
                )
                conn.commit()

                with pytest.raises(psycopg.errors.UniqueViolation):
                    conn.execute(
                        """
                        INSERT INTO host_defaults (
                            default_id, owner_id, pet_id, scope_key, host_id, source,
                            created_at, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (uuid.uuid4(), owner_uuid, pet_id, "media", id_b, "confirmed", now, now),
                    )
                    conn.commit()
                conn.rollback()

                row = conn.execute(
                    """
                    SELECT host_id, source FROM host_defaults
                    WHERE owner_id = %s AND pet_id = %s AND scope_key = 'media'
                    """,
                    (owner_uuid, pet_id),
                ).fetchone()
                assert row is not None
                assert row[0] == id_a
                assert row[1] == "confirmed"
        finally:
            await host_a.disconnect()
            await host_b.disconnect()
            await owner.aclose()


# =============================================================================
# Feature 4 — HostTwin & Dynamic Host Roster
# =============================================================================


async def test_t1_f4_01_host_twin_populated_in_ram_on_presence_and_resources(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F4-01: xuna/presence & xuna/resources populate live HostTwin in RAM only."""
    async with mesh_server_factory() as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        host_a = SimulatedXtpHost(platform="macos")
        try:
            id_a = await host_a.connect_enroll(srv.ws_url, owner, "DeskMac")
            await host_a.send_presence(
                idle_seconds=12,
                screen_locked=False,
                media_playing=True,
                battery_percent=78,
                is_charging=True,
                subnet_hash="a1b2c3d4e5f60718293a4b5c6d7e8f90",
                foreground_app_id="code_editor_x",
            )
            await host_a.send_resources(
                installed_apps=["code_editor_x", "audio_player_x"],
                workspaces=["workspace_alpha"],
                capabilities_version="r1_0001",
            )

            items = await owner.list_hosts()
            assert len(items) == 1
            item = items[0]
            assert item["hostId"] == str(id_a)
            assert item["online"] is True
            pres = item["presence"]
            assert pres is not None
            assert pres["idleSeconds"] == 12
            assert pres["screenLocked"] is False
            assert pres["mediaPlaying"] is True
            assert pres["batteryPercent"] == 78
            assert pres["isCharging"] is True
            assert pres["subnetHash"] == "a1b2c3d4e5f60718293a4b5c6d7e8f90"
            assert pres["foregroundAppId"] == "code_editor_x"

            with psycopg.connect(pg_dsn(srv.database_url)) as conn:
                cols = {
                    r[0]
                    for r in conn.execute(
                        """
                        SELECT column_name FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = 'hosts'
                        """
                    ).fetchall()
                }
                assert "presence" not in cols
                assert "idle_seconds" not in cols
                assert "foreground_app_id" not in cols
        finally:
            await host_a.disconnect()
            await owner.aclose()


async def test_t1_f4_02_host_twin_evicted_immediately_on_ws_disconnect(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F4-02: Disconnecting XTP WS immediately clears HostTwin (presence == None)."""
    async with mesh_server_factory() as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        host_a = SimulatedXtpHost(platform="windows")
        try:
            await host_a.connect_enroll(srv.ws_url, owner, "WorkWin")
            await host_a.send_presence(
                idle_seconds=4,
                screen_locked=False,
                media_playing=True,
                include_battery=False,
            )
            online_items = await owner.list_hosts()
            assert online_items[0]["online"] is True
            assert online_items[0]["presence"] is not None

            await host_a.disconnect()

            offline_items = await owner.list_hosts()
            assert offline_items[0]["online"] is False
            assert offline_items[0]["presence"] is None
        finally:
            await host_a.disconnect()
            await owner.aclose()


async def test_t1_f4_03_shared_owner_focus_state_across_desktop_xtp_and_mobile_http(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F4-03: Desktop xuna/focusRequest and Mobile POST /v1/actions/focus share pin."""
    async with mesh_server_factory() as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        host_a = SimulatedXtpHost(platform="macos")
        host_b = SimulatedXtpHost(platform="windows")
        try:
            id_a = await host_a.connect_enroll(srv.ws_url, owner, "HostAlpha")
            id_b = await host_b.connect_enroll(srv.ws_url, owner, "HostBeta")

            await host_a.send_focus_request(source="hotkey", follow_me=True)

            hosts_after_hotkey = {h["hostId"]: h for h in await owner.list_hosts()}
            assert hosts_after_hotkey[str(id_a)]["focused"] is True
            assert hosts_after_hotkey[str(id_b)]["focused"] is False

            focus_resp = await owner.set_focus(host_id=str(id_b), follow_me=False)
            assert focus_resp.status_code == 200, focus_resp.text
            focus_body = focus_resp.json()
            assert focus_body["focusedHostId"] == str(id_b)
            assert focus_body["followMe"] is False

            hosts_after_http = {h["hostId"]: h for h in await owner.list_hosts()}
            assert hosts_after_http[str(id_a)]["focused"] is False
            assert hosts_after_http[str(id_b)]["focused"] is True
        finally:
            await host_a.disconnect()
            await host_b.disconnect()
            await owner.aclose()


async def test_t1_f4_04_dynamic_host_roster_prompt_block_built_from_active_hosts(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F4-04: [HOST_ROSTER] dynamically reflects online/offline hosts and twins."""
    llm = ScriptedTextGenerator(scripts=[[TextDelta("Ready."), done_event()]])
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        host_a = SimulatedXtpHost(platform="macos")
        host_b = SimulatedXtpHost(platform="windows")
        try:
            id_a = await host_a.connect_enroll(srv.ws_url, owner, "StudioMac")
            id_b = await host_b.connect_enroll(srv.ws_url, owner, "HomePC")

            patch_a = await owner.patch_host(
                str(id_a),
                {"schemaVersion": 1, "place": "office", "nicknames": ["desk_a"]},
            )
            assert patch_a.status_code == 200, patch_a.text
            patch_b = await owner.patch_host(
                str(id_b),
                {"schemaVersion": 1, "place": "home"},
            )
            assert patch_b.status_code == 200, patch_b.text

            await host_a.send_presence(
                idle_seconds=9,
                screen_locked=False,
                media_playing=True,
                battery_percent=82,
                is_charging=False,
            )
            await host_a.send_resources(
                installed_apps=["code_editor_x"],
                workspaces=["workspace_alpha", "project_beta"],
            )
            await host_b.disconnect()

            await owner.stream_turn(pet_id, "check my hosts")
            assert len(llm.requests) >= 1
            sys_msgs = [m.content for m in llm.requests[0].messages if m.role == Role.SYSTEM]
            combined = "\n".join(sys_msgs)
            assert "[HOST_ROSTER]" in combined
            assert "[/HOST_ROSTER]" in combined
            assert "focusMode:" in combined
            assert "StudioMac" in combined
            assert "HomePC" in combined
            assert "state=online" in combined
            assert "state=offline" in combined
            assert "place=office" in combined
            assert "place=home" in combined
            assert "appsCount=1" in combined
            assert "workspacesCount=2" in combined
        finally:
            await host_a.disconnect()
            await host_b.disconnect()
            await owner.aclose()


async def test_t1_f4_05_host_twin_evicted_and_roster_updated_on_revoke_host(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F4-05: Revoking an online host closes its WS, evicts HostTwin, drops roster."""
    llm = ScriptedTextGenerator(scripts=[[TextDelta("Checked."), done_event()]])
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        host_a = SimulatedXtpHost(platform="macos")
        host_b = SimulatedXtpHost(platform="windows")
        try:
            id_a = await host_a.connect_enroll(srv.ws_url, owner, "RevokeTarget")
            await host_b.connect_enroll(srv.ws_url, owner, "KeptHost")
            await host_a.send_presence(idle_seconds=5, screen_locked=False, media_playing=False)

            del_resp = await owner.revoke_host(str(id_a))
            assert del_resp.status_code == 204
            await host_a.wait_closed(wait_s=3.0)

            items = await owner.list_hosts()
            assert [h["alias"] for h in items] == ["KeptHost"]

            await owner.stream_turn(pet_id, "list hosts")
            sys_prompt = "\n".join(
                m.content for m in llm.requests[0].messages if m.role == Role.SYSTEM
            )
            assert "RevokeTarget" not in sys_prompt
            assert "KeptHost" in sys_prompt
        finally:
            await host_a.disconnect()
            await host_b.disconnect()
            await owner.aclose()


# =============================================================================
# Feature 8 — Backend HTTP / SSE / WS Mesh APIs
# =============================================================================


async def test_t1_f8_01_patch_host_updates_spatial_fields_idempotent_and_audits(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F8-01: PATCH /v1/hosts/{id} updates spatial metadata, replays idem, audits."""
    async with mesh_server_factory() as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        host_a = SimulatedXtpHost(platform="macos")
        try:
            id_a = await host_a.connect_enroll(srv.ws_url, owner, "OldAlias")
            idem_key = str(uuid.uuid4())
            payload = {
                "schemaVersion": 1,
                "alias": "Mac Studio",
                "place": "office",
                "nicknames": ["mac nha", "desk_01"],
                "hostKind": "desktop",
                "wolMac": "AA:BB:CC:DD:EE:FF",
            }
            resp1 = await owner.patch_host(str(id_a), payload, idempotency_key=idem_key)
            assert resp1.status_code == 200, resp1.text
            body1 = resp1.json()
            assert body1["alias"] == "Mac Studio"
            assert body1["place"] == "office"
            assert body1["nicknames"] == ["mac nha", "desk_01"]
            assert body1["hostKind"] == "desktop"
            assert body1["wolMac"] == "AA:BB:CC:DD:EE:FF"

            resp2 = await owner.patch_host(str(id_a), payload, idempotency_key=idem_key)
            assert resp2.status_code == 200
            assert resp2.json() == body1

            events = await owner.audit_events(limit=50)
            updated_events = [e for e in events if e["action"] == "host.updated"]
            assert len(updated_events) == 1
            meta = updated_events[0]["metadata"]
            assert meta.get("hostId") == str(id_a)
            assert "fieldsCount" in meta
            assert "Mac Studio" not in json.dumps(meta)
        finally:
            await host_a.disconnect()
            await owner.aclose()


async def test_t1_f8_02_get_hosts_additive_backwards_compatible_shape(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F8-02: GET /v1/hosts keeps legacy keys and adds spatial/presence fields."""
    async with mesh_server_factory() as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        host_a = SimulatedXtpHost(platform="macos")
        host_b = SimulatedXtpHost(platform="windows")
        try:
            await host_a.connect_enroll(srv.ws_url, owner, "OnlineHost")
            await host_b.connect_enroll(srv.ws_url, owner, "OfflineHost")
            await host_a.send_presence(idle_seconds=3, screen_locked=False, media_playing=False)
            await host_b.disconnect()

            items = {h["alias"]: h for h in await owner.list_hosts()}
            legacy_keys = {
                "hostId",
                "alias",
                "platform",
                "hostname",
                "osVersion",
                "agentVersion",
                "status",
                "online",
                "keyFingerprint",
                "publicKey",
                "lanFingerprint",
                "lanPort",
                "createdAt",
                "lastSeenAt",
                "grants",
                "tools",
            }
            additive_keys = {"hostKind", "place", "nicknames", "wolMac", "focused", "presence"}
            for item in items.values():
                assert legacy_keys <= set(item.keys())
                assert additive_keys <= set(item.keys())

            assert items["OnlineHost"]["online"] is True
            assert items["OnlineHost"]["presence"] is not None
            assert items["OfflineHost"]["online"] is False
            assert items["OfflineHost"]["presence"] is None
        finally:
            await host_a.disconnect()
            await host_b.disconnect()
            await owner.aclose()


async def test_t1_f8_03_get_and_delete_actions_habits_and_audit_privacy(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F8-03: GET/DELETE /v1/actions/habits returns score/why and audits safely."""
    llm = ScriptedTextGenerator(
        scripts=[
            [
                ToolCallRequested(ToolCall(id="c1", name="media__play_pause", arguments={})),
                done_event(),
            ],
            [TextDelta("Done."), done_event()],
        ]
    )
    async with mesh_server_factory(generator=llm) as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        pet_id = await owner.bind_pet()
        await owner.enable_consents()
        host_a = SimulatedXtpHost(platform="macos")
        try:
            id_a = await host_a.connect_enroll(srv.ws_url, owner, "MediaHost")
            await owner.grant_all_standard(str(id_a), max_risk="R2")
            await owner.stream_turn(pet_id, "pause", target_host_id=str(id_a))

            get_resp = await owner.list_habits(pet_id=pet_id)
            assert get_resp.status_code == 200, get_resp.text
            items = get_resp.json()["items"]
            assert len(items) >= 1
            first = items[0]
            assert {
                "habitId",
                "petId",
                "contextKey",
                "hostId",
                "hostAlias",
                "hostPlatform",
                "positiveScore",
                "negativeScore",
                "netScore",
                "sampleCount",
                "gamma",
                "why",
                "updatedAt",
            } <= set(first.keys())
            assert first["gamma"] == pytest.approx(0.85)
            assert isinstance(first["why"], str) and first["why"]

            del_resp = await owner.delete_habit(first["habitId"])
            assert del_resp.status_code == 200, del_resp.text
            assert del_resp.json() == {"habitId": first["habitId"], "deleted": True}

            events = await owner.audit_events(limit=50)
            del_events = [e for e in events if e["action"] == "host.habit_deleted"]
            assert len(del_events) == 1
            assert "contextKey" not in del_events[0]["metadata"]
        finally:
            await host_a.disconnect()
            await owner.aclose()


async def test_t1_f8_04_post_actions_focus_pins_host_and_updates_twin(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F8-04: POST /v1/actions/focus pins/unpins host and toggles followMe."""
    async with mesh_server_factory() as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        host_a = SimulatedXtpHost(platform="macos")
        try:
            id_a = await host_a.connect_enroll(srv.ws_url, owner, "FocusMac")
            r1 = await owner.set_focus(host_id=str(id_a), follow_me=True)
            assert r1.status_code == 200, r1.text
            b1 = r1.json()
            assert b1["focusedHostId"] == str(id_a)
            assert b1["focusedAlias"] == "FocusMac"
            assert b1["followMe"] is True
            assert b1["source"] == "app"
            assert b1["updatedAt"]

            r2 = await owner.set_focus(host_id=None, follow_me=False)
            assert r2.status_code == 200
            b2 = r2.json()
            assert b2["focusedHostId"] is None
            assert b2["followMe"] is False
        finally:
            await host_a.disconnect()
            await owner.aclose()


async def test_t1_f8_05_post_actions_stop_scoped_host_id_vs_global(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F8-05: POST /v1/actions/stop with hostId stops only that host."""
    async with mesh_server_factory() as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        host_a = SimulatedXtpHost(platform="macos")
        host_b = SimulatedXtpHost(platform="windows")
        try:
            id_a = await host_a.connect_enroll(srv.ws_url, owner, "HostA")
            id_b = await host_b.connect_enroll(srv.ws_url, owner, "HostB")
            await owner.grant_all_standard(str(id_a), max_risk="R3")
            await owner.grant_all_standard(str(id_b), max_risk="R3")

            stop_a = await owner.emergency_stop(host_id=str(id_a))
            assert stop_a.status_code == 200, stop_a.text
            assert stop_a.json()["hostId"] == str(id_a)
            assert stop_a.json()["stopped"] is True
            await asyncio.sleep(0.05)

            assert len(host_a.received_stops) == 1
            assert len(host_b.received_stops) == 0

            hosts = {h["hostId"]: h for h in await owner.list_hosts()}
            a_r2 = {g["family"]: g for g in hosts[str(id_a)]["grants"]}["app"]
            b_r2 = {g["family"]: g for g in hosts[str(id_b)]["grants"]}["app"]
            assert a_r2["suspendedAt"] is not None
            assert b_r2["suspendedAt"] is None
        finally:
            await host_a.disconnect()
            await host_b.disconnect()
            await owner.aclose()


async def test_t1_f8_06_sse_action_target_required_pending_and_result_spatial_fields(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F8-06: SSE emits action_target_required and spatial fields on pending/result."""
    llm = ScriptedTextGenerator(
        scripts=[
            # Turn 1: R2 app.launch with identical hosts triggers ask_target
            [
                ToolCallRequested(
                    ToolCall(id="t1", name="app__launch", arguments={"id": "code_editor_x"})
                ),
                done_event(),
            ],
            [TextDelta("Please pick a host."), done_event()],
            # Turn 2: explicit targetHostId reaches action_pending & action_result
            [
                ToolCallRequested(
                    ToolCall(id="t2", name="app__launch", arguments={"id": "code_editor_x"})
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
        host_a = SimulatedXtpHost(platform="macos")
        host_b = SimulatedXtpHost(platform="windows")
        try:
            id_a = await host_a.connect_enroll(srv.ws_url, owner, "MacNode")
            id_b = await host_b.connect_enroll(srv.ws_url, owner, "WinNode")
            await owner.grant_all_standard(str(id_a), max_risk="R2")
            await owner.grant_all_standard(str(id_b), max_risk="R2")

            # Identical presence so R2 margin < 0.18 triggers action_target_required
            subnet = "00112233445566778899aabbccddeeff"
            for h in (host_a, host_b):
                await h.send_presence(
                    idle_seconds=10,
                    screen_locked=False,
                    media_playing=False,
                    subnet_hash=subnet,
                )
                await h.send_resources(installed_apps=["code_editor_x"], workspaces=[])

            turn1 = await owner.stream_turn(
                pet_id,
                "launch code_editor_x",
                client_subnet_hash=subnet,
            )
            req_ev = turn1.first("action_target_required")
            assert req_ev is not None, f"Expected action_target_required, got {turn1.events}"
            assert req_ev["reasonCode"] == "ask_target"
            assert req_ev["tool"] == "app.launch"
            assert isinstance(req_ev["targetReason"], str) and req_ev["targetReason"]
            assert 2 <= len(req_ev["candidates"]) <= 5
            for cand in req_ev["candidates"]:
                assert {
                    "hostId",
                    "alias",
                    "platform",
                    "hostKind",
                    "online",
                    "score",
                    "eligible",
                } <= set(cand.keys())
                assert "wolAvailable" in cand

            async def _approve_on_pending(ev_name: str, data: dict[str, Any]) -> None:
                if ev_name == "action_pending":
                    resp = await owner.decide_approval(data["approvalId"], "approve")
                    assert resp.status_code == 200, resp.text

            turn2 = await owner.stream_turn(
                pet_id,
                "launch code_editor_x",
                target_host_id=str(id_a),
                client_subnet_hash=subnet,
                on_event=_approve_on_pending,
            )
            pending_ev = turn2.first("action_pending")
            result_ev = turn2.first("action_result")
            assert pending_ev is not None
            assert pending_ev.get("targetReason")
            assert pending_ev.get("targetPlatform") == "macos"
            assert result_ev is not None
            assert result_ev.get("hostId") == str(id_a)
            assert result_ev.get("targetReason")
            assert result_ev.get("targetPlatform") == "macos"
        finally:
            await host_a.disconnect()
            await host_b.disconnect()
            await owner.aclose()


# =============================================================================
# Feature 9 (Backend Ingress) — Golden Vectors, mesh_frames.json, 512 KiB Limit
# =============================================================================


async def test_t1_f9_golden_vectors_and_mesh_frames_xtp_linux_ingress(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F9-01..04: Wire crypto matches golden_vectors.json; linux & mesh frames work."""
    verify_golden_vectors(FIXTURES_DIR / "golden_vectors.json")
    fixture = load_mesh_frames_fixture()
    frames = fixture["frames"]
    enroll_params = frames["helloEnrollLinuxRequest"]["params"]["enroll"]
    pres_battery = frames["presenceNotification"]["params"]
    pres_no_battery = frames["presenceDesktopNoBatteryNotification"]["params"]
    res_params = frames["resourcesNotification"]["params"]
    focus_hotkey = frames["focusRequestHotkeyNotification"]["params"]
    focus_tray = frames["focusRequestTrayNotification"]["params"]

    async with mesh_server_factory() as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        linux_host = SimulatedXtpHost(
            platform=enroll_params["platform"],
            os_version=enroll_params["osVersion"],
            agent_version=enroll_params["agentVersion"],
            hostname=enroll_params["hostname"],
        )
        try:
            hid = await linux_host.connect_enroll(srv.ws_url, owner, "LinuxLab")

            # Disconnect and reconnect using Mode B linux hello
            await linux_host.disconnect()
            await linux_host.connect_reconnect(srv.ws_url, host_id=hid)

            # Send presenceNotification from golden fixture verbatim
            await linux_host.send_raw_json(frames["presenceNotification"], sync=True)
            # Send resourcesNotification from golden fixture verbatim
            await linux_host.send_raw_json(frames["resourcesNotification"], sync=True)
            # Send focusRequestHotkeyNotification from golden fixture verbatim
            await linux_host.send_raw_json(frames["focusRequestHotkeyNotification"], sync=True)

            hosts = await owner.list_hosts()
            assert len(hosts) == 1
            h = hosts[0]
            assert h["platform"] == "linux"
            assert h["focused"] is True
            assert h["presence"]["idleSeconds"] == pres_battery["idleSeconds"]
            assert h["presence"]["batteryPercent"] == pres_battery["batteryPercent"]
            assert h["presence"]["foregroundAppId"] == pres_battery["foregroundAppId"]
            assert len(res_params["installedApps"]) == 2

            # Send battery-less presence fixture (batteryPercent omitted, foregroundAppId=None)
            await linux_host.send_raw_json(
                frames["presenceDesktopNoBatteryNotification"], sync=True
            )
            await linux_host.send_raw_json(frames["focusRequestTrayNotification"], sync=True)
            hosts2 = await owner.list_hosts()
            assert hosts2[0]["presence"]["batteryPercent"] is None
            assert hosts2[0]["presence"]["foregroundAppId"] is None
            assert hosts2[0]["presence"]["idleSeconds"] == pres_no_battery["idleSeconds"]
            assert focus_hotkey["source"] == "hotkey"
            assert focus_tray["source"] == "tray"
        finally:
            await linux_host.disconnect()
            await owner.aclose()


async def test_t1_f9_05_xtp_512kib_frame_ceiling_accepted(
    mesh_server_factory: Callable[..., Any],
) -> None:
    """[M1/M2] T1-F9-05: Backend XTP gateway accepts a valid 512 KiB (524,288 B) frame."""
    async with mesh_server_factory() as srv:
        owner = SimulatedOwnerClient(base_url=srv.base_url, ws_url=srv.ws_url, issuer=srv.issuer)
        host = SimulatedXtpHost(platform="linux")
        try:
            await host.connect_enroll(srv.ws_url, owner, "LargeFrameHost")
            prefix = '{"jsonrpc":"2.0","id":777,"result":{"blob":"'
            suffix = '"}}'
            pad_len = 524_288 - len(prefix.encode("utf-8")) - len(suffix.encode("utf-8"))
            exact_512kib = f"{prefix}{'a' * pad_len}{suffix}"
            assert len(exact_512kib.encode("utf-8")) == 524_288
            await host.send_raw_text(exact_512kib)
            pong = await host.ping()
            assert pong["result"] == {}
        finally:
            await host.disconnect()
            await owner.aclose()
