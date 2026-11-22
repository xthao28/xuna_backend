"""Application tests for `MeshProber` (`actions/application/mesh_prober.py`)."""

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

from xuna_backend.modules.actions.application.mesh_prober import (
    MeshProber,
    ProbeTarget,
)
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
PUB = b"\x04" + b"\x33" * 64


def _host(owner_id: uuid.UUID, alias: str, platform: HostPlatform = HostPlatform.MACOS) -> Host:
    return Host(
        host_id=HostId.generate(),
        owner_id=owner_id,
        alias=HostAlias(alias),
        platform=platform,
        hostname=alias.lower(),
        os_version="15.0",
        agent_version="0.2.0",
        public_key=PUB,
        status=HostStatus.ACTIVE,
        created_at=NOW,
        host_kind=HostKind.DESKTOP,
    )


def _desc(
    name: str,
    *,
    family: ToolFamily = ToolFamily.HOST,
    risk: RiskTier = RiskTier.R0,
    observes: bool = False,
) -> ToolDescriptor:
    return ToolDescriptor(
        name=name,
        description=f"Tool {name}",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        output_schema={"type": "object", "properties": {}, "additionalProperties": False},
        family=family,
        risk=risk,
        observes=observes,
    )


async def _connect_responder(
    hub: InMemoryHostSessionHub,
    owner_id: uuid.UUID,
    host: Host,
    handler: Any,
) -> None:
    tasks: set[asyncio.Task[None]] = set()

    async def _send(raw: str) -> None:
        import json

        frame = json.loads(raw)
        if frame.get("method") == "tools/call":
            req_id = frame["id"]
            tname = frame["params"]["name"]

            async def _reply() -> None:
                res = await handler(tname)
                if res is not None:
                    hub.on_response(str(host.host_id), req_id, res)

            task = asyncio.create_task(_reply())
            tasks.add(task)
            task.add_done_callback(tasks.discard)

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


def test_mesh_prober_concurrent_timeout_isolation_and_path_redaction() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        other_owner = uuid.uuid4()
        hub = InMemoryHostSessionHub()
        prober = MeshProber(hub=hub, clock=lambda: NOW, default_deadline_ms=150)

        h_fast = _host(owner, "FastNode")
        h_slow = _host(owner, "SlowNode", HostPlatform.WINDOWS)
        h_foreign = _host(other_owner, "ForeignNode")

        status_d = _desc("host.status", family=ToolFamily.HOST, risk=RiskTier.R0)
        vitals_d = _desc("system.vitals", family=ToolFamily.HOST, risk=RiskTier.R1, observes=True)
        r2_d = _desc("app.launch", family=ToolFamily.APP, risk=RiskTier.R2)
        m_fast = Manifest(host_id=h_fast.host_id, tools=(status_d, vitals_d, r2_d), updated_at=NOW)
        m_slow = Manifest(host_id=h_slow.host_id, tools=(status_d, vitals_d), updated_at=NOW)

        async def _fast_handler(_t: str) -> dict[str, Any]:
            return {
                "_meta": {"xuna/status": "ok", "xuna/durationMs": 5},
                "structuredContent": {
                    "cpuPercent": 12,
                    "path": "/secret/user/path",
                    "note": "/private/secret/file.txt",
                },
            }

        async def _slow_handler(_t: str) -> dict[str, Any]:
            await asyncio.sleep(0.5)
            return {"_meta": {"xuna/status": "ok"}, "structuredContent": {"ok": True}}

        await _connect_responder(hub, owner, h_fast, _fast_handler)
        await _connect_responder(hub, owner, h_slow, _slow_handler)
        await _connect_responder(hub, other_owner, h_foreign, _fast_handler)

        hub.update_presence(
            str(h_fast.host_id),
            HostPresence(
                idle_seconds=5,
                screen_locked=False,
                media_playing=False,
                battery_percent=10,
                is_charging=False,
                updated_at=NOW,
            ),
        )
        hub.update_presence(
            str(h_slow.host_id),
            HostPresence(
                idle_seconds=5,
                screen_locked=False,
                media_playing=False,
                battery_percent=90,
                is_charging=True,
                updated_at=NOW,
            ),
        )

        # R2 probe is always denied by MeshProber
        r2_out = await prober.probe_one(
            host=h_fast,
            descriptor=r2_d,
            grants=(Grant(h_fast.host_id, ToolFamily.APP, RiskTier.R2, True, NOW),),
            host_screen_consent=True,
        )
        assert r2_out.status == "denied"
        assert r2_out.error_code == "probe_risk_too_high"

        targets = [
            ProbeTarget(
                host=h_fast,
                manifest=m_fast,
                grants=(Grant(h_fast.host_id, ToolFamily.HOST, RiskTier.R1, True, NOW),),
                online=True,
            ),
            ProbeTarget(
                host=h_slow,
                manifest=m_slow,
                grants=(Grant(h_slow.host_id, ToolFamily.HOST, RiskTier.R0, True, NOW),),
                online=True,
            ),
            ProbeTarget(
                host=h_foreign,
                manifest=None,
                grants=(),
                online=True,
            ),
        ]

        # Verify foreign isolation, timeout isolation, R1 grant check, and path redaction
        res = await prober.query_status(
            owner_id=owner,
            targets=targets,
            include_vitals=True,
            host_screen_consent=True,
        )
        hosts_list = res["hosts"]
        assert isinstance(hosts_list, list)
        by_id = {row["hostId"]: row for row in hosts_list}
        assert str(h_foreign.host_id) not in by_id
        assert by_id[str(h_fast.host_id)]["probeStatus"] == "ok"
        assert by_id[str(h_fast.host_id)]["vitalsStatus"] == "ok"
        assert "path" not in by_id[str(h_fast.host_id)]["probe"]
        assert by_id[str(h_fast.host_id)]["probe"]["note"] == "[redacted_path]"
        assert by_id[str(h_fast.host_id)]["lowBattery"] is True
        assert "batteryAdvisories" in res

        assert by_id[str(h_slow.host_id)]["probeStatus"] == "timeout"
        assert by_id[str(h_slow.host_id)]["vitalsStatus"] == "denied"
        assert by_id[str(h_slow.host_id)]["vitalsErrorCode"] == "grant_denied"

        # Passing ONLY foreign hostIds returns empty hosts
        res_foreign = await prober.query_status(
            owner_id=owner,
            targets=targets,
            host_ids=(h_foreign.host_id.value,),
        )
        assert res_foreign["hosts"] == []

    asyncio.run(_run())


def test_mesh_prober_find_resource_twin_and_r0_app_list_fallback() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        hub = InMemoryHostSessionHub()
        prober = MeshProber(hub=hub, clock=lambda: NOW, default_deadline_ms=500)

        h_twin = _host(owner, "TwinHost")
        h_probe = _host(owner, "ProbeHost", HostPlatform.WINDOWS)
        app_list_d = _desc("app.list_allowed", family=ToolFamily.APP, risk=RiskTier.R0)
        tool_d = _desc("smarthome.dim", family=ToolFamily.APP, risk=RiskTier.R1)
        m_twin = Manifest(host_id=h_twin.host_id, tools=(tool_d,), updated_at=NOW)
        m_probe = Manifest(host_id=h_probe.host_id, tools=(app_list_d,), updated_at=NOW)

        async def _probe_handler(tname: str) -> dict[str, Any]:
            if tname == "app.list_allowed":
                return {
                    "_meta": {"xuna/status": "ok"},
                    "structuredContent": {
                        "apps": [
                            {"id": "synth_studio_x"},
                            {"id": "/bad/path/app"},
                        ]
                    },
                }
            return {"_meta": {"xuna/status": "ok"}, "structuredContent": {}}

        await _connect_responder(hub, owner, h_twin, _probe_handler)
        await _connect_responder(hub, owner, h_probe, _probe_handler)

        hub.update_resources(
            str(h_twin.host_id),
            HostResources(
                installed_apps=("editor_alpha",),
                workspaces=("Apollo_Repo",),
                updated_at=NOW,
            ),
        )
        # h_probe intentionally has no HostTwin.resources -> falls back to R0 app.list_allowed
        targets = [
            ProbeTarget(
                host=h_twin,
                manifest=m_twin,
                grants=(Grant(h_twin.host_id, ToolFamily.APP, RiskTier.R1, True, NOW),),
                online=True,
            ),
            ProbeTarget(
                host=h_probe,
                manifest=m_probe,
                grants=(Grant(h_probe.host_id, ToolFamily.APP, RiskTier.R0, True, NOW),),
                online=True,
            ),
        ]

        res_ws = await prober.find_resource(
            owner_id=owner, targets=targets, query="Apollo_Repo", kind="workspace"
        )
        assert [m["hostId"] for m in res_ws["matches"]] == [str(h_twin.host_id)]  # type: ignore[index]

        res_app = await prober.find_resource(
            owner_id=owner, targets=targets, query="synth_studio_x", kind="app"
        )
        assert [m["hostId"] for m in res_app["matches"]] == [str(h_probe.host_id)]  # type: ignore[index]
        assert res_app["matches"][0]["source"] == "probe"  # type: ignore[index]

        res_tool = await prober.find_resource(
            owner_id=owner, targets=targets, query="smarthome", kind="tool"
        )
        expected_ids = [str(h_twin.host_id)]
        assert [m["hostId"] for m in res_tool["matches"]] == expected_ids  # type: ignore[index]

        res_bad = await prober.find_resource(
            owner_id=owner, targets=targets, query="bad", kind="app"
        )
        assert res_bad["matches"] == []

        res_bad_path = await prober.find_resource(
            owner_id=owner, targets=targets, query="/bad/path/app", kind="app"
        )
        assert res_bad_path["matches"] == []

    asyncio.run(_run())


def test_mesh_prober_embedded_path_redaction_snake_keys_and_battery_advisories() -> None:
    async def _run() -> None:
        owner = uuid.uuid4()
        hub = InMemoryHostSessionHub()
        prober = MeshProber(hub=hub, clock=lambda: NOW, default_deadline_ms=500)

        h_laptop = _host(owner, "LaptopNode")
        h_mobile = Host(
            host_id=HostId.generate(),
            owner_id=owner,
            alias=HostAlias("PhoneNode"),
            platform=HostPlatform.IOS,
            hostname="phone-node",
            os_version="18.0",
            agent_version="0.2.0",
            public_key=PUB,
            status=HostStatus.ACTIVE,
            created_at=NOW,
            host_kind=HostKind.MOBILE,
        )
        h_server = Host(
            host_id=HostId.generate(),
            owner_id=owner,
            alias=HostAlias("ServerNode"),
            platform=HostPlatform.LINUX,
            hostname="server-node",
            os_version="6.8",
            agent_version="0.2.0",
            public_key=PUB,
            status=HostStatus.ACTIVE,
            created_at=NOW,
            host_kind=HostKind.SERVER,
        )
        status_d = _desc("host.status", family=ToolFamily.HOST, risk=RiskTier.R0)
        m_laptop = Manifest(host_id=h_laptop.host_id, tools=(status_d,), updated_at=NOW)
        m_mobile = Manifest(host_id=h_mobile.host_id, tools=(status_d,), updated_at=NOW)
        m_server = Manifest(host_id=h_server.host_id, tools=(status_d,), updated_at=NOW)

        async def _handler(_t: str) -> dict[str, Any]:
            return {
                "_meta": {"xuna/status": "ok"},
                "structuredContent": {
                    "embeddedPosix": "Failed opening /Users/alice/private.key now",
                    "embeddedWin": "Missing C:\\Users\\Bob\\secrets.txt in dir",
                    "embeddedUnc": "Denied \\\\fileserver\\share\\doc.txt here",
                    "embeddedFileUri": "open file:///Users/alice/private.key",
                    "safeUrl": "https://example.com/a/b",
                    "safeRatio": "app/v1",
                    "safeStatus": "status: ok",
                    "file_path": "/secret/a",
                    "working_dir": "/secret/b",
                    "executable_path": "/secret/c",
                },
            }

        for h in (h_laptop, h_mobile, h_server):
            await _connect_responder(hub, owner, h, _handler)

        hub.update_presence(
            str(h_laptop.host_id),
            HostPresence(
                idle_seconds=5,
                screen_locked=False,
                media_playing=False,
                battery_percent=10,
                is_charging=False,
                updated_at=NOW,
            ),
        )
        # Unmeasured mobile host (battery_percent=None, is_charging=False) must NOT be in advisories
        hub.update_presence(
            str(h_mobile.host_id),
            HostPresence(
                idle_seconds=5,
                screen_locked=False,
                media_playing=False,
                battery_percent=None,
                is_charging=False,
                updated_at=NOW,
            ),
        )
        targets = [
            ProbeTarget(
                host=h_laptop,
                manifest=m_laptop,
                grants=(Grant(h_laptop.host_id, ToolFamily.HOST, RiskTier.R0, True, NOW),),
                online=True,
            ),
            ProbeTarget(
                host=h_mobile,
                manifest=m_mobile,
                grants=(Grant(h_mobile.host_id, ToolFamily.HOST, RiskTier.R0, True, NOW),),
                online=True,
            ),
        ]
        res_no_srv = await prober.query_status(owner_id=owner, targets=targets)
        assert "batteryAdvisories" not in res_no_srv

        laptop_row = next(
            r
            for r in res_no_srv["hosts"]
            if r["hostId"] == str(h_laptop.host_id)  # type: ignore[union-attr]
        )
        probe_out = laptop_row["probe"]  # type: ignore[index]
        assert probe_out["embeddedPosix"] == "Failed opening [redacted_path] now"
        assert probe_out["embeddedWin"] == "Missing [redacted_path] in dir"
        assert probe_out["embeddedUnc"] == "Denied [redacted_path] here"
        assert probe_out["embeddedFileUri"] == "open [redacted_path]"
        assert probe_out["safeUrl"] == "https://example.com/a/b"
        assert probe_out["safeRatio"] == "app/v1"
        assert probe_out["safeStatus"] == "status: ok"
        assert "file_path" not in probe_out
        assert "working_dir" not in probe_out
        assert "executable_path" not in probe_out

        # SERVER with battery_percent=None IS a valid plugged alternative host
        hub.update_presence(
            str(h_server.host_id),
            HostPresence(
                idle_seconds=0,
                screen_locked=False,
                media_playing=False,
                battery_percent=None,
                is_charging=False,
                updated_at=NOW,
            ),
        )
        res_with_srv = await prober.query_status(
            owner_id=owner,
            targets=[
                *targets,
                ProbeTarget(
                    host=h_server,
                    manifest=m_server,
                    grants=(Grant(h_server.host_id, ToolFamily.HOST, RiskTier.R0, True, NOW),),
                    online=True,
                ),
            ],
        )
        advisories = res_with_srv["batteryAdvisories"]
        assert isinstance(advisories, list)
        assert "ServerNode" in advisories[0]
        assert "PhoneNode" not in advisories[0]

    asyncio.run(_run())


def test_mesh_prober_uppercase_file_uri_and_sentinel_skip() -> None:
    from xuna_backend.modules.actions.application.mesh_prober import (
        _extract_normalized_app_ids,
        _sanitize_probe_value,
    )

    sanitized = _sanitize_probe_value(
        {"upper": "open FILE:///etc/passwd", "mixed": "File:///tmp/a"}
    )
    assert isinstance(sanitized, dict)
    assert sanitized["upper"] == "open [redacted_path]"
    assert sanitized["mixed"] == "[redacted_path]"

    extracted = _extract_normalized_app_ids(
        {
            "apps": [
                "[redacted_path]",
                "[truncated]",
                "my_app [redacted_path]",
                "other [truncated]",
                "notes_app",
                "notes_app",
            ]
        }
    )
    assert extracted == ("notes_app",)

    sanitized_embedded = _sanitize_probe_value(
        {"apps": [{"id": "my_app /Users/secret/x"}, {"id": "editor_alpha"}]}
    )
    assert isinstance(sanitized_embedded, dict)
    assert _extract_normalized_app_ids(sanitized_embedded) == ("editor_alpha",)


def test_mesh_prober_redacts_home_and_relative_paths_preserving_urls() -> None:
    """Tier 5 (Gap 6): _OS_PATH_RE redacts ~/..., ./..., ../... while keeping URLs."""
    from xuna_backend.modules.actions.application.mesh_prober import _sanitize_probe_value

    sanitized = _sanitize_probe_value(
        {
            "homeUnix": "cat ~/Documents/secret.txt",
            "homeWin": r"type ~\AppData\secret.txt",
            "dotRel": "open ./.env and ../priv/key",
            "winRel": r"read ..\priv\key",
            "httpsUrl": "https://example.com/a/b",
            "httpUrl": "http://localhost:8080/health",
        }
    )
    assert isinstance(sanitized, dict)
    assert "~/Documents" not in str(sanitized["homeUnix"])
    assert sanitized["homeUnix"] == "cat [redacted_path]"
    assert "~\\AppData" not in str(sanitized["homeWin"])
    assert sanitized["homeWin"] == "type [redacted_path]"
    assert "./.env" not in str(sanitized["dotRel"])
    assert "../priv" not in str(sanitized["dotRel"])
    assert sanitized["dotRel"] == "open [redacted_path] and [redacted_path]"
    assert "..\\priv" not in str(sanitized["winRel"])
    assert sanitized["winRel"] == "read [redacted_path]"
    assert sanitized["httpsUrl"] == "https://example.com/a/b"
    assert sanitized["httpUrl"] == "http://localhost:8080/health"
