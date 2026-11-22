"""Unit tests for `actions.domain.battery_advisor` (Mesh §7 & Rule H2)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from xuna_backend.modules.actions.domain.battery_advisor import (
    evaluate_battery_advisory,
    is_alternative_power_host,
    is_low_battery,
)
from xuna_backend.modules.hosts.domain.entities import (
    Host,
    HostAlias,
    HostId,
    HostKind,
    HostPlatform,
    HostStatus,
)
from xuna_backend.modules.hosts.domain.twin import HostPresence, HostTwin

NOW = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)
PUB = b"\x04" + b"\x11" * 64


def _make_host(
    owner_id: uuid.UUID,
    alias: str,
    *,
    platform: HostPlatform = HostPlatform.MACOS,
    kind: HostKind = HostKind.DESKTOP,
) -> Host:
    return Host(
        host_id=HostId.generate(),
        owner_id=owner_id,
        alias=HostAlias(alias),
        platform=platform,
        host_kind=kind,
        hostname=alias.lower(),
        os_version="15.0",
        agent_version="0.2.0",
        public_key=PUB,
        status=HostStatus.ACTIVE,
        created_at=NOW,
    )


def _presence(
    *, battery_percent: int | None = None, is_charging: bool | None = None
) -> HostPresence:
    return HostPresence(
        idle_seconds=5,
        screen_locked=False,
        media_playing=False,
        updated_at=NOW,
        battery_percent=battery_percent,
        is_charging=is_charging,
    )


def test_low_battery_boundary_15_unplugged_vs_16_charging_and_null() -> None:
    assert is_low_battery(_presence(battery_percent=15, is_charging=False)) is True
    assert is_low_battery(_presence(battery_percent=16, is_charging=False)) is False
    assert is_low_battery(_presence(battery_percent=15, is_charging=True)) is False
    assert is_low_battery(_presence(battery_percent=10, is_charging=None)) is False
    assert is_low_battery(_presence(battery_percent=None, is_charging=False)) is False
    assert is_low_battery(None) is False


def test_alternative_power_host_includes_batteryless_desktop_server_bridge() -> None:
    owner_id = uuid.uuid4()
    desktop_nobatt = _make_host(owner_id, "WinTower", platform=HostPlatform.WINDOWS)
    server = _make_host(owner_id, "LabNode", platform=HostPlatform.LINUX, kind=HostKind.SERVER)
    bridge = _make_host(owner_id, "RoomBridge", platform=HostPlatform.LINUX, kind=HostKind.BRIDGE)
    laptop_low = _make_host(owner_id, "Laptop", platform=HostPlatform.MACOS)

    assert (
        is_alternative_power_host(desktop_nobatt, _presence(battery_percent=None, is_charging=None))
        is True
    )
    assert is_alternative_power_host(desktop_nobatt, None) is False
    assert is_alternative_power_host(server, None) is True
    assert is_alternative_power_host(bridge, None) is True
    assert (
        is_alternative_power_host(laptop_low, _presence(battery_percent=15, is_charging=False))
        is False
    )
    assert (
        is_alternative_power_host(laptop_low, _presence(battery_percent=15, is_charging=True))
        is True
    )


def test_evaluate_battery_advisory_suggests_batteryless_desktop_without_fabrication() -> None:
    owner_id = uuid.uuid4()
    low_laptop = _make_host(owner_id, "LowLaptop", platform=HostPlatform.MACOS)
    win_tower = _make_host(owner_id, "WinTower", platform=HostPlatform.WINDOWS)
    charging_pc = _make_host(owner_id, "PluggedPC", platform=HostPlatform.WINDOWS)

    twins = {
        str(low_laptop.host_id): HostTwin(
            host_id=low_laptop.host_id,
            owner_id=owner_id,
            presence=_presence(battery_percent=11, is_charging=False),
        ),
        str(win_tower.host_id): HostTwin(
            host_id=win_tower.host_id,
            owner_id=owner_id,
            presence=_presence(battery_percent=None, is_charging=None),
        ),
        str(charging_pc.host_id): HostTwin(
            host_id=charging_pc.host_id,
            owner_id=owner_id,
            presence=_presence(battery_percent=15, is_charging=True),
        ),
    }

    advisory = evaluate_battery_advisory(
        [low_laptop, win_tower, charging_pc],
        twins,
        online_host_ids={str(low_laptop.host_id), str(win_tower.host_id), str(charging_pc.host_id)},
        target_host_id=low_laptop.host_id,
    )
    assert advisory is not None
    assert len(advisory.low_battery_hosts) == 1
    assert advisory.low_battery_hosts[0][1] == "LowLaptop"
    alt_names = {alias for _, alias in advisory.alternative_hosts}
    assert alt_names == {"WinTower", "PluggedPC"}
    assert "LowLaptop" in advisory.messages[0].split(";")[0]
    assert "WinTower" in advisory.messages[0]
    assert "PluggedPC" in advisory.messages[0]


def test_evaluate_battery_advisory_without_alternatives_warns_without_fabricating() -> None:
    owner_id = uuid.uuid4()
    low_laptop = _make_host(owner_id, "SoloLaptop")
    twins = {
        str(low_laptop.host_id): HostTwin(
            host_id=low_laptop.host_id,
            owner_id=owner_id,
            presence=_presence(battery_percent=10, is_charging=False),
        )
    }
    advisory = evaluate_battery_advisory(
        [low_laptop],
        twins,
        online_host_ids={str(low_laptop.host_id)},
    )
    assert advisory is not None
    assert advisory.alternative_hosts == ()
    assert "SoloLaptop" in advisory.messages[0]
    assert "alternative hosts" not in advisory.messages[0]


def test_alternative_power_host_rejects_unmeasured_mobile_host() -> None:
    owner_id = uuid.uuid4()
    low_laptop = _make_host(owner_id, "LowLaptop", platform=HostPlatform.MACOS)
    mobile = _make_host(owner_id, "OwnerPhone", platform=HostPlatform.IOS, kind=HostKind.MOBILE)
    unmeasured = _presence(battery_percent=None, is_charging=None)

    assert is_alternative_power_host(mobile, unmeasured) is False
    assert (
        is_alternative_power_host(mobile, _presence(battery_percent=None, is_charging=True)) is True
    )

    twins = {
        str(low_laptop.host_id): HostTwin(
            host_id=low_laptop.host_id,
            owner_id=owner_id,
            presence=_presence(battery_percent=12, is_charging=False),
        ),
        str(mobile.host_id): HostTwin(
            host_id=mobile.host_id,
            owner_id=owner_id,
            presence=unmeasured,
        ),
    }
    advisory = evaluate_battery_advisory(
        [low_laptop, mobile],
        twins,
        online_host_ids={str(low_laptop.host_id), str(mobile.host_id)},
        target_host_id=low_laptop.host_id,
    )
    assert advisory is not None
    assert advisory.alternative_hosts == ()
