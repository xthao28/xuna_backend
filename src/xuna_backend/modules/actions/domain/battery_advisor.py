"""Pure-stdlib Capability & Battery Advisor for spatial multi-host control (Mesh §7).

Never fabricates battery percentages when telemetry is absent (`None`, Rule H2).
Recognizes plugged-in, adequately charged, and battery-less hosts (`HostKind.DESKTOP`,
`HostKind.SERVER`, `HostKind.BRIDGE`) as non-low-battery alternatives.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from xuna_backend.modules.hosts.domain.entities import Host, HostId, HostKind, HostStatus
from xuna_backend.modules.hosts.domain.twin import HostPresence, HostTwin

LOW_BATTERY_THRESHOLD_PCT = 15


@dataclass(frozen=True, slots=True)
class BatteryAdvisory:
    """Evaluated low-battery warning and suggested alternative hosts."""

    low_battery_hosts: tuple[tuple[HostId, str, int], ...]
    alternative_hosts: tuple[tuple[HostId, str], ...]
    messages: tuple[str, ...]


def is_low_battery(
    presence: HostPresence | None,
    *,
    threshold_pct: int = LOW_BATTERY_THRESHOLD_PCT,
) -> bool:
    """Returns `True` only when measured battery is `<= threshold_pct` and not charging."""
    return (
        presence is not None
        and presence.battery_percent is not None
        and presence.battery_percent <= threshold_pct
        and presence.is_charging is False
    )


def is_alternative_power_host(
    host: Host,
    presence: HostPresence | None,
    *,
    threshold_pct: int = LOW_BATTERY_THRESHOLD_PCT,
) -> bool:
    """Returns `True` when an online host is plugged in, above threshold, or battery-less."""
    if host.host_kind in (HostKind.SERVER, HostKind.BRIDGE):
        return True
    if presence is None:
        return False
    if presence.is_charging is True:
        return True
    if presence.battery_percent is not None:
        return presence.battery_percent > threshold_pct
    # Battery-less desktop workstations report `battery_percent=None, is_charging=None`.
    return host.host_kind == HostKind.DESKTOP and presence.is_charging is not False


def evaluate_battery_advisory(
    hosts: Sequence[Host],
    twins_by_host: Mapping[str, HostTwin],
    *,
    online_host_ids: Collection[str] | None = None,
    target_host_id: HostId | None = None,
    capable_host_ids: Collection[HostId] | None = None,
    threshold_pct: int = LOW_BATTERY_THRESHOLD_PCT,
) -> BatteryAdvisory | None:
    """Evaluates low-battery warnings and alternative hosts from live `HostTwin` presence."""
    active_online = [
        h
        for h in sorted(hosts, key=lambda item: (item.alias.value.lower(), str(item.host_id)))
        if h.status == HostStatus.ACTIVE
        and (
            str(h.host_id) in online_host_ids
            if online_host_ids is not None
            else str(h.host_id) in twins_by_host
        )
    ]

    low_hosts: list[tuple[HostId, str, int]] = []
    alt_hosts: list[tuple[HostId, str]] = []

    for host in active_online:
        twin = twins_by_host.get(str(host.host_id))
        presence = twin.presence if twin is not None else None
        if is_low_battery(presence, threshold_pct=threshold_pct):
            if (
                (target_host_id is None or host.host_id == target_host_id)
                and presence is not None
                and presence.battery_percent is not None
            ):
                low_hosts.append((host.host_id, host.alias.value, presence.battery_percent))
            continue

        if target_host_id is not None and host.host_id == target_host_id:
            continue
        if capable_host_ids is not None and host.host_id not in capable_host_ids:
            continue
        if is_alternative_power_host(host, presence, threshold_pct=threshold_pct):
            alt_hosts.append((host.host_id, host.alias.value))

    if not low_hosts:
        return None

    low_labels = ", ".join(f"{alias} ({pct}%)" for _, alias, pct in low_hosts)
    if alt_hosts:
        alt_aliases = ", ".join(alias for _, alias in alt_hosts[:3])
        msg = f"Low battery on {low_labels}; alternative hosts available: {alt_aliases}"
    else:
        msg = f"Low battery on {low_labels}; consider connecting power"
    return BatteryAdvisory(
        low_battery_hosts=tuple(low_hosts),
        alternative_hosts=tuple(alt_hosts),
        messages=(msg,),
    )
