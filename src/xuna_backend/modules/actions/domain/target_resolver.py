"""Deterministic host target resolution without fallback (UDC-4 §5.2-§5.3).

Rules:
1. An explicit alias (`requested_target`) wins when it matches an online active
   host that advertises the tool (case-insensitive alias match).
   If an explicit alias is provided but does not match any candidate host,
   never fall back to another host: return `ask_target` with available choices
   (or `no_host` if none exist).
2. Otherwise, if exactly 1 eligible host exists, select it.
3. Otherwise, if the conversation's last target within 5 minutes matches one of
   the eligible hosts, select it.
4. Otherwise, return `ask_target` with the eligible host aliases so the model
   asks the owner.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from xuna_backend.modules.hosts.domain.entities import Host

LAST_TARGET_TTL = timedelta(minutes=5)


class TargetResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    ASK_TARGET = "ask_target"
    NO_HOST = "no_host"


@dataclass(frozen=True, slots=True)
class TargetResolution:
    status: TargetResolutionStatus
    host: Host | None = None
    choices: tuple[str, ...] = ()


class TargetResolver:
    """Pure target resolver for multi-host action routing."""

    @staticmethod
    def resolve(
        *,
        requested_target: str | None,
        eligible_hosts: Sequence[Host],
        all_tool_hosts: Sequence[Host] | None = None,
        last_target_alias: str | None = None,
        last_target_at: datetime | None = None,
        now: datetime | None = None,
    ) -> TargetResolution:
        candidates_for_explicit = all_tool_hosts if all_tool_hosts is not None else eligible_hosts
        eligible_aliases = tuple(h.alias.value for h in eligible_hosts)

        if requested_target is not None and requested_target.strip():
            needle = requested_target.strip().lower()
            for host in candidates_for_explicit:
                if host.alias.value.lower() == needle:
                    return TargetResolution(
                        status=TargetResolutionStatus.RESOLVED,
                        host=host,
                        choices=eligible_aliases,
                    )
            if eligible_aliases:
                return TargetResolution(
                    status=TargetResolutionStatus.ASK_TARGET,
                    host=None,
                    choices=eligible_aliases,
                )
            return TargetResolution(
                status=TargetResolutionStatus.NO_HOST,
                host=None,
                choices=(),
            )

        if not eligible_hosts:
            return TargetResolution(
                status=TargetResolutionStatus.NO_HOST,
                host=None,
                choices=(),
            )

        if len(eligible_hosts) == 1:
            return TargetResolution(
                status=TargetResolutionStatus.RESOLVED,
                host=eligible_hosts[0],
                choices=eligible_aliases,
            )

        if last_target_alias and last_target_alias.strip():
            recent_valid = True
            if last_target_at is not None and now is not None:
                recent_valid = (now - last_target_at) <= LAST_TARGET_TTL
            if recent_valid:
                needle = last_target_alias.strip().lower()
                for host in eligible_hosts:
                    if host.alias.value.lower() == needle:
                        return TargetResolution(
                            status=TargetResolutionStatus.RESOLVED,
                            host=host,
                            choices=eligible_aliases,
                        )

        return TargetResolution(
            status=TargetResolutionStatus.ASK_TARGET,
            host=None,
            choices=eligible_aliases,
        )
