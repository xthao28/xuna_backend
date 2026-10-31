"""Deterministic host target resolution without fallback (UDC-4 §5.2-§5.3, UDC-7 S7.2).

Rules:
1. Pronouns (`"nó"`, `"máy đó"`, `"máy vừa rồi"`, `"it"`, `"that one"` vs `"máy kia"`,
   `"máy còn lại"`, `"pc kia"`, `"the other one"`):
   - Same-target pronoun resolves to `last_target_alias` iff within 5 minutes and
     present in `eligible_hosts`; otherwise `ask_target`.
   - Other-target pronoun requires a valid `last_target_alias` within 5 minutes and
     excludes that host from `eligible_hosts` (and filters by desktop/mobile when
     `"pc kia"` / `"điện thoại kia"` or when switching between 2 PCs). If a unique
     eligible host remains, select it; otherwise `ask_target`.
2. An explicit alias (`requested_target`) wins when it matches an online active
   host that advertises the tool (case-insensitive / diacritic-insensitive match).
   If an explicit alias is provided but does not match any candidate host,
   never fall back to another host: return `ask_target` with available choices
   (or `no_host` if none exist).
3. Capability filtering: if `capability_filter` is provided, filter `eligible_hosts`
   to those capable of the specific request.
4. If exactly 1 eligible host exists, select it.
5. Otherwise, if the conversation's last target within 5 minutes matches one of
   the eligible hosts, select it.
6. Otherwise, return `ask_target` with the eligible host aliases so the model
   asks the owner.
"""

import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from xuna_backend.modules.hosts.domain.entities import Host, HostPlatform

LAST_TARGET_TTL = timedelta(minutes=5)

_SAME_TARGET_PRONOUNS: frozenset[str] = frozenset(
    {
        "no",
        "may do",
        "may nay",
        "may vua roi",
        "may tinh do",
        "may tinh nay",
        "dien thoai do",
        "dien thoai nay",
        "thiet bi do",
        "thiet bi nay",
        "it",
        "that one",
        "that machine",
        "this one",
        "same one",
    }
)

_OTHER_TARGET_PRONOUNS: frozenset[str] = frozenset(
    {
        "may kia",
        "may con lai",
        "thiet bi kia",
        "pc kia",
        "may tinh kia",
        "dien thoai kia",
        "the other one",
        "the other pc",
        "the other phone",
        "other one",
        "other machine",
        "other pc",
        "other host",
        "other phone",
    }
)

_DESKTOP_OTHER_PRONOUNS: frozenset[str] = frozenset(
    {
        "pc kia",
        "may tinh kia",
        "the other pc",
        "other pc",
        "other computer",
    }
)

_MOBILE_OTHER_PRONOUNS: frozenset[str] = frozenset(
    {
        "dien thoai kia",
        "the other phone",
        "other phone",
    }
)


def _fold_text(raw: str) -> str:
    lowered = raw.strip().lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", lowered)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return " ".join(stripped.split())


def _is_desktop(host: Host) -> bool:
    return host.platform in (
        HostPlatform.WINDOWS,
        HostPlatform.MACOS,
        HostPlatform.LINUX,
    )


def _is_mobile(host: Host) -> bool:
    return host.platform in (HostPlatform.ANDROID, HostPlatform.IOS)


class TargetResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    ASK_TARGET = "ask_target"
    NO_HOST = "no_host"


@dataclass(frozen=True, slots=True)
class TargetResolution:
    status: TargetResolutionStatus
    host: Host | None = None
    choices: tuple[str, ...] = ()
    reason: str = ""


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
        capability_filter: Callable[[Host], bool] | None = None,
    ) -> TargetResolution:
        filtered_eligible = (
            [h for h in eligible_hosts if capability_filter(h)]
            if capability_filter is not None
            else list(eligible_hosts)
        )
        candidates_for_explicit = (
            [h for h in all_tool_hosts if capability_filter(h)]
            if all_tool_hosts is not None and capability_filter is not None
            else (list(all_tool_hosts) if all_tool_hosts is not None else filtered_eligible)
        )
        eligible_aliases = tuple(h.alias.value for h in filtered_eligible)

        last_valid = False
        last_host: Host | None = None
        if last_target_alias and last_target_alias.strip():
            recent_ok = True
            if last_target_at is not None and now is not None:
                recent_ok = (now - last_target_at) <= LAST_TARGET_TTL
            if recent_ok:
                last_valid = True
                last_folded = _fold_text(last_target_alias)
                for h in candidates_for_explicit:
                    if (
                        h.alias.value.lower() == last_target_alias.strip().lower()
                        or _fold_text(h.alias.value) == last_folded
                    ):
                        last_host = h
                        break

        if requested_target is not None and requested_target.strip():
            raw_needle = requested_target.strip().lower()
            folded_needle = _fold_text(requested_target)

            # 1a. Same-target pronoun ("nó", "máy đó", "it", ...)
            if folded_needle in _SAME_TARGET_PRONOUNS:
                if last_valid and last_host is not None and last_host in filtered_eligible:
                    return TargetResolution(
                        status=TargetResolutionStatus.RESOLVED,
                        host=last_host,
                        choices=eligible_aliases,
                        reason="pronoun_same_target",
                    )
                if eligible_aliases:
                    return TargetResolution(
                        status=TargetResolutionStatus.ASK_TARGET,
                        host=None,
                        choices=eligible_aliases,
                        reason="pronoun_expired_or_missing",
                    )
                return TargetResolution(
                    status=TargetResolutionStatus.NO_HOST,
                    host=None,
                    choices=(),
                    reason="no_host",
                )

            # 1b. Other-target pronoun ("máy kia", "pc kia", "the other one", ...)
            if folded_needle in _OTHER_TARGET_PRONOUNS:
                if not filtered_eligible:
                    return TargetResolution(
                        status=TargetResolutionStatus.NO_HOST,
                        host=None,
                        choices=(),
                        reason="no_host",
                    )
                if not last_valid or last_host is None:
                    return TargetResolution(
                        status=TargetResolutionStatus.ASK_TARGET,
                        host=None,
                        choices=eligible_aliases,
                        reason="pronoun_other_without_recent_target",
                    )
                remaining = [
                    h for h in filtered_eligible if h.host_id != last_host.host_id
                ]
                if folded_needle in _DESKTOP_OTHER_PRONOUNS:
                    remaining = [h for h in remaining if _is_desktop(h)]
                elif folded_needle in _MOBILE_OTHER_PRONOUNS:
                    remaining = [h for h in remaining if _is_mobile(h)]
                elif len(remaining) > 1 and _is_desktop(last_host):
                    desktop_remaining = [h for h in remaining if _is_desktop(h)]
                    if len(desktop_remaining) == 1:
                        remaining = desktop_remaining

                if len(remaining) == 1:
                    return TargetResolution(
                        status=TargetResolutionStatus.RESOLVED,
                        host=remaining[0],
                        choices=eligible_aliases,
                        reason="pronoun_other_target",
                    )
                return TargetResolution(
                    status=TargetResolutionStatus.ASK_TARGET,
                    host=None,
                    choices=eligible_aliases,
                    reason="pronoun_other_ambiguous",
                )

            # 2. Explicit alias match (exact case-insensitive first, then diacritic-folded)
            for host in candidates_for_explicit:
                if host.alias.value.lower() == raw_needle:
                    return TargetResolution(
                        status=TargetResolutionStatus.RESOLVED,
                        host=host,
                        choices=eligible_aliases,
                        reason="explicit_alias",
                    )
            folded_matches = [
                host
                for host in candidates_for_explicit
                if _fold_text(host.alias.value) == folded_needle
            ]
            if len(folded_matches) == 1:
                return TargetResolution(
                    status=TargetResolutionStatus.RESOLVED,
                    host=folded_matches[0],
                    choices=eligible_aliases,
                    reason="explicit_alias_folded",
                )

            if eligible_aliases:
                return TargetResolution(
                    status=TargetResolutionStatus.ASK_TARGET,
                    host=None,
                    choices=eligible_aliases,
                    reason="explicit_alias_not_found",
                )
            return TargetResolution(
                status=TargetResolutionStatus.NO_HOST,
                host=None,
                choices=(),
                reason="no_host",
            )

        if not filtered_eligible:
            return TargetResolution(
                status=TargetResolutionStatus.NO_HOST,
                host=None,
                choices=(),
                reason="no_host",
            )

        if len(filtered_eligible) == 1:
            return TargetResolution(
                status=TargetResolutionStatus.RESOLVED,
                host=filtered_eligible[0],
                choices=eligible_aliases,
                reason="single_eligible_host",
            )

        if last_valid and last_host is not None and last_host in filtered_eligible:
            return TargetResolution(
                status=TargetResolutionStatus.RESOLVED,
                host=last_host,
                choices=eligible_aliases,
                reason="last_target_5m",
            )

        return TargetResolution(
            status=TargetResolutionStatus.ASK_TARGET,
            host=None,
            choices=eligible_aliases,
            reason="ambiguous_multiple_hosts",
        )
