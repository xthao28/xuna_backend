"""Pure stdlib Evidence Confidence Scorer & safe fuzzy host matcher (Mesh §5.4).

Combines Resource Match, Live Presence, Contextual Habit (`gamma=0.85`), and Session
Continuity against risk-tiered thresholds (`R0`-`R3`), classifying `no_host`,
`host_offline`, and `grant_denied` before falling back to `ask_target`.
Zero I/O, zero frameworks, and zero hardcoded app/host/path/keyword tables (H5).
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Literal

from xuna_backend.modules.actions.domain.habit_learner import (
    day_part_for_dt,
    lookup_habit_score,
)
from xuna_backend.modules.actions.domain.habits import HostRoutingHabit
from xuna_backend.modules.actions.domain.target_resolver import (
    LAST_TARGET_TTL,
    is_target_pronoun,
)
from xuna_backend.modules.actions.domain.tool_names import is_forbidden_tool_name
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    Host,
    HostId,
    HostKind,
    HostStatus,
    Manifest,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
    ToolPermissionState,
    fold_text,
)
from xuna_backend.modules.hosts.domain.twin import (
    HostFocusState,
    HostPresence,
    HostResources,
    HostTwin,
)

FUZZY_MIN_SCORE = 0.82
FUZZY_MIN_MARGIN = 0.10
FUZZY_LEVENSHTEIN_WEIGHT = 0.7
FUZZY_PHONETIC_WEIGHT = 0.3

_U5_REQUIRED_FAMILIES: frozenset[ToolFamily] = frozenset(
    {
        ToolFamily.CU,
        ToolFamily.SCREEN,
        ToolFamily.UI,
        ToolFamily.INPUT,
    }
)
_INTERACTIVE_LOCK_FAMILIES: frozenset[ToolFamily] = frozenset(
    {
        ToolFamily.UI,
        ToolFamily.INPUT,
        ToolFamily.SCREEN,
        ToolFamily.APP,
        ToolFamily.CU,
    }
)
_INTERACTIVE_HOST_KINDS: frozenset[HostKind] = frozenset({HostKind.DESKTOP, HostKind.MOBILE})
_RESOURCE_ARG_KEYS: tuple[str, ...] = ("id", "appId", "project", "workspace", "capsuleId")


class TargetReasonCode(StrEnum):
    """Closed reason codes explaining why a host was selected or rejected (§5.4.5)."""

    PINNED_HOST = "pinned_host"
    EXPLICIT_ALIAS = "explicit_alias"
    PLACE_TAG = "place_tag"
    FOLLOW_ME_ACTIVE = "follow_me_active"
    UNIQUE_CAPABLE_HOST = "unique_capable_host"
    RESOURCE_AND_PRESENCE_MATCH = "resource_and_presence_match"
    HIGH_CONFIDENCE_HABIT_AND_PRESENCE = "high_confidence_habit_and_presence"
    SESSION_CONTINUITY = "session_continuity"
    HOST_OFFLINE = "host_offline"
    GRANT_DENIED = "grant_denied"
    AMBIGUOUS_CANDIDATES = "ambiguous_candidates"
    NO_HOST = "no_host"


class ActionTargetRequiredReason(StrEnum):
    """Wire `reasonCode` values for `action_target_required` (§5.6.7 #3)."""

    ASK_TARGET = "ask_target"
    HOST_OFFLINE = "host_offline"
    GRANT_DENIED = "grant_denied"
    NO_HOST = "no_host"


class CandidateUnavailableReason(StrEnum):
    """Per-candidate unavailability reasons on `action_target_required`."""

    HOST_OFFLINE = "host_offline"
    GRANT_DENIED = "grant_denied"
    U5_CONSENT_REQUIRED = "u5_consent_required"
    POLICY_BLOCKED = "policy_blocked"
    UNSUPPORTED = "unsupported"


RISK_THRESHOLDS: dict[RiskTier, tuple[float, float]] = {
    RiskTier.R0: (0.45, 0.08),
    RiskTier.R1: (0.58, 0.12),
    RiskTier.R2: (0.72, 0.18),
    RiskTier.R3: (0.85, 0.25),
    RiskTier.R4: (0.85, 0.25),
}


@dataclass(frozen=True, slots=True)
class TargetReason:
    """Structured explanation attached to routing decisions, approvals, and results."""

    reason_code: TargetReasonCode
    summary: str


@dataclass(frozen=True, slots=True)
class HostQueryMatch:
    """Outcome of safe unique-only alias/nickname/place matching (H3)."""

    status: Literal["unique", "ambiguous", "no_match"]
    host: Host | None = None
    candidates: tuple[Host, ...] = ()
    match_source: Literal["alias", "nickname", "place", "fuzzy"] | None = None
    scores: Mapping[HostId, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HostScoringCandidateInput:
    """All domain inputs needed to evaluate one candidate host for a tool call."""

    host: Host
    online: bool
    manifest: Manifest | None
    grants: Sequence[Grant]
    twin: HostTwin | None = None
    policy_blocked: bool = False
    has_active_cu_lane: bool = False
    effective_risk: RiskTier | None = None


@dataclass(frozen=True, slots=True)
class MeshScoringContext:
    """Turn-level scoring context passed into `score_hosts_for_tool`."""

    tool_name: str
    arguments: Mapping[str, object]
    now: datetime
    host_screen_consent: bool
    focus_state: HostFocusState | None = None
    client_subnet_hash: str | None = None
    last_target_alias: str | None = None
    last_target_at: datetime | None = None
    habits_by_key_host: Mapping[tuple[str, HostId], HostRoutingHabit] = field(default_factory=dict)
    default_host_id: HostId | None = None
    explicit_requested_host_id: HostId | None = None
    explicit_reason_code: TargetReasonCode | None = None
    client_tz_offset_minutes: int | None = None


@dataclass(frozen=True, slots=True)
class ScoredHostCandidate:
    """Scored candidate host with component breakdown and availability diagnostics."""

    host: Host
    score: float
    eligible: bool
    online: bool
    unavailable_reason: CandidateUnavailableReason | None
    reason: str
    wol_mac: str | None
    wol_available: bool
    m_res: float = 0.0
    m_pres: float = 0.0
    m_hab: float = 0.5
    m_cont: float = 0.3


@dataclass(frozen=True, slots=True)
class MeshScoringDecision:
    """Exhaustive resolution outcome from `score_hosts_for_tool`."""

    resolved: bool
    host: Host | None
    target_reason: TargetReason
    action_required_reason: ActionTargetRequiredReason | None
    candidates: tuple[ScoredHostCandidate, ...]


def levenshtein_distance(a: str, b: str) -> int:
    """Standard Levenshtein edit distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def levenshtein_ratio(a: str, b: str) -> float:
    """Normalized Levenshtein similarity in `[0.0, 1.0]`."""
    if not a and not b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - (levenshtein_distance(a, b) / max_len)


def phonetic_fold(raw: str) -> str:
    """Single-pass Vietnamese-safe consonant normalization without cross-rule collisions."""
    folded = fold_text(raw).replace(" ", "")
    out: list[str] = []
    i = 0
    n = len(folded)
    while i < n:
        if i + 1 < n:
            digraph = folded[i : i + 2]
            if digraph == "ph":
                out.append("f")
                i += 2
                continue
            if digraph in ("tr", "ch"):
                out.append("c")
                i += 2
                continue
            if digraph == "gi":
                out.append("z")
                i += 2
                continue
        ch = folded[i]
        if ch in ("k", "c", "q"):
            out.append("k")
        elif ch in ("d", "r"):
            out.append("z")
        elif ch in ("s", "x"):
            out.append("s")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def phonetic_bigram_sim(pa: str, pb: str) -> float:
    """Dice bigram similarity guarded against `< 2` character inputs (no ZeroDivisionError)."""
    if len(pa) < 2 or len(pb) < 2:
        return 1.0 if pa == pb else 0.0
    ba = {pa[i : i + 2] for i in range(len(pa) - 1)}
    bb = {pb[i : i + 2] for i in range(len(pb) - 1)}
    return (2.0 * len(ba & bb)) / (len(ba) + len(bb))


def label_similarity(query: str, candidate: str) -> float:
    """Weighted `0.7 * levenshtein_ratio + 0.3 * phonetic_bigram_sim` after folding."""
    qa = fold_text(query)
    cb = fold_text(candidate)
    if not qa or not cb:
        return 0.0
    if qa == cb:
        return 1.0
    lev = levenshtein_ratio(qa, cb)
    pho = phonetic_bigram_sim(phonetic_fold(qa), phonetic_fold(cb))
    return FUZZY_LEVENSHTEIN_WEIGHT * lev + FUZZY_PHONETIC_WEIGHT * pho


def match_hosts_by_query(query: str, hosts: Sequence[Host]) -> HostQueryMatch:
    """Safe unique-only matching across active hosts' `{alias, nicknames, place}` (H3)."""
    active_hosts = [h for h in hosts if h.status == HostStatus.ACTIVE]
    q_folded = fold_text(query)
    if not q_folded or not active_hosts:
        return HostQueryMatch(status="no_match", candidates=tuple(active_hosts))

    alias_exact = [
        h
        for h in active_hosts
        if (h.alias_folded or fold_text(h.alias.value)) == q_folded
        or fold_text(h.alias.value) == q_folded
    ]
    if len(alias_exact) == 1:
        h = alias_exact[0]
        return HostQueryMatch(
            status="unique",
            host=h,
            candidates=(h,),
            match_source="alias",
            scores={h.host_id: 1.0},
        )
    if len(alias_exact) > 1:
        return HostQueryMatch(
            status="ambiguous",
            candidates=tuple(alias_exact),
            scores={h.host_id: 1.0 for h in alias_exact},
        )

    nick_exact = [h for h in active_hosts if any(fold_text(n) == q_folded for n in h.nicknames)]
    place_exact = [
        h for h in active_hosts if h.place is not None and fold_text(h.place) == q_folded
    ]
    exact_other: list[Host] = []
    seen_ids: set[HostId] = set()
    for h in (*nick_exact, *place_exact):
        if h.host_id not in seen_ids:
            seen_ids.add(h.host_id)
            exact_other.append(h)

    if len(exact_other) == 1:
        h = exact_other[0]
        src: Literal["nickname", "place"] = "nickname" if h in nick_exact else "place"
        return HostQueryMatch(
            status="unique",
            host=h,
            candidates=(h,),
            match_source=src,
            scores={h.host_id: 1.0},
        )
    if len(exact_other) > 1:
        return HostQueryMatch(
            status="ambiguous",
            candidates=tuple(exact_other),
            scores={h.host_id: 1.0 for h in exact_other},
        )

    if len(q_folded) < 2 or is_target_pronoun(q_folded):
        return HostQueryMatch(
            status="no_match",
            candidates=(),
            scores={h.host_id: 0.0 for h in active_hosts},
        )

    scored_pairs: list[tuple[float, str, Host]] = []
    scores_map: dict[HostId, float] = {}
    for h in active_hosts:
        labels = [h.alias.value, *h.nicknames]
        if h.place:
            labels.append(h.place)
        best = max(label_similarity(q_folded, lbl) for lbl in labels)
        scores_map[h.host_id] = best
        scored_pairs.append((best, h.alias.value.lower(), h))

    scored_pairs.sort(key=lambda item: (-item[0], item[1]))
    top_score, _, top_host = scored_pairs[0]
    second_score = scored_pairs[1][0] if len(scored_pairs) > 1 else 0.0
    if top_score >= FUZZY_MIN_SCORE and (
        len(scored_pairs) == 1 or (top_score - second_score) >= FUZZY_MIN_MARGIN
    ):
        return HostQueryMatch(
            status="unique",
            host=top_host,
            candidates=(top_host,),
            match_source="fuzzy",
            scores=scores_map,
        )

    close = [h for s, _, h in scored_pairs if s >= 0.55]
    if not close:
        return HostQueryMatch(status="no_match", candidates=(), scores=scores_map)
    return HostQueryMatch(status="ambiguous", candidates=tuple(close[:5]), scores=scores_map)


match_host_query = match_hosts_by_query


def extract_resource_query(tool_name: str, arguments: Mapping[str, object]) -> str | None:
    """Extracts a normalized resource query (`id`, `project`, etc.) without reading URLs."""
    keys = (
        (*_RESOURCE_ARG_KEYS, "query") if tool_name == "mesh__find_resource" else _RESOURCE_ARG_KEYS
    )
    for key in keys:
        val = arguments.get(key)
        if isinstance(val, str):
            if any(ord(ch) < 32 or ord(ch) == 127 for ch in val):
                return None
            folded = fold_text(val)
            if folded:
                return folded
    return None


def score_resource_match(
    descriptor: ToolDescriptor,
    resources: HostResources | None,
    presence: HostPresence | None,
    resource_query: str | None,
) -> float:
    """Computes `M_res` in `[0.0, 1.0]` (§5.4.1 #1)."""
    if resource_query is None:
        return 0.70 if descriptor.permission == ToolPermissionState.GRANTED else 0.30
    q = resource_query.lower()
    if (
        presence is not None
        and presence.foreground_app_id is not None
        and fold_text(presence.foreground_app_id) == q
    ):
        return 1.00
    if resources is None:
        return 0.50
    pool = [fold_text(item) for item in (*resources.installed_apps, *resources.workspaces)]
    if q in pool:
        return 1.00
    matches = {item for item in pool if len(q) >= 2 and (item.startswith(q) or q in item)}
    if len(matches) == 1:
        return 0.85
    return 0.25


def score_live_presence(
    host_kind: HostKind,
    family: ToolFamily,
    *,
    online: bool,
    presence: HostPresence | None,
    client_subnet_hash: str | None,
) -> float:
    """Computes `M_pres` in `[0.0, 1.0]` with `media` weighting and interactive lock penalty."""
    if not online:
        return 0.0
    if presence is None:
        return 0.40
    is_interactive = host_kind in _INTERACTIVE_HOST_KINDS
    lock_factor = 0.15 if (is_interactive and presence.screen_locked) else 1.0
    r_idle = math.exp(-presence.idle_seconds / 300.0) if is_interactive else 1.0
    if client_subnet_hash and presence.subnet_hash:
        r_net = 1.0 if client_subnet_hash == presence.subnet_hash else 0.2
    else:
        r_net = 0.5

    if family == ToolFamily.MEDIA:
        r_media = 1.0 if presence.media_playing else 0.0
        score = 0.60 * r_media + 0.20 * lock_factor + 0.10 * r_idle + 0.10 * r_net
    else:
        score = 0.35 * lock_factor + 0.40 * r_idle + 0.25 * r_net

    if is_interactive and presence.screen_locked and family in _INTERACTIVE_LOCK_FAMILIES:
        score *= 0.35
    return max(0.0, min(1.0, score))


def score_session_continuity(
    host: Host,
    *,
    has_active_cu_lane: bool,
    is_focused: bool,
    last_target_alias: str | None,
    last_target_at: datetime | None,
    now: datetime,
    default_host_id: HostId | None,
) -> float:
    """Computes `M_cont` in `[0.0, 1.0]` (§5.4.1 #4)."""
    if has_active_cu_lane or is_focused:
        return 1.00
    if last_target_alias and last_target_alias.strip():
        recent_ok = True if last_target_at is None else (now - last_target_at) <= LAST_TARGET_TTL
        if recent_ok and fold_text(last_target_alias) == (
            host.alias_folded or fold_text(host.alias.value)
        ):
            return 0.80
    if default_host_id is not None and default_host_id == host.host_id:
        return 0.65
    return 0.30


def combine_confidence(
    m_res: float,
    m_pres: float,
    m_hab: float,
    m_cont: float,
    *,
    follow_me: bool,
) -> float:
    """Combines component scores into `C(h)` under normal or `follow_me` weights (§5.4.2)."""
    if follow_me:
        return 0.35 * m_res + 0.40 * m_pres + 0.20 * m_hab + 0.05 * m_cont
    return 0.35 * m_res + 0.30 * m_pres + 0.20 * m_hab + 0.15 * m_cont


def _evaluate_host_eligibility(
    item: HostScoringCandidateInput,
    *,
    tool_name: str,
    now: datetime,
    host_screen_consent: bool,
) -> tuple[bool, ToolDescriptor | None, bool, CandidateUnavailableReason | None]:
    """Returns `(is_capable, descriptor, eligible, unavailable_reason)`."""
    if is_forbidden_tool_name(tool_name):
        return False, None, False, CandidateUnavailableReason.UNSUPPORTED
    if not item.online and item.manifest is None:
        return False, None, False, CandidateUnavailableReason.HOST_OFFLINE
    descriptor = item.manifest.find_tool(tool_name) if item.manifest is not None else None
    if descriptor is None or descriptor.permission == ToolPermissionState.UNSUPPORTED:
        return False, descriptor, False, CandidateUnavailableReason.UNSUPPORTED

    if not item.online:
        return True, descriptor, False, CandidateUnavailableReason.HOST_OFFLINE
    if item.policy_blocked:
        return True, descriptor, False, CandidateUnavailableReason.POLICY_BLOCKED
    if descriptor.permission == ToolPermissionState.DENIED:
        return True, descriptor, False, CandidateUnavailableReason.GRANT_DENIED
    if (
        descriptor.observes or descriptor.family in _U5_REQUIRED_FAMILIES
    ) and not host_screen_consent:
        return True, descriptor, False, CandidateUnavailableReason.U5_CONSENT_REQUIRED

    eff_risk = item.effective_risk or descriptor.risk
    matching_grant = next((g for g in item.grants if g.family == descriptor.family), None)
    if (
        matching_grant is None
        or not matching_grant.is_active(now)
        or not matching_grant.max_risk.allows(eff_risk)
    ):
        return True, descriptor, False, CandidateUnavailableReason.GRANT_DENIED
    return True, descriptor, True, None


def _presence_summary(host: Host, twin: HostTwin | None, online: bool) -> str:
    if not online:
        return f"{host.alias.value} is offline"
    p = twin.presence if twin is not None else None
    if p is None:
        return f"{host.alias.value} is online"
    bits: list[str] = ["locked" if p.screen_locked else "unlocked"]
    if p.media_playing:
        bits.append("media playing")
    bits.append(f"active {p.idle_seconds}s ago")
    return ", ".join(bits)


def _make_unavailable_candidate(
    item: HostScoringCandidateInput,
    reason_enum: CandidateUnavailableReason,
) -> ScoredHostCandidate:
    host = item.host
    return ScoredHostCandidate(
        host=host,
        score=0.0,
        eligible=False,
        online=item.online,
        unavailable_reason=reason_enum,
        reason=_presence_summary(host, item.twin if item.online else None, item.online),
        wol_mac=host.wol_mac,
        wol_available=bool(host.wol_mac),
    )


def _score_eligible_item(
    item: HostScoringCandidateInput,
    descriptor: ToolDescriptor,
    *,
    ctx: MeshScoringContext,
    resource_query: str | None,
    day_part: str,
    follow_me: bool,
) -> ScoredHostCandidate:
    host = item.host
    twin = item.twin if item.online else None
    presence = twin.presence if twin is not None else None
    resources = twin.resources if twin is not None else None

    m_res = score_resource_match(descriptor, resources, presence, resource_query)
    m_pres = score_live_presence(
        host.host_kind,
        descriptor.family,
        online=item.online,
        presence=presence,
        client_subnet_hash=ctx.client_subnet_hash,
    )
    habit_res = lookup_habit_score(
        ctx.habits_by_key_host,
        family=descriptor.family,
        resource_id=resource_query,
        day_part=day_part,
        host_id=host.host_id,
    )
    m_hab = habit_res.score
    focused_id = ctx.focus_state.focused_host_id if ctx.focus_state is not None else None
    m_cont = score_session_continuity(
        host,
        has_active_cu_lane=item.has_active_cu_lane,
        is_focused=(focused_id is not None and focused_id == host.host_id),
        last_target_alias=ctx.last_target_alias,
        last_target_at=ctx.last_target_at,
        now=ctx.now,
        default_host_id=ctx.default_host_id,
    )
    total = combine_confidence(m_res, m_pres, m_hab, m_cont, follow_me=follow_me)
    return ScoredHostCandidate(
        host=host,
        score=total,
        eligible=True,
        online=True,
        unavailable_reason=None,
        reason=_presence_summary(host, twin, True),
        wol_mac=host.wol_mac,
        wol_available=bool(host.wol_mac),
        m_res=m_res,
        m_pres=m_pres,
        m_hab=m_hab,
        m_cont=m_cont,
    )


def _select_positive_reason(
    cand: ScoredHostCandidate,
    *,
    follow_me: bool,
) -> TargetReason:
    if follow_me and cand.m_pres >= 0.70:
        return TargetReason(
            reason_code=TargetReasonCode.FOLLOW_ME_ACTIVE,
            summary=f"Follow Me active · {cand.reason}",
        )
    if cand.m_res >= 0.85 or (cand.m_res >= 0.70 and cand.m_pres >= 0.60):
        return TargetReason(
            reason_code=TargetReasonCode.RESOURCE_AND_PRESENCE_MATCH,
            summary=f"Best resource & presence match · {cand.reason}",
        )
    if cand.m_hab >= 0.65:
        return TargetReason(
            reason_code=TargetReasonCode.HIGH_CONFIDENCE_HABIT_AND_PRESENCE,
            summary=f"Learned contextual habit · {cand.reason}",
        )
    if cand.m_cont >= 0.65:
        return TargetReason(
            reason_code=TargetReasonCode.SESSION_CONTINUITY,
            summary=f"Active session continuity · {cand.reason}",
        )
    return TargetReason(
        reason_code=TargetReasonCode.RESOURCE_AND_PRESENCE_MATCH,
        summary=f"Confidence {cand.score:.2f} · {cand.reason}",
    )


def score_hosts_for_tool(
    candidates: Sequence[HostScoringCandidateInput],
    ctx: MeshScoringContext,
) -> MeshScoringDecision:
    """Evaluates active hosts against a tool call using exhaustive pre-ask_target checks."""
    active_items = [item for item in candidates if item.host.status == HostStatus.ACTIVE]
    resource_query = extract_resource_query(ctx.tool_name, ctx.arguments)
    day_part = day_part_for_dt(ctx.now, ctx.client_tz_offset_minutes)
    follow_me = ctx.focus_state.follow_me if ctx.focus_state is not None else False

    if ctx.explicit_requested_host_id is not None:
        explicit_item = next(
            (item for item in active_items if item.host.host_id == ctx.explicit_requested_host_id),
            None,
        )
        if explicit_item is None:
            return MeshScoringDecision(
                resolved=False,
                host=None,
                target_reason=TargetReason(
                    reason_code=TargetReasonCode.NO_HOST,
                    summary="Requested host is not active",
                ),
                action_required_reason=ActionTargetRequiredReason.NO_HOST,
                candidates=(),
            )
        if not explicit_item.online and explicit_item.manifest is None:
            cand = _make_unavailable_candidate(
                explicit_item, CandidateUnavailableReason.HOST_OFFLINE
            )
            return MeshScoringDecision(
                resolved=False,
                host=None,
                target_reason=TargetReason(
                    reason_code=TargetReasonCode.HOST_OFFLINE,
                    summary=f"{explicit_item.host.alias.value} is offline",
                ),
                action_required_reason=ActionTargetRequiredReason.HOST_OFFLINE,
                candidates=(cand,),
            )
        capable, descriptor, eligible, unavail = _evaluate_host_eligibility(
            explicit_item,
            tool_name=ctx.tool_name,
            now=ctx.now,
            host_screen_consent=ctx.host_screen_consent,
        )
        if not capable or descriptor is None:
            cand = _make_unavailable_candidate(
                explicit_item, unavail or CandidateUnavailableReason.UNSUPPORTED
            )
            return MeshScoringDecision(
                resolved=False,
                host=None,
                target_reason=TargetReason(
                    reason_code=TargetReasonCode.NO_HOST,
                    summary=f"{explicit_item.host.alias.value} does not support {ctx.tool_name}",
                ),
                action_required_reason=ActionTargetRequiredReason.NO_HOST,
                candidates=(cand,),
            )
        if not explicit_item.online:
            cand = _make_unavailable_candidate(
                explicit_item, CandidateUnavailableReason.HOST_OFFLINE
            )
            return MeshScoringDecision(
                resolved=False,
                host=None,
                target_reason=TargetReason(
                    reason_code=TargetReasonCode.HOST_OFFLINE,
                    summary=f"{explicit_item.host.alias.value} is offline",
                ),
                action_required_reason=ActionTargetRequiredReason.HOST_OFFLINE,
                candidates=(cand,),
            )
        if not eligible:
            cand = _make_unavailable_candidate(
                explicit_item, unavail or CandidateUnavailableReason.GRANT_DENIED
            )
            return MeshScoringDecision(
                resolved=False,
                host=None,
                target_reason=TargetReason(
                    reason_code=TargetReasonCode.GRANT_DENIED,
                    summary=(
                        f"{explicit_item.host.alias.value} blocked by {cand.unavailable_reason}"
                    ),
                ),
                action_required_reason=ActionTargetRequiredReason.GRANT_DENIED,
                candidates=(cand,),
            )
        scored_explicit = _score_eligible_item(
            explicit_item,
            descriptor,
            ctx=ctx,
            resource_query=resource_query,
            day_part=day_part,
            follow_me=follow_me,
        )
        reason_code = ctx.explicit_reason_code or TargetReasonCode.PINNED_HOST
        return MeshScoringDecision(
            resolved=True,
            host=explicit_item.host,
            target_reason=TargetReason(
                reason_code=reason_code,
                summary=(
                    f"Explicit target {explicit_item.host.alias.value} · {scored_explicit.reason}"
                ),
            ),
            action_required_reason=None,
            candidates=(scored_explicit,),
        )

    capable_items: list[
        tuple[
            HostScoringCandidateInput,
            ToolDescriptor,
            bool,
            CandidateUnavailableReason | None,
        ]
    ] = []
    for item in active_items:
        capable, descriptor, eligible, unavail = _evaluate_host_eligibility(
            item,
            tool_name=ctx.tool_name,
            now=ctx.now,
            host_screen_consent=ctx.host_screen_consent,
        )
        if capable and descriptor is not None:
            capable_items.append((item, descriptor, eligible, unavail))

    if not capable_items:
        return MeshScoringDecision(
            resolved=False,
            host=None,
            target_reason=TargetReason(
                reason_code=TargetReasonCode.NO_HOST,
                summary="No active enrolled host advertises this tool",
            ),
            action_required_reason=ActionTargetRequiredReason.NO_HOST,
            candidates=(),
        )

    online_capable_count = sum(1 for item, _, _, _ in capable_items if item.online)
    eligible_pairs = [(item, desc) for item, desc, eligible, _ in capable_items if eligible]

    if not eligible_pairs:
        unavail_cands = [
            _make_unavailable_candidate(item, unavail or CandidateUnavailableReason.GRANT_DENIED)
            for item, _, _, unavail in capable_items
        ]
        if online_capable_count == 0:
            return MeshScoringDecision(
                resolved=False,
                host=None,
                target_reason=TargetReason(
                    reason_code=TargetReasonCode.HOST_OFFLINE,
                    summary="All capable hosts are offline",
                ),
                action_required_reason=ActionTargetRequiredReason.HOST_OFFLINE,
                candidates=tuple(unavail_cands[:5]),
            )
        unavail_cands.sort(
            key=lambda cand: (
                0 if cand.online else 1,
                cand.host.alias.value.lower(),
            )
        )
        return MeshScoringDecision(
            resolved=False,
            host=None,
            target_reason=TargetReason(
                reason_code=TargetReasonCode.GRANT_DENIED,
                summary="All online capable hosts lack required grants or consent",
            ),
            action_required_reason=ActionTargetRequiredReason.GRANT_DENIED,
            candidates=tuple(unavail_cands[:5]),
        )

    scored_eligible: list[tuple[ScoredHostCandidate, RiskTier]] = []
    for item, desc in eligible_pairs:
        cand = _score_eligible_item(
            item,
            desc,
            ctx=ctx,
            resource_query=resource_query,
            day_part=day_part,
            follow_me=follow_me,
        )
        eff_risk = item.effective_risk or desc.risk
        scored_eligible.append((cand, eff_risk))

    scored_eligible.sort(key=lambda pair: (-pair[0].score, pair[0].host.alias.value.lower()))

    if len(scored_eligible) == 1:
        c1, risk1 = scored_eligible[0]
        h1 = c1.host
        item1 = eligible_pairs[0][0]
        p1 = item1.twin.presence if item1.twin is not None else None
        if (
            risk1 in (RiskTier.R2, RiskTier.R3, RiskTier.R4)
            and h1.host_kind in _INTERACTIVE_HOST_KINDS
            and p1 is not None
            and p1.screen_locked is True
        ):
            return MeshScoringDecision(
                resolved=False,
                host=None,
                target_reason=TargetReason(
                    reason_code=TargetReasonCode.AMBIGUOUS_CANDIDATES,
                    summary=(
                        f"{h1.alias.value} is locked for {risk1.value} action; "
                        "owner confirmation required"
                    ),
                ),
                action_required_reason=ActionTargetRequiredReason.ASK_TARGET,
                candidates=(c1,),
            )
        return MeshScoringDecision(
            resolved=True,
            host=h1,
            target_reason=TargetReason(
                reason_code=TargetReasonCode.UNIQUE_CAPABLE_HOST,
                summary=f"Only eligible online host · {c1.reason}",
            ),
            action_required_reason=None,
            candidates=(c1,),
        )

    c1, _ = scored_eligible[0]
    c2, _ = scored_eligible[1]
    effective_tier = max((risk for _, risk in scored_eligible), key=lambda tier: tier.level)
    theta, delta = RISK_THRESHOLDS[effective_tier]
    if c1.score >= theta and (c1.score - c2.score) >= delta:
        return MeshScoringDecision(
            resolved=True,
            host=c1.host,
            target_reason=_select_positive_reason(c1, follow_me=follow_me),
            action_required_reason=None,
            candidates=tuple(c for c, _ in scored_eligible[:5]),
        )

    return MeshScoringDecision(
        resolved=False,
        host=None,
        target_reason=TargetReason(
            reason_code=TargetReasonCode.AMBIGUOUS_CANDIDATES,
            summary="Multiple eligible hosts have similar confidence",
        ),
        action_required_reason=ActionTargetRequiredReason.ASK_TARGET,
        candidates=tuple(c for c, _ in scored_eligible[:5]),
    )
