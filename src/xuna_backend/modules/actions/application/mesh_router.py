"""Deterministic Pre-Router, PendingIntent store, Host Roster & scorer bridge (§5.2.2, §5.3).

Runs deterministically before LLM calls (`targetHostId`, `/may`, `/follow`, `@tag`,
`PendingIntent` 1-word repair, `HostFocusState`), formats the dynamic `[HOST_ROSTER]`,
and bridges `Orchestrator._handle_host_call` to `mesh_scoring` + `habit_learner` without
bloating `orchestrator.py`. Zero hardcoded app names, host names, OS paths, or
keyword-intent tables (H5).
"""

import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal

from xuna_backend.modules.actions.application.ports import (
    ComputerUseSessionStore,
    HabitRepository,
    LastTargetStore,
)
from xuna_backend.modules.actions.domain.entities import ActionTargetRequiredEvent
from xuna_backend.modules.actions.domain.habit_learner import (
    HabitReasonCode,
    apply_dual_bucket_observation,
    context_keys_for_update,
    day_part_for_dt,
)
from xuna_backend.modules.actions.domain.habits import HostRoutingHabit
from xuna_backend.modules.actions.domain.mesh_scoring import (
    ActionTargetRequiredReason,
    CandidateUnavailableReason,
    HostQueryMatch,
    HostScoringCandidateInput,
    MeshScoringContext,
    MeshScoringDecision,
    ScoredHostCandidate,
    TargetReason,
    TargetReasonCode,
    extract_resource_query,
    match_hosts_by_query,
    score_hosts_for_tool,
)
from xuna_backend.modules.actions.domain.personalization import InMemoryHostDefaultStore
from xuna_backend.modules.actions.domain.target_resolver import (
    TargetResolutionStatus,
    TargetResolver,
    is_target_pronoun,
)
from xuna_backend.modules.actions.domain.wake_queue import InMemoryWakeQueueStore
from xuna_backend.modules.hosts.application.ports import HostSessionHub
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    Host,
    HostId,
    HostStatus,
    Manifest,
    ToolDescriptor,
    ToolFamily,
    fold_text,
    validate_tool_name,
)
from xuna_backend.modules.hosts.domain.twin import (
    HostFocusChangedNotice,
    HostFocusState,
    HostTwin,
)

if TYPE_CHECKING:
    from xuna_backend.modules.actions.application.orchestrator import ActionContext

PENDING_INTENT_TTL = timedelta(seconds=120)

_AT_TAG_RE = re.compile(r'(?:^|(?<=\s)|(?<=[\(\[\{]))@(?:(?:"([^"]*)")|([^\s,:;!?()\[\]{}"]+))')
_SLASH_MAY_RE = re.compile(r"^/may(?:\s+(.*))?$", re.IGNORECASE)
_SLASH_FOLLOW_RE = re.compile(r"^/follow(?:\s+(\S+))?$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class PendingIntent:
    """Short-TTL interrupted tool intent awaiting a 1-word host repair turn."""

    owner_id: uuid.UUID
    pet_id: str
    original_tool: str
    original_arguments: dict[str, object]
    original_text: str
    candidate_host_ids: tuple[HostId, ...]
    created_at: datetime
    suggested_host_id: HostId | None = None
    resource_id: str | None = None
    family: str | None = None
    client_caps: tuple[str, ...] = ()


class PendingIntentStore:
    """RAM-only `120 s` TTL store keyed by `(owner_id, pet_id)` for `ask_target` repair."""

    def __init__(self) -> None:
        self._entries: dict[tuple[uuid.UUID, str], PendingIntent] = {}

    def save(self, intent: PendingIntent) -> None:
        copied = PendingIntent(
            owner_id=intent.owner_id,
            pet_id=intent.pet_id,
            original_tool=intent.original_tool,
            original_arguments=dict(intent.original_arguments),
            original_text=intent.original_text,
            candidate_host_ids=tuple(intent.candidate_host_ids),
            created_at=intent.created_at,
            suggested_host_id=intent.suggested_host_id,
            resource_id=intent.resource_id,
            family=intent.family,
            client_caps=tuple(intent.client_caps),
        )
        self._entries[(intent.owner_id, intent.pet_id)] = copied

    def get(self, owner_id: uuid.UUID, pet_id: str, now: datetime) -> PendingIntent | None:
        key = (owner_id, pet_id)
        entry = self._entries.get(key)
        if entry is None:
            return None
        if (now - entry.created_at) > PENDING_INTENT_TTL:
            self._entries.pop(key, None)
            return None
        return entry

    def clear(self, owner_id: uuid.UUID, pet_id: str) -> bool:
        return self._entries.pop((owner_id, pet_id), None) is not None

    def clear_owner(self, owner_id: uuid.UUID) -> int:
        keys = [k for k in self._entries if k[0] == owner_id]
        for k in keys:
            self._entries.pop(k, None)
        return len(keys)


InMemoryPendingIntentStore = PendingIntentStore


@dataclass(frozen=True, slots=True)
class PreRouteOutcome:
    """Deterministic pre-routing result produced before invoking the LLM."""

    kind: Literal["pass", "pin", "slash_handled", "target_required", "repair_dispatch", "not_found"]
    pinned_host_id: HostId | None = None
    pinned_host: Host | None = None
    focus_notice: HostFocusChangedNotice | None = None
    reason_code: str | None = None
    target_reason: TargetReason | None = None
    candidates: tuple[ScoredHostCandidate, ...] = ()
    replay_tool: str | None = None
    replay_arguments: dict[str, object] | None = None
    cleaned_text: str | None = None
    habit_recorded_in_preroute: bool = False
    replay_client_caps: tuple[str, ...] = ()
    skip_focus_pin: bool = False


@dataclass(frozen=True, slots=True)
class HostCallRoutingResult:
    """Target resolution bridge output consumed by `Orchestrator._handle_host_call`."""

    resolved: bool
    host: Host | None = None
    error_status: str | None = None
    error_extra: dict[str, object] | None = None
    choices: tuple[str, ...] = ()
    target_reason: TargetReason | None = None
    scoring_decision: MeshScoringDecision | None = None


def _hub_get_twin(hub: HostSessionHub, host_id: str) -> HostTwin | None:
    fn = getattr(hub, "get_twin", None)
    if callable(fn):
        twin = fn(host_id)
        if isinstance(twin, HostTwin):
            return twin
    return None


def _hub_get_focus_state(hub: HostSessionHub, owner_id: uuid.UUID) -> HostFocusState | None:
    fn = getattr(hub, "get_focus_state", None)
    if callable(fn):
        state = fn(owner_id)
        if isinstance(state, HostFocusState):
            return state
    return None


def _hub_set_focus_state(
    hub: HostSessionHub,
    owner_id: uuid.UUID,
    *,
    focused_host_id: HostId | None = None,
    keep_focused_host: bool = False,
    follow_me: bool | None = None,
    source: str,
    now: datetime,
) -> HostFocusState:
    fn = getattr(hub, "set_focus_state", None)
    if callable(fn):
        state = fn(
            owner_id,
            focused_host_id=focused_host_id,
            keep_focused_host=keep_focused_host,
            follow_me=follow_me,
            source=source,
            now=now,
        )
        if isinstance(state, HostFocusState):
            return state
    prev = _hub_get_focus_state(hub, owner_id)
    eff_host = prev.focused_host_id if (keep_focused_host and prev is not None) else focused_host_id
    eff_follow = (
        follow_me if follow_me is not None else (prev.follow_me if prev is not None else False)
    )
    return HostFocusState(
        owner_id=owner_id,
        focused_host_id=eff_host,
        follow_me=eff_follow,
        updated_at=now,
        source=source,
    )


async def _hub_publish_focus_changed(hub: HostSessionHub, notice: HostFocusChangedNotice) -> None:
    fn = getattr(hub, "publish_focus_changed", None)
    if callable(fn):
        await fn(notice)


def _is_host_online(hub: HostSessionHub, host: Host, self_host_id: str | None) -> bool:
    hid_str = str(host.host_id)
    return hub.is_online(hid_str) or (self_host_id is not None and hid_str == self_host_id)


def _build_candidate_from_host(
    host: Host,
    *,
    online: bool,
    unavailable_reason: CandidateUnavailableReason | None,
    score: float = 0.0,
) -> ScoredHostCandidate:
    return ScoredHostCandidate(
        host=host,
        score=score,
        eligible=(unavailable_reason is None and online),
        online=online,
        unavailable_reason=unavailable_reason,
        reason=(f"{host.alias.value} is online" if online else f"{host.alias.value} is offline"),
        wol_mac=host.wol_mac,
        wol_available=bool(host.wol_mac),
    )


def _candidates_for_hosts(
    hosts: Sequence[Host],
    hub: HostSessionHub,
    self_host_id: str | None,
) -> tuple[ScoredHostCandidate, ...]:
    out: list[ScoredHostCandidate] = []
    for h in hosts[:5]:
        online = _is_host_online(hub, h, self_host_id)
        out.append(
            _build_candidate_from_host(
                h,
                online=online,
                unavailable_reason=None if online else CandidateUnavailableReason.HOST_OFFLINE,
            )
        )
    return tuple(out)


def _strip_single_tag(text: str, match: re.Match[str]) -> str:
    before = text[: match.start()].rstrip()
    after = text[match.end() :].lstrip(",;:")
    if after and after[0] in ".!?:;)]}\"'":
        joined = f"{before}{after}".strip()
    else:
        joined = f"{before} {after}".strip()
    collapsed = " ".join(joined.split())
    return collapsed or text.strip()


def _extract_tag_query(match: re.Match[str]) -> str:
    if match.group(1) is not None:
        return match.group(1).strip()
    raw = (match.group(2) or "").strip()
    return raw.rstrip(".,;:!?\"'()\u2019[]{}")


def _resolve_pending_family(pending: PendingIntent) -> str:
    if pending.family:
        return pending.family
    try:
        return validate_tool_name(pending.original_tool).value
    except ValueError:
        return pending.original_tool.split(".", 1)[0]


async def _record_habit_updates(
    habit_repo: HabitRepository,
    *,
    owner_id: uuid.UUID,
    pet_id: str,
    family: ToolFamily | str,
    resource_id: str | None,
    host_id: HostId,
    positive: bool,
    now: datetime,
    reason_code: HabitReasonCode,
    tz_offset_minutes: int | None = None,
) -> None:
    day_part = day_part_for_dt(now, tz_offset_minutes)
    keys = context_keys_for_update(family, resource_id, day_part)
    existing_map: dict[tuple[str, HostId], HostRoutingHabit] = {}
    for ctx_key in keys:
        row = await habit_repo.get_habit(owner_id, pet_id, ctx_key, host_id)
        if row is not None:
            existing_map[(ctx_key, host_id)] = row
    updated_rows = apply_dual_bucket_observation(
        existing_map,
        owner_id=owner_id,
        pet_id=pet_id,
        family=family,
        resource_id=resource_id,
        day_part=day_part,
        host_id=host_id,
        positive=positive,
        now=now,
        reason_code=reason_code,
    )
    for row in updated_rows:
        await habit_repo.upsert_habit(row)


class MeshPreRouter:
    """Deterministic Pre-Router executed before any LLM call (§5.3)."""

    def __init__(
        self,
        *,
        pending_store: PendingIntentStore | None = None,
        habit_repo: HabitRepository | None = None,
        wake_queue: InMemoryWakeQueueStore | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._pending_store = pending_store
        self._habit_repo = habit_repo
        self._wake_queue = wake_queue
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def wake_queue(self) -> InMemoryWakeQueueStore | None:
        return self._wake_queue

    def clock_now(self) -> datetime:
        return self._clock()

    def has_pending_intent(
        self, owner_id: uuid.UUID, pet_id: str, now: datetime | None = None
    ) -> bool:
        if self._pending_store is None:
            return False
        return self._pending_store.get(owner_id, pet_id, now or self._clock()) is not None

    async def route(
        self,
        *,
        owner_id: uuid.UUID,
        pet_id: str,
        text: str,
        hosts: Sequence[Host],
        hub: HostSessionHub,
        target_host_id: HostId | None = None,
        follow_me: bool | None = None,
        self_host_id: str | None = None,
        now: datetime | None = None,
        client_tz_offset_minutes: int | None = None,
    ) -> PreRouteOutcome:
        check_now = now or self._clock()
        active_hosts = [h for h in hosts if h.status == HostStatus.ACTIVE]

        # Priority 1: Explicit targetHostId (validate ownership before mutating focus state)
        if target_host_id is not None:
            match_host = next((h for h in active_hosts if h.host_id == target_host_id), None)
            if match_host is None:
                return PreRouteOutcome(kind="not_found")
            if follow_me is not None:
                _hub_set_focus_state(
                    hub,
                    owner_id,
                    keep_focused_host=True,
                    follow_me=follow_me,
                    source="app",
                    now=check_now,
                )
            if self._pending_store is not None:
                pending = self._pending_store.get(owner_id, pet_id, check_now)
                self._pending_store.clear(owner_id, pet_id)
                if (
                    pending is not None
                    and self._habit_repo is not None
                    and pending.suggested_host_id is not None
                    and pending.suggested_host_id != match_host.host_id
                ):
                    family = _resolve_pending_family(pending)
                    await _record_habit_updates(
                        self._habit_repo,
                        owner_id=owner_id,
                        pet_id=pet_id,
                        family=family,
                        resource_id=pending.resource_id,
                        host_id=pending.suggested_host_id,
                        positive=False,
                        now=check_now,
                        reason_code=HabitReasonCode.REDIRECT_PENALTY,
                        tz_offset_minutes=client_tz_offset_minutes,
                    )
            if not _is_host_online(hub, match_host, self_host_id):
                cand = _build_candidate_from_host(
                    match_host,
                    online=False,
                    unavailable_reason=CandidateUnavailableReason.HOST_OFFLINE,
                )
                return PreRouteOutcome(
                    kind="target_required",
                    reason_code=ActionTargetRequiredReason.HOST_OFFLINE.value,
                    target_reason=TargetReason(
                        reason_code=TargetReasonCode.HOST_OFFLINE,
                        summary=f"{match_host.alias.value} is offline",
                    ),
                    candidates=(cand,),
                )
            return PreRouteOutcome(
                kind="pin",
                pinned_host_id=match_host.host_id,
                pinned_host=match_host,
                target_reason=TargetReason(
                    reason_code=TargetReasonCode.PINNED_HOST,
                    summary=f"Explicit target {match_host.alias.value}",
                ),
            )

        if follow_me is not None:
            _hub_set_focus_state(
                hub,
                owner_id,
                keep_focused_host=True,
                follow_me=follow_me,
                source="app",
                now=check_now,
            )

        stripped = text.strip()

        # Priority 2: Slash commands (/may <alias|auto> and /follow <on|off>)
        may_match = _SLASH_MAY_RE.match(stripped)
        if may_match is not None:
            if self._pending_store is not None:
                self._pending_store.clear(owner_id, pet_id)
            raw_arg = (may_match.group(1) or "").strip().strip('"').strip("'")
            if fold_text(raw_arg) == "auto":
                state = _hub_set_focus_state(
                    hub,
                    owner_id,
                    focused_host_id=None,
                    keep_focused_host=False,
                    follow_me=None,
                    source="slash_cmd",
                    now=check_now,
                )
                notice = HostFocusChangedNotice(
                    owner_id=owner_id,
                    host_id=None,
                    alias=None,
                    platform=None,
                    follow_me=state.follow_me,
                    source="slash_cmd",
                    updated_at=check_now,
                )
                await _hub_publish_focus_changed(hub, notice)
                return PreRouteOutcome(kind="slash_handled", focus_notice=notice)

            if not raw_arg:
                return PreRouteOutcome(
                    kind="target_required",
                    reason_code=ActionTargetRequiredReason.ASK_TARGET.value,
                    target_reason=TargetReason(
                        reason_code=TargetReasonCode.AMBIGUOUS_CANDIDATES,
                        summary="Choose which host to focus",
                    ),
                    candidates=_candidates_for_hosts(active_hosts, hub, self_host_id),
                )

            query_match = match_hosts_by_query(raw_arg, active_hosts)
            if query_match.status == "unique" and query_match.host is not None:
                chosen = query_match.host
                state = _hub_set_focus_state(
                    hub,
                    owner_id,
                    focused_host_id=chosen.host_id,
                    keep_focused_host=False,
                    follow_me=None,
                    source="slash_cmd",
                    now=check_now,
                )
                notice = HostFocusChangedNotice(
                    owner_id=owner_id,
                    host_id=chosen.host_id,
                    alias=chosen.alias.value,
                    platform=chosen.platform.value,
                    follow_me=state.follow_me,
                    source="slash_cmd",
                    updated_at=check_now,
                )
                await _hub_publish_focus_changed(hub, notice)
                return PreRouteOutcome(
                    kind="slash_handled",
                    pinned_host_id=chosen.host_id,
                    pinned_host=chosen,
                    focus_notice=notice,
                    target_reason=TargetReason(
                        reason_code=TargetReasonCode.PINNED_HOST,
                        summary=f"Focused {chosen.alias.value} via /may",
                    ),
                )

            return PreRouteOutcome(
                kind="target_required",
                reason_code=ActionTargetRequiredReason.ASK_TARGET.value,
                target_reason=TargetReason(
                    reason_code=TargetReasonCode.AMBIGUOUS_CANDIDATES,
                    summary="Multiple or unclear host matches for /may",
                ),
                candidates=_candidates_for_hosts(
                    query_match.candidates or active_hosts, hub, self_host_id
                ),
            )

        follow_match = _SLASH_FOLLOW_RE.match(stripped)
        if follow_match is not None:
            raw_arg = (follow_match.group(1) or "").strip().lower()
            if raw_arg in ("on", "off"):
                if self._pending_store is not None:
                    self._pending_store.clear(owner_id, pet_id)
                state = _hub_set_focus_state(
                    hub,
                    owner_id,
                    keep_focused_host=True,
                    follow_me=(raw_arg == "on"),
                    source="slash_cmd",
                    now=check_now,
                )
                focused = (
                    next((h for h in active_hosts if h.host_id == state.focused_host_id), None)
                    if state.focused_host_id is not None
                    else None
                )
                notice = HostFocusChangedNotice(
                    owner_id=owner_id,
                    host_id=focused.host_id if focused is not None else None,
                    alias=focused.alias.value if focused is not None else None,
                    platform=focused.platform.value if focused is not None else None,
                    follow_me=state.follow_me,
                    source="slash_cmd",
                    updated_at=check_now,
                )
                await _hub_publish_focus_changed(hub, notice)
                return PreRouteOutcome(
                    kind="slash_handled",
                    pinned_host_id=focused.host_id if focused is not None else None,
                    pinned_host=focused,
                    focus_notice=notice,
                )

        # Priority 3: Inline @alias / @place / @nickname tags
        tag_matches = list(_AT_TAG_RE.finditer(text))
        if len(tag_matches) == 1:
            if self._pending_store is not None:
                self._pending_store.clear(owner_id, pet_id)
            tag_m = tag_matches[0]
            tag_query = _extract_tag_query(tag_m)
            if not tag_query:
                return PreRouteOutcome(
                    kind="target_required",
                    reason_code=ActionTargetRequiredReason.ASK_TARGET.value,
                    target_reason=TargetReason(
                        reason_code=TargetReasonCode.AMBIGUOUS_CANDIDATES,
                        summary="Choose which host to target",
                    ),
                    candidates=_candidates_for_hosts(active_hosts, hub, self_host_id),
                )
            query_match = match_hosts_by_query(tag_query, active_hosts)
            if query_match.status == "unique" and query_match.host is not None:
                chosen = query_match.host
                if not _is_host_online(hub, chosen, self_host_id):
                    cand = _build_candidate_from_host(
                        chosen,
                        online=False,
                        unavailable_reason=CandidateUnavailableReason.HOST_OFFLINE,
                    )
                    return PreRouteOutcome(
                        kind="target_required",
                        reason_code=ActionTargetRequiredReason.HOST_OFFLINE.value,
                        target_reason=TargetReason(
                            reason_code=TargetReasonCode.HOST_OFFLINE,
                            summary=f"{chosen.alias.value} is offline",
                        ),
                        candidates=(cand,),
                    )
                r_code = (
                    TargetReasonCode.PLACE_TAG
                    if query_match.match_source == "place"
                    else TargetReasonCode.EXPLICIT_ALIAS
                )
                return PreRouteOutcome(
                    kind="pin",
                    pinned_host_id=chosen.host_id,
                    pinned_host=chosen,
                    cleaned_text=_strip_single_tag(text, tag_m),
                    target_reason=TargetReason(
                        reason_code=r_code,
                        summary=f"Matched @{tag_query} to {chosen.alias.value}",
                    ),
                )
            tag_summary = (
                f"Ambiguous host tag @{tag_query}"
                if query_match.status == "ambiguous"
                else f"No host matched @{tag_query}"
            )
            return PreRouteOutcome(
                kind="target_required",
                reason_code=ActionTargetRequiredReason.ASK_TARGET.value,
                target_reason=TargetReason(
                    reason_code=TargetReasonCode.AMBIGUOUS_CANDIDATES,
                    summary=tag_summary,
                ),
                candidates=_candidates_for_hosts(
                    query_match.candidates or active_hosts, hub, self_host_id
                ),
            )

        # Priority 4: PendingIntent 1-word Voice/Chat Repair (TTL = 120s)
        if self._pending_store is not None:
            pending = self._pending_store.get(owner_id, pet_id, check_now)
            if pending is not None:
                cleaned_reply = stripped.strip("\"'.!?,;:()[]{}")
                tokens = [t for t in cleaned_reply.split() if t]
                if 1 <= len(tokens) <= 4:
                    cand_hosts = [
                        h for h in active_hosts if h.host_id in pending.candidate_host_ids
                    ]
                    repair_match = match_hosts_by_query(cleaned_reply, cand_hosts)
                    if repair_match.status == "unique" and repair_match.host is not None:
                        chosen = repair_match.host
                        self._pending_store.clear(owner_id, pet_id)
                        if self._habit_repo is not None:
                            family = _resolve_pending_family(pending)
                            await _record_habit_updates(
                                self._habit_repo,
                                owner_id=owner_id,
                                pet_id=pet_id,
                                family=family,
                                resource_id=pending.resource_id,
                                host_id=chosen.host_id,
                                positive=True,
                                now=check_now,
                                reason_code=HabitReasonCode.CONFIRMED_PICK,
                                tz_offset_minutes=client_tz_offset_minutes,
                            )
                            if (
                                pending.suggested_host_id is not None
                                and pending.suggested_host_id != chosen.host_id
                            ):
                                await _record_habit_updates(
                                    self._habit_repo,
                                    owner_id=owner_id,
                                    pet_id=pet_id,
                                    family=family,
                                    resource_id=pending.resource_id,
                                    host_id=pending.suggested_host_id,
                                    positive=False,
                                    now=check_now,
                                    reason_code=HabitReasonCode.REDIRECT_PENALTY,
                                    tz_offset_minutes=client_tz_offset_minutes,
                                )
                        if not _is_host_online(hub, chosen, self_host_id):
                            cand = _build_candidate_from_host(
                                chosen,
                                online=False,
                                unavailable_reason=CandidateUnavailableReason.HOST_OFFLINE,
                            )
                            return PreRouteOutcome(
                                kind="target_required",
                                reason_code=ActionTargetRequiredReason.HOST_OFFLINE.value,
                                target_reason=TargetReason(
                                    reason_code=TargetReasonCode.HOST_OFFLINE,
                                    summary=f"{chosen.alias.value} is offline",
                                ),
                                candidates=(cand,),
                            )
                        return PreRouteOutcome(
                            kind="repair_dispatch",
                            pinned_host_id=chosen.host_id,
                            pinned_host=chosen,
                            replay_tool=pending.original_tool,
                            replay_arguments=dict(pending.original_arguments),
                            cleaned_text=pending.original_text or None,
                            target_reason=TargetReason(
                                reason_code=TargetReasonCode.EXPLICIT_ALIAS,
                                summary=f"Owner selected {chosen.alias.value}",
                            ),
                            habit_recorded_in_preroute=(self._habit_repo is not None),
                            replay_client_caps=pending.client_caps,
                        )
                self._pending_store.clear(owner_id, pet_id)

        if len(tag_matches) >= 2:
            return PreRouteOutcome(kind="pass", skip_focus_pin=True)

        # Priority 5: Active HostFocusState pin
        focus = _hub_get_focus_state(hub, owner_id)
        if focus is not None and focus.focused_host_id is not None:
            pin = next((h for h in active_hosts if h.host_id == focus.focused_host_id), None)
            if pin is not None:
                if not _is_host_online(hub, pin, self_host_id):
                    has_other_online = any(
                        _is_host_online(hub, h, self_host_id)
                        for h in active_hosts
                        if h.host_id != pin.host_id
                    )
                    if focus.follow_me and has_other_online:
                        return PreRouteOutcome(kind="pass", skip_focus_pin=True)
                    cand = _build_candidate_from_host(
                        pin,
                        online=False,
                        unavailable_reason=CandidateUnavailableReason.HOST_OFFLINE,
                    )
                    return PreRouteOutcome(
                        kind="target_required",
                        reason_code=ActionTargetRequiredReason.HOST_OFFLINE.value,
                        target_reason=TargetReason(
                            reason_code=TargetReasonCode.HOST_OFFLINE,
                            summary=f"{pin.alias.value} is offline",
                        ),
                        candidates=(cand,),
                    )
                return PreRouteOutcome(
                    kind="pin",
                    pinned_host_id=pin.host_id,
                    pinned_host=pin,
                    target_reason=TargetReason(
                        reason_code=TargetReasonCode.PINNED_HOST,
                        summary=f"Focused host {pin.alias.value}",
                    ),
                )

        return PreRouteOutcome(kind="pass")


def format_host_roster(
    *,
    hosts: Sequence[Host],
    hub: HostSessionHub,
    owner_id: uuid.UUID,
    pinned_host_id: HostId | None = None,
    self_host_id: str | None = None,
) -> str:
    """Builds the dynamic content-free `[HOST_ROSTER]` block from live HostTwin state."""
    active_hosts = sorted(
        (h for h in hosts if h.status == HostStatus.ACTIVE),
        key=lambda h: (h.alias.value.lower(), str(h.host_id)),
    )
    if not active_hosts:
        return ""
    focus = _hub_get_focus_state(hub, owner_id)
    effective_pin_id = pinned_host_id or (focus.focused_host_id if focus is not None else None)
    pin_host = next((h for h in active_hosts if h.host_id == effective_pin_id), None)
    if pin_host is not None:
        focus_mode = f"pinned({pin_host.alias.value})"
    elif focus is not None and focus.follow_me:
        focus_mode = "follow_me"
    else:
        focus_mode = "auto"

    lines: list[str] = ["[HOST_ROSTER]", f"focusMode: {focus_mode}"]
    if pinned_host_id is not None and pin_host is not None:
        lines.append(f"pinnedHost: {pin_host.alias.value} (tools scoped strictly to this host)")
    for h in active_hosts:
        online = _is_host_online(hub, h, self_host_id)
        twin = _hub_get_twin(hub, str(h.host_id)) if online else None
        p = twin.presence if twin is not None else None
        r = twin.resources if twin is not None else None
        locked = str(p.screen_locked).lower() if p is not None else "unknown"
        media = str(p.media_playing).lower() if p is not None else "unknown"
        idle_s = str(p.idle_seconds) if p is not None else "unknown"
        battery = (
            f"{p.battery_percent}%"
            if (p is not None and p.battery_percent is not None)
            else "unknown"
        )
        apps_cnt = len(r.installed_apps) if r is not None else 0
        ws_cnt = len(r.workspaces) if r is not None else 0
        place_s = h.place if h.place is not None else "none"
        nicks_s = "[" + ", ".join(h.nicknames) + "]"
        state_s = "online" if online else "offline"
        lines.append(
            f"- {h.alias.value} (platform={h.platform.value}, kind={h.host_kind.value}, "
            f"place={place_s}, nicknames={nicks_s}, state={state_s}, locked={locked}, "
            f"mediaPlaying={media}, idleS={idle_s}, battery={battery}, "
            f"appsCount={apps_cnt}, workspacesCount={ws_cnt})"
        )
    lines.append("[/HOST_ROSTER]")
    return "\n".join(lines)


def wire_candidate_dict(cand: ScoredHostCandidate) -> dict[str, object]:
    return {
        "hostId": str(cand.host.host_id),
        "alias": cand.host.alias.value,
        "platform": cand.host.platform.value,
        "hostKind": cand.host.host_kind.value,
        "online": cand.online,
        "score": round(cand.score, 4),
        "eligible": cand.eligible,
        "unavailableReason": (
            cand.unavailable_reason.value if cand.unavailable_reason is not None else None
        ),
        "wolMac": cand.wol_mac,
        "wolAvailable": bool(cand.wol_mac),
        "reason": cand.reason,
    }


_wire_candidate_dict = wire_candidate_dict


def build_action_target_required_event(
    invocation_id: str,
    tool_name: str,
    routing: HostCallRoutingResult,
) -> ActionTargetRequiredEvent:
    """Builds `ActionTargetRequiredEvent` from a failed/ambiguous `HostCallRoutingResult`."""
    extra = routing.error_extra or {}
    raw_reason = (
        extra.get("reasonCode")
        or routing.error_status
        or ActionTargetRequiredReason.ASK_TARGET.value
    )
    reason_code = str(raw_reason)
    if reason_code == TargetReasonCode.AMBIGUOUS_CANDIDATES.value:
        reason_code = ActionTargetRequiredReason.ASK_TARGET.value
    raw_summary = extra.get("targetReason")
    if isinstance(raw_summary, str) and raw_summary:
        target_reason = raw_summary
    elif routing.target_reason is not None:
        target_reason = routing.target_reason.summary
    else:
        target_reason = "Please specify which host to target"
    raw_cands = extra.get("candidates")
    if isinstance(raw_cands, list | tuple):
        candidates = tuple(dict(c) for c in raw_cands if isinstance(c, Mapping))
    elif routing.scoring_decision is not None:
        candidates = tuple(wire_candidate_dict(c) for c in routing.scoring_decision.candidates)
    else:
        candidates = ()
    return ActionTargetRequiredEvent(
        invocation_id=invocation_id,
        tool=tool_name,
        reason_code=reason_code,
        target_reason=target_reason,
        candidates=candidates,
    )


async def resolve_host_target_for_call(
    *,
    tool_name: str,
    clean_args: Mapping[str, object],
    requested_target: str | None,
    ctx: "ActionContext",
    active_hosts: Sequence[Host],
    manifests_by_host: Mapping[HostId, Manifest],
    grants_by_host: Mapping[HostId, Sequence[Grant]],
    online_hosts_with_tool: Sequence[Host],
    reference_descriptor: ToolDescriptor,
    hub: HostSessionHub,
    last_target_store: LastTargetStore,
    host_default_store: InMemoryHostDefaultStore,
    habit_repo: HabitRepository | None,
    pending_store: PendingIntentStore | None,
    cu_session_store: ComputerUseSessionStore | None,
    now: datetime,
) -> HostCallRoutingResult:
    """Bridges `Orchestrator._handle_host_call` to `MeshScorer`, habits & `PendingIntentStore`."""
    last_target = last_target_store.get_last_target(ctx.owner_id, ctx.pet_id, now)
    last_alias = last_target[0] if last_target is not None else None
    last_at = last_target[1] if last_target is not None else None
    family_key = reference_descriptor.family.value
    default_alias = (
        ctx.host_defaults.get(family_key)
        if ctx.host_defaults is not None and family_key in ctx.host_defaults
        else host_default_store.get_default_alias(
            owner_id=ctx.owner_id,
            pet_id=ctx.pet_id,
            family=reference_descriptor.family,
        )
    )

    explicit_host_id: HostId | None = (
        ctx.pinned_host_id if isinstance(ctx.pinned_host_id, HostId) else None
    )
    explicit_reason: TargetReasonCode | None = (
        TargetReasonCode.PINNED_HOST if explicit_host_id is not None else None
    )

    if explicit_host_id is None and requested_target is not None and requested_target.strip():
        q_match = (
            HostQueryMatch(status="no_match", candidates=())
            if is_target_pronoun(requested_target)
            else match_hosts_by_query(requested_target, active_hosts)
        )
        if q_match.status == "unique" and q_match.host is not None:
            chosen = q_match.host
            explicit_host_id = chosen.host_id
            explicit_reason = (
                TargetReasonCode.PLACE_TAG
                if q_match.match_source == "place"
                else TargetReasonCode.EXPLICIT_ALIAS
            )
            # Preserve existing per-host grant-denial audit & ActionResultEvent when target named.
            if chosen in online_hosts_with_tool:
                return HostCallRoutingResult(
                    resolved=True,
                    host=chosen,
                    target_reason=TargetReason(
                        reason_code=explicit_reason,
                        summary=f"Explicit target {chosen.alias.value}",
                    ),
                )
        elif q_match.status == "ambiguous":
            cands = _candidates_for_hosts(q_match.candidates, hub, ctx.self_host_id)
            choices = tuple(c.host.alias.value for c in cands)
            amb_summary = (
                "Multiple hosts match the requested target"
                if len(cands) > 1
                else "Requested target did not clearly match a host"
            )
            if pending_store is not None and cands:
                pending_store.save(
                    PendingIntent(
                        owner_id=ctx.owner_id,
                        pet_id=ctx.pet_id,
                        original_tool=tool_name,
                        original_arguments=dict(clean_args),
                        original_text="",
                        candidate_host_ids=tuple(c.host.host_id for c in cands),
                        created_at=now,
                        suggested_host_id=None,
                        resource_id=extract_resource_query(tool_name, clean_args),
                        family=reference_descriptor.family.value,
                        client_caps=tuple(ctx.client_caps),
                    )
                )
            return HostCallRoutingResult(
                resolved=False,
                error_status="ask_target",
                choices=choices,
                target_reason=TargetReason(
                    reason_code=TargetReasonCode.AMBIGUOUS_CANDIDATES,
                    summary=amb_summary,
                ),
                error_extra={
                    "choices": list(choices),
                    "reasonCode": ActionTargetRequiredReason.ASK_TARGET.value,
                    "targetReason": amb_summary,
                    "candidates": [_wire_candidate_dict(c) for c in cands],
                },
            )
        else:
            tr_res = TargetResolver.resolve(
                requested_target=requested_target,
                eligible_hosts=online_hosts_with_tool,
                all_tool_hosts=online_hosts_with_tool,
                last_target_alias=last_alias,
                last_target_at=last_at,
                now=now,
                host_default_alias=default_alias,
            )
            if tr_res.status == TargetResolutionStatus.RESOLVED and tr_res.host is not None:
                return HostCallRoutingResult(
                    resolved=True,
                    host=tr_res.host,
                    target_reason=TargetReason(
                        reason_code=TargetReasonCode.SESSION_CONTINUITY,
                        summary=f"Resolved via pronoun to {tr_res.host.alias.value}",
                    ),
                )
            if tr_res.status == TargetResolutionStatus.NO_HOST:
                no_host_summary = "No online host supports this tool"
                return HostCallRoutingResult(
                    resolved=False,
                    error_status="no_host",
                    target_reason=TargetReason(
                        reason_code=TargetReasonCode.NO_HOST,
                        summary=no_host_summary,
                    ),
                    error_extra={
                        "choices": [],
                        "reasonCode": ActionTargetRequiredReason.NO_HOST.value,
                        "targetReason": no_host_summary,
                        "candidates": [],
                    },
                )
            tr_hosts = [h for h in online_hosts_with_tool if h.alias.value in tr_res.choices]
            tr_cands = _candidates_for_hosts(tr_hosts or active_hosts, hub, ctx.self_host_id)
            tr_summary = (
                "No host matched the requested target"
                if tr_res.reason == "explicit_alias_not_found"
                else "Please specify which host to target"
            )
            if pending_store is not None and tr_cands:
                pending_store.save(
                    PendingIntent(
                        owner_id=ctx.owner_id,
                        pet_id=ctx.pet_id,
                        original_tool=tool_name,
                        original_arguments=dict(clean_args),
                        original_text="",
                        candidate_host_ids=tuple(c.host.host_id for c in tr_cands),
                        created_at=now,
                        suggested_host_id=None,
                        resource_id=extract_resource_query(tool_name, clean_args),
                        family=reference_descriptor.family.value,
                        client_caps=tuple(ctx.client_caps),
                    )
                )
            return HostCallRoutingResult(
                resolved=False,
                error_status="ask_target",
                choices=tr_res.choices,
                target_reason=TargetReason(
                    reason_code=TargetReasonCode.AMBIGUOUS_CANDIDATES,
                    summary=tr_summary,
                ),
                error_extra={
                    "choices": list(tr_res.choices),
                    "reasonCode": ActionTargetRequiredReason.ASK_TARGET.value,
                    "targetReason": tr_summary,
                    "candidates": [_wire_candidate_dict(c) for c in tr_cands],
                },
            )

    resource_query = extract_resource_query(tool_name, clean_args)
    tz_offset_minutes = getattr(ctx, "client_tz_offset_minutes", None)
    day_part = day_part_for_dt(now, tz_offset_minutes)
    habits_map: dict[tuple[str, HostId], HostRoutingHabit] = {}
    default_host_id: HostId | None = None

    if habit_repo is not None:
        update_keys = context_keys_for_update(reference_descriptor.family, resource_query, day_part)
        for ctx_key in update_keys:
            rows = await habit_repo.list_habits(
                ctx.owner_id,
                pet_id=ctx.pet_id,
                context_key=ctx_key,
            )
            for row in rows:
                habits_map[(row.context_key, row.host_id)] = row
        default_rec = await habit_repo.get_default(
            ctx.owner_id, ctx.pet_id, reference_descriptor.family.value
        ) or await habit_repo.get_default(ctx.owner_id, ctx.pet_id, "*")
        if default_rec is not None:
            default_host_id = default_rec.host_id

    if default_host_id is None and default_alias is not None and default_alias.strip():
        def_folded = fold_text(default_alias)
        for h in active_hosts:
            if (h.alias_folded or fold_text(h.alias.value)) == def_folded:
                default_host_id = h.host_id
                break

    active_cu_hosts: set[HostId] = set()
    if cu_session_store is not None and hasattr(cu_session_store, "list_active_for_owner"):
        for lane in await cu_session_store.list_active_for_owner(ctx.owner_id):
            if lane.host_id is not None:
                active_cu_hosts.add(lane.host_id)

    candidate_inputs: list[HostScoringCandidateInput] = []
    for h in active_hosts:
        online = _is_host_online(hub, h, ctx.self_host_id)
        candidate_inputs.append(
            HostScoringCandidateInput(
                host=h,
                online=online,
                manifest=manifests_by_host.get(h.host_id),
                grants=grants_by_host.get(h.host_id, ()),
                twin=_hub_get_twin(hub, str(h.host_id)) if online else None,
                has_active_cu_lane=(h.host_id in active_cu_hosts),
            )
        )

    focus_state = _hub_get_focus_state(hub, ctx.owner_id)
    if ctx.follow_me and (focus_state is None or not focus_state.follow_me):
        focus_state = HostFocusState(
            owner_id=ctx.owner_id,
            focused_host_id=focus_state.focused_host_id if focus_state is not None else None,
            follow_me=True,
            updated_at=now,
            source="app",
        )

    decision = score_hosts_for_tool(
        candidate_inputs,
        MeshScoringContext(
            tool_name=tool_name,
            arguments=clean_args,
            now=now,
            host_screen_consent=ctx.host_screen_consent,
            focus_state=focus_state,
            client_subnet_hash=ctx.client_subnet_hash,
            last_target_alias=last_alias,
            last_target_at=last_at,
            habits_by_key_host=habits_map,
            default_host_id=default_host_id,
            explicit_requested_host_id=explicit_host_id,
            explicit_reason_code=explicit_reason,
            client_tz_offset_minutes=tz_offset_minutes,
        ),
    )

    if decision.resolved and decision.host is not None:
        return HostCallRoutingResult(
            resolved=True,
            host=decision.host,
            target_reason=decision.target_reason,
            scoring_decision=decision,
        )

    # Online host that explicitly marked this tool `unsupported`: let Orchestrator
    # run its `ToolPermissionState.UNSUPPORTED` audit + ActionResultEvent path.
    explicit_online_host = (
        next((h for h in online_hosts_with_tool if h.host_id == explicit_host_id), None)
        if explicit_host_id is not None
        else None
    )
    if decision.action_required_reason == ActionTargetRequiredReason.NO_HOST and (
        explicit_online_host is not None
        or (explicit_host_id is None and len(online_hosts_with_tool) == 1)
    ):
        return HostCallRoutingResult(
            resolved=True,
            host=explicit_online_host or online_hosts_with_tool[0],
            target_reason=decision.target_reason,
            scoring_decision=decision,
        )

    # Preserve UDC-10 in-memory HostDefault tie-breaker & single-host grant-denial audit
    # when Orchestrator runs without mesh habit/pending stores.
    if (
        habit_repo is None
        and pending_store is None
        and explicit_host_id is None
        and requested_target is None
    ):
        if (
            decision.action_required_reason == ActionTargetRequiredReason.GRANT_DENIED
            and len(online_hosts_with_tool) == 1
        ):
            return HostCallRoutingResult(
                resolved=True,
                host=online_hosts_with_tool[0],
                target_reason=decision.target_reason,
                scoring_decision=decision,
            )
        if (
            decision.action_required_reason == ActionTargetRequiredReason.ASK_TARGET
            and default_host_id is not None
        ):
            default_cand = next(
                (
                    c
                    for c in decision.candidates
                    if c.eligible and c.host.host_id == default_host_id
                ),
                None,
            )
            if default_cand is not None:
                return HostCallRoutingResult(
                    resolved=True,
                    host=default_cand.host,
                    target_reason=TargetReason(
                        reason_code=TargetReasonCode.SESSION_CONTINUITY,
                        summary=f"Default host {default_cand.host.alias.value}",
                    ),
                    scoring_decision=decision,
                )

    req_reason = decision.action_required_reason or ActionTargetRequiredReason.ASK_TARGET
    if req_reason == ActionTargetRequiredReason.NO_HOST:
        return HostCallRoutingResult(
            resolved=False,
            error_status="no_host",
            target_reason=decision.target_reason,
            scoring_decision=decision,
            error_extra={
                "choices": [],
                "reasonCode": ActionTargetRequiredReason.NO_HOST.value,
                "targetReason": decision.target_reason.summary,
                "candidates": [_wire_candidate_dict(c) for c in decision.candidates],
            },
        )

    choices = tuple(c.host.alias.value for c in decision.candidates)
    if (
        req_reason == ActionTargetRequiredReason.ASK_TARGET
        and pending_store is not None
        and decision.candidates
    ):
        pending_store.save(
            PendingIntent(
                owner_id=ctx.owner_id,
                pet_id=ctx.pet_id,
                original_tool=tool_name,
                original_arguments=dict(clean_args),
                original_text="",
                candidate_host_ids=tuple(c.host.host_id for c in decision.candidates),
                created_at=now,
                suggested_host_id=decision.candidates[0].host.host_id,
                resource_id=resource_query,
                family=reference_descriptor.family.value,
                client_caps=tuple(ctx.client_caps),
            )
        )

    return HostCallRoutingResult(
        resolved=False,
        error_status=req_reason.value,
        choices=choices,
        target_reason=decision.target_reason,
        scoring_decision=decision,
        error_extra={
            "choices": list(choices),
            "reasonCode": req_reason.value,
            "targetReason": decision.target_reason.summary,
            "candidates": [_wire_candidate_dict(c) for c in decision.candidates],
        },
    )


async def record_execution_habit(
    habit_repo: HabitRepository | None,
    *,
    ctx: "ActionContext",
    descriptor: ToolDescriptor,
    clean_args: Mapping[str, object],
    host_id: HostId,
    now: datetime,
) -> None:
    """Records a positive dual-bucket habit observation after successful tool execution."""
    if habit_repo is None or getattr(ctx, "habit_recorded_in_preroute", False):
        return
    family = descriptor.family.value
    resource_id = extract_resource_query(descriptor.name, clean_args)
    reason = (
        HabitReasonCode.CONFIRMED_PICK
        if ctx.pinned_host_id is not None
        else HabitReasonCode.EXECUTION_OK
    )
    await _record_habit_updates(
        habit_repo,
        owner_id=ctx.owner_id,
        pet_id=ctx.pet_id,
        family=family,
        resource_id=resource_id,
        host_id=host_id,
        positive=True,
        now=now,
        reason_code=reason,
        tz_offset_minutes=getattr(ctx, "client_tz_offset_minutes", None),
    )
