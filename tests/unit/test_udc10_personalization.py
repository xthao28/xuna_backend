"""Unit & acceptance tests for UDC-10 Learning and Personalization (S10.1-S10.3)."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from tests.helpers import MemoryAuditLog

from xuna_backend.modules.actions.application.catalog import CAP_HOST_ACTIONS_V1
from xuna_backend.modules.actions.application.orchestrator import (
    ActionContext,
    Orchestrator,
)
from xuna_backend.modules.actions.domain.entities import (
    ActionPendingEvent,
    ActionStreamEvent,
    ApprovalDecision,
)
from xuna_backend.modules.actions.domain.personalization import (
    OFFER_LABEL_HABIT_VI,
    OFFER_LABEL_RULE_VI,
    BondLevel,
    HostPickTracker,
    InMemoryHostDefaultStore,
    RoutineHistoryBucket,
    RoutineOfferEngine,
    evaluate_policy_at_bond_level,
    extract_active_host_defaults,
    hour_of_week,
    max_offers_per_day,
)
from xuna_backend.modules.actions.domain.target_resolver import (
    TargetResolutionStatus,
    TargetResolver,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.repository import (
    InMemoryApprovalRepository,
)
from xuna_backend.modules.cognition.domain.types import Locale, ToolCall
from xuna_backend.modules.conversation.domain.candidate_extractor import (
    extract_convention_candidate,
)
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    Host,
    HostAlias,
    HostId,
    HostPlatform,
    HostStatus,
    Manifest,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
)
from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.memory.domain.entities import (
    CandidateStatus,
    CandidateType,
    MemoryCandidate,
)


class _FakeHostRepo:
    def __init__(self) -> None:
        self.hosts: dict[HostId, Host] = {}
        self.manifests: dict[HostId, Manifest] = {}
        self.grants: dict[HostId, list[Grant]] = {}

    async def get_host(self, host_id: HostId) -> Host | None:
        return self.hosts.get(host_id)

    async def save_host(self, host: Host) -> None:
        self.hosts[host.host_id] = host

    async def list_hosts(self, owner_id: uuid.UUID) -> list[Host]:
        return [h for h in self.hosts.values() if h.owner_id == owner_id]

    async def get_manifest(self, host_id: HostId) -> Manifest | None:
        return self.manifests.get(host_id)

    async def save_manifest(self, manifest: Manifest) -> None:
        self.manifests[manifest.host_id] = manifest

    async def get_grants(self, host_id: HostId) -> list[Grant]:
        return list(self.grants.get(host_id, []))

    async def set_grants(self, host_id: HostId, grants: list[Grant]) -> None:
        self.grants[host_id] = list(grants)

    async def suspend_r2_plus_grants_for_owner(
        self, owner_id: uuid.UUID, now: datetime
    ) -> int:
        return 0


class _FakeHub:
    def __init__(self) -> None:
        self.online_ids: set[str] = set()
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def is_online(self, host_id: str) -> bool:
        return host_id in self.online_ids

    async def send_request(
        self,
        host_id: str,
        method: str,
        params: dict[str, Any],
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        self.calls.append((host_id, method, params))
        return {
            "_meta": {"xuna/status": "ok", "xuna/durationMs": 12},
            "structuredContent": {"status": "ok", "hostId": host_id, "method": method},
        }

    async def send_cancel_all_for_owner(self, host_ids: list[str], reason: str) -> int:
        return len(host_ids)


def _make_host(owner_id: uuid.UUID, alias: str, platform: HostPlatform) -> Host:
    now = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)
    hid = HostId(uuid.uuid4())
    return Host(
        host_id=hid,
        owner_id=owner_id,
        alias=HostAlias(alias),
        platform=platform,
        hostname="test-host",
        os_version="14.0",
        public_key=b"\x04" + b"\x11" * 64,
        agent_version="0.1.0",
        status=HostStatus.ACTIVE,
        created_at=now,
        last_seen_at=now,
    )


def _make_manifest(host_id: HostId) -> Manifest:
    now = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)
    return Manifest(
        host_id=host_id,
        updated_at=now,
        tools=(
            ToolDescriptor(
                name="media.play_pause",
                family=ToolFamily.MEDIA,
                risk=RiskTier.R1,
                observes=False,
                reversible=True,
                description="Play or pause media",
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
                output_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ),
            ToolDescriptor(
                name="url.open",
                family=ToolFamily.URL,
                risk=RiskTier.R2,
                observes=False,
                reversible=False,
                description="Open an HTTPS URL",
                input_schema={
                    "type": "object",
                    "properties": {"url": {"type": "string", "maxLength": 256}},
                    "required": ["url"],
                    "additionalProperties": False,
                },
                output_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ),
        ),
    )


def test_s10_1_host_default_extraction_and_3_pick_rule() -> None:
    """S10.1: `host_default` extracted from chat only as unconfirmed candidate,
    and 3 consecutive picks produce a rule-labelled offer."""
    extracted = extract_convention_candidate("nhạc thì mở trên Mac luôn nhé")
    assert extracted is not None
    assert extracted.candidate_type == CandidateType.KNOWLEDGE
    assert extracted.content["kind"] == "host_default"
    assert extracted.content["family"] == "media"
    assert extracted.content["hostAlias"] == "Mac"

    now = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)
    owner_id = uuid.uuid4()
    pet_id = PetId(uuid.uuid4().hex)

    unconfirmed = MemoryCandidate(
        candidate_id=uuid.uuid4(),
        owner_id=owner_id,
        pet_id=pet_id,
        source="conversation",
        candidate_type=CandidateType.KNOWLEDGE,
        content=dict(extracted.content),
        status=CandidateStatus.AWAITING_OWNER_CONFIRMATION,
        created_at=now,
        updated_at=now,
    )
    # Unconfirmed candidate never takes effect
    assert extract_active_host_defaults([unconfirmed]) == {}

    confirmed = MemoryCandidate(
        candidate_id=unconfirmed.candidate_id,
        owner_id=owner_id,
        pet_id=pet_id,
        source="conversation",
        candidate_type=CandidateType.KNOWLEDGE,
        content=dict(extracted.content),
        status=CandidateStatus.CONFIRMED,
        created_at=now,
        updated_at=now,
        confirmed_at=now,
    )
    assert extract_active_host_defaults([confirmed]) == {"media": "Mac"}

    deleted = MemoryCandidate(
        candidate_id=unconfirmed.candidate_id,
        owner_id=owner_id,
        pet_id=pet_id,
        source="conversation",
        candidate_type=CandidateType.KNOWLEDGE,
        content=dict(extracted.content),
        status=CandidateStatus.DELETED,
        created_at=now,
        updated_at=now,
    )
    assert extract_active_host_defaults([deleted]) == {}

    # 3-pick rule offer
    tracker = HostPickTracker()
    assert (
        tracker.record_pick(owner_id=owner_id, pet_id=str(pet_id), family="media", host_alias="Mac")
        is None
    )
    assert (
        tracker.record_pick(owner_id=owner_id, pet_id=str(pet_id), family="media", host_alias="Mac")
        is None
    )
    offer = tracker.record_pick(
        owner_id=owner_id, pet_id=str(pet_id), family="media", host_alias="Mac"
    )
    assert offer is not None
    assert offer.label == OFFER_LABEL_RULE_VI
    assert offer.prompt_vi == "Lần sau mở nhạc trên Mac luôn nhé?"
    assert "3 lần liên tiếp" in offer.why_vi


def test_s10_1_target_resolver_tie_breaker_and_immediate_deletion_without_skipping_approval() -> (
    None
):
    """S10.1: `host_default` acts only as tie-breaker, deleting it changes the next
    targeting decision immediately, and R2+ approval is never skipped."""

    async def _run() -> None:
        now = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)
        owner_id = uuid.uuid4()
        pet_id = "0123456789abcdef0123456789abcdef"
        repo = _FakeHostRepo()
        hub = _FakeHub()
        mac = _make_host(owner_id, "Mac", HostPlatform.MACOS)
        pc = _make_host(owner_id, "PC nhà", HostPlatform.WINDOWS)

        for h in (mac, pc):
            repo.hosts[h.host_id] = h
            repo.manifests[h.host_id] = _make_manifest(h.host_id)
            repo.grants[h.host_id] = [
                Grant(
                    host_id=h.host_id,
                    family=ToolFamily.MEDIA,
                    max_risk=RiskTier.R1,
                    enabled=True,
                    updated_at=now,
                ),
                Grant(
                    host_id=h.host_id,
                    family=ToolFamily.URL,
                    max_risk=RiskTier.R2,
                    enabled=True,
                    updated_at=now,
                ),
            ]
            hub.online_ids.add(str(h.host_id))

        # Direct TargetResolver check
        res_without_default = TargetResolver.resolve(
            requested_target=None,
            eligible_hosts=[mac, pc],
            now=now,
        )
        assert res_without_default.status == TargetResolutionStatus.ASK_TARGET

        res_with_default = TargetResolver.resolve(
            requested_target=None,
            eligible_hosts=[mac, pc],
            now=now,
            host_default_alias="Mac",
        )
        assert res_with_default.status == TargetResolutionStatus.RESOLVED
        assert res_with_default.host == mac
        assert res_with_default.reason == "host_default"

        # Explicit alias always overrides host_default
        res_explicit = TargetResolver.resolve(
            requested_target="PC nhà",
            eligible_hosts=[mac, pc],
            now=now,
            host_default_alias="Mac",
        )
        assert res_explicit.status == TargetResolutionStatus.RESOLVED
        assert res_explicit.host == pc
        assert res_explicit.reason == "explicit_alias"

        # End-to-end Orchestrator check: set default -> resolves -> delete default -> asks target
        approval_broker = InMemoryApprovalBroker()
        default_store = InMemoryHostDefaultStore()
        orchestrator = Orchestrator(
            host_repo=repo,
            hub=hub,  # type: ignore[arg-type]
            approval_repo=InMemoryApprovalRepository(),
            approval_broker=approval_broker,
            local_result_broker=InMemoryLocalResultBroker(),
            last_target_store=InMemoryLastTargetStore(),
            audit=MemoryAuditLog(),
            host_default_store=default_store,
            clock=lambda: now,
        )

        conv = orchestrator.set_host_default(
            owner_id=owner_id,
            pet_id=pet_id,
            family="media",
            host_alias="Mac",
        )
        url_conv = orchestrator.set_host_default(
            owner_id=owner_id,
            pet_id=pet_id,
            family="url",
            host_alias="Mac",
        )

        events: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)
            if isinstance(ev, ActionPendingEvent):
                await orchestrator.decide_approval(
                    approval_id=uuid.UUID(ev.approval_id),
                    owner_id=owner_id,
                    decision=ApprovalDecision.APPROVE,
                )

        ctx = ActionContext(
            owner_id=owner_id,
            pet_id=pet_id,
            locale=Locale.VI,
            client_caps={CAP_HOST_ACTIONS_V1},
            board_available=False,
            emit_event=_emit,
        )

        # 1. R2 tool (`url.open`) with `host_default` resolves to "Mac" AND still asks approval
        msg_r2 = await orchestrator.handle_call(
            ToolCall(
                id="call-r2",
                name="url__open",
                arguments={"url": "https://docs.flutter.dev"},
            ),
            ctx,
        )
        payload_r2 = json.loads(msg_r2.content)
        assert payload_r2["status"] == "ok"
        pending_events = [e for e in events if isinstance(e, ActionPendingEvent)]
        assert len(pending_events) == 1
        assert pending_events[0].target == "Mac"

        # Clear last_target so next call tests `host_default` tie-breaker directly
        orchestrator._last_target_store = InMemoryLastTargetStore()

        # 2. Delete the `media` default -> next `media__play_pause` call immediately asks target!
        assert orchestrator.delete_host_default(conv.candidate_id, owner_id=owner_id) is True
        assert orchestrator.delete_host_default(url_conv.candidate_id, owner_id=owner_id) is True

        msg_after_delete = await orchestrator.handle_call(
            ToolCall(id="call-media", name="media__play_pause", arguments={}),
            ctx,
        )
        payload_after_delete = json.loads(msg_after_delete.content)
        assert payload_after_delete["status"] == "ask_target"
        assert set(payload_after_delete["choices"]) == {"Mac", "PC nhà"}

    asyncio.run(_run())


def test_s10_2_routine_offer_rule_and_two_week_pause() -> None:
    """S10.2: Simulated 4-week history produces expected offer when >= 3/4 weeks
    match and routine is around; never auto-executes; pauses for 2 weeks after
    2 consecutive declines."""
    now = datetime(2026, 10, 30, 20, 0, tzinfo=UTC)  # Friday 20:00
    how = hour_of_week(now)
    buckets = [
        RoutineHistoryBucket(
            family="media", host_alias="Mac", hour_of_week=how, week_index=0, count=2
        ),
        RoutineHistoryBucket(
            family="media", host_alias="Mac", hour_of_week=how, week_index=1, count=1
        ),
        RoutineHistoryBucket(
            family="media", host_alias="Mac", hour_of_week=how, week_index=3, count=1
        ),
        # Only 2 weeks for PC -> must not trigger
        RoutineHistoryBucket(
            family="media", host_alias="PC", hour_of_week=how, week_index=0, count=3
        ),
        RoutineHistoryBucket(
            family="media", host_alias="PC", hour_of_week=how, week_index=1, count=2
        ),
    ]

    engine = RoutineOfferEngine()
    offers = engine.evaluate(
        buckets=buckets,
        now=now,
        routine_around=True,
        bond_level=BondLevel.AN_Y,
        temperament="balanced",
    )
    assert len(offers) == 1
    offer = offers[0]
    assert offer.host_alias == "Mac"
    assert offer.family == "media"
    assert offer.weeks_matched == 3
    assert offer.total_count == 4
    assert offer.auto_executed is False
    assert offer.label == OFFER_LABEL_HABIT_VI
    assert offer.text_vi == "Tối rồi, mở nhạc trên Mac nhé?"
    assert "3/4 tuần gần đây" in offer.why_vi
    assert "tổng 4 lần" in offer.why_vi

    # Not around -> 0 offers
    assert (
        engine.evaluate(
            buckets=buckets,
            now=now,
            routine_around=False,
            bond_level=BondLevel.AN_Y,
        )
        == []
    )

    # Declined twice in a row -> paused for 14 days
    engine.record_response(family="media", host_alias="Mac", accepted=False, now=now)
    assert engine.is_paused(family="media", host_alias="Mac", now=now) is False
    engine.record_response(family="media", host_alias="Mac", accepted=False, now=now)
    assert engine.is_paused(family="media", host_alias="Mac", now=now) is True
    assert (
        engine.evaluate(
            buckets=buckets,
            now=now + timedelta(days=7),
            routine_around=True,
            bond_level=BondLevel.AN_Y,
        )
        == []
    )
    # Resumes after 14 days
    resumed = engine.evaluate(
        buckets=buckets,
        now=now + timedelta(days=14),
        routine_around=True,
        bond_level=BondLevel.AN_Y,
    )
    assert len(resumed) == 1


def test_s10_3_bond_level_cap_policy_invariance_and_two_histories_differ() -> None:
    """S10.3: Bond level caps offer frequency only, policy result is identical
    across all bond levels, personality changes wording only, and two simulated
    owners get different, explainable offers."""
    assert max_offers_per_day(BondLevel.MOI_QUEN) == 0
    assert max_offers_per_day(BondLevel.BAT_NHIP) == 1
    assert max_offers_per_day(BondLevel.AN_Y) == 2
    assert max_offers_per_day(BondLevel.DONG_DOI) == 3

    # 1. Policy invariance across all bond levels
    now = datetime(2026, 10, 31, 20, 0, tzinfo=UTC)
    hid = HostId(uuid.uuid4())
    manifest = _make_manifest(hid)
    grants = [
        Grant(
            host_id=hid,
            family=ToolFamily.MEDIA,
            max_risk=RiskTier.R1,
            enabled=True,
            updated_at=now,
        ),
        Grant(
            host_id=hid,
            family=ToolFamily.URL,
            max_risk=RiskTier.R2,
            enabled=False,
            updated_at=now,
        ),
    ]
    for tool_desc in manifest.tools:
        snapshots = {
            evaluate_policy_at_bond_level(
                descriptor=tool_desc,
                grants=grants,
                now=now,
                bond_level=lvl,
            )
            for lvl in BondLevel
        }
        assert len(snapshots) == 1

    # 2. Two simulated owners with different histories get different, explainable offers
    friday_20h = datetime(2026, 10, 30, 20, 0, tzinfo=UTC)
    saturday_09h = datetime(2026, 10, 31, 9, 0, tzinfo=UTC)

    owner_a_buckets = [
        RoutineHistoryBucket(
            family="media",
            host_alias="MacBook Pro",
            hour_of_week=hour_of_week(friday_20h),
            week_index=w,
            count=2,
        )
        for w in (0, 1, 2, 3)
    ]
    owner_b_buckets = [
        RoutineHistoryBucket(
            family="media",
            host_alias="PC Phòng Khách",
            hour_of_week=hour_of_week(saturday_09h),
            week_index=w,
            count=1,
        )
        for w in (0, 2, 3)
    ]

    engine_a = RoutineOfferEngine()
    engine_b = RoutineOfferEngine()

    # At Mới quen (0), even with 4/4 weeks match, 0 offers are shown
    assert (
        engine_a.evaluate(
            buckets=owner_a_buckets,
            now=friday_20h,
            routine_around=True,
            bond_level=BondLevel.MOI_QUEN,
            temperament="playful",
        )
        == []
    )

    offers_a = engine_a.evaluate(
        buckets=owner_a_buckets,
        now=friday_20h,
        routine_around=True,
        bond_level=BondLevel.AN_Y,
        temperament="playful",
    )
    offers_b = engine_b.evaluate(
        buckets=owner_b_buckets,
        now=saturday_09h,
        routine_around=True,
        bond_level=BondLevel.BAT_NHIP,
        temperament="quiet",
    )

    assert len(offers_a) == 1
    assert len(offers_b) == 1
    assert offers_a[0].host_alias == "MacBook Pro"
    assert offers_b[0].host_alias == "PC Phòng Khách"
    assert offers_a[0].text_vi != offers_b[0].text_vi
    assert "cùng bạn nhé?" in offers_a[0].text_vi  # playful wording
    assert "Đã đến giờ quen thuộc" in offers_b[0].text_vi  # quiet wording
    assert "4/4 tuần gần đây" in offers_a[0].why_vi
    assert "3/4 tuần gần đây" in offers_b[0].why_vi
    assert offers_a[0].auto_executed is False
    assert offers_b[0].auto_executed is False
