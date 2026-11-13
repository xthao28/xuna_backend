"""UDC-17 S17.5: the approval card suggests "always allow" for a call the owner keeps approving.

The suggestion is only a count on the existing standing offer: it needs the same host, tool and
arguments approved at least `STANDING_SUGGEST_MIN_APPROVALS` times within the window, never
appears when the call cannot be pre-approved, and switches nothing on.
"""

import asyncio
import uuid
from datetime import timedelta

from tests.modules.actions.test_standing_approvals import (
    LAUNCH_ARGS,
    NOW,
    _call,
    _descriptor,
    _Hub,
    _World,
)

from xuna_backend.modules.actions.application.orchestrator import Orchestrator
from xuna_backend.modules.actions.domain.entities import (
    ActionPendingEvent,
    ApprovalRequest,
    ApprovalState,
)
from xuna_backend.modules.actions.domain.standing_approval import (
    STANDING_SUGGEST_MIN_APPROVALS,
    STANDING_SUGGEST_WINDOW,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.repository import InMemoryApprovalRepository
from xuna_backend.modules.hosts.domain.crypto import args_sha256
from xuna_backend.modules.hosts.domain.entities import HostId, RiskTier, ToolFamily

FIREFOX: dict[str, object] = {"app_id": "org.mozilla.firefox"}


def _approved(
    world: _World,
    args: dict[str, object],
    *,
    decided_ago: timedelta = timedelta(days=1),
    state: ApprovalState = ApprovalState.APPROVED,
    host_id: HostId | None = None,
    tool: str = "app.launch",
) -> ApprovalRequest:
    decided = NOW - decided_ago
    return ApprovalRequest(
        approval_id=uuid.uuid4(),
        owner_id=world.owner,
        host_id=host_id or world.pc.host_id,
        target_alias="PC",
        tool=tool,
        args_sha256=args_sha256(args),
        risk=RiskTier.R2,
        summary="Mở ứng dụng trên PC",
        created_at=decided,
        expires_at=decided + timedelta(minutes=2),
        state=state,
        decided_at=decided,
    )


async def _pending_for(
    world: _World, rows: list[ApprovalRequest], args: dict[str, object]
) -> ActionPendingEvent:
    repo = InMemoryApprovalRepository()
    for row in rows:
        await repo.save(row)
    hub = _Hub()
    hub.online.add(str(world.pc.host_id))
    orch = Orchestrator(
        host_repo=world.repo,
        hub=hub,
        approval_repo=repo,
        approval_broker=InMemoryApprovalBroker(),
        local_result_broker=InMemoryLocalResultBroker(),
        last_target_store=InMemoryLastTargetStore(),
        audit=world.audit,
        clock=lambda: world.now,
        standing_repo=world.standing,
    )
    status, events = await _call(world, orch, dict(args))
    assert status == "pending"
    return next(e for e in events if isinstance(e, ActionPendingEvent))


def test_the_card_carries_the_count_once_the_threshold_is_reached() -> None:
    async def run() -> None:
        world = _World()
        await world.setup()
        rows = [_approved(world, FIREFOX) for _ in range(STANDING_SUGGEST_MIN_APPROVALS)]
        pending = await _pending_for(world, rows, FIREFOX)
        assert pending.standing_tool == "app.launch"
        assert pending.standing_approved_count == STANDING_SUGGEST_MIN_APPROVALS

    asyncio.run(run())


def test_below_the_threshold_there_is_an_offer_but_no_suggestion() -> None:
    async def run() -> None:
        world = _World()
        await world.setup()
        rows = [_approved(world, FIREFOX) for _ in range(STANDING_SUGGEST_MIN_APPROVALS - 1)]
        pending = await _pending_for(world, rows, FIREFOX)
        assert pending.standing_tool == "app.launch"
        assert pending.standing_approved_count is None

    asyncio.run(run())


def test_only_approvals_of_exactly_this_call_inside_the_window_count() -> None:
    async def run() -> None:
        world = _World()
        await world.setup()
        other_host = HostId.generate()
        rows = [
            _approved(world, FIREFOX),
            _approved(world, FIREFOX),
            # None of these count:
            _approved(world, LAUNCH_ARGS),  # other arguments
            _approved(world, FIREFOX, host_id=other_host),  # other host
            _approved(world, FIREFOX, tool="app.close"),  # other tool
            _approved(world, FIREFOX, state=ApprovalState.REJECTED),
            _approved(world, FIREFOX, state=ApprovalState.EXPIRED),
            _approved(world, FIREFOX, decided_ago=STANDING_SUGGEST_WINDOW + timedelta(hours=1)),
        ]
        pending = await _pending_for(world, rows, FIREFOX)
        assert pending.standing_approved_count is None

        repo = InMemoryApprovalRepository()
        for row in rows:
            await repo.save(row)
        assert (
            await repo.count_approved(
                world.owner,
                world.pc.host_id,
                "app.launch",
                args_sha256(FIREFOX),
                NOW - STANDING_SUGGEST_WINDOW,
            )
            == 2
        )
        assert (
            await repo.count_approved(
                uuid.uuid4(),
                world.pc.host_id,
                "app.launch",
                args_sha256(FIREFOX),
                NOW - STANDING_SUGGEST_WINDOW,
            )
            == 0
        )

    asyncio.run(run())


def test_no_suggestion_when_the_call_cannot_be_pre_approved() -> None:
    async def run() -> None:
        # A free-text argument is never eligible, however often it was approved.
        world = _World(
            descriptor=_descriptor(
                "url.open",
                family=ToolFamily.URL,
                schema={
                    "type": "object",
                    "properties": {"url": {"type": "string", "maxLength": 256}},
                    "required": ["url"],
                    "additionalProperties": False,
                },
            )
        )
        await world.setup()
        args: dict[str, object] = {"url": "https://example.com"}
        rows = [_approved(world, args, tool="url.open") for _ in range(10)]
        repo = InMemoryApprovalRepository()
        for row in rows:
            await repo.save(row)
        hub = _Hub()
        hub.online.add(str(world.pc.host_id))
        orch = Orchestrator(
            host_repo=world.repo,
            hub=hub,
            approval_repo=repo,
            approval_broker=InMemoryApprovalBroker(),
            local_result_broker=InMemoryLocalResultBroker(),
            last_target_store=InMemoryLastTargetStore(),
            audit=world.audit,
            clock=lambda: world.now,
            standing_repo=world.standing,
        )
        status, events = await _call(world, orch, args, name="url__open")
        assert status == "pending"
        pending = next(e for e in events if isinstance(e, ActionPendingEvent))
        assert pending.standing_tool is None
        assert pending.standing_approved_count is None

    asyncio.run(run())


def test_the_wire_payload_carries_the_count_only_with_the_offer() -> None:
    base = {
        "approval_id": "a1",
        "target": "PC",
        "summary": "Mở ứng dụng",
        "risk": "R2",
        "expires_in": 120,
    }
    assert ActionPendingEvent(**base).standing_payload() is None
    offered = ActionPendingEvent(**base, standing_tool="app.launch", standing_arguments=FIREFOX)
    assert offered.standing_payload() == {"tool": "app.launch", "arguments": FIREFOX}
    suggested = ActionPendingEvent(
        **base,
        standing_tool="app.launch",
        standing_arguments=FIREFOX,
        standing_approved_count=4,
    )
    assert suggested.standing_payload() == {
        "tool": "app.launch",
        "arguments": FIREFOX,
        "approvedCount": 4,
    }
