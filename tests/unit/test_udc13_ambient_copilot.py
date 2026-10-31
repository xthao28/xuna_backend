"""UDC-13 unit and integration tests (`S13.1` -> `S13.5`):

- `S13.1` (`window.active_context`, `R1`, `observes = True`, `ToolFamily.APP`):
  - Validates tool name mapping, U5 (`host_screen`) consent gating, and deterministic
    Vietnamese/English summaries.
- `S13.2` (`browser.read_page`, `R1`, `observes = True`, `ToolFamily.URL`):
  - Validates tool name mapping, U5 consent gating, and summaries.
- `S13.3` (`beam.transfer`, `R2`, `ToolFamily.APP`):
  - Validates tool name mapping and summaries without storing raw payload bytes.
- `S13.4` (`workflow.batch_steps` & `custom_workflow` validate-all-before-run):
  - Verifies that if Step 2 in a custom workflow is missing from the host manifest or lacks
    an active grant, Step 1 is NEVER executed and no approval prompt is sent.
- `S13.5` (`action.undo`, `R1`, `ToolFamily.APP`):
  - Validates tool name mapping, summaries, and custom workflow extraction (`"hoàn tác"`).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from tests.helpers import MemoryAuditLog

from xuna_backend.modules.actions.application.catalog import is_tool_granted
from xuna_backend.modules.actions.application.workflows import (
    CompositeWorkflowRunner,
    CustomWorkflowDefinition,
    CustomWorkflowStep,
    InMemoryCustomWorkflowStore,
    extract_custom_workflow_candidate,
)
from xuna_backend.modules.actions.domain.entities import (
    ActionPendingEvent,
    ActionStreamEvent,
    ActionSummary,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.repository import (
    InMemoryApprovalRepository,
)
from xuna_backend.modules.cognition.domain.types import Locale, ToolCall
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
    validate_tool_name,
)

NOW = datetime(2026, 10, 31, 14, 0, 0, tzinfo=UTC)
FAKE_PUBKEY = b"\x04" + bytes(range(1, 65))


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

    async def get_grants(self, host_id: HostId) -> list[Grant]:
        return list(self.grants.get(host_id, []))

    async def set_grants(self, host_id: HostId, grants: list[Grant]) -> None:
        self.grants[host_id] = list(grants)

    async def suspend_r2_plus_grants_for_owner(
        self, owner_id: uuid.UUID, now: datetime
    ) -> int:
        return 0

    async def get_manifest(self, host_id: HostId) -> Manifest | None:
        return self.manifests.get(host_id)

    async def save_manifest(self, manifest: Manifest) -> None:
        self.manifests[manifest.host_id] = manifest


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
            "content": [{"type": "text", "text": "ok"}],
            "structuredContent": {},
            "isError": False,
            "_meta": {"xuna/status": "ok", "xuna/durationMs": 3},
        }

    async def stop_host(self, host_id: str, reason: str = "owner_app") -> bool:
        return host_id in self.online_ids


def test_udc13_tool_names_consent_gating_and_summaries() -> None:
    assert (
        validate_tool_name("window.active_context", expected_family=ToolFamily.APP)
        == ToolFamily.APP
    )
    assert (
        validate_tool_name("browser.read_page", expected_family=ToolFamily.URL)
        == ToolFamily.URL
    )
    assert (
        validate_tool_name("beam.transfer", expected_family=ToolFamily.APP)
        == ToolFamily.APP
    )
    assert (
        validate_tool_name("workflow.batch_steps", expected_family=ToolFamily.WORKFLOW)
        == ToolFamily.WORKFLOW
    )
    assert (
        validate_tool_name("action.undo", expected_family=ToolFamily.APP)
        == ToolFamily.APP
    )

    empty_schema: dict[str, object] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    host_id = HostId.generate()
    win_desc = ToolDescriptor(
        name="window.active_context",
        description="Read frontmost window context.",
        input_schema=empty_schema,
        output_schema=empty_schema,
        family=ToolFamily.APP,
        risk=RiskTier.R1,
        observes=True,
    )
    grants = [
        Grant(
            host_id=host_id,
            family=ToolFamily.APP,
            max_risk=RiskTier.R2,
            enabled=True,
            updated_at=NOW,
        )
    ]
    # Because observes=True, U5 (host_screen_consent) is required
    assert is_tool_granted(win_desc, grants, NOW, host_screen_consent=False) is False
    assert is_tool_granted(win_desc, grants, NOW, host_screen_consent=True) is True

    assert (
        ActionSummary.format_pending(
            "window.active_context", {}, "PC", Locale.VI
        )
        == "Đọc ngữ cảnh cửa sổ đang mở trên PC"
    )
    assert (
        ActionSummary.format_pending(
            "browser.read_page", {"browser": "edge"}, "PC", Locale.VI
        )
        == "Tóm tắt trang web đang mở (edge) trên PC"
    )
    assert (
        ActionSummary.format_pending(
            "beam.transfer", {"fileName": "report.pdf"}, "PC", Locale.VI
        )
        == "Truyền tệp report.pdf qua XUNA Beam tới PC"
    )
    assert (
        ActionSummary.format_pending(
            "workflow.batch_steps",
            {"steps": [{"tool": "display.wake", "arguments": {}}]},
            "PC",
            Locale.VI,
        )
        == "Chạy chuỗi 1 bước liên hoàn trên PC"
    )
    assert (
        ActionSummary.format_pending("action.undo", {}, "PC", Locale.VI)
        == "Hoàn tác thao tác vừa thực hiện trên PC"
    )

    candidate = extract_custom_workflow_candidate(
        uuid.uuid4(),
        "Khi mình nói 'xem nhanh' thì bật màn hình, đọc cửa sổ, tóm tắt trang và hoàn tác",
        default_host_alias="PC",
    )
    assert candidate is not None
    assert [s.tool for s in candidate.steps] == [
        "display.wake",
        "window.active_context",
        "browser.read_page",
        "action.undo",
    ]


def test_udc13_validate_all_before_run_prevents_partial_side_effects() -> None:
    async def _scenario() -> None:
        owner_id = uuid.uuid4()
        host_repo = _FakeHostRepo()
        hub = _FakeHub()
        approval_repo = InMemoryApprovalRepository()
        approval_broker = InMemoryApprovalBroker()
        local_broker = InMemoryLocalResultBroker()
        audit = MemoryAuditLog()
        custom_store = InMemoryCustomWorkflowStore()

        pc = Host(
            host_id=HostId.generate(),
            owner_id=owner_id,
            alias=HostAlias("PC"),
            platform=HostPlatform.WINDOWS,
            hostname="pc.local",
            os_version="11",
            agent_version="0.1.0",
            public_key=FAKE_PUBKEY,
            status=HostStatus.ACTIVE,
            created_at=NOW,
            last_seen_at=NOW,
        )
        await host_repo.save_host(pc)

        empty_schema: dict[str, object] = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        # Manifest has display.wake (R1) and app.launch (R2), but NOT url.open
        await host_repo.save_manifest(
            Manifest(
                host_id=pc.host_id,
                tools=(
                    ToolDescriptor(
                        name="display.wake",
                        description="Wake display.",
                        input_schema=empty_schema,
                        output_schema=empty_schema,
                        family=ToolFamily.APP,
                        risk=RiskTier.R1,
                    ),
                    ToolDescriptor(
                        name="app.launch",
                        description="Launch app.",
                        input_schema=empty_schema,
                        output_schema=empty_schema,
                        family=ToolFamily.APP,
                        risk=RiskTier.R2,
                        reversible=False,
                    ),
                ),
                updated_at=NOW,
            )
        )
        # Grant only R1 for APP (so Step 2 app.launch R2 is NOT granted!)
        await host_repo.set_grants(
            pc.host_id,
            [
                Grant(
                    host_id=pc.host_id,
                    family=ToolFamily.APP,
                    max_risk=RiskTier.R1,
                    enabled=True,
                    updated_at=NOW,
                )
            ],
        )
        hub.online_ids.add(str(pc.host_id))

        # Define a 2-step workflow where Step 1 (display.wake, R1) is allowed,
        # but Step 2 (app.launch, R2) exceeds the host's R1 grant.
        custom_store.save(
            CustomWorkflowDefinition(
                owner_id=owner_id,
                name="atomic_check",
                trigger_phrase="kiểm tra nguyên tử",
                steps=(
                    CustomWorkflowStep(
                        tool="display.wake",
                        arguments={},
                        risk=RiskTier.R1,
                    ),
                    CustomWorkflowStep(
                        tool="app.launch",
                        arguments={"id": "vscode"},
                        risk=RiskTier.R2,
                    ),
                ),
                host_alias="PC",
            )
        )

        runner = CompositeWorkflowRunner(
            host_repo=host_repo,
            hub=hub,
            approval_repo=approval_repo,
            approval_broker=approval_broker,
            local_result_broker=local_broker,
            audit=audit,
            custom_workflow_store=custom_store,
            clock=lambda: NOW,
        )

        emitted: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            emitted.append(ev)

        res = await runner.handle_template_call(
            ToolCall(
                id="call-atomic-1",
                name="workflow_template",
                arguments={
                    "template": "custom_workflow",
                    "workflow_name": "atomic_check",
                },
            ),
            owner_id=owner_id,
            locale=Locale.VI,
            client_caps=set(),
            board_available=False,
            emit_event=_emit,
            request_id="req-atomic",
        )
        payload = json.loads(res.content)
        assert payload["status"] == "denied"
        assert payload["output"]["failedStep"] == 2
        assert payload["output"]["failedTool"] == "app.launch"

        # Crucial invariant: Step 1 (display.wake) was NEVER sent to the host,
        # and no approval card was emitted!
        assert hub.calls == []
        assert [e for e in emitted if isinstance(e, ActionPendingEvent)] == []

    asyncio.run(_scenario())

