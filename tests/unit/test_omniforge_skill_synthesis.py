"""OMNIFORGE (`FORGE-1` -> `FORGE-4`) unit tests for Citizen-Developer Skill Synthesis:

- Validates the 5 new XTP tools (`forge.synthesize_skill`, `forge.run_skill`,
  `forge.list_and_manage`, `teach.record_session`, `forge.extract_invoices_to_sheet`)
  in `validate_tool_name`, `ActionSummary`, and `build_tool_catalog`.
- Tests `OmniForgeSynthesizerService` across categories (`web_harvest_invoices`,
  `document_to_sheet`, `smart_file_janitor`, `spreadsheet_reconciler`), `Show-Me-Once`
  demonstration steps, and hard refusal of executable extensions (`.exe`, `.bat`) or
  path-traversal sandbox folders (`../`).
- Tests `CompositeWorkflowRunner` for `forge_invoice_harvester` (`R2`, 1 approval) and
  `teach_and_compile` (`R1`).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from tests.helpers import MemoryAuditLog

from xuna_backend.modules.actions.application.catalog import (
    CAP_HOST_ACTIONS_V1,
    CAP_WORKFLOW_TEMPLATES_V1,
    build_tool_catalog,
)
from xuna_backend.modules.actions.application.workflows import CompositeWorkflowRunner
from xuna_backend.modules.actions.domain.entities import (
    ActionPendingEvent,
    ActionStreamEvent,
    ActionSummary,
    ApprovalDecision,
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
from xuna_backend.modules.symbiotic.application.services import (
    OmniForgeSynthesizerService,
)

NOW = datetime(2026, 11, 2, 10, 0, 0, tzinfo=UTC)
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

    async def suspend_r2_plus_grants_for_owner(self, owner_id: uuid.UUID, now: datetime) -> int:
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
        name = params.get("name", "")
        args = params.get("arguments", {})
        if name == "forge.synthesize_skill":
            target_dir = args.get("targetFolder", "HoaDon")
            return {
                "content": [{"type": "text", "text": "ok"}],
                "structuredContent": {
                    "skillId": "skill-web-harvest-invoices-1",
                    "titleVi": "Tải Hóa Đơn Hàng Loạt & Lập Bảng Kê",
                    "category": "web_harvest_invoices",
                    "detectedItemsCount": 48,
                    "dryRunSample": f"Mẫu thử #1: 2026-11_VNPT_HD001248.pdf -> {target_dir}/",
                },
                "isError": False,
                "_meta": {"xuna/status": "ok", "xuna/durationMs": 5},
            }
        if name == "forge.run_skill":
            return {
                "content": [{"type": "text", "text": "ok"}],
                "structuredContent": {
                    "skillId": args.get("skillId", "skill-web-harvest-invoices-1"),
                    "processedCount": 48,
                    "outputDirectory": args.get("outputFolder", "XUNA_HoaDon/Thang_11_2026"),
                    "exportedCsvFile": "XUNA_HoaDon/Thang_11_2026/BangKe_HoaDon_TongHop.csv",
                    "selfHealedSelectors": 1,
                    "undoSnapshotId": "undo-forge-1",
                },
                "isError": False,
                "_meta": {"xuna/status": "ok", "xuna/durationMs": 12},
            }
        if name == "forge.extract_invoices_to_sheet":
            src_dir = args.get("sourceFolder", "HoaDon")
            out_name = args.get("outputFileName", "BangKe.csv")
            return {
                "content": [{"type": "text", "text": "ok"}],
                "structuredContent": {
                    "invoiceCount": 3,
                    "totalAmountVnd": 7150000,
                    "duplicateCount": 0,
                    "outputCsvFile": f"{src_dir}/{out_name}",
                },
                "isError": False,
                "_meta": {"xuna/status": "ok", "xuna/durationMs": 6},
            }
        if name == "teach.record_session":
            return {
                "content": [{"type": "text", "text": "ok"}],
                "structuredContent": {
                    "recordingActive": False,
                    "capturedStepsCount": 2,
                    "compiledSkillId": "skill-web-harvest-invoices-2",
                    "generalizedLoopSummaryVi": (
                        "Đã đóng gói 2 bước thao tác mẫu thành công cụ tải hóa đơn tự động!"
                    ),
                },
                "isError": False,
                "_meta": {"xuna/status": "ok", "xuna/durationMs": 5},
            }
        return {
            "content": [{"type": "text", "text": "ok"}],
            "structuredContent": {"ok": True},
            "isError": False,
            "_meta": {"xuna/status": "ok", "xuna/durationMs": 5},
        }


def _build_omniforge_manifest(host_id: HostId) -> Manifest:
    empty_schema: dict[str, object] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    tools = (
        ToolDescriptor(
            name="forge.synthesize_skill",
            family=ToolFamily.APP,
            risk=RiskTier.R1,
            description="Synthesize sandboxed personal skill",
            input_schema=empty_schema,
            output_schema=empty_schema,
            observes=True,
            reversible=True,
        ),
        ToolDescriptor(
            name="forge.run_skill",
            family=ToolFamily.APP,
            risk=RiskTier.R2,
            description="Run forged personal skill in sandbox",
            input_schema=empty_schema,
            output_schema=empty_schema,
            observes=False,
            reversible=True,
        ),
        ToolDescriptor(
            name="forge.list_and_manage",
            family=ToolFamily.APP,
            risk=RiskTier.R1,
            description="List or manage forged skills",
            input_schema=empty_schema,
            output_schema=empty_schema,
            observes=True,
            reversible=True,
        ),
        ToolDescriptor(
            name="teach.record_session",
            family=ToolFamily.APP,
            risk=RiskTier.R1,
            description="Record Show-Me-Once demonstration",
            input_schema=empty_schema,
            output_schema=empty_schema,
            observes=True,
            reversible=True,
        ),
        ToolDescriptor(
            name="forge.extract_invoices_to_sheet",
            family=ToolFamily.APP,
            risk=RiskTier.R1,
            description="Extract electronic invoices to accounting CSV",
            input_schema=empty_schema,
            output_schema=empty_schema,
            observes=False,
            reversible=True,
        ),
    )
    return Manifest(
        host_id=host_id,
        tools=tools,
        updated_at=NOW,
    )


def test_omniforge_tool_validation_and_action_summaries() -> None:
    for tool_name in (
        "forge.synthesize_skill",
        "forge.run_skill",
        "forge.list_and_manage",
        "teach.record_session",
        "forge.extract_invoices_to_sheet",
    ):
        assert validate_tool_name(tool_name) == ToolFamily.APP

    vi = ActionSummary.format_pending(
        "forge.synthesize_skill",
        {"goal": "Tải hết hóa đơn PDF"},
        "PC Kế Toán",
        Locale.VI,
    )
    assert "Tự rèn kỹ năng mới" in vi
    assert "PC Kế Toán" in vi

    en = ActionSummary.format_pending(
        "forge.extract_invoices_to_sheet",
        {"sourceFolder": "XUNA_HoaDon"},
        "Accounting PC",
        Locale.EN,
    )
    assert "Extract invoices" in en
    assert "Accounting PC" in en


def test_omniforge_synthesizer_service_categories_and_safety_guards() -> None:
    svc = OmniForgeSynthesizerService()

    inv = svc.synthesize_skill(
        goal="Tôi muốn kéo hết hóa đơn tự động tải về máy, tải từng hóa đơn lâu quá",
        target_folder="XUNA_HoaDon/Thang_11",
        file_extensions=("pdf", "xml"),
        now=NOW,
    )
    assert inv.category == "web_harvest_invoices"
    assert inv.detected_items_count == 48
    assert "2026-11_VNPT_HD001248.pdf" in inv.dry_run_sample
    assert len(inv.parameter_fields) == 3

    reconcile = svc.synthesize_skill(
        goal="Đối soát sao kê ngân hàng với danh sách đơn hàng tháng 11",
        now=NOW,
    )
    assert reconcile.category == "spreadsheet_reconciler"

    janitor = svc.synthesize_skill(
        goal="Phân loại và dọn thư mục Downloads tự động",
        demonstrated_action="Kéo file HopDong.pdf vào thư mục TaiLieu_PDF",
        now=NOW,
    )
    assert janitor.category == "smart_file_janitor"
    assert any("Học từ thao tác mẫu" in s for s in janitor.script_steps)

    with pytest.raises(ValueError, match="forge_extension_forbidden"):
        svc.synthesize_skill(goal="Tải file", file_extensions=("exe",), now=NOW)

    with pytest.raises(ValueError, match="invalid_sandbox_folder"):
        svc.synthesize_skill(goal="Tải hóa đơn", target_folder="../System32", now=NOW)


def test_omniforge_workflows_invoice_harvester_and_teach_compile() -> None:
    async def _run() -> None:
        owner_id = uuid.uuid4()
        host_id = HostId.generate()
        repo = _FakeHostRepo()
        hub = _FakeHub()
        approval_repo = InMemoryApprovalRepository()
        approval_broker = InMemoryApprovalBroker()
        local_broker = InMemoryLocalResultBroker()
        audit = MemoryAuditLog()

        host = Host(
            host_id=host_id,
            owner_id=owner_id,
            alias=HostAlias("PC Kế Toán"),
            platform=HostPlatform.WINDOWS,
            hostname="ACCOUNTING-PC",
            os_version="Windows 11",
            agent_version="0.1.0",
            public_key=FAKE_PUBKEY,
            status=HostStatus.ACTIVE,
            created_at=NOW,
            last_seen_at=NOW,
        )
        manifest = _build_omniforge_manifest(host_id)
        grants = [
            Grant(
                host_id=host_id,
                family=ToolFamily.APP,
                max_risk=RiskTier.R2,
                enabled=True,
                updated_at=NOW,
            )
        ]
        await repo.save_host(host)
        await repo.save_manifest(manifest)
        await repo.set_grants(host_id, grants)
        hub.online_ids.add(str(host_id))

        catalog = build_tool_catalog(
            hosts=[host],
            manifests_by_host={host_id: manifest},
            grants_by_host={host_id: grants},
            online_host_ids={str(host_id)},
            client_caps=frozenset({CAP_HOST_ACTIONS_V1, CAP_WORKFLOW_TEMPLATES_V1}),
            board_available=True,
            host_screen_consent=True,
            now=NOW,
        )
        names = {t.name for t in catalog}
        assert "forge__synthesize_skill" in names
        assert "forge__run_skill" in names

        runner = CompositeWorkflowRunner(
            host_repo=repo,
            hub=hub,
            approval_repo=approval_repo,
            approval_broker=approval_broker,
            local_result_broker=local_broker,
            audit=audit,
            clock=lambda: NOW,
        )

        events: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            events.append(ev)
            if isinstance(ev, ActionPendingEvent):
                req_id = uuid.UUID(ev.approval_id)
                await approval_broker.decide(
                    req_id,
                    owner_id,
                    ApprovalDecision.APPROVE,
                    NOW,
                )

        msg = await runner.handle_template_call(
            ToolCall(
                id="call-forge-1",
                name="workflow_template",
                arguments={
                    "template": "forge_invoice_harvester",
                    "goal": "Tự động tải hết hóa đơn PDF/XML tháng 11",
                    "source_folder": "XUNA_HoaDon/Thang_11",
                    "file_extensions": "pdf,xml",
                },
            ),
            owner_id=owner_id,
            locale=Locale.VI,
            client_caps={CAP_HOST_ACTIONS_V1, CAP_WORKFLOW_TEMPLATES_V1},
            board_available=True,
            emit_event=_emit,
            request_id="req-forge-1",
        )
        payload = json.loads(msg.content)
        assert payload["status"] == "ok"
        out = payload["output"]
        assert out["template"] == "forge_invoice_harvester"
        assert out["processedCount"] == 48
        assert out["totalAmountVnd"] == 7150000
        assert out["undoSnapshotId"] == "undo-forge-1"

        teach_msg = await runner.handle_template_call(
            ToolCall(
                id="call-teach-1",
                name="workflow_template",
                arguments={
                    "template": "teach_and_compile",
                    "goal": "Tải hóa đơn MISA từ 1 lần làm mẫu",
                },
            ),
            owner_id=owner_id,
            locale=Locale.VI,
            client_caps={CAP_HOST_ACTIONS_V1, CAP_WORKFLOW_TEMPLATES_V1},
            board_available=True,
            emit_event=_emit,
            request_id="req-teach-1",
        )
        teach_payload = json.loads(teach_msg.content)
        assert teach_payload["status"] == "ok"
        assert teach_payload["output"]["compiledSkillId"] == "skill-web-harvest-invoices-2"

    asyncio.run(_run())
