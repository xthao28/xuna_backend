"""P4.1 / P4.2 (ADR-0021) tests for declarative capability adapters on backend.

Verifies:
1. `exec_fs_frames.json` capability descriptors (`capability.list`, `capability.run`),
   preflight frame, and completed result frame parse and validate cleanly.
2. Backend `SchemaValidator` accepts both `capability.list` and `capability.run`
   without silently dropping either descriptor from the catalog.
3. `capability.run` requires mandatory host preflight (`preflight_required` when
   missing) and uses `ToolFamily.EXEC`; `capability.list` uses `ToolFamily.HOST`.
4. Rule H5 compliance: zero hardcoded external application or toolchain names in
   backend capability/summary/catalog/entities modules.
"""

from __future__ import annotations

import ast
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from tests.e2e.test_wave_c_e2e import (
    EventCollector,
    _FakeAuditLog,
    _FakeHostRepository,
    _FakeSessionHub,
    _make_action_context,
    _make_host,
)

from xuna_backend.modules.actions.application.catalog import collect_eligible_host_tools
from xuna_backend.modules.actions.application.exec_fs_support import (
    requires_mandatory_preflight,
)
from xuna_backend.modules.actions.application.orchestrator import Orchestrator
from xuna_backend.modules.actions.domain.exec_fs_summary import exec_fs_pending_text
from xuna_backend.modules.actions.domain.schema_validator import SchemaValidator
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryApprovalScopeStore,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.repository import InMemoryApprovalRepository
from xuna_backend.modules.cognition.domain.types import ToolCall
from xuna_backend.modules.hosts.application.xtp_frames import parse_preflight_result
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    HostPlatform,
    Manifest,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
)

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "xtp"
BACKEND_ROOT = Path(__file__).resolve().parents[3]


def _load_frames() -> dict[str, Any]:
    raw: dict[str, Any] = json.loads(
        (FIXTURES_DIR / "exec_fs_frames.json").read_text(encoding="utf-8")
    )
    frames: dict[str, Any] = raw["frames"]
    return frames


def test_capability_fixture_descriptors_and_schema_validator_accept_both_slots() -> None:
    frames = _load_frames()
    wire_tools = frames["toolsListCapabilityResponse"]["result"]["tools"]
    descriptors = [ToolDescriptor.from_wire(t) for t in wire_tools]
    by_name = {d.name: d for d in descriptors}

    assert set(by_name.keys()) == {"capability.list", "capability.run"}
    assert by_name["capability.list"].family == ToolFamily.HOST
    assert by_name["capability.list"].risk == RiskTier.R1
    assert by_name["capability.run"].family == ToolFamily.EXEC
    assert by_name["capability.run"].risk == RiskTier.R2
    assert requires_mandatory_preflight("capability.run") is True
    assert requires_mandatory_preflight("capability.list") is False

    # Neither tool may be silently dropped by SchemaValidator!
    for desc in descriptors:
        SchemaValidator.validate_schema_definition(desc.input_schema)
    SchemaValidator.validate_arguments(
        by_name["capability.list"].input_schema,
        {"capability": "document.convert", "format": "pdf"},
    )
    SchemaValidator.validate_arguments(
        by_name["capability.run"].input_schema,
        {"adapterId": "seed_document_convert", "inputsJson": "{}"},
    )

    pf = parse_preflight_result(frames["preflightCapabilityRunResponse"]["result"])
    assert pf.status == "ok"
    assert pf.effective_risk == RiskTier.R2
    assert pf.reason_code == "capability_manifest"


def test_capability_tools_eligible_and_generic_summaries_have_no_app_names() -> None:
    now = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    owner_id = uuid.uuid4()
    host = _make_host(owner_id, alias="DevPC", platform=HostPlatform.WINDOWS)
    frames = _load_frames()
    descriptors = tuple(
        ToolDescriptor.from_wire(t)
        for t in frames["toolsListCapabilityResponse"]["result"]["tools"]
    )
    manifest = Manifest(host_id=host.host_id, tools=descriptors, updated_at=now, schema_version=1)
    grants = [
        Grant(
            host_id=host.host_id,
            family=ToolFamily.HOST,
            max_risk=RiskTier.R1,
            enabled=True,
            updated_at=now,
        ),
        Grant(
            host_id=host.host_id,
            family=ToolFamily.EXEC,
            max_risk=RiskTier.R3,
            enabled=True,
            updated_at=now,
        ),
    ]
    eligible = collect_eligible_host_tools(
        hosts=[host],
        manifests_by_host={host.host_id: manifest},
        grants_by_host={host.host_id: grants},
        online_host_ids={str(host.host_id)},
        host_screen_consent=True,
        now=now,
    )
    assert set(eligible.keys()) == {"capability.list", "capability.run"}

    vi = exec_fs_pending_text(
        "capability.run", {"adapterId": "custom_adapter"}, "DevPC", english=False
    )
    en = exec_fs_pending_text(
        "capability.run", {"adapterId": "custom_adapter"}, "DevPC", english=True
    )
    assert vi is not None and "custom_adapter" in vi
    assert en is not None and "custom_adapter" in en


@pytest.mark.anyio
async def test_capability_run_refuses_fail_closed_when_preflight_missing() -> None:
    now = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    owner_id = uuid.uuid4()
    host = _make_host(owner_id, alias="DevPC", platform=HostPlatform.WINDOWS)
    frames = _load_frames()
    descriptors = tuple(
        ToolDescriptor.from_wire(t)
        for t in frames["toolsListCapabilityResponse"]["result"]["tools"]
    )
    host_repo = _FakeHostRepository()
    host_repo.hosts[host.host_id] = host
    host_repo.manifests[host.host_id] = Manifest(
        host_id=host.host_id, tools=descriptors, updated_at=now, schema_version=1
    )
    host_repo.grants[host.host_id] = [
        Grant(
            host_id=host.host_id,
            family=ToolFamily.EXEC,
            max_risk=RiskTier.R3,
            enabled=True,
            updated_at=now,
        )
    ]

    hub = _FakeSessionHub()

    async def _no_preflight(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("rpc_error:-32601: Method not found")

    hub.send_request = _no_preflight  # type: ignore[method-assign]
    orch = Orchestrator(
        host_repo=host_repo,
        hub=hub,
        approval_repo=InMemoryApprovalRepository(),
        approval_broker=InMemoryApprovalBroker(),
        local_result_broker=InMemoryLocalResultBroker(),
        last_target_store=InMemoryLastTargetStore(),
        audit=_FakeAuditLog(),
        scope_store=InMemoryApprovalScopeStore(),
        clock=lambda: now,
    )
    collector = EventCollector(orch, owner_id)
    msg = await orch.handle_call(
        ToolCall(
            id="cap-1",
            name="capability__run",
            arguments={"adapterId": "seed_document_convert", "inputsJson": "{}"},
        ),
        _make_action_context(owner_id, collector),
    )
    payload = json.loads(msg.content)
    assert payload["status"] == "denied"
    assert payload["output"]["errorCode"] == "preflight_required"
    assert not collector.pending_events


def test_rule_h5_zero_app_names_in_backend_capability_layers() -> None:
    forbidden = re.compile(
        r"\b(libreoffice|soffice|pandoc|imagemagick|inkscape|blender|photoshop|gimp|ffmpeg)\b",
        re.IGNORECASE,
    )
    targets = [
        BACKEND_ROOT / "src/xuna_backend/modules/hosts/domain/entities.py",
        BACKEND_ROOT / "src/xuna_backend/modules/actions/application/exec_fs_support.py",
        BACKEND_ROOT / "src/xuna_backend/modules/actions/application/catalog.py",
        BACKEND_ROOT / "src/xuna_backend/modules/actions/domain/exec_fs_summary.py",
    ]
    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert not forbidden.search(text), f"Rule H5 violation in {path}"
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not forbidden.search(node.value), f"Rule H5 literal in {path}"
