"""Tests for Wave 4 W4-M2 (ADR-0019 §2): bounded ephemeral `xuna/preflight` previews."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from tests.e2e.test_wave_c_e2e import (
    NOW,
    EventCollector,
    _FakeAuditLog,
    _FakeHostRepository,
    _FakeSessionHub,
    _make_action_context,
    _make_host,
)

from xuna_backend.modules.actions.application.orchestrator import Orchestrator
from xuna_backend.modules.actions.domain.preflight_preview import (
    MAX_PREVIEW_SNIPPET_BYTES,
    sanitize_preflight_preview,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryApprovalScopeStore,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.repository import (
    InMemoryApprovalRepository,
)
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

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "xtp" / "exec_fs_frames.json"


def _frames() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    frames: dict[str, Any] = data["frames"]
    return frames


def test_golden_preflight_preview_fixtures_parse() -> None:
    frames = _frames()
    diff_pf = parse_preflight_result(frames["preflightFsPatchPreviewResponse"]["result"])
    assert diff_pf.status == "ok"
    assert diff_pf.effective_risk == RiskTier.R2
    assert diff_pf.preview is not None
    assert diff_pf.preview["kind"] == "diff"
    assert diff_pf.preview["truncated"] is False
    assert "+const port = 8080;" in str(diff_pf.preview["snippet"])

    paths_pf = parse_preflight_result(frames["preflightExecDeletePreviewResponse"]["result"])
    assert paths_pf.status == "ok"
    assert paths_pf.effective_risk == RiskTier.R3
    assert paths_pf.preview is not None
    assert paths_pf.preview["kind"] == "paths"
    assert paths_pf.preview["paths"] == ["logs/a.log", "logs/b.log", "logs/c.log"]
    assert paths_pf.preview["totalCount"] == 3


@pytest.mark.parametrize(
    "bad_preflight",
    [
        # Preview forbidden on refused preflight
        {
            "status": "denied",
            "errorCode": "exec_hard_denied",
            "preview": {
                "kind": "paths",
                "summary": "1 file",
                "paths": ["a.log"],
                "totalCount": 1,
                "truncated": False,
            },
        },
        # Diff snippet exceeding 2048 UTF-8 bytes
        {
            "status": "ok",
            "effectiveRisk": "R2",
            "preview": {
                "kind": "diff",
                "summary": "too big",
                "snippet": "x" * 2049,
                "truncated": False,
            },
        },
        # Diff snippet exceeding 40 lines
        {
            "status": "ok",
            "effectiveRisk": "R2",
            "preview": {
                "kind": "diff",
                "summary": "too many lines",
                "snippet": "\n".join(["+a"] * 41),
                "truncated": False,
            },
        },
        # Truncated diff missing ellipsis marker
        {
            "status": "ok",
            "effectiveRisk": "R2",
            "preview": {
                "kind": "diff",
                "summary": "missing marker",
                "snippet": "+line1",
                "truncated": True,
            },
        },
        # Paths exceeding 20 items
        {
            "status": "ok",
            "effectiveRisk": "R3",
            "preview": {
                "kind": "paths",
                "summary": "too many",
                "paths": [f"f{i}.log" for i in range(21)],
                "totalCount": 21,
                "truncated": True,
            },
        },
        # totalCount < len(paths)
        {
            "status": "ok",
            "effectiveRisk": "R3",
            "preview": {
                "kind": "paths",
                "summary": "bad total",
                "paths": ["a.log", "b.log"],
                "totalCount": 1,
                "truncated": False,
            },
        },
        # Absolute path or traversal or secret file rejected
        {
            "status": "ok",
            "effectiveRisk": "R3",
            "preview": {
                "kind": "paths",
                "summary": "bad path",
                "paths": ["../escape.log"],
                "totalCount": 1,
                "truncated": False,
            },
        },
        {
            "status": "ok",
            "effectiveRisk": "R3",
            "preview": {
                "kind": "paths",
                "summary": "secret file",
                "paths": [".env"],
                "totalCount": 1,
                "truncated": False,
            },
        },
        {
            "status": "ok",
            "effectiveRisk": "R3",
            "preview": {
                "kind": "paths",
                "summary": "secret dir",
                "paths": [".ssh/id_rsa"],
                "totalCount": 1,
                "truncated": False,
            },
        },
    ],
)
def test_parse_preflight_result_rejects_invalid_preview(bad_preflight: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        parse_preflight_result({"preflight": bad_preflight})


def test_sanitize_preflight_preview_redacts_secrets_and_blocks_bad_paths() -> None:
    raw_snippet = "--- a/config.ts\n+++ b/config.ts\n" "+const k = 'sk-1234567890abcdefghijklmnop';"
    cleaned = sanitize_preflight_preview(
        {
            "kind": "diff",
            "summary": "config.ts: 1 replacement",
            "snippet": raw_snippet,
            "truncated": False,
        }
    )
    assert cleaned is not None
    snippet = str(cleaned["snippet"])
    assert "sk-1234567890abcdefghijklmnop" not in snippet
    assert len(snippet.encode("utf-8")) <= MAX_PREVIEW_SNIPPET_BYTES

    assert (
        sanitize_preflight_preview(
            {
                "kind": "paths",
                "summary": "secret",
                "paths": ["nested/.env.local"],
                "totalCount": 1,
                "truncated": False,
            }
        )
        is None
    )
    assert sanitize_preflight_preview("not-a-dict") is None


@pytest.mark.anyio
async def test_orchestrator_forwards_ephemeral_preview_on_pending_without_persisting() -> None:
    owner_id = uuid.uuid4()
    host_repo = _FakeHostRepository()
    hub = _FakeSessionHub()
    approval_repo = InMemoryApprovalRepository()
    audit_log = _FakeAuditLog()
    host = _make_host(owner_id, platform=HostPlatform.WINDOWS)
    host_repo.hosts[host.host_id] = host
    tools = _frames()["toolsListExecFsResponse"]["result"]["tools"]
    run_desc = ToolDescriptor.from_wire(next(t for t in tools if t["name"] == "exec.run"))
    host_repo.manifests[host.host_id] = Manifest(
        host_id=host.host_id, tools=(run_desc,), updated_at=NOW, schema_version=1
    )
    host_repo.grants[host.host_id] = [
        Grant(
            host_id=host.host_id,
            family=ToolFamily.EXEC,
            max_risk=RiskTier.R3,
            enabled=True,
            updated_at=NOW,
        )
    ]
    hub.custom_responses["xuna/preflight"] = _frames()["preflightExecDeletePreviewResponse"][
        "result"
    ]

    orch = Orchestrator(
        host_repo=host_repo,
        hub=hub,
        approval_repo=approval_repo,
        approval_broker=InMemoryApprovalBroker(),
        local_result_broker=InMemoryLocalResultBroker(),
        last_target_store=InMemoryLastTargetStore(),
        audit=audit_log,
        scope_store=InMemoryApprovalScopeStore(),
        clock=lambda: NOW,
    )
    collector = EventCollector(orch, owner_id)
    await orch.handle_call(
        ToolCall(
            id="w4m2-r3",
            name="exec__run",
            arguments={"command": "del logs\\*.log", "cwd": "C:/ws"},
        ),
        _make_action_context(owner_id, collector),
    )

    pending_events = list(collector.pending_events)
    assert len(pending_events) == 1
    wire = pending_events[0].wire_payload()
    assert wire["preview"] == {
        "kind": "paths",
        "summary": "3 matching file(s) in workspace",
        "paths": ["logs/a.log", "logs/b.log", "logs/c.log"],
        "totalCount": 3,
        "truncated": False,
    }

    # Ephemeral invariant: persisted ApprovalRequest and AuditLog never contain preview data.
    saved = await approval_repo.get(uuid.UUID(pending_events[0].approval_id))
    assert saved is not None
    assert "logs/a.log" not in saved.summary
    for audit_ev in audit_log.events:
        dumped = json.dumps(audit_ev.metadata)
        assert "logs/a.log" not in dumped
        assert "preview" not in dumped
