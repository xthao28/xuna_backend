"""W4-M3 / W4-M4 tests: `draft.*` tools, `EffectManifest` golden vector, and `xuna.approval.v2`."""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils
from tests.e2e.test_wave_c_e2e import (
    _FakeAuditLog,
    _FakeHostRepository,
    _FakeSessionHub,
    _make_host,
)

from xuna_backend.modules.actions.application.orchestrator import ActionContext, Orchestrator
from xuna_backend.modules.actions.domain.entities import (
    ActionPendingEvent,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalScope,
    ApprovalState,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.repository import InMemoryApprovalRepository
from xuna_backend.modules.cognition.domain.types import Locale, ToolCall
from xuna_backend.modules.hosts.application.xtp_frames import parse_preflight_result
from xuna_backend.modules.hosts.domain.crypto import (
    approval_proof_message,
    approval_v2_proof_message,
    args_sha256,
    canonical_json,
)
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    HostId,
    HostPlatform,
    Manifest,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
)
from xuna_backend.modules.identity.infrastructure.p256 import P256SignatureVerifier

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "xtp"


def _load_json(name: str) -> dict[str, Any]:
    raw: dict[str, Any] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return raw


def _sign_raw_p256(priv: ec.EllipticCurvePrivateKey, message: bytes) -> bytes:
    sig_der = priv.sign(message, ec.ECDSA(hashes.SHA256()))
    r, s = utils.decode_dss_signature(sig_der)
    return r.to_bytes(32, "big") + s.to_bytes(32, "big")


def test_golden_approval_v2_vector_matches_python_crypto_and_verifier() -> None:
    golden = _load_json("golden_vectors.json")
    v2 = golden["approvalV2"]

    canon = canonical_json(v2["manifest"])
    assert canon == v2["canonicalManifest"]
    assert hashlib.sha256(canon.encode("utf-8")).hexdigest() == v2["effectSha256"]

    msg = approval_v2_proof_message(
        v2["approvalId"],
        v2["effectSha256"],
        v2["hostId"],
        v2["expiresAt"],
    )
    assert msg.hex() == v2["messageHex"]
    assert hashlib.sha256(msg).hexdigest() == v2["messageSha256Hex"]

    pub = bytes.fromhex(v2["publicKeyHex"])
    sig = bytes.fromhex(v2["signatureHex"])
    verifier = P256SignatureVerifier()
    assert verifier.verify(pub, msg, sig) is True

    # v1 message with same hash must NOT verify with v2 signature (domain separation).
    v1_msg = approval_proof_message(
        v2["approvalId"],
        v2["effectSha256"],
        v2["hostId"],
        v2["expiresAt"],
    )
    assert verifier.verify(pub, v1_msg, sig) is False


def test_draft_fixture_descriptors_and_preflight_effect_sha256_parse() -> None:
    frames = _load_json("exec_fs_frames.json")["frames"]
    draft_tools = [
        ToolDescriptor.from_wire(t) for t in frames["toolsListDraftResponse"]["result"]["tools"]
    ]
    assert {d.name for d in draft_tools} == {
        "draft.create",
        "draft.manifest",
        "draft.discard",
        "draft.apply",
    }
    for d in draft_tools:
        assert d.family == ToolFamily.FILES

    pf = parse_preflight_result(frames["preflightDraftApplyResponse"]["result"])
    assert pf.status == "ok"
    assert pf.effective_risk == RiskTier.R3
    assert pf.reason_code == "draft_apply"
    assert pf.effect_sha256 == _load_json("golden_vectors.json")["approvalV2"]["effectSha256"]
    assert pf.preview is not None and pf.preview["kind"] == "paths"

    # Fail-closed: refused preflight carrying effectSha256 is rejected.
    with pytest.raises(ValueError):
        parse_preflight_result(
            {
                "preflight": {
                    "status": "denied",
                    "errorCode": "draft_not_found",
                    "effectSha256": "a" * 64,
                }
            }
        )


def test_approval_request_validates_v1_and_v2_invariants() -> None:
    now = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    base_kwargs: dict[str, Any] = {
        "approval_id": uuid.uuid4(),
        "owner_id": uuid.uuid4(),
        "host_id": HostId.generate(),
        "target_alias": "DevPC",
        "tool": "draft.apply",
        "args_sha256": "a" * 64,
        "risk": RiskTier.R3,
        "summary": "Apply draft",
        "created_at": now,
        "expires_at": now + timedelta(seconds=60),
    }
    req_v2 = ApprovalRequest(
        **base_kwargs,
        approval_version=2,
        effect_sha256="b" * 64,
    )
    assert req_v2.approval_version == 2

    with pytest.raises(ValueError):
        ApprovalRequest(**base_kwargs, approval_version=2, effect_sha256=None)
    with pytest.raises(ValueError):
        ApprovalRequest(**base_kwargs, approval_version=1, effect_sha256="b" * 64)
    with pytest.raises(ValueError):
        ApprovalRequest(
            **base_kwargs,
            approval_version=2,
            effect_sha256="b" * 64,
            scope=ApprovalScope.CONVERSATION_15M,
        )


@pytest.mark.anyio
async def test_orchestrator_draft_apply_enforces_v2_signature_and_forwards_wire_meta() -> None:
    now = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    owner_id = uuid.uuid4()
    phone_priv = ec.generate_private_key(ec.SECP256R1())
    phone_pub = phone_priv.public_key().public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )

    desktop_host = _make_host(owner_id, alias="DevPC", platform=HostPlatform.WINDOWS)
    host_id = desktop_host.host_id
    phone_host = dataclasses.replace(
        _make_host(owner_id, alias="MyPhone", platform=HostPlatform.IOS),
        public_key=phone_pub,
    )

    frames = _load_json("exec_fs_frames.json")["frames"]
    draft_tools = tuple(
        ToolDescriptor.from_wire(t) for t in frames["toolsListDraftResponse"]["result"]["tools"]
    )
    host_repo = _FakeHostRepository()
    host_repo.hosts[host_id] = desktop_host
    host_repo.hosts[phone_host.host_id] = phone_host
    host_repo.manifests[host_id] = Manifest(
        host_id=host_id, tools=draft_tools, updated_at=now, schema_version=1
    )
    host_repo.grants[host_id] = [
        Grant(
            host_id=host_id,
            family=ToolFamily.FILES,
            max_risk=RiskTier.R3,
            enabled=True,
            updated_at=now,
        )
    ]

    effect_hash = "c" * 64
    captured_calls: list[dict[str, Any]] = []
    hub = _FakeSessionHub()
    hub.custom_responses["xuna/preflight"] = {
        "preflight": {
            "status": "ok",
            "effectiveRisk": "R3",
            "reasonCode": "draft_apply",
            "effectSha256": effect_hash,
            "preview": {
                "kind": "paths",
                "summary": "Apply 1 file",
                "paths": ["src/app.py"],
                "totalCount": 1,
                "truncated": False,
            },
        }
    }
    orig_send = hub.send_request

    async def _capturing_send(
        hid: HostId,
        method: str,
        params: dict[str, Any],
        *,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        if method == "tools/call":
            captured_calls.append(params)
            return {
                "content": [{"type": "text", "text": "applied"}],
                "structuredContent": {
                    "draftId": "draft_demo_1",
                    "effectSha256": effect_hash,
                    "appliedCount": 1,
                    "createdCount": 0,
                    "modifiedCount": 1,
                    "deletedCount": 0,
                    "undoTxId": "tx_1",
                    "undoAvailable": True,
                },
                "isError": False,
                "_meta": {"xuna/status": "ok", "xuna/durationMs": 12},
            }
        return await orig_send(hid, method, params, timeout_s=timeout_s)

    hub.send_request = _capturing_send  # type: ignore[method-assign]

    approval_repo = InMemoryApprovalRepository()
    broker = InMemoryApprovalBroker()
    orch = Orchestrator(
        host_repo=host_repo,
        hub=hub,
        approval_repo=approval_repo,
        approval_broker=broker,
        local_result_broker=InMemoryLocalResultBroker(),
        last_target_store=InMemoryLastTargetStore(),
        audit=_FakeAuditLog(),
        verifier=P256SignatureVerifier(),
        clock=lambda: now,
    )

    pending_events: list[ActionPendingEvent] = []

    async def _emit(event: Any) -> None:
        if isinstance(event, ActionPendingEvent):
            pending_events.append(event)

    ctx = ActionContext(
        owner_id=owner_id,
        pet_id="pet_1",
        locale=Locale.VI,
        client_caps=("host_actions_v1",),
        board_available=False,
        emit_event=_emit,
    )
    apply_args: dict[str, object] = {
        "draftId": "draft_demo_1",
        "effectSha256": effect_hash,
    }

    async def _approve_with_v2_only() -> None:
        for _ in range(50):
            if pending_events:
                break
            await asyncio.sleep(0.01)
        assert pending_events
        ev = pending_events[-1]
        wire = ev.wire_payload()
        assert wire["approvalVersion"] == 2
        assert wire["effectSha256"] == effect_hash
        assert wire["preview"]["kind"] == "paths"  # type: ignore[index]

        aid = uuid.UUID(ev.approval_id)
        # 1. Unsigned must fail closed.
        with pytest.raises(ValueError, match="approval_signature_invalid"):
            await orch.decide_approval(
                approval_id=aid,
                owner_id=owner_id,
                decision=ApprovalDecision.APPROVE,
                scope=ApprovalScope.ONCE,
            )
        # 2. v1 signature over argsSha256 must fail closed!
        v1_sig = _sign_raw_p256(
            phone_priv,
            approval_proof_message(
                ev.approval_id,
                args_sha256(apply_args),
                str(host_id),
                ev.expires_at,
            ),
        )
        with pytest.raises(ValueError, match="approval_signature_invalid"):
            await orch.decide_approval(
                approval_id=aid,
                owner_id=owner_id,
                decision=ApprovalDecision.APPROVE,
                scope=ApprovalScope.ONCE,
                signature=v1_sig,
                signer_public_key=phone_pub,
            )
        # 3. Valid v2 signature over effectSha256 succeeds!
        v2_sig = _sign_raw_p256(
            phone_priv,
            approval_v2_proof_message(ev.approval_id, effect_hash, str(host_id), ev.expires_at),
        )
        updated = await orch.decide_approval(
            approval_id=aid,
            owner_id=owner_id,
            decision=ApprovalDecision.APPROVE,
            scope=ApprovalScope.ONCE,
            signature=v2_sig,
            signer_public_key=phone_pub,
        )
        assert updated is not None and updated.state == ApprovalState.APPROVED

    approver = asyncio.create_task(_approve_with_v2_only())
    msg = await orch.handle_call(
        ToolCall(id="call_1", name="draft__apply", arguments=apply_args),
        ctx,
    )
    await approver

    payload = json.loads(msg.content)
    assert payload["status"] == "ok"
    assert len(captured_calls) == 1
    wire_approval = captured_calls[0]["_meta"]["xuna/approval"]
    assert wire_approval["version"] == 2
    assert wire_approval["effectSha256"] == effect_hash
    assert base64.b64decode(wire_approval["publicKey"]) == phone_pub


@pytest.mark.anyio
async def test_orchestrator_draft_apply_refuses_preflight_effect_mismatch() -> None:
    now = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    owner_id = uuid.uuid4()
    desktop_host = _make_host(owner_id, alias="DevPC", platform=HostPlatform.WINDOWS)
    host_id = desktop_host.host_id

    frames = _load_json("exec_fs_frames.json")["frames"]
    draft_tools = tuple(
        ToolDescriptor.from_wire(t) for t in frames["toolsListDraftResponse"]["result"]["tools"]
    )
    host_repo = _FakeHostRepository()
    host_repo.hosts[host_id] = desktop_host
    host_repo.manifests[host_id] = Manifest(
        host_id=host_id, tools=draft_tools, updated_at=now, schema_version=1
    )
    host_repo.grants[host_id] = [
        Grant(
            host_id=host_id,
            family=ToolFamily.FILES,
            max_risk=RiskTier.R3,
            enabled=True,
            updated_at=now,
        )
    ]

    hub = _FakeSessionHub()
    hub.custom_responses["xuna/preflight"] = {
        "preflight": {
            "status": "ok",
            "effectiveRisk": "R3",
            "reasonCode": "draft_apply",
            "effectSha256": "d" * 64,
        }
    }

    orch = Orchestrator(
        host_repo=host_repo,
        hub=hub,
        approval_repo=InMemoryApprovalRepository(),
        approval_broker=InMemoryApprovalBroker(),
        local_result_broker=InMemoryLocalResultBroker(),
        last_target_store=InMemoryLastTargetStore(),
        audit=_FakeAuditLog(),
        verifier=P256SignatureVerifier(),
        clock=lambda: now,
    )
    pending_events: list[ActionPendingEvent] = []

    async def _emit(event: Any) -> None:
        if isinstance(event, ActionPendingEvent):
            pending_events.append(event)

    ctx = ActionContext(
        owner_id=owner_id,
        pet_id="pet_1",
        locale=Locale.VI,
        client_caps=("host_actions_v1",),
        board_available=False,
        emit_event=_emit,
    )
    msg = await orch.handle_call(
        ToolCall(
            id="call_bad",
            name="draft__apply",
            arguments={"draftId": "draft_demo_1", "effectSha256": "e" * 64},
        ),
        ctx,
    )
    payload = json.loads(msg.content)
    assert payload["status"] == "denied"
    assert payload["output"]["errorCode"] == "effect_sha256_mismatch"
    assert not pending_events
