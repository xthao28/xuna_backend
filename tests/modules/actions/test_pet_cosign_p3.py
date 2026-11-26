"""Tests for ADR-0020 P3 (`F3-M2` pet co-signing & `F3-M3` signed E-Stop):

- Opt-in D6 policy default OFF; `GET`/`PUT /v1/actions/cosign-policy`.
- `ActionPendingEvent` includes `requiresPetCosign`, `cosignShortCode`, `cosignNonceHex`.
- Dual-signature `POST /v1/actions/approvals/{id}`: fails closed on missing, wrong shortCode,
  wrong target SHA-256, forged/revoked/unbound pet key, or replayed nonce; accepts valid dual sig.
- Signed `POST /v1/actions/stop`: verifies `ProofPurpose.STOP` when `petProof` supplied or required
  for `owner_board`; fails closed on invalid or replayed proof; audits `petSigned`.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.e2e.test_wave_c_e2e import _FakeAuditLog, _FakeHostRepository, _make_host
from tests.helpers import SimulatedDevice

from xuna_backend.core.db import request_session
from xuna_backend.core.errors import install_error_handlers
from xuna_backend.modules.accounts.api.dependencies import current_owner
from xuna_backend.modules.accounts.domain.owner import Owner
from xuna_backend.modules.actions.api.routes import router as actions_router
from xuna_backend.modules.actions.application.orchestrator import ActionContext, Orchestrator
from xuna_backend.modules.actions.domain.cosigning import (
    CosignCategory,
    InMemoryCosignChallengeStore,
    InMemoryCosignPolicyStore,
    InMemoryProofNonceStore,
    OwnerCosignPolicy,
    PetApproveProofInput,
    PetStopProofInput,
    classify_cosign_category,
)
from xuna_backend.modules.actions.domain.entities import (
    ActionPendingEvent,
    ActionStreamEvent,
    ApprovalDecision,
    ApprovalRequest,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.repository import InMemoryApprovalRepository
from xuna_backend.modules.cognition.domain.types import Locale, ToolCall
from xuna_backend.modules.hosts.domain.crypto import (
    approval_proof_message,
    approval_v2_proof_message,
)
from xuna_backend.modules.hosts.domain.entities import (
    Grant,
    Host,
    HostPlatform,
    HostStatus,
    Manifest,
    RiskTier,
    ToolDescriptor,
    ToolFamily,
)
from xuna_backend.modules.hosts.infrastructure.session_hub import InMemoryHostSessionHub
from xuna_backend.modules.identity.domain.device_proof import (
    ProofPurpose,
    approve_proof_payload,
    cosign_short_code,
    proof_message,
    stop_proof_payload,
)
from xuna_backend.modules.identity.domain.entities import Challenge, Pet
from xuna_backend.modules.identity.domain.pet_id import PetId, PetOrigin
from xuna_backend.modules.identity.infrastructure.p256 import P256SignatureVerifier

NOW = datetime(2026, 11, 26, 13, 0, 0, tzinfo=UTC)


class _FakeIdentityRepo:
    def __init__(self, pets: list[Pet] | None = None) -> None:
        self.pets: list[Pet] = list(pets or [])

    async def add_challenge(self, challenge: Challenge) -> None:
        pass

    async def consume_challenge(self, challenge_id: uuid.UUID) -> Challenge | None:
        return None

    async def get_pet(self, pet_id: PetId) -> Pet | None:
        return next((p for p in self.pets if p.pet_id == pet_id), None)

    async def get_pet_by_device(self, device_id: uuid.UUID) -> Pet | None:
        return next((p for p in self.pets if p.device_id == device_id), None)

    async def list_pets(self, owner_id: uuid.UUID) -> list[Pet]:
        return [p for p in self.pets if p.owner_id == owner_id]

    async def create_pet(
        self,
        *,
        pet_id: PetId,
        owner_id: uuid.UUID,
        origin: PetOrigin,
        public_key: bytes,
        generation: int,
        now: datetime,
    ) -> Pet:
        raise NotImplementedError

    async def update_generation(self, pet_id: PetId, generation: int) -> None:
        pass

    async def rebind_device(self, pet_id: PetId, public_key: bytes, now: datetime) -> Pet:
        raise NotImplementedError

    async def rotate_brain(self, pet_id: PetId, now: datetime) -> Pet:
        raise NotImplementedError

    async def revoke_device(self, pet_id: PetId, now: datetime) -> None:
        pass

    async def restore_device(self, pet_id: PetId, public_key: bytes, now: datetime) -> Pet:
        raise NotImplementedError


def _make_phone_host(owner_id: uuid.UUID, pub_key: bytes) -> Host:
    return dataclasses.replace(
        _make_host(owner_id, alias="iPhone", platform=HostPlatform.IOS),
        public_key=pub_key,
    )


def _make_pet(owner_id: uuid.UUID, device: SimulatedDevice, *, revoked: bool = False) -> Pet:
    return Pet(
        pet_id=PetId("d945c77026fd4f4a8a899bbff4aeed86"),
        owner_id=owner_id,
        origin=PetOrigin.HATCHED_VIA_APP,
        active_brain_id=uuid.uuid4(),
        device_id=uuid.uuid4(),
        device_public_key=device.public_key,
        last_generation=38,
        bound_at=NOW,
        device_revoked_at=NOW if revoked else None,
    )


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _sign_phone_v2(
    phone: SimulatedDevice,
    approval_id: str,
    effect_sha256: str,
    host_id: str,
    expires_at_iso: str,
) -> tuple[bytes, bytes]:
    msg = approval_v2_proof_message(approval_id, effect_sha256, host_id, expires_at_iso)
    return phone.sign(msg), phone.public_key


def _sign_phone_v1(
    phone: SimulatedDevice,
    approval_id: str,
    args_sha256_hex: str,
    host_id: str,
    expires_at_iso: str,
) -> tuple[bytes, bytes]:
    msg = approval_proof_message(approval_id, args_sha256_hex, host_id, expires_at_iso)
    return phone.sign(msg), phone.public_key


def _sign_pet_approve(
    pet_dev: SimulatedDevice,
    pet_id: PetId,
    generation: int,
    nonce: bytes,
    target_hex: str,
    *,
    short_code_override: str | None = None,
) -> PetApproveProofInput:
    target_bytes = bytes.fromhex(target_hex)
    code = short_code_override or cosign_short_code(nonce, target_bytes)
    msg = proof_message(
        nonce,
        pet_id,
        ProofPurpose.APPROVE,
        generation,
        approve_proof_payload(target_bytes, code),
    )
    return PetApproveProofInput(
        pet_id=pet_id.value,
        nonce_hex=nonce.hex(),
        generation=generation,
        short_code=code,
        signature=pet_dev.sign(msg),
        public_key=pet_dev.public_key,
    )


def _sign_pet_stop(
    pet_dev: SimulatedDevice,
    pet_id: PetId,
    generation: int,
    nonce: bytes,
) -> PetStopProofInput:
    msg = proof_message(nonce, pet_id, ProofPurpose.STOP, generation, stop_proof_payload())
    return PetStopProofInput(
        pet_id=pet_id.value,
        nonce_hex=nonce.hex(),
        generation=generation,
        signature=pet_dev.sign(msg),
        public_key=pet_dev.public_key,
    )


def _make_api_client(
    *,
    owner: Owner,
    phone_host: Host,
    approval_req: ApprovalRequest | None = None,
    challenge: tuple[bytes, str] | None = None,
    pets: list[Pet] | None = None,
) -> tuple[TestClient, FastAPI, _FakeAuditLog]:
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(actions_router)
    host_repo = _FakeHostRepository()
    asyncio.run(host_repo.save_host(phone_host))
    approval_repo = InMemoryApprovalRepository()
    broker = InMemoryApprovalBroker()
    audit = _FakeAuditLog()
    challenge_store = InMemoryCosignChallengeStore()
    policy_store = InMemoryCosignPolicyStore()
    nonce_store = InMemoryProofNonceStore()

    if approval_req is not None:
        asyncio.run(approval_repo.save(approval_req))
        asyncio.run(broker.create(approval_req))
        if challenge is not None:
            nonce, _ = challenge
            challenge_store.create(
                approval_id=approval_req.approval_id,
                owner_id=owner.owner_id,
                target_sha256_hex=approval_req.effect_sha256 or approval_req.args_sha256,
                category=CosignCategory.DRAFT_APPLY,
                nonce=nonce,
            )

    class _DummySession:
        async def commit(self) -> None:
            pass

    app.state.host_repository = lambda _s: host_repo
    app.state.approval_repository = lambda _s: approval_repo
    app.state.audit_log = lambda _s: audit
    app.state.host_session_hub = InMemoryHostSessionHub()
    app.state.approval_broker = broker
    app.state.local_result_broker = InMemoryLocalResultBroker()
    app.state.last_target_store = InMemoryLastTargetStore()
    app.state.signature_verifier = P256SignatureVerifier()
    app.state.identity_repository = _FakeIdentityRepo(pets)
    app.state.cosign_policy_store = policy_store
    app.state.cosign_challenge_store = challenge_store
    app.state.proof_nonce_store = nonce_store

    async def _owner_dep() -> Owner:
        return owner

    def _session_dep() -> Iterator[_DummySession]:
        yield _DummySession()

    app.dependency_overrides[current_owner] = _owner_dep
    app.dependency_overrides[request_session] = _session_dep
    return TestClient(app), app, audit


def test_cosign_policy_defaults_off_and_category_classification() -> None:
    store = InMemoryCosignPolicyStore()
    owner_id = uuid.uuid4()
    default_policy = store.get_policy(owner_id)
    assert default_policy.enabled_categories == frozenset()
    assert default_policy.require_signed_board_stop is False

    assert (
        classify_cosign_category("draft.apply", {}, risk=RiskTier.R3) == CosignCategory.DRAFT_APPLY
    )
    assert (
        classify_cosign_category(
            "fs.write_file", {}, risk=RiskTier.R3, preflight_reason="outside_workspace"
        )
        == CosignCategory.OUTSIDE_WORKSPACE_R3
    )
    assert (
        classify_cosign_category("exec.run", {"command": "del *.log"}, risk=RiskTier.R3)
        == CosignCategory.BULK_DELETE
    )
    assert (
        classify_cosign_category("exec.run", {"command": "push origin dev"}, risk=RiskTier.R3)
        == CosignCategory.REMOTE_PUSH
    )
    assert classify_cosign_category("draft.apply", {}, risk=RiskTier.R2) is None


def test_cosign_policy_routes_get_and_put() -> None:
    owner_id = uuid.uuid4()
    owner = Owner(owner_id=owner_id, issuer="https://auth.test.xuna", subject="owner-1")
    phone = SimulatedDevice()
    phone_host = _make_phone_host(owner_id, phone.public_key)
    client, _app, _audit = _make_api_client(owner=owner, phone_host=phone_host)

    r_get = client.get("/v1/actions/cosign-policy")
    assert r_get.status_code == 200
    assert r_get.json() == {
        "schemaVersion": 1,
        "enabledCategories": [],
        "requireSignedBoardStop": False,
    }

    r_put = client.put(
        "/v1/actions/cosign-policy",
        json={
            "schemaVersion": 1,
            "enabledCategories": ["draft_apply", "bulk_delete"],
            "requireSignedBoardStop": True,
        },
    )
    assert r_put.status_code == 200
    assert r_put.json()["enabledCategories"] == ["bulk_delete", "draft_apply"]
    assert r_put.json()["requireSignedBoardStop"] is True


def test_orchestrator_emits_cosign_challenge_only_when_owner_opted_in() -> None:
    owner_id = uuid.uuid4()
    work_host = dataclasses.replace(
        _make_host(owner_id, alias="WorkPC", platform=HostPlatform.WINDOWS),
        status=HostStatus.ACTIVE,
    )
    draft_desc = ToolDescriptor(
        name="draft.apply",
        family=ToolFamily.FILES,
        risk=RiskTier.R3,
        description="Apply draft workspace",
        input_schema={
            "type": "object",
            "properties": {
                "draftId": {"type": "string"},
                "effectSha256": {"type": "string"},
            },
            "required": ["draftId", "effectSha256"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"applied": {"type": "boolean"}},
            "additionalProperties": False,
        },
    )
    manifest = Manifest(host_id=work_host.host_id, tools=(draft_desc,), updated_at=NOW)
    grants = [
        Grant(
            host_id=work_host.host_id,
            family=ToolFamily.FILES,
            max_risk=RiskTier.R3,
            enabled=True,
            updated_at=NOW,
        )
    ]
    effect_hex = "a" * 64

    async def _run_case(opt_in: bool) -> ActionPendingEvent:
        host_repo = _FakeHostRepository()
        await host_repo.save_host(work_host)
        await host_repo.save_manifest(manifest)
        await host_repo.set_grants(work_host.host_id, grants)
        approval_repo = InMemoryApprovalRepository()
        broker = InMemoryApprovalBroker()

        class _StubHub(InMemoryHostSessionHub):
            def is_online(self, host_id: str) -> bool:
                return True

            async def send_request(
                self,
                host_id: str,
                method: str,
                params: dict[str, object],
                timeout_s: float = 10.0,
            ) -> dict[str, object]:
                if method == "xuna/preflight":
                    return {
                        "preflight": {
                            "status": "ok",
                            "effectiveRisk": "R3",
                            "reasonCode": "draft_apply",
                            "effectSha256": effect_hex,
                        }
                    }
                return {
                    "content": [{"type": "text", "text": "applied"}],
                    "structuredContent": {"applied": True},
                }

        hub = _StubHub()
        policy_store = InMemoryCosignPolicyStore()
        challenge_store = InMemoryCosignChallengeStore()
        if opt_in:
            policy_store.set_policy(
                owner_id,
                OwnerCosignPolicy(enabled_categories=frozenset({CosignCategory.DRAFT_APPLY})),
            )

        orch = Orchestrator(
            host_repo=host_repo,
            hub=hub,
            approval_repo=approval_repo,
            approval_broker=broker,
            local_result_broker=InMemoryLocalResultBroker(),
            last_target_store=InMemoryLastTargetStore(),
            audit=_FakeAuditLog(),
            verifier=P256SignatureVerifier(),
            cosign_policy_store=policy_store,
            cosign_challenge_store=challenge_store,
            nonce_store=InMemoryProofNonceStore(),
            clock=lambda: NOW,
        )
        recorded: list[ActionStreamEvent] = []

        async def _emit(ev: ActionStreamEvent) -> None:
            recorded.append(ev)

        ctx = ActionContext(
            owner_id=owner_id,
            pet_id="d945c77026fd4f4a8a899bbff4aeed86",
            locale=Locale.VI,
            client_caps=("host_actions_v1",),
            board_available=False,
            emit_event=_emit,
            request_id="req-cosign",
        )
        task = asyncio.create_task(
            orch.handle_call(
                ToolCall(
                    id="c1",
                    name="draft__apply",
                    arguments={
                        "target": "WorkPC",
                        "draftId": "d1",
                        "effectSha256": effect_hex,
                    },
                ),
                ctx,
            )
        )
        for _ in range(30):
            await asyncio.sleep(0.01)
            pend = [e for e in recorded if isinstance(e, ActionPendingEvent)]
            if pend:
                await broker.decide(
                    approval_id=uuid.UUID(pend[0].approval_id),
                    owner_id=owner_id,
                    decision=ApprovalDecision.REJECT,
                    now=NOW,
                )
                await task
                return pend[0]
        raise AssertionError("ActionPendingEvent was not emitted")

    default_ev = asyncio.run(_run_case(opt_in=False))
    assert default_ev.requires_pet_cosign is False
    assert "requiresPetCosign" not in default_ev.wire_payload()

    opt_in_ev = asyncio.run(_run_case(opt_in=True))
    assert opt_in_ev.requires_pet_cosign is True
    assert opt_in_ev.cosign_short_code is not None and len(opt_in_ev.cosign_short_code) == 6
    assert opt_in_ev.cosign_nonce_hex is not None and len(opt_in_ev.cosign_nonce_hex) == 64
    wire = opt_in_ev.wire_payload()
    assert wire["requiresPetCosign"] is True
    assert wire["cosignShortCode"] == opt_in_ev.cosign_short_code
    assert (
        cosign_short_code(bytes.fromhex(opt_in_ev.cosign_nonce_hex), bytes.fromhex(effect_hex))
        == opt_in_ev.cosign_short_code
    )


def test_dual_cosign_approval_enforces_matching_pet_proof() -> None:
    owner_id = uuid.uuid4()
    owner = Owner(owner_id=owner_id, issuer="https://auth.test.xuna", subject="owner-1")
    phone = SimulatedDevice()
    pet_dev = SimulatedDevice()
    stranger_pet = SimulatedDevice()
    phone_host = _make_phone_host(owner_id, phone.public_key)
    pet = _make_pet(owner_id, pet_dev)

    approval_id = uuid.uuid4()
    effect_hex = "b" * 64
    now_live = datetime.now(UTC)
    expires_at = now_live + timedelta(seconds=60)
    nonce = bytes(range(32))
    req = ApprovalRequest(
        approval_id=approval_id,
        owner_id=owner_id,
        host_id=phone_host.host_id,
        target_alias="WorkPC",
        tool="draft.apply",
        args_sha256="1" * 64,
        risk=RiskTier.R3,
        summary="Apply workspace draft",
        created_at=now_live,
        expires_at=expires_at,
        approval_version=2,
        effect_sha256=effect_hex,
    )
    phone_sig, phone_pub = _sign_phone_v2(
        phone,
        str(approval_id),
        effect_hex,
        str(phone_host.host_id),
        expires_at.astimezone(UTC).isoformat(),
    )

    client, _app, _audit = _make_api_client(
        owner=owner,
        phone_host=phone_host,
        approval_req=req,
        challenge=(nonce, cosign_short_code(nonce, bytes.fromhex(effect_hex))),
        pets=[pet],
    )

    # 1. Phone signature alone (no petProof) -> MUST FAIL CLOSED with pet_cosign_required.
    resp_missing = client.post(
        f"/v1/actions/approvals/{approval_id}",
        json={
            "schemaVersion": 1,
            "decision": "approve",
            "scope": "once",
            "signature": _b64(phone_sig),
            "publicKey": _b64(phone_pub),
        },
    )
    assert resp_missing.status_code == 400
    assert resp_missing.json()["code"] == "pet_cosign_required"

    # 2. Wrong short code -> pet_cosign_invalid.
    wrong_code_proof = _sign_pet_approve(
        pet_dev, pet.pet_id, 38, nonce, effect_hex, short_code_override="000000"
    )
    resp_wrong_code = client.post(
        f"/v1/actions/approvals/{approval_id}",
        json={
            "schemaVersion": 1,
            "decision": "approve",
            "scope": "once",
            "signature": _b64(phone_sig),
            "publicKey": _b64(phone_pub),
            "petProof": {
                "petId": wrong_code_proof.pet_id,
                "nonceHex": wrong_code_proof.nonce_hex,
                "generation": wrong_code_proof.generation,
                "shortCode": wrong_code_proof.short_code,
                "signature": _b64(wrong_code_proof.signature),
                "publicKey": _b64(wrong_code_proof.public_key),
            },
        },
    )
    assert resp_wrong_code.status_code == 400
    assert resp_wrong_code.json()["code"] == "pet_cosign_invalid"

    # 3. Unbound / stranger pet key -> pet_cosign_invalid.
    stranger_proof = _sign_pet_approve(stranger_pet, pet.pet_id, 38, nonce, effect_hex)
    resp_stranger = client.post(
        f"/v1/actions/approvals/{approval_id}",
        json={
            "schemaVersion": 1,
            "decision": "approve",
            "scope": "once",
            "signature": _b64(phone_sig),
            "publicKey": _b64(phone_pub),
            "petProof": {
                "petId": stranger_proof.pet_id,
                "nonceHex": stranger_proof.nonce_hex,
                "generation": stranger_proof.generation,
                "shortCode": stranger_proof.short_code,
                "signature": _b64(stranger_proof.signature),
                "publicKey": _b64(stranger_proof.public_key),
            },
        },
    )
    assert resp_stranger.status_code == 400
    assert resp_stranger.json()["code"] == "pet_cosign_invalid"

    # 4. Valid bound pet + matching code + valid phone v2 signature -> APPROVED.
    good_proof = _sign_pet_approve(pet_dev, pet.pet_id, 38, nonce, effect_hex)
    resp_ok = client.post(
        f"/v1/actions/approvals/{approval_id}",
        json={
            "schemaVersion": 1,
            "decision": "approve",
            "scope": "once",
            "signature": _b64(phone_sig),
            "publicKey": _b64(phone_pub),
            "petProof": {
                "petId": good_proof.pet_id,
                "nonceHex": good_proof.nonce_hex,
                "generation": good_proof.generation,
                "shortCode": good_proof.short_code,
                "signature": _b64(good_proof.signature),
                "publicKey": _b64(good_proof.public_key),
            },
        },
    )
    assert resp_ok.status_code == 200
    assert resp_ok.json()["state"] == "approved"


def test_dual_cosign_replay_check_with_store() -> None:
    """Directly checks nonce replay & revoked pet fail-closed across two approvals."""
    owner_id = uuid.uuid4()
    owner = Owner(owner_id=owner_id, issuer="https://auth.test.xuna", subject="owner-1")
    phone = SimulatedDevice()
    pet_dev = SimulatedDevice()
    phone_host = _make_phone_host(owner_id, phone.public_key)
    pet = _make_pet(owner_id, pet_dev)

    client, app, _audit = _make_api_client(owner=owner, phone_host=phone_host, pets=[pet])
    ap1 = uuid.uuid4()
    ap2 = uuid.uuid4()
    args_hex = "d" * 64
    now_live = datetime.now(UTC)
    expires = now_live + timedelta(seconds=60)
    nonce = bytes([7] * 32)
    for aid in (ap1, ap2):
        req = ApprovalRequest(
            approval_id=aid,
            owner_id=owner_id,
            host_id=phone_host.host_id,
            target_alias="WorkPC",
            tool="exec.run",
            args_sha256=args_hex,
            risk=RiskTier.R3,
            summary="Delete temp logs",
            created_at=now_live,
            expires_at=expires,
        )
        asyncio.run(app.state.approval_repository(None).save(req))
        asyncio.run(app.state.approval_broker.create(req))
        app.state.cosign_challenge_store.create(
            approval_id=aid,
            owner_id=owner_id,
            target_sha256_hex=args_hex,
            category=CosignCategory.BULK_DELETE,
            nonce=nonce,
        )

    proof = _sign_pet_approve(pet_dev, pet.pet_id, 38, nonce, args_hex)
    sig1, pub1 = _sign_phone_v1(
        phone, str(ap1), args_hex, str(phone_host.host_id), expires.astimezone(UTC).isoformat()
    )
    r1 = client.post(
        f"/v1/actions/approvals/{ap1}",
        json={
            "schemaVersion": 1,
            "decision": "approve",
            "scope": "once",
            "signature": _b64(sig1),
            "publicKey": _b64(pub1),
            "petProof": {
                "petId": proof.pet_id,
                "nonceHex": proof.nonce_hex,
                "generation": proof.generation,
                "shortCode": proof.short_code,
                "signature": _b64(proof.signature),
                "publicKey": _b64(proof.public_key),
            },
        },
    )
    assert r1.status_code == 200

    sig2, pub2 = _sign_phone_v1(
        phone, str(ap2), args_hex, str(phone_host.host_id), expires.astimezone(UTC).isoformat()
    )
    r2 = client.post(
        f"/v1/actions/approvals/{ap2}",
        json={
            "schemaVersion": 1,
            "decision": "approve",
            "scope": "once",
            "signature": _b64(sig2),
            "publicKey": _b64(pub2),
            "petProof": {
                "petId": proof.pet_id,
                "nonceHex": proof.nonce_hex,
                "generation": proof.generation,
                "shortCode": proof.short_code,
                "signature": _b64(proof.signature),
                "publicKey": _b64(proof.public_key),
            },
        },
    )
    assert r2.status_code == 400
    assert r2.json()["code"] == "pet_cosign_replayed"


def test_signed_emergency_stop_f3m3() -> None:
    owner_id = uuid.uuid4()
    owner = Owner(owner_id=owner_id, issuer="https://auth.test.xuna", subject="owner-1")
    phone = SimulatedDevice()
    pet_dev = SimulatedDevice()
    phone_host = _make_phone_host(owner_id, phone.public_key)
    pet = _make_pet(owner_id, pet_dev)
    client, _app, audit = _make_api_client(owner=owner, phone_host=phone_host, pets=[pet])

    # Default (requireSignedBoardStop=False): unsigned stop works and records petSigned=False.
    r_unsigned = client.post("/v1/actions/stop", json={"schemaVersion": 1, "reason": "owner_board"})
    assert r_unsigned.status_code == 200
    assert r_unsigned.json()["petSigned"] is False
    assert audit.events[-1].metadata["petSigned"] is False

    # Forged Stop proof fails closed even when optional.
    bad_nonce = bytes([9] * 32)
    r_bad = client.post(
        "/v1/actions/stop",
        json={
            "schemaVersion": 1,
            "reason": "owner_board",
            "petProof": {
                "petId": pet.pet_id.value,
                "nonceHex": bad_nonce.hex(),
                "generation": 38,
                "signature": _b64(bytes(64)),
                "publicKey": _b64(pet_dev.public_key),
            },
        },
    )
    assert r_bad.status_code == 400
    assert r_bad.json()["code"] == "pet_stop_proof_invalid"

    # Enable strict signed board stop policy -> unsigned refused, signed ok, replay fails.
    r_pol = client.put(
        "/v1/actions/cosign-policy",
        json={"schemaVersion": 1, "enabledCategories": [], "requireSignedBoardStop": True},
    )
    assert r_pol.status_code == 200

    r_req_missing = client.post(
        "/v1/actions/stop", json={"schemaVersion": 1, "reason": "owner_board"}
    )
    assert r_req_missing.status_code == 400
    assert r_req_missing.json()["code"] == "pet_stop_proof_required"

    good_nonce = bytes([11] * 32)
    stop_proof = _sign_pet_stop(pet_dev, pet.pet_id, 38, good_nonce)
    body = {
        "schemaVersion": 1,
        "reason": "owner_board",
        "petProof": {
            "petId": stop_proof.pet_id,
            "nonceHex": stop_proof.nonce_hex,
            "generation": stop_proof.generation,
            "signature": _b64(stop_proof.signature),
            "publicKey": _b64(stop_proof.public_key),
        },
    }
    r_signed = client.post("/v1/actions/stop", json=body)
    assert r_signed.status_code == 200
    assert r_signed.json()["petSigned"] is True
    assert audit.events[-1].metadata["petSigned"] is True
    assert audit.events[-1].metadata["petId"] == pet.pet_id.value

    r_replay = client.post("/v1/actions/stop", json=body)
    assert r_replay.status_code == 400
    assert r_replay.json()["code"] == "pet_stop_proof_replayed"
