"""Unit and PostgreSQL integration tests for KX-3 knowledge packs and brief-drafts."""

from __future__ import annotations

import base64
import os
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from tests.helpers import LocalIssuer, SimulatedDevice, b64, idem
from xuna_backend.core.config import Environment, Settings
from xuna_backend.main import create_app
from xuna_backend.modules.cognition.application.ports import ProviderUnavailableError
from xuna_backend.modules.cognition.domain.types import (
    ChatEvent,
    ChatFinished,
    ChatRequest,
    TextDelta,
)
from xuna_backend.modules.conversation.application.chat_turn import (
    TurnContext,
    build_request,
)
from xuna_backend.modules.conversation.domain.turn_mode import TurnMode
from xuna_backend.modules.identity.domain.device_proof import ProofPurpose, proof_message
from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.knowledge.application.brief_draft import (
    BRIEF_DRAFT_SYSTEM_VI,
    DraftExpertBriefUseCase,
)
from xuna_backend.modules.knowledge.domain.owner_context import DomainBrief, OwnerContext
from xuna_backend.modules.knowledge.domain.packs import get_pack

ROOT = Path(__file__).resolve().parents[3]
PET = "d945c77026fd4f4a8a899bbff4aeed86"
TABLES = (
    "knowledge_summaries, knowledge_suggestions, knowledge_domains, "
    "memory_candidates, memory_projections, idempotency_keys, "
    "device_challenges, devices, brains, pets, owners"
)


class _ScriptedBriefGenerator:
    def __init__(self, chunks: list[str], *, fail: bool = False) -> None:
        self._chunks = chunks
        self._fail = fail
        self.requests: list[ChatRequest] = []

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        self.requests.append(request)
        if self._fail:
            raise ProviderUnavailableError("offline")
        for chunk in self._chunks:
            yield TextDelta(chunk)
        yield ChatFinished(
            provider="fake",
            model="fake-expert",
            input_tokens=20,
            output_tokens=40,
            latency_ms=10,
        )


@pytest.mark.anyio
async def test_draft_expert_brief_use_case_sanitizes_and_truncates() -> None:
    gen = _ScriptedBriefGenerator(
        [
            "<g:bounce_play><say>Cách làm việc chuyên lĩnh vực Điện mặt trời:</say>\n",
            "-   Kiểm tra   công suất tấm pin và biến tần.\n\n\n\n",
            "- Tính toán dòng ngắn mạch Isc và áp hở mạch Voc. " + ("rất chi tiết " * 130),
        ]
    )
    uc = DraftExpertBriefUseCase(gen)
    draft = await uc.execute("Điện mặt trời", "Hệ thống điện mặt trời áp mái inverter")

    assert len(gen.requests) == 1
    req = gen.requests[0]
    assert req.max_output_tokens == 900
    assert req.timeout_s == 30.0
    assert req.messages[0].content == BRIEF_DRAFT_SYSTEM_VI
    assert "Tên lĩnh vực: Điện mặt trời" in req.messages[1].content
    assert "<say>" not in draft
    assert "</say>" not in draft
    assert "<g:" not in draft
    assert "\n\n\n" not in draft
    assert len(draft) <= 1500
    assert draft.startswith("Cách làm việc chuyên lĩnh vực Điện mặt trời:")


def test_high_stakes_pack_prompt_injection() -> None:
    law_pack = get_pack("law")
    assert law_pack is not None
    owner_ctx = OwnerContext(
        domain=DomainBrief(
            domain_id=uuid.uuid4(),
            name=law_pack.name,
            brief=law_pack.brief,
            level=2,
            safety_tier=law_pack.safety_tier.value,
        ),
        conventions=(),
        cases=(),
        snippets=(),
        domain_score=0.8,
        domain_explicit=True,
    )
    req = build_request(
        "Rà soát giúp mình điều khoản bồi thường thiệt hại trong hợp đồng.",
        context=TurnContext(owner=owner_ctx, mode=TurnMode.EXPERT),
        mode=TurnMode.EXPERT,
    )
    system_prompt = req.messages[0].content
    assert law_pack.brief in system_prompt
    assert (
        "Lĩnh vực này có quyết định hệ trọng: nêu căn cứ (điều luật, chuẩn mực, nghiên cứu)"
        in system_prompt
    )
    assert "nhắc bạn ấy kiểm tra với chuyên gia có thẩm quyền trước khi hành động." in system_prompt


@pytest.fixture
def issuer() -> LocalIssuer:
    return LocalIssuer()


@pytest.fixture
def client(database_url: str, issuer: LocalIssuer) -> Iterator[TestClient]:
    env = {**os.environ, "XUNA_DATABASE_URL": database_url}
    subprocess.run(  # noqa: S603 - fixed argv, test-only
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=ROOT,
        env=env,
        check=True,
    )
    with psycopg.connect(database_url.replace("+psycopg", "")) as connection:
        connection.execute(f"TRUNCATE {TABLES} CASCADE")
    gen = _ScriptedBriefGenerator(
        [
            "Cách làm việc chuyên lĩnh vực Pháp lý:\n"
            "- Rà soát kỹ quyền và nghĩa vụ của từng bên.\n"
            "- Nêu rõ rủi ro pháp lý và điều kiện chấm dứt hợp đồng."
        ]
    )
    app = create_app(
        Settings(environment=Environment.TEST, database_url=database_url),
        verifier=issuer.verifier(),
        expert_generator=gen,
    )
    with TestClient(app) as test_client:
        yield test_client


def _auth(issuer: LocalIssuer, subject: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {issuer.token(subject)}"}


def _bind_pet(client: TestClient, headers: dict[str, str], device: SimulatedDevice) -> None:
    challenge = client.post(
        "/v1/devices/challenges",
        json={"schemaVersion": 1, "petId": PET},
        headers={**headers, **idem()},
    ).json()
    message = proof_message(
        base64.b64decode(challenge["nonce"]),
        PetId(PET),
        ProofPurpose.BIND,
        37,
        device.public_key,
    )
    resp = client.post(
        "/v1/pets",
        json={
            "schemaVersion": 1,
            "petId": PET,
            "origin": "migrated_from_v1",
            "snapshotGeneration": 37,
            "deviceProof": {
                "challengeId": challenge["challengeId"],
                "publicKey": b64(device.public_key),
                "signature": b64(device.sign(message)),
            },
        },
        headers={**headers, **idem()},
    )
    assert resp.status_code == 201


@pytest.mark.postgres
def test_knowledge_packs_and_brief_drafts_api(client: TestClient, issuer: LocalIssuer) -> None:
    headers = _auth(issuer, "owner-kx3-1")
    other_headers = _auth(issuer, "owner-kx3-2")
    device = SimulatedDevice()
    _bind_pet(client, headers, device)

    # 1. GET /v1/knowledge-packs returns 7 packs without requiring U3 consent
    packs_resp = client.get("/v1/knowledge-packs", headers=headers)
    assert packs_resp.status_code == 200
    packs = packs_resp.json()
    assert len(packs) == 7
    assert [p["packId"] for p in packs] == [
        "embedded",
        "software",
        "finance",
        "marketing",
        "law",
        "biomed",
        "language",
    ]

    # 2. POST domain with invalid packId -> 422 validation_failed
    bad_pack_resp = client.post(
        f"/v1/pets/{PET}/knowledge-domains",
        json={"name": "Test", "deviceSlot": 0, "packId": "invalid_pack"},
        headers=headers,
    )
    assert bad_pack_resp.status_code == 422
    assert bad_pack_resp.json()["code"] == "validation_failed"

    # 3. POST domain with packId="law" -> 201 with copied brief and high_stakes tier
    create_resp = client.post(
        f"/v1/pets/{PET}/knowledge-domains",
        json={"name": "Pháp lý", "deviceSlot": 0, "packId": "law"},
        headers=headers,
    )
    assert create_resp.status_code == 201
    dom = create_resp.json()
    domain_id = dom["domainId"]
    assert dom["packId"] == "law"
    assert dom["safetyTier"] == "high_stakes"
    assert dom["expertBrief"].startswith("Cách làm việc chuyên lĩnh vực Pháp lý & hợp đồng:")
    assert "Pháp luật quy định hợp đồng" in dom["description"]

    # 4. PATCH set_brief and apply_pack
    brief_resp = client.patch(
        f"/v1/knowledge-domains/{domain_id}",
        json={"action": "set_brief", "expertBrief": "Chỉ dẫn riêng của chủ."},
        headers=headers,
    )
    assert brief_resp.status_code == 200
    assert brief_resp.json()["expertBrief"] == "Chỉ dẫn riêng của chủ."

    apply_resp = client.patch(
        f"/v1/knowledge-domains/{domain_id}",
        json={"action": "apply_pack", "packId": "embedded"},
        headers=headers,
    )
    assert apply_resp.status_code == 200
    assert apply_resp.json()["packId"] == "embedded"
    assert apply_resp.json()["safetyTier"] == "general"
    assert "Nhúng & phần cứng" in apply_resp.json()["expertBrief"]

    # 5. POST /v1/knowledge-domains/{domain_id}/brief-drafts without U3 -> 403 consent_required
    no_consent = client.post(
        f"/v1/knowledge-domains/{domain_id}/brief-drafts",
        json={"schemaVersion": 1},
        headers=headers,
    )
    assert no_consent.status_code == 403
    assert no_consent.json()["code"] == "consent_required"

    # 6. Grant U3 consent
    consent_resp = client.put(
        "/v1/consents",
        json={
            "schemaVersion": 1,
            "aggregateMetrics": True,
            "experienceEvents": True,
            "conversationText": True,
            "expectedVersion": 0,
        },
        headers={**headers, **idem()},
    )
    assert consent_resp.status_code == 200

    # 7. Other owner cannot draft or get domain -> 404
    other_consent = client.put(
        "/v1/consents",
        json={
            "schemaVersion": 1,
            "aggregateMetrics": True,
            "experienceEvents": True,
            "conversationText": True,
            "expectedVersion": 0,
        },
        headers={**other_headers, **idem()},
    )
    assert other_consent.status_code == 200
    other_draft = client.post(
        f"/v1/knowledge-domains/{domain_id}/brief-drafts",
        json={"schemaVersion": 1},
        headers=other_headers,
    )
    assert other_draft.status_code == 404

    # 8. Owner drafts brief -> 200 and does NOT overwrite DB brief automatically
    draft_resp = client.post(
        f"/v1/knowledge-domains/{domain_id}/brief-drafts",
        json={"schemaVersion": 1},
        headers=headers,
    )
    assert draft_resp.status_code == 200
    assert "Cách làm việc chuyên lĩnh vực Pháp lý:" in draft_resp.json()["draft"]

    get_resp = client.get(f"/v1/knowledge-domains/{domain_id}", headers=headers)
    assert get_resp.status_code == 200
    assert "Nhúng & phần cứng" in get_resp.json()["expertBrief"]
