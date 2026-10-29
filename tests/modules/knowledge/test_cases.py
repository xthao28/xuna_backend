"""Unit and API tests for ProposeKnowledgeCaseUseCase and knowledge-cases (KX-4)."""

from __future__ import annotations

import base64
import os
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from tests.helpers import LocalIssuer, SimulatedDevice, b64, idem
from tests.modules.knowledge.test_use_cases import FakeCandidatePort, FakeKnowledgeRepo
from xuna_backend.core.config import Environment, Settings
from xuna_backend.main import create_app
from xuna_backend.modules.cognition.application.ports import ProviderUnavailableError
from xuna_backend.modules.cognition.domain.types import (
    ChatEvent,
    ChatFinished,
    ChatRequest,
    TextDelta,
)
from xuna_backend.modules.identity.domain.device_proof import ProofPurpose, proof_message
from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.knowledge.application.cases import (
    CASE_SUMMARY_SYSTEM_VI,
    ProposeKnowledgeCaseUseCase,
)
from xuna_backend.modules.knowledge.application.schemas import ProposeKnowledgeCaseRequest
from xuna_backend.modules.knowledge.domain.entities import (
    DomainOrigin,
    DomainStatus,
    KnowledgeDomain,
    compute_fact_key,
)

ROOT = Path(__file__).resolve().parents[3]
PET_STR = "d945c77026fd4f4a8a899bbff4aeed86"
PET = PetId(PET_STR)
OWNER = uuid.uuid4()
NOW = datetime(2026, 10, 29, 10, 0, tzinfo=UTC)


class _ScriptedCaseGenerator:
    def __init__(self, output: str = "", *, fail: bool = False) -> None:
        self.output = output
        self.fail = fail
        self.requests: list[ChatRequest] = []

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        self.requests.append(request)
        if self.fail:
            raise ProviderUnavailableError("provider down")
        if self.output:
            yield TextDelta(self.output)
        yield ChatFinished("fake", "fake-m", 10, 10, 5)


@pytest.mark.anyio
async def test_propose_case_use_case_sanitizes_deduplicates_and_assigns_domain() -> None:
    repo = FakeKnowledgeRepo()
    cands = FakeCandidatePort()
    dom = KnowledgeDomain(
        domain_id=uuid.uuid4(),
        owner_id=OWNER,
        pet_id=PET,
        name="Nhúng ESP32",
        description="Mạch I2C ESP32 treo bus, thêm điện trở kéo lên",
        status=DomainStatus.ACTIVE,
        origin=DomainOrigin.OWNER,
        created_at=NOW,
        updated_at=NOW,
        device_slot=0,
    )
    await repo.save_domain(dom)

    gen = _ScriptedCaseGenerator(
        "<say>Chào!</say> <g:happy> Vấn đề: I2C ESP32 treo bus.   "
        "Cách giải: thêm điện trở kéo lên 4.7kΩ."
    )
    uc = ProposeKnowledgeCaseUseCase(gen, repo, cands)
    req = ProposeKnowledgeCaseRequest(
        schemaVersion=1,
        domainId=None,
        question="Mạch I2C ESP32 của mình bị treo bus khi đọc cảm biến?",
        answer="Bạn hãy kiểm tra và thêm điện trở kéo lên 4.7kΩ trên SDA/SCL.",
    )
    res1 = await uc.execute(PET, OWNER, req, now=NOW)
    expected_text = "Chào! Vấn đề: I2C ESP32 treo bus. Cách giải: thêm điện trở kéo lên 4.7kΩ."
    assert res1.text == expected_text
    assert res1.kind == "case"
    assert res1.fact_key == compute_fact_key("case:" + expected_text)
    assert res1.proposed_domain_id == dom.domain_id
    assert gen.requests[0].messages[0].content == CASE_SUMMARY_SYSTEM_VI

    # Calling again with identical summary returns the existing candidate without creating a 2nd one
    res2 = await uc.execute(PET, OWNER, req, now=NOW)
    assert res2.candidate_id == res1.candidate_id
    assert len(cands.candidates) == 1


@pytest.mark.anyio
async def test_propose_case_rejects_deleted_or_foreign_domain_and_empty_summary() -> None:
    repo = FakeKnowledgeRepo()
    cands = FakeCandidatePort()
    deleted_dom = KnowledgeDomain(
        domain_id=uuid.uuid4(),
        owner_id=OWNER,
        pet_id=PET,
        name="Đã xoá",
        description="",
        status=DomainStatus.DELETED,
        origin=DomainOrigin.OWNER,
        created_at=NOW,
        updated_at=NOW,
        device_slot=None,
    )
    await repo.save_domain(deleted_dom)

    uc = ProposeKnowledgeCaseUseCase(
        _ScriptedCaseGenerator("Vấn đề: A. Cách giải: B."), repo, cands
    )
    with pytest.raises(KeyError):
        await uc.execute(
            PET,
            OWNER,
            ProposeKnowledgeCaseRequest(
                schemaVersion=1,
                domainId=deleted_dom.domain_id,
                question="Câu hỏi",
                answer="Câu trả lời",
            ),
            now=NOW,
        )

    empty_uc = ProposeKnowledgeCaseUseCase(_ScriptedCaseGenerator("<say></say>   "), repo, cands)
    with pytest.raises(ProviderUnavailableError):
        await empty_uc.execute(
            PET,
            OWNER,
            ProposeKnowledgeCaseRequest(
                schemaVersion=1,
                domainId=None,
                question="Câu hỏi",
                answer="Câu trả lời",
            ),
            now=NOW,
        )


def _reset(url: str) -> None:
    env = {**os.environ, "XUNA_DATABASE_URL": url}
    subprocess.run(  # noqa: S603
        [sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, env=env, check=True
    )
    with psycopg.connect(url.replace("+psycopg", "")) as connection:
        for table in ("audit_events", "consent_records"):
            connection.execute(f"ALTER TABLE {table} DISABLE TRIGGER {table}_append_only")
            connection.execute(f"TRUNCATE {table}")
            connection.execute(f"ALTER TABLE {table} ENABLE TRIGGER {table}_append_only")
        connection.execute(
            "TRUNCATE idempotency_keys, device_challenges, devices, brains, pets, owners CASCADE"
        )


@pytest.mark.postgres
def test_knowledge_cases_api_consent_dedup_503_and_never_stores_question_or_answer(
    database_url: str,
) -> None:
    _reset(database_url)
    issuer = LocalIssuer()
    gen = _ScriptedCaseGenerator(
        "Vấn đề: Lỗi brownout ESP32 khi bật WiFi. Cách giải: thêm tụ 100uF sát chân 3V3."
    )
    app = create_app(
        Settings(environment=Environment.TEST, database_url=database_url),
        verifier=issuer.verifier(),
        generator=gen,
        expert_generator=gen,
    )
    with TestClient(app) as client:
        h = {"Authorization": f"Bearer {issuer.token('owner-1')}"}
        other_h = {"Authorization": f"Bearer {issuer.token('owner-2')}"}

        # Bind pet
        device = SimulatedDevice()
        challenge = client.post(
            "/v1/devices/challenges",
            json={"schemaVersion": 1, "petId": PET_STR},
            headers={**h, **idem()},
        ).json()
        msg = proof_message(
            base64.b64decode(challenge["nonce"]),
            PET,
            ProofPurpose.BIND,
            1,
            device.public_key,
        )
        assert (
            client.post(
                "/v1/pets",
                json={
                    "schemaVersion": 1,
                    "petId": PET_STR,
                    "origin": "migrated_from_v1",
                    "snapshotGeneration": 1,
                    "deviceProof": {
                        "challengeId": challenge["challengeId"],
                        "publicKey": b64(device.public_key),
                        "signature": b64(device.sign(msg)),
                    },
                },
                headers={**h, **idem()},
            ).status_code
            == 201
        )

        raw_question = "CHUOI_CAU_HOI_998877_KHONG_DUOC_LUU"
        raw_answer = "CHUOI_CAU_TRA_LOI_112233_KHONG_DUOC_LUU"
        payload = {
            "schemaVersion": 1,
            "domainId": None,
            "question": raw_question,
            "answer": raw_answer,
        }

        # 1. Without U3 consent -> 403
        r_no_consent = client.post(f"/v1/pets/{PET_STR}/knowledge-cases", json=payload, headers=h)
        assert r_no_consent.status_code == 403

        # Grant U3 consent to both owners
        for hdr in (h, other_h):
            assert (
                client.put(
                    "/v1/consents",
                    json={
                        "schemaVersion": 1,
                        "aggregateMetrics": False,
                        "experienceEvents": False,
                        "conversationText": True,
                        "expectedVersion": 0,
                    },
                    headers={**hdr, **idem()},
                ).status_code
                == 200
            )

        # 2. Other owner -> 404
        r_other = client.post(f"/v1/pets/{PET_STR}/knowledge-cases", json=payload, headers=other_h)
        assert r_other.status_code == 404

        # 3. Create case -> 201
        r_ok = client.post(f"/v1/pets/{PET_STR}/knowledge-cases", json=payload, headers=h)
        assert r_ok.status_code == 201
        body1 = r_ok.json()
        assert body1["kind"] == "case"
        assert body1["text"] == (
            "Vấn đề: Lỗi brownout ESP32 khi bật WiFi. Cách giải: thêm tụ 100uF sát chân 3V3."
        )
        assert body1["factKey"] == compute_fact_key("case:" + body1["text"])

        # 4. Duplicate summary -> returns existing candidateId
        r_dup = client.post(f"/v1/pets/{PET_STR}/knowledge-cases", json=payload, headers=h)
        assert r_dup.status_code == 201
        assert r_dup.json()["candidateId"] == body1["candidateId"]

        # 5. Provider failure -> 503
        gen.fail = True
        r_fail = client.post(f"/v1/pets/{PET_STR}/knowledge-cases", json=payload, headers=h)
        assert r_fail.status_code == 503

        # 6. Verify neither raw_question nor raw_answer is stored anywhere in PostgreSQL
        with psycopg.connect(database_url.replace("+psycopg", "")) as connection:
            tables = [
                r[0]
                for r in connection.execute(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
                )
            ]
            for table in tables:
                dump = str(connection.execute(f"SELECT * FROM {table}").fetchall())  # noqa: S608
                assert raw_question not in dump, table
                assert raw_answer not in dump, table
