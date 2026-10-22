"""Conversation turn over SSE: U3 consent, pet ownership, no storage, fallback."""

import base64
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from tests.helpers import LocalIssuer, SimulatedDevice, b64, idem
from xuna_backend.core.config import Environment, Settings
from xuna_backend.main import create_app
from xuna_backend.modules.cognition.application.ports import ProviderUnavailableError
from xuna_backend.modules.cognition.domain.types import ChatEvent, ChatRequest, Role
from xuna_backend.modules.cognition.infrastructure.echo import EchoTextGenerator
from xuna_backend.modules.identity.domain.device_proof import ProofPurpose, proof_message
from xuna_backend.modules.identity.domain.pet_id import PetId

pytestmark = pytest.mark.postgres
ROOT = Path(__file__).resolve().parents[3]
PET = "d945c77026fd4f4a8a899bbff4aeed86"
PRIVATE_TEXT = "mã cửa nhà mình là 1234"


class RecordingGenerator(EchoTextGenerator):
    def __init__(self, fail: bool = False) -> None:
        self.requests: list[ChatRequest] = []
        self.fail = fail

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        self.requests.append(request)
        if self.fail:
            raise ProviderUnavailableError("vpn down")
        async for event in super().stream(request):
            yield event


def _reset(url: str) -> None:
    env = {**os.environ, "XUNA_DATABASE_URL": url}
    subprocess.run(  # noqa: S603 - fixed argv, test-only
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


def _client(url: str, issuer: LocalIssuer, generator: RecordingGenerator) -> TestClient:
    app = create_app(
        Settings(environment=Environment.TEST, database_url=url),
        verifier=issuer.verifier(),
        generator=generator,
    )
    return TestClient(app)


@pytest.fixture
def issuer() -> LocalIssuer:
    return LocalIssuer()


@pytest.fixture
def setup(
    database_url: str, issuer: LocalIssuer
) -> Iterator[tuple[TestClient, RecordingGenerator]]:
    _reset(database_url)
    generator = RecordingGenerator()
    with _client(database_url, issuer, generator) as client:
        yield client, generator


def headers(issuer: LocalIssuer, subject: str = "owner-1") -> dict[str, str]:
    return {"Authorization": f"Bearer {issuer.token(subject)}"}


def bind(client: TestClient, h: dict[str, str]) -> None:
    device = SimulatedDevice()
    challenge = client.post(
        "/v1/devices/challenges", json={"schemaVersion": 1, "petId": PET}, headers={**h, **idem()}
    ).json()
    message = proof_message(
        base64.b64decode(challenge["nonce"]), PetId(PET), ProofPurpose.BIND, 1, device.public_key
    )
    body = {
        "schemaVersion": 1,
        "petId": PET,
        "origin": "migrated_from_v1",
        "snapshotGeneration": 1,
        "deviceProof": {
            "challengeId": challenge["challengeId"],
            "publicKey": b64(device.public_key),
            "signature": b64(device.sign(message)),
        },
    }
    assert client.post("/v1/pets", json=body, headers={**h, **idem()}).status_code == 201


def opt_in_u3(client: TestClient, h: dict[str, str]) -> None:
    body = {
        "schemaVersion": 1,
        "aggregateMetrics": False,
        "experienceEvents": False,
        "conversationText": True,
        "expectedVersion": 0,
    }
    assert client.put("/v1/consents", json=body, headers={**h, **idem()}).status_code == 200


def turn(client: TestClient, h: dict[str, str], text: str = "chào XUNA nhé") -> object:
    return client.post(
        f"/v1/conversations/{PET}/turns", json={"schemaVersion": 1, "text": text}, headers=h
    )


def test_requires_u3_consent_then_streams(
    setup: tuple[TestClient, RecordingGenerator], issuer: LocalIssuer
) -> None:
    client, generator = setup
    h = headers(issuer)
    bind(client, h)
    denied = turn(client, h)
    assert denied.status_code == 403  # type: ignore[attr-defined]
    assert generator.requests == []
    opt_in_u3(client, h)
    response = turn(client, h)
    assert response.headers["content-type"].startswith("text/event-stream")  # type: ignore[attr-defined]
    text = response.text  # type: ignore[attr-defined]
    assert 'event: delta\ndata: {"text": "chào"}' in text
    assert "event: finished" in text
    sent = generator.requests[0]
    assert sent.messages[0].role is Role.SYSTEM
    assert "XUNA" in sent.messages[0].content
    assert sent.messages[1].content == "chào XUNA nhé"


def test_only_own_pet_and_nothing_is_stored(
    setup: tuple[TestClient, RecordingGenerator], issuer: LocalIssuer, database_url: str
) -> None:
    client, _ = setup
    owner, other = headers(issuer), headers(issuer, "other")
    bind(client, owner)
    opt_in_u3(client, other)
    assert turn(client, other).status_code == 404  # type: ignore[attr-defined]
    opt_in_u3(client, owner)
    assert turn(client, owner, PRIVATE_TEXT).status_code == 200  # type: ignore[attr-defined]
    with psycopg.connect(database_url.replace("+psycopg", "")) as connection:
        tables = [
            r[0]
            for r in connection.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
        ]
        for table in tables:
            dump = str(connection.execute(f"SELECT * FROM {table}").fetchall())  # noqa: S608
            assert "1234" not in dump, table


def test_provider_outage_becomes_error_event(database_url: str, issuer: LocalIssuer) -> None:
    _reset(database_url)
    with _client(database_url, issuer, RecordingGenerator(fail=True)) as client:
        h = headers(issuer)
        bind(client, h)
        opt_in_u3(client, h)
        response = turn(client, h)
        assert response.status_code == 200  # type: ignore[attr-defined]
        assert 'event: error\ndata: {"code": "provider_unavailable"}' in response.text  # type: ignore[attr-defined]


def test_rejects_oversized_text(
    setup: tuple[TestClient, RecordingGenerator], issuer: LocalIssuer
) -> None:
    client, generator = setup
    h = headers(issuer)
    bind(client, h)
    opt_in_u3(client, h)
    assert turn(client, h, "a" * 501).status_code == 422  # type: ignore[attr-defined]
    assert generator.requests == []


def test_turn_context_reaches_the_prompt_and_bad_context_is_rejected(
    setup: tuple[TestClient, RecordingGenerator], issuer: LocalIssuer
) -> None:
    # Owner 2026-10-23: time, earlier lines and the device's view of XUNA.
    client, generator = setup
    h = headers(issuer)
    bind(client, h)
    opt_in_u3(client, h)
    url = f"/v1/conversations/{PET}/turns"
    body = {
        "schemaVersion": 1,
        "text": "hôm nay thứ mấy?",
        "clientTime": "2026-10-23T21:05:00+07:00",
        "history": [{"role": "owner", "text": "chào"}, {"role": "xuna", "text": "Chào bạn!"}],
        "pet": {"activeSeconds": 7200, "learnedEvents": 3, "quietChoice": "idle"},
    }
    assert client.post(url, json=body, headers=h).status_code == 200
    sent = generator.requests[-1]
    assert "thứ Sáu, ngày 23/10/2026" in sent.messages[0].content
    assert "khi yên tĩnh bạn thích nằm nghỉ" in sent.messages[0].content
    assert [m.role for m in sent.messages] == [Role.SYSTEM, Role.USER, Role.ASSISTANT, Role.USER]
    for bad in (
        {"clientTime": "2026-10-23T21:05:00"},
        {"history": [{"role": "owner", "text": "a"}] * 13},
        {"pet": {"current": "dancing"}},
        {"pet": {"name": "Bob"}},
    ):
        assert client.post(url, json={**body, **bad}, headers=h).status_code == 422
