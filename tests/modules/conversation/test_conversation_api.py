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
    assert '"signals": []' in text  # PE-4c: plain greeting, no tag
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
        "pet": {
            "activeSeconds": 7200,
            "learnedEvents": 3,
            "quietChoice": "idle",
            "personality": "affectionate",
            "mood": "glad",
        },
    }
    assert client.post(url, json=body, headers=h).status_code == 200
    sent = generator.requests[-1]
    assert "thứ Sáu, ngày 23/10/2026" in sent.messages[0].content
    assert "khi yên tĩnh bạn thích nằm nghỉ" in sent.messages[0].content
    assert "là quấn chủ" in sent.messages[0].content
    assert "Tâm trạng của bạn lúc này là vui" in sent.messages[0].content
    assert [m.role for m in sent.messages] == [Role.SYSTEM, Role.USER, Role.ASSISTANT, Role.USER]
    for bad in (
        {"clientTime": "2026-10-23T21:05:00"},
        {"history": [{"role": "owner", "text": "a"}] * 13},
        {"pet": {"current": "dancing"}},
        {"pet": {"name": "Bob"}},
        {"pet": {"personality": "grumpy"}},
        {"pet": {"mood": "angry"}},
    ):
        assert client.post(url, json={**body, **bad}, headers=h).status_code == 422


def test_finished_carries_the_conversation_tags(
    setup: tuple[TestClient, RecordingGenerator], issuer: LocalIssuer
) -> None:
    # PE-4c: closed tags from the owner's words travel with `finished` only.
    client, _ = setup
    h = headers(issuer)
    bind(client, h)
    opt_in_u3(client, h)
    response = client.post(
        f"/v1/conversations/{PET}/turns",
        json={"schemaVersion": 1, "text": "XUNA giỏi quá, thương ghê"},
        headers=h,
    )
    assert '"signals": ["praise", "affection"]' in response.text


def test_conversation_extracts_candidate_and_deduplicates(
    setup: tuple[TestClient, RecordingGenerator], issuer: LocalIssuer
) -> None:
    client, _ = setup
    h = headers(issuer)
    bind(client, h)
    opt_in_u3(client, h)

    # 1. Stated preference creates candidate in awaiting_owner_confirmation
    res = client.post(
        f"/v1/conversations/{PET}/turns",
        json={"schemaVersion": 1, "text": "mình thích XUNA nhảy nhót khi chơi đùa"},
        headers=h,
    )
    assert res.status_code == 200
    assert '"candidate"' in res.text
    assert '"type": "stated_preference"' in res.text
    assert "nhảy nhót vui vẻ" in res.text

    # Verify candidate is in database via memory API
    candidates_res = client.get(f"/v1/memory-candidates?petId={PET}", headers=h)
    assert candidates_res.status_code == 200
    candidates = candidates_res.json()["items"]
    assert len(candidates) == 1
    assert candidates[0]["candidateType"] == "stated_preference"
    assert candidates[0]["status"] == "awaiting_owner_confirmation"
    assert candidates[0]["source"] == "conversation"

    # 2. Duplicate turn does not produce another candidate
    res2 = client.post(
        f"/v1/conversations/{PET}/turns",
        json={"schemaVersion": 1, "text": "mình thích XUNA nhảy nhót khi chơi đùa"},
        headers=h,
    )
    assert res2.status_code == 200
    assert '"candidate"' not in res2.text  # already exists, skipped

    candidates_res2 = client.get(f"/v1/memory-candidates?petId={PET}", headers=h)
    assert len(candidates_res2.json()["items"]) == 1

    # 3. Personal fact creates candidate
    res3 = client.post(
        f"/v1/conversations/{PET}/turns",
        json={"schemaVersion": 1, "text": "tên mình là Thảo"},
        headers=h,
    )
    assert res3.status_code == 200
    assert '"candidate"' in res3.text
    assert '"type": "fact"' in res3.text

    candidates_res3 = client.get(f"/v1/memory-candidates?petId={PET}", headers=h)
    assert len(candidates_res3.json()["items"]) == 2


def test_conversation_injects_confirmed_facts_into_prompt(
    setup: tuple[TestClient, RecordingGenerator], issuer: LocalIssuer
) -> None:
    # Layer B: confirmed facts recall into system prompt (CLD-14, D17)
    client, generator = setup
    h = headers(issuer)
    bind(client, h)
    opt_in_u3(client, h)

    # 1. Turn creates a fact candidate
    res1 = client.post(
        f"/v1/conversations/{PET}/turns",
        json={"schemaVersion": 1, "text": "tên mình là Thảo"},
        headers=h,
    )
    assert res1.status_code == 200
    # Before confirmation, the first turn does NOT have confirmed facts injected
    assert "Tên của bạn ấy là Thảo" not in generator.requests[-1].messages[0].content

    # 2. Get the candidate and confirm + commit it
    candidates = client.get(f"/v1/memory-candidates?petId={PET}", headers=h).json()["items"]
    fact_candidate = next(c for c in candidates if c["candidateType"] == "fact")
    cand_id = fact_candidate["candidateId"]

    patch_res = client.patch(
        f"/v1/memory-candidates/{cand_id}",
        json={"schemaVersion": 1, "action": "confirm"},
        headers={**h, **idem()},
    )
    assert patch_res.status_code == 200

    commit_res = client.patch(
        f"/v1/memory-candidates/{cand_id}",
        json={"schemaVersion": 1, "action": "commit", "committedGeneration": 1},
        headers={**h, **idem()},
    )
    assert commit_res.status_code == 200

    # Also add a favorite color fact directly and confirm + commit
    post_fact = client.post(
        "/v1/memory-candidates",
        json={
            "schemaVersion": 1,
            "petId": PET,
            "source": "owner_input",
            "candidateType": "fact",
            "content": {
                "fact": "favorite_color",
                "value": "xanh ngọc",
                "summary": "Màu yêu thích của bạn là xanh ngọc",
            },
        },
        headers={**h, **idem()},
    )
    assert post_fact.status_code == 201
    color_cand_id = post_fact.json()["candidateId"]
    client.patch(
        f"/v1/memory-candidates/{color_cand_id}",
        json={"schemaVersion": 1, "action": "confirm"},
        headers={**h, **idem()},
    )
    client.patch(
        f"/v1/memory-candidates/{color_cand_id}",
        json={"schemaVersion": 1, "action": "commit", "committedGeneration": 2},
        headers={**h, **idem()},
    )

    # 3. Next turn: generator should receive the confirmed facts in the system prompt!
    res2 = client.post(
        f"/v1/conversations/{PET}/turns",
        json={"schemaVersion": 1, "text": "bạn có nhớ mình là ai không?"},
        headers=h,
    )
    assert res2.status_code == 200
    system_prompt = generator.requests[-1].messages[0].content
    assert "Những điều bạn biết chắc chắn về người bạn đang trò chuyện cùng" in system_prompt
    assert "Tên của bạn ấy là Thảo" in system_prompt
    assert "Màu yêu thích của bạn ấy là xanh ngọc" in system_prompt
    assert "Người bạn đang nói chuyện cùng tên là Thảo" in system_prompt


def test_reply_gesture_is_sent_once_and_never_as_text(
    setup: tuple[TestClient, RecordingGenerator], issuer: LocalIssuer
) -> None:
    # Owner 2026-10-27: an expression cue from a closed set, only when the app says
    # the device can play it. The echo generator repeats the tag the owner typed.
    client, generator = setup
    h = headers(issuer)
    bind(client, h)
    opt_in_u3(client, h)
    response = client.post(
        f"/v1/conversations/{PET}/turns",
        json={
            "schemaVersion": 1,
            "text": "<g:greet_nuzzle> chào <g:look_around>",
            "gestures": True,
        },
        headers=h,
    )
    text = response.text
    assert text.count("event: gesture") == 1
    assert 'event: gesture\ndata: {"name": "greet_nuzzle"}' in text
    deltas = [block for block in text.split("\n\n") if block.startswith("event: delta")]
    assert deltas and not any("<g:" in block for block in deltas)
    assert "<g:bounce_play>" in generator.requests[0].messages[0].content
    plain = turn(client, h)
    assert "event: gesture" not in plain.text  # type: ignore[attr-defined]
    assert "<g:" not in generator.requests[1].messages[0].content


def test_legacy_client_has_no_mode_event_and_finished_reports_chat(
    setup: tuple[TestClient, RecordingGenerator], issuer: LocalIssuer
) -> None:
    import json

    client, _ = setup
    h = headers(issuer)
    bind(client, h)
    opt_in_u3(client, h)
    res = turn(client, h, "chào XUNA")
    text = res.text  # type: ignore[attr-defined]
    assert "event: mode\n" not in text
    assert "event: detail\n" not in text
    finished_blocks = [b for b in text.strip().split("\n\n") if b.startswith("event: finished\n")]
    assert len(finished_blocks) == 1
    finished_data = json.loads(finished_blocks[0].split("\n", 1)[1].removeprefix("data: "))
    assert finished_data["mode"] == "chat"
    assert finished_data["domainId"] is None
    assert finished_data["hasDetail"] is False


def test_expert_mode_dual_output_spoken_say_only_and_length_limits(
    database_url: str, issuer: LocalIssuer
) -> None:
    import json

    from tests.modules.conversation.test_voice_api import FakeSynthesizer
    from xuna_backend.modules.cognition.domain.types import ChatFinished, TextDelta

    class ScriptedExpertGenerator(RecordingGenerator):
        async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
            self.requests.append(request)
            yield TextDelta("<say>Ý chính.</say>## Chi tiết\n- Mục 1\n- Mục 2")
            yield ChatFinished("fake", "expert-m", 10, 20, 5)

    _reset(database_url)
    chat_gen = RecordingGenerator()
    expert_gen = ScriptedExpertGenerator()
    synth = FakeSynthesizer()
    app = create_app(
        Settings(environment=Environment.TEST, database_url=database_url),
        verifier=issuer.verifier(),
        generator=chat_gen,
        expert_generator=expert_gen,
        synthesizer=synth,  # type: ignore[arg-type]
    )
    with TestClient(app) as client:
        h = headers(issuer)
        bind(client, h)
        opt_in_u3(client, h)
        url = f"/v1/conversations/{PET}/turns"

        # 600 chars with mode="chat" -> 422
        r_chat_long = client.post(
            url,
            json={"schemaVersion": 1, "text": "a" * 600, "mode": "chat"},
            headers=h,
        )
        assert r_chat_long.status_code == 422

        # 600 chars with mode="expert" and speak=True -> 200, uses expert_gen, TTS speaks only say
        r_expert = client.post(
            url,
            json={"schemaVersion": 1, "text": "a" * 600, "mode": "expert", "speak": True},
            headers=h,
        )
        assert r_expert.status_code == 200
        assert len(chat_gen.requests) == 0
        assert len(expert_gen.requests) == 1
        assert expert_gen.requests[0].max_output_tokens == 3000

        events = []
        for block in r_expert.text.strip().split("\n\n"):
            ev_line, data_line = block.split("\n", 1)
            events.append(
                (ev_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: ")))
            )

        names = [n for n, _ in events]
        assert names[0] == "mode"
        assert events[0][1] == {
            "mode": "expert",
            "domainId": None,
            "reason": "requested_expert",
        }
        assert "delta" in names
        assert "detail" in names
        assert "audio" in names
        # Every detail must arrive BEFORE finished
        assert max(i for i, (n, _) in enumerate(events) if n == "detail") < names.index("finished")
        delta_text = "".join(d["text"] for n, d in events if n == "delta")
        detail_text = "".join(d["text"] for n, d in events if n == "detail")
        assert delta_text == "Ý chính."
        assert detail_text == "## Chi tiết\n- Mục 1\n- Mục 2"
        # Audio only speaks say, never detail
        assert synth.texts == ["Ý chính."]
        assert events[-1][0] == "finished"
        assert events[-1][1]["mode"] == "expert"
        assert events[-1][1]["hasDetail"] is True


def test_foreign_or_unknown_domain_id_is_ignored_without_error(
    setup: tuple[TestClient, RecordingGenerator], issuer: LocalIssuer
) -> None:
    import json
    import uuid

    client, _ = setup
    h = headers(issuer)
    bind(client, h)
    opt_in_u3(client, h)

    res = client.post(
        f"/v1/conversations/{PET}/turns",
        json={
            "schemaVersion": 1,
            "text": "Giải thích nguyên lý hoạt động?",
            "mode": "auto",
            "domainId": str(uuid.uuid4()),
            "knowledge": [{"slot": 0, "level": 2, "state": "active"}],
        },
        headers=h,
    )
    assert res.status_code == 200
    first_block = res.text.strip().split("\n\n")[0]
    mode_data = json.loads(first_block.split("\n", 1)[1].removeprefix("data: "))
    assert mode_data["domainId"] is None
    assert mode_data["mode"] == "chat"
    assert mode_data["reason"] == "casual"
