"""End-to-end backend acceptance suite for M06 Knowledge Expert (KX-6, plan §10)."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from tests.helpers import LocalIssuer, SimulatedDevice, b64, idem
from tests.modules.conversation.test_conversation_api import RecordingGenerator, _reset
from tests.modules.conversation.test_voice_api import FakeRecognizer, FakeSynthesizer
from xuna_backend.core.config import Environment, Settings
from xuna_backend.main import create_app
from xuna_backend.modules.cognition.domain.types import (
    ChatEvent,
    ChatFinished,
    ChatRequest,
    TextDelta,
)
from xuna_backend.modules.identity.domain.device_proof import ProofPurpose, proof_message
from xuna_backend.modules.identity.domain.pet_id import PetId
from xuna_backend.modules.knowledge.domain.packs import get_pack

PET_STR = "d945c77026fd4f4a8a899bbff4aeed86"
PET = PetId(PET_STR)


class _Kx6ExpertGenerator(RecordingGenerator):
    """Simulates expert LLM responses and case summaries for KX-6 acceptance."""

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        self.requests.append(request)
        sys_prompt = request.messages[0].content
        if "Tóm tắt một việc người dùng và trợ lý đã cùng giải quyết" in sys_prompt:
            yield TextDelta(
                "Vấn đề: Bus I2C bị treo khi nhiễu nguồn. "
                "Cách giải: Thêm trở kéo 4.7kΩ và phát 9 xung SCL để giải phóng SDA."
            )
            yield ChatFinished("fake", "expert-case", 20, 25, 5)
            return

        if "<say>" in sys_prompt:
            yield TextDelta(
                "<say>Mình nghĩ bus I2C bị kẹt mức thấp trên chân SDA do thiếu xung clock; "
                "bạn thử phát 9 xung SCL và kiểm tra trở kéo 4.7kΩ nhé.</say>"
                "## Nguyên nhân gốc\n"
                "- Slave giữ `SDA = LOW` khi giao dịch bị ngắt giữa chừng.\n\n"
                "## Các bước kiểm tra\n"
                "1. Đo áp trên `SDA`/`SCL`.\n"
                "2. Cấu hình tạm `SCL` thành GPIO output và phát **9 xung clock**.\n"
                "```c\n"
                "i2c_master_bus_reset(bus_handle);\n"
                "```"
            )
            yield ChatFinished("fake", "expert-model", 50, 90, 12)
            return

        yield TextDelta("Mình nghe bạn đây, mình có thể giúp gì tiếp theo?")
        yield ChatFinished("fake", "chat-model", 20, 15, 5)


def _bind_and_consent(client: TestClient, h: dict[str, str]) -> None:
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
    r_bind = client.post(
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
    )
    assert r_bind.status_code == 201
    r_consent = client.put(
        "/v1/consents",
        json={
            "schemaVersion": 1,
            "aggregateMetrics": False,
            "experienceEvents": False,
            "conversationText": True,
            "conversationVoice": True,
            "ownerDocuments": True,
            "expectedVersion": 0,
        },
        headers={**h, **idem()},
    )
    assert r_consent.status_code == 200


def _parse_sse(raw: str) -> list[tuple[str, dict[str, object]]]:
    events: list[tuple[str, dict[str, object]]] = []
    for block in raw.strip().split("\n\n"):
        if not block.strip():
            continue
        ev_line, data_line = block.split("\n", 1)
        events.append(
            (ev_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: ")))
        )
    return events


@pytest.mark.postgres
def test_kx6_full_acceptance_matrix(database_url: str) -> None:
    """Covers KX-6 criteria 1, 2, 3, 5, and 6 end-to-end against PostgreSQL."""
    _reset(database_url)
    issuer = LocalIssuer()
    chat_gen = _Kx6ExpertGenerator()
    expert_gen = _Kx6ExpertGenerator()
    synth = FakeSynthesizer(pcm=False)
    app = create_app(
        Settings(environment=Environment.TEST, database_url=database_url),
        verifier=issuer.verifier(),
        generator=chat_gen,
        expert_generator=expert_gen,
        recognizer=FakeRecognizer(),
        synthesizer=synth,  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        h = {"Authorization": f"Bearer {issuer.token('owner-kx6')}"}
        _bind_and_consent(client, h)

        # --- Criterion 1: Two domains (Embedded from pack + custom domain) ---
        r_dom1 = client.post(
            f"/v1/pets/{PET_STR}/knowledge-domains",
            json={
                "name": "Lập trình nhúng",
                "description": "",
                "deviceSlot": 0,
                "packId": "embedded",
            },
            headers={**h, **idem()},
        )
        assert r_dom1.status_code == 201
        dom1 = r_dom1.json()
        dom1_id = dom1["domainId"]
        embedded_pack = get_pack("embedded")
        assert embedded_pack is not None
        assert dom1["packId"] == "embedded"
        assert dom1["expertBrief"] == embedded_pack.brief

        # Activate dom1 on device slot 0
        assert (
            client.patch(
                f"/v1/knowledge-domains/{dom1_id}",
                json={"action": "device_ack", "generation": 2},
                headers={**h, **idem()},
            ).status_code
            == 200
        )

        r_dom2 = client.post(
            f"/v1/pets/{PET_STR}/knowledge-domains",
            json={
                "name": "Nhà thông minh",
                "description": "Cảm biến I2C, Zigbee, Home Assistant",
                "deviceSlot": 1,
            },
            headers={**h, **idem()},
        )
        assert r_dom2.status_code == 201
        dom2_id = r_dom2.json()["domainId"]
        custom_brief = (
            "Bạn là kỹ sư tự động hoá nhà thông minh, ưu tiên độ ổn định cảm biến "
            "và cảnh báo mất kết nối trên Home Assistant."
        )
        assert (
            client.patch(
                f"/v1/knowledge-domains/{dom2_id}",
                json={"action": "device_ack", "generation": 3},
                headers={**h, **idem()},
            ).status_code
            == 200
        )
        assert (
            client.patch(
                f"/v1/knowledge-domains/{dom2_id}",
                json={
                    "action": "set_brief",
                    "expertBrief": custom_brief,
                },
                headers={**h, **idem()},
            ).status_code
            == 200
        )

        question = "Tại sao cảm biến nhiệt độ I2C thỉnh thoảng bị treo bus?"
        slots_lvl0 = [
            {"slot": 0, "level": 0, "state": "active"},
            {"slot": 1, "level": 0, "state": "active"},
        ]

        # Ask in Domain 1 (Embedded pack) at Level 0
        r_turn1 = client.post(
            f"/v1/conversations/{PET_STR}/turns",
            json={
                "schemaVersion": 1,
                "text": question,
                "mode": "auto",
                "domainId": dom1_id,
                "knowledge": slots_lvl0,
            },
            headers=h,
        )
        assert r_turn1.status_code == 200
        ev1 = _parse_sse(r_turn1.text)
        assert ev1[0][0] == "mode"
        assert ev1[0][1] == {
            "mode": "expert",
            "domainId": dom1_id,
            "reason": "domain_task",
        }
        say1 = "".join(str(d["text"]) for n, d in ev1 if n == "delta")
        detail1 = "".join(str(d["text"]) for n, d in ev1 if n == "detail")
        assert 1 <= len(say1) <= 200
        assert "##" not in say1 and "```" not in say1
        assert "## Nguyên nhân gốc" in detail1 and "```c" in detail1
        prompt_dom1_lvl0 = expert_gen.requests[-1].messages[0].content
        assert embedded_pack.brief in prompt_dom1_lvl0
        assert "(độ ăn ý: Mới quen)" in prompt_dom1_lvl0

        # Same question in Domain 2 (Custom domain)
        r_turn2 = client.post(
            f"/v1/conversations/{PET_STR}/turns",
            json={
                "schemaVersion": 1,
                "text": question,
                "mode": "auto",
                "domainId": dom2_id,
                "knowledge": slots_lvl0,
            },
            headers=h,
        )
        assert r_turn2.status_code == 200
        prompt_dom2 = expert_gen.requests[-1].messages[0].content
        assert custom_brief in prompt_dom2

        # --- Criterion 3: Level 0 -> 2 with conventions, solved cases, and U4 documents ---
        # 3a. Teach working convention
        r_teach = client.post(
            f"/v1/pets/{PET_STR}/knowledge",
            json={
                "text": (
                    "Dự án của mình dùng ESP-IDF 5.2, driver i2c_master mới "
                    "và không dùng Arduino."
                ),
                "domainId": dom1_id,
            },
            headers={**h, **idem()},
        )
        assert r_teach.status_code == 201
        conv_cand_id = r_teach.json()["candidateId"]
        assert r_teach.json()["kind"] == "convention"
        assert (
            client.patch(
                f"/v1/memory-candidates/{conv_cand_id}",
                json={"schemaVersion": 1, "action": "confirm"},
                headers={**h, **idem()},
            ).status_code
            == 200
        )
        assert (
            client.patch(
                f"/v1/memory-candidates/{conv_cand_id}",
                json={"schemaVersion": 1, "action": "commit", "committedGeneration": 4},
                headers={**h, **idem()},
            ).status_code
            == 200
        )

        # 3b. Propose and confirm solved case
        r_case = client.post(
            f"/v1/pets/{PET_STR}/knowledge-cases",
            json={
                "schemaVersion": 1,
                "domainId": dom1_id,
                "question": question,
                "answer": f"{say1}\n\n{detail1}",
            },
            headers={**h, **idem()},
        )
        assert r_case.status_code == 201
        case_cand_id = r_case.json()["candidateId"]
        assert r_case.json()["kind"] == "case"
        assert (
            client.patch(
                f"/v1/memory-candidates/{case_cand_id}",
                json={"schemaVersion": 1, "action": "confirm"},
                headers={**h, **idem()},
            ).status_code
            == 200
        )
        assert (
            client.patch(
                f"/v1/memory-candidates/{case_cand_id}",
                json={"schemaVersion": 1, "action": "commit", "committedGeneration": 5},
                headers={**h, **idem()},
            ).status_code
            == 200
        )

        # 3c. Upload and confirm U4 owner reference document
        doc_md = (
            "# Sơ đồ chân Board V2\n\n"
            "Bus I2C cảm biến nhiệt độ nối vào GPIO8 (SDA) và GPIO9 (SCL)."
        )
        r_doc = client.post(
            f"/v1/pets/{PET_STR}/knowledge-documents",
            json={
                "schemaVersion": 1,
                "title": "Sơ đồ chân Board V2.md",
                "domainId": dom1_id,
                "mediaType": "text/markdown",
                "data": base64.b64encode(doc_md.encode("utf-8")).decode("ascii"),
            },
            headers=h,
        )
        assert r_doc.status_code == 201
        doc_cand_id = r_doc.json()["candidateId"]
        assert (
            client.patch(
                f"/v1/memory-candidates/{doc_cand_id}",
                json={"schemaVersion": 1, "action": "confirm"},
                headers={**h, **idem()},
            ).status_code
            == 200
        )

        # Ask at Level 2 (Ăn ý): prompt proactively applies conventions, cases, and snippets
        slots_lvl2 = [
            {"slot": 0, "level": 2, "state": "active"},
            {"slot": 1, "level": 0, "state": "active"},
        ]
        r_turn_lvl2 = client.post(
            f"/v1/conversations/{PET_STR}/turns",
            json={
                "schemaVersion": 1,
                "text": "Viết code khởi tạo bus I2C cảm biến nhiệt độ cho board của mình?",
                "mode": "auto",
                "domainId": dom1_id,
                "knowledge": slots_lvl2,
            },
            headers=h,
        )
        assert r_turn_lvl2.status_code == 200
        prompt_lvl2 = expert_gen.requests[-1].messages[0].content
        assert "Độ ăn ý của bạn với bạn ấy trong lĩnh vực này: Ăn ý." in prompt_lvl2
        assert "ESP-IDF 5.2" in prompt_lvl2
        assert "Vấn đề: Bus I2C bị treo khi nhiễu nguồn." in prompt_lvl2
        assert "[Sơ đồ chân Board V2.md]" in prompt_lvl2
        assert "GPIO8 (SDA)" in prompt_lvl2

        # --- Criterion 2: Voice WebSocket expert turn speaks only say, never detail ---
        synth.texts.clear()
        token = h["Authorization"].removeprefix("Bearer ")
        with client.websocket_connect("/v1/voice") as ws:
            ws.send_json(
                {
                    "type": "hello",
                    "schemaVersion": 1,
                    "auth": {"token": token},
                    "petId": PET_STR,
                    "speak": True,
                }
            )
            assert ws.receive_json()["type"] == "hello.ok"
            ws.send_json(
                {
                    "type": "turn.commit",
                    "text": "Giải thích cách reset bus I2C trên ESP32 khi bị treo?",
                    "mode": "auto",
                    "context": {
                        "clientTime": "2026-10-29T15:00:00+07:00",
                        "domainId": dom1_id,
                        "knowledge": slots_lvl2,
                    },
                }
            )
            ws_events = []
            while True:
                msg = ws.receive()
                if msg.get("text"):
                    parsed = json.loads(msg["text"])
                    ws_events.append(parsed)
                    if parsed["type"] in ("reply.finished", "error"):
                        break
            ws.send_json({"type": "bye"})

        ws_types = [e["type"] for e in ws_events]
        assert ws_types[0] == "turn.mode"
        assert ws_events[0]["mode"] == "expert"
        assert ws_events[0]["domainId"] == dom1_id
        ws_say = "".join(e["text"] for e in ws_events if e["type"] == "reply.delta")
        ws_detail = "".join(e["text"] for e in ws_events if e["type"] == "detail.delta")
        assert 1 <= len(ws_say) <= 200
        assert "## Nguyên nhân gốc" in ws_detail
        # TTS only synthesized the short say sentence(s), never the markdown detail
        assert len(synth.texts) >= 1
        for spoken in synth.texts:
            assert "##" not in spoken
            assert "i2c_master_bus_reset" not in spoken

        # --- Criterion 5: Deleting a solved case removes it from owner context ---
        r_del_case = client.patch(
            f"/v1/memory-candidates/{case_cand_id}",
            json={"schemaVersion": 1, "action": "delete"},
            headers={**h, **idem()},
        )
        assert r_del_case.status_code == 200
        assert r_del_case.json()["status"] == "deleted"

        r_turn_after_del = client.post(
            f"/v1/conversations/{PET_STR}/turns",
            json={
                "schemaVersion": 1,
                "text": "Kiểm tra lại bus I2C cho mình?",
                "mode": "auto",
                "domainId": dom1_id,
                "knowledge": [{"slot": 0, "level": 1, "state": "active"}],
            },
            headers=h,
        )
        assert r_turn_after_del.status_code == 200
        prompt_after_del = expert_gen.requests[-1].messages[0].content
        assert "Vấn đề: Bus I2C bị treo khi nhiễu nguồn." not in prompt_after_del
        assert "Độ ăn ý của bạn với bạn ấy trong lĩnh vực này: Bắt nhịp." in prompt_after_del

        # --- Criterion 6: Legacy app (pre-KX-2, no mode field) still chats seamlessly ---
        chat_gen.requests.clear()
        r_legacy = client.post(
            f"/v1/conversations/{PET_STR}/turns",
            json={
                "schemaVersion": 1,
                "text": "Giải thích chi tiết nguyên lý I2C trên ESP32?",
            },
            headers=h,
        )
        assert r_legacy.status_code == 200
        legacy_events = _parse_sse(r_legacy.text)
        legacy_names = [n for n, _ in legacy_events]
        assert "mode" not in legacy_names
        assert "detail" not in legacy_names
        assert legacy_events[-1][0] == "finished"
        assert legacy_events[-1][1]["mode"] == "chat"
        assert legacy_events[-1][1]["hasDetail"] is False
        assert len(chat_gen.requests) == 1
        assert chat_gen.requests[0].max_output_tokens == 400
