"""Phone voice V1 over HTTP: U3V consent, transcription, spoken replies."""

import base64
import json
from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from tests.helpers import LocalIssuer, idem
from tests.modules.conversation.test_conversation_api import (
    PET,
    RecordingGenerator,
    _reset,
    bind,
    headers,
)
from xuna_backend.core.config import Environment, Settings
from xuna_backend.main import create_app
from xuna_backend.modules.cognition.application.ports import ProviderUnavailableError
from xuna_backend.modules.cognition.domain.types import AudioChunk, Locale, Transcript

pytestmark = pytest.mark.postgres
OGG = b"OggS" + b"\0" * 60


class FakeRecognizer:
    def __init__(self, text: str = "chào XUNA", fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls = 0

    async def transcribe(
        self, audio: AsyncIterator[AudioChunk], locale: Locale
    ) -> AsyncIterator[Transcript]:
        self.calls += 1
        _ = [chunk async for chunk in audio]
        if self.fail:
            raise ProviderUnavailableError("down")
        yield Transcript(self.text, final=True)


class FakeSynthesizer:
    def __init__(self, fail_from: int = 99, pcm: bool = False) -> None:
        self.texts: list[str] = []
        self.fail_from = fail_from
        self.pcm = pcm

    async def synthesize(self, text: str, locale: Locale) -> AsyncIterator[AudioChunk]:
        self.texts.append(text)
        if len(self.texts) >= self.fail_from:
            raise ProviderUnavailableError("tts down")
        if self.pcm:
            yield AudioChunk(b"\0\0" * 8, "pcm16", 22050)
        else:
            yield AudioChunk(b"ID3" + text.encode(), "mp3", 24000)


def consents(client: TestClient, h: dict[str, str], *, text: bool, voice: bool) -> None:
    body = {
        "schemaVersion": 1,
        "aggregateMetrics": False,
        "experienceEvents": False,
        "conversationText": text,
        "conversationVoice": voice,
        "expectedVersion": client.get("/v1/consents", headers=h).json()["version"],
    }
    assert client.put("/v1/consents", json=body, headers={**h, **idem()}).status_code == 200


def make(
    url: str, issuer: LocalIssuer, recognizer: object | None, synthesizer: object | None
) -> TestClient:
    _reset(url)
    app = create_app(
        Settings(environment=Environment.TEST, database_url=url),
        verifier=issuer.verifier(),
        generator=RecordingGenerator(),
        recognizer=recognizer,  # type: ignore[arg-type]
        synthesizer=synthesizer,  # type: ignore[arg-type]
    )
    return TestClient(app)


@pytest.fixture
def issuer() -> LocalIssuer:
    return LocalIssuer()


def speech(
    client: TestClient, h: dict[str, str], audio: bytes = OGG, fmt: str = "ogg_opus"
) -> object:
    body = {"schemaVersion": 1, "audio": base64.b64encode(audio).decode(), "format": fmt}
    return client.post("/v1/speech/transcriptions", json=body, headers=h)


def test_transcription_needs_voice_consent_not_text_consent(
    database_url: str, issuer: LocalIssuer
) -> None:
    recognizer = FakeRecognizer()
    with make(database_url, issuer, recognizer, None) as client:
        h = headers(issuer)
        consents(client, h, text=True, voice=False)
        denied = speech(client, h)
        assert (denied.status_code, denied.json()["code"]) == (403, "consent_required")  # type: ignore[attr-defined]
        assert recognizer.calls == 0
        consents(client, h, text=False, voice=True)
        ok = speech(client, h)
        assert ok.status_code == 200  # type: ignore[attr-defined]
        assert ok.json()["text"] == "chào XUNA"  # type: ignore[attr-defined]
        assert isinstance(ok.json()["latencyMs"], int)  # type: ignore[attr-defined]


def test_transcription_errors(database_url: str, issuer: LocalIssuer) -> None:
    with make(database_url, issuer, FakeRecognizer(fail=True), None) as client:
        h = headers(issuer)
        consents(client, h, text=False, voice=True)
        down = speech(client, h)
        assert (down.status_code, down.json()["code"]) == (503, "provider_unavailable")  # type: ignore[attr-defined]
        wrong = speech(client, h, fmt="wav")
        assert (wrong.status_code, wrong.json()["code"]) == (422, "validation_failed")  # type: ignore[attr-defined]
        not_b64 = client.post(
            "/v1/speech/transcriptions",
            json={"schemaVersion": 1, "audio": "@@@@", "format": "ogg_opus"},
            headers=h,
        )
        assert not_b64.status_code == 422
    with make(database_url, issuer, None, None) as client:
        h = headers(issuer)
        consents(client, h, text=False, voice=True)
        off = speech(client, h)
        assert (off.status_code, off.json()["code"]) == (503, "voice_unavailable")  # type: ignore[attr-defined]


def _events(text: str) -> list[tuple[str, dict[str, object]]]:
    out = []
    for block in text.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        out.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def spoken_turn(
    client: TestClient, h: dict[str, str], text: str
) -> list[tuple[str, dict[str, object]]]:
    response = client.post(
        f"/v1/conversations/{PET}/turns",
        json={"schemaVersion": 1, "text": text, "speak": True},
        headers=h,
    )
    assert response.status_code == 200, response.text
    return _events(response.text)


def test_spoken_reply_streams_audio_per_sentence_in_order(
    database_url: str, issuer: LocalIssuer
) -> None:
    synthesizer = FakeSynthesizer()
    with make(database_url, issuer, None, synthesizer) as client:
        h = headers(issuer)
        bind(client, h)
        consents(client, h, text=True, voice=False)
        events = spoken_turn(client, h, "Chào XUNA. Hôm nay vui quá! Còn gì")
    audio = [data for name, data in events if name == "audio"]
    assert [a["seq"] for a in audio] == [1, 2, 3]
    assert [a["text"] for a in audio] == ["Chào XUNA.", "Hôm nay vui quá!", "Còn gì"]
    assert synthesizer.texts[0] == "Chào Xu Na."  # pronunciation fix, not shown text
    assert base64.b64decode(str(audio[0]["data"])).startswith(b"ID3")
    assert audio[0]["format"] == "mp3"
    names = [name for name, _ in events]
    assert names.index("audio") > names.index("delta")
    assert events[-1][0] == "finished"
    assert events[-1][1]["spoken"] is True


def test_tts_failure_reports_once_and_text_continues(
    database_url: str, issuer: LocalIssuer
) -> None:
    with make(database_url, issuer, None, FakeSynthesizer(fail_from=2, pcm=True)) as client:
        h = headers(issuer)
        bind(client, h)
        consents(client, h, text=True, voice=False)
        events = spoken_turn(client, h, "Một. Hai. Ba.")
        names = [name for name, _ in events]
        assert names.count("audio") == 1
        assert names.count("speech_unavailable") == 1
        first = next(d for n, d in events if n == "audio")
        assert first["format"] == "wav"
        assert base64.b64decode(str(first["data"]))[:4] == b"RIFF"
        assert events[-1][1]["spoken"] is False
        assert names.count("delta") == 3
    with make(database_url, issuer, None, None) as client:
        h = headers(issuer)
        bind(client, h)
        consents(client, h, text=True, voice=False)
        events = spoken_turn(client, h, "Một. Hai.")
        assert [n for n, _ in events].count("speech_unavailable") == 1
        assert events[-1][1]["spoken"] is False


def test_text_turn_without_speak_is_unchanged(database_url: str, issuer: LocalIssuer) -> None:
    with make(database_url, issuer, None, FakeSynthesizer()) as client:
        h = headers(issuer)
        bind(client, h)
        consents(client, h, text=True, voice=False)
        response = client.post(
            f"/v1/conversations/{PET}/turns",
            json={"schemaVersion": 1, "text": "Một. Hai."},
            headers=h,
        )
        names = [n for n, _ in _events(response.text)]
        assert "audio" not in names
        assert "spoken" not in _events(response.text)[-1][1]
