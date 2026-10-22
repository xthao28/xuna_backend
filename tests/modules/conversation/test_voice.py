"""Phone voice V1 application rules: audio checks, sentence cutting, WAV framing."""

import asyncio
import io
import wave
from collections.abc import AsyncIterator

import pytest

from xuna_backend.modules.cognition.domain.types import AudioChunk, Locale, Transcript
from xuna_backend.modules.conversation.application.voice import (
    MAX_AUDIO_BYTES,
    AudioFormat,
    SentenceSplitter,
    check_audio,
    speakable,
    transcribe_once,
    wav_container,
)

OGG = b"OggS" + b"\0" * 60
M4A = b"\0\0\0\x20ftypM4A " + b"\0" * 40
WAV = wav_container(b"\0\0" * 160, 16000)


@pytest.mark.parametrize(
    ("data", "audio_format"),
    [(OGG, AudioFormat.OGG_OPUS), (M4A, AudioFormat.M4A_AAC), (WAV, AudioFormat.WAV)],
)
def test_accepts_matching_containers(data: bytes, audio_format: AudioFormat) -> None:
    check_audio(data, audio_format)


@pytest.mark.parametrize(
    ("data", "audio_format"),
    [
        (b"", AudioFormat.OGG_OPUS),
        (OGG, AudioFormat.WAV),
        (WAV, AudioFormat.M4A_AAC),
        (b"OggS" + b"\0" * MAX_AUDIO_BYTES, AudioFormat.OGG_OPUS),
    ],
)
def test_rejects_empty_oversized_or_mislabelled_audio(
    data: bytes, audio_format: AudioFormat
) -> None:
    with pytest.raises(ValueError):
        check_audio(data, audio_format)


class FakeRecognizer:
    def __init__(self, *texts: str) -> None:
        self.texts = texts
        self.received: list[AudioChunk] = []

    async def transcribe(
        self, audio: AsyncIterator[AudioChunk], locale: Locale
    ) -> AsyncIterator[Transcript]:
        self.received += [chunk async for chunk in audio]
        for index, text in enumerate(self.texts):
            yield Transcript(text, final=index == len(self.texts) - 1)


def test_transcribe_once_keeps_the_final_text_and_labels_the_codec() -> None:
    recognizer = FakeRecognizer("chào", " chào XUNA ")
    text = asyncio.run(transcribe_once(recognizer, OGG, AudioFormat.OGG_OPUS, Locale.VI))
    assert text == "chào XUNA"
    assert recognizer.received[0].codec == "ogg_opus"
    assert (
        asyncio.run(transcribe_once(FakeRecognizer(), OGG, AudioFormat.OGG_OPUS, Locale.VI)) == ""
    )
    with pytest.raises(ValueError):
        asyncio.run(transcribe_once(FakeRecognizer("x"), b"nope", AudioFormat.WAV, Locale.VI))


def test_sentences_are_cut_as_the_reply_streams() -> None:
    splitter = SentenceSplitter()
    assert splitter.feed("Chào bạn") == []
    assert splitter.feed("! Hôm nay giá là 3.") == ["Chào bạn!"]
    assert splitter.feed("5 nghìn. Vui") == ["Hôm nay giá là 3.5 nghìn."]  # decimal kept
    assert splitter.feed(" quá…\nThật") == ["Vui quá…"]
    assert splitter.flush() == ["Thật"]
    assert splitter.flush() == []


def test_a_sentence_without_punctuation_is_cut_at_a_space_when_too_long() -> None:
    splitter = SentenceSplitter()
    parts = splitter.feed("chữ " * 100)
    assert parts and all(len(p) <= 240 for p in parts)


def test_pronunciation_and_wav_framing() -> None:
    assert speakable("Chào XUNA, Xuna ơi") == "Chào Xu Na, Xu Na ơi"
    with wave.open(io.BytesIO(WAV)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (
            1,
            2,
            16000,
            160,
        )
    with pytest.raises(ValueError):
        wav_container(b"\0", 16000)
