"""Phone voice V1 (owner D16, 2026-10-22): speech in, spoken reply out.

The app records, the backend transcribes (consent U3V), and the app sends the
transcript through its usual text path, so teaching phrases still need the
owner's confirmation. A spoken reply is the text reply rendered by TTS sentence
by sentence. Audio and transcripts are never stored; transcripts are untrusted.
"""

import struct
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum

from xuna_backend.modules.cognition.application.ports import SpeechRecognizer
from xuna_backend.modules.cognition.domain.types import AudioChunk, Locale

# 15 s of AAC at 96 kb/s is ~180 KB; Opus at 16-32 kb/s is far smaller.
MAX_AUDIO_BYTES = 256 * 1024
MAX_SENTENCE_CHARS = 240


class AudioFormat(StrEnum):
    """Containers a phone records directly; the wire value is also the codec."""

    OGG_OPUS = "ogg_opus"
    M4A_AAC = "m4a_aac"
    WAV = "wav"


def check_audio(data: bytes, audio_format: AudioFormat) -> None:
    """Size and container magic check; raises ValueError. Not a full decode."""
    if not data or len(data) > MAX_AUDIO_BYTES:
        raise ValueError("audio must be 1 byte .. 256 KiB")
    ok = {
        AudioFormat.OGG_OPUS: data[:4] == b"OggS",
        AudioFormat.M4A_AAC: data[4:8] == b"ftyp",
        AudioFormat.WAV: data[:4] == b"RIFF" and data[8:12] == b"WAVE",
    }[audio_format]
    if not ok:
        raise ValueError("audio does not match its format")


async def _one(chunk: AudioChunk) -> AsyncIterator[AudioChunk]:
    yield chunk


async def transcribe_once(
    recognizer: SpeechRecognizer, data: bytes, audio_format: AudioFormat, locale: Locale
) -> str:
    """Final transcript of one push-to-talk utterance ("" = nothing heard).

    Raises ProviderUnavailableError from the adapter on outage or timeout.
    """
    check_audio(data, audio_format)
    final = ""
    async for transcript in recognizer.transcribe(
        _one(AudioChunk(data, audio_format.value, 0)), locale
    ):
        if transcript.final:
            final = transcript.text.strip()
    return final


@dataclass
class SentenceSplitter:
    """Cuts streamed reply text into sentences so speech starts early."""

    _buffer: str = ""

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        sentences: list[str] = []
        start = 0
        for index, char in enumerate(self._buffer):
            # Wait for the following character: "3." may still become "3.5".
            end_of_sentence = char == "\n" or (
                char in ".!?…"
                and index + 1 < len(self._buffer)
                and self._buffer[index + 1].isspace()
            )
            too_long = index - start + 1 >= MAX_SENTENCE_CHARS and char.isspace()
            if end_of_sentence or too_long:
                sentence = self._buffer[start : index + 1].strip()
                if sentence:
                    sentences.append(sentence)
                start = index + 1
        self._buffer = self._buffer[start:]
        return sentences

    def flush(self) -> list[str]:
        rest, self._buffer = self._buffer.strip(), ""
        return [rest] if rest else []


# TTS reads the name letter by letter or as "Xun" (V0); spell how it is said.
_PRONUNCIATION = {"XUNA": "Xu Na", "Xuna": "Xu Na"}


def speakable(sentence: str) -> str:
    for written, spoken in _PRONUNCIATION.items():
        sentence = sentence.replace(written, spoken)
    return sentence


def wav_container(pcm16: bytes, sample_rate: int) -> bytes:
    """Mono 16-bit PCM → WAV, so the app can play each sentence directly."""
    if not 8000 <= sample_rate <= 48000 or len(pcm16) % 2:
        raise ValueError("invalid PCM")
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(pcm16),
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        len(pcm16),
    )
    return header + pcm16
