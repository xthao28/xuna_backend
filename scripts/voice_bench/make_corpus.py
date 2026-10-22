"""Build the synthetic Vietnamese corpus: Edge TTS voices → 16 kHz mono WAV, plus noisy copies.

Usage: .venv/bin/python make_corpus.py --out corpus
Writes <out>/manifest.json with id, voice, condition, path and reference text.
Synthetic speech is a stand-in until the owner records real voices (V0 plan).
"""

import argparse
import asyncio
import json
import subprocess
import wave
from pathlib import Path

import edge_tts
import numpy as np
from sentences import SENTENCES

VOICES = ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"]
SAMPLE_RATE = 16000
NOISE_SNR_DB = 10.0


def to_wav(src: Path, dst: Path) -> None:
    subprocess.run(  # noqa: S603 - fixed argv, dev-only script
        [  # noqa: S607 - ffmpeg from PATH, documented requirement
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-sample_fmt",
            "s16",
            str(dst),
        ],
        check=True,
    )


def add_noise(src: Path, dst: Path, seed: int) -> None:
    """Pink-ish room noise at NOISE_SNR_DB (deterministic per seed)."""
    with wave.open(str(src)) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(len(audio))
    pink = np.convolve(white, np.ones(8) / 8, mode="same")  # crude low-pass
    speech_power = np.mean(audio**2)
    noise_power = np.mean(pink**2)
    pink *= np.sqrt(speech_power / (noise_power * 10 ** (NOISE_SNR_DB / 10)))
    mixed = np.clip(audio + pink, -32768, 32767).astype(np.int16)
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(mixed.tobytes())


async def synthesize(text: str, voice: str, mp3: Path, attempts: int = 8) -> None:
    """Edge TTS is a free online service and fails now and then: retry with backoff."""
    for attempt in range(attempts):
        try:
            await edge_tts.Communicate(text, voice).save(str(mp3))
            return
        except Exception:
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(5.0 * (attempt + 1))


async def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for v, voice in enumerate(VOICES):
        for i, text in enumerate(SENTENCES):
            stem = f"s{i:02d}_{voice.split('-')[2].removesuffix('Neural').lower()}"
            mp3 = out / f"{stem}.mp3"
            clean = out / f"{stem}_clean.wav"
            noisy = out / f"{stem}_noise{int(NOISE_SNR_DB)}.wav"
            if not clean.exists():
                await synthesize(text, voice, mp3)
                await asyncio.sleep(1.0)  # be gentle with the free service
                to_wav(mp3, clean)
                mp3.unlink()
            if not noisy.exists():
                add_noise(clean, noisy, seed=v * 1000 + i)
            for condition, path in (("clean", clean), (f"noise{int(NOISE_SNR_DB)}db", noisy)):
                items.append(
                    {
                        "id": f"{stem}_{condition}",
                        "voice": voice,
                        "condition": condition,
                        "path": path.name,
                        "text": text,
                    }
                )
    (out / "manifest.json").write_text(json.dumps(items, ensure_ascii=False, indent=1))
    print(f"{len(items)} utterances in {out}")  # noqa: T201 - CLI output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("corpus"))
    asyncio.run(main(parser.parse_args().out))
