"""V0 TTS fallback: Piper (local, offline) with the vi_VN vais1000 medium voice.

Usage: .venv/bin/python piper_bench.py --limit 6 [--check-asr]
Downloads (once, gitignored): models/vi_VN-vais1000-medium.onnx(.json) from
huggingface.co/rhasspy/piper-voices (voice dataset VAIS-1000, CC BY 4.0; the
piper-tts runtime is GPL-3.0). Measures time to the first audio chunk and the whole
sentence. --check-asr transcribes the result with the Groq Whisper endpoint
(backend/.env.groq) as an intelligibility proxy: WER of ASR on synthetic speech.
"""

import argparse
import json
import statistics
import time
import wave
from pathlib import Path
from typing import Any

import jiwer
from asr_bench import normalise
from asr_cloud_bench import transcribe
from piper import PiperVoice
from sentences import SENTENCES

MODEL = Path("models/vi_VN-vais1000-medium.onnx")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--check-asr", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/piper.json"))
    args = parser.parse_args()
    t0 = time.perf_counter()
    voice = PiperVoice.load(str(MODEL))
    load_s = time.perf_counter() - t0
    out_dir = args.out.parent / "piper_wav"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for i, text in enumerate(SENTENCES[: args.limit]):
        started = time.perf_counter()
        first: float | None = None
        pcm = b""
        rate = 22050
        for chunk in voice.synthesize(text):
            if first is None:
                first = time.perf_counter() - started
            pcm += chunk.audio_int16_bytes
            rate = chunk.sample_rate
        total = time.perf_counter() - started
        path = out_dir / f"s{i:02d}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm)
        audio_s = len(pcm) / 2 / rate
        row: dict[str, Any] = {
            "text": text,
            "firstAudioMs": round((first or total) * 1000),
            "totalMs": round(total * 1000),
            "audioS": round(audio_s, 2),
        }
        if args.check_asr:
            hyp = transcribe(path, "whisper-large-v3-turbo", 30.0, "opus")
            row["asr"] = hyp
            row["wer"] = jiwer.wer(normalise(text), normalise(hyp) or "∅")
            time.sleep(3.2)
        rows.append(row)
    summary: dict[str, Any] = {
        "voice": MODEL.stem,
        "loadS": round(load_s, 2),
        "n": len(rows),
        "firstAudioMsMedian": statistics.median(r["firstAudioMs"] for r in rows),
        "rtfMedian": round(statistics.median(r["totalMs"] / 1000 / r["audioS"] for r in rows), 3),
    }
    if args.check_asr:
        summary["asrWer"] = round(
            jiwer.wer([normalise(r["text"]) for r in rows], [normalise(r["asr"]) for r in rows]), 4
        )
    args.out.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1)
    )
    print(json.dumps(summary, ensure_ascii=False))  # noqa: T201 - CLI output


if __name__ == "__main__":
    main()
