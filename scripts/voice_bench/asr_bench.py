"""V0 ASR benchmark: local faster-whisper models on the Vietnamese corpus.

Usage: .venv/bin/python asr_bench.py --corpus corpus --models small medium --out results/asr.json
Reports word error rate (normalised: lowercase, punctuation removed) per model and
condition, and wall-clock decode time. Latency on this machine (2014 Intel i7, CPU
int8) is only a relative number; production latency must be measured on the target
server. Transcripts stay local (results/ is gitignored); the corpus is synthetic.
"""

import argparse
import json
import re
import statistics
import time
import unicodedata
import wave
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
from faster_whisper import WhisperModel

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    return " ".join(_PUNCT.sub(" ", text).split())


def load_wav(path: Path) -> "np.ndarray[Any, np.dtype[np.float32]]":
    """16 kHz mono s16 WAV → float32 in [-1, 1] (avoids PyAV, which faster-whisper 1.1.1
    calls with an argument newer PyAV releases removed)."""
    with wave.open(str(path)) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f"{path.name}: expected 16 kHz mono s16 WAV")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def run_model(
    name: str, items: list[dict[str, Any]], corpus: Path, threads: int, prompt: str | None
) -> dict[str, Any]:
    started = time.perf_counter()
    model = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=threads)
    load_s = time.perf_counter() - started
    rows = []
    for item in items:
        t0 = time.perf_counter()
        segments, info = model.transcribe(
            load_wav(corpus / item["path"]),
            language="vi",
            beam_size=1,
            vad_filter=False,
            initial_prompt=prompt,
        )
        hyp = " ".join(s.text for s in segments).strip()
        decode_s = time.perf_counter() - t0
        rows.append(
            {
                "id": item["id"],
                "condition": item["condition"],
                "voice": item["voice"],
                "reference": item["text"],
                "hypothesis": hyp,
                "wer": jiwer.wer(normalise(item["text"]), normalise(hyp) or "∅"),
                "decodeS": round(decode_s, 3),
                "audioS": round(info.duration, 3),
            }
        )
    summary: dict[str, Any] = {"model": name, "prompt": prompt, "loadS": round(load_s, 1)}
    for condition in sorted({r["condition"] for r in rows}):
        sel = [r for r in rows if r["condition"] == condition]
        refs = [normalise(r["reference"]) for r in sel]
        hyps = [normalise(r["hypothesis"]) or "∅" for r in sel]
        summary[condition] = {
            "n": len(sel),
            "wer": round(jiwer.wer(refs, hyps), 4),
            "decodeSMedian": round(statistics.median(r["decodeS"] for r in sel), 3),
            "rtfMedian": round(statistics.median(r["decodeS"] / r["audioS"] for r in sel), 3),
        }
    return {"summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("corpus"))
    parser.add_argument("--models", nargs="+", default=["small"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--prompt", default=None, help="initial prompt, e.g. the pet name XUNA")
    parser.add_argument("--limit", type=int, default=0, help="first N items only (smoke run)")
    parser.add_argument("--out", type=Path, default=Path("results/asr.json"))
    args = parser.parse_args()
    items = json.loads((args.corpus / "manifest.json").read_text())
    if args.limit:
        items = items[: args.limit]
    results = [run_model(m, items, args.corpus, args.threads, args.prompt) for m in args.models]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    for r in results:
        print(json.dumps(r["summary"], ensure_ascii=False))  # noqa: T201 - CLI output


if __name__ == "__main__":
    main()
