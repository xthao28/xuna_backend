"""V0 TTS benchmark: Edge TTS (free online, Vietnamese voices) first-audio latency and reliability.

Usage: .venv/bin/python tts_bench.py --runs 1 --out results/tts.json
For each sentence and voice: time to the first audio chunk (what the listener waits
for when audio is streamed), total time, audio bytes, and whether the request failed.
No retries here on purpose: the failure rate is part of the result.
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

import edge_tts
from sentences import SENTENCES

VOICES = ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"]


async def one(text: str, voice: str, timeout_s: float) -> dict[str, Any]:
    started = time.perf_counter()
    first: float | None = None
    size = 0
    try:
        async with asyncio.timeout(timeout_s):
            async for chunk in edge_tts.Communicate(text, voice).stream():
                if chunk["type"] == "audio":
                    if first is None:
                        first = time.perf_counter() - started
                    size += len(chunk["data"])
        ok = size > 0
        error = None
    except Exception as exc:  # the failure kind is the result
        ok = False
        error = type(exc).__name__
    return {
        "voice": voice,
        "chars": len(text),
        "ok": ok,
        "error": error,
        "firstAudioMs": round(first * 1000) if first is not None else None,
        "totalMs": round((time.perf_counter() - started) * 1000),
        "bytes": size,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--pause", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("results/tts.json"))
    args = parser.parse_args()
    rows = []
    for _ in range(args.runs):
        for voice in VOICES:
            for text in SENTENCES:
                rows.append(await one(text, voice, args.timeout))
                await asyncio.sleep(args.pause)
    ok = [r for r in rows if r["ok"]]
    firsts = sorted(r["firstAudioMs"] for r in ok)
    summary = {
        "provider": "edge-tts",
        "calls": len(rows),
        "failures": len(rows) - len(ok),
        "failureRate": round((len(rows) - len(ok)) / len(rows), 3),
        "firstAudioMsMedian": statistics.median(firsts) if firsts else None,
        "firstAudioMsP95": firsts[int(0.95 * (len(firsts) - 1))] if firsts else None,
        "errors": sorted({r["error"] for r in rows if r["error"]}),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary))  # noqa: T201 - CLI output


if __name__ == "__main__":
    asyncio.run(main())
