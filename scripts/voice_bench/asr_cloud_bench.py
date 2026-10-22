"""V0 cloud ASR benchmark (free tiers) on the synthetic Vietnamese corpus.

Providers:
  gemini      Gemini via its OpenAI-compatible chat endpoint with audio input. Uses the
              existing backend/.env.gemini (XUNA_TEXT_BASE_URL, XUNA_TEXT_API_KEY, model).
  transcribe  Any OpenAI-compatible /audio/transcriptions endpoint (e.g. Groq Whisper):
              XUNA_ASR_BASE_URL, XUNA_ASR_API_KEY, XUNA_ASR_MODEL.

Usage (keys only from the environment, never printed):
  set -a; source ../../.env.gemini; set +a
  .venv/bin/python asr_cloud_bench.py --provider gemini --model gemini-3.5-flash-lite --limit 10
Free tiers may use inputs to improve products: synthetic corpus only, never owner audio
without consent. Results stay local (results/ is gitignored).
"""

import argparse
import base64
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import jiwer
import requests
from asr_bench import normalise

PROMPT = (
    "Chép lại chính xác lời nói tiếng Việt trong đoạn âm thanh. Chỉ trả về đúng câu nói, "
    "không giải thích. Tên riêng có thể gặp: XUNA."
)


def gemini(path: Path, model: str, timeout_s: float) -> str:
    base = os.environ["XUNA_TEXT_BASE_URL"].rstrip("/")
    audio = base64.b64encode(path.read_bytes()).decode()
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}},
                ],
            }
        ],
        "temperature": 0,
    }
    effort = os.environ.get("XUNA_ASR_REASONING_EFFORT", "none")
    if effort:
        body["reasoning_effort"] = effort  # thinking adds seconds before the transcript
    r = requests.post(
        f"{base}/chat/completions",
        json=body,
        timeout=timeout_s,
        headers={"Authorization": f"Bearer {os.environ['XUNA_TEXT_API_KEY']}"},
    )
    r.raise_for_status()
    return str(r.json()["choices"][0]["message"]["content"]).strip()


def transcribe(path: Path, model: str, timeout_s: float) -> str:
    base = os.environ["XUNA_ASR_BASE_URL"].rstrip("/")
    with path.open("rb") as f:
        r = requests.post(
            f"{base}/audio/transcriptions",
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {os.environ['XUNA_ASR_API_KEY']}"},
            files={"file": (path.name, f, "audio/wav")},
            data={"model": model, "language": "vi", "prompt": "XUNA", "temperature": "0"},
        )
    r.raise_for_status()
    return str(r.json()["text"]).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["gemini", "transcribe"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", type=Path, default=Path("corpus"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--pause", type=float, default=2.0, help="seconds between calls (rate limits)"
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    call = gemini if args.provider == "gemini" else transcribe
    items = json.loads((args.corpus / "manifest.json").read_text())[: args.limit]
    rows = []
    for item in items:
        t0 = time.perf_counter()
        try:
            hyp, error = call(args.corpus / item["path"], args.model, args.timeout), None
        except Exception as exc:  # status code only, never the response body
            hyp = ""
            code = getattr(getattr(exc, "response", None), "status_code", None)
            error = f"{type(exc).__name__}{'' if code is None else f' {code}'}"
        rows.append(
            {
                "id": item["id"],
                "condition": item["condition"],
                "reference": item["text"],
                "hypothesis": hyp,
                "error": error,
                "latencyMs": round((time.perf_counter() - t0) * 1000),
                "wer": jiwer.wer(normalise(item["text"]), normalise(hyp) or "∅"),
            }
        )
        time.sleep(args.pause)
    ok = [r for r in rows if not r["error"]]
    summary: dict[str, Any] = {
        "provider": args.provider,
        "model": args.model,
        "calls": len(rows),
        "failures": len(rows) - len(ok),
        "errors": sorted({r["error"] for r in rows if r["error"]}),
    }
    for condition in sorted({r["condition"] for r in ok}):
        sel = [r for r in ok if r["condition"] == condition]
        summary[condition] = {
            "n": len(sel),
            "wer": round(
                jiwer.wer(
                    [normalise(r["reference"]) for r in sel],
                    [normalise(r["hypothesis"]) or "∅" for r in sel],
                ),
                4,
            ),
            "latencyMsMedian": statistics.median(r["latencyMs"] for r in sel),
        }
    out = args.out or Path(f"results/asr_{args.provider}_{args.model}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False))  # noqa: T201 - CLI output


if __name__ == "__main__":
    main()
