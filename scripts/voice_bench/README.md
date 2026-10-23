# Voice V0 benchmark tools

Research tools for the [voice parity plan](../../../docs/roadmap/voice/voice-parity-plan-2026-10-22.md)
(Phase V0). Not part of the backend package: own venv, own `requirements.txt`. The
corpus is synthetic (Edge TTS voices); `corpus/` and `results/` are gitignored and
are rebuilt on each machine.

## Set up on a new machine

Needs Python 3.12 and `ffmpeg` on PATH.

```sh
cd backend/scripts/voice_bench
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python make_corpus.py --out corpus     # 120 WAVs: 30 sentences × 2 voices × clean/noise10db
```

Edge TTS is a free online service and fails now and then (403, "No audio received");
`make_corpus.py` retries with backoff and resumes where it stopped.

## Run

Keep runs small on weak machines (`--limit`, `--threads`).

```sh
# Local Whisper (faster-whisper, CPU int8). Add --prompt XUNA so the name is heard.
.venv/bin/python asr_bench.py --models small --threads 4 --limit 10 --prompt XUNA

# Cloud ASR, free tiers. Keys only from the environment; never print or commit them.
set -a; source ../../.env.gemini; set +a        # existing Gemini key (XUNA_TEXT_*)
.venv/bin/python asr_cloud_bench.py --provider gemini --model gemini-3.5-flash-lite --limit 10
# Groq Whisper: put the key in backend/.env.groq (gitignored; template below)
set -a; source ../../.env.groq; set +a
.venv/bin/python asr_cloud_bench.py --provider transcribe --model "$XUNA_ASR_MODEL" --limit 10

# Edge TTS first-audio latency and failure rate
.venv/bin/python tts_bench.py --runs 1

# Piper local TTS fallback (download the voice once into models/, gitignored)
mkdir -p models && for f in vi_VN-vais1000-medium.onnx vi_VN-vais1000-medium.onnx.json; do
  curl -sSfL -o models/$f https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vais1000/medium/$f; done
.venv/bin/python piper_bench.py --limit 6 --check-asr   # --check-asr uses backend/.env.groq
```

`backend/.env.groq` is not in Git, so create it on each machine (mode 600):

```sh
XUNA_ASR_BASE_URL=https://api.groq.com/openai/v1
XUNA_ASR_MODEL=whisper-large-v3-turbo
XUNA_ASR_API_KEY=<key from https://console.groq.com/keys>
```

## Files

| File | Purpose |
| --- | --- |
| `sentences.py` | 30 Vietnamese reference sentences (owner-style speech, numbers, names) |
| `make_corpus.py` | Edge TTS → 16 kHz mono WAV + 10 dB noisy copies, `manifest.json` |
| `asr_bench.py` | Local faster-whisper WER (normalised) and decode time |
| `asr_cloud_bench.py` | Gemini audio input or any OpenAI-compatible `/audio/transcriptions` |
| `tts_bench.py` | Edge TTS first audio / total time / failures (no retries on purpose) |
| `piper_bench.py` | Piper local TTS (vi_VN vais1000 medium) latency; optional ASR round trip as intelligibility proxy |

Free tiers may use inputs to improve their products: use the synthetic corpus only,
never owner audio without consent.
