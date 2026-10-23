# XUNA backend architecture

Status: scaffold (CLD-10), 2026-10-17. Rules: [AGENTS.md](../AGENTS.md).

## Shape

A **modular monolith** (ADR-0009): one FastAPI process, one PostgreSQL database, modules with hard boundaries enforced by `tests/test_architecture.py`. Each module can be extracted into its own service later because it only talks to others through `application`/`domain` and shares IDs, not tables.

```text
 Flutter app ──HTTPS/SSE──▶ api ──▶ application (use cases, ports) ──▶ domain
                                         │ ports implemented by
                                         ▼
                                  infrastructure (PostgreSQL, provider adapters)
```

Async end to end: cloud AI work is dominated by waiting on providers, so routers, use cases, DB sessions and provider ports are `async`, and long outputs stream (`AsyncIterator`).

## Modules today

| Module | Contents | Status |
| --- | --- | --- |
| `health` | `/healthz` (process), `/readyz` (database ping, 503 problem when down) | Implemented |
| `accounts` | Generic OIDC JWT verification (RS256/ES256 via JWKS, vendor-neutral), owner auto-provisioning by `(issuer, subject)`, `GET /v1/me`, `current_owner` dependency | CLD-11 implemented; vendor not chosen |
| `identity` | `PetId`, `ProjectionKey`, device proof message (ECDSA P-256), challenges (single-use, 5 min), TOFU pet binding creating Pet + active Brain + Device; `POST /v1/devices/challenges`, `POST/GET /v1/pets`, `GET /v1/pets/{petId}` | CLD-11 implemented; firmware Identity V2 signing pending |
| `exports` | `xuna.export.v1` envelope/hash check and strict payload schema; golden fixture produced by the mobile Dart code | Validation only; storing projections needs gate (c) + CLD-14 |
| `cognition` | Provider-independent chat/voice types, ports (`TextGenerator`, `SpeechRecognizer`, `SpeechSynthesizer`, `Tool`), `ProviderRegistry`, `echo` adapter, **`openai_compatible` streaming adapter** (owner's internal AI over VPN now, rented vendor later), **`openai_transcription` ASR** (Groq Whisper, kept-alive client), **`edge` TTS** (free voices) and optional **`piper` TTS** (`.[piper]`, GPL) | Text + phone voice V1 (owner D16, 2026-10-22); device voice by VOICE gates |
| `conversation` | `POST /v1/conversations/{petId}/turns` SSE (`delta`/`finished`/`error`, plus `audio`/`speech_unavailable` with `speak`), XUNA Vietnamese persona, U3 guard, own-pet check, nothing stored, `provider_unavailable` fallback event; `POST /v1/speech/transcriptions` (U3V guard, container check, audio never stored) | Text implemented; phone voice V1 slice 1 (2026-10-22) |

| `consent` | U0 default, append-only history (version per change, policy tag), optimistic `expectedVersion`, `GET/PUT /v1/consents`, `RequireConsent` route guard; U3V `conversation_voice` (migration 0004) | CLD-12 implemented; U3V 2026-10-22 |
| `audit` | Append-only `audit_events` (DB trigger), content-free typed metadata, `GET /v1/audit-events` for the owner's own trail; binding and consent changes recorded | CLD-12 implemented |

Cross-cutting in `core`: `idempotency` (24 h `Idempotency-Key` replay store), `db` (UTC sessions, `DbSession`, `commit_or_conflict`).

Planned (Cloud plan §8): `consent`, `conversation`, `memory`, `backup`, `suggestion`, `audit`.

## Cognition design and the Xiaozhi reference

[xiaozhi-esp32-server](https://github.com/xinnan-tech/xiaozhi-esp32-server) (MIT, inspected 2026-10-17; ideas only, no code copied) is the reference voice-assistant server for ESP32 devices. What we keep and what we deliberately do better:

| Topic | xiaozhi-esp32-server | XUNA backend |
| --- | --- | --- |
| Capability families | `vad`, `asr`, `llm`, `vllm`, `tts`, `memory`, `intent`, `tools` provider folders | Same split as `Capability` (`text`, `vision`, `speech_detection`, `speech_recognition`, `speech_synthesis`); memory/intent become their own modules with device-confirmed semantics |
| Provider selection | `selected_module` in `config.yaml` → class lookup | `ProviderRegistry`: typed factory per `(capability, name)`, unknown/duplicate names fail at startup |
| Interfaces | ABC base classes with sync generators, threads per connection | `typing.Protocol` ports, `async` streams, typed events (`TextDelta`, `ToolCallRequested`, `ChatFinished`) |
| Tools | plugin functions + MCP (server/device/endpoint) | `Tool` port with JSON-Schema `ToolSpec`; MCP can be an adapter. **Device actions are never tools**: they go through device-validated suggestions |
| Session handling | one large `ConnectionHandler` (~1,800 lines) owning audio, ASR, LLM, TTS, memory, reports | Per-module use cases; the future voice/chat session orchestrator composes ports and stays small and testable. `conversation.application.spoken_turn` (2026-10-23) overlaps TTS with the text stream (≤ 2 parallel, in-order audio, cancel on close) |
| Memory | provider writes memory directly (mem0, local) | Model output is untrusted; memory is confirmed by owner + durable device commit, projected with `(PetId, generation)`, deletable with tombstones |
| Privacy | report/record features optional | U0 default, opt-in classes checked server-side, no raw audio, no conversation text retention by default, content-free logs |
| Device transport | WebSocket/MQTT audio streaming from the device | Phone gateway HTTPS first; device MQTT/TLS after E14, voice after VOICE-01–04 |
| Quality gates | tests folder, performance tester | `ruff` + `mypy --strict` + architecture test + migrations round-trip + pytest in CI |

Features to reach parity later, each behind its gate: streaming ASR/TTS adapters (VOICE), VAD, vision LLM, MCP tool adapter, per-role system prompts, intent routing, performance tester for provider latency (CLD-13 benchmark).

## Adding a provider adapter (recipe)

1. `modules/cognition/infrastructure/<vendor>.py`: class implementing the port; timeout; map vendor errors to `ProviderUnavailableError`; credentials from `XUNA_<VENDOR>_*` env vars.
2. `register(registry)` function registering `(Capability.X, "<vendor>")`.
3. Tests with recorded fakes, never live calls by default.
4. Owner approval for the SDK dependency and CLD-13 benchmark evidence before production selection.

## Contracts

- `xuna.export.v1`: produced by mobile `ExportDocumentCodec`/`DeviceExportMapper`; verified by `exports.domain.export_document` + `exports.application.schemas`. Golden fixture `tests/fixtures/xuna_export_v1_golden.json` was generated by running the Dart repository code (2026-10-17). If the mobile format changes, regenerate the fixture from Dart and update both sides in the same sprint.
- HTTP contract draft (not frozen): parent `docs/architecture/XUNA_CLOUD_CONTRACT_DRAFT.md`.
- Threat model: parent `docs/architecture/XUNA_CLOUD_THREAT_MODEL.md`.
