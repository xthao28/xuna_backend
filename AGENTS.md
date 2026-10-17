# XUNA backend — coding rules

> **Mandatory before writing or changing any code in this repository.** Applies to every coding agent (Claude Code, Codex, Cursor, Copilot) and engineer.
>
> - Product intent, approved decisions and gates: the parent repository's [AGENTS.md](../AGENTS.md) (canonical), then [PROCESS.md](../PROCESS.md), the [Cloud plan](../docs/architecture/XUNA_CLOUD_IMPLEMENTATION_PLAN.md), [ADR-0008](../docs/decisions/ADR-0008-cloud-brain-authority-and-identity.md), [ADR-0009](../docs/decisions/ADR-0009-cloud-platform-sequencing-and-privacy.md), the [threat model](../docs/architecture/XUNA_CLOUD_THREAT_MODEL.md) and the [contract draft](../docs/architecture/XUNA_CLOUD_CONTRACT_DRAFT.md).
> - How backend code is written: **this file** and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
> - If a request conflicts with either, stop and ask the owner. Changing these rules requires owner approval.
>
> Status: created 2026-10-17 (owner delegated structure and rules). Stack: Python 3.12, FastAPI, SQLAlchemy 2 (async), Alembic, Pydantic 2, PostgreSQL. Machine-checked by `scripts/check.sh` and `tests/test_architecture.py`.

## 0. Workflow

1. Read the documents above. Confirm the work item (CLD-xx) is allowed by its gate; do not build later phases speculatively.
2. Tell the owner in Vietnamese: purpose, scope, modules/layers touched, whether a database, provider or device is needed (parent AGENTS §10).
3. Design top-down (endpoint → use case → ports), implement bottom-up: **domain → application → infrastructure → api → wiring → migration → tests**.
4. Run `scripts/check.sh` (with `XUNA_TEST_DATABASE_URL` set when the change touches the database). It must pass.
5. Report in Vietnamese: files changed, commands and results, what was **not** run, risks, next step. Update README/ARCHITECTURE and parent handoff docs when behavior or contracts change. Commit only when asked.

## 1. Product invariants (non-negotiable)

1. **The device is authoritative** for identity, life state, learning evidence, personality, growth and physical actions. The cloud stores *projections*, labelled with `(PetId, snapshotGeneration)` and receive time. It never writes device state directly.
2. **PetId is device-issued.** The server validates (`PetId` value object) and references it; it never generates or reassigns one. Keep `UserId`, `PetId`, `BrainId`, `DeviceId` distinct types.
3. **Provider output is untrusted.** LLM/ASR text is display text or a *candidate*. Nothing becomes a fact, preference or growth change without owner confirmation **and** a durable device commit (CLD-14). Tool-call arguments from a model are untrusted input and are validated like any request.
4. **No device actuation from the cloud.** Device behaviour changes only through typed, signed, bounded, expiring, replay-protected suggestions that the device `ActionGate` re-validates. A device action is never a model tool.
5. **Privacy (D23):** U0 (no upload) is the default; U1/U2/U3 are separate opt-ins checked server-side before storing anything. Raw audio is never stored. Conversation text is not retained after the provider call unless U3 is on. Confirmed facts are correctable, exportable, deletable; deletion writes a tombstone that blocks resurrection.
6. **Offline-first:** a provider or service outage degrades features; it must never break identity or require the device to change behaviour.
7. **Gates:** storing device-data projections (CLD-14+) needs Cloud gate (a)–(c). Choosing a vendor needs CLD-13 evidence (D11a open). Voice capabilities (VAD/ASR/TTS adapters) are interface-only until VOICE-01–04.

## 2. Structure and the dependency rule

Modular monolith. One deployable, one database, modules with hard internal boundaries so any module can later become its own service.

```text
src/xuna_backend/
├── main.py                  # composition root: create_app()
├── core/                    # business-free framework: config, db, errors, observability
└── modules/<module>/
    ├── domain/              # entities, value objects, pure rules — stdlib only
    ├── application/         # use cases, ports (typing.Protocol), DTO schemas
    ├── infrastructure/      # SQLAlchemy models/repositories, provider adapters, clients
    ├── api/                 # FastAPI routers + request/response schemas
    └── wiring.py            # module composition root (optional)
```

| Layer | May import | Must not import |
| --- | --- | --- |
| `domain` | stdlib, own/other modules' `domain` | `pydantic`, any framework (`fastapi`, `starlette`, `sqlalchemy`, `alembic`, `psycopg`, `httpx`, `uvicorn`), `core`, any `application`/`infrastructure`/`api` |
| `application` | stdlib, `pydantic`, `domain`, `application` | frameworks, `core`, `infrastructure`, `api` |
| `infrastructure` | `domain`, `application`, `core`, frameworks, vendor SDKs | `api` |
| `api` | `domain`, `application`, `core`, `fastapi` | `infrastructure` (inject through dependencies wired in `main.py`/`wiring.py`) |
| `core` | stdlib, frameworks | `modules` |

- These rules are enforced by `tests/test_architecture.py`. Do not weaken that test to make code pass; fix the code.
- Cross-module calls go through the other module's `application` use cases or `domain` types, never its `infrastructure` or tables. Modules do not join each other's tables; share IDs.
- Vendor SDKs (`openai`, `anthropic`, `google`, `dashscope`, `boto3`, …) may be imported **only** in an `infrastructure/` adapter.
- Module names are nouns for a business capability: `identity`, `consent`, `conversation`, `memory`, `backup`, `suggestion`, `cognition`, `exports`, `health`. Create a module only when a work item needs it; no empty skeletons.

## 3. Domain

- Frozen `@dataclass(frozen=True, slots=True)` value objects/entities that validate in `__post_init__` and raise `ValueError`.
- Closed sets are `StrEnum` whose values are the wire strings (snake_case). Never use enum ordinals in storage or wire formats.
- Device integers keep their device semantics: u32 generations compare with serial-number arithmetic (`ProjectionKey`); Q8.8 masses stay integers.
- No I/O, clock or randomness in domain: pass `now`, IDs and entropy in.

## 4. Application

- One use case per business action: a function or small class named `verb_noun` / `VerbNoun`, taking ports through parameters or the constructor.
- Ports are `typing.Protocol` classes in `application/ports.py`. Long-running provider ports stream with `AsyncIterator`.
- Request/response schemas are Pydantic models with `ConfigDict(extra="forbid", frozen=True, strict=True)`; JSON field names follow the wire contract (camelCase where the device/app contract is camelCase).
- Raise module exceptions (subclasses of `ValueError`/`Exception` defined in the module) — the API layer maps them to problem codes.
- Every mutating use case is idempotent on its `Idempotency-Key`, and records an audit event (CLD-12) once that module exists.

## 5. Infrastructure

- ORM models subclass `core.db.Base` (naming convention fixed). Tables: snake_case plural; primary keys are UUIDs or device-issued IDs; timestamps are `timestamptz` in UTC; versioned payloads are `JSONB` with an explicit `schema_version` column.
- Async SQLAlchemy only (`AsyncSession`); one transaction per use case; no lazy loading across awaits.
- Provider adapters: one file per vendor, register with `ProviderRegistry` under a stable name, enforce their own timeout, map vendor errors to `ProviderUnavailableError`, never log prompts/outputs, read credentials only from `XUNA_*` env vars.
- Every external call has an explicit timeout. Retries only for idempotent calls, with backoff and a cap.

## 6. API

- Versioned paths `/v1/...`; resource names plural nouns. Health probes `/healthz`, `/readyz` are unversioned.
- Errors are RFC 9457 `application/problem+json` with a stable `code` (`core.errors.AppError`). Never return `str(exception)`, stack traces, user content or secrets.
- Mutating requests require `Idempotency-Key` and carry `schemaVersion` in the body (contract draft §2).
- Authentication: managed OIDC/magic-link for owners and per-device asymmetric credentials (CLD-11). No custom password system, no shared fleet secret, no API keys in source.
- Streaming responses (chat, voice) use SSE or WebSocket with typed event payloads mirroring `cognition.domain.types`.

## 7. Database migrations

- Every schema change is an Alembic revision `NNNN_slug.py` with a working `downgrade()`. Revisions are linear (single head).
- Import new ORM models in `migrations/env.py`.
- `tests/test_migrations.py` must round-trip (upgrade → downgrade → upgrade) against PostgreSQL in CI.
- Data-destroying migrations need owner approval and a backup note in the PR.

## 8. Security, privacy, logging

- Configuration only from `XUNA_*` environment variables (`core.config.Settings`); no secret has a default. `.env` files are never committed.
- Logs are content-free: method, route path, status, duration, request ID, error code. Never log bodies, query strings, headers, tokens, prompts, transcripts, provider output or personal facts. No `print` (ruff `T20`).
- Validation errors report field locations only.
- Run uvicorn with `--no-access-log`: its default access log includes query strings. `core.observability` writes the only access log.
- Treat every inbound payload (app, device, provider, tool arguments) as untrusted: size limits, strict schemas, enum ranges.
- Dependencies are pinned exactly in `pyproject.toml`. New dependencies need owner approval with license and maintenance check; vendor SDKs additionally need CLD-13 evidence.

## 9. Testing and checks

- Test paths mirror `src/`: `tests/modules/<module>/test_<topic>.py`; framework tests at `tests/`.
- Domain and application: table-driven tests including negative cases; no database.
- Infrastructure: tests marked `@pytest.mark.postgres` using the `database_url` fixture (skipped without `XUNA_TEST_DATABASE_URL`, always run in CI).
- API: `TestClient` against `create_app(Settings(...))`; error paths assert the problem `code`.
- Providers: tests use `EchoTextGenerator`-style deterministic fakes; real vendor calls are never part of the default test run.
- Contract fixtures shared with mobile/firmware live in `tests/fixtures/` and are regenerated only from the producing side (see ARCHITECTURE §Contracts).
- **Definition of Done** — before reporting:

```sh
XUNA_TEST_DATABASE_URL=postgresql+psycopg://<user>@localhost/xuna_test scripts/check.sh
```

  It runs `ruff format --check`, `ruff check`, `mypy --strict`, `alembic upgrade head` and `pytest`. All must pass. Report anything skipped.

## 10. Style

- Type hints everywhere (`mypy --strict`). `from __future__` not needed on 3.12; use `X | None`, `list[...]`.
- Imports absolute (`xuna_backend.…`), sorted by ruff. Line length 100.
- Docstrings on modules, public ports, use cases and anything with a non-obvious rule; no comments that restate code.
- English identifiers and docs; owner-facing reports in Vietnamese.

## 11. Definition of Done checklist

- [ ] Work item allowed by its gate; nothing from later phases.
- [ ] Layer rules respected; `tests/test_architecture.py` passes unmodified.
- [ ] Product invariants §1 respected (authority, untrusted output, consent, no actuation).
- [ ] Problem-details errors with stable codes; no content in logs.
- [ ] Migration with downgrade when the schema changed.
- [ ] Tests for each layer touched; `scripts/check.sh` passes; skipped checks reported.
- [ ] README/ARCHITECTURE/contract docs and parent PROCESS/PLAN/CHANGELOG updated.
