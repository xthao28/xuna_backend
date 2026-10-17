# xuna_backend

Optional Cloud Brain for XUNA. The device stays authoritative; the cloud is a projection, backup and cognitive service (ADR-0008). Status: **CLD-10 scaffold + CLD-11 identity** — health probes, OIDC owner auth, device-proof (ECDSA P-256) pet binding with Pet/Brain/Device records, idempotency, `xuna.export.v1` verification and cognition provider ports. No device-data projection is stored yet (CLD-14 needs gate c).

Read [AGENTS.md](AGENTS.md) before coding and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design.

## Setup (macOS, Python 3.12, PostgreSQL 14+)

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
createdb xuna_test
export XUNA_TEST_DATABASE_URL=postgresql+psycopg://$(whoami)@localhost/xuna_test
scripts/check.sh          # format, lint, mypy --strict, migrations, tests
```

## Run locally

```sh
createdb xuna_local
export XUNA_DATABASE_URL=postgresql+psycopg://$(whoami)@localhost/xuna_local
.venv/bin/alembic upgrade head
.venv/bin/uvicorn xuna_backend.main:app --reload --no-access-log  # app logs its own content-free access line
curl localhost:8000/healthz   # {"status":"ok"}
curl localhost:8000/readyz    # {"status":"ready"}
```

Configuration is only through `XUNA_*` environment variables: `XUNA_DATABASE_URL` (required), `XUNA_ENV` (`local|test|production`), `XUNA_LOG_LEVEL`, `XUNA_OIDC_ISSUER`, `XUNA_OIDC_AUDIENCE`, `XUNA_OIDC_JWKS_URL` (defaults to `<issuer>/.well-known/jwks.json`). Without an issuer every authenticated route returns `503 auth_unavailable`.

CI: `.github/workflows/ci.yml` runs `scripts/check.sh` against PostgreSQL 16.
