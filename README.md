# xuna_backend

Optional Cloud Brain for XUNA. The device stays authoritative; the cloud is a projection, backup and cognitive service (ADR-0008). Status: **CLD-10 scaffold + CLD-11 identity + CLD-12 consent/audit + CLD-13 text provider adapter and conversation turn** — health probes, OIDC owner auth, device-proof (ECDSA P-256) pet binding with Pet/Brain/Device records, idempotency, `xuna.export.v1` verification and cognition provider ports. No device-data projection is stored yet (CLD-14 needs gate c).

Read [AGENTS.md](AGENTS.md) before coding and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design.

## Owner machines: Mac vs Windows (do not mix them up)

Each machine has its **own** PostgreSQL and its own databases. Nothing is shared between them: the dev pet binding in the Mac's `xuna_local` does not exist on Windows. Use the command for the machine you are on.

| | **Mac** (`/Users/xthao28`) | **Windows** (`C:\Users\Thao`) |
| --- | --- | --- |
| Python | 3.12.8 | 3.12.10 (winget, `%LOCALAPPDATA%\Programs\Python\Python312`) |
| venv tools | `.venv/bin/` | `.venv/Scripts/` |
| PostgreSQL | **14.15**, local socket, user = your login (`$(whoami)`), no password | **16.15** (winget EDB installer, Windows service `postgresql-x64-16`, port 5432), user **`xuna`** / password **`xuna`** (local test only, same as CI); superuser `postgres` password in `C:\Users\Thao\.xuna_pg_superuser` (outside Git) |
| `psql` | on `PATH` | `"C:\Program Files\PostgreSQL\16\bin\psql.exe"` (not on `PATH`) |
| Test database URL | `postgresql+psycopg://$(whoami)@localhost/xuna_test` | `postgresql+psycopg://xuna:xuna@localhost:5432/xuna_test` |
| Dev database `xuna_local` (uvicorn, phone tests) | yes (bound real pet lives here) | **not created** |
| Run uvicorn for the phone | yes (tested) | not tried; psycopg async needs the Selector event loop on Windows (tests set it in `tests/conftest.py`) |
| Shell for `scripts/check.sh` | zsh/bash | Git Bash |

### Mac (Python 3.12, PostgreSQL 14)

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
createdb xuna_test
export XUNA_TEST_DATABASE_URL=postgresql+psycopg://$(whoami)@localhost/xuna_test
scripts/check.sh          # format, lint, mypy --strict, migrations, tests
```

### Windows (Python 3.12, PostgreSQL 16, Git Bash)

```sh
~/AppData/Local/Programs/Python/Python312/python.exe -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
# once: role + database (superuser password from ~/.xuna_pg_superuser)
PGPASSWORD=$(cat ~/.xuna_pg_superuser) "/c/Program Files/PostgreSQL/16/bin/psql.exe" -h localhost -U postgres \
  -c "CREATE ROLE xuna LOGIN PASSWORD 'xuna';" -c "CREATE DATABASE xuna_test OWNER xuna;"
XUNA_TEST_DATABASE_URL=postgresql+psycopg://xuna:xuna@localhost:5432/xuna_test PY=.venv/Scripts bash scripts/check.sh
```

## Run locally (Mac)

```sh
createdb xuna_local
export XUNA_DATABASE_URL=postgresql+psycopg://$(whoami)@localhost/xuna_local
.venv/bin/alembic upgrade head
.venv/bin/uvicorn xuna_backend.main:app --reload --no-access-log  # app logs its own content-free access line
curl localhost:8000/healthz   # {"status":"ok"}
curl localhost:8000/readyz    # {"status":"ready"}
```

Configuration is only through `XUNA_*` environment variables: `XUNA_DATABASE_URL` (required), `XUNA_ENV` (`local|test|production`), `XUNA_LOG_LEVEL`, `XUNA_OIDC_ISSUER`, `XUNA_OIDC_AUDIENCE`, `XUNA_OIDC_JWKS_URL` (defaults to `<issuer>/.well-known/jwks.json`). Without an issuer every authenticated route returns `503 auth_unavailable`. **Firebase Auth** (owner choice): set only `XUNA_FIREBASE_PROJECT_ID`; issuer, audience and JWKS are derived.

Text provider: `XUNA_TEXT_PROVIDER` (`echo` default, or `openai_compatible`), `XUNA_TEXT_BASE_URL`, `XUNA_TEXT_MODEL`, `XUNA_TEXT_API_KEY`, `XUNA_TEXT_TIMEOUT_S`. Internal AI over OpenVPN: [docs/ops/internal-ai-over-vpn.md](docs/ops/internal-ai-over-vpn.md). Vietnamese benchmark: `scripts/text_benchmark.py`.

Phone-free end-to-end check (simulated board + app-shaped requests against a real server with a local test issuer; adds rows, so use a dev/test database): `.venv/bin/python scripts/device_e2e.py --database-url postgresql+psycopg://$(whoami)@localhost/xuna_test`.

CI: `.github/workflows/ci.yml` runs `scripts/check.sh` against PostgreSQL 16.
