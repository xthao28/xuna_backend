"""Process configuration from `XUNA_*` environment variables only.

No secret has a default value; secrets never appear in source or logs.
"""

import os
from dataclasses import dataclass
from enum import StrEnum


class Environment(StrEnum):
    LOCAL = "local"
    TEST = "test"
    PRODUCTION = "production"


@dataclass(frozen=True, slots=True)
class Settings:
    environment: Environment
    database_url: str
    log_level: str = "INFO"
    # Managed OIDC owner auth (vendor open). Empty issuer = auth unavailable.
    oidc_issuer: str = ""
    oidc_audience: str = ""
    oidc_jwks_url: str = ""
    # Firebase Auth preset (owner choice 2026-10-17): fills issuer/audience/JWKS.
    firebase_project_id: str = ""
    # Text provider (cognition). `echo` = offline fake; `openai_compatible` =
    # any OpenAI-style /chat/completions server (internal AI over VPN, vLLM,
    # Ollama /v1, hosted vendors later).
    text_provider: str = "echo"
    text_base_url: str = ""
    text_model: str = ""
    text_api_key: str = ""
    # Header carrying the key: "authorization" (Bearer, default) or a gateway
    # header such as Bifrost's "x-bf-vk" (raw key).
    text_api_key_header: str = "authorization"
    text_timeout_s: float = 30.0
    # Optional OpenAI `reasoning_effort` (e.g. "none"/"low") for thinking models,
    # whose hidden reasoning otherwise eats short output budgets. Empty = not sent.
    text_reasoning_effort: str = ""

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        source = os.environ if env is None else env
        database_url = source.get("XUNA_DATABASE_URL", "")
        if not database_url:
            raise RuntimeError("XUNA_DATABASE_URL is required")
        return cls(
            environment=Environment(source.get("XUNA_ENV", Environment.LOCAL)),
            database_url=database_url,
            log_level=source.get("XUNA_LOG_LEVEL", "INFO"),
            oidc_issuer=source.get("XUNA_OIDC_ISSUER", ""),
            oidc_audience=source.get("XUNA_OIDC_AUDIENCE", ""),
            oidc_jwks_url=source.get("XUNA_OIDC_JWKS_URL", ""),
            firebase_project_id=source.get("XUNA_FIREBASE_PROJECT_ID", ""),
            text_provider=source.get("XUNA_TEXT_PROVIDER", "echo"),
            text_base_url=source.get("XUNA_TEXT_BASE_URL", ""),
            text_model=source.get("XUNA_TEXT_MODEL", ""),
            text_api_key=source.get("XUNA_TEXT_API_KEY", ""),
            text_api_key_header=source.get("XUNA_TEXT_API_KEY_HEADER", "authorization"),
            text_timeout_s=float(source.get("XUNA_TEXT_TIMEOUT_S", "30")),
            text_reasoning_effort=source.get("XUNA_TEXT_REASONING_EFFORT", ""),
        )

    def __repr__(self) -> str:
        # Never print secrets (database password, API key) in logs or errors.
        return (
            f"Settings(environment={self.environment.value!r}, "
            f"text_provider={self.text_provider!r})"
        )
