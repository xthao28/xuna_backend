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
    # Expert-mode text provider (KX-1, plan §6.1). Empty provider = reuse `text_generator`.
    text_expert_provider: str = ""
    text_expert_base_url: str = ""
    text_expert_model: str = ""
    text_expert_api_key: str = ""
    text_expert_api_key_header: str = ""
    text_expert_reasoning_effort: str = ""
    text_expert_timeout_s: float = 60.0
    text_expert_max_tokens: int = 3000
    # Phone voice V1 (owner D16, 2026-10-22). Empty provider = voice off (the
    # endpoints answer 503 voice_unavailable). ASR `openai_transcription` = any
    # OpenAI-style /audio/transcriptions server (Groq Whisper in V0).
    asr_provider: str = ""
    asr_base_url: str = ""
    asr_model: str = ""
    asr_api_key: str = ""
    asr_timeout_s: float = 10.0
    asr_auto_detect: bool = True
    # TTS `edge` (free online voices) or `piper` (local, optional GPL extra).
    tts_provider: str = ""
    tts_voice: str = ""
    tts_piper_model: str = ""
    tts_timeout_s: float = 10.0
    # Embedding adapter (KD-4, optional). Empty provider = embedding off (lexical fallback).
    embedding_provider: str = ""
    embedding_base_url: str = ""
    embedding_model: str = ""
    embedding_api_key: str = ""
    embedding_api_key_header: str = "authorization"
    embedding_timeout_s: float = 10.0
    # Backup encryption key (Phase 2, CLD-20). Empty = derives deterministic fallback.
    backup_encryption_key: str = ""
    # Dev-only host tool invocation route (`XUNA_DEV_HOST_TOOLS=1`, UDC-3 §5.4).
    # Never enabled in production.
    dev_host_tools: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.text_expert_timeout_s <= 120:
            raise ValueError("text_expert_timeout_s must be in (0, 120]")
        if not 512 <= self.text_expert_max_tokens <= 8192:
            raise ValueError("text_expert_max_tokens must be in 512..8192")
        if self.dev_host_tools and self.environment == Environment.PRODUCTION:
            raise ValueError("dev_host_tools must never be enabled in production")

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
            text_expert_provider=source.get("XUNA_TEXT_EXPERT_PROVIDER", ""),
            text_expert_base_url=source.get("XUNA_TEXT_EXPERT_BASE_URL", ""),
            text_expert_model=source.get("XUNA_TEXT_EXPERT_MODEL", ""),
            text_expert_api_key=source.get("XUNA_TEXT_EXPERT_API_KEY", ""),
            text_expert_api_key_header=source.get("XUNA_TEXT_EXPERT_API_KEY_HEADER", ""),
            text_expert_reasoning_effort=source.get("XUNA_TEXT_EXPERT_REASONING_EFFORT", ""),
            text_expert_timeout_s=float(source.get("XUNA_TEXT_EXPERT_TIMEOUT_S", "60") or "60"),
            text_expert_max_tokens=int(source.get("XUNA_TEXT_EXPERT_MAX_TOKENS", "3000") or "3000"),
            asr_provider=source.get("XUNA_ASR_PROVIDER", ""),
            asr_base_url=source.get("XUNA_ASR_BASE_URL", ""),
            asr_model=source.get("XUNA_ASR_MODEL", ""),
            asr_api_key=source.get("XUNA_ASR_API_KEY", ""),
            asr_timeout_s=float(source.get("XUNA_ASR_TIMEOUT_S", "10")),
            asr_auto_detect=source.get("XUNA_ASR_AUTO_DETECT", "true").lower()
            in ("true", "1", "yes"),
            tts_provider=source.get("XUNA_TTS_PROVIDER", ""),
            tts_voice=source.get("XUNA_TTS_VOICE", ""),
            tts_piper_model=source.get("XUNA_TTS_PIPER_MODEL", ""),
            tts_timeout_s=float(source.get("XUNA_TTS_TIMEOUT_S", "10")),
            embedding_provider=source.get("XUNA_EMBEDDING_PROVIDER", ""),
            embedding_base_url=source.get("XUNA_EMBEDDING_BASE_URL", ""),
            embedding_model=source.get("XUNA_EMBEDDING_MODEL", ""),
            embedding_api_key=source.get("XUNA_EMBEDDING_API_KEY", ""),
            embedding_api_key_header=source.get("XUNA_EMBEDDING_API_KEY_HEADER", "authorization"),
            embedding_timeout_s=float(source.get("XUNA_EMBEDDING_TIMEOUT_S", "10")),
            backup_encryption_key=source.get("XUNA_BACKUP_ENCRYPTION_KEY", ""),
            dev_host_tools=source.get("XUNA_DEV_HOST_TOOLS", "").lower() in ("1", "true", "yes"),
        )

    def __repr__(self) -> str:
        # Never print secrets (database password, API key) in logs or errors.
        return (
            f"Settings(environment={self.environment.value!r}, "
            f"text_provider={self.text_provider!r}, "
            f"text_expert_provider={self.text_expert_provider!r}, "
            f"asr_provider={self.asr_provider!r}, "
            f"tts_provider={self.tts_provider!r})"
        )
