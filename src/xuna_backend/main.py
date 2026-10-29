"""Composition root: builds the FastAPI app. Run: `uvicorn xuna_backend.main:app`."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI

from xuna_backend import __version__
from xuna_backend.core.config import Settings
from xuna_backend.core.db import Database
from xuna_backend.core.errors import install_error_handlers
from xuna_backend.core.observability import configure_logging, install_request_context
from xuna_backend.modules.accounts.api.routes import router as accounts_router
from xuna_backend.modules.accounts.application.ports import TokenVerifier
from xuna_backend.modules.accounts.infrastructure.oidc import (
    OidcTokenVerifier,
    UnconfiguredTokenVerifier,
    firebase_settings,
    jwks_resolver,
)
from xuna_backend.modules.accounts.infrastructure.repository import SqlOwnerRepository
from xuna_backend.modules.audit.api.routes import router as audit_router
from xuna_backend.modules.audit.infrastructure.repository import SqlAuditLog
from xuna_backend.modules.backups.api.routes import router as backups_router
from xuna_backend.modules.backups.infrastructure.repository import SqlBackupRepository
from xuna_backend.modules.cognition.application.ports import (
    Embedder,
    SpeechRecognizer,
    SpeechSynthesizer,
    TextGenerator,
)
from xuna_backend.modules.cognition.application.registry import ProviderRegistry
from xuna_backend.modules.cognition.domain.types import Capability
from xuna_backend.modules.cognition.infrastructure import (
    echo,
    edge_tts_adapter,
    openai_compatible,
    openai_transcription,
    piper_adapter,
)
from xuna_backend.modules.consent.api.routes import router as consent_router
from xuna_backend.modules.consent.infrastructure.repository import SqlConsentRepository
from xuna_backend.modules.conversation.api.routes import router as conversation_router
from xuna_backend.modules.health.api.routes import router as health_router
from xuna_backend.modules.identity.api.routes import router as identity_router
from xuna_backend.modules.identity.infrastructure.p256 import P256SignatureVerifier
from xuna_backend.modules.identity.infrastructure.repository import SqlIdentityRepository
from xuna_backend.modules.knowledge.api.routes import router as knowledge_router
from xuna_backend.modules.knowledge.infrastructure.repository import (
    SqlKnowledgeCandidateAdapter,
    SqlKnowledgeRepository,
)
from xuna_backend.modules.memory.api.routes import router as memory_router
from xuna_backend.modules.memory.infrastructure.repository import SqlMemoryRepository
from xuna_backend.modules.suggestions.api.routes import router as suggestions_router
from xuna_backend.modules.suggestions.infrastructure.repository import SqlSuggestionRepository
from xuna_backend.modules.transport.api.routes import router as transport_router
from xuna_backend.modules.transport.domain.device_session import (
    GatewayLimits,
    SlidingWindowLimiter,
)
from xuna_backend.modules.transport.infrastructure.memory_broker import InMemoryMqttBroker
from xuna_backend.modules.transport.infrastructure.repository import SqlTransportStatusRepository
from xuna_backend.modules.transport.infrastructure.session_hub import InMemoryDeviceSessionHub


def token_verifier(settings: Settings) -> TokenVerifier:
    if settings.firebase_project_id:
        issuer, audience, jwks_url = firebase_settings(settings.firebase_project_id)
        return OidcTokenVerifier(issuer, audience, jwks_resolver(jwks_url))
    if not settings.oidc_issuer:
        return UnconfiguredTokenVerifier()
    jwks_url = settings.oidc_jwks_url or (
        settings.oidc_issuer.rstrip("/") + "/.well-known/jwks.json"
    )
    return OidcTokenVerifier(settings.oidc_issuer, settings.oidc_audience, jwks_resolver(jwks_url))


def text_generator(settings: Settings) -> TextGenerator:
    """Fails at startup on an unknown or misconfigured provider."""
    registry = ProviderRegistry()
    echo.register(registry)
    openai_compatible.register(registry)
    options = {
        "base_url": settings.text_base_url,
        "model": settings.text_model,
        "api_key": settings.text_api_key,
        "api_key_header": settings.text_api_key_header,
        "reasoning_effort": settings.text_reasoning_effort,
    }
    return cast(TextGenerator, registry.create(Capability.TEXT, settings.text_provider, options))


def expert_text_generator(settings: Settings) -> TextGenerator | None:
    """None when expert provider is unset (reuses `text_generator`); fails fast if misconfigured."""
    if not settings.text_expert_provider:
        return None
    registry = ProviderRegistry()
    echo.register(registry)
    openai_compatible.register(registry)
    options = {
        "base_url": settings.text_expert_base_url or settings.text_base_url,
        "model": settings.text_expert_model,
        "api_key": settings.text_expert_api_key or settings.text_api_key,
        "api_key_header": settings.text_expert_api_key_header or settings.text_api_key_header,
        "reasoning_effort": settings.text_expert_reasoning_effort,
    }
    return cast(
        TextGenerator,
        registry.create(Capability.TEXT, settings.text_expert_provider, options),
    )


def speech_recognizer(settings: Settings) -> SpeechRecognizer | None:
    """None when voice is off; fails at startup on an unknown/misconfigured provider."""
    if not settings.asr_provider:
        return None
    registry = ProviderRegistry()
    openai_transcription.register(registry)
    options = {
        "base_url": settings.asr_base_url,
        "model": settings.asr_model,
        "api_key": settings.asr_api_key,
        "timeout_s": settings.asr_timeout_s,
    }
    return cast(
        SpeechRecognizer,
        registry.create(Capability.SPEECH_RECOGNITION, settings.asr_provider, options),
    )


def speech_synthesizer(settings: Settings) -> SpeechSynthesizer | None:
    """None when spoken replies are off (text replies still work)."""
    if not settings.tts_provider:
        return None
    registry = ProviderRegistry()
    edge_tts_adapter.register(registry)
    piper_adapter.register(registry)
    options = {
        "voice": settings.tts_voice,
        "model_path": settings.tts_piper_model,
        "timeout_s": settings.tts_timeout_s,
    }
    return cast(
        SpeechSynthesizer,
        registry.create(Capability.SPEECH_SYNTHESIS, settings.tts_provider, options),
    )


def embedder(settings: Settings) -> Embedder | None:
    """None when embedding is off (lexical matching used); fails at startup if misconfigured."""
    if not settings.embedding_provider:
        return None
    registry = ProviderRegistry()
    openai_compatible.register(registry)
    options = {
        "base_url": settings.embedding_base_url,
        "model": settings.embedding_model,
        "api_key": settings.embedding_api_key,
        "api_key_header": settings.embedding_api_key_header,
        "timeout_s": settings.embedding_timeout_s,
    }
    return cast(
        Embedder,
        registry.create(Capability.EMBEDDING, settings.embedding_provider, options),
    )


def create_app(
    settings: Settings | None = None,
    *,
    verifier: TokenVerifier | None = None,
    generator: TextGenerator | None = None,
    expert_generator: TextGenerator | None = None,
    recognizer: SpeechRecognizer | None = None,
    synthesizer: SpeechSynthesizer | None = None,
    embedder_instance: Embedder | None = None,
) -> FastAPI:
    resolved = settings or Settings.from_env()
    configure_logging(resolved.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.database = Database(resolved.database_url)
        try:
            yield
        finally:
            await app.state.database.dispose()
            close = getattr(app.state.speech_recognizer, "aclose", None)
            if close is not None:
                await close()

    app = FastAPI(title="XUNA Cloud", version=__version__, lifespan=lifespan)
    app.state.settings = resolved
    # Ports for routers (api never imports infrastructure).
    app.state.token_verifier = verifier or token_verifier(resolved)
    app.state.owner_repository = SqlOwnerRepository
    app.state.identity_repository = SqlIdentityRepository
    app.state.signature_verifier = P256SignatureVerifier()
    app.state.audit_log = SqlAuditLog
    app.state.consent_repository = SqlConsentRepository
    app.state.memory_repository = SqlMemoryRepository
    app.state.knowledge_repository = SqlKnowledgeRepository
    app.state.knowledge_candidate_adapter = SqlKnowledgeCandidateAdapter
    app.state.backup_repository = SqlBackupRepository
    app.state.suggestion_repository = SqlSuggestionRepository
    app.state.transport_status_repository = SqlTransportStatusRepository
    app.state.mqtt_publisher = InMemoryMqttBroker()
    # Device-direct gateway (DC-3): limits, process-local rate limits and session registry.
    limits = GatewayLimits()
    app.state.device_gateway_limits = limits
    app.state.device_challenge_limiter = SlidingWindowLimiter(
        limits.challenges_per_pet, limits.challenge_window_s
    )
    app.state.device_connect_limiter = SlidingWindowLimiter(
        limits.connects_per_peer, limits.connect_window_s
    )
    app.state.device_session_hub = InMemoryDeviceSessionHub()
    app.state.text_generator = generator or text_generator(resolved)
    app.state.expert_text_generator = expert_generator or expert_text_generator(resolved)
    app.state.speech_recognizer = recognizer or speech_recognizer(resolved)
    app.state.speech_synthesizer = synthesizer or speech_synthesizer(resolved)
    app.state.embedder = embedder_instance or embedder(resolved)
    install_error_handlers(app)
    install_request_context(app)
    app.include_router(health_router)
    app.include_router(accounts_router)
    app.include_router(identity_router)
    app.include_router(consent_router)
    app.include_router(audit_router)
    app.include_router(conversation_router)
    app.include_router(memory_router)
    app.include_router(knowledge_router)
    app.include_router(backups_router)
    app.include_router(suggestions_router)
    app.include_router(transport_router)
    return app


def __getattr__(name: str) -> FastAPI:
    # Lazy `app` for uvicorn so importing this module never reads the env.
    if name == "app":
        return create_app()
    raise AttributeError(name)
