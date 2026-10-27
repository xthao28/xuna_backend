"""Knowledge application package."""

from xuna_backend.modules.knowledge.application.ports import (
    KnowledgeCandidatePort,
    KnowledgeRepository,
)
from xuna_backend.modules.knowledge.application.schemas import (
    CreateDomainRequest,
    DecideSuggestionRequest,
    DomainResponse,
    KnowledgeSuggestionResponse,
    TeachKnowledgeRequest,
    TeachKnowledgeResponse,
    UpdateDomainRequest,
)
from xuna_backend.modules.knowledge.application.suggest import (
    ExpireKnowledgeSuggestionsUseCase,
    GenerateKnowledgeSuggestionUseCase,
)
from xuna_backend.modules.knowledge.application.use_cases import (
    CreateDomainUseCase,
    DecideSuggestionUseCase,
    GetDomainUseCase,
    ListDomainsUseCase,
    ListSuggestionsUseCase,
    TeachKnowledgeUseCase,
    UpdateDomainUseCase,
)

__all__ = [
    "CreateDomainRequest",
    "CreateDomainUseCase",
    "DecideSuggestionRequest",
    "DecideSuggestionUseCase",
    "DomainResponse",
    "ExpireKnowledgeSuggestionsUseCase",
    "GenerateKnowledgeSuggestionUseCase",
    "GetDomainUseCase",
    "KnowledgeCandidatePort",
    "KnowledgeRepository",
    "KnowledgeSuggestionResponse",
    "ListDomainsUseCase",
    "ListSuggestionsUseCase",
    "TeachKnowledgeRequest",
    "TeachKnowledgeResponse",
    "TeachKnowledgeUseCase",
    "UpdateDomainRequest",
    "UpdateDomainUseCase",
]
