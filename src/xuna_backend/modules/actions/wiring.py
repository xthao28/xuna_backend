"""Composition root wiring for the `actions` module (UDC-4 §5.2, UDC-7, UDC-8)."""

from fastapi import FastAPI

from xuna_backend.modules.actions.api.routes import router as actions_router
from xuna_backend.modules.actions.infrastructure.approval_scope_store import (
    SqlApprovalScopeStore,
)
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.cu_session_store import (
    SqlComputerUseSessionStore,
)
from xuna_backend.modules.actions.infrastructure.repository import SqlApprovalRepository
from xuna_backend.modules.actions.infrastructure.standing_repository import (
    SqlStandingApprovalRepository,
)


def wire_actions(app: FastAPI) -> None:
    app.state.approval_repository = SqlApprovalRepository
    app.state.standing_approval_repository = SqlStandingApprovalRepository
    app.state.approval_broker = InMemoryApprovalBroker()
    app.state.local_result_broker = InMemoryLocalResultBroker()
    app.state.last_target_store = InMemoryLastTargetStore()
    app.state.approval_scope_store = SqlApprovalScopeStore(lambda: app.state.database.sessions())
    # UDC-17 S17.6: durable sessions. The database is created in the app lifespan, so the
    # store reaches it at call time.
    app.state.cu_session_store = SqlComputerUseSessionStore(lambda: app.state.database.sessions())
    app.include_router(actions_router)
