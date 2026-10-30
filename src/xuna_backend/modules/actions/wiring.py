"""Composition root wiring for the `actions` module (UDC-4 §5.2)."""

from fastapi import FastAPI

from xuna_backend.modules.actions.api.routes import router as actions_router
from xuna_backend.modules.actions.infrastructure.brokers import (
    InMemoryApprovalBroker,
    InMemoryLastTargetStore,
    InMemoryLocalResultBroker,
)
from xuna_backend.modules.actions.infrastructure.repository import SqlApprovalRepository


def wire_actions(app: FastAPI) -> None:
    app.state.approval_repository = SqlApprovalRepository
    app.state.approval_broker = InMemoryApprovalBroker()
    app.state.local_result_broker = InMemoryLocalResultBroker()
    app.state.last_target_store = InMemoryLastTargetStore()
    app.include_router(actions_router)
