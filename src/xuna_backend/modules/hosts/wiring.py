"""Composition root wiring for the `hosts` module (UDC-3 §5.4)."""

from fastapi import FastAPI

from xuna_backend.modules.hosts.api.routes import router as hosts_router
from xuna_backend.modules.hosts.domain.session import HostGatewayLimits, HostRateLimiter
from xuna_backend.modules.hosts.infrastructure.repository import (
    SqlEnrollmentRepository,
    SqlHostRepository,
)
from xuna_backend.modules.hosts.infrastructure.session_hub import InMemoryHostSessionHub


def wire_hosts(app: FastAPI) -> None:
    limits = HostGatewayLimits()
    app.state.host_repository = SqlHostRepository
    app.state.host_enrollment_repository = SqlEnrollmentRepository
    app.state.host_gateway_limits = limits
    app.state.host_claim_limiter = HostRateLimiter(limits.claims_per_owner_per_min, 60.0)
    app.state.host_connect_limiter = HostRateLimiter(limits.connects_per_peer_per_min, 60.0)
    app.state.host_session_hub = InMemoryHostSessionHub()
    app.include_router(hosts_router)
