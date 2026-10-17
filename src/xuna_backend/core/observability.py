"""Request IDs and content-free access logs (D23: logs redact content).

Only method, route path, status, duration and request ID are logged. Never
log bodies, query strings, headers, tokens or conversation text.
"""

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response

REQUEST_ID_HEADER = "X-Request-ID"
logger = logging.getLogger("xuna.access")


def configure_logging(level: str) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def _request_id(request: Request) -> str:
    candidate = request.headers.get(REQUEST_ID_HEADER, "")
    if 8 <= len(candidate) <= 64 and candidate.replace("-", "").isalnum():
        return candidate
    return uuid.uuid4().hex


def install_request_context(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = _request_id(request)
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "request method=%s path=%s status=%d ms=%.1f request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
            request_id,
        )
        return response
