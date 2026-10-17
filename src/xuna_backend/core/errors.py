"""RFC 9457 problem details. Every error response uses this shape.

Raise `AppError` (or a module subclass) from application code; never return
ad-hoc error dicts and never put user content or secrets in `detail`.
"""

from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

PROBLEM_JSON = "application/problem+json"


class AppError(Exception):
    """A failure the client may handle. `code` is stable and machine-readable."""

    def __init__(self, code: str, status: HTTPStatus, detail: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.status = status
        self.detail = detail


def problem(status: HTTPStatus, code: str, detail: str = "") -> JSONResponse:
    body: dict[str, object] = {
        "type": f"https://xuna.dev/problems/{code}",
        "title": status.phrase,
        "status": status.value,
        "code": code,
    }
    if detail:
        body["detail"] = detail
    return JSONResponse(body, status_code=status.value, media_type=PROBLEM_JSON)


def install_error_handlers(app: FastAPI) -> None:
    async def on_app_error(_: Request, error: Exception) -> JSONResponse:
        if not isinstance(error, AppError):  # registered for AppError only
            raise error
        return problem(error.status, error.code, error.detail)

    async def on_validation(_: Request, error: Exception) -> JSONResponse:
        if not isinstance(error, RequestValidationError):
            raise error
        # Field locations only: request values may contain personal data.
        fields = sorted({".".join(str(p) for p in e["loc"]) for e in error.errors()})
        return problem(HTTPStatus.UNPROCESSABLE_ENTITY, "validation_failed", ", ".join(fields))

    async def on_unexpected(_: Request, __: Exception) -> JSONResponse:
        return problem(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error")

    app.add_exception_handler(AppError, on_app_error)
    app.add_exception_handler(RequestValidationError, on_validation)
    app.add_exception_handler(Exception, on_unexpected)
