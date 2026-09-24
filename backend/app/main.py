import logging
import re
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.health import router as health_router
from app.api.auth import router as auth_router
from app.api.academic import router as academic_router
from app.api.materials import router as materials_router
from app.api.retrieval import router as retrieval_router
from app.api.templates import router as templates_router
from app.api.questions import router as questions_router
from app.api.errors import ErrorResponse
from app.core.config import get_settings
from app.core.logging import configure_logging, correlation_id


settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version=settings.app_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    supplied_request_id = request.headers.get("X-Request-ID")
    request_id = (
        supplied_request_id
        if supplied_request_id and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", supplied_request_id)
        else str(uuid4())
    )
    token = correlation_id.set(request_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info("request complete: %s %s %s", request.method, request.url.path, response.status_code)
        return response
    finally:
        correlation_id.reset(token)


app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(auth_router, prefix=settings.api_prefix)
app.include_router(academic_router, prefix=settings.api_prefix)
app.include_router(materials_router, prefix=settings.api_prefix)
app.include_router(retrieval_router, prefix=settings.api_prefix)
app.include_router(templates_router, prefix=settings.api_prefix)
app.include_router(questions_router, prefix=settings.api_prefix)


def error_response(
    request: Request,
    *,
    status_code: int,
    error: str,
    message: str,
    details: list[dict[str, object]] | None = None,
) -> JSONResponse:
    payload = ErrorResponse(
        error=error,
        message=message,
        request_id=correlation_id.get(),
        details=details,
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    message = exc.detail if isinstance(exc.detail, str) else "The request could not be completed."
    return error_response(
        request,
        status_code=exc.status_code,
        error="http_error",
        message=message,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return error_response(
        request,
        status_code=422,
        error="validation_error",
        message="The request is invalid.",
        details=[dict(error) for error in exc.errors()],
    )
