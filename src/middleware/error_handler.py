from fastapi import Request
from fastapi.responses import JSONResponse

from src.utils.logging import get_logger

logger = get_logger(__name__)


class CircuitOpenError(Exception):
    """CMS write-back circuit breaker is open."""


class TranscriptionError(Exception):
    """Failure in /v1/transcribe* surface."""


class ImageEmbeddingError(Exception):
    """Failure in /v1/embed/image surface."""


async def global_error_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")

    if isinstance(exc, CircuitOpenError):
        return JSONResponse(
            status_code=503,
            content={
                "error": "CMS service unavailable — circuit breaker is open",
                "error_code": "CIRCUIT_OPEN",
                "retryable": True,
                "retry_after_seconds": 30,
                "request_id": request_id,
            },
        )

    if isinstance(exc, TranscriptionError):
        return JSONResponse(
            status_code=422,
            content={
                "error": str(exc),
                "error_code": "TRANSCRIPTION_FAILED",
                "retryable": False,
                "request_id": request_id,
            },
        )

    if isinstance(exc, ImageEmbeddingError):
        return JSONResponse(
            status_code=422,
            content={
                "error": str(exc),
                "error_code": "IMAGE_EMBEDDING_FAILED",
                "retryable": False,
                "request_id": request_id,
            },
        )

    logger.exception("unhandled_error", request_id=request_id, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "retryable": True,
            "retry_after_seconds": 5,
            "request_id": request_id,
        },
    )
