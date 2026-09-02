from fastapi import Request
from fastapi.responses import JSONResponse

from src.utils.logging import get_logger
from src.services.workload import WorkloadOverloadedError

logger = get_logger(__name__)


class CircuitOpenError(Exception):
    """CMS write-back circuit breaker is open."""


class TranscriptionError(Exception):
    """Failure in /v1/transcribe* surface."""


class AsyncTranscriptionURLRequiredError(TranscriptionError):
    """Asynchronous jobs must reference already-persisted durable media."""


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

    if isinstance(exc, WorkloadOverloadedError):
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": "1"},
            content={
                "error": "Media workload is temporarily saturated",
                "error_code": "WORKLOAD_OVERLOADED",
                "retryable": True,
                "retry_after_seconds": 1,
                "request_id": request_id,
            },
        )

    if isinstance(exc, AsyncTranscriptionURLRequiredError):
        return JSONResponse(
            status_code=422,
            content={
                "error": "Async transcription accepts only url; persist uploads through Aggregation first",
                "error_code": "ASYNC_TRANSCRIPTION_URL_REQUIRED",
                "retryable": False,
                "request_id": request_id,
            },
        )

    if isinstance(exc, TranscriptionError):
        return JSONResponse(
            status_code=422,
            content={
                "error": "Transcription request failed",
                "error_code": "TRANSCRIPTION_FAILED",
                "retryable": False,
                "request_id": request_id,
            },
        )

    if isinstance(exc, ImageEmbeddingError):
        return JSONResponse(
            status_code=422,
            content={
                "error": "Image embedding request failed",
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
