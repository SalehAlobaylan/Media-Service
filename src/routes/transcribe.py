import asyncio
import os
import tempfile
from typing import Any, Literal
from uuid import UUID, uuid4

from arq.jobs import Job
from arq.jobs import JobStatus as ArqJobStatus
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from src.auth.service_auth import verify_service_token
from src.middleware.error_handler import AsyncTranscriptionURLRequiredError, TranscriptionError
from src.middleware.request_id import current_request_id
from src.schemas.transcribe import TranscribeResponse
from src.services.transcription import TranscriptionService
from src.utils.logging import get_logger
from src.utils.metrics import transcribe_jobs_total, transcriptions_total
from src.utils.url_guard import safe_url_host

logger = get_logger(__name__)
router = APIRouter(dependencies=[Depends(verify_service_token)])

UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB
SUPPORTED_LANGUAGES = {"ar", "en", "multi"}
ALLOWED_AUDIO_SUFFIXES = {
    ".mp3",
    ".m4a",
    ".mp4",
    ".wav",
    ".ogg",
    ".opus",
    ".flac",
    ".webm",
}


async def _get_arq_pool(request: Request):
    """Use the lifecycle manager when present; retain direct test injection."""
    manager = getattr(request.app.state, "queue_manager", None)
    if manager is not None:
        pool = await manager.get_pool()
        request.app.state.arq_pool = pool
        return pool
    return getattr(request.app.state, "arq_pool", None)


async def _mark_arq_unavailable(request: Request, pool: Any) -> None:
    manager = getattr(request.app.state, "queue_manager", None)
    if manager is not None:
        await manager.mark_unavailable(pool)
        request.app.state.arq_pool = None


JobStatus = Literal["queued", "in_progress", "completed", "failed", "not_found"]


class JobAcceptedResponse(BaseModel):
    """Returned from POST /v1/transcribe/jobs (HTTP 202)."""

    job_id: str
    status: Literal["queued"] = "queued"


class JobStatusResponse(BaseModel):
    """Returned from GET /v1/transcribe/jobs/{id}."""

    job_id: str
    status: JobStatus
    result: TranscribeResponse | None = None
    error: str | None = None


class JobCancelResponse(BaseModel):
    job_id: str
    status: str
    canceled: bool
    reason: str | None = None


def _validate_transcription_metadata(
    content_id: str | None,
    transcription_job_id: str | None,
    language: str | None,
    media_size_bytes: int | None,
) -> None:
    for value, label in (
        (content_id, "content_id"),
        (transcription_job_id, "transcription_job_id"),
    ):
        if value:
            try:
                UUID(value)
            except ValueError as exc:
                raise TranscriptionError(f"Invalid {label}") from exc
    if language is not None and language not in SUPPORTED_LANGUAGES:
        raise TranscriptionError("Unsupported language")
    if media_size_bytes is not None and media_size_bytes < 0:
        raise TranscriptionError("media_size_bytes must be nonnegative")


def _validate_audio_source(filename: str | None, content_type: str | None) -> None:
    suffix = os.path.splitext(filename or "")[1].lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise TranscriptionError("Unsupported media type")
    if content_type and not (
        content_type.startswith("audio/") or content_type in {"video/mp4", "video/webm"}
    ):
        raise TranscriptionError("Unsupported media type")


async def _spool_upload_to_disk(
    upload: UploadFile, tmp_path: str, max_bytes: int
) -> int:
    """Stream upload to disk; abort if it exceeds max_bytes. Returns bytes written."""
    written = 0
    with open(tmp_path, "wb") as f:
        while True:
            chunk = await upload.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                raise TranscriptionError(
                    f"Upload exceeds maximum size of {max_bytes // (1024 * 1024)} MB"
                )
            f.write(chunk)
    return written


async def _mark_transcription_job_failed(
    request: Request,
    transcription_job_id: str | None,
    message: str,
    media_size_bytes: int | None = None,
) -> None:
    if not transcription_job_id:
        return
    model_manager = request.app.state.model_manager
    service = TranscriptionService(model_manager.stt, request.app.state.cms_client)
    await service._update_job(
        transcription_job_id,
        {
            "status": "failed",
            "provider": model_manager.stt.name,
            "model": model_manager.stt.model_size,
            "error_message": message,
            "provider_error_code": "stt_not_ready",
            "metadata": {
                "media_size_bytes": media_size_bytes,
            }
            if media_size_bytes is not None
            else None,
        },
    )


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    request: Request,
    audio_file: UploadFile | None = File(None),
    url: str | None = Form(None),
    content_id: str | None = Form(None),
    transcription_job_id: str | None = Form(None),
    language: str | None = Form(None),
    media_size_bytes: int | None = Form(None),
    word_timestamps: bool = Form(False),
) -> TranscribeResponse:
    settings = request.app.state.settings
    model_manager = request.app.state.model_manager
    cms_client = request.app.state.cms_client
    service = TranscriptionService(
        model_manager.stt,
        cms_client,
        temp_dir=getattr(request.app.state, "temp_dir", None),
        fetch_client=getattr(request.app.state, "fetch_client", None),
        max_download_bytes=settings.MAX_UPLOAD_MB * 1024 * 1024,
        admission=getattr(request.app.state, "workload_admission", None),
    )

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024

    if not model_manager.stt.is_loaded:
        await _mark_transcription_job_failed(
            request, transcription_job_id, "STT engine is not ready", media_size_bytes
        )
        raise TranscriptionError("STT engine is not ready")

    has_file = bool(audio_file and audio_file.filename)
    has_url = bool(url and url.strip())
    if has_file == has_url:
        raise TranscriptionError("Provide exactly one of audio_file or url")
    _validate_transcription_metadata(
        content_id, transcription_job_id, language, media_size_bytes
    )
    if has_file and audio_file is not None:
        _validate_audio_source(audio_file.filename, audio_file.content_type)

    # Fast path: reject oversize uploads via Content-Length before streaming.
    if audio_file is not None:
        content_length = request.headers.get("content-length")
        if (
            content_length
            and content_length.isdigit()
            and int(content_length) > max_bytes + UPLOAD_CHUNK_SIZE
        ):
            await _mark_transcription_job_failed(
                request,
                transcription_job_id,
                f"Upload exceeds maximum size of {settings.MAX_UPLOAD_MB} MB",
                media_size_bytes,
            )
            raise TranscriptionError(
                f"Upload exceeds maximum size of {settings.MAX_UPLOAD_MB} MB"
            )

    try:
        if has_file and audio_file is not None:
            # Starlette may already have a named, disk-backed spool. Reuse it
            # only when it has a stable path; otherwise copy into our validated
            # scratch directory so the provider never observes a dying handle.
            existing_path = getattr(audio_file.file, "name", None)
            tmp_path = (
                existing_path
                if isinstance(existing_path, str) and os.path.isfile(existing_path)
                else None
            )
            try:
                if tmp_path is None:
                    suffix = os.path.splitext(audio_file.filename)[1] or ".mp3"
                    fd, tmp_path = tempfile.mkstemp(
                        suffix=suffix,
                        dir=getattr(
                            request.app.state, "temp_dir", tempfile.gettempdir()
                        ),
                    )
                    os.close(fd)
                    await _spool_upload_to_disk(audio_file, tmp_path, max_bytes)

                return await service.transcribe_file(
                    tmp_path,
                    content_id=content_id,
                    transcription_job_id=transcription_job_id,
                    language=language,
                    media_size_bytes=media_size_bytes,
                    word_timestamps=word_timestamps,
                )
            finally:
                if existing_path != tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        elif has_url and url is not None:
            return await service.transcribe_url(
                url,
                content_id=content_id,
                transcription_job_id=transcription_job_id,
                language=language,
                media_size_bytes=media_size_bytes,
                word_timestamps=word_timestamps,
            )

    except TranscriptionError:
        raise
    except Exception as exc:
        transcriptions_total.labels(
            status="failure", model_size=model_manager.stt.model_size
        ).inc()
        logger.error("transcription_failed", error_code="transcription_failed")
        raise TranscriptionError("Transcription failed") from exc


# ─── Async transcription endpoints ─────────────────────────────────────


@router.post(
    "/transcribe/jobs",
    response_model=JobAcceptedResponse,
    status_code=202,
)
async def submit_transcribe_job(
    request: Request,
) -> JobAcceptedResponse:
    """Enqueue an async transcription job. Use for long-form (>2 min) audio.

    The worker process (separate from this API) consumes from Redis and runs
    the transcription. Poll GET /v1/transcribe/jobs/{id} for status.
    """
    form = await request.form()
    audio_file = form.get("audio_file")
    raw_url = form.get("url")
    raw_content_id = form.get("content_id")
    raw_job_id = form.get("transcription_job_id")
    raw_language = form.get("language")
    raw_size = form.get("media_size_bytes")
    raw_timestamps = form.get("word_timestamps", False)
    url = raw_url.strip() if isinstance(raw_url, str) else None
    content_id = raw_content_id if isinstance(raw_content_id, str) else None
    transcription_job_id = raw_job_id if isinstance(raw_job_id, str) else None
    language = raw_language if isinstance(raw_language, str) else None
    try:
        media_size_bytes = int(raw_size) if isinstance(raw_size, str) and raw_size else None
    except ValueError as exc:
        raise TranscriptionError("media_size_bytes must be an integer") from exc
    word_timestamps = str(raw_timestamps).lower() in {"1", "true", "yes", "on"}
    has_file = isinstance(audio_file, UploadFile) and bool(audio_file.filename)
    has_url = bool(url)
    if has_file:
        raise AsyncTranscriptionURLRequiredError(
            "Async transcription accepts only url; persist uploads through Aggregation first"
        )
    if not has_url:
        raise AsyncTranscriptionURLRequiredError(
            "Async transcription accepts only url; persist uploads through Aggregation first"
        )
    _validate_transcription_metadata(
        content_id, transcription_job_id, language, media_size_bytes
    )

    arq_pool = await _get_arq_pool(request)

    if arq_pool is None:
        await _mark_transcription_job_failed(
            request,
            transcription_job_id,
            "Async transcription unavailable: Redis (arq) not reachable",
            media_size_bytes,
        )
        raise HTTPException(
            status_code=503,
            detail="Async transcription unavailable: Redis (arq) not reachable",
        )

    dedupe_key = transcription_job_id or content_id
    dedupe_id = f"transcribe:{dedupe_key}" if dedupe_key else None
    job_id = dedupe_id or f"transcribe:{uuid4().hex}"
    enqueue_attempted = False
    try:
        # Forward the current request_id into the job so the worker logs line
        # up with this API call when debugging cross-process traces.
        request_id = current_request_id()
        # Deterministic job id when we have a content_id: an at-least-once
        # re-submit (e.g. an upstream BullMQ retry of the AI job) coalesces to
        # the same arq job within keep_result instead of re-transcribing.
        enqueue_attempted = True
        job = await arq_pool.enqueue_job(
            "transcribe_task",
            url,
            content_id,
            transcription_job_id,
            language,
            word_timestamps,
            request_id,
            media_size_bytes,
            _job_id=job_id,
        )
        if job is None:
            # arq returns None when a job with this id already exists (queued,
            # in-progress, or recently completed) — it's already being handled.
            if dedupe_id is not None:
                logger.info(
                    "transcribe_job_deduped", job_id=dedupe_id, content_id=content_id
                )
                return JobAcceptedResponse(job_id=dedupe_id)
            raise TranscriptionError("Failed to enqueue transcription job")

        transcribe_jobs_total.labels(state="queued").inc()
        logger.info(
            "transcribe_job_queued",
            job_id=job.job_id,
            content_id=content_id,
            url_host=safe_url_host(url),
        )
        return JobAcceptedResponse(job_id=job.job_id)
    except BaseException as exc:
        if enqueue_attempted:
            await _mark_arq_unavailable(request, arq_pool)
        if not isinstance(exc, asyncio.CancelledError):
            await _mark_transcription_job_failed(
                request,
                transcription_job_id,
                "Failed to enqueue transcription job",
                media_size_bytes,
            )
        raise


@router.get("/transcribe/jobs/{job_id}", response_model=JobStatusResponse)
async def get_transcribe_job(job_id: str, request: Request) -> JobStatusResponse:
    """Poll status / fetch result for an async transcription job."""
    arq_pool = await _get_arq_pool(request)
    if arq_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Async transcription unavailable: Redis (arq) not reachable",
        )

    job = Job(job_id, redis=arq_pool)
    status = await job.status()

    # Map arq's status enum to our public status string.
    status_map: dict[ArqJobStatus, JobStatus] = {
        ArqJobStatus.deferred: "queued",
        ArqJobStatus.queued: "queued",
        ArqJobStatus.in_progress: "in_progress",
        ArqJobStatus.complete: "completed",
        ArqJobStatus.not_found: "not_found",
    }
    public_status = status_map.get(status, "not_found")

    if public_status == "not_found":
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if public_status != "completed":
        return JobStatusResponse(job_id=job_id, status=public_status)

    # Completed — fetch the result. result_info() reads the worker's stored
    # output; for failed jobs that's the exception, which can occasionally
    # carry non-deserializable state. Don't let that bubble as a 500.
    try:
        info = await job.result_info()
    except Exception as exc:
        logger.warning(
            "transcribe_job_result_unreadable",
            job_id=job_id,
            error=str(exc),
        )
        return JobStatusResponse(
            job_id=job_id,
            status="failed",
            error=f"Job result could not be deserialized: {exc}",
        )

    if info is None:
        return JobStatusResponse(job_id=job_id, status="completed")

    if info.success:
        payload: dict[str, Any] = info.result if isinstance(info.result, dict) else {}
        try:
            response = TranscribeResponse.model_validate(payload)
            return JobStatusResponse(job_id=job_id, status="completed", result=response)
        except Exception as exc:
            return JobStatusResponse(
                job_id=job_id,
                status="failed",
                error=f"Result deserialization failed: {exc}",
            )
    else:
        return JobStatusResponse(
            job_id=job_id,
            status="failed",
            error=str(info.result) if info.result else "Job failed",
        )


@router.delete("/transcribe/jobs/{job_id}", response_model=JobCancelResponse)
async def cancel_transcribe_job(job_id: str, request: Request) -> JobCancelResponse:
    """Best-effort async transcription cancellation.

    Queued/deferred ARQ jobs can usually be aborted. In-progress jobs may already
    be inside the hosted provider call, so callers must treat cancellation as
    advisory and keep CMS as the source of truth.
    """
    arq_pool = await _get_arq_pool(request)
    if arq_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Async transcription unavailable: Redis (arq) not reachable",
        )
    job = Job(job_id, redis=arq_pool)
    status = await job.status()
    if status == ArqJobStatus.not_found:
        return JobCancelResponse(
            job_id=job_id,
            status="not_found",
            canceled=False,
            reason="job not found",
        )
    if status == ArqJobStatus.complete:
        return JobCancelResponse(
            job_id=job_id,
            status="completed",
            canceled=False,
            reason="job already completed",
        )
    if status == ArqJobStatus.in_progress:
        return JobCancelResponse(
            job_id=job_id,
            status="in_progress",
            canceled=False,
            reason="job already running; provider call may finish",
        )
    try:
        abort = getattr(job, "abort", None)
        if abort is None:
            return JobCancelResponse(
                job_id=job_id,
                status=str(status),
                canceled=False,
                reason="arq job abort is unavailable",
            )
        # abort(timeout=None) waits FOREVER for the job result — bound it so a
        # cancel request can't hang the handler. The worker honors the abort
        # flag (allow_abort_jobs=True in WorkerSettings); if confirmation
        # doesn't arrive in time, report it as advisory.
        await abort(timeout=5)
        return JobCancelResponse(job_id=job_id, status=str(status), canceled=True)
    except (TimeoutError, asyncio.TimeoutError):
        return JobCancelResponse(
            job_id=job_id,
            status=str(status),
            canceled=False,
            reason="abort requested; confirmation timed out (advisory)",
        )
    except Exception:
        logger.warning(
            "transcribe_job_cancel_failed", job_id=job_id, error_code="abort_failed"
        )
        return JobCancelResponse(
            job_id=job_id,
            status=str(status),
            canceled=False,
            reason="abort failed",
        )
