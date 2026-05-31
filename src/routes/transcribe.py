import os
import tempfile
from typing import Any, Literal
from uuid import uuid4

from arq.jobs import Job
from arq.jobs import JobStatus as ArqJobStatus
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from src.auth.service_auth import verify_service_token
from src.middleware.error_handler import TranscriptionError
from src.middleware.request_id import current_request_id
from src.schemas.transcribe import TranscribeResponse
from src.services.transcription import TranscriptionService
from src.utils.logging import get_logger
from src.utils.metrics import transcribe_jobs_total, transcriptions_total

logger = get_logger(__name__)
router = APIRouter(dependencies=[Depends(verify_service_token)])

TEMP_DIR = os.environ.get("MEDIA_TEMP_DIR") or tempfile.gettempdir()
UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB


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


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    request: Request,
    audio_file: UploadFile | None = File(None),
    url: str | None = Form(None),
    content_id: str | None = Form(None),
    language: str | None = Form(None),
    word_timestamps: bool = Form(False),
) -> TranscribeResponse:
    settings = request.app.state.settings
    model_manager = request.app.state.model_manager
    cms_client = request.app.state.cms_client
    service = TranscriptionService(model_manager.whisper, cms_client)

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024

    if not model_manager.whisper.is_loaded:
        raise TranscriptionError("Whisper model is not loaded")

    # Fast path: reject oversize uploads via Content-Length before streaming.
    if audio_file is not None:
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) > max_bytes:
            raise TranscriptionError(
                f"Upload exceeds maximum size of {settings.MAX_UPLOAD_MB} MB"
            )

    try:
        if audio_file and audio_file.filename:
            suffix = os.path.splitext(audio_file.filename)[1] or ".mp3"
            fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=TEMP_DIR)
            os.close(fd)  # we'll reopen via _spool_upload_to_disk
            try:
                await _spool_upload_to_disk(audio_file, tmp_path, max_bytes)

                return await service.transcribe_file(
                    tmp_path,
                    content_id=content_id,
                    language=language,
                    word_timestamps=word_timestamps,
                )
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        elif url:
            return await service.transcribe_url(
                url,
                content_id=content_id,
                language=language,
                word_timestamps=word_timestamps,
            )

        else:
            raise TranscriptionError("Provide either 'audio_file' or 'url'")

    except TranscriptionError:
        raise
    except Exception as exc:
        transcriptions_total.labels(
            status="failure", model_size=model_manager.whisper.model_size
        ).inc()
        logger.error("transcription_failed", error=str(exc))
        raise TranscriptionError(f"Transcription failed: {exc}") from exc


# ─── Async transcription endpoints ─────────────────────────────────────


@router.post(
    "/transcribe/jobs",
    response_model=JobAcceptedResponse,
    status_code=202,
)
async def submit_transcribe_job(
    request: Request,
    audio_file: UploadFile | None = File(None),
    url: str | None = Form(None),
    content_id: str | None = Form(None),
    language: str | None = Form(None),
    word_timestamps: bool = Form(False),
) -> JobAcceptedResponse:
    """Enqueue an async transcription job. Use for long-form (>2 min) audio.

    The worker process (separate from this API) consumes from Redis and runs
    the transcription. Poll GET /v1/transcribe/jobs/{id} for status.
    """
    settings = request.app.state.settings
    arq_pool = request.app.state.arq_pool
    storage = request.app.state.storage_client

    if arq_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Async transcription unavailable: Redis (arq) not reachable",
        )

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024

    # Same Content-Length gate as the sync route.
    if audio_file is not None:
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) > max_bytes:
            raise TranscriptionError(
                f"Upload exceeds maximum size of {settings.MAX_UPLOAD_MB} MB"
            )

    # Uploads are streamed to object storage (R2) rather than the API's local
    # disk: queued audio then lives in R2 (not piling up on the server while it
    # waits for a worker), and the separate worker container can fetch it
    # without a shared filesystem. URL-based jobs skip storage entirely — the
    # worker downloads the URL directly.
    storage_key: str | None = None
    try:
        if audio_file and audio_file.filename:
            if not storage.is_configured:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Async file uploads require object storage (S3_* not "
                        "configured). Submit a URL, or configure R2/S3."
                    ),
                )
            suffix = os.path.splitext(audio_file.filename)[1] or ".mp3"
            storage_key = f"transcribe-jobs/{uuid4().hex}{suffix}"
            await storage.upload_fileobj(
                audio_file.file, storage_key, content_type=audio_file.content_type
            )
        elif not url:
            raise TranscriptionError("Provide either 'audio_file' or 'url'")

        # Forward the current request_id into the job so the worker logs line
        # up with this API call when debugging cross-process traces.
        request_id = current_request_id()
        # Deterministic job id when we have a content_id: an at-least-once
        # re-submit (e.g. an upstream BullMQ retry of the AI job) coalesces to
        # the same arq job within keep_result instead of re-transcribing.
        dedupe_id = f"transcribe:{content_id}" if content_id else None
        job = await arq_pool.enqueue_job(
            "transcribe_task",
            None,  # audio_path — unused now that uploads go through storage
            url,
            content_id,
            language,
            word_timestamps,
            request_id,
            storage_key,
            _job_id=dedupe_id,
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
            storage_key=storage_key,
            url=url,
        )
        return JobAcceptedResponse(job_id=job.job_id)
    except (TranscriptionError, HTTPException):
        # Object was uploaded but enqueue failed — remove it so it doesn't orphan.
        if storage_key and storage.is_configured:
            try:
                await storage.delete_object(storage_key)
            except Exception:
                pass
        raise


@router.get("/transcribe/jobs/{job_id}", response_model=JobStatusResponse)
async def get_transcribe_job(job_id: str, request: Request) -> JobStatusResponse:
    """Poll status / fetch result for an async transcription job."""
    arq_pool = request.app.state.arq_pool
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
