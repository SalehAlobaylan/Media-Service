"""arq worker for long-running Media-Service jobs.

Currently handles one job type: transcription. The worker process uses the
configured hosted STT provider and runs jobs from Redis until shut down.

Run via: `arq src.worker.WorkerSettings`
Or via:  `make worker`
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

from redis.asyncio import Redis

from src.clients.cms import CMSClient
from src.clients.storage import StorageClient
from src.clients.safe_fetch import SafeFetchClient
from src.config import Settings
from src.middleware.request_id import _request_id_ctx
from src.models.manager import ModelManager
from src.queue import build_redis_settings
from src.services.transcription import TranscriptionService, provider_error_code
from src.services.workload import WorkloadAdmission
from src.utils.logging import get_logger, setup_logging
from src.utils.metrics import transcribe_jobs_total
from src.utils.tempdir import resolve_media_temp_dir

logger = get_logger("media-worker")


async def _startup(ctx: dict[str, Any]) -> None:
    settings = Settings()
    setup_logging(log_level=settings.LOG_LEVEL, json_output=settings.is_production)
    logger.info("worker_starting", env=settings.ENV)

    config_errors, config_warnings = settings.validate_startup(expected_role="worker")
    for warning in config_warnings:
        logger.warning("config_warning", error=warning)
    if config_errors:
        raise RuntimeError("Refusing worker startup: " + "; ".join(config_errors))

    model_manager = ModelManager(settings)
    cms_client = CMSClient(settings)
    storage_client = StorageClient(settings)
    fetch_client = SafeFetchClient()
    admission = WorkloadAdmission(stt_limit=1, clip_limit=1)
    temp_dir = resolve_media_temp_dir(settings.MEDIA_TEMP_DIR)

    # The worker only runs transcribe_task (STT). Load the active STT engine only
    # — CLIP (~600 MB) is image-embedding, which is synchronous in the API and
    # never touched here. For a hosted engine (Deepgram) this is a no-op.
    await model_manager.warmup(["stt"])
    if not model_manager.stt.is_loaded:
        await cms_client.close()
        raise RuntimeError("Refusing worker startup: STT provider is not ready")

    if not await cms_client.health_check():
        await cms_client.close()
        raise RuntimeError("Refusing worker startup: CMS is not reachable")

    redis = Redis.from_url(settings.REDIS_URL, db=settings.ARQ_REDIS_DB)
    try:
        await redis.ping()
    except Exception as exc:
        await cms_client.close()
        raise RuntimeError("Refusing worker startup: Redis is not reachable") from exc
    finally:
        await redis.aclose()

    ctx["settings"] = settings
    ctx["model_manager"] = model_manager
    ctx["cms_client"] = cms_client
    ctx["storage_client"] = storage_client
    ctx["temp_dir"] = temp_dir
    ctx["fetch_client"] = fetch_client
    ctx["workload_admission"] = admission
    logger.info(
        "worker_ready",
        stt_provider=model_manager.stt.name,
        stt_loaded=model_manager.stt.is_loaded,
        storage=storage_client.is_configured,
    )


async def _shutdown(ctx: dict[str, Any]) -> None:
    cms_client = ctx.get("cms_client")
    fetch_client = ctx.get("fetch_client")
    if cms_client is not None:
        try:
            await cms_client.close()
        except Exception:
            pass
    if fetch_client is not None:
        await fetch_client.aclose()
    model_manager = ctx.get("model_manager")
    if model_manager is not None:
        await model_manager.stt.aclose()
    admission = ctx.get("workload_admission")
    if admission is not None:
        admission.shutdown()
    logger.info("worker_shutdown")


async def transcribe_task(
    ctx: dict[str, Any],
    audio_path: str | None,
    url: str | None,
    content_id: str | None,
    transcription_job_id: str | None,
    language: str | None,
    word_timestamps: bool,
    request_id: str | None = None,
    storage_key: str | None = None,
    media_size_bytes: int | None = None,
) -> dict[str, Any]:
    """Run transcription. Exactly one of (storage_key, url, audio_path) is set.

    On success returns the serialized TranscribeResponse. On error raises so
    arq marks the job as failed (result preserved in Redis).

    - storage_key: the API uploaded the audio to object storage (R2); the
      worker downloads it to its own temp dir, transcribes, then deletes the
      object. This is the path used for file uploads — no shared filesystem.
    - url: the worker downloads the URL directly (no storage involved).
    - audio_path: legacy/co-located path where the worker shares the API's
      filesystem (e.g. the combined-container dev setup).
    """
    # Restore the request-id contextvar so structured logs + outbound headers
    # in this worker process carry the same trace id as the enqueueing API call.
    token = _request_id_ctx.set(request_id) if request_id else None
    storage_client: StorageClient | None = ctx.get("storage_client")
    local_download_path: str | None = None
    succeeded = False
    service: TranscriptionService | None = None
    try:
        model_manager: ModelManager = ctx["model_manager"]
        cms_client: CMSClient = ctx["cms_client"]
        service = TranscriptionService(
            model_manager.stt,
            cms_client,
            temp_dir=ctx.get("temp_dir"),
            fetch_client=ctx.get("fetch_client"),
            max_download_bytes=ctx["settings"].MAX_UPLOAD_MB * 1024 * 1024 if "settings" in ctx else 200 * 1024 * 1024,
            admission=ctx.get("workload_admission"),
        )

        if not model_manager.stt.is_loaded:
            if transcription_job_id:
                await service._update_job(
                    transcription_job_id,
                    {
                        "status": "failed",
                        "provider": model_manager.stt.name,
                        "model": model_manager.stt.model_size,
                        "error_message": "STT engine is not ready in worker",
                        "provider_error_code": "stt_not_ready",
                    },
                )
            transcribe_jobs_total.labels(state="failed").inc()
            raise RuntimeError("STT engine is not ready in worker")

        logger.info(
            "transcribe_task_started",
            job_id=ctx.get("job_id"),
            content_id=content_id,
            has_storage_key=bool(storage_key),
            has_url=bool(url),
            has_audio_path=bool(audio_path),
        )

        try:
            if storage_key:
                if storage_client is None or not storage_client.is_configured:
                    raise RuntimeError(
                        "storage_key given but object storage not configured in worker"
                    )
                suffix = os.path.splitext(storage_key)[1] or ".mp3"
                fd, local_download_path = tempfile.mkstemp(
                    suffix=suffix, prefix="media_async_", dir=ctx.get("temp_dir")
                )
                os.close(fd)
                await storage_client.download_to_path(storage_key, local_download_path)
                response = await service.transcribe_file(
                    local_download_path,
                    content_id=content_id,
                    transcription_job_id=transcription_job_id,
                    language=language,
                    media_size_bytes=media_size_bytes,
                    word_timestamps=word_timestamps,
                )
            elif url:
                response = await service.transcribe_url(
                    url,
                    content_id=content_id,
                    transcription_job_id=transcription_job_id,
                    language=language,
                    media_size_bytes=media_size_bytes,
                    word_timestamps=word_timestamps,
                )
            elif audio_path:
                response = await service.transcribe_file(
                    audio_path,
                    content_id=content_id,
                    transcription_job_id=transcription_job_id,
                    language=language,
                    media_size_bytes=media_size_bytes,
                    word_timestamps=word_timestamps,
                )
            else:
                raise ValueError("Must provide storage_key, url, or audio_path")
            succeeded = True
        finally:
            # Always drop the worker-local temp files (downloaded object +
            # legacy spool). The remote object is handled separately below.
            for path in (local_download_path, audio_path):
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

        # Delete the storage object only on success. On failure we leave it so
        # an arq retry can re-download; orphans from terminal failures should be
        # swept by a bucket lifecycle/expiry rule.
        if succeeded and storage_key and storage_client and storage_client.is_configured:
            try:
                await storage_client.delete_object(storage_key)
            except Exception as exc:
                logger.warning(
                    "transcribe_storage_cleanup_failed",
                    storage_key=storage_key,
                    error=str(exc),
                )

        transcribe_jobs_total.labels(state="completed").inc()
        logger.info(
            "transcribe_task_completed",
            job_id=ctx.get("job_id"),
            content_id=content_id,
            write_back_status=response.write_back_status,
        )
        return response.model_dump()
    except Exception as exc:
        if transcription_job_id and service is not None:
            await service._update_job(
                transcription_job_id,
                {
                    "status": "failed",
                    "error_message": str(exc),
                    "provider_error_code": provider_error_code(exc),
                    "metadata": {
                        "media_size_bytes": media_size_bytes,
                    } if media_size_bytes is not None else None,
                },
            )
        transcribe_jobs_total.labels(state="failed").inc()
        logger.error(
            "transcribe_task_failed",
            job_id=ctx.get("job_id"),
            content_id=content_id,
            error=str(exc),
        )
        raise
    finally:
        if token is not None:
            _request_id_ctx.reset(token)


def _build_redis_settings():
    """Parse REDIS_URL into arq's RedisSettings, applying ARQ_REDIS_DB."""
    return build_redis_settings(Settings())


class WorkerSettings:
    """arq picks this up via `arq src.worker.WorkerSettings`."""

    functions = [transcribe_task]
    on_startup = _startup
    on_shutdown = _shutdown
    redis_settings = _build_redis_settings()
    # Jobs are network/billing-sensitive. One concurrent job per worker process
    # is the conservative default — operators add replicas to scale.
    max_jobs = 1
    # Keep finished job results around long enough for clients to poll.
    keep_result = 3600  # 1 hour
    # Refresh the health-check key every 30s (arq default is 3600s). The key's
    # TTL is interval+1s, so a dead worker's key expires within ~31s — this is
    # what makes the admin dashboard's "worker alive" signal timely instead of
    # up to an hour stale.
    health_check_interval = 30
    # Required for DELETE /v1/transcribe/jobs/{id} (batch cancel). Without this,
    # arq ignores Job.abort() and queued jobs run anyway.
    allow_abort_jobs = True
    # Long jobs need long timeouts; hosted providers still process long podcasts
    # asynchronously and can take several minutes end to end.
    job_timeout = 1800  # 30 min
