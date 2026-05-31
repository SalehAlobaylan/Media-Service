"""arq worker for long-running Media-Service jobs.

Currently handles one job type: transcription. The worker process loads its
own copy of the Whisper model (separate process from the FastAPI API) and
runs jobs from Redis until shut down.

Run via: `arq src.worker.WorkerSettings`
Or via:  `make worker`
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

from arq.connections import RedisSettings
from redis.asyncio import Redis

from src.clients.cms import CMSClient
from src.clients.storage import StorageClient
from src.config import Settings
from src.middleware.request_id import _request_id_ctx
from src.models.manager import ModelManager
from src.services.transcription import TranscriptionService
from src.utils.logging import get_logger, setup_logging
from src.utils.metrics import transcribe_jobs_total

logger = get_logger("media-worker")


async def _startup(ctx: dict[str, Any]) -> None:
    settings = Settings()
    setup_logging(log_level=settings.LOG_LEVEL, json_output=settings.is_production)
    logger.info("worker_starting", env=settings.ENV)

    model_manager = ModelManager(settings)
    cms_client = CMSClient(settings)
    storage_client = StorageClient(settings)

    # The worker only runs transcribe_task (Whisper). Load Whisper only —
    # CLIP (~600 MB) is image-embedding, which is synchronous in the API and
    # never touched here.
    await model_manager.warmup(["whisper"])

    ctx["settings"] = settings
    ctx["model_manager"] = model_manager
    ctx["cms_client"] = cms_client
    ctx["storage_client"] = storage_client
    logger.info(
        "worker_ready",
        whisper_loaded=model_manager.whisper.is_loaded,
        storage=storage_client.is_configured,
    )


async def _shutdown(ctx: dict[str, Any]) -> None:
    cms_client = ctx.get("cms_client")
    if cms_client is not None:
        try:
            await cms_client.close()
        except Exception:
            pass
    logger.info("worker_shutdown")


async def transcribe_task(
    ctx: dict[str, Any],
    audio_path: str | None,
    url: str | None,
    content_id: str | None,
    language: str | None,
    word_timestamps: bool,
    request_id: str | None = None,
    storage_key: str | None = None,
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
    try:
        model_manager: ModelManager = ctx["model_manager"]
        cms_client: CMSClient = ctx["cms_client"]
        service = TranscriptionService(model_manager.whisper, cms_client)

        if not model_manager.whisper.is_loaded:
            transcribe_jobs_total.labels(state="failed").inc()
            raise RuntimeError("Whisper model is not loaded in worker")

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
                    suffix=suffix, prefix="media_async_"
                )
                os.close(fd)
                await storage_client.download_to_path(storage_key, local_download_path)
                response = await service.transcribe_file(
                    local_download_path,
                    content_id=content_id,
                    language=language,
                    word_timestamps=word_timestamps,
                )
            elif url:
                response = await service.transcribe_url(
                    url,
                    content_id=content_id,
                    language=language,
                    word_timestamps=word_timestamps,
                )
            elif audio_path:
                response = await service.transcribe_file(
                    audio_path,
                    content_id=content_id,
                    language=language,
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


def _build_redis_settings() -> RedisSettings:
    """Parse REDIS_URL into arq's RedisSettings, applying ARQ_REDIS_DB."""
    settings = Settings()
    # arq parses host/port/password from RedisSettings; rather than re-implementing
    # URL parsing, use redis-py to crack the URL then re-pack for arq.
    parsed = Redis.from_url(settings.REDIS_URL).connection_pool.connection_kwargs
    return RedisSettings(
        host=parsed.get("host", "localhost"),
        port=parsed.get("port", 6379),
        password=parsed.get("password"),
        database=settings.ARQ_REDIS_DB,
    )


class WorkerSettings:
    """arq picks this up via `arq src.worker.WorkerSettings`."""

    functions = [transcribe_task]
    on_startup = _startup
    on_shutdown = _shutdown
    redis_settings = _build_redis_settings()
    # Jobs are slow (Whisper transcription on CPU). One concurrent job per
    # worker process is the right default — operators add replicas to scale.
    max_jobs = 1
    # Keep finished job results around long enough for clients to poll.
    keep_result = 3600  # 1 hour
    # Refresh the health-check key every 30s (arq default is 3600s). The key's
    # TTL is interval+1s, so a dead worker's key expires within ~31s — this is
    # what makes the admin dashboard's "worker alive" signal timely instead of
    # up to an hour stale.
    health_check_interval = 30
    # Long jobs need long timeouts. Whisper-base on a 90-min podcast is ~10 min.
    job_timeout = 1800  # 30 min
