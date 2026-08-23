"""arq worker for long-running Media-Service jobs.

Currently handles one job type: transcription. The worker process uses the
configured hosted STT provider and runs jobs from Redis until shut down.

Run via: `arq src.worker.WorkerSettings`
Or via:  `make worker`
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from typing import Any

import httpx

from arq import Retry, cron
from arq.worker import func

from redis.asyncio import Redis

from src.clients.cms import CMSClient
from src.clients.storage import StorageClient
from src.clients.safe_fetch import SafeFetchClient
from src.config import Settings
from src.middleware.request_id import _request_id_ctx
from src.models.manager import ModelManager
from src.queue import build_redis_settings
from src.services.transcription import TranscriptionService, provider_error_code
from src.services.segmented_audio import extract_audio_segment
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

    # The API is the sole CLIP owner. This worker only needs STT and forwards
    # durable image claims to the authenticated API route.
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

    ctx["settings"] = settings
    ctx["model_manager"] = model_manager
    ctx["cms_client"] = cms_client
    ctx["storage_client"] = storage_client
    ctx["temp_dir"] = temp_dir
    ctx["fetch_client"] = fetch_client
    ctx["workload_admission"] = admission
    ctx["migration_redis"] = redis
    logger.info(
        "worker_ready",
        stt_provider=model_manager.stt.name,
        stt_loaded=model_manager.stt.is_loaded,
        storage=storage_client.is_configured,
    )


async def _embed_via_api(
    ctx: dict[str, Any],
    image_url: str,
    content_id: str,
    *,
    content_stage: dict[str, str] | None = None,
    artifact_recovery: dict[str, str] | None = None,
) -> dict[str, Any]:
    settings: Settings = ctx["settings"]
    token = settings.service_auth_token
    if not token:
        raise RuntimeError("Media API token is unavailable for image embedding")
    async with httpx.AsyncClient(timeout=httpx.Timeout(90, connect=10)) as client:
        response = await client.post(
            f"{settings.MEDIA_API_BASE_URL.rstrip('/')}/v1/embed/image",
            data={
                "url": image_url,
                "content_id": content_id,
                "content_stage_json": json.dumps(content_stage) if content_stage else "",
                "artifact_recovery_json": json.dumps(artifact_recovery) if artifact_recovery else "",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Media API returned invalid image embedding response")
    return payload


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
    migration_redis = ctx.get("migration_redis")
    if migration_redis is not None:
        await migration_redis.aclose()
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
    artifact_recovery: dict[str, str] | None = None,
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
    migration_redis = ctx.get("migration_redis")
    if migration_redis is not None and await migration_redis.exists("wahb:database-migration:media-quiesced"):
        raise Retry(defer=5)
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
            max_download_bytes=ctx["settings"].MAX_UPLOAD_MB * 1024 * 1024
            if "settings" in ctx
            else 200 * 1024 * 1024,
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
                    artifact_recovery=artifact_recovery,
                )
            elif url:
                response = await service.transcribe_url(
                    url,
                    content_id=content_id,
                    transcription_job_id=transcription_job_id,
                    language=language,
                    media_size_bytes=media_size_bytes,
                    word_timestamps=word_timestamps,
                    artifact_recovery=artifact_recovery,
                )
            elif audio_path:
                response = await service.transcribe_file(
                    audio_path,
                    content_id=content_id,
                    transcription_job_id=transcription_job_id,
                    language=language,
                    media_size_bytes=media_size_bytes,
                    word_timestamps=word_timestamps,
                    artifact_recovery=artifact_recovery,
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
        if (
            succeeded
            and storage_key
            and storage_client
            and storage_client.is_configured
        ):
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
                    }
                    if media_size_bytes is not None
                    else None,
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


async def artifact_coverage_dispatch_tick(ctx: dict[str, Any]) -> None:
    claim = await ctx["cms_client"].claim_artifact_coverage()
    if not claim:
        return
    await ctx["redis"].enqueue_job(
        "artifact_coverage_task",
        claim,
        _job_id=claim["deterministic_job_id"],
    )


async def content_stage_dispatch_tick(ctx: dict[str, Any]) -> None:
    claim = await ctx["cms_client"].claim_content_stage()
    if not claim:
        return
    await ctx["redis"].enqueue_job(
        "content_stage_transcript_task",
        claim,
        _job_id=claim["deterministic_job_id"],
    )
    await ctx["cms_client"].content_stage_transition(claim, "accepted")


async def content_stage_transcript_task(
    ctx: dict[str, Any], claim: dict[str, Any]
) -> dict[str, str]:
    cms: CMSClient = ctx["cms_client"]
    if claim.get("stage") not in {"pods_transcript", "pods_image_embedding"}:
        await cms.content_stage_transition(
            claim,
            "failed",
            failure_class="unsupported_media_stage",
            summary="Media worker only accepts pods_transcript",
        )
        return {"request_id": claim["request_id"], "state": "failed"}
    await cms.content_stage_transition(claim, "begin")
    stop = asyncio.Event()

    async def heartbeat() -> None:
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=15)
                return
            except TimeoutError:
                await cms.content_stage_transition(claim, "heartbeat")

    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        content = claim["bounded_input"]
        long_form_generation = await ensure_long_form_transcription_generation(ctx, claim, content)
        if long_form_generation is not None:
            # The durable segment dispatcher owns the actual STT work. The
            # stage remains verifiable until CMS observes the merged transcript.
            return {"request_id": claim["request_id"], "state": "verifying"}
        if claim["stage"] == "pods_image_embedding":
            image_url = content.get("thumbnail_url")
            if not image_url:
                await cms.content_stage_transition(
                    claim,
                    "failed",
                    failure_class="invalid_input",
                    summary="Image embedding stage has no CMS-approved thumbnail URL",
                )
                return {"request_id": claim["request_id"], "state": "failed"}
            response = await _embed_via_api(
                ctx, image_url, claim["content_item_id"],
                content_stage=cms.content_stage_correlation(claim),
            )
            if response.get("write_back_status") != "ok":
                await cms.content_stage_transition(
                    claim, "uncertain", summary="Image embedding writeback outcome is unknown"
                )
                return {"request_id": claim["request_id"], "state": "uncertain"}
            return {"request_id": claim["request_id"], "state": "verifying"}
        caption = content.get("caption_artifact")
        if isinstance(caption, dict):
            full_text = caption.get("full_text")
            segments = caption.get("segments")
            if (
                isinstance(full_text, str)
                and full_text.strip()
                and len(full_text) <= 2_000_000
                and isinstance(segments, list)
                and 0 < len(segments) <= 10_000
            ):
                chapters = caption.get("chapters")
                await cms.create_transcript(
                    content_item_id=claim["content_item_id"],
                    full_text=full_text,
                    language=str(caption.get("language") or "und"),
                    segments=segments,
                    chapters=chapters if isinstance(chapters, list) else None,
                    source=str(caption.get("source") or "youtube_auto"),
                    provider=str(caption.get("provider") or "youtube"),
                    content_stage=cms.content_stage_correlation(claim),
                )
                return {"request_id": claim["request_id"], "state": "verifying"}
        url = content.get("playback_url") or content.get("media_url")
        if not url:
            await cms.content_stage_transition(
                claim,
                "failed",
                failure_class="invalid_input",
                summary="Transcript stage has no CMS-approved media URL",
            )
            return {"request_id": claim["request_id"], "state": "failed"}
        service = TranscriptionService(
            ctx["model_manager"].stt,
            cms,
            temp_dir=ctx.get("temp_dir"),
            fetch_client=ctx.get("fetch_client"),
            admission=ctx.get("workload_admission"),
        )
        response = await service.transcribe_url(
            url,
            content_id=claim["content_item_id"],
            word_timestamps=True,
            content_stage=cms.content_stage_correlation(claim),
        )
        if response.write_back_status != "ok":
            await cms.content_stage_transition(
                claim, "uncertain", summary="Transcript writeback outcome is unknown"
            )
            return {"request_id": claim["request_id"], "state": "uncertain"}
        return {"request_id": claim["request_id"], "state": "verifying"}
    except Exception:
        try:
            await cms.content_stage_transition(
                claim, "uncertain", summary="Transcript effect may have started"
            )
        except Exception:
            pass
        raise
    finally:
        stop.set()
        await heartbeat_task


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


async def ensure_long_form_transcription_generation(
    ctx: dict[str, Any], claim: dict[str, Any], content: dict[str, Any]
) -> dict[str, Any] | None:
    duration_raw = content.get("duration_sec")
    try:
        duration_sec = float(duration_raw or 0)
    except (TypeError, ValueError):
        duration_sec = 0
    if duration_sec <= 2400:
        return None
    url = content.get("playback_url") or content.get("media_url")
    if not isinstance(url, str) or not url.strip():
        raise RuntimeError("long-form transcript stage has no CMS-approved playback URL")

    settings: Settings = ctx["settings"]
    cms: CMSClient = ctx["cms_client"]
    content_id = str(claim["content_item_id"])
    tenant_id = str(claim.get("tenant_id") or "default")
    provider = ctx["model_manager"].stt.name
    model = ctx["model_manager"].stt.model_size
    language = str(content.get("content_language") or settings.STT_DEFAULT_LANGUAGE)
    input_digest = _digest(f"long-form-analysis-audio:v1|{url}|{duration_sec:.3f}|{language}|{provider}|{model}")
    existing_manifest_id = content.get("analysis_audio_manifest_id")
    if not isinstance(existing_manifest_id, str) or not existing_manifest_id.strip():
        # A repaired/legacy parent may reuse its source manifest only when the
        # CMS already proves it is audio. Never invent a verified manifest for
        # an arbitrary public URL; Aggregation must first materialize and
        # verify analysis audio for a video source.
        existing_manifest_id = content.get("media_artifact_manifest_id")
        if not isinstance(existing_manifest_id, str) or not existing_manifest_id.strip():
            raise RuntimeError("long-form transcription requires a manifest-owned analysis-audio artifact")
    manifest = await cms.get_artifact_manifest(existing_manifest_id, tenant_id)
    manifest_content_type = str(manifest.get("content_type") or "").lower()
    if not manifest_content_type.startswith("audio/"):
        raise RuntimeError("long-form transcription source manifest is not audio")
    if str(manifest.get("state") or "") not in {"verified", "active"}:
        raise RuntimeError("long-form transcription source manifest is not verified")
    url = str(manifest.get("public_url") or content.get("analysis_audio_url") or url)
    if not url.strip():
        raise RuntimeError("long-form transcription manifest has no public URL")
    input_digest = _digest(f"long-form-analysis-audio:v1|{url}|{duration_sec:.3f}|{language}|{provider}|{model}")

    segment_length_ms = 15 * 60 * 1000
    overlap_ms = 5_000
    total_ms = max(1, int(duration_sec * 1000))
    segments: list[dict[str, Any]] = []
    index = 0
    target_start = 0
    while target_start < total_ms:
        target_end = min(total_ms, target_start + segment_length_ms)
        start_ms = max(0, target_start - (overlap_ms if target_start else 0))
        segment_digest = _digest(f"{input_digest}|{index}|{start_ms}|{target_end}|{overlap_ms}")
        segments.append({
            "start_ms": start_ms,
            "end_ms": target_end,
            "overlap_ms": target_start - start_ms,
            "source_digest": input_digest,
            "segment_digest": segment_digest,
            "artifact_manifest_id": manifest["id"],
        })
        index += 1
        target_start = target_end
    return await cms.create_transcription_generation({
        "tenant_id": tenant_id,
        "content_item_id": content_id,
        "transcription_job_id": claim["request_id"],
        "input_digest": input_digest,
        "analysis_audio_manifest_id": manifest["id"],
        "provider": provider,
        "model": model,
        "language": language,
        "content_stage": cms.content_stage_correlation(claim),
        "segments": segments,
    })


async def transcription_segment_dispatch_tick(ctx: dict[str, Any]) -> None:
    claim = await ctx["cms_client"].claim_transcription_segment()
    if not claim:
        return
    unit = claim.get("unit") or {}
    unit_id = str(unit.get("id") or "")
    attempt = str(unit.get("attempt_count") or "0")
    if not unit_id:
        return
    await ctx["redis"].enqueue_job(
        "transcription_segment_task",
        claim,
        _job_id=f"transcription-segment-{unit_id}-{attempt}",
    )


async def transcription_segment_task(ctx: dict[str, Any], claim: dict[str, Any]) -> dict[str, str]:
    cms: CMSClient = ctx["cms_client"]
    unit = claim.get("unit") or {}
    generation = claim.get("generation") or {}
    unit_id = str(unit["id"])
    claim_token = str(unit["claim_token"])
    manifest_id = unit.get("artifact_manifest_id") or generation.get("analysis_audio_manifest_id")
    if not manifest_id:
        await cms.transition_transcription_segment(unit_id, "failed", {
            "claim_token": claim_token,
            "failure_class": "missing_analysis_audio_manifest",
            "summary": "segment has no analysis-audio manifest",
        })
        return {"segment_id": unit_id, "state": "failed"}
    manifest = await cms.get_artifact_manifest(str(manifest_id), str(generation.get("tenant_id") or "default"))
    url = manifest.get("public_url")
    if not isinstance(url, str) or not url.strip():
        await cms.transition_transcription_segment(unit_id, "failed", {
            "claim_token": claim_token,
            "failure_class": "missing_analysis_audio_url",
            "summary": "analysis-audio manifest has no public URL",
        })
        return {"segment_id": unit_id, "state": "failed"}

    stop = asyncio.Event()

    async def heartbeat() -> None:
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=15)
                return
            except TimeoutError:
                await cms.heartbeat_transcription_segment(unit_id, claim_token)

    heartbeat_task = asyncio.create_task(heartbeat())
    local_path: str | None = None
    try:
        start_ms = int(unit["start_ms"])
        end_ms = int(unit["end_ms"])
        fd, local_path = tempfile.mkstemp(suffix=".wav", prefix=f"transcript-segment-{unit_id}-", dir=ctx.get("temp_dir"))
        os.close(fd)
        await extract_audio_segment(
            url,
            local_path,
            start_ms,
            end_ms,
            timeout_sec=max(1800, int(ctx["settings"].TRANSCRIBE_TIMEOUT_SEC) * 2),
            trusted_cms_source=True,
        )
        service = TranscriptionService(
            ctx["model_manager"].stt,
            cms,
            temp_dir=ctx.get("temp_dir"),
            admission=ctx.get("workload_admission"),
        )
        response = await service.transcribe_file(
            local_path,
            language=str(generation.get("language") or ctx["settings"].STT_DEFAULT_LANGUAGE),
            word_timestamps=True,
        )
        segments = [segment.model_dump() for segment in response.segments]
        await cms.transition_transcription_segment(unit_id, "verified", {
            "claim_token": claim_token,
            "transcript_text": response.text,
            "transcript_segments": segments,
            "summary": "segment transcription verified",
            "artifact_manifest_ids": [str(manifest_id)],
        })
        try:
            await cms.finalize_transcription_generation(str(generation["id"]), _generation_content_stage(generation))
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 409 or "TRANSCRIPTION_INCOMPLETE" not in exc.response.text:
                raise
        return {"segment_id": unit_id, "state": "verified"}
    except Exception as exc:
        try:
            await cms.transition_transcription_segment(unit_id, "failed", {
                "claim_token": claim_token,
                "failure_class": provider_error_code(exc),
                "summary": str(exc)[-1000:],
            })
        except Exception:
            logger.exception("transcription_segment_failure_receipt_failed", segment_id=unit_id)
        raise
    finally:
        stop.set()
        await heartbeat_task
        if local_path and os.path.exists(local_path):
            try:
                os.unlink(local_path)
            except OSError:
                pass


def _generation_content_stage(generation: dict[str, Any]) -> dict[str, str] | None:
    proof = generation.get("terminal_proof")
    if not isinstance(proof, dict):
        return None
    value = proof.get("content_stage")
    if not isinstance(value, dict):
        return None
    required = ("request_id", "attempt_id", "claim_token", "fence_token", "input_fingerprint", "producer_event_id")
    if not all(isinstance(value.get(key), str) and value[key] for key in required):
        return None
    return {key: str(value[key]) for key in required}


async def artifact_coverage_task(
    ctx: dict[str, Any], claim: dict[str, Any]
) -> dict[str, str]:
    cms: CMSClient = ctx["cms_client"]
    request_id = claim["id"]
    token = claim["claim_token"]
    await cms.begin_artifact_coverage(request_id, token)
    heartbeat_stop = asyncio.Event()

    async def heartbeat() -> None:
        while True:
            try:
                await asyncio.wait_for(heartbeat_stop.wait(), timeout=15)
                return
            except TimeoutError:
                await cms.heartbeat_artifact_coverage(request_id, token)

    heartbeat_task = asyncio.create_task(heartbeat())
    correlation = {
        "request_id": request_id,
        "attempt_id": claim["attempt_id"],
        "claim_token": token,
        "fence_token": claim["fence_token"],
        "input_digest": claim["input_digest"],
        "producer_event_id": f"media:{claim['attempt_id']}",
    }
    content = claim["content"]
    artifact = claim["artifact"]
    model_manager: ModelManager = ctx["model_manager"]
    try:
        if artifact == "transcript":
            url = content.get("media_url") or content.get("original_url")
            if not url:
                raise RuntimeError("CMS transcript claim has no media URL")
            service = TranscriptionService(
                model_manager.stt,
                cms,
                temp_dir=ctx.get("temp_dir"),
                fetch_client=ctx.get("fetch_client"),
                admission=ctx.get("workload_admission"),
            )
            response = await service.transcribe_url(
                url,
                content_id=content["id"],
                word_timestamps=True,
                artifact_recovery=correlation,
            )
            if response.write_back_status != "ok":
                raise RuntimeError("transcript recovery was not persisted")
        elif artifact == "image_embedding":
            url = content.get("thumbnail_url") or content.get("media_url")
            if not url:
                raise RuntimeError("CMS image claim has no image URL")
            response = await _embed_via_api(
                ctx, url, content["id"], artifact_recovery=correlation
            )
            if response.get("write_back_status") != "ok":
                raise RuntimeError("image recovery was not persisted")
        else:
            raise RuntimeError("CMS returned an artifact outside Media capability")
        await cms.accept_artifact_coverage(
            request_id, token, {"artifact": artifact, "write_back_status": "ok"}
        )
        return {"request_id": request_id, "state": "verifying"}
    except Exception:
        try:
            await cms.mark_artifact_coverage_uncertain(request_id, token)
        except Exception:
            # CMS lease recovery performs the same transition if the explicit
            # uncertainty receipt cannot be delivered during an outage.
            pass
        raise
    finally:
        heartbeat_stop.set()
        await heartbeat_task


def _build_redis_settings():
    """Parse REDIS_URL into arq's RedisSettings, applying ARQ_REDIS_DB."""
    return build_redis_settings(Settings())


class WorkerSettings:
    """arq picks this up via `arq src.worker.WorkerSettings`."""

    functions = [
        transcribe_task,
        func(artifact_coverage_task, max_tries=1),
        func(content_stage_transcript_task, max_tries=1),
        func(transcription_segment_task, max_tries=1),
    ]
    cron_jobs = [
        cron(
            artifact_coverage_dispatch_tick,
            second={0, 10, 20, 30, 40, 50},
            run_at_startup=True,
        ),
        cron(
            content_stage_dispatch_tick,
            second={5, 15, 25, 35, 45, 55},
            run_at_startup=True,
        ),
        cron(
            transcription_segment_dispatch_tick,
            second={2, 12, 22, 32, 42, 52},
            run_at_startup=True,
        ),
    ]
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
    job_timeout = 3600  # long-form segment extraction + STT
