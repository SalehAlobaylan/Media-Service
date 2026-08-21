from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from src.clients.cms import CMSClient
from src.clients.storage import StorageClient
from src.clients.safe_fetch import SafeFetchClient
from src.config import Settings
from src.middleware.error_handler import (
    CircuitOpenError,
    ImageEmbeddingError,
    TranscriptionError,
    global_error_handler,
)
from src.middleware.body_limit import RequestBodyLimitMiddleware
from src.middleware.logging import LoggingMiddleware
from src.middleware.request_id import RequestIDMiddleware
from src.models.manager import ModelManager
from src.queue import ArqPoolManager
from src.routes import embed_image, health, transcribe
from src.services.workload import WorkloadAdmission
from src.utils.logging import get_logger, setup_logging
from src.utils.tempdir import resolve_media_temp_dir
from src.migration_control import MigrationFenceMiddleware, router as migration_router
from src.migration_control import owner as migration_owner


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = Settings()
    setup_logging(log_level=settings.LOG_LEVEL, json_output=settings.is_production)
    logger = get_logger("media-service")

    logger.info("starting", port=settings.PORT, env=settings.ENV)

    config_errors, config_warnings = settings.validate_startup(expected_role="api")
    for warn in config_warnings:
        logger.warning("config_warning", error=warn)
    if config_errors:
        for err in config_errors:
            logger.error("config_invalid", error=err)
        raise RuntimeError(
            "Refusing to start: invalid configuration — " + "; ".join(config_errors)
        )

    model_manager = ModelManager(settings)
    cms_client = CMSClient(settings)
    storage_client = StorageClient(settings)
    fetch_client = SafeFetchClient()
    admission = WorkloadAdmission()
    temp_dir = resolve_media_temp_dir(settings.MEDIA_TEMP_DIR)

    # Redis is optional for synchronous work. The manager reconnects with
    # bounded backoff, so async submission recovers after a down-at-boot Redis
    # without restarting the API process.
    queue_manager = ArqPoolManager(settings)
    await queue_manager.start()
    arq_pool = queue_manager.pool

    if arq_pool is not None:
        try:
            await migration_owner.restore(arq_pool)
        except Exception as exc:
            logger.error("migration_owner_restore_failed", reason=str(exc))

    if arq_pool is not None and storage_client.is_configured:
        try:
            reclaimed = await transcribe.sweep_orphaned_spools(storage_client, arq_pool)
            if reclaimed:
                logger.info("transcribe_spool_sweep_complete", reclaimed=reclaimed)
        except Exception as exc:
            logger.warning("transcribe_spool_sweep_failed", reason=str(exc))

    await model_manager.warmup()

    app.state.settings = settings
    app.state.model_manager = model_manager
    app.state.cms_client = cms_client
    app.state.arq_pool = arq_pool
    app.state.queue_manager = queue_manager
    app.state.storage_client = storage_client
    app.state.temp_dir = temp_dir
    app.state.fetch_client = fetch_client
    app.state.workload_admission = admission

    logger.info("ready", models=model_manager.is_ready)
    yield

    await cms_client.close()
    await fetch_client.aclose()
    await model_manager.stt.aclose()
    admission.shutdown()
    await queue_manager.aclose()
    logger.info("shutdown_complete")


app = FastAPI(
    title="Media Service",
    description="Media-processing microservice for the Wahb platform "
    "(hosted transcription, CLIP image embedding, future OCR/video).",
    version="1.0.0",
    lifespan=lifespan,
)

# Multipart framing adds a small bounded overhead beyond the media-byte cap.
_body_limit_settings = Settings()
app.add_middleware(
    RequestBodyLimitMiddleware,
    max_bytes=(_body_limit_settings.MAX_UPLOAD_MB * 1024 * 1024) + (1024 * 1024),
)

# CORS — read from env at import time. Defaults are dev-friendly; production
# operators must set CORS_ALLOWED_ORIGINS explicitly or to "" to disable.
_cors_setting = Settings()
_cors_origins = [
    o.strip()
    for o in (_cors_setting.CORS_ALLOWED_ORIGINS or "").split(",")
    if o.strip()
]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        allow_credentials=False,
    )

# Middleware (order matters — outermost first)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(MigrationFenceMiddleware)

# Prometheus metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Routes
app.include_router(health.router)
app.include_router(transcribe.router, prefix="/v1")
app.include_router(embed_image.router, prefix="/v1")
app.include_router(migration_router)

# Error handlers
for exc_class in (CircuitOpenError, TranscriptionError, ImageEmbeddingError):
    app.add_exception_handler(exc_class, global_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(Exception, global_error_handler)  # type: ignore[arg-type]
