import re
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request, Response

from src.auth.service_auth import verify_service_token
from src.schemas.common import (
    HealthResponse,
    ModelInfoItem,
    ModelsResponse,
    QueueStatusResponse,
    ReadyResponse,
)

router = APIRouter()

VERSION = "1.0.0"


def _model_items(model_manager) -> list[ModelInfoItem]:
    """Per-model identity (name + dims), shared by /ready and /v1/models."""
    return [
        ModelInfoItem(
            name=model_manager.stt.model_size,
            loaded=model_manager.stt.is_loaded,
            type="stt",
            dimensions=None,
            readiness_reason=model_manager.stt_readiness_reason,
        ),
        _clip_item(model_manager.clip, model_manager.clip_readiness_reason),
    ]


def _clip_item(clip, readiness_reason: str | None = None) -> ModelInfoItem:
    raw_desc = clip.space_descriptor() if clip.is_loaded else {}
    desc = raw_desc if isinstance(raw_desc, dict) else {}
    return ModelInfoItem(
        name=clip.model_name,
        loaded=clip.is_loaded,
        type="clip",
        dimensions=clip.dimensions if clip.is_loaded else None,
        revision=desc.get("revision"),
        normalized=desc.get("normalized"),
        pooling=desc.get("pooling"),
        space_id=desc.get("space_id"),
        producer_recipe=desc.get("producer_recipe"),
        producer_id=desc.get("producer_id"),
        readiness_reason=readiness_reason,
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=VERSION,
    )


@router.get("/ready", response_model=ReadyResponse)
async def ready(request: Request, response: Response) -> ReadyResponse:
    model_manager = request.app.state.model_manager
    cms_client = request.app.state.cms_client

    models_status = model_manager.is_ready
    cms_reachable = await cms_client.health_check()

    all_ready = model_manager.all_ready and cms_reachable

    if not all_ready:
        response.status_code = 503
        response.headers["Retry-After"] = "10"

    return ReadyResponse(
        status="ok" if all_ready else "not_ready",
        models=models_status,
        dependencies={"cms": cms_reachable},
        models_detail=_model_items(model_manager),
    )


# arq's default queue (WorkerSettings doesn't override queue_name) is the
# sorted set "arq:queue"; the worker writes "<queue_name>:health-check" every
# health_check_interval while running. The API observes both to report on the
# separate worker deployment.
_ARQ_QUEUE_KEY = "arq:queue"
_ARQ_HEALTH_KEY = b"arq:queue:health-check"
# arq writes "<ts> j_complete=N j_failed=N j_retried=N j_ongoing=N queued=N".
_HEALTH_COUNTS_RE = re.compile(
    r"j_complete=(\d+)\s+j_failed=(\d+)\s+j_retried=(\d+)\s+j_ongoing=(\d+)"
)


@router.get("/health/queue", response_model=QueueStatusResponse)
async def queue_status(request: Request) -> QueueStatusResponse:
    """Async-transcription worker + queue depth + throughput.

    Unauthenticated like /health and /ready — exposes only job counts and a
    worker-alive flag (no sensitive data). The worker has no HTTP listener, so
    this is the only window into it; the API reads the shared Redis (db=2)
    queue via its arq pool. `worker_alive` is the presence of arq's health-check
    key, which the worker refreshes every WorkerSettings.health_check_interval.
    """
    manager = getattr(request.app.state, "queue_manager", None)
    pool = await manager.get_pool() if manager is not None else getattr(request.app.state, "arq_pool", None)
    if pool is None:
        settings = request.app.state.settings
        return QueueStatusResponse(
            configured=bool(settings.REDIS_URL),
            reachable=False,
            worker_alive=False,
            queued=0,
        )

    reachable = True
    try:
        queued = int(await pool.zcard(_ARQ_QUEUE_KEY) or 0)
    except Exception:
        reachable = False
        if manager is not None:
            await manager.mark_unavailable(pool)
        queued = 0

    try:
        health = await pool.get(_ARQ_HEALTH_KEY)
    except Exception:
        reachable = False
        if manager is not None:
            await manager.mark_unavailable(pool)
        health = None
    detail = health.decode() if isinstance(health, bytes) else health

    complete = failed = retried = ongoing = 0
    if detail:
        m = _HEALTH_COUNTS_RE.search(detail)
        if m:
            complete, failed, retried, ongoing = (int(g) for g in m.groups())

    return QueueStatusResponse(
        configured=True,
        reachable=reachable,
        worker_alive=health is not None,
        queued=queued,
        jobs_complete=complete,
        jobs_failed=failed,
        jobs_retried=retried,
        jobs_ongoing=ongoing,
        detail=detail,
    )


@router.get(
    "/v1/models",
    response_model=ModelsResponse,
    dependencies=[Depends(verify_service_token)],
)
async def models(request: Request) -> ModelsResponse:
    return ModelsResponse(models=_model_items(request.app.state.model_manager))
