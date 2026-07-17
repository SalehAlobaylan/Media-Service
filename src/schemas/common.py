from pydantic import BaseModel


class ErrorResponse(BaseModel):
    error: str
    error_code: str
    retryable: bool
    retry_after_seconds: int | None = None
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


class ModelInfoItem(BaseModel):
    name: str
    loaded: bool
    type: str
    dimensions: int | None = None
    # Immutable vector-space descriptor (stage 10). Present on the clip item;
    # absent/None on stt (no vector space). "" revision/space_id ⇒ not ready.
    revision: str | None = None
    normalized: bool | None = None
    pooling: str | None = None
    space_id: str | None = None
    producer_recipe: str | None = None
    producer_id: str | None = None
    readiness_reason: str | None = None


class ReadyResponse(BaseModel):
    status: str
    models: dict[str, bool]
    dependencies: dict[str, bool]
    # Richer per-model identity (name + dims), additive to the bool `models`
    # map. `type` is the role (stt/clip) — the admin dashboard joins on it
    # to show model names/dims instead of bare load booleans.
    models_detail: list[ModelInfoItem] = []


class QueueStatusResponse(BaseModel):
    """Async-transcription worker + queue health.

    The arq worker is a separate deployment with no HTTP port; the API reports
    on it by observing the shared Redis (db=2) queue. `configured` means a
    Redis DSN exists; `reachable` reports the lifecycle-managed connection.
    """

    configured: bool
    reachable: bool = False
    worker_alive: bool
    queued: int
    # Cumulative worker throughput, parsed from arq's health-check record
    # ("j_complete=.. j_failed=.. j_retried=.. j_ongoing=.."). Zero when no
    # worker has reported yet.
    jobs_complete: int = 0
    jobs_failed: int = 0
    jobs_retried: int = 0
    jobs_ongoing: int = 0
    detail: str | None = None


class ModelsResponse(BaseModel):
    models: list[ModelInfoItem]
