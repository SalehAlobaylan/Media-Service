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


class ReadyResponse(BaseModel):
    status: str
    models: dict[str, bool]
    dependencies: dict[str, bool]


class ModelInfoItem(BaseModel):
    name: str
    loaded: bool
    type: str
    dimensions: int | None = None


class ModelsResponse(BaseModel):
    models: list[ModelInfoItem]
