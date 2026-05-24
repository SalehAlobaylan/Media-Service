from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Server
    PORT: int = 5051
    ENV: str = "development"
    LOG_LEVEL: str = "info"
    WORKERS: int = 1

    # Auth — same fallback chain as Enrichment-Service so a single shared
    # SERVICE_AUTH_TOKEN in .env.local works for both services.
    SERVICE_AUTH_TOKEN: str = ""
    MEDIA_SERVICE_TOKEN: str = ""
    CMS_SERVICE_TOKEN: str = ""
    CMS_BASE_URL: str = "http://localhost:8080"

    # Models
    WHISPER_MODEL_SIZE: str = "base"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    CLIP_MODEL: str = "clip-ViT-B-32"
    MODELS_DIR: str = "./models"

    # Media handling — MEDIA_TEMP_DIR empty string means "fall back to
    # system tempdir at runtime" (handled in the routes that need it).
    MEDIA_TEMP_DIR: str = ""
    MAX_UPLOAD_MB: int = 200

    # Circuit Breaker (CMS write-back)
    CB_FAILURE_THRESHOLD: int = 5
    CB_RESET_TIMEOUT_SEC: int = 30
    CB_HALF_OPEN_REQUESTS: int = 3

    # Timeouts
    TRANSCRIBE_TIMEOUT_SEC: int = 600
    CMS_REQUEST_TIMEOUT_SEC: int = 10

    # Redis (arq job queue) — db=2 by convention (db=0 is Aggregation
    # BullMQ, db=1 is Enrichment LLM cache).
    REDIS_URL: str = "redis://localhost:6379"
    ARQ_REDIS_DB: int = 2

    # CORS — CSV of allowed origins. Empty disables CORS; default is
    # wide-open in dev for convenience.
    CORS_ALLOWED_ORIGINS: str = "*"

    @property
    def is_production(self) -> bool:
        return self.ENV == "production"

    @property
    def service_auth_token(self) -> str:
        """Resolution order mirrors Enrichment-Service. The shared
        SERVICE_AUTH_TOKEN in .env.local always wins; the per-service
        var is a per-deployment override; CMS_SERVICE_TOKEN is a final
        fallback for ops who only configured one token."""
        return (
            self.SERVICE_AUTH_TOKEN
            or self.MEDIA_SERVICE_TOKEN
            or self.CMS_SERVICE_TOKEN
        )

    def validate_startup(self) -> tuple[list[str], list[str]]:
        """Return (fatal_errors, warnings).

        In production the auth token must be set, otherwise Aggregation
        can't authenticate against this service. In dev we downgrade to
        a warning so local stacks can boot without explicit token setup.
        """
        errors: list[str] = []
        warnings: list[str] = []

        if not self.service_auth_token:
            msg = (
                "SERVICE_AUTH_TOKEN (or MEDIA_SERVICE_TOKEN / "
                "CMS_SERVICE_TOKEN) must be set"
            )
            (errors if self.is_production else warnings).append(msg)

        return errors, warnings
