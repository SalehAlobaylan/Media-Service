import hmac

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Server
    PORT: int = 5051
    ENV: str = "development"
    LOG_LEVEL: str = "info"
    WORKERS: int = 1

    # Inbound auth is deliberately separate from the token used to mutate CMS.
    # SERVICE_AUTH_TOKEN is retained as a documented legacy inbound alias for
    # local stacks; it never grants CMS access by itself.
    SERVICE_AUTH_TOKEN: str = ""
    MEDIA_SERVICE_TOKEN: str = ""
    MEDIA_SERVICE_TOKEN_PREVIOUS: str = ""
    # Dedicated outbound CMS identity. CMS_SERVICE_TOKEN is retained only for
    # local and migration compatibility.
    CMS_MEDIA_SERVICE_TOKEN: str = ""
    CMS_SERVICE_TOKEN: str = ""
    CMS_BASE_URL: str = "http://localhost:8080"
    MEDIA_ROLE: str = "api"
    # Worker-to-API route for the single local CLIP owner. Production provides
    # the API service URL; local development keeps the loopback default.
    MEDIA_API_BASE_URL: str = "http://127.0.0.1:5051"

    # STT engine selector (boot-time infra selector — Config Discipline). The
    # toggle/budget that govern WHEN STT runs live in the CMS transcription_config
    # table, not here. Default Deepgram Nova-3 (Arabic dialect coverage).
    STT_PROVIDER: str = "deepgram"
    STT_DEFAULT_LANGUAGE: str = (
        "multi"  # 'multi' = code-switching (ar+en); or 'ar'/'en'
    )
    DEEPGRAM_API_KEY: str = ""
    DEEPGRAM_MODEL: str = "nova-3"

    # Models
    WHISPER_MODEL_SIZE: str = "base"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    CLIP_MODEL: str = "clip-ViT-B-32"
    # Immutable CLIP artifact revision (commit digest). Boot-time model config
    # per Config Discipline, NOT a tuning knob. "" ⇒ auto-resolve from the local
    # HF snapshot cache; a bare branch label is not a revision and leaves the
    # image space lifecycle-not-ready (no false-stable identity stamped).
    CLIP_MODEL_REVISION: str = ""
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
    def inbound_service_tokens(self) -> tuple[str, ...]:
        """Current and optional previous inbound tokens, without duplicates."""
        tokens = tuple(
            token
            for token in (
                self.MEDIA_SERVICE_TOKEN or self.SERVICE_AUTH_TOKEN,
                self.MEDIA_SERVICE_TOKEN_PREVIOUS,
            )
            if token
        )
        if tokens or self.is_production:
            return tokens
        # Local-only convenience. Production must never inherit CMS mutation
        # authority for inbound requests.
        return (self.CMS_SERVICE_TOKEN,) if self.CMS_SERVICE_TOKEN else ()

    @property
    def service_auth_token(self) -> str:
        """Primary inbound token retained for existing callers."""
        return self.inbound_service_tokens[0] if self.inbound_service_tokens else ""

    @property
    def cms_writeback_token(self) -> str:
        return self.CMS_MEDIA_SERVICE_TOKEN or self.CMS_SERVICE_TOKEN

    def validate_startup(
        self, expected_role: str | None = None
    ) -> tuple[list[str], list[str]]:
        """Return (fatal_errors, warnings).

        Roles are explicit so an API image cannot accidentally consume jobs and
        a worker cannot report API-grade readiness. Production additionally
        requires separate inbound and CMS credentials. Development may use the
        CMS token as an inbound convenience only when no dedicated token exists.
        """
        errors: list[str] = []
        warnings: list[str] = []

        if self.MEDIA_ROLE not in {"api", "worker"}:
            errors.append("MEDIA_ROLE must be one of: api, worker")
        elif expected_role and self.MEDIA_ROLE != expected_role:
            errors.append(f"MEDIA_ROLE must be {expected_role} for this process")

        primary_inbound = self.MEDIA_SERVICE_TOKEN or self.SERVICE_AUTH_TOKEN
        if not primary_inbound:
            msg = "MEDIA_SERVICE_TOKEN (or legacy SERVICE_AUTH_TOKEN) must be set"
            (errors if self.is_production else warnings).append(msg)

        if self.is_production and not self.CMS_MEDIA_SERVICE_TOKEN:
            errors.append("CMS_MEDIA_SERVICE_TOKEN must be set in production")

        if (
            self.is_production
            and primary_inbound
            and self.CMS_MEDIA_SERVICE_TOKEN
            and hmac.compare_digest(primary_inbound, self.CMS_MEDIA_SERVICE_TOKEN)
        ):
            errors.append(
                "MEDIA_SERVICE_TOKEN/SERVICE_AUTH_TOKEN must differ from CMS_MEDIA_SERVICE_TOKEN in production"
            )

        return errors, warnings
