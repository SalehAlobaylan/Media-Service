"""STT provider selection.

Picks the hosted engine from STT_PROVIDER at boot. Whisper fallback is disabled:
missing credentials or unknown providers must surface as visible readiness/job
failures instead of silently changing transcript quality.
"""
from __future__ import annotations

from src.config import Settings
from src.models.whisper import WhisperWrapper
from src.providers.base import STTProvider
from src.providers.deepgram_provider import DeepgramProvider
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_stt_provider(settings: Settings, whisper_wrapper: WhisperWrapper) -> STTProvider:
    provider = (settings.STT_PROVIDER or "deepgram").strip().lower()

    if provider in ("whisper", "faster-whisper"):
        raise RuntimeError("Whisper STT is disabled; configure a hosted STT provider")

    if provider == "deepgram":
        if not settings.DEEPGRAM_API_KEY:
            logger.warning(
                "deepgram_key_missing_stt_not_ready",
                hint="set DEEPGRAM_API_KEY to use Deepgram",
            )
        logger.info("stt_provider_selected", provider="deepgram", model=settings.DEEPGRAM_MODEL)
        return DeepgramProvider(
            api_key=settings.DEEPGRAM_API_KEY,
            model=settings.DEEPGRAM_MODEL,
            default_language=settings.STT_DEFAULT_LANGUAGE,
            timeout_sec=settings.TRANSCRIBE_TIMEOUT_SEC,
        )

    raise RuntimeError(f"Unknown STT_PROVIDER: {provider}")
