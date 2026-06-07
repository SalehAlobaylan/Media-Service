"""STT provider selection.

Picks the engine from STT_PROVIDER at boot (a model-selector env var — allowed
by Config Discipline). Defaults to Deepgram Nova-3; falls back to faster-whisper
when whisper is selected or when Deepgram has no API key.
"""
from __future__ import annotations

from src.config import Settings
from src.models.whisper import WhisperWrapper
from src.providers.base import STTProvider
from src.providers.deepgram_provider import DeepgramProvider
from src.providers.whisper_provider import WhisperProvider
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_stt_provider(settings: Settings, whisper_wrapper: WhisperWrapper) -> STTProvider:
    provider = (settings.STT_PROVIDER or "deepgram").strip().lower()

    if provider in ("whisper", "faster-whisper"):
        logger.info("stt_provider_selected", provider="faster-whisper")
        return WhisperProvider(whisper_wrapper)

    if provider == "deepgram":
        if not settings.DEEPGRAM_API_KEY:
            logger.warning(
                "deepgram_key_missing_falling_back_to_whisper",
                hint="set DEEPGRAM_API_KEY to use Deepgram",
            )
            return WhisperProvider(whisper_wrapper)
        logger.info("stt_provider_selected", provider="deepgram", model=settings.DEEPGRAM_MODEL)
        return DeepgramProvider(
            api_key=settings.DEEPGRAM_API_KEY,
            model=settings.DEEPGRAM_MODEL,
            default_language=settings.STT_DEFAULT_LANGUAGE,
            timeout_sec=settings.TRANSCRIBE_TIMEOUT_SEC,
        )

    logger.warning("unknown_stt_provider_falling_back_to_whisper", provider=provider)
    return WhisperProvider(whisper_wrapper)
