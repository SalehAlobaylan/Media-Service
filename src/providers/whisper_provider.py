"""faster-whisper STT provider — kept as a disabled offline fallback.

Thin adapter over the existing WhisperWrapper so the established loading /
transcription path is unchanged; it just now sits behind the STTProvider
interface. Selected only when STT_PROVIDER=whisper (or when the configured
hosted provider has no API key).
"""
from __future__ import annotations

from src.models.whisper import TranscribeResult, WhisperWrapper
from src.providers.base import STTProvider


class WhisperProvider(STTProvider):
    def __init__(self, wrapper: WhisperWrapper) -> None:
        self._wrapper = wrapper

    @property
    def name(self) -> str:
        return "faster-whisper"

    @property
    def source_label(self) -> str:
        return "stt_whisper"

    @property
    def model_size(self) -> str:
        return self._wrapper.model_size

    @property
    def is_loaded(self) -> bool:
        return self._wrapper.is_loaded

    @property
    def requires_warmup(self) -> bool:
        return True

    def load(self) -> None:
        self._wrapper.load()

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        word_timestamps: bool = False,
    ) -> TranscribeResult:
        return self._wrapper.transcribe(
            audio_path,
            language=language,
            word_timestamps=word_timestamps,
        )
