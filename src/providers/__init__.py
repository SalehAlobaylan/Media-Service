"""Swappable STT providers (Deepgram default, Whisper fallback)."""
from src.providers.base import STTProvider, TranscribeResult
from src.providers.deepgram_provider import DeepgramProvider
from src.providers.factory import build_stt_provider
from src.providers.whisper_provider import WhisperProvider

__all__ = [
    "STTProvider",
    "TranscribeResult",
    "DeepgramProvider",
    "WhisperProvider",
    "build_stt_provider",
]
