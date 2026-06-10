"""Swappable hosted STT providers."""
from src.providers.base import STTProvider, TranscribeResult
from src.providers.deepgram_provider import DeepgramProvider
from src.providers.factory import build_stt_provider

__all__ = [
    "STTProvider",
    "TranscribeResult",
    "DeepgramProvider",
    "build_stt_provider",
]
