"""Swappable speech-to-text provider interface.

Media-Service was Whisper-only. This abstraction keeps CMS/Console contracts
engine-agnostic while the active hosted provider is selected at boot via
STT_PROVIDER. Adding ElevenLabs Scribe / OpenAI transcription later is one new
subclass + a factory branch.

All providers return the shared TranscribeResult so the TranscriptionService and
CMS write-back are engine-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.whisper import TranscribeResult

__all__ = ["STTProvider", "TranscribeResult"]


class STTProvider(ABC):
    """Common interface every STT engine implements."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Concrete engine name, e.g. 'deepgram'. Stored as Transcript.provider."""

    @property
    @abstractmethod
    def source_label(self) -> str:
        """Transcript provenance written to CMS, e.g. 'stt_deepgram'.
        CMS maps this to caption_state=stt_done."""

    @property
    @abstractmethod
    def model_size(self) -> str:
        """Engine identifier used for metrics labels + /ready display."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the engine is ready to transcribe (local model loaded, or API
        key present for hosted engines)."""

    @property
    def requires_warmup(self) -> bool:
        """True for engines that load a local model. Hosted API engines return
        False so the worker/API skip local model warmup."""
        return False

    def load(self) -> None:
        """Prepare the engine. No-op for hosted-API engines; loads the model for
        local engines. Safe to call multiple times."""
        return None

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        word_timestamps: bool = False,
    ) -> TranscribeResult:
        """Transcribe a local audio file. Synchronous — the service runs it in a
        thread. Returns the shared TranscribeResult."""

    async def transcribe_async(
        self, audio_path: str, language: str | None = None, word_timestamps: bool = False
    ) -> TranscribeResult:
        """Hosted providers override this; legacy local providers remain sync."""
        import asyncio

        return await asyncio.to_thread(
            self.transcribe, audio_path, language=language, word_timestamps=word_timestamps
        )

    async def aclose(self) -> None:
        return None
