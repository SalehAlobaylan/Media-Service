"""Deepgram Nova-3 STT provider (default engine).

Calls Deepgram's pre-recorded REST API directly via httpx (already a dependency
— no SDK). Nova-3 ships a dedicated Arabic model (MSA + dialect variants) with
word-level timestamps; `language=multi` enables code-switching (Arabic+English).

The TranscriptionService runs transcribe() in a worker thread, so a synchronous
httpx call here is fine even for long podcasts.
"""
from __future__ import annotations

import os

import anyio
import httpx

from src.models.whisper import TranscribeResult
from src.providers.base import STTProvider
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"

# Best-effort content-type by extension (Deepgram also sniffs; octet-stream works).
_CONTENT_TYPES = {
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
}


class DeepgramProvider(STTProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "nova-3",
        default_language: str = "multi",
        timeout_sec: int = 600,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._default_language = default_language
        self._timeout_sec = timeout_sec
        self._client = httpx.AsyncClient(timeout=timeout_sec)

    @property
    def name(self) -> str:
        return "deepgram"

    @property
    def source_label(self) -> str:
        return "stt_deepgram"

    @property
    def model_size(self) -> str:
        return f"deepgram:{self._model}"

    @property
    def is_loaded(self) -> bool:
        # Hosted API: "ready" means we have a key to call it with.
        return bool(self._api_key)

    def load(self) -> None:
        if not self._api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set")

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        word_timestamps: bool = False,
    ) -> TranscribeResult:
        if not self._api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set")

        suffix = os.path.splitext(audio_path)[1].lower()
        content_type = _CONTENT_TYPES.get(suffix, "application/octet-stream")

        params = {
            "model": self._model,
            "smart_format": "true",
            "punctuate": "true",
            "utterances": "true",
            "language": language or self._default_language,
        }
        headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": content_type,
            # Stream the file from disk (content=f) to avoid loading large
            # podcasts fully into memory, but set Content-Length explicitly:
            # without it httpx falls back to chunked transfer-encoding, which
            # Deepgram's pre-recorded REST endpoint rejects.
            "Content-Length": str(os.path.getsize(audio_path)),
        }

        with httpx.Client(timeout=self._timeout_sec) as client:
            with open(audio_path, "rb") as f:
                resp = client.post(
                    DEEPGRAM_URL, params=params, headers=headers, content=f
                )
            resp.raise_for_status()
            data = resp.json()

        return self._parse(data, requested_language=language)

    async def transcribe_async(
        self, audio_path: str, language: str | None = None, word_timestamps: bool = False
    ) -> TranscribeResult:
        if not self._api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set")
        suffix = os.path.splitext(audio_path)[1].lower()
        content_type = _CONTENT_TYPES.get(suffix, "application/octet-stream")

        async def stream_file():
            async with await anyio.open_file(audio_path, "rb") as source:
                while chunk := await source.read(1024 * 1024):
                    yield chunk

        response = await self._client.post(
            DEEPGRAM_URL,
            params={
                "model": self._model,
                "smart_format": "true",
                "punctuate": "true",
                "utterances": "true",
                "language": language or self._default_language,
            },
            headers={
                "Authorization": f"Token {self._api_key}",
                "Content-Type": content_type,
                "Content-Length": str(os.path.getsize(audio_path)),
            },
            content=stream_file(),
        )
        response.raise_for_status()
        return self._parse(response.json(), requested_language=language)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _parse(self, data: dict, requested_language: str | None) -> TranscribeResult:
        results = data.get("results", {}) or {}
        metadata = data.get("metadata", {}) or {}

        channels = results.get("channels", []) or []
        alternative = {}
        if channels:
            alts = channels[0].get("alternatives", []) or []
            if alts:
                alternative = alts[0]

        full_text = (alternative.get("transcript") or "").strip()
        confidence = float(alternative.get("confidence") or 0.0)

        # Prefer utterance-level segments; fall back to one whole-text segment.
        segments: list[dict] = []
        for utt in results.get("utterances", []) or []:
            text = (utt.get("transcript") or "").strip()
            if not text:
                continue
            segments.append(
                {
                    "start": round(float(utt.get("start", 0.0)), 3),
                    "end": round(float(utt.get("end", 0.0)), 3),
                    "text": text,
                }
            )
        if not segments and full_text:
            duration = float(metadata.get("duration", 0.0) or 0.0)
            segments = [{"start": 0.0, "end": round(duration, 3), "text": full_text}]

        # Language: Deepgram may detect per-word for multi; surface the requested
        # language, else the model's configured default.
        language = requested_language or self._default_language

        return TranscribeResult(
            text=full_text,
            language=language,
            language_probability=round(confidence, 4),
            segments=segments,
            duration_sec=round(float(metadata.get("duration", 0.0) or 0.0), 3),
        )
