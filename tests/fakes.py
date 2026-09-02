"""Deterministic boundary doubles for Media contract tests.

They deliberately model only the observable Media boundaries.  No fake opens a
network connection, loads a model, or depends on time passing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.models.whisper import TranscribeResult


class RecordingCMS:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def create_transcript(self, **payload: Any) -> dict[str, str]:
        self.calls.append(("create_transcript", payload))
        return {"id": "transcript-1"}

    async def update_transcription_job(
        self, job_id: str, payload: dict[str, Any]
    ) -> dict[str, str]:
        self.calls.append((f"update:{job_id}", payload))
        return {"id": job_id}


class RecordingSTT:
    name = "deepgram"
    source_label = "stt_deepgram"
    model_size = "deepgram:nova-3"
    is_loaded = True

    def __init__(self, result: TranscribeResult | None = None) -> None:
        self.result = result or TranscribeResult(
            text="مرحبا world",
            language="multi",
            language_probability=0.92,
            segments=[{"start": 0.0, "end": 1.0, "text": "مرحبا world"}],
            duration_sec=1.0,
        )
        self.calls: list[tuple[str, str | None, bool]] = []

    def transcribe(
        self, audio_path: str, language: str | None = None, word_timestamps: bool = False
    ) -> TranscribeResult:
        self.calls.append((audio_path, language, word_timestamps))
        return self.result


@dataclass
class EnqueuedJob:
    job_id: str


class RecordingARQ:
    def __init__(self, job: EnqueuedJob | None = None) -> None:
        self.job = job or EnqueuedJob("arq-1")
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def enqueue_job(self, *args: Any, **kwargs: Any) -> EnqueuedJob | None:
        self.calls.append((args, kwargs))
        return self.job
