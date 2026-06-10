import asyncio
import os
import tempfile

import httpx

from src.clients.cms import CMSClient
from src.providers.base import STTProvider
from src.schemas.transcribe import TranscribeResponse, TranscribeSegment
from src.utils.logging import get_logger
from src.utils.metrics import transcription_duration, transcriptions_total

logger = get_logger(__name__)

# MEDIA_TEMP_DIR is the shared spool path between the FastAPI process (where
# uploads land) and the arq worker process (which reads them). Empty / unset
# falls back to the system tempdir so dev works without extra config.
TEMP_DIR = os.environ.get("MEDIA_TEMP_DIR") or tempfile.gettempdir()


class TranscriptionService:
    def __init__(self, stt: STTProvider, cms_client: CMSClient):
        self.stt = stt
        self.cms_client = cms_client

    async def transcribe_file(
        self,
        audio_path: str,
        content_id: str | None = None,
        transcription_job_id: str | None = None,
        language: str | None = None,
        word_timestamps: bool = False,
    ) -> TranscribeResponse:
        model_size = self.stt.model_size

        if transcription_job_id:
            await self._update_job(
                transcription_job_id,
                {
                    "status": "running",
                    "provider": self.stt.name,
                    "model": model_size,
                    "language": language,
                },
            )

        try:
            with transcription_duration.labels(model_size=model_size).time():
                result = await asyncio.to_thread(
                    self.stt.transcribe,
                    audio_path,
                    language=language,
                    word_timestamps=word_timestamps,
                )
        except Exception as exc:
            if transcription_job_id:
                await self._update_job(
                    transcription_job_id,
                    {
                        "status": "failed",
                        "provider": self.stt.name,
                        "model": model_size,
                        "error_message": str(exc),
                    },
                )
            raise

        transcriptions_total.labels(status="success", model_size=model_size).inc()

        response = TranscribeResponse(
            text=result.text,
            language=result.language,
            language_probability=result.language_probability,
            provider=self.stt.name,
            model=model_size,
            segments=[TranscribeSegment(**seg) for seg in result.segments],
            duration_sec=result.duration_sec,
        )

        if content_id:
            status, error = await self._write_back(
                content_id, response, transcription_job_id=transcription_job_id
            )
            response.write_back_status = status
            response.write_back_error = error
        elif transcription_job_id:
            await self._update_job(
                transcription_job_id,
                {
                    "status": "succeeded",
                    "provider": self.stt.name,
                    "model": model_size,
                    "language": response.language,
                    "duration_sec": response.duration_sec,
                    "metadata": {
                        "language_probability": response.language_probability,
                    },
                },
            )

        return response

    async def transcribe_url(
        self,
        url: str,
        content_id: str | None = None,
        transcription_job_id: str | None = None,
        language: str | None = None,
        word_timestamps: bool = False,
    ) -> TranscribeResponse:
        try:
            audio_path = await self._download(url)
        except Exception as exc:
            if transcription_job_id:
                await self._update_job(
                    transcription_job_id,
                    {
                        "status": "failed",
                        "provider": self.stt.name,
                        "model": self.stt.model_size,
                        "error_message": f"media download failed: {exc}",
                    },
                )
            raise
        try:
            return await self.transcribe_file(
                audio_path,
                content_id=content_id,
                transcription_job_id=transcription_job_id,
                language=language,
                word_timestamps=word_timestamps,
            )
        finally:
            self._cleanup(audio_path)

    async def _write_back(
        self,
        content_id: str,
        result: TranscribeResponse,
        transcription_job_id: str | None = None,
    ) -> tuple[str, str | None]:
        """Persist transcript to CMS. Returns (status, error_message).

        Retries once on transient failure. Caller is expected to surface the
        status to the API response so the orchestrator knows whether the
        transcript was actually persisted.
        """
        segments_data = [seg.model_dump() for seg in result.segments]
        if not segments_data:
            segments_data = None

        last_error: str | None = None
        for attempt in range(2):
            try:
                await self.cms_client.create_transcript(
                    content_item_id=content_id,
                    full_text=result.text,
                    language=result.language,
                    word_timestamps=segments_data,
                    segments=segments_data,
                    source=self.stt.source_label,
                    provider=self.stt.name,
                    transcription_job_id=transcription_job_id,
                    language_probability=result.language_probability,
                    duration_sec=result.duration_sec,
                )
                logger.info("transcript_writeback_complete", content_id=content_id)
                return "ok", None
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "transcript_writeback_attempt_failed",
                    content_id=content_id,
                    attempt=attempt + 1,
                    error=last_error,
                )
                if attempt == 0:
                    await asyncio.sleep(1.0)

        logger.error(
            "transcript_writeback_failed",
            content_id=content_id,
            error=last_error,
        )
        if transcription_job_id:
            await self._update_job(
                transcription_job_id,
                {
                    "status": "writeback_failed",
                    "provider": self.stt.name,
                    "model": self.stt.model_size,
                    "language": result.language,
                    "duration_sec": result.duration_sec,
                    "error_message": last_error,
                    "metadata": {
                        "language_probability": result.language_probability,
                    },
                },
            )
        return "failed", last_error

    async def _update_job(self, job_id: str, payload: dict) -> None:
        try:
            await self.cms_client.update_transcription_job(job_id, payload)
        except Exception as exc:
            logger.warning(
                "transcription_job_update_failed",
                job_id=job_id,
                error=str(exc),
            )

    async def _download(self, url: str) -> str:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        suffix = ".mp3"
        if "." in url.split("/")[-1]:
            suffix = "." + url.split("/")[-1].split(".")[-1].split("?")[0]

        fd, path = tempfile.mkstemp(suffix=suffix, dir=TEMP_DIR)
        with os.fdopen(fd, "wb") as f:
            f.write(resp.content)
        return path

    @staticmethod
    def _cleanup(path: str) -> None:
        try:
            os.unlink(path)
        except OSError:
            pass
