import asyncio
import inspect
import os
import tempfile

import httpx

from src.clients.cms import CMSClient
from src.clients.safe_fetch import SafeFetchClient
from src.providers.base import STTProvider
from src.services.workload import WorkloadAdmission
from src.schemas.transcribe import TranscribeResponse, TranscribeSegment
from src.utils.logging import get_logger
from src.utils.metrics import transcription_duration, transcriptions_total
from src.utils.url_guard import UnsafeURLError, validate_public_url

logger = get_logger(__name__)


def provider_error_code(exc: Exception) -> str:
    message = str(exc).lower()
    if "api_key" in message or "not set" in message or "not ready" in message:
        return "missing_key"
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)) or "timeout" in message:
        return "timeout"
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 429:
            return "quota_or_rate_limit"
        if status == 413:
            return "payload_too_large"
        if status in (400, 422) and "language" in message:
            return "unsupported_language"
        return f"provider_http_{status}"
    if "payload too large" in message or "request entity too large" in message:
        return "payload_too_large"
    if "quota" in message or "rate limit" in message or "too many requests" in message:
        return "quota_or_rate_limit"
    if "language" in message and ("unsupported" in message or "invalid" in message):
        return "unsupported_language"
    return "provider_error"


def is_retryable_transcript_delivery_error(exc: Exception) -> bool:
    """Whether a CMS transcript delivery can be safely retried by this caller.

    This does not make creation idempotent; the CMS delivery-key contract is
    still required before retries can be relied on for lost responses. Until
    then, never retry deterministic auth, validation, not-found, or conflict
    responses that could only repeat a rejected mutation.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))


class TranscriptionService:
    def __init__(
        self,
        stt: STTProvider,
        cms_client: CMSClient,
        temp_dir: str | None = None,
        fetch_client: SafeFetchClient | None = None,
        max_download_bytes: int = 200 * 1024 * 1024,
        admission: WorkloadAdmission | None = None,
    ):
        self.stt = stt
        self.cms_client = cms_client
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.fetch_client = fetch_client or SafeFetchClient()
        self.max_download_bytes = max_download_bytes
        self.admission = admission or WorkloadAdmission()

    async def transcribe_file(
        self,
        audio_path: str,
        content_id: str | None = None,
        transcription_job_id: str | None = None,
        language: str | None = None,
        media_size_bytes: int | None = None,
        word_timestamps: bool = False,
        artifact_recovery: dict[str, str] | None = None,
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
                    "metadata": {
                        "media_size_bytes": media_size_bytes,
                    }
                    if media_size_bytes is not None
                    else None,
                },
            )

        try:
            async with self.admission.acquire("stt"):
                with transcription_duration.labels(model_size=model_size).time():
                    transcribe_async = getattr(self.stt, "transcribe_async", None)
                    if inspect.iscoroutinefunction(transcribe_async):
                        result = await transcribe_async(
                            audio_path,
                            language=language,
                            word_timestamps=word_timestamps,
                        )
                    else:
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
                        "provider_error_code": provider_error_code(exc),
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
            media_size_bytes=media_size_bytes,
        )

        if content_id:
            status, error = await self._write_back(
                content_id,
                response,
                transcription_job_id=transcription_job_id,
                artifact_recovery=artifact_recovery,
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
                        "media_size_bytes": media_size_bytes,
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
        media_size_bytes: int | None = None,
        word_timestamps: bool = False,
        artifact_recovery: dict[str, str] | None = None,
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
                        "provider_error_code": "media_download_failed",
                        "metadata": {
                            "media_size_bytes": media_size_bytes,
                        }
                        if media_size_bytes is not None
                        else None,
                    },
                )
            raise
        try:
            return await self.transcribe_file(
                audio_path,
                content_id=content_id,
                transcription_job_id=transcription_job_id,
                language=language,
                media_size_bytes=media_size_bytes,
                word_timestamps=word_timestamps,
                artifact_recovery=artifact_recovery,
            )
        finally:
            self._cleanup(audio_path)

    async def _write_back(
        self,
        content_id: str,
        result: TranscribeResponse,
        transcription_job_id: str | None = None,
        artifact_recovery: dict[str, str] | None = None,
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
                    artifact_recovery=artifact_recovery,
                )
                logger.info("transcript_writeback_complete", content_id=content_id)
                return "ok", None
            except Exception as exc:
                last_error = provider_error_code(exc)
                logger.warning(
                    "transcript_writeback_attempt_failed",
                    content_id=content_id,
                    attempt=attempt + 1,
                    error_code=last_error,
                )
                if attempt == 0 and is_retryable_transcript_delivery_error(exc):
                    await asyncio.sleep(1.0)
                    continue
                break

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
                    "writeback_status": "failed",
                    "writeback_error": last_error,
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
        try:
            validate_public_url(url)
        except UnsafeURLError as exc:
            raise ValueError(f"Refusing to fetch unsafe URL: {exc}") from exc
        suffix = ".mp3"
        if "." in url.split("/")[-1]:
            suffix = "." + url.split("/")[-1].split(".")[-1].split("?")[0]

        fd, path = tempfile.mkstemp(suffix=suffix, dir=self.temp_dir)
        os.close(fd)
        try:
            await self.fetch_client.download_to_path(
                url,
                path,
                self.max_download_bytes,
                ("audio/", "video/mp4", "video/webm"),
            )
        except Exception:
            self._cleanup(path)
            raise
        return path

    @staticmethod
    def _cleanup(path: str) -> None:
        try:
            os.unlink(path)
        except OSError:
            pass
