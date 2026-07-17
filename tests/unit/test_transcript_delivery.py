from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.schemas.transcribe import TranscribeResponse
from src.services.transcription import (
    TranscriptionService,
    is_retryable_transcript_delivery_error,
)


def http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://cms/internal/transcripts")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("delivery failed", request=request, response=response)


def response() -> TranscribeResponse:
    return TranscribeResponse(
        text="transcript",
        language="ar",
        language_probability=0.9,
        provider="deepgram",
        model="nova-3",
        segments=[],
        duration_sec=0,
    )


@pytest.mark.asyncio
async def test_delivery_does_not_retry_validation_error() -> None:
    cms = AsyncMock()
    cms.create_transcript.side_effect = http_error(422)
    stt = MagicMock(name="deepgram", source_label="stt_deepgram")
    service = TranscriptionService(stt, cms)

    status, _ = await service._write_back("content-1", response())

    assert status == "failed"
    assert cms.create_transcript.await_count == 1


@pytest.mark.asyncio
async def test_delivery_retries_transient_server_error_once() -> None:
    cms = AsyncMock()
    cms.create_transcript.side_effect = [http_error(503), {"id": "transcript-1"}]
    stt = MagicMock(name="deepgram", source_label="stt_deepgram")
    service = TranscriptionService(stt, cms)

    with patch("src.services.transcription.asyncio.sleep", new=AsyncMock()):
        status, error = await service._write_back("content-1", response())

    assert (status, error) == ("ok", None)
    assert cms.create_transcript.await_count == 2


def test_delivery_retry_classifier_only_allows_transport_429_and_5xx() -> None:
    assert not is_retryable_transcript_delivery_error(http_error(400))
    assert not is_retryable_transcript_delivery_error(http_error(409))
    assert is_retryable_transcript_delivery_error(http_error(429))
    assert is_retryable_transcript_delivery_error(http_error(503))
    assert is_retryable_transcript_delivery_error(httpx.ReadTimeout("timeout"))
