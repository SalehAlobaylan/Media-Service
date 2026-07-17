from fastapi import Request
import pytest

from src.middleware.error_handler import ImageEmbeddingError, TranscriptionError, global_error_handler


async def _request() -> Request:
    return Request({"type": "http", "method": "POST", "path": "/v1/transcribe", "headers": []})


@pytest.mark.asyncio
async def test_known_media_errors_do_not_echo_signed_urls() -> None:
    secret = "https://cdn.example/audio.mp3?signature=secret-value"
    response = await global_error_handler(await _request(), TranscriptionError(secret))
    assert response.status_code == 422
    assert secret.encode() not in response.body
    assert b"TRANSCRIPTION_FAILED" in response.body


@pytest.mark.asyncio
async def test_image_errors_do_not_echo_local_paths() -> None:
    secret = "/private/tmp/customer-image.png"
    response = await global_error_handler(await _request(), ImageEmbeddingError(secret))
    assert response.status_code == 422
    assert secret.encode() not in response.body
    assert b"IMAGE_EMBEDDING_FAILED" in response.body
