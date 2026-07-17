from fastapi.testclient import TestClient


def test_sync_transcription_rejects_ambiguous_or_missing_source(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    both = client.post(
        "/v1/transcribe",
        headers=auth_headers,
        data={"url": "https://example.com/audio.mp3"},
        files={"audio_file": ("audio.mp3", b"audio", "audio/mpeg")},
    )
    neither = client.post("/v1/transcribe", headers=auth_headers)

    assert both.status_code == 422
    assert neither.status_code == 422
    assert both.json()["error_code"] == "TRANSCRIPTION_FAILED"


def test_async_transcription_validates_source_before_redis_availability(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.post("/v1/transcribe/jobs", headers=auth_headers)
    assert response.status_code == 422
    assert response.json()["error_code"] == "TRANSCRIPTION_FAILED"


def test_transcription_rejects_invalid_typed_metadata_before_queueing(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    invalid_id = client.post(
        "/v1/transcribe/jobs",
        headers=auth_headers,
        data={"url": "https://example.com/audio.mp3", "content_id": "not-a-uuid"},
    )
    invalid_language = client.post(
        "/v1/transcribe/jobs",
        headers=auth_headers,
        data={"url": "https://example.com/audio.mp3", "language": "fr"},
    )
    invalid_size = client.post(
        "/v1/transcribe/jobs",
        headers=auth_headers,
        data={"url": "https://example.com/audio.mp3", "media_size_bytes": "-1"},
    )

    for response in (invalid_id, invalid_language, invalid_size):
        assert response.status_code == 422
        assert response.json()["error_code"] == "TRANSCRIPTION_FAILED"


def test_transcription_rejects_unsupported_uploaded_media_type(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.post(
        "/v1/transcribe",
        headers=auth_headers,
        files={"audio_file": ("payload.exe", b"not-media", "application/octet-stream")},
    )

    assert response.status_code == 422
    assert response.json()["error_code"] == "TRANSCRIPTION_FAILED"


def test_image_embedding_rejects_ambiguous_or_missing_source(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    both = client.post(
        "/v1/embed/image",
        headers=auth_headers,
        data={"url": "https://example.com/image.png"},
        files={"image_file": ("image.png", b"not an image", "image/png")},
    )
    neither = client.post("/v1/embed/image", headers=auth_headers)

    assert both.status_code == 422
    assert neither.status_code == 422
    assert both.json()["error_code"] == "IMAGE_EMBEDDING_FAILED"
