from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from src.config import Settings
from src.models.whisper import WhisperWrapper
from src.providers.deepgram_provider import DeepgramProvider
from src.providers.factory import build_stt_provider
from src.services.image_embedding import ImageEmbeddingService


def test_provider_factory_is_explicit_about_missing_and_unknown_provider() -> None:
    settings = Settings(STT_PROVIDER="deepgram", DEEPGRAM_API_KEY="")
    provider = build_stt_provider(settings, WhisperWrapper())
    assert provider.name == "deepgram"
    assert provider.is_loaded is False

    with pytest.raises(RuntimeError, match="Whisper STT is disabled"):
        build_stt_provider(Settings(STT_PROVIDER="whisper"), WhisperWrapper())
    with pytest.raises(RuntimeError, match="Unknown STT_PROVIDER"):
        build_stt_provider(Settings(STT_PROVIDER="other"), WhisperWrapper())


class _DeepgramResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "metadata": {"duration": 2.5},
            "results": {
                "channels": [{"alternatives": [{"transcript": "مرحبا hello", "confidence": 0.87654}]}],
                "utterances": [
                    {"start": 0.0, "end": 1.2, "transcript": "مرحبا"},
                    {"start": 1.2, "end": 2.5, "transcript": "hello"},
                ],
            },
        }


class _DeepgramClient:
    def __init__(self, **_: object) -> None:
        self.call: tuple[tuple, dict] | None = None

    def __enter__(self) -> "_DeepgramClient":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def post(self, *args: object, **kwargs: object) -> _DeepgramResponse:
        self.call = (args, kwargs)
        _DeepgramClient.last_call = self.call
        return _DeepgramResponse()


def test_deepgram_request_and_arabic_code_switched_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.providers.deepgram_provider as module

    monkeypatch.setattr(module.httpx, "Client", _DeepgramClient)
    path = tmp_path / "clip.mp3"
    path.write_bytes(b"audio")
    result = DeepgramProvider("secret", default_language="multi").transcribe(
        str(path), word_timestamps=True
    )

    _, kwargs = _DeepgramClient.last_call
    assert kwargs["headers"]["Authorization"] == "Token secret"
    assert kwargs["headers"]["Content-Type"] == "audio/mpeg"
    assert kwargs["headers"]["Content-Length"] == "5"
    assert kwargs["params"]["language"] == "multi"
    assert result.text == "مرحبا hello"
    assert result.language == "multi"
    assert result.language_probability == 0.8765
    assert result.segments == [
        {"start": 0.0, "end": 1.2, "text": "مرحبا"},
        {"start": 1.2, "end": 2.5, "text": "hello"},
    ]


def test_deepgram_empty_response_has_an_honest_empty_contract() -> None:
    result = DeepgramProvider("secret")._parse({"results": {}, "metadata": {}}, None)

    assert result.text == ""
    assert result.language == "multi"
    assert result.language_probability == 0.0
    assert result.segments == []
    assert result.duration_sec == 0.0


@pytest.mark.asyncio
async def test_image_writeback_forwards_resolved_vector_space_identity() -> None:
    clip = MagicMock()
    clip.is_loaded = True
    clip.model_name = "clip-ViT-B-32"
    clip.dimensions = 512
    clip.space_descriptor.return_value = {
        "model": "clip-ViT-B-32",
        "revision": "immutable-revision",
        "dimensions": 512,
        "normalized": True,
        "space_id": "clip-space",
        "producer_id": "clip-producer",
    }
    cms = MagicMock()
    cms.store_image_embedding = AsyncMock(return_value={"ok": True})
    service = ImageEmbeddingService(clip, cms)

    status, error = await service._write_back(
        "content-1", [0.1] * 512, clip.space_descriptor()
    )

    assert (status, error) == ("ok", None)
    cms.store_image_embedding.assert_awaited_once_with(
        "content-1",
        [0.1] * 512,
        model="clip-ViT-B-32",
        space_id="clip-space",
        producer_id="clip-producer",
        artifact_recovery=None,
        content_stage=None,
    )


@pytest.mark.asyncio
async def test_image_decode_encode_and_writeback_stay_inside_boundary_fakes() -> None:
    clip = MagicMock()
    clip.is_loaded = True
    clip.model_name = "clip-ViT-B-32"
    clip.dimensions = 512
    clip.encode_image.return_value = [0.25] * 512
    clip.space_descriptor.return_value = {
        "model": "clip-ViT-B-32",
        "revision": "immutable-revision",
        "dimensions": 512,
        "normalized": True,
        "space_id": "clip-space",
        "producer_id": "clip-producer",
    }
    cms = MagicMock()
    cms.store_image_embedding = AsyncMock(return_value={"ok": True})
    image = Image.new("RGB", (2, 2), color="white")
    from io import BytesIO

    payload = BytesIO()
    image.save(payload, format="PNG")

    response = await ImageEmbeddingService(clip, cms).embed_bytes(
        payload.getvalue(), content_id="content-1"
    )

    assert response.dimensions == 512
    assert response.write_back_status == "ok"
    clip.encode_image.assert_called_once()
    assert cms.store_image_embedding.await_count == 1
