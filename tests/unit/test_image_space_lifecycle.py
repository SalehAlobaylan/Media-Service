from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.manager import ModelManager
from src.services.image_embedding import ImageEmbeddingService


def clip_descriptor(**overrides):
    descriptor = {
        "model": "clip-ViT-B-32",
        "revision": "immutable-revision",
        "dimensions": 512,
        "normalized": True,
        "space_id": "clip-space",
        "producer_id": "clip-producer",
    }
    descriptor.update(overrides)
    return descriptor


@pytest.mark.asyncio
async def test_image_writeback_refuses_unresolved_space() -> None:
    clip = MagicMock()
    clip.model_name = "clip-ViT-B-32"
    clip.dimensions = 512
    clip.space_descriptor.return_value = clip_descriptor(revision="", space_id="")
    cms = AsyncMock()
    service = ImageEmbeddingService(clip, cms)

    status, error = await service._write_back("content-1", [0.1] * 512, clip.space_descriptor())

    assert (status, error) == ("failed", "image_embedding_space_unresolved")
    cms.store_image_embedding.assert_not_awaited()


def test_clip_readiness_requires_immutable_512_dimension_identity() -> None:
    manager = object.__new__(ModelManager)
    manager.clip = MagicMock()
    manager.clip.is_loaded = True
    manager.clip.model_name = "clip-ViT-B-32"
    manager.clip.dimensions = 512
    manager.clip.space_descriptor.return_value = clip_descriptor()
    assert manager.clip_ready is True

    manager.clip.space_descriptor.return_value = clip_descriptor(producer_id="")
    assert manager.clip_ready is False
    manager.clip.dimensions = 256
    assert manager.clip_ready is False
