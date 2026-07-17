from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from src.services.image_embedding import ImageEmbeddingService


def service() -> ImageEmbeddingService:
    clip = MagicMock()
    clip.is_loaded = True
    return ImageEmbeddingService(clip, AsyncMock())


@pytest.mark.asyncio
async def test_image_embedding_rejects_animated_input_before_clip_inference() -> None:
    first = Image.new("RGB", (2, 2), "red")
    second = Image.new("RGB", (2, 2), "blue")
    raw = BytesIO()
    first.save(raw, format="GIF", save_all=True, append_images=[second])
    embedder = service()

    with pytest.raises(ValueError, match="Unsupported image format|Animated images"):
        await embedder.embed_bytes(raw.getvalue())

    embedder.clip.encode_image.assert_not_called()


@pytest.mark.asyncio
async def test_image_embedding_rejects_oversized_dimensions_before_clip_inference() -> None:
    image = Image.new("RGB", (8193, 1), "black")
    raw = BytesIO()
    image.save(raw, format="PNG")
    embedder = service()

    with pytest.raises(ValueError, match="dimensions exceed"):
        await embedder.embed_bytes(raw.getvalue())

    embedder.clip.encode_image.assert_not_called()
