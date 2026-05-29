"""CLIP image embedding model wrapper.

Uses sentence-transformers' clip-ViT-B-32. Output is a 512-dim L2-normalized
vector suitable for cosine-similarity search in pgvector.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.utils.logging import get_logger

if TYPE_CHECKING:
    from PIL.Image import Image

logger = get_logger(__name__)

CLIP_MODEL_NAME = "clip-ViT-B-32"
CLIP_DIMENSIONS = 512


class CLIPWrapper:
    def __init__(
        self,
        model_name: str = CLIP_MODEL_NAME,
        cache_folder: str = "./models",
    ) -> None:
        self._model_name = model_name
        self._cache_folder = cache_folder
        self._model = None
        self._dimensions: int = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        logger.info("loading_clip", model_name=self._model_name)
        self._model = SentenceTransformer(
            self._model_name,
            cache_folder=self._cache_folder,
        )
        # Validate dimensions by encoding a small synthetic image so we fail
        # at boot rather than at first user request if the model is broken.
        from PIL import Image

        probe = Image.new("RGB", (224, 224), color="black")
        test_embedding = self._model.encode([probe])
        self._dimensions = len(test_embedding[0])
        logger.info(
            "clip_loaded",
            model_name=self._model_name,
            dimensions=self._dimensions,
        )

    def encode_image(self, image: "Image") -> list[float]:
        """Encode a single PIL image into a normalized 512-dim vector."""
        if self._model is None:
            raise RuntimeError("CLIP model is not loaded. Call load() first.")
        # normalize_embeddings=True so dot product == cosine similarity downstream.
        embedding = self._model.encode([image], normalize_embeddings=True)[0]
        return embedding.tolist()
