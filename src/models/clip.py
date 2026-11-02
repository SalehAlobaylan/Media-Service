"""CLIP image embedding model wrapper.

Uses sentence-transformers' clip-ViT-B-32. Output is a 512-dim L2-normalized
vector suitable for cosine-similarity search in pgvector.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from src.utils.logging import get_logger
from src.utils.spaceid import (
    RECIPE_CONTENT_IMAGE,
    compute_producer_id,
    compute_space_id,
)

if TYPE_CHECKING:
    from PIL.Image import Image

logger = get_logger(__name__)

CLIP_MODEL_NAME = "clip-ViT-B-32"
CLIP_DIMENSIONS = 512


def _resolve_hf_revision(model_name: str, cache_folder: str, revision: str = "") -> str:
    """Resolve the selected SentenceTransformer manifest to an immutable commit.

    CLIP stores its transformer config under 0_CLIPModel/, not at repository
    root. The root modules.json manifest belongs to the exact snapshot that
    SentenceTransformer loads and works for this multi-module artifact.
    """
    from huggingface_hub import hf_hub_download

    repo = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
    try:
        config_path = Path(
            hf_hub_download(
                repo,
                "modules.json",
                cache_dir=cache_folder,
                revision=revision or None,
            )
        )
    except (OSError, ValueError):
        return ""
    for parent in config_path.parents:
        if parent.parent.name == "snapshots" and re.fullmatch(
            r"[0-9a-f]{40}", parent.name
        ):
            return parent.name
    return ""


class CLIPWrapper:
    def __init__(
        self,
        model_name: str = CLIP_MODEL_NAME,
        cache_folder: str = "./models",
        revision: str = "",
    ) -> None:
        self._model_name = model_name
        self._cache_folder = cache_folder
        self._revision_selector = revision.strip()
        self._revision = ""
        self._model = None
        self._dimensions: int = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def revision(self) -> str:
        return self._revision

    def space_descriptor(self) -> dict:
        """CLIP image vector-space basis + producer identity (stage 10)."""
        space_id = compute_space_id(
            model=self._model_name,
            revision=self._revision,
            dimensions=self._dimensions,
            normalized=True,
            pooling="clip-visual-proj",
        )
        return {
            "model": self._model_name,
            "revision": self._revision,
            "dimensions": self._dimensions,
            "normalized": True,
            "pooling": "clip-visual-proj",
            "space_id": space_id,
            "producer_recipe": RECIPE_CONTENT_IMAGE,
            "producer_id": compute_producer_id(space_id, RECIPE_CONTENT_IMAGE),
        }

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
        resolved_revision = _resolve_hf_revision(
            self._model_name,
            self._cache_folder,
            self._revision_selector,
        )
        self._model = SentenceTransformer(
            self._model_name,
            cache_folder=self._cache_folder,
            revision=resolved_revision or self._revision_selector or None,
        )
        # Validate dimensions by encoding a small synthetic image so we fail
        # at boot rather than at first user request if the model is broken.
        from PIL import Image

        probe = Image.new("RGB", (224, 224), color="black")
        test_embedding = self._model.encode([probe])
        self._dimensions = len(test_embedding[0])
        self._revision = resolved_revision
        logger.info(
            "clip_loaded",
            model_name=self._model_name,
            dimensions=self._dimensions,
            revision=self._revision or "unresolved",
        )

    def encode_image(self, image: "Image") -> list[float]:
        """Encode a single PIL image into a normalized 512-dim vector."""
        if self._model is None:
            raise RuntimeError("CLIP model is not loaded. Call load() first.")
        # normalize_embeddings=True so dot product == cosine similarity downstream.
        embedding = self._model.encode([image], normalize_embeddings=True)[0]
        return embedding.tolist()
