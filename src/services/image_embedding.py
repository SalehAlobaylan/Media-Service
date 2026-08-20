"""Image embedding via CLIP-ViT-B-32.

Accepts either raw image bytes (upload) or a URL (we download). Outputs a
512-dim L2-normalized vector. When content_id is supplied, writes back to
CMS via PATCH /internal/content-items/:id/image-embedding — same shape as
transcribe write-back.
"""

from __future__ import annotations

from io import BytesIO
import warnings

from PIL import Image

from src.clients.cms import CMSClient
from src.clients.safe_fetch import SafeFetchClient
from src.models.clip import CLIPWrapper
from src.services.workload import WorkloadAdmission
from src.schemas.embed_image import ImageEmbedResponse
from src.utils.logging import get_logger
from src.utils.metrics import image_embeddings_total

logger = get_logger(__name__)

# Cap downloaded images to ~50 MB so a bogus URL can't OOM the worker.
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
DOWNLOAD_TIMEOUT_SEC = 30
MAX_IMAGE_PIXELS = 24_000_000
MAX_IMAGE_DIMENSION = 8_192
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}


def _space_descriptor(clip: CLIPWrapper) -> dict:
    factory = getattr(clip, "space_descriptor", None)
    if not callable(factory):
        return {}
    descriptor = factory()
    return descriptor if isinstance(descriptor, dict) else {}


def _is_writable_clip_descriptor(clip: CLIPWrapper, descriptor: dict) -> bool:
    return bool(
        clip.dimensions == 512
        and descriptor.get("model") == clip.model_name
        and descriptor.get("revision")
        and descriptor.get("dimensions") == 512
        and descriptor.get("normalized") is True
        and descriptor.get("space_id")
        and descriptor.get("producer_id")
    )


class ImageEmbeddingService:
    def __init__(
        self,
        clip: CLIPWrapper,
        cms_client: CMSClient,
        fetch_client: SafeFetchClient | None = None,
        admission: WorkloadAdmission | None = None,
    ):
        self.clip = clip
        self.cms_client = cms_client
        self.fetch_client = fetch_client or SafeFetchClient()
        self.admission = admission or WorkloadAdmission()

    async def embed_bytes(
        self,
        image_bytes: bytes,
        content_id: str | None = None,
        artifact_recovery: dict[str, str] | None = None,
        content_stage: dict[str, str] | None = None,
    ) -> ImageEmbedResponse:
        if not self.clip.is_loaded:
            image_embeddings_total.labels(status="failure").inc()
            raise RuntimeError("CLIP model is not loaded")

        try:
            # Treat Pillow's decompression-bomb warning as a hard rejection;
            # validate metadata before RGB conversion allocates a pixel buffer.
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(image_bytes)) as decoded:
                    if decoded.format not in ALLOWED_IMAGE_FORMATS:
                        raise ValueError("Unsupported image format")
                    if getattr(decoded, "n_frames", 1) != 1:
                        raise ValueError("Animated images are not supported")
                    width, height = decoded.size
                    if (
                        width <= 0
                        or height <= 0
                        or width > MAX_IMAGE_DIMENSION
                        or height > MAX_IMAGE_DIMENSION
                        or width * height > MAX_IMAGE_PIXELS
                    ):
                        raise ValueError("Image dimensions exceed the allowed limit")
                    image = decoded.convert("RGB")
        except Exception as exc:
            image_embeddings_total.labels(status="failure").inc()
            raise ValueError(f"Could not decode image: {exc}") from exc

        # CLIP encoder is CPU/GPU-bound; offload off the event loop.
        vector = await self.admission.run_clip(self.clip.encode_image, image)
        image_embeddings_total.labels(status="success").inc()

        descriptor = _space_descriptor(self.clip)
        response = ImageEmbedResponse(
            embedding=vector,
            model=self.clip.model_name,
            dimensions=self.clip.dimensions,
        )

        if content_id:
            status, error = await self._write_back(
                content_id, vector, descriptor, artifact_recovery, content_stage
            )
            response.write_back_status = status
            response.write_back_error = error

        return response

    async def embed_url(
        self,
        url: str,
        content_id: str | None = None,
        artifact_recovery: dict[str, str] | None = None,
        content_stage: dict[str, str] | None = None,
    ) -> ImageEmbedResponse:
        image_bytes = await self._download(url)
        return await self.embed_bytes(
            image_bytes, content_id=content_id, artifact_recovery=artifact_recovery, content_stage=content_stage
        )

    async def _download(self, url: str) -> bytes:
        return await self.fetch_client.get_bytes(
            url, MAX_DOWNLOAD_BYTES, ("image/jpeg", "image/png", "image/webp")
        )

    async def _write_back(
        self,
        content_id: str,
        vector: list[float],
        descriptor: dict | None = None,
        artifact_recovery: dict[str, str] | None = None,
        content_stage: dict[str, str] | None = None,
    ) -> tuple[str, str | None]:
        if descriptor is None or not _is_writable_clip_descriptor(
            self.clip, descriptor
        ):
            logger.warning("image_embedding_writeback_refused", content_id=content_id)
            return "failed", "image_embedding_space_unresolved"
        try:
            await self.cms_client.store_image_embedding(
                content_id,
                vector,
                model=descriptor.get("model") if descriptor else None,
                space_id=descriptor.get("space_id") if descriptor else None,
                producer_id=descriptor.get("producer_id") if descriptor else None,
                artifact_recovery=artifact_recovery,
                content_stage=content_stage,
            )
            logger.info("image_embedding_writeback_complete", content_id=content_id)
            return "ok", None
        except Exception:
            logger.error(
                "image_embedding_writeback_failed",
                content_id=content_id,
                error_code="cms_writeback_failed",
            )
            return "failed", "cms_writeback_failed"
