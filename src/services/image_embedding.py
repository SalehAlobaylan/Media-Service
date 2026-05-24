"""Image embedding via CLIP-ViT-B-32.

Accepts either raw image bytes (upload) or a URL (we download). Outputs a
512-dim L2-normalized vector. When content_id is supplied, writes back to
CMS via PATCH /internal/content-items/:id/image-embedding — same shape as
transcribe write-back.
"""
from __future__ import annotations

import asyncio
from io import BytesIO

import httpx
from PIL import Image

from src.clients.cms import CMSClient
from src.models.clip import CLIPWrapper
from src.schemas.embed_image import ImageEmbedResponse
from src.utils.logging import get_logger
from src.utils.metrics import image_embeddings_total

logger = get_logger(__name__)

# Cap downloaded images to ~50 MB so a bogus URL can't OOM the worker.
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
DOWNLOAD_TIMEOUT_SEC = 30


class ImageEmbeddingService:
    def __init__(self, clip: CLIPWrapper, cms_client: CMSClient):
        self.clip = clip
        self.cms_client = cms_client

    async def embed_bytes(
        self,
        image_bytes: bytes,
        content_id: str | None = None,
    ) -> ImageEmbedResponse:
        if not self.clip.is_loaded:
            image_embeddings_total.labels(status="failure").inc()
            raise RuntimeError("CLIP model is not loaded")

        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            image_embeddings_total.labels(status="failure").inc()
            raise ValueError(f"Could not decode image: {exc}") from exc

        # CLIP encoder is CPU/GPU-bound; offload off the event loop.
        vector = await asyncio.to_thread(self.clip.encode_image, image)
        image_embeddings_total.labels(status="success").inc()

        response = ImageEmbedResponse(
            embedding=vector,
            model=self.clip.model_name,
            dimensions=self.clip.dimensions,
        )

        if content_id:
            status, error = await self._write_back(content_id, vector)
            response.write_back_status = status
            response.write_back_error = error

        return response

    async def embed_url(
        self,
        url: str,
        content_id: str | None = None,
    ) -> ImageEmbedResponse:
        image_bytes = await self._download(url)
        return await self.embed_bytes(image_bytes, content_id=content_id)

    async def _download(self, url: str) -> bytes:
        async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT_SEC) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                # Reject early on declared Content-Length over cap.
                content_length = resp.headers.get("content-length")
                if (
                    content_length
                    and content_length.isdigit()
                    and int(content_length) > MAX_DOWNLOAD_BYTES
                ):
                    raise ValueError(
                        f"Image at {url} declares {content_length} bytes, "
                        f"exceeds {MAX_DOWNLOAD_BYTES} cap"
                    )

                buf = BytesIO()
                written = 0
                async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                    written += len(chunk)
                    if written > MAX_DOWNLOAD_BYTES:
                        raise ValueError(
                            f"Image at {url} exceeded {MAX_DOWNLOAD_BYTES} bytes during download"
                        )
                    buf.write(chunk)
                return buf.getvalue()

    async def _write_back(
        self, content_id: str, vector: list[float]
    ) -> tuple[str, str | None]:
        try:
            await self.cms_client.store_image_embedding(content_id, vector)
            logger.info("image_embedding_writeback_complete", content_id=content_id)
            return "ok", None
        except Exception as exc:
            err = str(exc)
            logger.error(
                "image_embedding_writeback_failed",
                content_id=content_id,
                error=err,
            )
            return "failed", err
