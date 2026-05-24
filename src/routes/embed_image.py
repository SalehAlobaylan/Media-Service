"""POST /v1/embed/image — CLIP image embedding endpoint.

Accepts either:
  - upload: `image_file` UploadFile
  - URL form field: `url`

Plus optional `content_id` for CMS write-back. Returns a 512-dim vector.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from src.auth.service_auth import verify_service_token
from src.middleware.error_handler import ImageEmbeddingError
from src.schemas.embed_image import ImageEmbedResponse
from src.services.image_embedding import ImageEmbeddingService
from src.utils.logging import get_logger
from src.utils.metrics import image_embeddings_total

logger = get_logger(__name__)
router = APIRouter(dependencies=[Depends(verify_service_token)])

# Tighter cap than transcribe — images shouldn't ever approach 200 MB.
IMAGE_UPLOAD_MAX_BYTES = 50 * 1024 * 1024  # 50 MB


@router.post("/embed/image", response_model=ImageEmbedResponse)
async def embed_image(
    request: Request,
    image_file: UploadFile | None = File(None),
    url: str | None = Form(None),
    content_id: str | None = Form(None),
) -> ImageEmbedResponse:
    model_manager = request.app.state.model_manager
    cms_client = request.app.state.cms_client
    service = ImageEmbeddingService(model_manager.clip, cms_client)

    if not model_manager.clip.is_loaded:
        raise ImageEmbeddingError("CLIP model is not loaded")

    if image_file is not None:
        content_length = request.headers.get("content-length")
        if (
            content_length
            and content_length.isdigit()
            and int(content_length) > IMAGE_UPLOAD_MAX_BYTES
        ):
            raise ImageEmbeddingError(
                f"Image upload exceeds maximum size of "
                f"{IMAGE_UPLOAD_MAX_BYTES // (1024 * 1024)} MB"
            )

    try:
        if image_file is not None and image_file.filename:
            data = await image_file.read()
            if len(data) > IMAGE_UPLOAD_MAX_BYTES:
                raise ImageEmbeddingError(
                    f"Image upload exceeds maximum size of "
                    f"{IMAGE_UPLOAD_MAX_BYTES // (1024 * 1024)} MB"
                )
            return await service.embed_bytes(data, content_id=content_id)
        if url:
            return await service.embed_url(url, content_id=content_id)
        raise ImageEmbeddingError("Provide either 'image_file' or 'url'")
    except ImageEmbeddingError:
        raise
    except ValueError as exc:
        # Bad image / oversize download — 4xx, not 5xx.
        image_embeddings_total.labels(status="failure").inc()
        raise ImageEmbeddingError(str(exc)) from exc
    except Exception as exc:
        image_embeddings_total.labels(status="failure").inc()
        logger.error("image_embedding_failed", error=str(exc))
        raise ImageEmbeddingError(f"Image embedding failed: {exc}") from exc
