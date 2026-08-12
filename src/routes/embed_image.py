"""POST /v1/embed/image — CLIP image embedding endpoint.

Accepts either:
  - upload: `image_file` UploadFile
  - URL form field: `url`

Plus optional `content_id` for CMS write-back. Returns a 512-dim vector.
"""

from __future__ import annotations

from uuid import UUID

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
    service = ImageEmbeddingService(
        model_manager.clip,
        cms_client,
        getattr(request.app.state, "fetch_client", None),
        getattr(request.app.state, "workload_admission", None),
    )

    if not model_manager.clip.is_loaded:
        raise ImageEmbeddingError("CLIP model is not loaded")

    has_file = bool(image_file and image_file.filename)
    has_url = bool(url and url.strip())
    if has_file == has_url:
        raise ImageEmbeddingError("Provide exactly one of image_file or url")
    if content_id:
        try:
            UUID(content_id)
        except ValueError as exc:
            raise ImageEmbeddingError("Invalid content_id") from exc

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
        if has_file and image_file is not None:
            data = await image_file.read()
            if len(data) > IMAGE_UPLOAD_MAX_BYTES:
                raise ImageEmbeddingError(
                    f"Image upload exceeds maximum size of "
                    f"{IMAGE_UPLOAD_MAX_BYTES // (1024 * 1024)} MB"
                )
            return await service.embed_bytes(data, content_id=content_id)
        if has_url and url is not None:
            return await service.embed_url(url, content_id=content_id)
    except ImageEmbeddingError:
        raise
    except ValueError as exc:
        # Bad image / oversize download — 4xx, not 5xx.
        image_embeddings_total.labels(status="failure").inc()
        raise ImageEmbeddingError("Invalid image input") from exc
    except Exception as exc:
        image_embeddings_total.labels(status="failure").inc()
        logger.error("image_embedding_failed", error_code="image_embedding_failed")
        raise ImageEmbeddingError("Image embedding failed") from exc
