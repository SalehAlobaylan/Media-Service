from typing import Literal

from pydantic import BaseModel

# Same status enum as transcribe write-back.
WriteBackStatus = Literal["not_attempted", "ok", "failed"]


class ImageEmbedResponse(BaseModel):
    """Response for POST /v1/embed/image.

    The image embedding is a CLIP-ViT-B-32 512-dim vector — independent from
    the text embedding (lives in Enrichment-Service) on the same
    content_item. Both can coexist on different pgvector columns.
    """

    embedding: list[float]
    model: str
    dimensions: int
    # When content_id is supplied, Media-Service writes the vector to CMS via
    # PATCH /internal/content-items/:id/image-embedding. These fields surface
    # silent failures (the body still contains the vector for direct use).
    write_back_status: WriteBackStatus = "not_attempted"
    write_back_error: str | None = None
