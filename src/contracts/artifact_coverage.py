from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

UUID_PATTERN = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
DIGEST_PATTERN = r"^[0-9a-f]{64}$"


class ArtifactCoverageContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=UUID_PATTERN)
    type: Literal["VIDEO", "PODCAST"]
    original_url: str | None = None
    media_url: str | None = None
    thumbnail_url: str | None = None
    title: str | None = None
    excerpt: str | None = None
    body_text: str | None = None
    duration_sec: int | None = Field(default=None, gt=0)


class ArtifactCoverageClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=UUID_PATTERN)
    attempt_id: str = Field(pattern=UUID_PATTERN)
    claim_token: str = Field(pattern=UUID_PATTERN)
    fence_token: str = Field(pattern=UUID_PATTERN)
    input_digest: str = Field(pattern=DIGEST_PATTERN)
    artifact: Literal["transcript", "image_embedding"]
    deterministic_job_id: str = Field(pattern=r"^artifact:[0-9a-f]{64}$")
    content: ArtifactCoverageContent
