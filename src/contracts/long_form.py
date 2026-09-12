from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.contracts.artifact_coverage import UUID_PATTERN


class TranscriptionSegmentClaimUnit(BaseModel):
    """Capability-bearing unit returned only by the CMS claim endpoint."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(pattern=UUID_PATTERN)
    generation_id: str = Field(pattern=UUID_PATTERN)
    segment_index: int = Field(ge=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    attempt_count: int = Field(ge=1)
    claim_token: str = Field(pattern=UUID_PATTERN)
    unit_fence_token: str = Field(pattern=UUID_PATTERN)
    fence_token: str = Field(pattern=UUID_PATTERN)
    lease_expires_at: str = Field(min_length=1)


class TranscriptionSegmentClaim(BaseModel):
    model_config = ConfigDict(extra="ignore")

    unit: TranscriptionSegmentClaimUnit
    generation: dict[str, Any]
