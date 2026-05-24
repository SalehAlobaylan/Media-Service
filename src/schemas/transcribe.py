from typing import Literal

from pydantic import BaseModel

# Possible outcomes for write-back to CMS:
#   - "not_attempted": no content_id supplied (stateless tool mode)
#   - "ok":            transcript created and linked successfully
#   - "failed":        all retries exhausted; transcript was NOT persisted
WriteBackStatus = Literal["not_attempted", "ok", "failed"]


class TranscribeSegment(BaseModel):
    start: float
    end: float
    text: str


class TranscribeResponse(BaseModel):
    text: str
    language: str
    language_probability: float
    segments: list[TranscribeSegment]
    duration_sec: float
    write_back_status: WriteBackStatus = "not_attempted"
    write_back_error: str | None = None
