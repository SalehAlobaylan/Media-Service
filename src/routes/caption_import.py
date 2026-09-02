"""Media-owned import of provider captions."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from src.auth.service_auth import verify_service_token

router = APIRouter(dependencies=[Depends(verify_service_token)])


class CaptionSegment(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str = Field(min_length=1, max_length=20_000)


class CaptionChapter(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    title: str = Field(min_length=1, max_length=500)
    source: Literal["youtube", "derived"] = "youtube"


class CaptionImportRequest(BaseModel):
    content_item_id: UUID
    full_text: str = Field(min_length=1, max_length=2_000_000)
    language: str = Field(min_length=1, max_length=32)
    segments: list[CaptionSegment] = Field(default_factory=list, max_length=100_000)
    chapters: list[CaptionChapter] = Field(default_factory=list, max_length=2_000)
    source: Literal["youtube_human", "youtube_auto"]
    provider: Literal["youtube"] = "youtube"


@router.post("/captions/import")
async def import_caption(payload: CaptionImportRequest, request: Request) -> dict[str, object]:
    if any(segment.end <= segment.start for segment in payload.segments):
        return {"status": "rejected", "error": "caption segment bounds are invalid"}
    if any(chapter.end <= chapter.start for chapter in payload.chapters):
        return {"status": "rejected", "error": "caption chapter bounds are invalid"}
    result = await request.app.state.cms_client.create_transcript(
        content_item_id=str(payload.content_item_id),
        full_text=payload.full_text,
        language=payload.language,
        segments=[segment.model_dump(mode="json") for segment in payload.segments],
        chapters=[chapter.model_dump(mode="json") for chapter in payload.chapters],
        source=payload.source,
        provider=payload.provider,
    )
    return {"status": "ok", "transcript_id": result.get("id")}
