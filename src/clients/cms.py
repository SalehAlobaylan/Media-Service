"""Trimmed CMS client for Media-Service.

Only exposes the write-back methods Media needs:
- create_transcript / link_transcript (Whisper transcription pipeline)
- store_image_embedding (CLIP image embedding write-back)
- health_check / update_status (operational)

Text embedding write-back stays in Enrichment-Service. If Media ever
needs more CMS surface, copy the method shape from Enrichment-Service/
src/clients/cms.py exactly — the patterns are identical by design.
"""
from typing import Any

import httpx

from src.clients.circuit_breaker import CircuitBreaker
from src.config import Settings
from src.middleware.request_id import current_request_id
from src.utils.logging import get_logger
from src.utils.metrics import cms_writeback_total

logger = get_logger(__name__)


class CMSClient:
    def __init__(self, settings: Settings):
        raw_base_url = settings.CMS_BASE_URL.rstrip("/")
        self.base_url = raw_base_url
        self.public_base_url = (
            raw_base_url.removesuffix("/internal")
            if raw_base_url.endswith("/internal")
            else raw_base_url
        )
        self.token = settings.CMS_SERVICE_TOKEN
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=settings.CB_FAILURE_THRESHOLD,
            reset_timeout_sec=settings.CB_RESET_TIMEOUT_SEC,
            half_open_requests=settings.CB_HALF_OPEN_REQUESTS,
        )
        headers = {
            "Content-Type": "application/json",
            "X-Service-Name": "media-service",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self.client = httpx.AsyncClient(
            timeout=settings.CMS_REQUEST_TIMEOUT_SEC,
            headers=headers,
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def health_check(self) -> bool:
        try:
            resp = await self.client.get(f"{self.public_base_url}/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def create_transcript(
        self,
        content_item_id: str,
        full_text: str,
        language: str,
        word_timestamps: list[dict] | None = None,
        summary: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "content_item_id": content_item_id,
            "full_text": full_text,
            "language": language,
        }
        if word_timestamps:
            payload["word_timestamps"] = word_timestamps
        if summary:
            payload["summary"] = summary

        return await self._request(
            "POST",
            "/internal/transcripts",
            json=payload,
            metric_label="create_transcript",
        )

    async def link_transcript(self, content_id: str, transcript_id: str) -> dict[str, Any]:
        return await self._request(
            "PATCH",
            f"/internal/content-items/{content_id}/transcript",
            json={"transcript_id": transcript_id},
            metric_label="link_transcript",
        )

    async def store_image_embedding(
        self,
        content_id: str,
        embedding: list[float],
    ) -> dict[str, Any]:
        """Persist a 512-dim CLIP image embedding to content_items.image_embedding."""
        return await self._request(
            "PATCH",
            f"/internal/content-items/{content_id}/image-embedding",
            json={"embedding": embedding},
            metric_label="store_image_embedding",
        )

    async def update_status(
        self,
        content_id: str,
        status: str,
        failure_reason: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": status}
        if failure_reason:
            payload["failure_reason"] = failure_reason
        return await self._request(
            "PATCH",
            f"/internal/content-items/{content_id}/status",
            json=payload,
            metric_label="update_status",
        )

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        metric_label: str = "unknown",
    ) -> dict[str, Any]:
        async def _do_request() -> dict[str, Any]:
            url = self._build_url(path)
            request_id = current_request_id()
            headers = {"X-Request-ID": request_id} if request_id else None
            resp = await self.client.request(method, url, json=json, headers=headers)
            resp.raise_for_status()
            return resp.json()

        try:
            result = await self.circuit_breaker.execute(_do_request)
            cms_writeback_total.labels(endpoint=metric_label, status="success").inc()
            return result
        except Exception as exc:
            cms_writeback_total.labels(endpoint=metric_label, status="failure").inc()
            logger.error(
                "cms_request_failed",
                method=method,
                path=path,
                error=str(exc),
            )
            raise

    def _build_url(self, path: str) -> str:
        if self.base_url.endswith("/internal") and path.startswith("/internal/"):
            return f"{self.base_url}{path.removeprefix('/internal')}"
        return f"{self.base_url}{path}"
