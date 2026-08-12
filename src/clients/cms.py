"""Trimmed CMS client for Media-Service.

Only exposes the write-back methods Media needs:
- create_transcript (hosted transcription pipeline)
- store_image_embedding (CLIP image embedding write-back)
- health_check / update_status (operational)

Text embedding write-back stays in Enrichment-Service. If Media ever
needs more CMS surface, copy the method shape from Enrichment-Service/
src/clients/cms.py exactly — the patterns are identical by design.
"""

from typing import Any

import httpx

from src.contracts.artifact_coverage import ArtifactCoverageClaim

from src.clients.circuit_breaker import CircuitBreaker
from src.config import Settings
from src.middleware.request_id import current_request_id
from src.utils.logging import get_logger
from src.utils.metrics import cms_writeback_total

logger = get_logger(__name__)


def _is_countable_cms_failure(exc: Exception) -> bool:
    """Only CMS availability failures can open persistence's breaker."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, httpx.HTTPError)


class CMSClient:
    def __init__(self, settings: Settings):
        raw_base_url = settings.CMS_BASE_URL.rstrip("/")
        self.base_url = raw_base_url
        self.public_base_url = (
            raw_base_url.removesuffix("/internal")
            if raw_base_url.endswith("/internal")
            else raw_base_url
        )
        self.token = settings.cms_writeback_token
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
            # CMS /health aggregates this service's readiness. Liveness avoids
            # turning that dependency check into a circular readiness failure.
            resp = await self.client.get(f"{self.public_base_url}/live")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def claim_artifact_coverage(self) -> dict[str, Any] | None:
        result = await self._request(
            "POST",
            "/internal/artifact-coverage/media/claim",
            json={},
            metric_label="claim_artifact_coverage",
        )
        if not result:
            return None
        return ArtifactCoverageClaim.model_validate(result).model_dump(mode="json")

    async def begin_artifact_coverage(self, request_id: str, claim_token: str) -> None:
        await self._request(
            "POST",
            f"/internal/artifact-coverage/media/{request_id}/begin",
            json={"claim_token": claim_token},
            metric_label="begin_artifact_coverage",
        )

    async def heartbeat_artifact_coverage(
        self, request_id: str, claim_token: str
    ) -> None:
        await self._request(
            "POST",
            f"/internal/artifact-coverage/media/{request_id}/heartbeat",
            json={"claim_token": claim_token},
            metric_label="heartbeat_artifact_coverage",
        )

    async def accept_artifact_coverage(
        self, request_id: str, claim_token: str, proof: dict[str, Any]
    ) -> None:
        await self._request(
            "POST",
            f"/internal/artifact-coverage/media/{request_id}/accepted",
            json={"claim_token": claim_token, "proof": proof},
            metric_label="accept_artifact_coverage",
        )

    async def mark_artifact_coverage_uncertain(
        self, request_id: str, claim_token: str
    ) -> None:
        await self._request(
            "POST",
            f"/internal/artifact-coverage/media/{request_id}/uncertain",
            json={"claim_token": claim_token},
            metric_label="uncertain_artifact_coverage",
        )

    async def create_transcript(
        self,
        content_item_id: str,
        full_text: str,
        language: str,
        word_timestamps: list[dict] | None = None,
        summary: str | None = None,
        segments: list[dict] | None = None,
        chapters: list[dict] | None = None,
        source: str | None = None,
        provider: str | None = None,
        transcription_job_id: str | None = None,
        language_probability: float | None = None,
        duration_sec: float | None = None,
        artifact_recovery: dict[str, str] | None = None,
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
        if segments:
            payload["segments"] = segments
        if chapters:
            payload["chapters"] = chapters
        if source:
            payload["source"] = source
        if provider:
            payload["provider"] = provider
        if transcription_job_id:
            payload["transcription_job_id"] = transcription_job_id
        if language_probability is not None:
            payload["language_probability"] = language_probability
        if duration_sec is not None:
            payload["duration_sec"] = duration_sec
        if artifact_recovery:
            payload["artifact_recovery"] = artifact_recovery

        return await self._request(
            "POST",
            "/internal/transcripts",
            json=payload,
            metric_label="create_transcript",
        )

    async def update_transcription_job(
        self,
        job_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._request(
            "PATCH",
            f"/internal/transcription-jobs/{job_id}",
            json=payload,
            metric_label="update_transcription_job",
        )

    async def store_image_embedding(
        self,
        content_id: str,
        embedding: list[float],
        model: str | None = None,
        space_id: str | None = None,
        producer_id: str | None = None,
        artifact_recovery: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Persist a 512-dim CLIP image embedding to content_items.image_embedding.

        model/space_id/producer_id are the immutable vector-space identities
        (stage 10). Sent only when resolved; an unresolved space leaves the row
        unstamped debt rather than stamping a false-stable identity.
        """
        payload: dict[str, Any] = {"embedding": embedding}
        if model:
            payload["model"] = model
        if space_id:
            payload["space_id"] = space_id
        if producer_id:
            payload["producer_id"] = producer_id
        if artifact_recovery:
            payload["artifact_recovery"] = artifact_recovery
        return await self._request(
            "PATCH",
            f"/internal/content-items/{content_id}/image-embedding",
            json=payload,
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
            if resp.status_code == 204:
                return {}
            return resp.json()

        try:
            result = await self.circuit_breaker.execute(
                _do_request, count_failure=_is_countable_cms_failure
            )
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
