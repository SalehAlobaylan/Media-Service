from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import pytest
from arq.jobs import JobStatus
from fastapi.testclient import TestClient

from src.models.whisper import TranscribeResult
from src.routes.transcribe import TranscribeResponse
from src.worker import transcribe_task
from tests.fakes import RecordingARQ, RecordingCMS, RecordingSTT


def test_url_submission_enqueues_only_a_durable_url_and_stable_dedupe_id(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    arq = RecordingARQ()
    client.app.state.arq_pool = arq

    response = client.post(
        "/v1/transcribe/jobs",
        headers=auth_headers,
        data={
            "url": "https://media.example.test/episode.m4a",
            "content_id": "11111111-1111-4111-8111-111111111111",
            "media_size_bytes": "42",
        },
    )

    assert response.status_code == 202, response.text
    assert response.json() == {"job_id": "arq-1", "status": "queued"}
    args, kwargs = arq.calls[0]
    assert args[:6] == (
        "transcribe_task",
        "https://media.example.test/episode.m4a",
        "11111111-1111-4111-8111-111111111111",
        None,
        None,
        False,
    )
    assert isinstance(args[6], str) and args[6]
    assert args[7] == 42
    assert kwargs["_job_id"] == "transcribe:11111111-1111-4111-8111-111111111111"


def test_async_multipart_is_rejected_before_queue_admission(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    arq = RecordingARQ()
    client.app.state.arq_pool = arq
    response = client.post(
        "/v1/transcribe/jobs",
        headers=auth_headers,
        files={"audio_file": ("clip.mp3", b"audio", "audio/mpeg")},
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "ASYNC_TRANSCRIPTION_URL_REQUIRED"
    assert "persist uploads through Aggregation" in response.json()["error"]
    assert arq.calls == []


@dataclass
class _ResultInfo:
    success: bool
    result: object


class _PollingJob:
    status_value = JobStatus.complete
    result_value: _ResultInfo | None = None
    abort_value: object | None = None

    def __init__(self, job_id: str, redis: object) -> None:
        self.job_id = job_id
        self.redis = redis

    async def status(self) -> JobStatus:
        return self.status_value

    async def result_info(self) -> _ResultInfo | None:
        return self.result_value

    async def abort(self, timeout: int) -> object:
        assert timeout == 5
        return self.abort_value


def test_polling_and_cancellation_remain_contract_stable(
    client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.routes.transcribe as route

    monkeypatch.setattr(route, "Job", _PollingJob)
    client.app.state.arq_pool = object()
    _PollingJob.status_value = JobStatus.complete
    _PollingJob.result_value = _ResultInfo(
        success=True,
        result=TranscribeResponse(
            text="done", language="ar", language_probability=0.8,
            provider="deepgram", model="deepgram:nova-3", segments=[], duration_sec=1.2,
        ).model_dump(),
    )
    completed = client.get("/v1/transcribe/jobs/job-1", headers=auth_headers)
    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"

    _PollingJob.status_value = JobStatus.queued
    _PollingJob.abort_value = None
    canceled = client.delete("/v1/transcribe/jobs/job-1", headers=auth_headers)
    assert canceled.status_code == 200
    assert canceled.json()["canceled"] is True


@pytest.mark.asyncio
async def test_worker_url_job_writes_back_without_remote_spool(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    cms = RecordingCMS()
    stt = RecordingSTT(
        TranscribeResult(
            text="contract transcript", language="ar", language_probability=0.91,
            segments=[{"start": 0.0, "end": 2.0, "text": "contract transcript"}], duration_sec=2.0,
        )
    )
    ctx = {
        "job_id": "arq-1", "model_manager": SimpleNamespace(stt=stt),
        "cms_client": cms, "temp_dir": str(tmp_path),
    }
    from src.services.transcription import TranscriptionService

    async def fake_transcribe_url(self, url, **kwargs):
        assert url == "https://media.example.test/episode.m4a"
        source = tmp_path / "source.m4a"
        source.write_bytes(b"audio")
        return await self.transcribe_file(str(source), **kwargs)

    monkeypatch.setattr(TranscriptionService, "transcribe_url", fake_transcribe_url)
    result = await transcribe_task(
        ctx, "https://media.example.test/episode.m4a", "content-1", "cms-job-1",
        "ar", True, media_size_bytes=42,
    )

    assert result["text"] == "contract transcript"
    assert [name for name, _ in cms.calls] == ["update:cms-job-1", "create_transcript"]


@pytest.mark.asyncio
async def test_worker_url_failure_marks_cms_job_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    cms = RecordingCMS()
    stt = RecordingSTT()
    ctx = {"model_manager": SimpleNamespace(stt=stt), "cms_client": cms}
    from src.services.transcription import TranscriptionService

    async def fail_fetch(self, _url, **_kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(TranscriptionService, "transcribe_url", fail_fetch)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await transcribe_task(ctx, "https://media.example.test/episode.m4a", "content-1", "cms-job-1", None, False)
    assert cms.calls[-1][0] == "update:cms-job-1"
    assert cms.calls[-1][1]["status"] == "failed"
