from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from arq.jobs import JobStatus
from fastapi.testclient import TestClient

from src.models.whisper import TranscribeResult
from src.schemas.transcribe import TranscribeResponse
from src.worker import transcribe_task
from src.routes.transcribe import sweep_orphaned_spools
from tests.fakes import RecordingARQ, RecordingCMS, RecordingSTT, RecordingStorage


def test_url_submission_enqueues_only_the_url_and_stable_dedupe_id(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    arq = RecordingARQ()
    client.app.state.arq_pool = arq
    client.app.state.storage_client = RecordingStorage()

    response = client.post(
        "/v1/transcribe/jobs",
        headers=auth_headers,
        data={"url": "https://media.example.test/episode.mp3", "content_id": "11111111-1111-4111-8111-111111111111"},
    )

    assert response.status_code == 202, response.text
    assert response.json() == {"job_id": "arq-1", "status": "queued"}
    args, kwargs = arq.calls[0]
    assert args[0] == "transcribe_task"
    assert args[1:4] == (None, "https://media.example.test/episode.mp3", "11111111-1111-4111-8111-111111111111")
    assert args[-2] is None  # URL jobs never create an object-storage key.
    assert kwargs["_job_id"] == "transcribe:11111111-1111-4111-8111-111111111111"


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
        if isinstance(self.result_value, Exception):
            raise self.result_value
        return self.result_value

    async def abort(self, timeout: int) -> object:
        assert timeout == 5
        if isinstance(self.abort_value, Exception):
            raise self.abort_value
        return self.abort_value


def test_poll_completed_result_and_queued_cancel_are_contract_stable(
    client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.routes.transcribe as route

    monkeypatch.setattr(route, "Job", _PollingJob)
    client.app.state.arq_pool = object()
    _PollingJob.status_value = JobStatus.complete
    _PollingJob.result_value = _ResultInfo(
        success=True,
        result=TranscribeResponse(
            text="done",
            language="ar",
            language_probability=0.8,
            provider="deepgram",
            model="deepgram:nova-3",
            segments=[],
            duration_sec=1.2,
        ).model_dump(),
    )
    completed = client.get("/v1/transcribe/jobs/job-1", headers=auth_headers)
    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"
    assert completed.json()["result"]["text"] == "done"

    _PollingJob.status_value = JobStatus.queued
    _PollingJob.abort_value = None
    canceled = client.delete("/v1/transcribe/jobs/job-1", headers=auth_headers)
    assert canceled.status_code == 200
    assert canceled.json()["canceled"] is True
    assert canceled.json()["status"] == str(JobStatus.queued)


def test_poll_failed_and_unreadable_results_are_safe_failures(
    client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.routes.transcribe as route

    monkeypatch.setattr(route, "Job", _PollingJob)
    client.app.state.arq_pool = object()
    _PollingJob.status_value = JobStatus.complete
    _PollingJob.result_value = _ResultInfo(success=False, result="provider failed")
    failed = client.get("/v1/transcribe/jobs/job-2", headers=auth_headers)
    assert failed.status_code == 200
    assert failed.json()["status"] == "failed"
    assert failed.json()["error"] == "provider failed"

    _PollingJob.result_value = RuntimeError("stored object is unreadable")
    unreadable = client.get("/v1/transcribe/jobs/job-2", headers=auth_headers)
    assert unreadable.status_code == 200
    assert unreadable.json()["status"] == "failed"
    assert "could not be deserialized" in unreadable.json()["error"]


def test_confirmed_queued_cancel_reclaims_only_its_recorded_spool_key(
    client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.routes.transcribe as route

    class JobWithSpool(_PollingJob):
        async def info(self):
            return SimpleNamespace(
                function="transcribe_task",
                args=(None, None, "content-1", "cms-job-1", None, False, None, "ignored", "transcribe-jobs/owned.mp3"),
            )

    monkeypatch.setattr(route, "Job", JobWithSpool)
    client.app.state.arq_pool = object()
    client.app.state.storage_client = RecordingStorage()
    JobWithSpool.status_value = JobStatus.queued
    JobWithSpool.abort_value = None

    canceled = client.delete("/v1/transcribe/jobs/job-owned", headers=auth_headers)

    assert canceled.status_code == 200
    assert canceled.json()["canceled"] is True
    assert client.app.state.storage_client.calls == [
        ("delete", "transcribe-jobs/owned.mp3")
    ]


@pytest.mark.asyncio
async def test_spool_sweeper_reclaims_only_old_noncanonical_objects(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.routes.transcribe as route

    class SweepStorage:
        async def list_objects(self, prefix, max_keys):
            assert (prefix, max_keys) == ("transcribe-jobs/", 100)
            old = datetime.now(UTC) - timedelta(minutes=16)
            return [
                ("transcribe-jobs/transcribe:job-1/original.mp3", old),
                ("transcribe-jobs/transcribe:job-1/duplicate.mp3", old),
            ]

        def __init__(self):
            self.deleted: list[str] = []

        async def delete_object(self, key):
            self.deleted.append(key)

    class SweepJob:
        def __init__(self, *_args, **_kwargs):
            pass

        async def info(self):
            return SimpleNamespace(
                function="transcribe_task",
                args=(None, None, None, None, None, False, None, None, "transcribe-jobs/transcribe:job-1/original.mp3"),
            )

    monkeypatch.setattr(route, "Job", SweepJob)
    storage = SweepStorage()

    assert await sweep_orphaned_spools(storage, object()) == 1
    assert storage.deleted == ["transcribe-jobs/transcribe:job-1/duplicate.mp3"]


@pytest.mark.asyncio
async def test_worker_file_job_delivers_transcript_then_deletes_only_successful_object(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    storage = RecordingStorage()
    cms = RecordingCMS()
    stt = RecordingSTT(
        TranscribeResult(
            text="contract transcript",
            language="ar",
            language_probability=0.91,
            segments=[{"start": 0.0, "end": 2.0, "text": "contract transcript"}],
            duration_sec=2.0,
        )
    )
    model_manager = SimpleNamespace(stt=stt)
    ctx = {
        "job_id": "arq-1",
        "model_manager": model_manager,
        "cms_client": cms,
        "storage_client": storage,
        "temp_dir": str(tmp_path),
    }

    result = await transcribe_task(
        ctx,
        audio_path=None,
        url=None,
        content_id="content-1",
        transcription_job_id="cms-job-1",
        language="ar",
        word_timestamps=True,
        storage_key="transcribe-jobs/input.mp3",
        media_size_bytes=42,
    )

    assert result["text"] == "contract transcript"
    assert [name for name, _ in storage.calls] == ["download", "delete"]
    assert storage.calls[-1] == ("delete", "transcribe-jobs/input.mp3")
    assert [name for name, _ in cms.calls] == [
        "update:cms-job-1",
        "create_transcript",
    ]
    transcript = cms.calls[-1][1]
    assert transcript["content_item_id"] == "content-1"
    assert transcript["transcription_job_id"] == "cms-job-1"
    assert transcript["segments"] == [
        {"start": 0.0, "end": 2.0, "text": "contract transcript"}
    ]
    assert not any(Path(path).exists() for path, _, _ in stt.calls)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_worker_failure_keeps_remote_object_and_marks_cms_job_failed() -> None:
    storage = RecordingStorage()
    cms = RecordingCMS()
    stt = RecordingSTT()
    stt.transcribe = MagicMock(side_effect=RuntimeError("provider unavailable"))
    ctx = {
        "model_manager": SimpleNamespace(stt=stt),
        "cms_client": cms,
        "storage_client": storage,
    }

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await transcribe_task(
            ctx,
            audio_path=None,
            url=None,
            content_id="content-1",
            transcription_job_id="cms-job-1",
            language=None,
            word_timestamps=False,
            storage_key="transcribe-jobs/input.mp3",
        )

    assert storage.calls == [("download", "transcribe-jobs/input.mp3")]
    assert cms.calls[-1][0] == "update:cms-job-1"
    assert cms.calls[-1][1]["status"] == "failed"
