from dataclasses import dataclass

from fastapi.testclient import TestClient


class FakeStorage:
    is_configured = True

    def __init__(self) -> None:
        self.uploaded: list[str] = []
        self.deleted: list[str] = []

    async def upload_fileobj(self, fileobj, key, **kwargs) -> None:
        fileobj.read()
        self.uploaded.append(key)

    async def delete_object(self, key: str) -> None:
        self.deleted.append(key)


@dataclass
class FakeJob:
    job_id: str


class DuplicateARQ:
    async def enqueue_job(self, *args, **kwargs):
        return None


class FailingARQ:
    async def enqueue_job(self, *args, **kwargs):
        raise RuntimeError("redis unavailable")


def test_duplicate_submission_reclaims_only_newly_uploaded_object(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    storage = FakeStorage()
    client.app.state.storage_client = storage
    client.app.state.arq_pool = DuplicateARQ()

    response = client.post(
        "/v1/transcribe/jobs",
        headers=auth_headers,
        data={"content_id": "11111111-1111-4111-8111-111111111111"},
        files={"audio_file": ("clip.mp3", b"audio", "audio/mpeg")},
    )

    assert response.status_code == 202, response.text
    assert response.json()["job_id"] == "transcribe:11111111-1111-4111-8111-111111111111"
    assert len(storage.uploaded) == 1
    assert storage.deleted == storage.uploaded


def test_enqueue_exception_reclaims_request_owned_object(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    storage = FakeStorage()
    client.app.state.storage_client = storage
    client.app.state.arq_pool = FailingARQ()

    response = client.post(
        "/v1/transcribe/jobs",
        headers=auth_headers,
        data={"content_id": "11111111-1111-4111-8111-111111111111"},
        files={"audio_file": ("clip.mp3", b"audio", "audio/mpeg")},
    )

    assert response.status_code == 500, response.text
    assert len(storage.uploaded) == 1
    assert storage.deleted == storage.uploaded
