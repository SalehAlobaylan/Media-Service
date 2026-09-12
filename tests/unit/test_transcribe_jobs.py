from fastapi.testclient import TestClient


class DuplicateARQ:
    async def enqueue_job(self, *args, **kwargs):
        return None


class FailingARQ:
    async def enqueue_job(self, *args, **kwargs):
        raise RuntimeError("redis unavailable")


class BusyARQ:
    async def zcard(self, key):
        assert key == "arq:queue"
        return 1

    async def exists(self, *keys):
        assert keys
        return 0

    async def enqueue_job(self, *args, **kwargs):
        raise AssertionError("busy queue must not receive another job")


def test_duplicate_url_submission_preserves_deterministic_job_id(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    client.app.state.arq_pool = DuplicateARQ()
    response = client.post(
        "/v1/transcribe/jobs", headers=auth_headers,
        data={"url": "https://media.example.test/clip.mp3", "content_id": "11111111-1111-4111-8111-111111111111"},
    )
    assert response.status_code == 202, response.text
    assert response.json()["job_id"] == "transcribe:11111111-1111-4111-8111-111111111111"


def test_enqueue_exception_creates_no_remote_storage_state(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    client.app.state.arq_pool = FailingARQ()
    response = client.post(
        "/v1/transcribe/jobs", headers=auth_headers,
        data={"url": "https://media.example.test/clip.mp3", "content_id": "11111111-1111-4111-8111-111111111111"},
    )
    assert response.status_code == 500, response.text


def test_busy_arq_queue_rejects_new_transcription_without_marking_it_failed(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    client.app.state.arq_pool = BusyARQ()
    response = client.post(
        "/v1/transcribe/jobs",
        headers=auth_headers,
        data={
            "url": "https://media.example.test/clip.mp3",
            "content_id": "11111111-1111-4111-8111-111111111111",
        },
    )
    assert response.status_code == 429, response.text
    assert response.headers["retry-after"] == "15"
