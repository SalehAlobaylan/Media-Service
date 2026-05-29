"""Health / readiness / queue endpoint tests.

These exercise the unauthenticated observability surface. The `client` fixture
(see conftest) injects mocks onto `app.state` and does NOT run the lifespan, so
no real models load and `arq_pool` defaults to None.
"""

from unittest.mock import AsyncMock

from src.main import app


def test_health_ok(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"]


def test_ready_reports_models_and_cms(client) -> None:
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models"] == {"whisper": True, "clip": True}
    assert body["dependencies"]["cms"] is True


def test_queue_status_not_configured(client) -> None:
    # conftest sets app.state.arq_pool = None — the arq pool isn't wired.
    r = client.get("/health/queue")
    assert r.status_code == 200
    assert r.json() == {
        "configured": False,
        "worker_alive": False,
        "queued": 0,
        "detail": None,
    }


def test_queue_status_worker_alive_parses_counts(client) -> None:
    pool = AsyncMock()
    pool.zcard.return_value = 3
    # Real arq health-check record shape.
    pool.get.return_value = (
        b"May-29 14:00:00 j_complete=10 j_failed=2 j_retried=1 j_ongoing=1 queued=3"
    )
    app.state.arq_pool = pool
    try:
        r = client.get("/health/queue")
    finally:
        app.state.arq_pool = None

    assert r.status_code == 200
    body = r.json()
    assert body["configured"] is True
    assert body["worker_alive"] is True
    assert body["queued"] == 3
    assert body["jobs_complete"] == 10
    assert body["jobs_failed"] == 2
    assert body["jobs_retried"] == 1
    assert body["jobs_ongoing"] == 1
    assert body["detail"] is not None
    pool.zcard.assert_awaited_once_with("arq:queue")


def test_queue_status_worker_down(client) -> None:
    # Pool is reachable (configured) but no worker has written a health key.
    pool = AsyncMock()
    pool.zcard.return_value = 0
    pool.get.return_value = None
    app.state.arq_pool = pool
    try:
        r = client.get("/health/queue")
    finally:
        app.state.arq_pool = None

    assert r.status_code == 200
    body = r.json()
    assert body["configured"] is True
    assert body["worker_alive"] is False
    assert body["queued"] == 0
    assert body["detail"] is None
