import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from src import worker


@pytest.mark.asyncio
async def test_fair_dispatch_visits_empty_lanes_and_rotates_priority(monkeypatch):
    stop = asyncio.Event()
    calls = []

    async def tick(ctx, lane, callback):
        calls.append(lane)
        await asyncio.sleep(0)  # Simulate a CMS round trip in an empty lane.
        if len(calls) == 6:
            stop.set()

    monkeypatch.setattr(worker, "_run_dispatch_once", tick)
    monkeypatch.setattr(worker, "_DISPATCH_SPECS", (
        ("artifacts", None, 0.001),
        ("content", None, 0.001),
        ("transcript", None, 0.001),
    ))
    await asyncio.wait_for(worker._fair_dispatch_loop({"dispatch_stop": stop}), 1)
    assert calls == ["artifacts", "content", "transcript", "content", "transcript", "artifacts"]


def test_media_dispatch_is_single_effect_and_recognizes_arq_cron_ids() -> None:
    """The old arq cron IDs are function-name-prefixed, not ``cron:`` IDs."""

    assert worker._MAX_ARQ_JOBS == 1
    assert worker._MAX_ARQ_QUEUE_DEPTH == 1
    for function_name in (
        "artifact_coverage_dispatch_tick",
        "content_stage_dispatch_tick",
        "transcription_segment_dispatch_tick",
    ):
        assert f"{function_name}:1700000000000".startswith(
            worker._CRON_JOB_PREFIXES
        )


@pytest.mark.asyncio
async def test_worker_startup_fails_before_ready_context_when_stt_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = MagicMock()
    settings.ENV = "test"
    settings.LOG_LEVEL = "info"
    settings.is_production = False
    settings.validate_startup.return_value = ([], [])
    manager = MagicMock()
    manager.stt.is_loaded = False
    manager.warmup = AsyncMock()
    cms = MagicMock()
    cms.close = AsyncMock()

    monkeypatch.setattr(worker, "Settings", lambda: settings)
    monkeypatch.setattr(worker, "ModelManager", lambda _: manager)
    monkeypatch.setattr(worker, "CMSClient", lambda _: cms)

    context: dict = {}
    with pytest.raises(RuntimeError, match="STT provider is not ready"):
        await worker._startup(context)

    manager.warmup.assert_awaited_once_with(["stt"])
    cms.close.assert_awaited_once()
    assert context == {}


@pytest.mark.asyncio
async def test_worker_startup_accepts_no_spool_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = MagicMock()
    settings.ENV = "test"
    settings.LOG_LEVEL = "info"
    settings.is_production = False
    settings.REDIS_URL = "redis://localhost:6379"
    settings.ARQ_REDIS_DB = 2
    settings.MEDIA_TEMP_DIR = ""
    settings.validate_startup.return_value = ([], [])

    manager = MagicMock()
    manager.stt.is_loaded = True
    manager.stt.name = "deepgram"
    manager.warmup = AsyncMock()
    manager.stt.aclose = AsyncMock()
    cms = MagicMock()
    cms.health_check = AsyncMock(return_value=True)
    cms.close = AsyncMock()
    fetch = MagicMock()
    fetch.aclose = AsyncMock()
    redis = MagicMock()
    redis.ping = AsyncMock()
    redis.aclose = AsyncMock()

    monkeypatch.setattr(worker, "Settings", lambda: settings)
    monkeypatch.setattr(worker, "ModelManager", lambda _: manager)
    monkeypatch.setattr(worker, "CMSClient", lambda _: cms)
    monkeypatch.setattr(worker, "SafeFetchClient", lambda: fetch)
    monkeypatch.setattr(worker.Redis, "from_url", lambda *args, **kwargs: redis)

    context: dict = {}
    await worker._startup(context)
    assert "storage_client" not in context
    await worker._shutdown(context)
    redis.aclose.assert_awaited_once()
    cms.close.assert_awaited_once()
    fetch.aclose.assert_awaited_once()
