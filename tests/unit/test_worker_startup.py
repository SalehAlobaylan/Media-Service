from unittest.mock import AsyncMock, MagicMock

import pytest

from src import worker


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
    monkeypatch.setattr(worker, "StorageClient", lambda _: MagicMock())

    context: dict = {}
    with pytest.raises(RuntimeError, match="STT provider is not ready"):
        await worker._startup(context)

    manager.warmup.assert_awaited_once_with(["stt"])
    cms.close.assert_awaited_once()
    assert context == {}
