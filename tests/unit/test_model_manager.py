"""ModelManager.warmup selective-load tests (M2).

The arq worker must load Whisper only — loading CLIP there wastes ~600 MB and
cold-start time since the worker never runs image embedding.
"""
import asyncio
from unittest.mock import MagicMock

from src.config import Settings
from src.models.manager import ModelManager


def _manager() -> ModelManager:
    settings = Settings(
        SERVICE_AUTH_TOKEN="t",
        CMS_SERVICE_TOKEN="t",
        CMS_BASE_URL="http://localhost:8080",
        MODELS_DIR="./test-models",
        ENV="test",
    )
    mgr = ModelManager(settings)
    # Replace the heavy load() calls — we only assert which ones fire.
    mgr.whisper.load = MagicMock()
    mgr.clip.load = MagicMock()
    return mgr


def test_warmup_whisper_only_skips_clip() -> None:
    mgr = _manager()
    asyncio.run(mgr.warmup(["whisper"]))
    mgr.whisper.load.assert_called_once()
    mgr.clip.load.assert_not_called()


def test_warmup_default_loads_both() -> None:
    mgr = _manager()
    asyncio.run(mgr.warmup())
    mgr.whisper.load.assert_called_once()
    mgr.clip.load.assert_called_once()
