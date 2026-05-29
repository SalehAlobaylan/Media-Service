"""Slim model manager for Media-Service.

Loads Whisper + CLIP only. Text embedder and reranker live in
Enrichment-Service. Same loading pattern as Enrichment's manager so the
two services stay structurally aligned.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.config import Settings
from src.models.clip import CLIPWrapper
from src.models.whisper import WhisperWrapper
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModelManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Two concurrent loaders — Whisper + CLIP. Cold start is bottlenecked
        # on the slowest model (typically Whisper).
        self._executor = ThreadPoolExecutor(max_workers=2)

        self.whisper = WhisperWrapper(
            model_size=settings.WHISPER_MODEL_SIZE,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            download_root=settings.MODELS_DIR,
        )
        self.clip = CLIPWrapper(
            model_name=settings.CLIP_MODEL,
            cache_folder=settings.MODELS_DIR,
        )

    @property
    def is_ready(self) -> dict[str, bool]:
        return {
            "whisper": self.whisper.is_loaded,
            "clip": self.clip.is_loaded,
        }

    @property
    def all_ready(self) -> bool:
        return self.whisper.is_loaded and self.clip.is_loaded

    async def warmup(self) -> None:
        loop = asyncio.get_event_loop()

        logger.info("loading_models")

        whisper_task = loop.run_in_executor(self._executor, self.whisper.load)
        clip_task = loop.run_in_executor(self._executor, self.clip.load)

        results = await asyncio.gather(whisper_task, clip_task, return_exceptions=True)

        for name, result in zip(["whisper", "clip"], results):
            if isinstance(result, Exception):
                logger.error("model_load_failed", model=name, error=str(result))
            else:
                logger.info("model_loaded", model=name)
