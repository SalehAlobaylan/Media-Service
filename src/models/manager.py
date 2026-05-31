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

    async def warmup(self, models: list[str] | None = None) -> None:
        """Load models concurrently.

        `models=None` loads everything (Whisper + CLIP) — the API process serves
        both transcription and image embedding. The arq worker passes
        `["whisper"]` since `transcribe_task` only needs Whisper; loading CLIP
        there would waste ~600 MB + cold-start time for nothing.
        """
        loaders = {"whisper": self.whisper.load, "clip": self.clip.load}
        selected = list(loaders) if models is None else [m for m in models if m in loaders]

        loop = asyncio.get_event_loop()
        logger.info("loading_models", models=selected)

        tasks = [loop.run_in_executor(self._executor, loaders[name]) for name in selected]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for name, result in zip(selected, results):
            if isinstance(result, Exception):
                logger.error("model_load_failed", model=name, error=str(result))
            else:
                logger.info("model_loaded", model=name)
