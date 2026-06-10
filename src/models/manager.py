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
from src.providers.factory import build_stt_provider
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModelManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Two concurrent loaders — STT + CLIP. Cold start is bottlenecked
        # on the slowest model (Whisper if it's the active STT engine).
        self._executor = ThreadPoolExecutor(max_workers=2)

        # WhisperWrapper is still constructed for legacy model metadata paths,
        # but hosted STT is the only supported transcription provider now.
        self.whisper = WhisperWrapper(
            model_size=settings.WHISPER_MODEL_SIZE,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            download_root=settings.MODELS_DIR,
        )
        self.stt = build_stt_provider(settings, self.whisper)
        self.clip = CLIPWrapper(
            model_name=settings.CLIP_MODEL,
            cache_folder=settings.MODELS_DIR,
        )

    @property
    def is_ready(self) -> dict[str, bool]:
        return {
            "stt": self.stt.is_loaded,
            "clip": self.clip.is_loaded,
        }

    @property
    def all_ready(self) -> bool:
        return self.stt.is_loaded and self.clip.is_loaded

    async def warmup(self, models: list[str] | None = None) -> None:
        """Load models concurrently.

        `models=None` loads everything (active STT engine + CLIP) — the API
        process serves both transcription and image embedding. The arq worker
        passes `["stt"]` since `transcribe_task` only needs the STT engine;
        loading CLIP there would waste ~600 MB + cold-start time for nothing.

        For a hosted STT engine (Deepgram) the "stt" loader is a no-op, so no
        local Whisper model is loaded — saving memory on the API + worker.
        """
        loaders = {"stt": self.stt.load, "clip": self.clip.load}
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
