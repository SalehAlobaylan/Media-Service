from dataclasses import dataclass, field

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TranscribeResult:
    text: str
    language: str
    language_probability: float
    segments: list[dict] = field(default_factory=list)
    duration_sec: float = 0.0


class WhisperWrapper:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: str = "./models",
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._download_root = download_root
        self._model = None

    @property
    def model_size(self) -> str:
        return self._model_size

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        logger.info(
            "loading_whisper",
            model_size=self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
            download_root=self._download_root,
        )
        logger.info("whisper_loaded", model_size=self._model_size)

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        word_timestamps: bool = False,
    ) -> TranscribeResult:
        if self._model is None:
            raise RuntimeError("Whisper model is not loaded. Call load() first.")

        segments_iter, info = self._model.transcribe(
            audio_path,
            language=language,
            word_timestamps=word_timestamps,
            vad_filter=True,
        )

        segments = []
        full_text_parts = []

        for segment in segments_iter:
            segments.append(
                {
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "text": segment.text.strip(),
                }
            )
            full_text_parts.append(segment.text.strip())

        full_text = " ".join(full_text_parts)

        return TranscribeResult(
            text=full_text,
            language=info.language,
            language_probability=round(info.language_probability, 4),
            segments=segments,
            duration_sec=round(info.duration, 3),
        )
