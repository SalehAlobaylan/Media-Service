# Media-Service: Whisper transcription (sync + async via arq) + CLIP image
# embedding. CPU-only — the torch CPU pin lives in requirements.txt (Phase 8).
#
# Single-stage build because Media has no Playwright/Chromium dependency and
# the model download (~3 GB on disk: Whisper-base + CLIP) is small enough
# that splitting stages adds layer-coordination overhead without saving
# meaningful image size. Layers ordered so source code changes don't bust
# the cache on the expensive `pip install` + model download steps.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps:
#   ffmpeg          — faster-whisper's audio decoding fallback path
#   libgomp1        — OpenMP runtime, used by sentence-transformers / CLIP
#                     (without this, torch raises ImportError on slim images)
#   curl            — for the healthcheck below
#   ca-certificates — HTTPS to Hugging Face / CMS write-back
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python deps first — most-frequent invalidation is source code, so anything
# that depends only on requirements.txt should live above the source copy.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-download models into the image. Cached as long as download_models.py
# is unchanged. faster-whisper (Whisper-base, ~500 MB) + sentence-transformers
# (CLIP-ViT-B-32, ~600 MB) total ~1.1 GB on disk.
COPY scripts/download_models.py ./scripts/download_models.py
RUN mkdir -p /app/models && python scripts/download_models.py --output /app/models

COPY src ./src

ENV MODELS_DIR=/app/models \
    PORT=5051

EXPOSE 5051

# Inline healthcheck — Media doesn't have a separate healthcheck.sh because
# the curl + grep is one line. Allows Cranl + docker to detect a hung process
# even if the TCP listener is still bound.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:${PORT:-5051}/health > /dev/null || exit 1

# CMD is overridden for the arq worker — Cranl deploys two entries from this
# same image: an API container (this CMD) and a worker container running
# `python -m arq src.worker.WorkerSettings`.
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-5051}"]
