FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for faster-whisper (ffmpeg for any audio fallback paths) and
# Pillow (libjpeg/zlib are bundled in Pillow wheels on Debian — no extra apt
# needed in slim).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-download models into the image so cold starts don't re-download every time.
COPY scripts/download_models.py ./scripts/download_models.py
RUN mkdir -p /app/models && python scripts/download_models.py --output /app/models

COPY src ./src

ENV MODELS_DIR=/app/models \
    PORT=5051

EXPOSE 5051

# CMD is overridden by docker-compose / Cranl for the worker process.
# Default = API. The arq worker uses: `python -m arq src.worker.WorkerSettings`.
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-5051}"]
