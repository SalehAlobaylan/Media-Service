# Media-Service

The media-processing microservice for the Wahb platform. Owns speech-to-text transcription (sync + async) and CLIP image embedding, writing both back to CMS. **Aggregation** calls it during ingest; results go straight to CMS via `/internal`.

It does **not** run text embeddings, LLM ops, FFmpeg transcoding, pipeline orchestration, or serve user-facing traffic — those belong to CMS, Enrichment, and Aggregation.

**Port:** 5051 · **Deployment:** internal-only (Cranl — two deploys from one image: API + arq worker) · **Stack:** Python 3.11, FastAPI, Deepgram / faster-whisper, CLIP, arq

> Full feature, architecture, and endpoint reference: [`../docs/media-service.md`](../docs/media-service.md). Product intent: [`../docs/PRD.md`](../docs/PRD.md).

## What's inside

| Capability | Model / impl | Endpoint |
|------------|--------------|----------|
| Speech-to-text (sync, short clips) | Deepgram Nova-3 (default) / faster-whisper | `POST /v1/transcribe` |
| Speech-to-text (async, long-form) | arq queue + durable media reference | `POST /v1/transcribe/jobs`, `GET`/`DELETE /v1/transcribe/jobs/:id` |
| Image embedding | CLIP-ViT-B-32 (sentence-transformers, 512-dim) | `POST /v1/embed/image` |
| Models registry | — | `GET /v1/models` |
| Liveness / readiness | models + CMS | `GET /health`, `GET /ready` |
| Queue depth | arq worker liveness + throughput | `GET /health/queue` |
| Prometheus metrics | `media_*` prefix | `GET /metrics` |

The STT engine sits behind a stable HTTP boundary so it can be swapped (Deepgram, faster-whisper, a Saudi-tuned model, …) without rippling into Aggregation or CMS. Default is **Deepgram Nova-3** (`STT_PROVIDER=deepgram`, chosen for Arabic dialect + code-switching coverage); **faster-whisper** is the local/self-hosted alternative.

## Atomization Transcript Role

Media owns timestamped parent transcripts used by the atomization engine. Aggregation may force STT for >40m atomization candidates that lack a usable transcript. Parent transcripts remain available for provenance, search, and future re-chaptering; child transcript slices are derived downstream and embedded through Enrichment. Media does not plan chapters, cut media, or decide feed visibility.

## Quick Start

```bash
cp .env.example .env       # then edit
make install               # one-time: create venv + install deps
make dev                   # API on :5051 (--reload)

# in another shell — the async-transcription worker (separate process/deployment):
make worker
```

`./start.sh` at the repo root starts the API + worker automatically alongside the other services.

### Docker

```bash
make docker-build          # API image (Dockerfile)
# the worker builds from Dockerfile.worker — same source, different entrypoint
```

## Async transcription (how it works)

Long audio does not block the API: `POST /v1/transcribe/jobs` accepts one CMS- or Aggregation-approved durable URL, enqueues that reference in Redis (db=2), and returns a `job_id`. The separate worker downloads the reference only when executing, transcribes it in worker-local scratch, writes back to CMS, and removes local scratch. Async multipart submissions are rejected; user uploads must first be persisted through CMS → Aggregation. API and worker therefore need no shared filesystem or duplicate object-storage bucket.

## Configuration

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `PORT` | no | 5051 | API port |
| `ENV` | no | development | `production` makes a missing auth token fatal |
| `SERVICE_AUTH_TOKEN` | prod | — | Bearer for inbound `/v1/*` (falls back to `MEDIA_SERVICE_TOKEN` / `CMS_SERVICE_TOKEN`) |
| `CMS_BASE_URL` | yes | http://localhost:8080 | Where transcripts + image embeddings are written |
| `CMS_SERVICE_TOKEN` | yes | — | Bearer for outbound CMS `/internal/*` writes |
| `STT_PROVIDER` | no | deepgram | STT engine (`deepgram` / whisper) |
| `STT_DEFAULT_LANGUAGE` | no | multi | `multi` (ar+en code-switch), `ar`, or `en` |
| `DEEPGRAM_API_KEY` | if deepgram | — | Deepgram key |
| `DEEPGRAM_MODEL` | no | nova-3 | Deepgram model |
| `WHISPER_MODEL_SIZE` | no | base | faster-whisper model size |
| `WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE` | no | cpu / int8 | Whisper runtime |
| `CLIP_MODEL` | no | clip-ViT-B-32 | Image embedder selector |
| `CLIP_MODEL_REVISION` | no | empty | Immutable model commit; empty auto-resolves the loaded snapshot for lifecycle provenance |
| `MODELS_DIR` | no | ./models | Model cache dir |
| `MEDIA_TEMP_DIR` | no | system tmp | Per-process temp dir (not shared between API/worker) |
| `MAX_UPLOAD_MB` | no | 200 | Upload cap |
| `TRANSCRIBE_TIMEOUT_SEC` | no | 600 | Transcription timeout |
| `REDIS_URL` / `ARQ_REDIS_DB` | no | redis://localhost:6379 / 2 | arq queue (db=2 — isolated from Aggregation db=0, Enrichment db=1) |
| `CB_FAILURE_THRESHOLD` / `CB_RESET_TIMEOUT_SEC` / `CB_HALF_OPEN_REQUESTS` | no | 5 / 30 / 3 | CMS circuit breaker |
| `CORS_ALLOWED_ORIGINS` | no | * | CSV; "" disables CORS |

## Commands

| Command | Purpose |
|---------|---------|
| `make dev` | Dev server with reload (:5051) |
| `make run` | Prod-style server (no reload) |
| `make worker` | arq async-transcription worker |
| `make install` / `make install-dev` | Install runtime / dev deps into `.venv` |
| `make test` / `make test-unit` / `make test-coverage` | Tests |
| `make lint` / `make format` | Ruff |
| `make download-models` | Pre-cache Whisper + CLIP for Docker builds |
| `make docker-build` | Build the container image |

## Service Boundaries

**Owns:** audio/image → AI output; writes to CMS via `POST /internal/transcripts`, `PATCH /internal/content-items/:id/transcript`, `PATCH /internal/content-items/:id/image-embedding`.

**Does not own:** text embeddings (Enrichment-Service), retrieval / search (CMS + Enrichment), pipeline orchestration / BullMQ (Aggregation), chapter planning (Enrichment), feed assembly/visibility (CMS), FFmpeg transcoding/cutting (Aggregation).

## Project Structure

```
src/
├── main.py        # app, middleware, metrics, route mounting
├── config.py      # Settings (env)
├── worker.py      # arq worker entrypoint (transcribe_task)
├── routes/        # transcribe, embed_image, health
├── schemas/       # Pydantic response shapes
├── models/        # ModelManager (loads/warms Whisper + CLIP)
├── clients/       # CMS write-back and safe-fetch clients
└── middleware/    # logging, request-id, error handler
tests/             # unit + integration
scripts/           # model download
Dockerfile         # API image
Dockerfile.worker  # arq worker image (same source, worker entrypoint)
```
