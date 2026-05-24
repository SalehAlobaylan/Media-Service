# Media-Service

Media-processing microservice for the Wahb platform. Owns every operation that
turns raw binary media (audio, image, future video) into structured AI output.

## What's inside

| Capability | Model / impl | Endpoint |
|------------|--------------|----------|
| Speech-to-text (sync, short clips) | faster-whisper | `POST /v1/transcribe` |
| Speech-to-text (async, long-form) | arq job queue + faster-whisper | `POST /v1/transcribe/jobs` + `GET /v1/transcribe/jobs/:id` |
| Image embedding | CLIP-ViT-B-32 (sentence-transformers, 512-dim) | `POST /v1/embed/image` |
| Liveness | — | `GET /health` |
| Readiness | model + CMS check | `GET /ready` |
| Models registry | — | `GET /v1/models` |
| Prometheus metrics | `media_*` prefix | `GET /metrics` |

The Whisper implementation is intentionally behind a stable HTTP boundary so
it can be swapped (AssemblyAI Arabic, Whisper-large-v3, a Saudi-tuned model,
etc.) without rippling into Aggregation or CMS.

## Quick start

```bash
cp .env.example .env       # then edit
make install               # one-time: create venv + install deps
make dev                   # API on :5051 (--reload)
# in another shell:
make worker                # arq worker (transcribe_task)
```

`./start.sh` at the repo root starts the API + worker automatically alongside
all other services.

## Dev commands

| Command | Purpose |
|---------|---------|
| `make dev` | Start dev server with reload (port 5051) |
| `make run` | Start prod-style server (no reload) |
| `make worker` | Run the arq async-transcription worker |
| `make test` | Run all tests |
| `make test-unit` | Unit tests only |
| `make lint` | Ruff linter |
| `make format` | Ruff formatter |
| `make download-models` | Pre-cache Whisper + CLIP for Docker builds |
| `make docker-build` | Build the container image |

## Required env vars (subset of `.env.example`)

| Var | Default | Purpose |
|-----|---------|---------|
| `PORT` | `5051` | API port |
| `SERVICE_AUTH_TOKEN` | _required_ | Shared bearer for inbound `/v1/*` calls |
| `CMS_BASE_URL` | `http://localhost:8080` | Where transcripts + image embeddings are written |
| `CMS_SERVICE_TOKEN` | _required_ | Bearer for outbound CMS `/internal/*` writes |
| `REDIS_URL` | `redis://localhost:6379` | arq job queue (db=2) |
| `ARQ_REDIS_DB` | `2` | Logical Redis DB — isolated from Aggregation (0) and Enrichment LLM cache (1) |
| `MEDIA_TEMP_DIR` | system tempdir | Shared spool path between API and worker processes |
| `MAX_UPLOAD_MB` | `200` | Upload cap for `/v1/transcribe` |
| `WHISPER_MODEL_SIZE` | `base` | faster-whisper model size |
| `CLIP_MODEL` | `clip-ViT-B-32` | sentence-transformers CLIP model |

## Service boundaries

Owns: audio/image → AI output. Writes back to CMS via `/internal/transcripts`
+ `PATCH /internal/content-items/:id/transcript` + `PATCH /internal/content-items/:id/image-embedding`.

Does NOT own: text embeddings (Enrichment-Service), retrieval / search (CMS +
Enrichment-Service), pipeline orchestration (Aggregation-Service), feed
assembly (CMS).
