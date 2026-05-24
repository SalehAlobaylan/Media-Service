# CLAUDE.md — Media-Service

> Instructions for Claude Code (AI agent). Direct directives only.

---
> ## ABSOLUTE RULE
> **NEVER add AI agents as co-authors in git commits.**
> Do NOT include `Co-Authored-By: Claude` or any AI attribution — ever.
---

## What This Is

Python microservice for the Wahb platform. Owns all media-processing:

- Whisper transcription (faster-whisper, sync + async via arq)
- CLIP image embedding (clip-ViT-B-32, 512-dim)
- Future: Arabic OCR, audio language detection, video processing, image moderation

Created by splitting Whisper + CLIP out of Enrichment-Service. The implementation
behind each endpoint is meant to be swappable — Whisper today, possibly
AssemblyAI / Azure / a Saudi-tuned model tomorrow.

## Architecture

Triangle model — Aggregation calls Media for ingestion-time work; Media writes
results directly back to CMS:

- **Aggregation → Media**: per-content transcription + image embedding
- **Media → CMS**: write-back via `/internal/transcripts` and `/internal/content-items/:id/{transcript,image-embedding}`

CMS does NOT call Media-Service today. (When the admin retry trigger fires from
Platform-Console, CMS goes through its own `triggerTranscription` helper, which
also points at Media.)

## Running

```bash
# Development
cp .env.example .env
make install
make dev         # API on :5051

# arq worker (separate process)
make worker

# Docker
make docker-build

# Tests
make test
```

## Dev Commands

| Command | Purpose |
|---------|---------|
| `make dev` | Start dev server (port 5051) |
| `make run` | Start prod-style server |
| `make worker` | Run arq async-transcription worker |
| `make test` | Run all tests |
| `make test-unit` | Unit tests only |
| `make lint` | Ruff linter |
| `make format` | Ruff formatter |
| `make docker-build` | Build Docker image |
| `make download-models` | Pre-cache Whisper + CLIP |

## Key Patterns

**content_id write-back**: Every AI endpoint accepts an optional `content_id`.
If present, Media-Service writes results directly to CMS via internal API and
surfaces the outcome via `write_back_status` / `write_back_error`. If absent,
stateless tool mode (caller gets the result, no persistence).

**Auth**: Bearer token (`SERVICE_AUTH_TOKEN`) on all `/v1/*` routes. Health,
ready, and metrics endpoints are unauthenticated.

**Circuit breaker**: CMS client uses CLOSED→OPEN→HALF_OPEN state machine
(5 failures → open, 30s reset). Same shape as Enrichment-Service.

**arq async jobs**: Long transcriptions go through `POST /v1/transcribe/jobs`
which spools the upload to `MEDIA_TEMP_DIR`, enqueues to Redis (db=2), and
returns a job ID. The worker process (separate from the API) loads its own
Whisper instance and runs the job. The API never blocks on transcription.

**X-Request-ID propagation**: Inbound `X-Request-ID` is bound to a contextvar
and forwarded to outbound CMS writes — so a single trace ID flows
Aggregation → Media → CMS for one ingestion.

## Service Boundaries

| Media Owns | Media Cannot |
|---|---|
| Whisper transcription (sync + async) | Orchestrate pipelines |
| CLIP image embedding | Manage BullMQ queues |
| Future: OCR, audio lang-id, video, image moderation | Serve user-facing APIs |
| Write transcripts + image embeddings to CMS | Write text embeddings (Enrichment owns those) |
| `MEDIA_TEMP_DIR` lifecycle | LLM-backed text ops (translate/summarize/tag) |

## API Endpoints

| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/health` | GET | No | Liveness check |
| `/ready` | GET | No | Readiness (Whisper + CLIP loaded + CMS reachable) |
| `/metrics` | GET | No | Prometheus metrics (`media_*` prefix) |
| `/v1/models` | GET | Yes | Loaded model info (whisper, clip) |
| `/v1/transcribe` | POST | Yes | Sync audio → text (short clips) |
| `/v1/transcribe/jobs` | POST | Yes | Enqueue async transcription, returns job_id |
| `/v1/transcribe/jobs/:id` | GET | Yes | Poll async job status / fetch result |
| `/v1/embed/image` | POST | Yes | Image (upload or URL) → 512-dim CLIP vector |

## Redis usage

- db=0 → Aggregation BullMQ (not ours)
- db=1 → Enrichment LLM cache (not ours)
- db=2 → **Media arq job queue** (this service)
