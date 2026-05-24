"""Prometheus metrics for Media-Service.

Metric prefix is `media_*` (not `enrichment_*`) so both services'
`/metrics` endpoints can be scraped into the same Prometheus instance
without collisions.
"""
from prometheus_client import Counter, Gauge, Histogram

# ─── Transcription ──────────────────────────────────────────

transcriptions_total = Counter(
    "media_transcriptions_total",
    "Total transcription requests",
    ["status", "model_size"],
)

transcription_duration = Histogram(
    "media_transcription_duration_seconds",
    "Transcription processing time",
    ["model_size"],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)

transcribe_jobs_total = Counter(
    "media_transcribe_jobs_total",
    "Async transcription jobs by state (queued | started | completed | failed).",
    ["state"],
)

# ─── Image embedding (CLIP) ─────────────────────────────────

image_embeddings_total = Counter(
    "media_image_embeddings_total",
    "Total CLIP image embedding requests",
    ["status"],
)

# ─── CMS write-back (transcript + image embedding) ──────────

cms_writeback_total = Counter(
    "media_cms_writeback_total",
    "CMS write-back attempts",
    ["endpoint", "status"],
)

# ─── Health / circuit breaker ───────────────────────────────

circuit_state = Gauge(
    "media_circuit_state",
    "CMS circuit breaker state (0=closed, 1=open, 2=half_open)",
)

model_loaded = Gauge(
    "media_model_loaded",
    "Whether a model is loaded (1=yes, 0=no)",
    ["model_name"],
)
