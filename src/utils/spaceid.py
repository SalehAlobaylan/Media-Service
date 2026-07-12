"""Immutable vector-space identity for the Embedding & Model Lifecycle System.

MUST stay byte-identical to the Go implementation at
Content-Management-System/src/spaceid/spaceid.go and the Enrichment mirror at
Enrichment-Service/src/common/spaceid.py. Canonical serialization is compact,
sorted-key JSON with no ASCII escaping. See the Go file for the full contract.

This copy lives in Media-Service for the CLIP (image) space. It is intentionally
duplicated rather than shared, because the three services deploy independently;
the golden fixture test in each service pins them together.
"""
from __future__ import annotations

import hashlib
import json

# Producer-recipe constant for the CLIP image-embedding surface. MUST match the
# Go constant spaceid.RecipeContentImage.
RECIPE_CONTENT_IMAGE = "content-hero-thumbnail:v1"


def _canonical(obj: dict) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_space_id(
    model: str,
    revision: str,
    dimensions: int,
    normalized: bool,
    pooling: str,
) -> str:
    if not revision or not revision.strip():
        return ""
    return _sha256_hex(
        _canonical(
            {
                "model": model,
                "revision": revision,
                "dimensions": dimensions,
                "normalized": normalized,
                "pooling": pooling,
            }
        )
    )


def compute_producer_id(space_id: str, recipe: str) -> str:
    if not space_id or not space_id.strip():
        return ""
    return _sha256_hex(_canonical({"recipe": recipe, "space_id": space_id}))
