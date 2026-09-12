import pytest
from pydantic import ValidationError

from src.contracts.long_form import TranscriptionSegmentClaim


UNIT = {
    "id": "11111111-1111-4111-8111-111111111111",
    "generation_id": "22222222-2222-4222-8222-222222222222",
    "segment_index": 0,
    "start_ms": 0,
    "end_ms": 120000,
    "attempt_count": 1,
    "claim_token": "33333333-3333-4333-8333-333333333333",
    "unit_fence_token": "44444444-4444-4444-8444-444444444444",
    "fence_token": "44444444-4444-4444-8444-444444444444",
    "lease_expires_at": "2026-09-08T10:00:00Z",
}


def test_transcription_claim_requires_capability_credentials() -> None:
    with pytest.raises(ValidationError):
        TranscriptionSegmentClaim.model_validate({"unit": {**UNIT, "claim_token": None}, "generation": {}})
    with pytest.raises(ValidationError):
        TranscriptionSegmentClaim.model_validate({"unit": {**UNIT, "unit_fence_token": None}, "generation": {}})


def test_transcription_claim_preserves_post_increment_attempt_and_fences() -> None:
    claim = TranscriptionSegmentClaim.model_validate({"unit": UNIT, "generation": {"id": UNIT["generation_id"]}})
    assert claim.unit.attempt_count == 1
    assert claim.unit.claim_token == UNIT["claim_token"]
    assert claim.unit.unit_fence_token == claim.unit.fence_token
