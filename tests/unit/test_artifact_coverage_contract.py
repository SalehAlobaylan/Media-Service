from src import worker
from pydantic import ValidationError
import pytest

from src.contracts.artifact_coverage import ArtifactCoverageClaim


def test_artifact_worker_is_single_try_and_owner_closed() -> None:
    registered = {getattr(item, "name", getattr(item, "__name__", "")): item for item in worker.WorkerSettings.functions}
    artifact = registered["artifact_coverage_task"]
    assert artifact.max_tries == 1


def test_artifact_dispatch_has_no_browser_or_queue_arguments() -> None:
    names = set(worker.artifact_coverage_task.__annotations__)
    assert names == {"ctx", "claim", "return"}
    source = worker.artifact_coverage_task.__code__.co_names
    assert "claim_artifact_coverage" not in source
    assert "enqueue_job" not in source


def test_artifact_claim_rejects_wrong_owner_artifact_and_extra_arguments() -> None:
    payload = {
        "id": "11111111-1111-4111-8111-111111111111",
        "attempt_id": "22222222-2222-4222-8222-222222222222",
        "claim_token": "33333333-3333-4333-8333-333333333333",
        "fence_token": "44444444-4444-4444-8444-444444444444",
        "input_digest": "a" * 64,
        "artifact": "text_embedding",
        "deterministic_job_id": "artifact:" + "b" * 64,
        "content": {"id": "55555555-5555-4555-8555-555555555555", "type": "VIDEO"},
        "queue_name": "arbitrary",
    }
    with pytest.raises(ValidationError):
        ArtifactCoverageClaim.model_validate(payload)
