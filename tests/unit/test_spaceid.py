"""Cross-language golden fixtures for the vector-space identity contract.

MUST match the Go and Enrichment implementations. See Enrichment's copy.
"""
from src.utils.spaceid import compute_producer_id, compute_space_id

GOLDEN_SPACE = "58ce573ed10df8af2a0197fde7f7114cd26f844de65b14f0bc95633d36d8a70f"
GOLDEN_PRODUCER = "dd5e4491b5dfb06a6696e765a2a5e813de7e61ae043ebd9bc9585e3f132f0791"


def test_golden_space_id():
    assert (
        compute_space_id("test-model", "abc123", 4, True, "mean") == GOLDEN_SPACE
    )


def test_golden_producer_id():
    assert compute_producer_id(GOLDEN_SPACE, "r:v1") == GOLDEN_PRODUCER


def test_unresolved_revision_yields_empty():
    assert compute_space_id("m", "  ", 512, True, "p") == ""
    assert compute_producer_id("", "r:v1") == ""
