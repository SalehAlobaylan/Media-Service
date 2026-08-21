import json
from uuid import uuid4

import pytest
from fastapi import HTTPException

from src.migration_control import MigrationOwner, OwnerRequest


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}

    async def get(self, key: str):
        return self.values.get(key)


@pytest.mark.asyncio
async def test_owner_restores_durable_quiescence_after_process_restart() -> None:
    program_id = uuid4()
    redis = FakeRedis()
    redis.values[MigrationOwner.key] = json.dumps(
        {
            "program_id": str(program_id),
            "epoch": 7,
            "since": "2026-08-20T00:00:00+00:00",
        }
    ).encode()
    owner = MigrationOwner()

    await owner.restore(redis)

    assert owner.evidence()["state"] == "quiesced"
    assert owner.program_id == str(program_id)
    assert owner.epoch == 7


@pytest.mark.asyncio
async def test_owner_rejects_a_different_program_or_epoch() -> None:
    owner = MigrationOwner()
    first = OwnerRequest(program_id=uuid4(), expected_epoch=3)
    await owner.quiesce(first)

    with pytest.raises(HTTPException) as caught:
        await owner.quiesce(OwnerRequest(program_id=uuid4(), expected_epoch=3))

    assert caught.value.status_code == 409
