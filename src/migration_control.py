import asyncio
import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from src.auth.service_auth import verify_migration_coordinator_token


class OwnerRequest(BaseModel):
    program_id: UUID
    expected_epoch: int = Field(ge=0)


class MigrationOwner:
    key = "wahb:database-migration:media-quiesced"

    def __init__(self) -> None:
        self.program_id: str | None = None
        self.epoch = 0
        self.active = 0
        self.since: datetime | None = None
        self.lock = asyncio.Lock()
        self.persistence_ready = False

    async def restore(self, pool: Any | None) -> None:
        if pool is None:
            self.persistence_ready = False
            return
        raw = await pool.get(self.key)
        self.persistence_ready = True
        if not raw:
            return
        payload = json.loads(raw)
        self.program_id = str(UUID(payload["program_id"]))
        self.epoch = int(payload["epoch"])
        self.since = datetime.fromisoformat(payload["since"])

    async def quiesce(self, request: OwnerRequest) -> None:
        async with self.lock:
            if self.program_id and (
                self.program_id != str(request.program_id)
                or self.epoch != request.expected_epoch
            ):
                raise HTTPException(409, "migration owner precondition changed")
            self.program_id = str(request.program_id)
            self.epoch = request.expected_epoch
            self.since = self.since or datetime.now(UTC)

    async def resume(self, request: OwnerRequest) -> None:
        async with self.lock:
            if (
                self.program_id != str(request.program_id)
                or self.epoch != request.expected_epoch
            ):
                raise HTTPException(409, "migration owner precondition changed")
            self.program_id = None
            self.since = None

    def evidence(self) -> dict[str, Any]:
        if not self.persistence_ready:
            state = "unknown"
        elif not self.program_id:
            state = "not_quiesced"
        elif self.active == 0:
            state = "quiesced"
        else:
            state = "draining"
        return {
            "state": state,
            "owner": {
                "program_id": self.program_id,
                "epoch": self.epoch,
                "since": self.since,
            },
            "active_count": self.active,
            "persistence_ready": self.persistence_ready,
            "observed_at": datetime.now(UTC),
        }


owner = MigrationOwner()
router = APIRouter(
    prefix="/internal/database-migration",
    dependencies=[Depends(verify_migration_coordinator_token)],
)


@router.get("/quiescence")
async def status(request: Request) -> dict[str, Any]:
    try:
        await owner.restore(getattr(request.app.state, "arq_pool", None))
    except Exception:
        return {
            "state": "unknown",
            "reason": "durable_owner_unavailable",
            "observed_at": datetime.now(UTC),
        }
    return owner.evidence()


@router.post("/quiesce")
async def quiesce(body: OwnerRequest, request: Request) -> dict[str, Any]:
    pool = getattr(request.app.state, "arq_pool", None)
    if pool is None:
        raise HTTPException(503, "durable migration owner store unavailable")
    await owner.restore(pool)
    await owner.quiesce(body)
    await pool.set(
        owner.key,
        json.dumps(
            {
                "program_id": str(body.program_id),
                "epoch": body.expected_epoch,
                "since": owner.since.isoformat() if owner.since else "",
            }
        ),
    )
    return owner.evidence()


@router.post("/resume")
async def resume(body: OwnerRequest, request: Request) -> dict[str, Any]:
    pool = getattr(request.app.state, "arq_pool", None)
    if pool is None:
        raise HTTPException(503, "durable migration owner store unavailable")
    await owner.restore(pool)
    if owner.program_id != str(body.program_id) or owner.epoch != body.expected_epoch:
        raise HTTPException(409, "migration owner precondition changed")
    await pool.delete(owner.key)
    await owner.resume(body)
    return owner.evidence()


class MigrationFenceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Any:
        tracked = request.method in {"POST", "PUT", "PATCH", "DELETE"} and not (
            request.url.path.startswith("/internal/database-migration/")
        )
        if tracked:
            async with owner.lock:
                if owner.program_id:
                    raise HTTPException(
                        423, "media effects are quiesced for database migration"
                    )
                owner.active += 1
        try:
            return await call_next(request)
        finally:
            if tracked:
                async with owner.lock:
                    owner.active -= 1
