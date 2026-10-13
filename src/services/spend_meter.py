"""Bounded asynchronous delivery with durable per-UUID acknowledgements."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from src.utils.logging import get_logger
from prometheus_client import Counter, Gauge

ai_spend_delivery_total = Counter(
    "media_ai_spend_delivery_total", "Delivery outcomes", ["outcome"]
)
ai_spend_delivery_retries_total = Counter(
    "media_ai_spend_delivery_retries_total", "Delivery retries"
)
ai_spend_queue_depth = Gauge("media_ai_spend_queue_depth", "Buffered events")
ai_spend_queue_oldest_age_seconds = Gauge(
    "media_ai_spend_queue_oldest_age_seconds", "Oldest buffered age"
)

logger = get_logger(__name__)
QUEUE_CAPACITY = 1_000
BATCH_SIZE = 25
MAX_DELIVERY_ATTEMPTS = 3
SHUTDOWN_DRAIN_SEC = 3.0


class SpendDeliveryClient(Protocol):
    async def emit_ai_spend_events(
        self, events: list[dict[str, Any]], health: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...


class SpendMeter:
    """Process-local buffering; cumulative health makes loss visible in CMS."""

    def __init__(
        self, client: SpendDeliveryClient, capacity: int = QUEUE_CAPACITY
    ) -> None:
        self._client = client
        self._queue: asyncio.Queue[tuple[dict[str, Any], float]] = asyncio.Queue(
            capacity
        )
        self._worker: asyncio.Task[None] | None = None
        self._accepting = True
        self._closed = False
        self._process_id = str(uuid4())
        self._started_at = datetime.now(UTC).isoformat()
        self._counts: dict[str, dict[str, int]] = {}
        self._inflight: list[dict[str, Any]] = []

    def _counter(self, tenant: str) -> dict[str, int]:
        return self._counts.setdefault(
            tenant, dict(emitted=0, accepted=0, dropped=0, unobservable=0, sequence=0)
        )

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="ai-spend-meter")

    def unobservable_attempt(self, tenant: str) -> None:
        self._counter(tenant)["unobservable"] += 1

    def enqueue(self, event: dict[str, Any]) -> None:
        # Freeze occurrence before buffering or network delay.
        event = {
            "occurred_at": datetime.now(UTC).isoformat(),
            "event_count": 1,
            **event,
        }
        tenant = event.get("tenant_id") or "default"
        counters = self._counter(tenant)
        counters["emitted"] += 1
        if not self._accepting:
            counters["dropped"] += 1
            ai_spend_delivery_total.labels(outcome="dropped_shutdown").inc()
            return
        try:
            self._queue.put_nowait((event, time.monotonic()))
            ai_spend_queue_depth.set(self._queue.qsize())
        except asyncio.QueueFull:
            counters["dropped"] += 1
            ai_spend_delivery_total.labels(outcome="dropped_full").inc()
            logger.warning("ai_spend_meter_queue_full")

    async def publish_health(self) -> None:
        for tenant, counts in list(self._counts.items()):
            counts["sequence"] += 1
            await self._client.emit_ai_spend_events(
                [],
                {
                    **counts,
                    "tenant_id": tenant,
                    "process_id": self._process_id,
                    "started_at": self._started_at,
                    "source_service": "media",
                    "closed": self._closed,
                },
            )

    def _drop(self, events: list[dict[str, Any]], outcome: str) -> None:
        for event in events:
            self._counter(event.get("tenant_id") or "default")["dropped"] += 1
        if events:
            ai_spend_delivery_total.labels(outcome=outcome).inc(len(events))

    async def close(self, drain_sec: float = SHUTDOWN_DRAIN_SEC) -> None:
        self._accepting = False
        if self._worker is not None:
            try:
                await asyncio.wait_for(self._queue.join(), timeout=drain_sec)
            except TimeoutError:
                pass
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            self._worker = None
        remaining = list(self._inflight)
        self._inflight.clear()
        while not self._queue.empty():
            event, _ = self._queue.get_nowait()
            remaining.append(event)
            self._queue.task_done()
        self._drop(remaining, "dropped_shutdown")
        ai_spend_queue_depth.set(0)
        self._closed = True
        try:
            await asyncio.wait_for(self.publish_health(), timeout=1.0)
        except Exception:
            logger.warning("ai_spend_final_health_unavailable")

    async def _run(self) -> None:
        while True:
            try:
                first = await asyncio.wait_for(self._queue.get(), timeout=30)
            except TimeoutError:
                try:
                    await self.publish_health()
                except Exception:
                    logger.warning("ai_spend_health_unavailable")
                continue
            batch = [first]
            while len(batch) < BATCH_SIZE:
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            pending = [event for event, _ in batch]
            self._inflight = pending
            ai_spend_queue_oldest_age_seconds.set(
                max(0.0, time.monotonic() - min(at for _, at in batch))
            )
            try:
                for attempt in range(MAX_DELIVERY_ATTEMPTS):
                    try:
                        response = await self._client.emit_ai_spend_events(pending)
                        acknowledgements = {
                            a["event_id"]: a["status"]
                            for a in response.get("acknowledgements", [])
                        }
                        unresolved = []
                        for event in pending:
                            status = acknowledgements.get(event["event_id"])
                            if status in ("inserted", "duplicate"):
                                self._counter(event.get("tenant_id") or "default")[
                                    "accepted"
                                ] += 1
                                ai_spend_delivery_total.labels(
                                    outcome="delivered"
                                ).inc()
                            elif status == "invalid":
                                self._drop([event], "dropped_invalid")
                            else:
                                unresolved.append(event)
                        pending = unresolved
                        self._inflight = pending
                        if not pending:
                            break
                    except Exception:
                        logger.warning("ai_spend_meter_delivery_failed")
                    if attempt + 1 < MAX_DELIVERY_ATTEMPTS:
                        ai_spend_delivery_retries_total.inc()
                        await asyncio.sleep(0.1 * (attempt + 1))
                self._drop(pending, "dropped_failed")
                self._inflight = []
                try:
                    await self.publish_health()
                except Exception:
                    logger.warning("ai_spend_health_unavailable")
            finally:
                for _ in batch:
                    self._queue.task_done()
                ai_spend_queue_depth.set(self._queue.qsize())
