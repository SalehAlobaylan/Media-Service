"""Finite, independent admission gates for paid STT and CLIP work."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.utils.metrics import workload_admission_total
from contextlib import asynccontextmanager


class WorkloadOverloadedError(RuntimeError):
    pass


class WorkloadAdmission:
    # One hosted STT and one CLIP inference fit the smallest supported API
    # instance; they are deliberately independent so neither starves the other.
    def __init__(
        self, stt_limit: int = 1, clip_limit: int = 1, wait_seconds: float = 0.25
    ):
        self._semaphores = {
            "stt": asyncio.Semaphore(stt_limit),
            "clip": asyncio.Semaphore(clip_limit),
        }
        self._wait_seconds = wait_seconds
        self._clip_executor = ThreadPoolExecutor(
            max_workers=clip_limit, thread_name_prefix="media-clip"
        )

    @asynccontextmanager
    async def acquire(self, workload: str):
        semaphore = self._semaphores[workload]
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=self._wait_seconds)
        except TimeoutError as exc:
            workload_admission_total.labels(workload=workload, outcome="rejected").inc()
            raise WorkloadOverloadedError(f"{workload} workload is saturated") from exc
        workload_admission_total.labels(workload=workload, outcome="admitted").inc()
        try:
            yield
        finally:
            semaphore.release()

    async def run_clip(self, fn, *args):
        async with self.acquire("clip"):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._clip_executor, fn, *args)

    def shutdown(self) -> None:
        self._clip_executor.shutdown(wait=False, cancel_futures=True)
