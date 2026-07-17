import asyncio
from pathlib import Path

import httpx
import pytest

from src.providers.deepgram_provider import DeepgramProvider
from src.services.workload import WorkloadAdmission, WorkloadOverloadedError


@pytest.mark.asyncio
async def test_bulkheads_are_independent_and_release_after_cancellation() -> None:
    admission = WorkloadAdmission(stt_limit=1, clip_limit=1, wait_seconds=0.01)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold_stt() -> None:
        async with admission.acquire("stt"):
            entered.set()
            await release.wait()

    holder = asyncio.create_task(hold_stt())
    await entered.wait()
    async with admission.acquire("clip"):
        pass
    with pytest.raises(WorkloadOverloadedError):
        async with admission.acquire("stt"):
            pass
    holder.cancel()
    with pytest.raises(asyncio.CancelledError):
        await holder
    async with admission.acquire("stt"):
        pass


@pytest.mark.asyncio
async def test_deepgram_async_path_streams_without_default_executor(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp3"
    path.write_bytes(b"audio")

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["content-length"] == "5"
        assert await request.aread() == b"audio"
        return httpx.Response(
            200,
            json={
                "metadata": {"duration": 1.0},
                "results": {"channels": [{"alternatives": [{"transcript": "مرحبا", "confidence": 1}]}]},
            },
        )

    provider = DeepgramProvider("key")
    await provider._client.aclose()
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = await provider.transcribe_async(str(path), language="ar")

    assert result.text == "مرحبا"
    assert result.language == "ar"
    await provider.aclose()
