import asyncio

import httpx
import pytest

from src.clients.circuit_breaker import CircuitBreaker, CircuitState
from src.clients.cms import _is_countable_cms_failure
from src.middleware.error_handler import CircuitOpenError


def http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://cms/internal/test")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("CMS error", request=request, response=response)


def test_only_availability_failures_are_countable() -> None:
    assert not _is_countable_cms_failure(http_error(400))
    assert not _is_countable_cms_failure(http_error(401))
    assert not _is_countable_cms_failure(http_error(409))
    assert not _is_countable_cms_failure(http_error(422))
    assert _is_countable_cms_failure(http_error(429))
    assert _is_countable_cms_failure(http_error(500))
    assert _is_countable_cms_failure(httpx.ConnectError("unavailable"))


@pytest.mark.asyncio
async def test_non_countable_failure_does_not_open_breaker() -> None:
    breaker = CircuitBreaker(failure_threshold=1)

    async def bad_request():
        raise http_error(422)

    with pytest.raises(httpx.HTTPStatusError):
        await breaker.execute(bad_request, count_failure=_is_countable_cms_failure)

    assert breaker.state is CircuitState.CLOSED


@pytest.mark.asyncio
async def test_half_open_limits_parallel_probe_permits() -> None:
    breaker = CircuitBreaker(half_open_requests=2)
    breaker._state = CircuitState.HALF_OPEN
    started = asyncio.Event()
    release = asyncio.Event()
    active = 0

    async def probe():
        nonlocal active
        active += 1
        if active == 2:
            started.set()
        await release.wait()
        return "ok"

    first = asyncio.create_task(breaker.execute(probe))
    second = asyncio.create_task(breaker.execute(probe))
    await started.wait()
    with pytest.raises(CircuitOpenError):
        await breaker.execute(probe)
    release.set()
    assert await first == "ok"
    assert await second == "ok"


@pytest.mark.asyncio
async def test_cancelled_half_open_probe_releases_its_permit() -> None:
    breaker = CircuitBreaker(half_open_requests=1)
    breaker._state = CircuitState.HALF_OPEN
    entered = asyncio.Event()

    async def blocked_probe():
        entered.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(breaker.execute(blocked_probe))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    async def succeeding_probe():
        return "ok"

    assert await breaker.execute(succeeding_probe) == "ok"


@pytest.mark.asyncio
async def test_stale_half_open_result_cannot_close_a_reopened_breaker() -> None:
    breaker = CircuitBreaker(half_open_requests=2)
    breaker._state = CircuitState.HALF_OPEN
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_success():
        entered.set()
        await release.wait()
        return "late"

    late = asyncio.create_task(breaker.execute(slow_success))
    await entered.wait()

    async def failing_probe():
        raise http_error(503)

    with pytest.raises(httpx.HTTPStatusError):
        await breaker.execute(failing_probe, count_failure=_is_countable_cms_failure)
    assert breaker.state is CircuitState.OPEN
    release.set()
    assert await late == "late"
    assert breaker.state is CircuitState.OPEN
