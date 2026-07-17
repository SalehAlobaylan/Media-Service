"""ARQ connection settings and lifecycle-managed API queue access."""
from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlparse

from arq.connections import RedisSettings
from redis.asyncio import Redis

from src.config import Settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_redis_settings(settings: Settings) -> RedisSettings:
    """Preserve Redis DSN security details while reserving Media database 2."""
    parsed_url = urlparse(settings.REDIS_URL)
    parsed = Redis.from_url(settings.REDIS_URL).connection_pool.connection_kwargs
    # RedisSettings does not accept every redis-py tuning option, but it does
    # carry the identity and TLS/certificate settings that affect safe access.
    supported = {
        "host",
        "port",
        "username",
        "password",
        "ssl_cert_reqs",
        "ssl_ca_certs",
        "ssl_ca_data",
        "ssl_certfile",
        "ssl_keyfile",
        "ssl_check_hostname",
        "max_connections",
    }
    kwargs = {key: value for key, value in parsed.items() if key in supported}
    if parsed_url.scheme == "unix":
        kwargs["unix_socket_path"] = parsed_url.path
    else:
        kwargs["ssl"] = parsed_url.scheme == "rediss"
    kwargs["database"] = settings.ARQ_REDIS_DB
    return RedisSettings(**kwargs)


CreatePool = Callable[[RedisSettings], Awaitable[Any]]


class ArqPoolManager:
    """Keep async submission available across temporary Redis outages."""

    def __init__(
        self,
        settings: Settings,
        create_pool: CreatePool | None = None,
        *,
        base_delay_seconds: float = 0.25,
        max_delay_seconds: float = 10.0,
    ) -> None:
        self._settings = settings
        self._create_pool = create_pool
        self._base_delay = base_delay_seconds
        self._max_delay = max_delay_seconds
        self._pool: Any | None = None
        self._lock = asyncio.Lock()
        self._closed = False
        self._task: asyncio.Task[None] | None = None

    @property
    def pool(self) -> Any | None:
        return self._pool

    @property
    def reachable(self) -> bool:
        return self._pool is not None

    async def start(self) -> None:
        await self._connect_once()
        self._task = asyncio.create_task(self._reconnect_loop(), name="media-arq-reconnect")

    async def get_pool(self) -> Any | None:
        if self._pool is None:
            await self._connect_once()
        return self._pool

    async def mark_unavailable(self, pool: Any | None = None) -> None:
        async with self._lock:
            if pool is not None and pool is not self._pool:
                return
            stale, self._pool = self._pool, None
        if stale is not None:
            try:
                await stale.aclose()
            except Exception:
                pass

    async def aclose(self) -> None:
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.mark_unavailable()

    async def _connect_once(self) -> bool:
        if self._closed:
            return False
        async with self._lock:
            if self._pool is not None:
                return True
            try:
                create_pool = self._create_pool
                if create_pool is None:
                    from arq import create_pool as arq_create_pool

                    create_pool = arq_create_pool
                self._pool = await create_pool(build_redis_settings(self._settings))
                logger.info("arq_pool_ready", db=self._settings.ARQ_REDIS_DB)
                return True
            except Exception as exc:
                logger.warning("arq_pool_unreachable", error=str(exc))
                return False

    async def _reconnect_loop(self) -> None:
        delay = self._base_delay
        while not self._closed:
            if self._pool is not None:
                delay = self._base_delay
                await asyncio.sleep(1)
                continue
            connected = await self._connect_once()
            if connected:
                delay = self._base_delay
                continue
            # A small jitter keeps replicas from retrying in lockstep.
            await asyncio.sleep(delay * (1 + random.uniform(0, 0.2)))
            delay = min(self._max_delay, delay * 2)
