import asyncio

import pytest

from src.config import Settings
from src.queue import ArqPoolManager, build_redis_settings


def test_redis_settings_preserve_acl_tls_ipv6_and_override_database() -> None:
    settings = Settings(
        REDIS_URL=(
            "rediss://media%40user:p%40ss@[2001:db8::1]:6380/7?"
            "ssl_cert_reqs=required&ssl_check_hostname=true&ssl_ca_certs=%2Fetc%2Fca.pem"
        ),
        ARQ_REDIS_DB=2,
    )
    redis = build_redis_settings(settings)
    assert redis.host == "2001:db8::1"
    assert redis.port == 6380
    assert redis.username == "media@user"
    assert redis.password == "p@ss"
    assert redis.ssl is True
    assert redis.ssl_check_hostname is True
    assert redis.ssl_ca_certs == "/etc/ca.pem"
    assert redis.database == 2


class _Pool:
    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_queue_manager_recovers_after_down_at_boot() -> None:
    attempts = 0

    async def create_pool(_settings):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("redis unavailable")
        return _Pool()

    manager = ArqPoolManager(
        Settings(), create_pool, base_delay_seconds=0.001, max_delay_seconds=0.002
    )
    await manager.start()
    for _ in range(20):
        if manager.reachable:
            break
        await asyncio.sleep(0.005)
    try:
        assert attempts >= 2
        assert manager.reachable is True
        assert await manager.get_pool() is not None
    finally:
        await manager.aclose()
