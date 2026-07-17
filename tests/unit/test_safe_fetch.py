from unittest.mock import AsyncMock

import httpx
import pytest

from src.clients.safe_fetch import SafeFetchClient, _PinnedBackend
from src.utils.url_guard import UnsafeURLError, validate_public_url


@pytest.mark.asyncio
async def test_pinned_backend_dials_the_validated_address_not_the_hostname(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.clients.safe_fetch as module

    backend = _PinnedBackend()
    backend._delegate.connect_tcp = AsyncMock(return_value="stream")
    monkeypatch.setattr(module, "validate_public_host", lambda host, port: "203.0.113.8")

    assert await backend.connect_tcp("media.example", 443, timeout=2) == "stream"
    backend._delegate.connect_tcp.assert_awaited_once_with("203.0.113.8", 443, timeout=2)


@pytest.mark.asyncio
async def test_safe_fetch_rejects_redirects_and_streamed_oversize(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.clients.safe_fetch as module

    monkeypatch.setattr(module, "validate_public_url", lambda url: url)

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/redirect":
            return httpx.Response(302, headers={"location": "https://private.example"})
        return httpx.Response(200, content=b"12345")

    fetcher = SafeFetchClient()
    await fetcher._client.aclose()
    fetcher._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="redirects"):
        await fetcher.get_bytes("https://example.test/redirect", 10)
    with pytest.raises(ValueError, match="allowed size"):
        await fetcher.get_bytes("https://example.test/large", 4)
    await fetcher.aclose()


def test_url_guard_rejects_userinfo_and_private_rebinding_answer(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.utils.url_guard as guard

    with pytest.raises(UnsafeURLError, match="userinfo"):
        validate_public_url("https://user:secret@example.test/audio.mp3")
    monkeypatch.setattr(
        guard.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(None, None, None, None, ("127.0.0.1", 443))],
    )
    with pytest.raises(UnsafeURLError, match="non-public"):
        validate_public_url("https://rebind.example/audio.mp3")
