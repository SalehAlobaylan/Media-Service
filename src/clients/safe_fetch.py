"""Pooled, byte-bounded HTTP fetches pinned to a verified public address."""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpcore
import httpx

from src.utils.url_guard import validate_public_host, validate_public_url


class _PinnedBackend(httpcore.AsyncNetworkBackend):
    """Resolve at connect time and dial that public IP, retaining original SNI."""

    def __init__(self) -> None:
        self._delegate = httpcore.AnyIOBackend()

    async def connect_tcp(self, host: str, port: int, **kwargs: Any):
        address = validate_public_host(host, port)
        return await self._delegate.connect_tcp(address, port, **kwargs)

    async def connect_unix_socket(self, *args: Any, **kwargs: Any):
        raise httpcore.ConnectError("unix sockets are not allowed for remote media")

    async def sleep(self, seconds: float) -> None:
        await self._delegate.sleep(seconds)


class _PinnedTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self._pool = httpcore.AsyncConnectionPool(
            network_backend=_PinnedBackend(), max_connections=20, max_keepalive_connections=10
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._pool.handle_async_request(
            httpcore.Request(
                method=request.method,
                url=httpcore.URL(
                    scheme=request.url.raw_scheme,
                    host=request.url.raw_host,
                    port=request.url.port,
                    target=request.url.raw_path,
                ),
                headers=request.headers.raw,
                content=request.stream,
                extensions=request.extensions,
            )
        )
        return httpx.Response(
            status_code=response.status,
            headers=response.headers,
            stream=_CoreStream(response.stream),
            extensions=response.extensions,
        )

    async def aclose(self) -> None:
        await self._pool.aclose()


class _CoreStream(httpx.AsyncByteStream):
    def __init__(self, stream: AsyncIterator[bytes]) -> None:
        self._stream = stream

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._stream:
            yield chunk

    async def aclose(self) -> None:
        close = getattr(self._stream, "aclose", None)
        if close is not None:
            await close()


class SafeFetchClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            transport=_PinnedTransport(),
            follow_redirects=False,
            timeout=httpx.Timeout(connect=10, read=30, write=10, pool=10),
        )

    async def get_bytes(
        self, url: str, max_bytes: int, allowed_content_prefixes: tuple[str, ...] = ()
    ) -> bytes:
        validate_public_url(url)
        async with self._client.stream("GET", url) as response:
            if response.is_redirect:
                raise ValueError("redirects are not accepted for media inputs")
            response.raise_for_status()
            declared = response.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise ValueError("remote media exceeds the allowed size")
            content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            if allowed_content_prefixes and not any(
                content_type.startswith(prefix) for prefix in allowed_content_prefixes
            ):
                raise ValueError("remote media has an unsupported content type")
            payload = bytearray()
            async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                payload.extend(chunk)
                if len(payload) > max_bytes:
                    raise ValueError("remote media exceeds the allowed size")
            return bytes(payload)

    async def download_to_path(
        self, url: str, path: str, max_bytes: int, allowed_content_prefixes: tuple[str, ...] = ()
    ) -> None:
        payload = await self.get_bytes(url, max_bytes, allowed_content_prefixes)
        with open(path, "wb") as destination:
            destination.write(payload)

    async def aclose(self) -> None:
        await self._client.aclose()
