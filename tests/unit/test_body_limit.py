import pytest

from src.middleware.body_limit import RequestBodyLimitMiddleware


@pytest.mark.asyncio
async def test_transcription_body_limit_rejects_chunked_oversize_before_app() -> None:
    called = False
    sent: list[dict] = []
    messages = iter(
        [
            {"type": "http.request", "body": b"abc", "more_body": True},
            {"type": "http.request", "body": b"def", "more_body": False},
        ]
    )

    async def receive() -> dict:
        return next(messages)

    async def send(message: dict) -> None:
        sent.append(message)

    async def downstream(scope, receive, send) -> None:
        nonlocal called
        called = True
        await receive()
        await receive()

    middleware = RequestBodyLimitMiddleware(downstream, max_bytes=5)
    await middleware({"type": "http", "path": "/v1/transcribe/jobs", "headers": []}, receive, send)

    assert called is True
    assert sent[0]["status"] == 413


@pytest.mark.asyncio
async def test_transcription_body_limit_rejects_declared_oversize_without_downstream() -> None:
    called = False
    sent: list[dict] = []

    async def receive() -> dict:
        raise AssertionError("body must not be read after a declared oversize")

    async def send(message: dict) -> None:
        sent.append(message)

    async def downstream(scope, receive, send) -> None:
        nonlocal called
        called = True

    middleware = RequestBodyLimitMiddleware(downstream, max_bytes=5)
    await middleware(
        {"type": "http", "path": "/v1/transcribe", "headers": [(b"content-length", b"6")]},
        receive,
        send,
    )

    assert called is False
    assert sent[0]["status"] == 413
