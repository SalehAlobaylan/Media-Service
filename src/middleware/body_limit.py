"""ASGI request-body guard for media uploads before multipart spooling."""
from __future__ import annotations

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestBodyLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] not in {
            "/v1/transcribe",
            "/v1/transcribe/jobs",
        }:
            await self.app(scope, receive, send)
            return

        content_length = next(
            (value for key, value in scope.get("headers", []) if key == b"content-length"), None
        )
        if content_length and content_length.isdigit() and int(content_length) > self.max_bytes:
            await JSONResponse(
                {"error": "Request body exceeds the configured upload limit"}, status_code=413
            )(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestBodyTooLarge:
            await JSONResponse(
                {"error": "Request body exceeds the configured upload limit"}, status_code=413
            )(scope, receive, send)


class _RequestBodyTooLarge(Exception):
    pass
