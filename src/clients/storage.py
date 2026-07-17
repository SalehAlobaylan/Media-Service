"""Object-storage client (Cloudflare R2 / S3-compatible).

Used for async-transcription spooling: the API streams the incoming upload to
R2 and enqueues only the object key; the separate worker process downloads the
object when it runs. This keeps queued audio in object storage instead of the
API container's local disk, and means the API and worker need no shared volume.

boto3 is synchronous, so every call is dispatched to a thread via
`asyncio.to_thread` to avoid blocking the event loop.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import IO, Any

import boto3
from botocore.config import Config

from src.config import Settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class StorageUploadTooLargeError(ValueError):
    """Raised when the actual streamed object exceeds the caller's byte cap."""


class _CountingReader:
    """File-object proxy that enforces a cap on bytes consumed by boto3."""

    def __init__(self, source: IO[bytes], max_bytes: int) -> None:
        self._source = source
        self._max_bytes = max_bytes
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._source.read(size)
        self.bytes_read += len(chunk)
        if self.bytes_read > self._max_bytes:
            raise StorageUploadTooLargeError(
                f"Upload exceeds maximum size of {self._max_bytes} bytes"
            )
        return chunk

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)


class StorageClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._bucket = settings.S3_BUCKET
        self._client = None
        if settings.s3_configured:
            self._client = boto3.client(
                "s3",
                endpoint_url=settings.S3_ENDPOINT_URL,
                aws_access_key_id=settings.S3_ACCESS_KEY_ID,
                aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY,
                region_name=settings.S3_REGION or "auto",
                config=Config(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "standard"},
                ),
            )
            logger.info("storage_client_ready", bucket=self._bucket)
        else:
            logger.info("storage_client_disabled", reason="S3_* not configured")

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    async def upload_fileobj(
        self,
        fileobj: IO[bytes],
        key: str,
        content_type: str | None = None,
        max_bytes: int | None = None,
    ) -> None:
        """Upload a request-owned object and compensate for every failure.

        Content-Length is only an early rejection: chunked/misreported uploads
        are constrained by this reader. A failed multipart transfer can leave
        a partial object, so delete the owned key before propagating failure.
        """
        extra = {"ContentType": content_type} if content_type else {}
        source: IO[bytes] = (
            _CountingReader(fileobj, max_bytes) if max_bytes is not None else fileobj
        )
        try:
            await asyncio.to_thread(
                self._client.upload_fileobj, source, self._bucket, key, ExtraArgs=extra
            )
        except BaseException:
            try:
                await self.delete_object(key)
            except Exception:
                logger.warning("storage_partial_upload_cleanup_failed", key=key)
            raise

    async def download_to_path(self, key: str, dest_path: str) -> None:
        await asyncio.to_thread(self._client.download_file, self._bucket, key, dest_path)

    async def delete_object(self, key: str) -> None:
        await asyncio.to_thread(
            self._client.delete_object, Bucket=self._bucket, Key=key
        )

    async def list_objects(self, prefix: str, max_keys: int) -> list[tuple[str, datetime]]:
        """Return one bounded page of object keys and last-modified timestamps."""
        response = await asyncio.to_thread(
            self._client.list_objects_v2,
            Bucket=self._bucket,
            Prefix=prefix,
            MaxKeys=max_keys,
        )
        return [
            (entry["Key"], entry["LastModified"])
            for entry in response.get("Contents", [])
            if entry.get("Key") and entry.get("LastModified")
        ]
