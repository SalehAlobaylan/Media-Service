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
from typing import IO

import boto3
from botocore.config import Config

from src.config import Settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


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
        self, fileobj: IO[bytes], key: str, content_type: str | None = None
    ) -> None:
        extra = {"ContentType": content_type} if content_type else {}
        await asyncio.to_thread(
            self._client.upload_fileobj, fileobj, self._bucket, key, ExtraArgs=extra
        )

    async def download_to_path(self, key: str, dest_path: str) -> None:
        await asyncio.to_thread(self._client.download_file, self._bucket, key, dest_path)

    async def delete_object(self, key: str) -> None:
        await asyncio.to_thread(
            self._client.delete_object, Bucket=self._bucket, Key=key
        )
