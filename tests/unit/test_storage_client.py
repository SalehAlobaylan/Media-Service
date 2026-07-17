import io

import pytest

from src.clients.storage import StorageClient, StorageUploadTooLargeError


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.deleted: list[str] = []

    def upload_fileobj(self, stream, bucket, key, ExtraArgs):  # noqa: N803
        self.objects[key] = stream.read()

    def delete_object(self, Bucket, Key):  # noqa: N803
        self.deleted.append(Key)
        self.objects.pop(Key, None)


def storage_with(fake: FakeS3) -> StorageClient:
    client = object.__new__(StorageClient)
    client._client = fake
    client._bucket = "test-bucket"
    return client


@pytest.mark.asyncio
async def test_upload_fileobj_enforces_actual_byte_cap_and_cleans_partial_object() -> None:
    fake = FakeS3()
    storage = storage_with(fake)

    with pytest.raises(StorageUploadTooLargeError):
        await storage.upload_fileobj(io.BytesIO(b"12345"), "owned-key", max_bytes=4)

    assert fake.deleted == ["owned-key"]
    assert fake.objects == {}


@pytest.mark.asyncio
async def test_upload_fileobj_accepts_exact_byte_cap() -> None:
    fake = FakeS3()
    storage = storage_with(fake)

    await storage.upload_fileobj(io.BytesIO(b"1234"), "owned-key", max_bytes=4)

    assert fake.deleted == []
    assert fake.objects == {"owned-key": b"1234"}
