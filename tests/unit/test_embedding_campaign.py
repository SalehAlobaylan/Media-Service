import json
from io import BytesIO
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image

from src.clients.cms import CMSClient
from src.config import Settings

CONTENT_ID = "dba3dd23-9718-4a69-a06d-7551f22ef271"
REQUEST_ID = "dba3dd23-9718-4a69-a06d-7551f22ef272"
CAMPAIGN = {"campaign_id": 17, "action_id": 23, "request_id": REQUEST_ID}


def configure_clip(client):
    clip = client.app.state.model_manager.clip
    clip.encode_image.return_value = [0.1] * 512
    clip.space_descriptor.return_value = {
        "model": clip.model_name,
        "revision": "a" * 40,
        "dimensions": 512,
        "normalized": True,
        "space_id": "space",
        "producer_id": "producer",
    }
    image = BytesIO()
    Image.new("RGB", (2, 2)).save(image, format="PNG")
    return image.getvalue()


@pytest.mark.parametrize("source", ["upload", "url"])
@pytest.mark.parametrize("campaign", [CAMPAIGN, None])
def test_image_writeback_preserves_campaign_without_changing_legacy_calls(
    client,
    auth_headers,
    mock_cms_client,
    source,
    campaign,
):
    image = configure_clip(client)
    data = {"content_id": CONTENT_ID}
    if campaign is not None:
        data["embedding_campaign_json"] = json.dumps(campaign)
    files = None
    if source == "url":
        client.app.state.fetch_client = AsyncMock()
        client.app.state.fetch_client.get_bytes.return_value = image
        data["url"] = "https://images.test/image.png"
    else:
        files = {"image_file": ("image.png", image, "image/png")}

    response = client.post(
        "/v1/embed/image", data=data, files=files, headers=auth_headers
    )

    assert response.status_code == 200
    assert response.json()["write_back_status"] == "ok"
    mock_cms_client.store_image_embedding.assert_awaited_once()
    kwargs = mock_cms_client.store_image_embedding.call_args.kwargs
    if campaign is None:
        assert "embedding_campaign" not in kwargs
    else:
        assert kwargs["embedding_campaign"] == campaign
    assert kwargs["content_stage"] is None
    assert kwargs["artifact_recovery"] is None


@pytest.mark.parametrize(
    "correlation", ["content_stage_json", "artifact_recovery_json"]
)
def test_image_campaign_rejects_other_correlations_before_inference(
    client,
    auth_headers,
    correlation,
):
    image = configure_clip(client)
    response = client.post(
        "/v1/embed/image",
        headers=auth_headers,
        data={
            "content_id": CONTENT_ID,
            "embedding_campaign_json": json.dumps(CAMPAIGN),
            correlation: "{}",
        },
        files={"image_file": ("image.png", image, "image/png")},
    )
    assert response.status_code == 422
    client.app.state.model_manager.clip.encode_image.assert_not_called()


def test_image_campaign_requires_target_before_inference(client, auth_headers):
    image = configure_clip(client)
    response = client.post(
        "/v1/embed/image",
        headers=auth_headers,
        data={"embedding_campaign_json": json.dumps(CAMPAIGN)},
        files={"image_file": ("image.png", image, "image/png")},
    )
    assert response.status_code == 422
    client.app.state.model_manager.clip.encode_image.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("campaign", [CAMPAIGN, None])
async def test_cms_image_client_preserves_opaque_campaign(campaign):
    cms = CMSClient(
        Settings(CMS_BASE_URL="http://cms.test", CMS_SERVICE_TOKEN="test-token")
    )
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"ok": True})

    await cms.client.aclose()
    cms.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        await cms.store_image_embedding(CONTENT_ID, [0.1], embedding_campaign=campaign)
        assert (
            str(requests[0].url)
            == f"http://cms.test/internal/content-items/{CONTENT_ID}/image-embedding"
        )
        payload = json.loads(requests[0].content)
        if campaign is None:
            assert "embedding_campaign" not in payload
        else:
            assert payload["embedding_campaign"] == campaign
    finally:
        await cms.close()
