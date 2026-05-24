from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.clients.cms import CMSClient
from src.config import Settings
from src.main import app


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        SERVICE_AUTH_TOKEN="test-token",
        CMS_SERVICE_TOKEN="test-cms-token",
        CMS_BASE_URL="http://localhost:8080",
        MODELS_DIR="./test-models",
        ENV="test",
    )


@pytest.fixture
def mock_model_manager() -> MagicMock:
    manager = MagicMock()
    manager.is_ready = {"whisper": True, "clip": True}
    manager.all_ready = True

    # Whisper mock
    manager.whisper.is_loaded = True
    manager.whisper.model_size = "base"

    # CLIP mock
    manager.clip.is_loaded = True
    manager.clip.model_name = "clip-ViT-B-32"
    manager.clip.dimensions = 512

    return manager


@pytest.fixture
def mock_cms_client() -> AsyncMock:
    client = AsyncMock(spec=CMSClient)
    client.health_check.return_value = True
    client.create_transcript.return_value = {"id": "transcript-123"}
    client.link_transcript.return_value = {"ok": True}
    client.store_image_embedding.return_value = {"ok": True}
    client.update_status.return_value = {"ok": True}
    return client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


@pytest.fixture
def client(
    test_settings: Settings,
    mock_model_manager: MagicMock,
    mock_cms_client: AsyncMock,
) -> TestClient:
    app.state.settings = test_settings
    app.state.model_manager = mock_model_manager
    app.state.cms_client = mock_cms_client
    app.state.arq_pool = None  # async transcribe endpoints will 503 in tests
    return TestClient(app, raise_server_exceptions=False)
