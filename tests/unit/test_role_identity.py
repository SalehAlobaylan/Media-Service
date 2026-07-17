from src.config import Settings


def _production_settings(**overrides: str) -> Settings:
    values = {
        "ENV": "production",
        "MEDIA_ROLE": "api",
        "MEDIA_SERVICE_TOKEN": "media-current",
        "CMS_MEDIA_SERVICE_TOKEN": "cms-writeback",
    }
    values.update(overrides)
    return Settings(**values)


def test_production_rejects_shared_inbound_and_cms_credentials() -> None:
    settings = _production_settings(MEDIA_SERVICE_TOKEN="shared", CMS_MEDIA_SERVICE_TOKEN="shared")
    errors, _ = settings.validate_startup(expected_role="api")
    assert any("must differ" in error for error in errors)


def test_production_accepts_independent_rotating_inbound_credentials() -> None:
    settings = _production_settings(MEDIA_SERVICE_TOKEN_PREVIOUS="media-previous")
    errors, _ = settings.validate_startup(expected_role="api")
    assert errors == []
    assert settings.inbound_service_tokens == ("media-current", "media-previous")


def test_production_never_falls_back_to_cms_token_for_inbound_auth() -> None:
    settings = Settings(ENV="production", MEDIA_ROLE="api", CMS_MEDIA_SERVICE_TOKEN="cms-writeback")
    errors, _ = settings.validate_startup(expected_role="api")
    assert settings.inbound_service_tokens == ()
    assert any("MEDIA_SERVICE_TOKEN" in error for error in errors)


def test_role_and_partial_storage_contract_are_startup_errors() -> None:
    settings = _production_settings(MEDIA_ROLE="worker", S3_BUCKET="only-a-bucket")
    errors, _ = settings.validate_startup(expected_role="api")
    assert any("MEDIA_ROLE" in error for error in errors)
    assert any("S3 configuration" in error for error in errors)


def test_service_auth_accepts_current_and_previous_rotation_tokens(client) -> None:
    previous = client.app.state.settings
    client.app.state.settings = Settings(
        ENV="production",
        MEDIA_SERVICE_TOKEN="media-current",
        MEDIA_SERVICE_TOKEN_PREVIOUS="media-previous",
        CMS_MEDIA_SERVICE_TOKEN="cms-writeback",
    )
    try:
        assert client.get("/v1/models", headers={"Authorization": "Bearer media-current"}).status_code == 200
        assert client.get("/v1/models", headers={"Authorization": "Bearer media-previous"}).status_code == 200
        assert client.get("/v1/models", headers={"Authorization": "Bearer cms-writeback"}).status_code == 401
    finally:
        client.app.state.settings = previous
