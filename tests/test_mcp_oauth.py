"""Tests for remote MCP OAuth token persistence."""

import stat

import pytest
from mcp.shared.auth import AuthorizationCodeResult, OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from pydantic import AnyUrl

from ai_assist.mcp_oauth import (
    ConfiguredOAuthClientProvider,
    MCPTokenStorage,
    normalize_root_issuer,
    populate_missing_response_issuer,
    register_oauth_client,
    replace_authorization_endpoint,
    store_pre_registered_client,
)


@pytest.mark.asyncio
async def test_token_storage_round_trips_credentials_with_private_permissions(tmp_path):
    storage = MCPTokenStorage(tmp_path / "oauth.json")
    token = OAuthToken(access_token="access", refresh_token="refresh")
    client = OAuthClientInformationFull(client_id="client-id", redirect_uris=["http://127.0.0.1/callback"])

    await storage.set_tokens(token)
    await storage.set_client_info(client)

    assert (await storage.get_tokens()).access_token == "access"
    assert (await storage.get_client_info()).client_id == "client-id"
    assert stat.S_IMODE((tmp_path / "oauth.json").stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_token_storage_returns_none_for_missing_values(tmp_path):
    storage = MCPTokenStorage(tmp_path / "oauth.json")

    assert await storage.get_tokens() is None
    assert await storage.get_client_info() is None


@pytest.mark.asyncio
async def test_registration_endpoint_persists_client_credentials(tmp_path, monkeypatch):
    class Response:
        status_code = 201

        @staticmethod
        def json():
            return {
                "client_id": "client-id",
                "client_secret": "client-secret",
                "redirect_uris": ["http://127.0.0.1:8765/callback"],
            }

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, endpoint, json):
            assert endpoint == "https://auth.example.test/register"
            assert json["redirect_uris"] == ["http://127.0.0.1:8765/callback"]
            return Response()

    monkeypatch.setattr("ai_assist.mcp_oauth.httpx2.AsyncClient", Client)
    storage = MCPTokenStorage(tmp_path / "oauth.json")
    metadata = OAuthClientMetadata(redirect_uris=[AnyUrl("http://127.0.0.1:8765/callback")])

    await register_oauth_client("https://auth.example.test/register", metadata, storage)

    assert (await storage.get_client_info()).client_secret == "client-secret"


def test_explicit_authorization_endpoint_preserves_oauth_query_parameters():
    result = replace_authorization_endpoint(
        "https://wrong.example/authorize?client_id=client&state=state",
        "https://www.tradingview.com/mcp/oauth/authorize",
    )

    assert result == "https://www.tradingview.com/mcp/oauth/authorize?client_id=client&state=state"


def test_normalize_root_issuer_removes_only_a_trailing_root_slash():
    assert normalize_root_issuer("https://accounts.google.com/") == "https://accounts.google.com"
    assert normalize_root_issuer("https://example.test/oauth/") == "https://example.test/oauth/"


def test_configured_provider_normalizes_opted_in_root_issuer(tmp_path):
    provider = ConfiguredOAuthClientProvider(
        "https://gmailmcp.googleapis.com/mcp/v1",
        OAuthClientMetadata(redirect_uris=[AnyUrl("http://127.0.0.1:8765/callback")]),
        MCPTokenStorage(tmp_path / "oauth.json"),
        normalize_root_issuer=True,
    )
    provider.context.auth_server_url = "https://accounts.google.com/"

    assert provider._expected_issuer() == "https://accounts.google.com"


@pytest.mark.asyncio
async def test_configured_provider_requests_json_token_response(tmp_path):
    provider = ConfiguredOAuthClientProvider(
        "https://mcp.example.test",
        OAuthClientMetadata(redirect_uris=[AnyUrl("http://127.0.0.1:8765/callback")]),
        MCPTokenStorage(tmp_path / "oauth.json"),
    )
    provider.context.client_info = OAuthClientInformationFull(
        client_id="client-id",
        client_secret="client-secret",
        redirect_uris=["http://127.0.0.1:8765/callback"],
        token_endpoint_auth_method="client_secret_post",
    )

    request = await provider._exchange_token_authorization_code("code", "verifier")

    assert request.headers["Accept"] == "application/json"

    provider.context.current_tokens = OAuthToken(access_token="access", refresh_token="refresh")
    refresh_request = await provider._refresh_token()

    assert refresh_request.headers["Accept"] == "application/json"


def test_populate_missing_response_issuer_is_opt_in():
    result = AuthorizationCodeResult(code="code", state="state")

    assert populate_missing_response_issuer(result, "https://accounts.google.com", False).iss is None
    assert (
        populate_missing_response_issuer(result, "https://accounts.google.com", True).iss
        == "https://accounts.google.com"
    )


@pytest.mark.asyncio
async def test_pre_registered_client_is_saved_for_oauth_flow(tmp_path):
    storage = MCPTokenStorage(tmp_path / "oauth.json")
    metadata = OAuthClientMetadata(redirect_uris=[AnyUrl("http://127.0.0.1:8765/callback")])

    await store_pre_registered_client(
        storage,
        metadata,
        client_id="google-client-id",
        client_secret="google-client-secret",
        token_endpoint_auth_method="client_secret_post",
    )

    client = await storage.get_client_info()
    assert client.client_id == "google-client-id"
    assert client.client_secret == "google-client-secret"
    assert client.token_endpoint_auth_method == "client_secret_post"
