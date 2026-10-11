"""Tests for remote MCP OAuth token persistence."""

import stat

import pytest
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from pydantic import AnyUrl

from ai_assist.mcp_oauth import MCPTokenStorage, register_oauth_client


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
