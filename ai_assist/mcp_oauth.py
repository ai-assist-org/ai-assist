"""OAuth helpers for remote MCP servers."""

import asyncio
import json
import logging
import os
import stat
import webbrowser
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import httpx2
from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from pydantic import AnyUrl

from .output import console_print

logger = logging.getLogger(__name__)


class MCPTokenStorage:
    """Store one remote MCP server's OAuth credentials in a private JSON file."""

    def __init__(self, path: Path):
        self.path = path

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except FileNotFoundError, json.JSONDecodeError:
            return {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.path.parent.chmod(0o700)
        except OSError:
            pass
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data))
        temporary.chmod(stat.S_IRUSR | stat.S_IWUSR)
        os.replace(temporary, self.path)

    async def get_tokens(self) -> OAuthToken | None:
        value = self._read().get("tokens")
        return OAuthToken.model_validate(value) if value else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        data = self._read()
        data["tokens"] = tokens.model_dump(mode="json")
        self._write(data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        value = self._read().get("client_info")
        return OAuthClientInformationFull.model_validate(value) if value else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        data = self._read()
        data["client_info"] = client_info.model_dump(mode="json")
        self._write(data)

    async def clear(self) -> None:
        self._write({})


class OAuthCallbackServer:
    """Temporary loopback endpoint used by the OAuth authorization-code flow."""

    def __init__(self, port: int):
        self.port = port
        self._result: asyncio.Future[tuple[str, str | None]] | None = None
        self._server: asyncio.AbstractServer | None = None
        self.redirect_uri = ""

    async def __aenter__(self) -> OAuthCallbackServer:
        self._result = asyncio.get_running_loop().create_future()
        self._server = await asyncio.start_server(self._handle_request, "127.0.0.1", self.port)
        self.redirect_uri = f"http://127.0.0.1:{self.port}/callback"
        return self

    async def __aexit__(self, *_exc_info) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_request(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await reader.readline()
            target = request_line.decode("ascii", errors="replace").split(" ")[1]
            query = parse_qs(urlparse(target).query)
            code = query.get("code", [""])[0]
            state = query.get("state", [None])[0]
            if code and self._result and not self._result.done():
                self._result.set_result((code, state))
                body = "Authorization complete. You can close this browser tab."
                status = "200 OK"
            else:
                body = "Authorization failed or was cancelled. Return to ai-assist for details."
                status = "400 Bad Request"
            encoded = body.encode()
            writer.write(
                f"HTTP/1.1 {status}\r\nContent-Type: text/plain\r\nContent-Length: {len(encoded)}\r\nConnection: close\r\n\r\n".encode()
                + encoded
            )
            await writer.drain()
        except IndexError, UnicodeDecodeError:
            logger.warning("Received malformed OAuth callback")
        finally:
            writer.close()
            await writer.wait_closed()

    async def redirect(self, authorization_url: str) -> None:
        console_print(f"\nAuthorize MCP server in your browser:\n{authorization_url}\n")
        opened = await asyncio.to_thread(webbrowser.open, authorization_url)
        if not opened:
            logger.warning("Could not open a browser for MCP OAuth; open the printed URL manually")

    async def callback(self) -> tuple[str, str | None]:
        assert self._result is not None
        return await self._result


async def register_oauth_client(endpoint: str, metadata: OAuthClientMetadata, storage: MCPTokenStorage) -> None:
    """Register a client at an explicitly configured OAuth registration endpoint."""
    existing = await storage.get_client_info()
    if existing and existing.redirect_uris == metadata.redirect_uris:
        return

    if existing:
        await storage.clear()

    async with httpx2.AsyncClient() as client:
        response = await client.post(
            endpoint,
            json=metadata.model_dump(by_alias=True, mode="json", exclude_none=True),
        )
    if response.status_code not in (200, 201):
        raise RuntimeError(f"OAuth client registration failed with HTTP {response.status_code}")
    await storage.set_client_info(OAuthClientInformationFull.model_validate(response.json()))


@asynccontextmanager
async def oauth_streamable_http_client(
    server_url: str,
    storage_path: Path,
    registration_endpoint: str | None = None,
    callback_port: int = 8765,
) -> AsyncIterator[httpx2.AsyncClient]:
    """Build an OAuth-authenticated HTTP client for an MCP streamable HTTP connection."""
    async with OAuthCallbackServer(callback_port) as callback:
        storage = MCPTokenStorage(storage_path)
        metadata = OAuthClientMetadata(
            redirect_uris=[AnyUrl(callback.redirect_uri)],
            client_name="ai-assist",
        )
        if registration_endpoint:
            await register_oauth_client(registration_endpoint, metadata, storage)
        provider = OAuthClientProvider(
            server_url,
            metadata,
            storage,
            redirect_handler=callback.redirect,
            callback_handler=cast(Any, callback.callback),
        )
        async with httpx2.AsyncClient(auth=cast(httpx2.Auth, provider)) as client:
            yield client
