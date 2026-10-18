"""OAuth helpers for remote MCP servers."""

import asyncio
import fcntl
import json
import logging
import os
import stat
import time
import webbrowser
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, cast
from urllib.parse import parse_qs, urlparse, urlunparse

import httpx2
from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import AuthorizationCodeResult, OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from pydantic import AnyUrl

from .output import console_print

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OAuthConnectionConfig:
    """Provider-specific OAuth settings for a remote MCP server."""

    registration_endpoint: str | None = None
    callback_port: int = 8765
    authorization_endpoint: str | None = None
    token_endpoint: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    token_endpoint_auth_method: str = "client_secret_post"
    scopes: list[str] | None = None
    normalize_root_issuer: bool = False
    assume_authorization_response_issuer: bool = False


class OAuthFlowLock:
    """Serialize one remote server's OAuth flow across local ai-assist processes."""

    def __init__(self, path: Path):
        self.path = path
        self._file: IO[str] | None = None

    async def __aenter__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._file = await asyncio.to_thread(open, self.path, "a+")
        try:
            while True:
                try:
                    await asyncio.to_thread(fcntl.flock, self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return
                except BlockingIOError:
                    await asyncio.sleep(0.1)
        except BaseException:
            await asyncio.to_thread(self._file.close)
            self._file = None
            raise

    async def __aexit__(self, *_exc_info) -> None:
        file = self._file
        assert file is not None
        await asyncio.to_thread(fcntl.flock, file.fileno(), fcntl.LOCK_UN)
        await asyncio.to_thread(file.close)
        self._file = None


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
        # ``expires_in`` is relative to the time the token response was received.
        # The MCP SDK keeps its calculated absolute expiry only in memory, so retain
        # it here to prevent a restarted client from presenting an expired token.
        data["token_expiry_time"] = time.time() + int(tokens.expires_in) if tokens.expires_in is not None else None
        self._write(data)

    async def get_token_expiry_time(self) -> float | None:
        value = self._read().get("token_expiry_time")
        return float(value) if value is not None else None

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
        self.redirect_uri = f"http://127.0.0.1:{self.port}/callback"

    async def __aenter__(self) -> OAuthCallbackServer:
        await self.start()
        return self

    async def __aexit__(self, *_exc_info) -> None:
        await self.close()

    async def start(self) -> None:
        if self._server:
            return
        self._result = asyncio.get_running_loop().create_future()
        self._server = await asyncio.start_server(self._handle_request, "127.0.0.1", self.port)

    async def close(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

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

    async def callback(self) -> AuthorizationCodeResult:
        assert self._result is not None
        code, state = await self._result
        return AuthorizationCodeResult(code=code, state=state)


def replace_authorization_endpoint(authorization_url: str, endpoint: str | None) -> str:
    """Use a configured endpoint while preserving the SDK's OAuth query parameters."""
    if not endpoint:
        return authorization_url
    original = urlparse(authorization_url)
    configured = urlparse(endpoint)
    return urlunparse(configured._replace(query=original.query, fragment=""))


def normalize_root_issuer(issuer: str) -> str:
    """Normalize a trailing slash only when it denotes a URL origin's root."""
    parsed = urlparse(issuer)
    if parsed.path == "/" and not parsed.params and not parsed.query and not parsed.fragment:
        return urlunparse(parsed._replace(path=""))
    return issuer


def populate_missing_response_issuer(
    result: AuthorizationCodeResult, issuer: str | None, enabled: bool
) -> AuthorizationCodeResult:
    """Fill a known issuer for providers that omit RFC 9207's callback parameter."""
    if enabled and result.iss is None and issuer:
        return result.model_copy(update={"iss": issuer})
    return result


class ConfiguredOAuthClientProvider(OAuthClientProvider):
    """OAuth provider with compatibility for imperfect discovery and token metadata."""

    def __init__(
        self,
        *args,
        token_endpoint: str | None = None,
        normalize_root_issuer: bool = False,
        authorization_lock_path: Path,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._configured_token_endpoint = token_endpoint
        self._normalize_root_issuer = normalize_root_issuer
        self._authorization_lock_path = authorization_lock_path

    async def _initialize(self) -> None:
        """Restore the absolute expiry that the base provider keeps only in memory."""
        await super()._initialize()
        if self.context.current_tokens and self.context.current_tokens.expires_in is not None:
            expiry_time = await cast(MCPTokenStorage, self.context.storage).get_token_expiry_time()
            # Caches written before absolute expiry persistence are conservatively
            # expired.  This lets their refresh token recover the session instead
            # of sending an already-expired access token and starting browser OAuth.
            self.context.token_expiry_time = expiry_time if expiry_time is not None else -1.0

    def _get_token_endpoint(self) -> str:
        return self._configured_token_endpoint or super()._get_token_endpoint()

    def _expected_issuer(self) -> str:
        issuer = super()._expected_issuer()
        return normalize_root_issuer(issuer) if self._normalize_root_issuer else issuer

    async def _exchange_token_authorization_code(self, auth_code: str, code_verifier: str) -> httpx2.Request:
        request = await super()._exchange_token_authorization_code(auth_code, code_verifier)
        request.headers["Accept"] = "application/json"
        return request

    async def _refresh_token(self) -> httpx2.Request:
        request = await super()._refresh_token()
        request.headers["Accept"] = "application/json"
        return request

    @asynccontextmanager
    async def _authorization_lock_if_needed(self) -> AsyncIterator[None]:
        if not self._initialized:
            await self._initialize()
        if self.context.is_token_valid():
            yield
        else:
            async with OAuthFlowLock(self._authorization_lock_path):
                self._initialized = False
                await self._initialize()
                yield

    async def _auth_flow(self, request):
        """Relay the parent auth generator while retaining its response values."""
        if request.method == "DELETE":
            # streamable_http_client sends DELETE while closing a session.  A 401
            # from that best-effort cleanup request must not open a browser as the
            # application is shutting down.
            if not self._initialized:
                await self._initialize()
            if self.context.is_token_valid():
                self._add_auth_header(request)
            yield request
            return

        async with self._authorization_lock_if_needed():
            flow = super()._auth_flow(request)
            try:
                outgoing = await anext(flow)
                while True:
                    response = yield outgoing
                    outgoing = await flow.asend(response)
            except StopAsyncIteration:
                return
            finally:
                await flow.aclose()


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


async def store_pre_registered_client(
    storage: MCPTokenStorage,
    metadata: OAuthClientMetadata,
    client_id: str,
    client_secret: str | None,
    token_endpoint_auth_method: str,
) -> None:
    """Persist credentials supplied by a provider instead of using dynamic registration."""
    existing = await storage.get_client_info()
    if (
        existing
        and existing.client_id == client_id
        and existing.client_secret == client_secret
        and existing.redirect_uris == metadata.redirect_uris
        and existing.token_endpoint_auth_method == token_endpoint_auth_method
    ):
        return

    if existing:
        await storage.clear()
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uris=metadata.redirect_uris,
            token_endpoint_auth_method=token_endpoint_auth_method,
        )
    )


@asynccontextmanager
async def oauth_streamable_http_client(
    server_url: str,
    storage_path: Path,
    config: OAuthConnectionConfig = OAuthConnectionConfig(),
) -> AsyncIterator[httpx2.AsyncClient]:
    """Build an OAuth-authenticated HTTP client for an MCP streamable HTTP connection."""
    callback = OAuthCallbackServer(config.callback_port)
    storage = MCPTokenStorage(storage_path)
    authorization_lock_path = storage_path.with_suffix(".lock")
    try:
        metadata = OAuthClientMetadata(
            redirect_uris=[AnyUrl(callback.redirect_uri)],
            client_name="ai-assist",
            scope=" ".join(config.scopes) if config.scopes else None,
        )
        async with OAuthFlowLock(authorization_lock_path):
            if config.client_id:
                await store_pre_registered_client(
                    storage,
                    metadata,
                    config.client_id,
                    config.client_secret,
                    config.token_endpoint_auth_method,
                )
            elif config.registration_endpoint:
                await register_oauth_client(config.registration_endpoint, metadata, storage)

        async def redirect(authorization_url: str) -> None:
            await callback.start()
            await callback.redirect(replace_authorization_endpoint(authorization_url, config.authorization_endpoint))

        async def callback_handler() -> AuthorizationCodeResult:
            try:
                result = await callback.callback()
                metadata = provider.context.oauth_metadata
                issuer = str(metadata.issuer) if metadata else None
                return populate_missing_response_issuer(result, issuer, config.assume_authorization_response_issuer)
            finally:
                await callback.close()

        provider = ConfiguredOAuthClientProvider(
            server_url,
            metadata,
            storage,
            redirect_handler=redirect,
            callback_handler=callback_handler,
            token_endpoint=config.token_endpoint,
            normalize_root_issuer=config.normalize_root_issuer,
            authorization_lock_path=authorization_lock_path,
        )
        async with httpx2.AsyncClient(auth=cast(httpx2.Auth, provider)) as client:
            yield client
    finally:
        await callback.close()
