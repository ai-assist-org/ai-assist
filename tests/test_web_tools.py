"""Tests for the web tools (web_search + web_fetch) and their providers."""

from __future__ import annotations

from types import SimpleNamespace

import httpx

from ai_assist.web_tools import WebTools, web_search_configured
from ai_assist.web_tools.providers import brave, duckduckgo, exa, firecrawl, http_fetch, perplexity, searxng, tavily


def make_config(**overrides):
    """A minimal config stand-in with the web-tool fields."""
    base = {
        "tavily_api_key": None,
        "brave_api_key": None,
        "exa_api_key": None,
        "perplexity_api_key": None,
        "firecrawl_api_key": None,
        "web_searxng_url": None,
        "web_search_provider": "auto",
        "web_fetch_provider": "auto",
        "web_search_enabled": True,
        "web_fetch_enabled": True,
        "web_fetch_max_chars": 20000,
        "web_timeout_seconds": 30,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeResponse:
    def __init__(self, *, json_data=None, text="", status_code=200, headers=None, url=""):
        self._json = json_data
        self.text = text
        self.status_code = status_code
        self.headers = headers or {}
        self.url = url or "https://example.com"

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=self)


class _FakeClient:
    """Stands in for httpx.AsyncClient used as an async context manager.

    Behavior is per-method so a single patch of ``httpx.AsyncClient`` can drive
    both providers at once (Tavily uses ``post``, the HTTP provider uses ``get``).
    """

    def __init__(self, *, post_response=None, post_error=None, get_response=None, get_error=None):
        self._post_response = post_response
        self._post_error = post_error
        self._get_response = get_response
        self._get_error = get_error

    def __call__(self, *args, **kwargs):  # AsyncClient(...) call
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, *args, **kwargs):
        if self._post_error:
            raise self._post_error
        return self._post_response

    async def get(self, *args, **kwargs):
        if self._get_error:
            raise self._get_error
        return self._get_response


# --- Definitions / gating -------------------------------------------------


class TestWebToolDefinitions:
    def test_both_offered_without_key(self):
        # web_fetch is keyless (http) and web_search is keyless (DuckDuckGo fallback).
        defs = WebTools(make_config()).get_tool_definitions()
        names = [d["name"] for d in defs]
        assert "internal__web_fetch" in names
        assert "internal__web_search" in names

    def test_search_offered_with_tavily_key(self):
        defs = WebTools(make_config(tavily_api_key="k")).get_tool_definitions()
        names = [d["name"] for d in defs]
        assert "internal__web_search" in names
        assert "internal__web_fetch" in names

    def test_fetch_disabled(self):
        names = [d["name"] for d in WebTools(make_config(web_fetch_enabled=False)).get_tool_definitions()]
        assert "internal__web_fetch" not in names

    def test_both_disabled(self):
        cfg = make_config(web_fetch_enabled=False, web_search_enabled=False)
        assert WebTools(cfg).get_tool_definitions() == []

    def test_search_disabled_even_with_key(self):
        cfg = make_config(tavily_api_key="k", web_search_enabled=False)
        names = [d["name"] for d in WebTools(cfg).get_tool_definitions()]
        assert "internal__web_search" not in names

    def test_definition_metadata(self):
        defs = WebTools(make_config(tavily_api_key="k")).get_tool_definitions()
        for d in defs:
            assert d["_server"] == "internal"
            assert d["_readonly"] is True
            assert d["name"] == f"internal__{d['_original_name']}"
            assert d["input_schema"]["type"] == "object"

    def test_web_search_configured_helper(self):
        # Keyless DuckDuckGo makes search available even without a Tavily key.
        assert web_search_configured(make_config(tavily_api_key="k")) is True
        assert web_search_configured(make_config()) is True
        assert web_search_configured(make_config(web_search_enabled=False)) is False


# --- Execution ------------------------------------------------------------


class TestWebSearchExecution:
    async def test_search_success(self, mocker):
        response = _FakeResponse(
            json_data={
                "answer": "Python 3.14 is the latest.",
                "results": [
                    {"title": "Python 3.14", "url": "https://python.org", "content": "release notes"},
                ],
            }
        )
        mocker.patch.object(tavily.httpx, "AsyncClient", _FakeClient(post_response=response))
        out = await WebTools(make_config(tavily_api_key="k")).execute_tool("web_search", {"query": "latest python"})
        assert "Python 3.14 is the latest." in out
        assert "https://python.org" in out
        assert "provider: tavily" in out

    async def test_search_requires_query(self):
        out = await WebTools(make_config(tavily_api_key="k")).execute_tool("web_search", {"query": "  "})
        assert out.startswith("Error:")

    async def test_search_unavailable_when_disabled(self):
        out = await WebTools(make_config(web_search_enabled=False)).execute_tool("web_search", {"query": "x"})
        assert "unavailable" in out

    async def test_search_provider_error(self, mocker):
        mocker.patch.object(tavily.httpx, "AsyncClient", _FakeClient(post_error=httpx.ConnectError("boom")))
        out = await WebTools(make_config(tavily_api_key="k")).execute_tool("web_search", {"query": "x"})
        assert out.startswith("Error: web search failed")

    async def test_search_duckduckgo_fallback(self, mocker):
        # No Tavily key, no SearXNG url -> keyless DuckDuckGo provider is selected.
        raw = [{"title": "Py", "href": "https://python.org", "body": "the site"}]
        mocker.patch.object(duckduckgo.DuckDuckGoProvider, "_search_sync", staticmethod(lambda *a, **k: raw))
        out = await WebTools(make_config()).execute_tool("web_search", {"query": "python"})
        assert "https://python.org" in out
        assert "provider: duckduckgo" in out

    async def test_search_brave(self, mocker):
        response = _FakeResponse(
            json_data={"web": {"results": [{"title": "Py", "url": "https://python.org", "description": "desc"}]}}
        )
        mocker.patch.object(brave.httpx, "AsyncClient", _FakeClient(get_response=response))
        out = await WebTools(make_config(brave_api_key="k")).execute_tool("web_search", {"query": "python"})
        assert "https://python.org" in out
        assert "provider: brave" in out

    async def test_search_exa(self, mocker):
        response = _FakeResponse(
            json_data={"results": [{"title": "Py", "url": "https://python.org", "text": "neural"}]}
        )
        mocker.patch.object(exa.httpx, "AsyncClient", _FakeClient(post_response=response))
        out = await WebTools(make_config(exa_api_key="k")).execute_tool("web_search", {"query": "python"})
        assert "provider: exa" in out

    async def test_search_perplexity_answer(self, mocker):
        response = _FakeResponse(
            json_data={
                "choices": [{"message": {"content": "Python is a language."}}],
                "search_results": [{"title": "Py", "url": "https://python.org"}],
            }
        )
        mocker.patch.object(perplexity.httpx, "AsyncClient", _FakeClient(post_response=response))
        out = await WebTools(make_config(perplexity_api_key="k")).execute_tool("web_search", {"query": "python"})
        assert "Python is a language." in out
        assert "provider: perplexity" in out

    async def test_search_provider_override(self, mocker):
        # Both Tavily and Brave keyed, but override forces Brave.
        response = _FakeResponse(
            json_data={"web": {"results": [{"title": "B", "url": "https://b.com", "description": "d"}]}}
        )
        mocker.patch.object(brave.httpx, "AsyncClient", _FakeClient(get_response=response))
        cfg = make_config(tavily_api_key="k", brave_api_key="k", web_search_provider="brave")
        out = await WebTools(cfg).execute_tool("web_search", {"query": "x"})
        assert "provider: brave" in out

    async def test_search_duckduckgo_retries_then_succeeds(self, mocker):
        from ddgs.exceptions import DDGSException

        calls = {"n": 0}

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise DDGSException("No results found.")
            return [{"title": "Py", "href": "https://python.org", "body": "b"}]

        mocker.patch.object(duckduckgo.DuckDuckGoProvider, "_search_sync", staticmethod(flaky))
        mocker.patch.object(duckduckgo.asyncio, "sleep", mocker.AsyncMock())
        out = await WebTools(make_config()).execute_tool("web_search", {"query": "python"})
        assert "https://python.org" in out
        assert calls["n"] == 2

    async def test_search_duckduckgo_fails_after_retries(self, mocker):
        from ddgs.exceptions import DDGSException

        def always_fail(*a, **k):
            raise DDGSException("No results found.")

        mocker.patch.object(duckduckgo.DuckDuckGoProvider, "_search_sync", staticmethod(always_fail))
        mocker.patch.object(duckduckgo.asyncio, "sleep", mocker.AsyncMock())
        out = await WebTools(make_config()).execute_tool("web_search", {"query": "python"})
        assert out.startswith("Error: web search failed")
        assert "after 3 attempts" in out

    async def test_search_searxng_preferred_over_ddg(self, mocker):
        # A SearXNG url wins over keyless DuckDuckGo (but still loses to Tavily).
        response = _FakeResponse(
            json_data={"results": [{"title": "Py", "url": "https://python.org", "content": "docs"}]}
        )
        mocker.patch.object(searxng.httpx, "AsyncClient", _FakeClient(get_response=response))
        out = await WebTools(make_config(web_searxng_url="https://searx.example")).execute_tool(
            "web_search", {"query": "python"}
        )
        assert "https://python.org" in out
        assert "provider: searxng" in out


class TestWebFetchExecution:
    async def test_fetch_html_to_markdown(self, mocker):
        html = "<html><head><title>Hi</title></head><body><h1>Heading</h1><p>Body text</p></body></html>"
        response = _FakeResponse(text=html, headers={"content-type": "text/html; charset=utf-8"}, url="https://e.com")
        mocker.patch.object(http_fetch.httpx, "AsyncClient", _FakeClient(get_response=response))
        out = await WebTools(make_config()).execute_tool("web_fetch", {"url": "https://e.com"})
        assert "Title: Hi" in out
        assert "Heading" in out
        assert "Body text" in out
        assert "Provider: http" in out

    async def test_fetch_truncation(self, mocker):
        html = "<p>" + ("x" * 500) + "</p>"
        response = _FakeResponse(text=html, headers={"content-type": "text/html"})
        mocker.patch.object(http_fetch.httpx, "AsyncClient", _FakeClient(get_response=response))
        out = await WebTools(make_config()).execute_tool("web_fetch", {"url": "https://e.com", "max_chars": 50})
        assert "Truncated: yes" in out

    async def test_fetch_rejects_non_http(self):
        out = await WebTools(make_config()).execute_tool("web_fetch", {"url": "ftp://x"})
        assert out.startswith("Error:")

    async def test_fetch_http_first_cost_aware(self, mocker):
        # Auto fetch tries free HTTP before keyed providers, even when Tavily is set.
        response = _FakeResponse(text="<p>ok</p>", headers={"content-type": "text/html"})
        mocker.patch.object(http_fetch.httpx, "AsyncClient", _FakeClient(get_response=response))
        out = await WebTools(make_config(tavily_api_key="k")).execute_tool("web_fetch", {"url": "https://e.com"})
        assert "ok" in out
        assert "Provider: http" in out

    async def test_fetch_escalates_to_firecrawl_on_403(self, mocker):
        # HTTP 403 (anti-bot) -> escalate to Firecrawl. http uses get, firecrawl uses
        # post, so one fake on the shared httpx module drives both.
        forbidden = _FakeResponse(status_code=403, headers={"content-type": "text/html"})
        fc = _FakeResponse(json_data={"data": {"markdown": "scraped body", "metadata": {"title": "T"}}})
        mocker.patch.object(http_fetch.httpx, "AsyncClient", _FakeClient(get_response=forbidden, post_response=fc))
        out = await WebTools(make_config(firecrawl_api_key="k")).execute_tool("web_fetch", {"url": "https://e.com"})
        assert "scraped body" in out
        assert "Provider: firecrawl" in out

    async def test_fetch_provider_override(self, mocker):
        # Forcing firecrawl skips HTTP entirely.
        fc = _FakeResponse(json_data={"data": {"markdown": "scraped", "metadata": {}}})
        mocker.patch.object(firecrawl.httpx, "AsyncClient", _FakeClient(post_response=fc))
        cfg = make_config(firecrawl_api_key="k", web_fetch_provider="firecrawl")
        out = await WebTools(cfg).execute_tool("web_fetch", {"url": "https://e.com"})
        assert "Provider: firecrawl" in out


class TestWebToolDispatch:
    async def test_unknown_tool(self):
        out = await WebTools(make_config()).execute_tool("nope", {})
        assert out.startswith("Error: unknown web tool")
