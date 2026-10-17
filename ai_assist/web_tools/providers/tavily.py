"""Tavily provider: web search and URL extraction via the Tavily API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import FetchResult, ProviderError, SearchResult

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://api.tavily.com/search"
_EXTRACT_URL = "https://api.tavily.com/extract"


class TavilyProvider:
    """Search + fetch backed by Tavily. Needs ``config.tavily_api_key``."""

    id = "tavily"

    @staticmethod
    def is_configured(config: Any) -> bool:
        return bool(getattr(config, "tavily_api_key", None))

    async def search(self, config: Any, query: str, *, count: int, **opts: Any) -> SearchResult:
        payload: dict[str, Any] = {
            "api_key": config.tavily_api_key,
            "query": query,
            "max_results": count,
            "include_answer": True,
        }
        freshness = opts.get("freshness")
        if freshness:
            payload["time_range"] = freshness
        data = await self._post(config, _SEARCH_URL, payload)
        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
                "published": r.get("published_date"),
            }
            for r in data.get("results", [])
        ]
        return SearchResult(query=query, results=results, answer=data.get("answer"), provider=self.id)

    async def fetch(self, config: Any, url: str, *, max_chars: int, **opts: Any) -> FetchResult:
        data = await self._post(config, _EXTRACT_URL, {"api_key": config.tavily_api_key, "urls": [url]})
        extracted = data.get("results") or []
        if not extracted:
            raise ProviderError(f"Tavily returned no content for {url}")
        content = extracted[0].get("raw_content") or ""
        truncated = len(content) > max_chars
        return FetchResult(
            url=url,
            final_url=extracted[0].get("url", url),
            status=200,
            text=content[:max_chars],
            truncated=truncated,
            provider=self.id,
        )

    async def _post(self, config: Any, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        timeout = float(getattr(config, "web_timeout_seconds", 30))
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.warning("Tavily request to %s returned %s", endpoint, e.response.status_code)
            raise ProviderError(f"Tavily HTTP {e.response.status_code}") from e
        except httpx.HTTPError as e:
            logger.exception("Tavily request to %s failed", endpoint)
            raise ProviderError(f"Tavily request failed: {e}") from e
