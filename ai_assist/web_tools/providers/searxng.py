"""SearXNG provider: keyless web search against a SearXNG instance's JSON API.

Configured by ``config.web_searxng_url`` (a self-hosted or public instance). No
API key — SearXNG aggregates results from many upstream engines.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import ProviderError, SearchResult

logger = logging.getLogger(__name__)

# SearXNG time_range tokens: day/week/month/year (no "hour" mapping needed here).
_FRESHNESS = {"day": "day", "week": "week", "month": "month", "year": "year"}


class SearxngProvider:
    """Keyless web search via a configured SearXNG instance."""

    id = "searxng"

    @staticmethod
    def is_configured(config: Any) -> bool:
        return bool(getattr(config, "web_searxng_url", None))

    async def search(self, config: Any, query: str, *, count: int, **opts: Any) -> SearchResult:
        base = str(config.web_searxng_url).rstrip("/")
        params: dict[str, Any] = {"q": query, "format": "json", "categories": "general"}
        time_range = _FRESHNESS.get(opts.get("freshness") or "")
        if time_range:
            params["time_range"] = time_range
        timeout = float(getattr(config, "web_timeout_seconds", 30))
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{base}/search", params=params)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            logger.warning("SearXNG request returned %s", e.response.status_code)
            raise ProviderError(f"SearXNG HTTP {e.response.status_code}") from e
        except httpx.HTTPError as e:
            logger.exception("SearXNG request failed")
            raise ProviderError(f"SearXNG request failed: {e}") from e

        results = [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
            for r in data.get("results", [])[:count]
        ]
        return SearchResult(query=query, results=results, provider=self.id)
