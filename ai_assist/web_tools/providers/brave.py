"""Brave provider: web search via the Brave Search API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import ProviderError, SearchResult

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
# Brave freshness tokens.
_FRESHNESS = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}


class BraveProvider:
    """Web search backed by Brave. Needs ``config.brave_api_key``."""

    id = "brave"

    @staticmethod
    def is_configured(config: Any) -> bool:
        return bool(getattr(config, "brave_api_key", None))

    async def search(self, config: Any, query: str, *, count: int, **opts: Any) -> SearchResult:
        params: dict[str, Any] = {"q": query, "count": count}
        freshness = _FRESHNESS.get(opts.get("freshness") or "")
        if freshness:
            params["freshness"] = freshness
        headers = {"Accept": "application/json", "X-Subscription-Token": config.brave_api_key}
        timeout = float(getattr(config, "web_timeout_seconds", 30))
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(_SEARCH_URL, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            logger.warning("Brave request returned %s", e.response.status_code)
            raise ProviderError(f"Brave HTTP {e.response.status_code}") from e
        except httpx.HTTPError as e:
            logger.exception("Brave request failed")
            raise ProviderError(f"Brave request failed: {e}") from e

        results = [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("description", "")}
            for r in (data.get("web") or {}).get("results", [])
        ]
        return SearchResult(query=query, results=results, provider=self.id)
