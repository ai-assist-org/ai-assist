"""Exa provider: neural/semantic web search via the Exa API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import ProviderError, SearchResult

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://api.exa.ai/search"


class ExaProvider:
    """Web search backed by Exa. Needs ``config.exa_api_key``."""

    id = "exa"

    @staticmethod
    def is_configured(config: Any) -> bool:
        return bool(getattr(config, "exa_api_key", None))

    async def search(self, config: Any, query: str, *, count: int, **opts: Any) -> SearchResult:
        payload: dict[str, Any] = {
            "query": query,
            "numResults": count,
            "contents": {"text": {"maxCharacters": 500}},
        }
        headers = {"x-api-key": config.exa_api_key, "Content-Type": "application/json"}
        timeout = float(getattr(config, "web_timeout_seconds", 30))
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(_SEARCH_URL, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            logger.warning("Exa request returned %s", e.response.status_code)
            raise ProviderError(f"Exa HTTP {e.response.status_code}") from e
        except httpx.HTTPError as e:
            logger.exception("Exa request failed")
            raise ProviderError(f"Exa request failed: {e}") from e

        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": (r.get("text") or "").strip(),
                "published": r.get("publishedDate"),
            }
            for r in data.get("results", [])
        ]
        return SearchResult(query=query, results=results, provider=self.id)
