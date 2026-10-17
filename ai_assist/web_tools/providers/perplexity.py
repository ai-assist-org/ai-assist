"""Perplexity provider: answer-with-citations search via the Sonar API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import ProviderError, SearchResult

logger = logging.getLogger(__name__)

_CHAT_URL = "https://api.perplexity.ai/chat/completions"
_MODEL = "sonar"
_FRESHNESS = {"day": "day", "week": "week", "month": "month", "year": "year"}


class PerplexityProvider:
    """Search that returns a synthesized answer + citations. Needs ``config.perplexity_api_key``."""

    id = "perplexity"

    @staticmethod
    def is_configured(config: Any) -> bool:
        return bool(getattr(config, "perplexity_api_key", None))

    async def search(self, config: Any, query: str, *, count: int, **opts: Any) -> SearchResult:
        payload: dict[str, Any] = {"model": _MODEL, "messages": [{"role": "user", "content": query}]}
        recency = _FRESHNESS.get(opts.get("freshness") or "")
        if recency:
            payload["search_recency_filter"] = recency
        headers = {"Authorization": f"Bearer {config.perplexity_api_key}", "Content-Type": "application/json"}
        timeout = float(getattr(config, "web_timeout_seconds", 30))
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(_CHAT_URL, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            logger.warning("Perplexity request returned %s", e.response.status_code)
            raise ProviderError(f"Perplexity HTTP {e.response.status_code}") from e
        except httpx.HTTPError as e:
            logger.exception("Perplexity request failed")
            raise ProviderError(f"Perplexity request failed: {e}") from e

        choices = data.get("choices") or []
        answer = choices[0].get("message", {}).get("content", "") if choices else ""
        # Newer API returns `search_results` (title+url); older returns `citations` (urls).
        if data.get("search_results"):
            results = [
                {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": ""}
                for r in data["search_results"][:count]
            ]
        else:
            results = [{"title": "", "url": url, "snippet": ""} for url in (data.get("citations") or [])[:count]]
        return SearchResult(query=query, results=results, answer=answer or None, provider=self.id)
