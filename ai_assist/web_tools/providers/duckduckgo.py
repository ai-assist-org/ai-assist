"""DuckDuckGo provider: keyless web search via the ``ddgs`` package.

Always available (no credentials), so it serves as the web_search fallback when no
API-based provider (Tavily) is configured. ``ddgs`` is synchronous, so the blocking
call runs in a thread to keep ``search`` async.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ddgs import DDGS
from ddgs.exceptions import DDGSException

from .base import ProviderError, SearchResult

logger = logging.getLogger(__name__)

# ddgs freshness tokens: day/week/month/year -> d/w/m/y.
_FRESHNESS = {"day": "d", "week": "w", "month": "m", "year": "y"}
# DuckDuckGo is flaky (rate-limits / backend hiccups return "No results found");
# a quick retry almost always succeeds.
_MAX_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 1.0


class DuckDuckGoProvider:
    """Keyless web search backed by DuckDuckGo (via ddgs)."""

    id = "duckduckgo"

    @staticmethod
    def is_configured(config: Any) -> bool:
        return True

    async def search(self, config: Any, query: str, *, count: int, **opts: Any) -> SearchResult:
        timelimit = _FRESHNESS.get(opts.get("freshness") or "")
        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                raw = await asyncio.to_thread(self._search_sync, query, count, timelimit)
            except DDGSException as e:
                # Expected, transient failure — log concisely and retry.
                last_error = e
                logger.warning("DuckDuckGo search attempt %d/%d failed: %s", attempt, _MAX_ATTEMPTS, e)
                if attempt < _MAX_ATTEMPTS:
                    await asyncio.sleep(_RETRY_DELAY_SECONDS * attempt)
                continue
            except Exception as e:  # unexpected — surface with a traceback
                logger.exception("DuckDuckGo search failed")
                raise ProviderError(f"DuckDuckGo request failed: {e}") from e
            results = [
                {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")} for r in raw
            ]
            return SearchResult(query=query, results=results, provider=self.id)
        raise ProviderError(f"DuckDuckGo request failed after {_MAX_ATTEMPTS} attempts: {last_error}")

    @staticmethod
    def _search_sync(query: str, count: int, timelimit: str | None) -> list[dict[str, Any]]:
        with DDGS() as ddgs:
            return ddgs.text(query, max_results=count, timelimit=timelimit)
