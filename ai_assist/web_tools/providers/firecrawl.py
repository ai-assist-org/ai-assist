"""Firecrawl provider: fetch JS-rendered / anti-bot pages via the Firecrawl API.

Handles pages that return 403 to a plain HTTP GET (Cloudflare-style bot
protection, client-rendered sites). Needs ``config.firecrawl_api_key``.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import FetchResult, ProviderError

logger = logging.getLogger(__name__)

_SCRAPE_URL = "https://api.firecrawl.dev/v1/scrape"


class FirecrawlProvider:
    """Keyed fetch for hard pages. Needs ``config.firecrawl_api_key``."""

    id = "firecrawl"

    @staticmethod
    def is_configured(config: Any) -> bool:
        return bool(getattr(config, "firecrawl_api_key", None))

    async def fetch(self, config: Any, url: str, *, max_chars: int, **opts: Any) -> FetchResult:
        fmt = "markdown"  # Firecrawl always returns clean markdown; text mode is handled downstream.
        payload = {"url": url, "formats": [fmt], "onlyMainContent": True}
        headers = {"Authorization": f"Bearer {config.firecrawl_api_key}", "Content-Type": "application/json"}
        timeout = float(getattr(config, "web_timeout_seconds", 30))
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(_SCRAPE_URL, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            logger.warning("Firecrawl request for %s returned %s", url, e.response.status_code)
            raise ProviderError(f"Firecrawl HTTP {e.response.status_code}") from e
        except httpx.HTTPError as e:
            logger.exception("Firecrawl request for %s failed", url)
            raise ProviderError(f"Firecrawl request failed: {e}") from e

        result = data.get("data") or {}
        content = result.get("markdown") or ""
        if not content:
            raise ProviderError(f"Firecrawl returned no content for {url}")
        metadata = result.get("metadata") or {}
        truncated = len(content) > max_chars
        return FetchResult(
            url=url,
            final_url=metadata.get("sourceURL", url),
            status=int(metadata.get("statusCode") or 200),
            title=metadata.get("title", ""),
            text=content[:max_chars],
            truncated=truncated,
            provider=self.id,
        )
