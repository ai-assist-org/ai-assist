"""Built-in keyless HTTP fetch provider: GET a URL and extract readable content.

HTML is reduced to the main article with ``trafilatura`` (falling back to a full
``markdownify`` conversion when trafilatura finds no article); JSON is
pretty-printed; other text is returned as-is. Always available — the fetch
fallback when no API-based provider is configured.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
import trafilatura
from markdownify import markdownify

from .base import FetchResult, ProviderError

logger = logging.getLogger(__name__)

# A browser-like UA: many sites 403 a generic/bot User-Agent outright.
_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) " "Chrome/124.0.0.0 Safari/537.36"
_DEFAULT_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def _extract_html(body: str, extract_mode: str) -> str:
    """Extract the readable article from HTML; fall back to full-page markdown."""
    output_format = "txt" if extract_mode == "text" else "markdown"
    try:
        article = trafilatura.extract(body, output_format=output_format, include_links=(extract_mode != "text"))
    except Exception:  # trafilatura can raise on malformed markup; degrade gracefully
        logger.exception("trafilatura extraction failed; falling back to markdownify")
        article = None
    if article:
        return article.strip()
    strip = ["a", "img"] if extract_mode == "text" else None
    return markdownify(body, strip=strip).strip()


class HttpFetchProvider:
    """Fetch a URL over plain HTTP(S). No credentials required."""

    id = "http"

    @staticmethod
    def is_configured(config: Any) -> bool:
        return True

    async def fetch(self, config: Any, url: str, *, max_chars: int, **opts: Any) -> FetchResult:
        extract_mode = opts.get("extract_mode", "markdown")
        timeout = float(getattr(config, "web_timeout_seconds", 30))
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.get(url, headers=_DEFAULT_HEADERS)
                response.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Expected, handled outcome (e.g. 403/404) — log concisely, no traceback.
            logger.warning("HTTP fetch of %s returned %s", url, e.response.status_code)
            raise ProviderError(f"HTTP {e.response.status_code}") from e
        except httpx.HTTPError as e:
            logger.exception("HTTP fetch of %s failed", url)
            raise ProviderError(f"HTTP request failed: {e}") from e

        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        body = response.text
        title = ""
        if "html" in content_type:
            match = _TITLE_RE.search(body)
            title = match.group(1).strip() if match else ""
            text = _extract_html(body, extract_mode)
        elif "json" in content_type:
            try:
                text = json.dumps(response.json(), indent=2)
            except ValueError:
                text = body
        else:
            text = body

        truncated = len(text) > max_chars
        return FetchResult(
            url=url,
            final_url=str(response.url),
            status=response.status_code,
            content_type=content_type,
            title=title,
            text=text[:max_chars],
            truncated=truncated,
            provider=self.id,
        )
