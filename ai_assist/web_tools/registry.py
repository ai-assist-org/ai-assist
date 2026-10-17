"""Provider selection for the web tools.

In "auto" mode (the default) a provider is picked by priority order: the first one
whose credentials are configured wins. The user can force a specific provider with
``config.web_search_provider`` / ``config.web_fetch_provider`` (a provider id, or
"auto"). Add a new backend by appending it to the ordered list below.
"""

from __future__ import annotations

from typing import Any

from .providers.base import FetchProvider, SearchProvider
from .providers.brave import BraveProvider
from .providers.duckduckgo import DuckDuckGoProvider
from .providers.exa import ExaProvider
from .providers.firecrawl import FirecrawlProvider
from .providers.http_fetch import HttpFetchProvider
from .providers.perplexity import PerplexityProvider
from .providers.searxng import SearxngProvider
from .providers.tavily import TavilyProvider

# Priority order for "auto": keyed/quality providers first, keyless fallbacks last.
SEARCH_PROVIDERS: list[type[SearchProvider]] = [
    TavilyProvider,
    BraveProvider,
    ExaProvider,
    PerplexityProvider,
    SearxngProvider,
    DuckDuckGoProvider,
]
# Fetch "auto" is cost-aware: try free HTTP first, escalate to keyed providers (for
# JS/anti-bot pages) only when HTTP fails.
FETCH_PROVIDERS: list[type[FetchProvider]] = [HttpFetchProvider, FirecrawlProvider, TavilyProvider]


def _select(providers: list, override: str) -> list:
    """Narrow an ordered provider list to a forced choice, or keep it for 'auto'."""
    if override and override != "auto":
        return [p for p in providers if p.id == override]
    return providers


def resolve_search_provider(config: Any) -> SearchProvider | None:
    """The first configured search provider (respecting any override), or None."""
    if not getattr(config, "web_search_enabled", False):
        return None
    override = getattr(config, "web_search_provider", "auto")
    for provider_cls in _select(SEARCH_PROVIDERS, override):
        if provider_cls.is_configured(config):
            return provider_cls()
    return None


def resolve_fetch_providers(config: Any) -> list[FetchProvider]:
    """Ordered list of configured fetch providers to try in turn (respecting any override)."""
    override = getattr(config, "web_fetch_provider", "auto")
    return [cls() for cls in _select(FETCH_PROVIDERS, override) if cls.is_configured(config)]


def web_search_configured(config: Any) -> bool:
    """True when web_search can be offered (enabled and a provider is configured)."""
    return resolve_search_provider(config) is not None
