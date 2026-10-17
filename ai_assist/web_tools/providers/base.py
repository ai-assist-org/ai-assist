"""Provider seam for the web tools.

A provider knows how to talk to one backend. ``SearchProvider`` and
``FetchProvider`` are structural protocols — a class satisfies them by having the
right attributes/methods, no inheritance required (mirroring the duck-typed tool
contract used elsewhere in the codebase). The registry picks a provider by asking
each one whether it ``is_configured`` for the given config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class ProviderError(Exception):
    """A provider request failed (network, HTTP status, or malformed response)."""


@dataclass
class SearchResult:
    """Normalized web-search result from any provider."""

    query: str
    results: list[dict[str, Any]] = field(default_factory=list)  # {title, url, snippet, published?}
    answer: str | None = None
    provider: str = ""


@dataclass
class FetchResult:
    """Normalized web-fetch result from any provider."""

    url: str
    text: str
    final_url: str = ""
    status: int = 0
    content_type: str = ""
    title: str = ""
    truncated: bool = False
    provider: str = ""


@runtime_checkable
class SearchProvider(Protocol):
    """A backend that can search the web."""

    id: str

    @staticmethod
    def is_configured(config: Any) -> bool:
        """True when this provider has the credentials it needs."""
        ...

    async def search(self, config: Any, query: str, *, count: int, **opts: Any) -> SearchResult:
        """Run a search and return normalized results. Raise ProviderError on failure."""
        ...


@runtime_checkable
class FetchProvider(Protocol):
    """A backend that can fetch a single URL and extract readable content."""

    id: str

    @staticmethod
    def is_configured(config: Any) -> bool:
        """True when this provider has the credentials it needs."""
        ...

    async def fetch(self, config: Any, url: str, *, max_chars: int, **opts: Any) -> FetchResult:
        """Fetch a URL and return normalized content. Raise ProviderError on failure."""
        ...
