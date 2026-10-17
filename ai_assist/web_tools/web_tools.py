"""Agent-facing web tools: ``internal__web_search`` and ``internal__web_fetch``.

Duck-typed tool class (like ``think_tool.py`` / ``jev_tools.py``): offers its
definitions only when enabled, dispatches in ``execute_tool``, and always returns a
string — failures come back as ``"Error: ..."`` rather than raised exceptions.
"""

from __future__ import annotations

from typing import Any

from .providers.base import FetchResult, ProviderError, SearchResult
from .registry import resolve_fetch_providers, resolve_search_provider


class WebTools:
    """Web search + fetch tools over pluggable providers."""

    def __init__(self, config: Any):
        self._config = config

    def get_tool_definitions(self) -> list[dict]:
        """Definitions for the tools that are currently available."""
        definitions: list[dict] = []
        if resolve_search_provider(self._config) is not None:
            definitions.append(
                {
                    "name": "internal__web_search",
                    "description": (
                        "Search the web and get back ranked results (title, URL, snippet) and, "
                        "when available, a short synthesized answer. Use for current events, facts "
                        "that may have changed, or to find pages to fetch with web_fetch."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "What to search for."},
                            "count": {
                                "type": "integer",
                                "description": "Max number of results (default 5).",
                            },
                            "freshness": {
                                "type": "string",
                                "description": "Optional recency filter: 'day', 'week', 'month', or 'year'.",
                            },
                        },
                        "required": ["query"],
                    },
                    "_server": "internal",
                    "_original_name": "web_search",
                    "_readonly": True,
                }
            )
        if getattr(self._config, "web_fetch_enabled", False):
            definitions.append(
                {
                    "name": "internal__web_fetch",
                    "description": (
                        "Fetch a single http(s) URL and return its readable content as markdown "
                        "(or plain text). Use after web_search, or when the user gives a URL. Large "
                        "pages are truncated — pass __save_to_file to keep the full content."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "The http(s) URL to fetch."},
                            "max_chars": {
                                "type": "integer",
                                "description": "Max characters of content to return.",
                            },
                            "extract_mode": {
                                "type": "string",
                                "enum": ["markdown", "text"],
                                "description": "Output format (default markdown).",
                            },
                        },
                        "required": ["url"],
                    },
                    "_server": "internal",
                    "_original_name": "web_fetch",
                    "_readonly": True,
                }
            )
        return definitions

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Dispatch to the right web tool and return a readable result."""
        if tool_name == "web_search":
            return await self._search(arguments)
        if tool_name == "web_fetch":
            return await self._fetch(arguments)
        return f"Error: unknown web tool '{tool_name}'"

    async def _search(self, arguments: dict[str, Any]) -> str:
        query = (arguments.get("query") or "").strip()
        if not query:
            return "Error: 'query' is required."
        provider = resolve_search_provider(self._config)
        if provider is None:
            return "Error: web search is unavailable (disabled, or the selected provider has no key)."
        count = int(arguments.get("count") or 5)
        opts = {"freshness": arguments["freshness"]} if arguments.get("freshness") else {}
        try:
            result = await provider.search(self._config, query, count=count, **opts)
        except ProviderError as e:
            return f"Error: web search failed: {e}"
        return _format_search(result)

    async def _fetch(self, arguments: dict[str, Any]) -> str:
        url = (arguments.get("url") or "").strip()
        if not url:
            return "Error: 'url' is required."
        if not url.startswith(("http://", "https://")):
            return "Error: 'url' must be an http(s) URL."
        providers = resolve_fetch_providers(self._config)
        if not providers:
            return "Error: web fetch has no available provider."
        raw_max = arguments.get("max_chars")
        max_chars = int(raw_max) if raw_max else int(getattr(self._config, "web_fetch_max_chars", 20000))
        opts = {"extract_mode": arguments.get("extract_mode", "markdown")}
        errors: list[str] = []
        for provider in providers:
            try:
                result = await provider.fetch(self._config, url, max_chars=max_chars, **opts)
            except ProviderError as e:
                errors.append(f"{provider.id}: {e}")
                continue
            return _format_fetch(result)
        return f"Error: web fetch failed ({'; '.join(errors)})."


def _format_search(result: SearchResult) -> str:
    """Render search results as a compact, readable block."""
    lines: list[str] = []
    if result.answer:
        lines.append(f"Answer: {result.answer}\n")
    if not result.results:
        lines.append("No results found.")
    for i, r in enumerate(result.results, 1):
        lines.append(f"{i}. {r.get('title') or '(untitled)'} — {r.get('url', '')}")
        snippet = r.get("snippet")
        if snippet:
            lines.append(f"   {snippet}")
    lines.append(f"\n(provider: {result.provider})")
    return "\n".join(lines)


def _format_fetch(result: FetchResult) -> str:
    """Render fetched content with a short header."""
    header = [f"URL: {result.final_url or result.url}", f"Status: {result.status}"]
    if result.title:
        header.append(f"Title: {result.title}")
    if result.truncated:
        header.append("Truncated: yes (pass __save_to_file or raise max_chars for the full page)")
    header.append(f"Provider: {result.provider}")
    return "\n".join(header) + "\n\n" + result.text
