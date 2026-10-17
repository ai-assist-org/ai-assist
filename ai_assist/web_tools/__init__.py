"""Web tools for the agent: web_search and web_fetch over pluggable providers.

``web_fetch`` always works keyless via a built-in HTTP provider; ``web_search``
is only offered when a search provider (Tavily) is configured. See ``web_tools.py``
for the agent-facing tool class and ``registry.py`` for provider selection.
"""

from .registry import web_search_configured
from .web_tools import WebTools

__all__ = ["WebTools", "web_search_configured"]
