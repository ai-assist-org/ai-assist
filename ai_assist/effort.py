"""Helper for the optional Anthropic ``output_config.effort`` request parameter."""

from __future__ import annotations

from typing import Any

from .config import EFFORT_NONE


def effort_output_config(effort: str | None) -> dict[str, Any]:
    """Return the ``output_config`` request kwargs for ``effort``, or ``{}`` to omit it.

    Omits effort both when it is unset (provider default) and when it is the
    explicit ``no`` setting (for models/tiers that reject the effort parameter).
    Any other value is sent as-is; use ``/doctor`` to check up front whether the
    configured models accept it.
    """
    if not effort or effort == EFFORT_NONE:
        return {}
    return {"output_config": {"effort": effort}}
