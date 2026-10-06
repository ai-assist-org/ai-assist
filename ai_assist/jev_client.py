"""Client for TypeSafe's jev "System One" decision model.

jev is a non-generative decision model: given a ``state`` plus typed
``questions``, it returns calibrated, typed answers (no generated text). This
module is the single place that talks to the jev System One API, shared by the
AWL runtime (transparent goal-success checks) and any future jev caller.

jev is reachable from TypeSafe directly or via OpenRouter; both expose an
identical System One endpoint, so only the base URL, model id, and key differ
(all config). jev is optional: callers must gate on ``jev_configured``.

Request/response shape (verified against the live System One endpoint):
  request:  {"model", "state", "questions": {<name>: {"type": "noul", "instructions": <q>}}}
  response: {"answers": {<name>: {"type": "noul", "noul": <p_yes>}}, "usage": {...}}
``questions`` is an object keyed by question name, not a list. All shape
knowledge is confined to this module.
"""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# jev responds in 70-500ms; allow generous slack for network round-trips.
_TIMEOUT = httpx.Timeout(30.0)


class JevError(Exception):
    """A jev request failed (network, HTTP status, or malformed response)."""


def jev_configured(config: Any) -> bool:
    """True when jev is enabled and has an API key."""
    return bool(getattr(config, "jev_enabled", False) and getattr(config, "jev_api_key", None))


def noul(question: str) -> dict[str, Any]:
    """Shape a single Noul (yes/no) question spec for the ``questions`` record."""
    return {"type": "noul", "instructions": question}


def choice(question: str, options: dict[str, str]) -> dict[str, Any]:
    """Shape a Choice question: pick one named option from ``options``.

    ``options`` maps each option name to a short description jev judges against.
    """
    return {"type": "choice", "instructions": question, "criteria": options}


def score(question: str, levels: list[str]) -> dict[str, Any]:
    """Shape a Score question: rate on an ordered rubric.

    ``levels`` is the rubric ordered low → high; jev returns a score on that scale.
    """
    return {"type": "score", "instructions": question, "criteria": levels}


async def jev_decide(config: Any, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Call jev's System One endpoint and return the parsed JSON response.

    Raises ``JevError`` on connection failure, non-2xx status, or invalid JSON.
    """
    body = {"model": config.jev_model, "state": state, "questions": questions}
    headers = {
        "Authorization": f"Bearer {config.jev_api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(config.jev_api_url, json=body, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        logger.exception("jev request failed with HTTP status")
        raise JevError(f"jev HTTP error: {e.response.status_code}") from e
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        logger.exception("jev request failed to connect")
        raise JevError(f"jev connection error: {e}") from e
    except ValueError as e:  # includes JSONDecodeError
        logger.exception("jev returned invalid JSON")
        raise JevError("jev returned invalid JSON") from e


def noul_probability(response: dict[str, Any], name: str) -> float | None:
    """Extract the yes-probability for Noul question ``name`` from a response.

    The answer for a Noul carries the probability under the ``noul`` key
    (``{"type": "noul", "noul": 0.99}``). Returns None if the answer or its
    probability cannot be found.
    """
    answers = response.get("answers")
    if not isinstance(answers, dict):
        return None
    answer = answers.get(name)
    if not isinstance(answer, dict):
        return None
    val = answer.get("noul")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return None


def choice_result(response: dict[str, Any], name: str) -> dict[str, Any] | None:
    """Extract a Choice answer: the winning option, its distribution, confidence.

    Returns ``{"choice", "probabilities", "confidence"}`` or None if absent.
    """
    answer = _answer(response, name)
    if answer is None or not isinstance(answer.get("choice"), str):
        return None
    return {
        "choice": answer["choice"],
        "probabilities": answer.get("probabilities", {}),
        "confidence": answer.get("confidence"),
    }


def score_result(response: dict[str, Any], name: str) -> dict[str, Any] | None:
    """Extract a Score answer: the numeric score, rubric legend, distribution.

    Returns ``{"score", "legend", "probabilities", "confidence", "nearest"}`` or
    None if absent. ``nearest`` is the rubric label closest to the score.
    """
    answer = _answer(response, name)
    if answer is None or not isinstance(answer.get("score"), (int, float)) or isinstance(answer.get("score"), bool):
        return None
    legend = answer.get("legend", {})
    nearest = legend.get(str(round(float(answer["score"])))) if isinstance(legend, dict) else None
    return {
        "score": float(answer["score"]),
        "legend": legend,
        "probabilities": answer.get("probabilities", {}),
        "confidence": answer.get("confidence"),
        "nearest": nearest,
    }


def _answer(response: dict[str, Any], name: str) -> dict[str, Any] | None:
    """Return the answer record for question ``name``, or None if missing."""
    answers = response.get("answers")
    if not isinstance(answers, dict):
        return None
    answer = answers.get(name)
    return answer if isinstance(answer, dict) else None
