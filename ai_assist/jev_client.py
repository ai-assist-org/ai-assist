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

import asyncio
import logging
import threading
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# jev responds in 70-500ms; allow generous slack for network round-trips.
_TIMEOUT = httpx.Timeout(30.0)

# Relevance rubric for reranking, ordered low -> high (what jev Score rates each
# retrieved candidate on). Kept here so the production reranker and the eval that
# measures it share one definition.
_RELEVANCE_LEVELS = ["not relevant", "slightly relevant", "relevant", "highly relevant"]

# System One returns a structured answer for every question.  Keeping batches
# small prevents the answer itself from exhausting the endpoint's token budget.
_RERANK_BATCH_SIZE = 8
_RERANK_CONTENT_CHARS = 2_000


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
        detail = _http_error_detail(e.response, config.jev_api_key)
        message = f"jev HTTP error: {e.response.status_code}"
        if detail:
            message += f": {detail}"
        raise JevError(message) from e
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        logger.exception("jev request failed to connect")
        raise JevError(f"jev connection error: {e}") from e
    except ValueError as e:  # includes JSONDecodeError
        logger.exception("jev returned invalid JSON")
        raise JevError("jev returned invalid JSON") from e


def _http_error_detail(response: httpx.Response, api_key: str) -> str:
    try:
        body = response.json()
    except ValueError:
        return ""
    error = body.get("error") if isinstance(body, dict) else None
    message = error.get("message") if isinstance(error, dict) else error
    if not isinstance(message, str):
        return ""
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    return " ".join(message.split())[:500]


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


async def rerank_candidates(
    config: Any, query: str, candidates: list[dict[str, Any]], *, content_key: str = "content"
) -> list[dict[str, Any]]:
    """Reorder retrieval ``candidates`` by a jev Score relevance judgment.

    Sends bounded batches of candidates to jev, then returns a NEW list ordered
    by score (ties keep the incoming, e.g. cosine, order). On any jev error the
    input order is returned unchanged, so reranking can never break retrieval.
    """
    if len(candidates) < 2:
        return list(candidates)
    state = f"User search query: {query}"
    scores = [-1.0] * len(candidates)
    for start in range(0, len(candidates), _RERANK_BATCH_SIZE):
        batch = candidates[start : start + _RERANK_BATCH_SIZE]
        questions = {
            f"c{i}": score(
                f"Knowledge-base entry:\n{_rerank_content(candidate.get(content_key, ''))}\n\n"
                "How relevant is this entry to the user's search query?",
                _RELEVANCE_LEVELS,
            )
            for i, candidate in enumerate(batch, start=start)
        }
        try:
            response = await jev_decide(config, state, questions)
        except JevError:
            logger.exception("jev rerank failed; keeping original order")
            return list(candidates)
        for i in range(start, start + len(batch)):
            scores[i] = (score_result(response, f"c{i}") or {}).get("score", -1.0)
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
    return [candidates[i] for i in order]


def _rerank_content(content: Any) -> str:
    text = str(content)
    if len(text) <= _RERANK_CONTENT_CHARS:
        return text
    return text[:_RERANK_CONTENT_CHARS] + "\n[entry truncated for reranking]"


def rerank_candidates_sync(
    config: Any, query: str, candidates: list[dict[str, Any]], *, content_key: str = "content"
) -> list[dict[str, Any]]:
    """Blocking wrapper around :func:`rerank_candidates`.

    Runs the async rerank to completion in a dedicated thread, so it is safe to
    call from ordinary sync code AND from code already running inside an event loop
    (e.g. ``semantic_search`` invoked from an async tool handler).
    """
    box: dict[str, Any] = {}

    def _runner() -> None:
        try:
            box["value"] = asyncio.run(rerank_candidates(config, query, candidates, content_key=content_key))
        except BaseException as exc:  # noqa: BLE001 - re-raised on the calling thread
            box["error"] = exc

    thread = threading.Thread(target=_runner)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]
