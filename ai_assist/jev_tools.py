"""Ad-hoc jev access for the interactive agent.

Exposes jev (TypeSafe's non-generative System One decision model) as agent tools
for calibrated decisions over a state the agent provides. Three primitives, one
per tool:

* ``internal__jev_decide`` — Noul: yes/no + probability.
* ``internal__jev_choose`` — Choice: pick one named option + a distribution.
* ``internal__jev_score``  — Score: rate on an ordered rubric + a distribution.

All go through ``jev_client``. The tools are only offered when jev is configured
(``jev_configured``), so they never appear without a key. jev has no tools and no
world knowledge — it only judges the ``state`` text it is handed — so the agent
must include the relevant facts in ``state`` itself.
"""

from typing import Any

from .jev_client import (
    JevError,
    choice,
    choice_result,
    jev_decide,
    noul,
    noul_probability,
    score,
    score_result,
)


class JevTools:
    """jev decision tools: Noul (yes/no), Choice (pick one), Score (rate)."""

    def __init__(self, config: Any):
        self._config = config

    def get_tool_definitions(self) -> list[dict]:
        """Get tool definitions for the agent."""
        return [
            {
                "name": "internal__jev_decide",
                "description": (
                    "Ask jev — a fast, calibrated, non-generative decision model — a yes/no "
                    "question about a state you provide. Returns the probability the answer is "
                    "yes (0.0-1.0). jev has no tools and no outside knowledge: it only judges the "
                    "'state' text you give it, so include every relevant fact there. Good for a "
                    "quick, cheap, calibrated verdict (e.g. 'do these results meet the criterion?')."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "state": {"type": "string", "description": "All facts jev should judge, as plain text."},
                        "question": {"type": "string", "description": "The yes/no question to answer about the state."},
                    },
                    "required": ["state", "question"],
                },
                "_server": "internal",
                "_original_name": "jev_decide",
                "_readonly": True,
            },
            {
                "name": "internal__jev_choose",
                "description": (
                    "Ask jev to pick exactly one option from a finite set, given a state. Returns "
                    "the chosen option plus a calibrated probability for every option. jev only "
                    "judges the 'state' you provide (no tools, no outside knowledge). Use for "
                    "classification/routing among known categories (e.g. sentiment, severity bucket)."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "state": {"type": "string", "description": "All facts jev should judge, as plain text."},
                        "question": {"type": "string", "description": "What to decide among the options."},
                        "options": {
                            "type": "object",
                            "description": "Map of option name -> short description of what that option means.",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                    "required": ["state", "question", "options"],
                },
                "_server": "internal",
                "_original_name": "jev_choose",
                "_readonly": True,
            },
            {
                "name": "internal__jev_score",
                "description": (
                    "Ask jev to rate a state on an ordered rubric. Provide the rubric levels from "
                    "lowest to highest; jev returns a numeric score on that scale, the nearest "
                    "level label, and a calibrated distribution over levels. jev only judges the "
                    "'state' you provide. Use for graded assessments (e.g. urgency, risk, quality)."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "state": {"type": "string", "description": "All facts jev should judge, as plain text."},
                        "question": {"type": "string", "description": "What to rate about the state."},
                        "levels": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Rubric levels ordered lowest to highest (e.g. ['low','medium','high']).",
                        },
                    },
                    "required": ["state", "question", "levels"],
                },
                "_server": "internal",
                "_original_name": "jev_score",
                "_readonly": True,
            },
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Dispatch to the right jev primitive and return a readable result."""
        if tool_name == "jev_decide":
            return await self._decide(arguments)
        if tool_name == "jev_choose":
            return await self._choose(arguments)
        if tool_name == "jev_score":
            return await self._score(arguments)
        return f"Error: unknown jev tool '{tool_name}'"

    async def _decide(self, arguments: dict[str, Any]) -> str:
        state = arguments.get("state", "")
        question = arguments.get("question", "")
        if not state or not question:
            return "Error: both 'state' and 'question' are required."
        try:
            response = await jev_decide(self._config, state=state, questions={"decision": noul(question)})
        except JevError as e:
            return f"Error: jev request failed: {e}"
        probability = noul_probability(response, "decision")
        if probability is None:
            return "Error: jev returned no probability for the question."
        verdict = "yes" if probability >= 0.5 else "no"
        return f"jev answer: {verdict} (p(yes)={probability:.2f})"

    async def _choose(self, arguments: dict[str, Any]) -> str:
        state = arguments.get("state", "")
        question = arguments.get("question", "")
        options = arguments.get("options") or {}
        if isinstance(options, list):
            options = {str(o): str(o) for o in options}
        if not state or not question or not isinstance(options, dict) or len(options) < 2:
            return "Error: 'state', 'question', and at least two 'options' are required."
        try:
            response = await jev_decide(self._config, state=state, questions={"decision": choice(question, options)})
        except JevError as e:
            return f"Error: jev request failed: {e}"
        result = choice_result(response, "decision")
        if result is None:
            return "Error: jev returned no choice for the question."
        dist = ", ".join(f"{k}={float(v):.2f}" for k, v in result["probabilities"].items())
        return f"jev choice: {result['choice']} (confidence={_fmt(result['confidence'])}; {dist})"

    async def _score(self, arguments: dict[str, Any]) -> str:
        state = arguments.get("state", "")
        question = arguments.get("question", "")
        levels = arguments.get("levels") or []
        if not state or not question or not isinstance(levels, list) or len(levels) < 2:
            return "Error: 'state', 'question', and at least two 'levels' are required."
        try:
            response = await jev_decide(
                self._config, state=state, questions={"decision": score(question, [str(lv) for lv in levels])}
            )
        except JevError as e:
            return f"Error: jev request failed: {e}"
        result = score_result(response, "decision")
        if result is None:
            return "Error: jev returned no score for the question."
        nearest = f", nearest='{result['nearest']}'" if result["nearest"] is not None else ""
        return (
            f"jev score: {result['score']:.2f} on a 0-{len(levels) - 1} scale"
            f"{nearest} (confidence={_fmt(result['confidence'])})"
        )


def _fmt(value: Any) -> str:
    """Format an optional confidence float."""
    return f"{float(value):.2f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "n/a"
