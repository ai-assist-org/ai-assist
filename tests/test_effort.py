"""Tests for the effort output-config helper and model:effort parsing."""

from ai_assist.config import (
    _env_role_effort,
    _env_role_model,
    _env_session_model_effort,
    _split_model_effort,
)
from ai_assist.effort import effort_output_config


def test_output_config_omitted_when_effort_unset():
    assert effort_output_config(None) == {}
    assert effort_output_config("") == {}


def test_output_config_present_when_effort_set():
    assert effort_output_config("medium") == {"output_config": {"effort": "medium"}}
    assert effort_output_config("max") == {"output_config": {"effort": "max"}}


def test_output_config_omitted_for_explicit_no():
    # "no" explicitly suppresses effort (for models/tiers that reject the param).
    assert effort_output_config("no") == {}


def test_split_model_effort_parses_levels_and_no():
    assert _split_model_effort("claude-haiku-4-5:no") == ("claude-haiku-4-5", "no")
    assert _split_model_effort("claude-sonnet-4-6:medium") == ("claude-sonnet-4-6", "medium")


def test_split_model_effort_without_suffix():
    assert _split_model_effort("claude-sonnet-4-6") == ("claude-sonnet-4-6", None)
    # An unknown suffix is not treated as an effort (kept as part of the model).
    assert _split_model_effort("some:thing") == ("some:thing", None)


def test_no_wins_over_fallback_in_awl_or_chain():
    """'no' is truthy, so an explicit tier/task 'no' overrides the session effort."""
    task_effort, tier_effort, saved_effort = None, "no", "medium"
    assert (task_effort or tier_effort or saved_effort) == "no"


def test_default_effort_none_when_not_specified(monkeypatch):
    """Unspecified default effort is None (no effort), even if a tier has a suffix."""
    monkeypatch.delenv("AI_ASSIST_MODEL", raising=False)
    # A high-tier suffix must NOT become the default effort.
    monkeypatch.setenv("AI_ASSIST_MODEL_HIGH", "claude-opus-4-8:medium")
    assert _env_session_model_effort() is None


def test_default_effort_from_explicit_model_suffix(monkeypatch):
    monkeypatch.setenv("AI_ASSIST_MODEL", "claude-opus-4-8:high")
    assert _env_session_model_effort() == "high"


def test_role_model_and_effort_suffix(monkeypatch):
    """Per-role model vars accept a :effort suffix, split like the others."""
    monkeypatch.setenv("AI_ASSIST_SYNTHESIS_MODEL", "claude-haiku-4-5:no")
    assert _env_role_model("AI_ASSIST_SYNTHESIS_MODEL") == "claude-haiku-4-5"
    assert _env_role_effort("AI_ASSIST_SYNTHESIS_MODEL") == "no"


def test_role_model_without_suffix_has_no_effort(monkeypatch):
    monkeypatch.setenv("AI_ASSIST_COMPACTION_MODEL", "claude-sonnet-4-6")
    assert _env_role_model("AI_ASSIST_COMPACTION_MODEL") == "claude-sonnet-4-6"
    assert _env_role_effort("AI_ASSIST_COMPACTION_MODEL") is None  # inherits session effort
