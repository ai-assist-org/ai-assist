"""Tests for the /doctor configured-model check."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from ai_assist.main import _doctor_model_checks, run_doctor


def _config(**overrides) -> SimpleNamespace:
    base = {
        "model": "claude-sonnet-4-6",
        "effort": "medium",
        "model_tiers": {},
        "model_tier_efforts": {},
        "synthesis_model": None,
        "synthesis_effort": None,
        "compaction_model": None,
        "compaction_effort": None,
        # jev defaults: disabled, so jev_configured() is False unless a test opts in
        "jev_enabled": False,
        "jev_api_key": None,
        "jev_api_url": "http://jev.test/v1/decide",
        "jev_model": "~typesafe/jev-latest",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _agent(create=None, **config_overrides):
    """An agent stub exposing config, _model_for/_effort_for, and anthropic.create."""
    config = _config(**config_overrides)
    agent = MagicMock()
    agent.config = config
    agent._model_for.side_effect = lambda role: {
        "synthesis": config.synthesis_model or config.model,
        "compaction": config.compaction_model or config.model,
    }.get(role, config.model)
    agent._effort_for.side_effect = lambda role: {
        "synthesis": config.synthesis_effort or config.effort,
        "compaction": config.compaction_effort or config.effort,
    }.get(role, config.effort)
    if create is not None:
        agent.anthropic.messages.create.side_effect = create
    else:
        agent.anthropic.messages.create.return_value = MagicMock()
    return agent


def test_checks_default_model_only():
    checks = _doctor_model_checks(_agent())
    assert checks == [("default", "claude-sonnet-4-6", "medium")]


def test_checks_include_tiers_and_roles():
    checks = _doctor_model_checks(
        _agent(
            model_tiers={"low": "claude-haiku-5-5", "high": "claude-opus-4-8"},
            model_tier_efforts={"high": "max"},
            synthesis_model="claude-opus-4-8",
        )
    )
    combos = {(model, effort) for _, model, effort in checks}
    assert ("claude-haiku-5-5", "medium") in combos  # suffix-less tier inherits config.effort
    assert ("claude-opus-4-8", "max") in combos  # high tier uses its own effort
    assert ("claude-opus-4-8", "medium") in combos  # synthesis model, session effort


def test_checks_synthesis_and_compaction_models():
    """Both the synthesis and compaction models are checked, with their own effort."""
    checks = _doctor_model_checks(
        _agent(
            synthesis_model="claude-opus-4-8",
            compaction_model="claude-haiku-4-5",
            compaction_effort="no",  # compaction opts out of effort
        )
    )
    by_label = {label: (model, effort) for label, model, effort in checks}
    assert by_label["synthesis"] == ("claude-opus-4-8", "medium")  # inherits session effort
    assert by_label["compaction"] == ("claude-haiku-4-5", "no")  # own effort suffix wins


def test_duplicate_model_effort_combo_merged():
    # synthesis uses the same (model, effort) as default -> one check, labels joined.
    checks = _doctor_model_checks(_agent(synthesis_model="claude-sonnet-4-6"))
    assert len(checks) == 1
    labels, model, effort = checks[0]
    assert model == "claude-sonnet-4-6"
    assert "default" in labels and "synthesis" in labels


async def test_run_doctor_reports_failures():
    def create(*, model, **kwargs):
        if model == "broken-model":
            raise RuntimeError("This model does not support the effort parameter.")
        return MagicMock()

    agent = _agent(create=create, model_tiers={"low": "broken-model"})
    assert await run_doctor(agent) == 1  # one model failed


async def test_run_doctor_all_ok():
    assert await run_doctor(_agent()) == 0


async def test_run_doctor_hides_no_effort_shows_real_effort(capsys):
    agent = _agent(
        effort="medium",
        model_tiers={"low": "claude-haiku-4-5", "high": "claude-opus-4-8"},
        model_tier_efforts={"low": "no"},  # low opts out
    )
    await run_doctor(agent)
    out = capsys.readouterr().out

    assert "effort=no" not in out  # ':no' is not shown as an effort
    assert "effort=medium" in out  # a real effort still is


async def test_run_doctor_one_call_per_model_effort_combo():
    """No (model, effort) combo is queried more than once, even with overlap."""
    agent = _agent(
        # synthesis + low tier both resolve to (sonnet, medium) like default;
        # compaction + high tier both resolve to (opus, medium).
        synthesis_model="claude-sonnet-4-6",
        compaction_model="claude-opus-4-8",
        model_tiers={"low": "claude-sonnet-4-6", "high": "claude-opus-4-8"},
        model_tier_efforts={},  # tiers inherit config.effort == "medium"
    )
    await run_doctor(agent)

    combos = [
        (c.kwargs["model"], c.kwargs.get("output_config", {}).get("effort"))
        for c in agent.anthropic.messages.create.call_args_list
    ]
    # Unique combos only: (sonnet, medium) and (opus, medium).
    assert set(combos) == {("claude-sonnet-4-6", "medium"), ("claude-opus-4-8", "medium")}
    assert len(combos) == len(set(combos))


async def test_run_doctor_probes_jev_when_configured(monkeypatch):
    """When jev is configured, doctor probes it and reports pass/fail."""
    calls = []

    async def fake_decide(config, state, questions):
        calls.append((config.jev_model, state))
        return {"ok": True}

    monkeypatch.setattr("ai_assist.jev_client.jev_decide", fake_decide)

    agent = _agent(jev_enabled=True, jev_api_key="secret", jev_model="~typesafe/jev-latest")
    assert await run_doctor(agent) == 0
    assert calls == [("~typesafe/jev-latest", "ping")]


async def test_run_doctor_reports_jev_failure(monkeypatch):
    async def fake_decide(config, state, questions):
        raise RuntimeError("jev connection error")

    monkeypatch.setattr("ai_assist.jev_client.jev_decide", fake_decide)

    agent = _agent(jev_enabled=True, jev_api_key="secret")
    assert await run_doctor(agent) == 1  # jev failure counts


async def test_run_doctor_skips_jev_when_not_configured(monkeypatch):
    async def fake_decide(config, state, questions):  # pragma: no cover - must not run
        raise AssertionError("jev_decide should not be called when disabled")

    monkeypatch.setattr("ai_assist.jev_client.jev_decide", fake_decide)
    assert await run_doctor(_agent()) == 0  # jev_enabled is False by default
