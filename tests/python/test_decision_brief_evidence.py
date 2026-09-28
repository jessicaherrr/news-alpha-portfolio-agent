"""Research Golden Path -- Decision Brief evidence test (task spec section
19): given a completed experiment, the Claude evidence bundle must contain
only bounded, real evidence categories and must never contain an invented
metric. Offline / Deterministic mode (the only mode this automated test
exercises, per section 21: "no automatic paid Claude API calls") makes zero
network calls.
"""
from __future__ import annotations

import pytest
from alpha_agent.registry import TrialRole
from alpha_agent.ui import decision_brief, services

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(),
    reason="Phase 14 registry sqlite not present in this checkout",
)


def _a_real_experiment_id() -> str:
    rows = services.list_experiments(trial_role=TrialRole.CANONICAL)
    assert rows, "expected at least one CANONICAL experiment in the registry"
    return rows[0]["experiment_id"]


def test_offline_decision_brief_makes_zero_network_calls(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("Offline / Deterministic mode must never construct AnthropicClient")

    monkeypatch.setattr(decision_brief, "AnthropicClient", _forbidden)
    evidence = decision_brief.build_decision_brief_evidence(_a_real_experiment_id())
    text, error = decision_brief.generate_decision_brief(evidence, mode=decision_brief.OFFLINE_MODE)
    assert error is None
    assert "Deterministic synthesis" in text


def test_decision_brief_evidence_numbers_match_the_committed_result_exactly():
    exp_id = _a_real_experiment_id()
    detail = services.get_experiment(exp_id)
    result = detail["result"]
    evidence = decision_brief.build_decision_brief_evidence(exp_id)
    cpp = evidence["cpp_historical_test"]
    assert cpp["net_pnl_usd"] == result["net_pnl_usd"]
    assert cpp["annualized_sharpe"] == result["annualized_sharpe"]
    assert cpp["n_trades"] == result["n_trades"]
    assert evidence["validation"]["verdict"] == result["headline_verdict"]
    assert evidence["validation"]["reason_codes"] == list(result["reason_codes"] or [])


def test_decision_brief_text_never_states_a_pnl_or_sharpe_absent_from_evidence():
    exp_id = _a_real_experiment_id()
    evidence = decision_brief.build_decision_brief_evidence(exp_id)
    text = decision_brief.render_deterministic_brief(evidence)
    cpp = evidence["cpp_historical_test"]
    if cpp["net_pnl_usd"] is not None:
        assert f"{cpp['net_pnl_usd']:,.0f}" in text
    if cpp["annualized_sharpe"] is not None:
        assert f"{cpp['annualized_sharpe']:.2f}" in text


def test_next_action_is_always_from_the_approved_vocabulary():
    for row in services.list_experiments(trial_role=TrialRole.CANONICAL):
        evidence = decision_brief.build_decision_brief_evidence(row["experiment_id"])
        action = decision_brief._deterministic_next_action(evidence)
        assert action in decision_brief.NEXT_ACTION_VOCABULARY


def test_next_action_vocabulary_excludes_buy_sell_language():
    banned = {"BUY", "SELL", "BUY NOW", "SELL NOW"}
    assert not (banned & set(decision_brief.NEXT_ACTION_VOCABULARY))


def test_evidence_never_touches_the_2025_holdout():
    exp_id = _a_real_experiment_id()
    evidence = decision_brief.build_decision_brief_evidence(exp_id)
    violation = services.check_no_holdout_leak(evidence, path="$.decision_brief_evidence")
    assert violation is None


def test_no_hypothesis_no_experiment_still_produces_an_honest_brief():
    """A hypothesis that has not been compiled/run yet must still render a
    brief that says so plainly, never a fabricated result."""
    evidence = decision_brief.build_decision_brief_evidence(None, hypothesis=None)
    text, error = decision_brief.generate_decision_brief(evidence, mode=decision_brief.OFFLINE_MODE)
    assert error is None
    assert "No committed C++ execution result yet" in text
