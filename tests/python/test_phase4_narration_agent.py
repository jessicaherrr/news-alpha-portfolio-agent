"""Phase 4 -- optional Claude narration over a Failure Autopsy
(`alpha_agent.learn.narration_agent`). Every test uses `ScriptedLLMClient` --
no network, ever (mirrors `test_phase_20_streamlit_ui.py`'s Phase 16/17 agent
tests)."""
from __future__ import annotations

import pytest
from alpha_agent.agents.llm import ScriptedLLMClient
from alpha_agent.learn.autopsy import FailureAutopsy, GateOutcome
from alpha_agent.learn.narration_agent import FailureAutopsyNarrator, NarrationRejected


def _autopsy(**overrides) -> FailureAutopsy:
    defaults = {
        "experiment_id": "CL__BREAKOUT__CANONICAL__VALIDATION_2023_2024",
        "experiment_identity": "experiment1:deadbeef",
        "root_symbol": "CL",
        "strategy_family": "breakout",
        "trial_role": "CANONICAL",
        "verdict": "REJECT",
        "reason_codes": ("null_hypothesis_not_rejected",),
        "gates": (GateOutcome(label="Null Hypothesis (Bootstrap)", state="FAIL", concept_id="bootstrap_null", evidence="p=0.77"),),
        "first_failed_gate": "Null Hypothesis (Bootstrap)",
        "first_failed_gate_concept_id": "bootstrap_null",
        "what_looked_promising": (),
        "what_failed": "Null Hypothesis (Bootstrap) failed (p=0.77).",
        "what_would_need_to_change": "A materially larger or more consistent edge.",
        "similar_failures": (),
    }
    defaults.update(overrides)
    return FailureAutopsy(**defaults)


def test_narrate_returns_the_models_text_on_a_clean_response():
    client = ScriptedLLMClient(["This result did not clear the null-hypothesis bar -- the edge looked like noise."])
    narrator = FailureAutopsyNarrator(client)
    text = narrator.narrate(_autopsy())
    assert "null-hypothesis" in text
    assert client.call_count == 1


def test_narrate_passes_only_typed_autopsy_facts_never_a_second_source():
    client = ScriptedLLMClient(["ok"])
    narrator = FailureAutopsyNarrator(client)
    narrator.narrate(_autopsy())
    sent = client.calls[0]["messages"][0]["content"]
    assert "REJECT" in sent
    assert "null_hypothesis_not_rejected" in sent


def test_narrate_returns_none_for_an_empty_response():
    client = ScriptedLLMClient([""])
    narrator = FailureAutopsyNarrator(client)
    assert narrator.narrate(_autopsy()) is None


def test_narrate_rejects_a_response_that_contradicts_the_committed_verdict():
    client = ScriptedLLMClient(["Good news -- this result is PASS and the strategy is ready to trade."])
    narrator = FailureAutopsyNarrator(client)
    with pytest.raises(NarrationRejected):
        narrator.narrate(_autopsy(verdict="REJECT"))


def test_narrate_allows_a_bare_mention_of_another_verdict_word_as_contrast():
    client = ScriptedLLMClient(["Unlike a PASS result, this one failed the very first statistical test."])
    narrator = FailureAutopsyNarrator(client)
    text = narrator.narrate(_autopsy(verdict="REJECT"))
    assert text is not None


def test_narrate_never_checks_verdict_contradiction_when_verdict_is_none():
    client = ScriptedLLMClient(["This experiment has not been adjudicated yet."])
    narrator = FailureAutopsyNarrator(client)
    text = narrator.narrate(_autopsy(verdict=None, first_failed_gate=None, first_failed_gate_concept_id=None, gates=()))
    assert text is not None


def test_narrate_holdout_guards_the_input_payload():
    """A 2025+ value anywhere in the autopsy must fail loudly before any
    model call is made (CLAUDE.md holdout rule) -- this proves the guard is
    actually wired, not just present elsewhere in the codebase."""
    client = ScriptedLLMClient(["should never be reached"])
    narrator = FailureAutopsyNarrator(client)
    tainted = _autopsy(what_failed="Observed on 2025-03-01: Null Hypothesis (Bootstrap) failed.")
    with pytest.raises(Exception):  # noqa: B017 -- the real HoldoutAccessError type
        narrator.narrate(tainted)
    assert client.call_count == 0


# ---------------------------------------------------------------------------
# `alpha_agent.ui.services.learn_failure_autopsy_narrative` -- the UI-facing
# wrapper, which must never let a raw transport/holdout/rejection exception
# reach the page (mirrors `decision_brief.generate_decision_brief`'s own
# defensive boundary).
# ---------------------------------------------------------------------------


class _BoomClient:
    def complete(self, **kwargs):
        raise RuntimeError("simulated network failure")


def test_services_wrapper_never_raises_on_a_transport_failure():
    pytest.importorskip("streamlit")
    from alpha_agent.ui import services

    if not services.REGISTRY_PATH.exists():
        pytest.skip("Phase 14 registry sqlite not present in this checkout")
    rows = services.list_experiments(trial_role=None)
    canonical = [r for r in rows if r["trial_role"] == "CANONICAL" and r["verdict"] not in (None, "PASS")]
    assert canonical
    result = services.learn_failure_autopsy_narrative(canonical[0]["experiment_id"], client=_BoomClient())
    assert result is None
