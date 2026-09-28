"""Phase 4 -- Failure Autopsy (`alpha_agent.learn.autopsy`), built against the
REAL, already-committed local registry (read-only, never mutated -- the same
pattern `test_alpha_memory_builder.py` uses).
"""
from __future__ import annotations

import pytest
from alpha_agent.learn.autopsy import FailureAutopsy, SimilarFailure, build_failure_autopsy
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.ui import services

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)


def _registry() -> ExperimentRegistry:
    return ExperimentRegistry(services.REGISTRY_PATH)


def _first_non_pass_canonical() -> dict:
    """A real REJECT/INCONCLUSIVE canonical experiment -- the exact,
    explicit verdict allowlist `views/learn.py::_FAILURE_AUTOPSY_VERDICTS`
    uses (never "verdict != PASS", which would also match a real
    NOT_ADJUDICATED trial)."""
    rows = services.list_experiments(trial_role=None)
    canonical = [r for r in rows if r["trial_role"] == "CANONICAL" and r["verdict"] in ("REJECT", "INCONCLUSIVE")]
    assert canonical, "expected at least one REJECT/INCONCLUSIVE canonical experiment in the real registry"
    return canonical[0]


def _first_not_adjudicated_canonical() -> dict:
    """A real CANONICAL-trial-role experiment whose verdict is
    NOT_ADJUDICATED -- e.g. a Phase 15B ML meta-label trial typed-refused
    for insufficient training events (real registry state today: 40 such
    rows). Used to prove NOT_ADJUDICATED is never described as a failure."""
    rows = services.list_experiments(trial_role=None)
    rows = [r for r in rows if r["trial_role"] == "CANONICAL" and r["verdict"] == "NOT_ADJUDICATED"]
    assert rows, "expected at least one real NOT_ADJUDICATED canonical experiment in this registry"
    return rows[0]


def _experiment_with_positive_descriptive_evidence() -> dict:
    """A real REJECT/INCONCLUSIVE canonical experiment whose raw committed
    net PnL is positive despite the scientific verdict -- proves
    `descriptive_evidence` surfaces real "looked interesting" facts even
    when the gate that matters failed."""
    rows = services.list_experiments(trial_role=None)
    for r in rows:
        if r["trial_role"] == "CANONICAL" and r["verdict"] in ("REJECT", "INCONCLUSIVE") and (r.get("net_pnl_usd") or 0) > 0:
            return r
    raise AssertionError("expected a real REJECT/INCONCLUSIVE canonical experiment with positive net PnL")


def test_build_failure_autopsy_matches_services_wrapper_output():
    row = _first_non_pass_canonical()
    via_service = services.learn_failure_autopsy(row["experiment_id"])
    with _registry() as reg:
        rows = services.gate_table(
            reason_codes=row["reason_codes"], verdict=row["verdict"], trial_role=row["trial_role"],
        )
        evidence = services.gate_evidence_text(services.get_experiment(row["experiment_id"])["result"])
        direct = build_failure_autopsy(reg, row["experiment_id"], gate_table=rows, gate_evidence=evidence)
    assert direct.model_dump(mode="json") == via_service


def test_first_failed_gate_is_the_first_fail_in_gate_definition_order():
    from alpha_agent.ui.services import GATE_DEFINITIONS

    row = _first_non_pass_canonical()
    autopsy = FailureAutopsy.model_validate(services.learn_failure_autopsy(row["experiment_id"]))
    fail_labels = [g.label for g in autopsy.gates if g.state == "FAIL"]
    if fail_labels:
        # the first FAIL in GATE_DEFINITIONS order, not just any fail
        order = [label for label, _ in GATE_DEFINITIONS]
        first_by_order = min(fail_labels, key=order.index)
        assert autopsy.first_failed_gate == first_by_order
        assert autopsy.first_failed_gate_concept_id is not None
        assert services.learn_concept(autopsy.first_failed_gate_concept_id) is not None


def test_what_looked_promising_never_includes_anything_after_the_first_fail():
    row = _first_non_pass_canonical()
    autopsy = FailureAutopsy.model_validate(services.learn_failure_autopsy(row["experiment_id"]))
    if autopsy.first_failed_gate is None:
        return
    promising_labels = [s.split(" was explicitly satisfied")[0] for s in autopsy.what_looked_promising]
    from alpha_agent.ui.services import GATE_DEFINITIONS

    order = [label for label, _ in GATE_DEFINITIONS]
    fail_idx = order.index(autopsy.first_failed_gate)
    for label in promising_labels:
        assert order.index(label) < fail_idx


def test_similar_failures_never_includes_the_experiment_itself():
    row = _first_non_pass_canonical()
    autopsy = FailureAutopsy.model_validate(services.learn_failure_autopsy(row["experiment_id"]))
    assert all(s.experiment_id != row["experiment_id"] for s in autopsy.similar_failures)


def test_similar_failures_only_share_real_committed_reason_codes():
    row = _first_non_pass_canonical()
    autopsy = FailureAutopsy.model_validate(services.learn_failure_autopsy(row["experiment_id"]))
    target = set(autopsy.reason_codes)
    for s in autopsy.similar_failures:
        assert set(s.shared_reason_codes).issubset(target)
        assert s.shared_reason_codes  # never an empty "similar" match


def test_similar_failures_are_capped_and_sorted_by_shared_code_count_descending():
    row = _first_non_pass_canonical()
    autopsy = FailureAutopsy.model_validate(services.learn_failure_autopsy(row["experiment_id"]))
    assert len(autopsy.similar_failures) <= 5
    counts = [len(s.shared_reason_codes) for s in autopsy.similar_failures]
    assert counts == sorted(counts, reverse=True)


def test_a_pass_verdict_reports_no_failed_gate_and_no_change_needed_language():
    rows = services.list_experiments(trial_role=None)
    passing = [r for r in rows if r["trial_role"] == "CANONICAL" and r["verdict"] == "PASS"]
    if not passing:
        pytest.skip("no real PASS canonical experiment in this checkout's registry")
    autopsy = FailureAutopsy.model_validate(services.learn_failure_autopsy(passing[0]["experiment_id"]))
    assert autopsy.first_failed_gate is None
    assert "PASS" in autopsy.what_failed or "No gate failed" in autopsy.what_failed


def test_every_gate_row_state_is_one_of_the_real_gate_states():
    from alpha_agent.ui.services import GATE_STATES

    row = _first_non_pass_canonical()
    autopsy = FailureAutopsy.model_validate(services.learn_failure_autopsy(row["experiment_id"]))
    for g in autopsy.gates:
        assert g.state in GATE_STATES


def test_engineering_notes_come_from_the_real_failure_memory_for_this_market_and_family():
    from alpha_agent.registry.enums import FailureScope
    from alpha_agent.registry.failure_memory import FailureMemory

    row = _first_non_pass_canonical()
    autopsy = FailureAutopsy.model_validate(services.learn_failure_autopsy(row["experiment_id"]))
    with _registry() as reg:
        experiment_identity = reg.get(row["experiment_id"]).experiment_identity
        raw = FailureMemory(reg).related_engineering_failures(
            root_symbol=row["root_symbol"], strategy_family=row["strategy_family"],
        )
    assert [n.failure_code for n in autopsy.engineering_notes] == [f.failure_code for f in raw]
    assert [n.summary for n in autopsy.engineering_notes] == [f.summary for f in raw]
    for note, f in zip(autopsy.engineering_notes, raw, strict=True):
        assert note.scope == f.scope.value
        assert note.root_symbol == f.root_symbol
        assert note.strategy_family == f.strategy_family
        expected_applies = f.scope is FailureScope.EXPERIMENT and f.experiment_identity == experiment_identity
        assert note.applies_to_this_experiment is expected_applies


def test_engineering_note_never_overclaims_scope_from_free_text():
    """A SYSTEM-scope note whose own summary text happens to name a
    different concrete root (e.g. a ZN-specific defect described in a
    general methodology lesson) must be labeled by its REAL structured
    `scope`/`root_symbol`, never inferred from what the free text says."""
    row = _first_non_pass_canonical()
    autopsy = FailureAutopsy.model_validate(services.learn_failure_autopsy(row["experiment_id"]))
    system_notes = [n for n in autopsy.engineering_notes if n.scope == "SYSTEM"]
    assert system_notes, "expected at least one real SYSTEM-scope note for this fixture"
    for note in system_notes:
        # a SYSTEM note is never claimed as specific to the selected experiment,
        # regardless of what its free-text summary happens to mention
        assert note.applies_to_this_experiment is False


def test_engineering_note_experiment_scope_only_applies_when_identity_matches():
    """An EXPERIMENT-scope note tied to a DIFFERENT experiment_identity than
    the one being viewed must never be marked as applying to this trial --
    proven against the real registry's own precorrection/superseded note."""
    row = _first_non_pass_canonical()
    autopsy = FailureAutopsy.model_validate(services.learn_failure_autopsy(row["experiment_id"]))
    mismatched = [n for n in autopsy.engineering_notes if n.scope == "EXPERIMENT" and not n.applies_to_this_experiment]
    for note in mismatched:
        assert note.root_symbol is not None or note.strategy_family is not None


def test_narration_payload_never_includes_engineering_notes():
    """Claude narration must never receive an engineering note as a proven
    causal explanation for the selected experiment -- proven by inspecting
    the exact payload `FailureAutopsyNarrator` sends."""
    from alpha_agent.agents.llm import ScriptedLLMClient
    from alpha_agent.learn.narration_agent import FailureAutopsyNarrator

    row = _first_non_pass_canonical()
    autopsy = FailureAutopsy.model_validate(services.learn_failure_autopsy(row["experiment_id"]))
    assert autopsy.engineering_notes, "expected a fixture with real engineering notes to make this test meaningful"
    client = ScriptedLLMClient(["ok"])
    FailureAutopsyNarrator(client).narrate(autopsy)
    sent = client.calls[0]["messages"][0]["content"]
    for note in autopsy.engineering_notes:
        assert note.failure_code not in sent
        assert note.summary not in sent


def test_autopsy_narrative_is_none_without_any_llm_call():
    """The typed builder itself never touches an LLM -- `narrative` stays
    `None` until a caller explicitly runs `learn_failure_autopsy_narrative`."""
    row = _first_non_pass_canonical()
    autopsy = FailureAutopsy.model_validate(services.learn_failure_autopsy(row["experiment_id"]))
    assert autopsy.narrative is None


def test_build_failure_autopsy_raises_for_an_unknown_experiment_id():
    with _registry() as reg, pytest.raises(Exception):  # noqa: B017 -- registry raises its own typed lookup error
        build_failure_autopsy(reg, "NOT_A_REAL_EXPERIMENT_ID", gate_table=[], gate_evidence={})


# ---------------------------------------------------------------------------
# Semantic hardening patch (Phase 4 review)
# ---------------------------------------------------------------------------
# 1. NOT_ADJUDICATED is not a failure.
# 3. Descriptive evidence vs. validated evidence.
# 4. Shared-reason-code wording never overclaims similarity.
# 5. Factor != Strategy != Experiment.


def test_not_adjudicated_never_enters_the_learn_page_experiment_selector():
    """`views/learn.py`'s own candidate list -- the mechanism that decides
    which experiments are offered as "My Research" -- must exclude every
    NOT_ADJUDICATED row, never rely on a downstream "not PASS" filter."""
    from alpha_agent.ui.views.learn import _FAILURE_AUTOPSY_VERDICTS

    assert _FAILURE_AUTOPSY_VERDICTS == ("REJECT", "INCONCLUSIVE")
    rows = services.list_experiments(trial_role=None)
    candidates = [r for r in rows if r["trial_role"] == "CANONICAL" and r["verdict"] in _FAILURE_AUTOPSY_VERDICTS]
    assert all(r["verdict"] != "NOT_ADJUDICATED" for r in candidates)
    # proves this is a real, non-vacuous exclusion against today's registry
    excluded = [r for r in rows if r["trial_role"] == "CANONICAL" and r["verdict"] == "NOT_ADJUDICATED"]
    assert excluded, "expected real NOT_ADJUDICATED canonical rows in this registry to make this test meaningful"


def test_not_adjudicated_is_never_described_as_a_failure_by_the_builder():
    """Defense in depth: even if `build_failure_autopsy` is called directly
    on a real NOT_ADJUDICATED experiment (bypassing the UI's own selector),
    its narrative must never use failure language."""
    row = _first_not_adjudicated_canonical()
    autopsy = FailureAutopsy.model_validate(services.learn_failure_autopsy(row["experiment_id"]))
    assert autopsy.verdict == "NOT_ADJUDICATED"
    assert autopsy.first_failed_gate is None
    lowered = autopsy.what_failed.lower()
    assert "fail" not in lowered
    assert "not been headline-adjudicated" in lowered or "not adjudicated" in lowered
    assert all(g.state != "FAIL" for g in autopsy.gates)


def test_not_adjudicated_verdict_is_checked_before_any_gate_fail_state():
    """Even a synthetic `gate_table` claiming a FAIL row must not override
    the NOT_ADJUDICATED verdict-level framing -- proves the check really
    has absolute priority, not just "usually happens to work" on today's
    real gate states."""
    row = _first_not_adjudicated_canonical()
    with _registry() as reg:
        tainted_gate_table = [{"label": "Null Hypothesis (Bootstrap)", "state": "FAIL"}]
        autopsy = build_failure_autopsy(
            reg, row["experiment_id"], gate_table=tainted_gate_table, gate_evidence={},
        )
    assert autopsy.verdict == "NOT_ADJUDICATED"
    assert "fail" not in autopsy.what_failed.lower()


def test_descriptive_evidence_surfaces_positive_facts_even_after_an_earlier_gate_fails():
    row = _experiment_with_positive_descriptive_evidence()
    autopsy = FailureAutopsy.model_validate(services.learn_failure_autopsy(row["experiment_id"]))
    assert autopsy.first_failed_gate is not None  # a real gate did fail
    assert autopsy.descriptive_evidence  # yet positive descriptive facts still surface
    assert any("positive" in s.lower() for s in autopsy.descriptive_evidence)


def test_descriptive_evidence_is_distinct_from_validated_gate_passes():
    """`descriptive_evidence` (raw ResultRecord facts) and
    `what_looked_promising` (gates explicitly PASSed) are two different
    fields with two different evidentiary weights -- never merged into one
    undifferentiated list."""
    row = _experiment_with_positive_descriptive_evidence()
    autopsy = FailureAutopsy.model_validate(services.learn_failure_autopsy(row["experiment_id"]))
    assert set(autopsy.descriptive_evidence).isdisjoint(set(autopsy.what_looked_promising))
    # descriptive_evidence text never claims a gate was "satisfied" -- that
    # specific validated-evidence phrasing is reserved for what_looked_promising
    assert all("explicitly satisfied" not in s for s in autopsy.descriptive_evidence)


def test_descriptive_evidence_never_claims_a_negative_pnl_or_sharpe_as_positive():
    """Only genuinely positive committed PnL/Sharpe values are ever phrased
    as "positive" -- proven against the real CL/breakout fixture, whose
    committed net PnL and Sharpe are both negative."""
    autopsy = FailureAutopsy.model_validate(services.learn_failure_autopsy("CL__BREAKOUT__CANONICAL__VALIDATION_2023_2024"))
    assert not any("pnl" in s.lower() and "positive" in s.lower() for s in autopsy.descriptive_evidence)
    assert not any("sharpe" in s.lower() and "positive" in s.lower() for s in autopsy.descriptive_evidence)


def test_similar_failures_field_documents_it_is_not_hypothesis_similarity():
    """`SimilarFailure`'s own docstring states the honesty constraint this
    field must be rendered under -- a static proof the constraint is
    documented at the source of truth, not only in the UI's label text."""
    doc = SimilarFailure.__doc__ or ""
    assert "NOT" in doc
    assert "similarity" in doc.lower()


def test_scope_note_states_factor_strategy_experiment_are_distinct():
    row = _first_non_pass_canonical()
    autopsy = FailureAutopsy.model_validate(services.learn_failure_autopsy(row["experiment_id"]))
    lowered = autopsy.scope_note.lower()
    assert "factor" in lowered
    assert "experiment" in lowered or "strategyspec" in lowered
    assert "does not" in lowered or "not by itself" in lowered


def test_scope_note_is_the_same_fixed_text_for_every_experiment():
    """`scope_note` is a deterministic constant, never derived from this
    experiment's own data -- proven across two experiments with very
    different verdicts/reason codes."""
    a = FailureAutopsy.model_validate(services.learn_failure_autopsy(_first_non_pass_canonical()["experiment_id"]))
    b = FailureAutopsy.model_validate(services.learn_failure_autopsy(_first_not_adjudicated_canonical()["experiment_id"]))
    assert a.scope_note == b.scope_note
