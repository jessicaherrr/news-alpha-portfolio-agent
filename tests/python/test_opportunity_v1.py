"""Opportunity V1 (Product Consolidation + Opportunity V1 campaign, Checkpoint
B/F; renamed/decoupled in the subsequent Product Acceptance Fix Pass).
`alpha_agent.opportunity.schemas` is pure and Streamlit-free -- every test
below runs with no live data, no network, and no Streamlit import.
`alpha_agent.ui.opportunity_context` is the separate real-evidence-assembly
boundary; its own scientific-isolation properties are covered at the bottom
of this file using only the plain, offline `alpha_agent.ui.services` reads
the rest of this codebase's test suite already relies on.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import pytest
from alpha_agent.opportunity.schemas import (
    FORBIDDEN_FIELD_NAMES,
    CatalystInfo,
    OpportunityInputs,
    OpportunitySnapshot,
    OpportunityState,
    TrendDirection,
    build_opportunity,
    rank_opportunities,
)

_NOW = datetime(2026, 9, 18, 12, 0, 0, tzinfo=UTC)


def _inputs(**overrides) -> OpportunityInputs:
    base = {
        "root_symbol": "CL",
        "display_name": "Crude Oil",
        "trend": TrendDirection.UP,
        "volatility": "Moderate",
        "volume_context": "Normal",
        "session_position": "Upper Half",
        "window_low": 68.5,
        "window_high": 71.2,
        "window_bars": 72,
        "window_timeframe": "1h",
        "curve_shape": "Backwardation",
        "peer_confirming_count": 1,
        "peer_total_count": 2,
        "news_count_24h": 2,
        "catalyst": None,
        "research_verdict": None,
        "research_promise_label": None,
        "best_strategy_family": None,
        "observed_at": _NOW,
    }
    base.update(overrides)
    return OpportunityInputs(**base)


# ---------------------------------------------------------------------------
# schema shape -- no probability/confidence/expected-return field, ever, and
# no field name that would re-merge Opportunity triage with Scientific
# Research Status.
# ---------------------------------------------------------------------------


def test_schema_never_carries_a_forbidden_field():
    fields = set(OpportunitySnapshot.model_fields)
    assert not fields & FORBIDDEN_FIELD_NAMES
    # exact, minimal schema (task spec section 7) -- catches an accidental
    # field addition just as much as a forbidden-name one.
    assert fields == {
        "product", "display_name", "setup", "why_now", "risks", "catalyst",
        "strengthen_if", "invalidate_if", "opportunity_state", "next_action",
        "observed_at", "evidence_refs",
    }


def test_schema_has_no_numeric_score_field():
    for name, field in OpportunitySnapshot.model_fields.items():
        assert field.annotation not in (int, float, "int", "float"), (
            f"OpportunitySnapshot.{name} is numeric -- Opportunity V1 must never carry a bare "
            "numeric score/confidence/expected-return field"
        )


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_build_opportunity_is_deterministic_same_input_same_output():
    inputs = _inputs()
    first = build_opportunity(inputs)
    second = build_opportunity(inputs)
    assert first == second


def test_rank_opportunities_is_deterministic_same_input_same_output():
    all_inputs = [_inputs(root_symbol="CL"), _inputs(root_symbol="NQ", trend=TrendDirection.SIDEWAYS)]
    assert rank_opportunities(all_inputs) == rank_opportunities(all_inputs)


# ---------------------------------------------------------------------------
# why_now / risks derived from real typed CURRENT OBSERVATIONAL evidence only
# ---------------------------------------------------------------------------


def test_why_now_reflects_trend_volume_curve_and_peer_confirmation():
    snap = build_opportunity(_inputs(
        trend=TrendDirection.UP, volume_context="Elevated", curve_shape="Backwardation",
        peer_confirming_count=2, peer_total_count=3,
    ))
    assert any("Up trend" in line for line in snap.why_now)
    assert any("Volume elevated" in line for line in snap.why_now)
    assert any("Backwardation" in line for line in snap.why_now)
    assert any("2/3" in line for line in snap.why_now)


def test_why_now_is_honest_when_no_evidence_supports_it():
    snap = build_opportunity(_inputs(
        trend=TrendDirection.SIDEWAYS, volume_context="Quiet", curve_shape=None,
        peer_confirming_count=0, peer_total_count=0,
    ))
    assert snap.why_now == ("No strengthening evidence observed in the current window",)


def test_risks_reflect_volatility_and_catalyst():
    catalyst = CatalystInfo(
        name="EIA Petroleum Status", source_name="EIA", scheduled_at=_NOW + timedelta(hours=20), importance="HIGH",
    )
    snap = build_opportunity(_inputs(volatility="High", catalyst=catalyst))
    assert any("volatility" in r.lower() for r in snap.risks)
    assert any("EIA Petroleum Status" in r for r in snap.risks)


def test_risks_reflect_non_confirming_peers():
    snap = build_opportunity(_inputs(peer_total_count=3, peer_confirming_count=0))
    assert any("not confirming" in r.lower() for r in snap.risks)


def test_risks_never_mention_a_registry_verdict():
    """No PASS/REJECT/INCONCLUSIVE/validation language leaks into risks --
    that is Research Evidence, surfaced separately as "Scientific Status",
    never a reason the CURRENT market itself is risky (Product Acceptance
    Fix Pass, item 3)."""
    for verdict in (None, "PASS", "REJECT", "INCONCLUSIVE", "NOT_ADJUDICATED"):
        snap = build_opportunity(_inputs(research_verdict=verdict))
        joined = " ".join(snap.risks).lower()
        assert "pass" not in joined
        assert "reject" not in joined
        assert "inconclusive" not in joined
        assert "validated" not in joined


# ---------------------------------------------------------------------------
# catalyst -- derived from EventStore evidence only, never fabricated
# ---------------------------------------------------------------------------


def test_catalyst_label_is_honest_when_none_exists():
    snap = build_opportunity(_inputs(catalyst=None))
    assert snap.catalyst == "No major mapped catalyst"


def test_catalyst_label_shows_name_time_and_importance():
    catalyst = CatalystInfo(
        name="EIA Petroleum Status", source_name="EIA", scheduled_at=_NOW + timedelta(hours=3, minutes=42),
        importance="HIGH",
    )
    snap = build_opportunity(_inputs(catalyst=catalyst))
    assert "EIA Petroleum Status" in snap.catalyst
    assert "3h 42m" in snap.catalyst
    assert "HIGH IMPACT" in snap.catalyst


# ---------------------------------------------------------------------------
# strengthen_if / invalidate_if -- deterministic, from observed levels only,
# and never labeled as a technical support/resistance/predicted-level model
# that does not exist (Product Acceptance Fix Pass, item 4).
# ---------------------------------------------------------------------------


def test_strengthen_and_invalidate_use_the_same_level_for_an_uptrend():
    snap = build_opportunity(_inputs(trend=TrendDirection.UP, window_low=100.0, window_high=110.0))
    assert any("100.00" in s and "holds above" in s for s in snap.strengthen_if)
    assert any("100.00" in s and "breaks below" in s for s in snap.invalidate_if)


def test_strengthen_and_invalidate_use_the_range_for_sideways():
    snap = build_opportunity(_inputs(trend=TrendDirection.SIDEWAYS, window_low=100.0, window_high=110.0))
    assert any("100.00" in s and "110.00" in s and "stays within" in s for s in snap.strengthen_if)
    assert any("100.00" in s and "110.00" in s and "breaks out" in s for s in snap.invalidate_if)


def test_strengthen_invalidate_are_deterministic():
    inputs = _inputs()
    a_s, a_i = build_opportunity(inputs).strengthen_if, build_opportunity(inputs).invalidate_if
    b_s, b_i = build_opportunity(inputs).strengthen_if, build_opportunity(inputs).invalidate_if
    assert a_s == b_s
    assert a_i == b_i


def test_strengthen_invalidate_never_claim_support_resistance_or_a_prediction():
    for trend in (TrendDirection.UP, TrendDirection.DOWN, TrendDirection.SIDEWAYS):
        snap = build_opportunity(_inputs(trend=trend, window_low=100.0, window_high=110.0))
        joined = " ".join(snap.strengthen_if) + " ".join(snap.invalidate_if)
        for banned in ("support", "resistance", "predicted"):
            assert banned not in joined.lower(), f"{banned!r} leaked into level text for trend={trend}"


def test_strengthen_invalidate_state_the_real_window_size():
    snap = build_opportunity(_inputs(window_bars=72, window_timeframe="1h", window_low=100.0, window_high=110.0))
    joined = " ".join(snap.strengthen_if) + " ".join(snap.invalidate_if)
    assert "72" in joined
    assert "1h" in joined


# ---------------------------------------------------------------------------
# opportunity_state -- WATCH / INVESTIGATE / WAIT, opportunity triage over
# CURRENT observational evidence only -- never a scientific research status.
# ---------------------------------------------------------------------------


def test_investigate_requires_directional_evidence_plus_support():
    snap = build_opportunity(_inputs(
        trend=TrendDirection.UP, volume_context="Elevated", research_verdict="REJECT", catalyst=None,
    ))
    assert snap.opportunity_state is OpportunityState.INVESTIGATE
    assert snap.next_action == "Investigate trend-continuation mechanism"


def test_watch_when_evidence_is_directional_but_unsupported():
    snap = build_opportunity(_inputs(
        trend=TrendDirection.UP, volume_context="Normal", peer_confirming_count=0, peer_total_count=0,
        catalyst=None,
    ))
    assert snap.opportunity_state is OpportunityState.WATCH


def test_investigate_even_with_an_existing_pass():
    """Product Acceptance Fix Pass, item 3: an existing authoritative PASS
    must NOT change whether the current market looks interesting -- that
    distinction belongs only to the separate Scientific Status surface."""
    snap = build_opportunity(_inputs(
        trend=TrendDirection.UP, volume_context="Elevated", research_verdict="PASS", catalyst=None,
    ))
    assert snap.opportunity_state is OpportunityState.INVESTIGATE


def test_wait_when_a_high_impact_catalyst_is_imminent():
    catalyst = CatalystInfo(
        name="FOMC Rate Decision", source_name="Federal Reserve", scheduled_at=_NOW + timedelta(hours=2),
        importance="HIGH",
    )
    snap = build_opportunity(_inputs(
        trend=TrendDirection.UP, volume_context="Elevated", catalyst=catalyst,
    ))
    assert snap.opportunity_state is OpportunityState.WAIT
    assert "catalyst" in snap.next_action.lower()


def test_wait_does_not_trigger_for_a_distant_high_impact_catalyst():
    catalyst = CatalystInfo(
        name="FOMC Rate Decision", source_name="Federal Reserve", scheduled_at=_NOW + timedelta(days=5),
        importance="HIGH",
    )
    snap = build_opportunity(_inputs(
        trend=TrendDirection.UP, volume_context="Elevated", catalyst=catalyst,
    ))
    assert snap.opportunity_state is not OpportunityState.WAIT


def test_wait_does_not_trigger_for_a_low_importance_imminent_catalyst():
    catalyst = CatalystInfo(
        name="Minor Release", source_name="USDA", scheduled_at=_NOW + timedelta(hours=1), importance="LOW",
    )
    snap = build_opportunity(_inputs(
        trend=TrendDirection.UP, volume_context="Elevated", catalyst=catalyst,
    ))
    assert snap.opportunity_state is not OpportunityState.WAIT


# ---------------------------------------------------------------------------
# CURRENT OPPORTUNITY independence (Product Acceptance Fix Pass, item 3) --
# opportunity_state / why_now / risks / internal ranking must be identical
# regardless of research_verdict / research_promise_label / best_strategy_family.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("verdict", [None, "PASS", "REJECT", "INCONCLUSIVE", "NOT_ADJUDICATED"])
def test_opportunity_state_is_independent_of_research_verdict(verdict):
    baseline = build_opportunity(_inputs(research_verdict=None))
    varied = build_opportunity(_inputs(research_verdict=verdict))
    assert baseline.opportunity_state == varied.opportunity_state


@pytest.mark.parametrize("verdict", [None, "PASS", "REJECT", "INCONCLUSIVE", "NOT_ADJUDICATED"])
def test_why_now_is_independent_of_research_verdict(verdict):
    baseline = build_opportunity(_inputs(research_verdict=None))
    varied = build_opportunity(_inputs(research_verdict=verdict))
    assert baseline.why_now == varied.why_now


@pytest.mark.parametrize("verdict", [None, "PASS", "REJECT", "INCONCLUSIVE", "NOT_ADJUDICATED"])
def test_risks_are_independent_of_research_verdict(verdict):
    baseline = build_opportunity(_inputs(research_verdict=None))
    varied = build_opportunity(_inputs(research_verdict=verdict))
    assert baseline.risks == varied.risks


def test_ranking_is_independent_of_research_verdict():
    a = _inputs(root_symbol="CL", research_verdict=None)
    b = _inputs(root_symbol="CL", research_verdict="PASS")
    ranked_a = rank_opportunities([a])[0]
    ranked_b = rank_opportunities([b])[0]
    assert ranked_a.opportunity_state == ranked_b.opportunity_state
    assert ranked_a.why_now == ranked_b.why_now
    assert ranked_a.risks == ranked_b.risks


def test_research_evidence_is_still_traceable_via_evidence_refs_only():
    """Research evidence is not deleted -- it stays available for the
    separate Scientific Status / Research Context / Next Action surfaces,
    just never as an input to the opportunity decision itself."""
    snap = build_opportunity(_inputs(research_verdict="REJECT"))
    assert any("research_verdict=REJECT" in ref for ref in snap.evidence_refs)


# ---------------------------------------------------------------------------
# ranking -- internal priority only, never exposed
# ---------------------------------------------------------------------------


def test_rank_opportunities_orders_investigate_above_watch_above_wait():
    catalyst = CatalystInfo(
        name="FOMC Rate Decision", source_name="Federal Reserve", scheduled_at=_NOW + timedelta(hours=1),
        importance="HIGH",
    )
    waiting = _inputs(root_symbol="GC", trend=TrendDirection.UP, volume_context="Elevated", catalyst=catalyst)
    watching = _inputs(root_symbol="ZN", trend=TrendDirection.SIDEWAYS, volume_context="Normal", catalyst=None)
    investigating = _inputs(
        root_symbol="CL", trend=TrendDirection.UP, volume_context="Elevated", research_verdict="REJECT",
        catalyst=None,
    )
    ranked = rank_opportunities([waiting, watching, investigating])
    states = [s.opportunity_state for s in ranked]
    assert states.index(OpportunityState.INVESTIGATE) < states.index(OpportunityState.WATCH) < states.index(
        OpportunityState.WAIT
    )


def test_rank_opportunities_never_pads_with_a_fabricated_entry():
    assert rank_opportunities([]) == []


# ---------------------------------------------------------------------------
# no evidence_refs fabrication -- always traceable to a typed input field
# ---------------------------------------------------------------------------


def test_evidence_refs_trace_back_to_typed_inputs():
    inputs = _inputs(trend=TrendDirection.DOWN, volatility="High", volume_context="Quiet")
    snap = build_opportunity(inputs)
    joined = " ".join(snap.evidence_refs)
    assert "trend_state=Down" in joined
    assert "volatility_state=High" in joined
    assert "volume_context_state=Quiet" in joined


# ---------------------------------------------------------------------------
# scientific isolation -- opportunity_context.py never writes the registry,
# never touches 2025, and reading it never mutates registry content.
# ---------------------------------------------------------------------------


def test_opportunity_context_module_never_calls_a_registry_write_method():
    from alpha_agent.ui import opportunity_context

    src_path = opportunity_context.__file__
    with open(src_path, encoding="utf-8") as fh:
        text = fh.read()
    forbidden = [
        r"\.insert_experiment\(", r"\.record_failure\(", r"\.record_lineage\(",
        r"\.apply_bundle\(", r"\.record_attempt", r"INSERT OR REPLACE",
    ]
    for pattern in forbidden:
        assert not re.search(pattern, text), f"opportunity_context.py must never call {pattern!r}"


def test_top_opportunities_never_changes_registry_content():
    pytest.importorskip("streamlit")
    from alpha_agent.ui import opportunity_context, services

    if not services.REGISTRY_PATH.exists():
        pytest.skip("Phase 14 registry sqlite not present in this checkout")
    before = services.registry_summary()["content_digest"]
    opportunity_context.top_opportunities(limit=3)
    after = services.registry_summary()["content_digest"]
    assert before == after
