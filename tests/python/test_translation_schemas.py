"""Phase 1 -- typed Observation/translation schema tests (prompt 1 sections
3/4). Covers the Observation Plane boundary invariants: a naive datetime is
refused, a root outside the certified Phase 1 universe is refused, and
`origin_vintage_fields` produces the same honest pre/post-holdout marker the
rest of the app already relies on (`alpha_agent.ui.views.agent
._origin_vintage_fields`).
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from alpha_agent.translation.schemas import (
    CERTIFIED_ROOTS,
    FactorCandidate,
    FactorProvenance,
    MeasurableVariable,
    MechanismCandidate,
    MechanismProvenance,
    Observation,
    ResearchabilityStatus,
    origin_vintage_fields,
)
from pydantic import ValidationError


def _obs(**overrides) -> Observation:
    base = dict(
        event_type="MARKET_NEWS",
        root_symbol="CL",
        observed_at=datetime(2024, 3, 1, tzinfo=UTC),
        source="EIA",
        summary="EIA reports a larger-than-expected crude draw",
        **origin_vintage_fields(datetime(2024, 3, 1, tzinfo=UTC)),
    )
    base.update(overrides)
    return Observation(**base)


def test_certified_roots_is_the_phase_1_five():
    assert CERTIFIED_ROOTS == ("ES", "NQ", "CL", "GC", "ZN")


def test_observation_rejects_naive_datetime():
    with pytest.raises(ValidationError):
        _obs(observed_at=datetime(2024, 3, 1))  # noqa: DTZ001 -- intentionally naive, this is what's under test


def test_observation_rejects_root_outside_certified_universe():
    with pytest.raises(ValidationError):
        _obs(root_symbol="NG")


@pytest.mark.parametrize("root", CERTIFIED_ROOTS)
def test_observation_accepts_every_certified_root(root):
    obs = _obs(root_symbol=root)
    assert obs.root_symbol == root


def test_observation_is_frozen():
    obs = _obs()
    with pytest.raises(ValidationError):
        obs.root_symbol = "ES"  # type: ignore[misc]


def test_origin_vintage_fields_pre_holdout_is_eligible():
    fields = origin_vintage_fields(datetime(2024, 6, 15, tzinfo=UTC))
    assert fields == {"origin_vintage": "2024-06-15", "holdout_eligible": True}


def test_origin_vintage_fields_post_holdout_is_not_eligible():
    fields = origin_vintage_fields(datetime(2026, 1, 2, tzinfo=UTC))
    assert fields["holdout_eligible"] is False
    assert fields["origin_vintage"] == "2026-01-02"


def test_mechanism_candidate_roundtrip():
    m = MechanismCandidate(
        mechanism="TREND", explanation="x", causal_chain=("a", "b"), evidence_basis="y",
        provenance=MechanismProvenance.DETERMINISTIC_LIBRARY,
    )
    assert m.mechanism.value == "TREND"
    assert m.provenance is MechanismProvenance.DETERMINISTIC_LIBRARY


def test_measurable_variable_requires_explicit_point_in_time_fields():
    with pytest.raises(ValidationError):
        MeasurableVariable(name="n", economic_meaning="m", required_source="s")  # type: ignore[call-arg]


def test_factor_candidate_default_empty_tuples():
    f = FactorCandidate(
        concept="c", mechanism="TREND", transform_or_proxy="t",
        researchability=ResearchabilityStatus.AVAILABLE, researchability_reason="r",
        provenance=FactorProvenance.REGISTERED_FEATURE,
    )
    assert f.proposed_feature_kinds == ()
    assert f.missing_requirements == ()
