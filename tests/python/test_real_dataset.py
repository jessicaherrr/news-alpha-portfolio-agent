"""Phase 13.5A -- real-dataset acquisition design (no network, no download)."""
from __future__ import annotations

import pytest
from alpha_agent.data.real_dataset import (
    DATASET,
    ComponentKind,
    RealDatasetManifest,
    SplitRole,
    load_plan,
)

CONFIG = "configs/real_dataset.yaml"


def test_plans_load_and_are_chronological_with_one_locked_holdout():
    for pid in ("A", "B"):
        plan = load_plan(CONFIG, pid)
        assert plan.dataset == DATASET == "GLBX.MDP3"
        assert plan.research_schema == "ohlcv-1m"
        assert plan.roots == ("ES", "NQ", "CL", "GC", "ZN")
        roles = [w.role for w in plan.windows]
        assert roles.count(SplitRole.LOCKED_HOLDOUT) == 1
        assert plan.windows[-1].role == SplitRole.LOCKED_HOLDOUT
        assert plan.holdout_span() == ("2025-01-01", "2026-01-01")
        # non-overlapping, chronological
        for a, b in zip(plan.windows, plan.windows[1:]):
            assert b.start >= a.end


def test_plan_a_and_b_spans():
    a = load_plan(CONFIG, "A")
    b = load_plan(CONFIG, "B")
    assert a.research_validation_span() == ("2021-01-01", "2025-01-01")
    assert b.research_validation_span() == ("2018-01-01", "2025-01-01")
    assert a.plan_fingerprint() != b.plan_fingerprint()


def test_components_exclude_holdout_by_default_and_cover_every_root():
    plan = load_plan(CONFIG, "B")
    comps = plan.components(include_holdout=False)
    assert all(c.window.role != SplitRole.LOCKED_HOLDOUT for c in comps)
    with_ho = plan.components(include_holdout=True)
    assert any(c.window.role == SplitRole.LOCKED_HOLDOUT for c in with_ho)
    # each non-holdout window: one continuous + one definitions + one roll-overlap per root
    kinds = {(c.kind, c.root_symbol, c.window.role.value) for c in comps}
    for w in plan.windows:
        if w.role == SplitRole.LOCKED_HOLDOUT:
            continue
        for root in plan.roots:
            assert (ComponentKind.CONTINUOUS_FRONT, root, w.role.value) in kinds
            assert (ComponentKind.DEFINITIONS, root, w.role.value) in kinds
            assert (ComponentKind.ROLL_OVERLAP_RAW, root, w.role.value) in kinds


def test_continuous_front_component_symbology_is_the_project_convention():
    plan = load_plan(CONFIG, "A")
    cont = next(c for c in plan.components() if c.kind == ComponentKind.CONTINUOUS_FRONT
               and c.root_symbol == "NQ")
    assert cont.symbols == ("NQ.v.0",)
    assert cont.schema_ == "ohlcv-1m"
    assert cont.stype_in == "continuous"
    assert cont.stype_out == "instrument_id"          # continuous->raw_symbol is unsupported
    defs = next(c for c in plan.components() if c.kind == ComponentKind.DEFINITIONS
                and c.root_symbol == "NQ")
    assert defs.symbols == ("NQ.FUT",) and defs.stype_in == "parent"
    roll = next(c for c in plan.components() if c.kind == ComponentKind.ROLL_OVERLAP_RAW
                and c.root_symbol == "NQ")
    assert roll.stype_in == "raw_symbol" and roll.resolved_from_transitions
    assert roll.symbols == ()                         # resolved from observed transitions


def test_query_identity_is_deterministic_and_request_specific():
    plan = load_plan(CONFIG, "A")
    comps = plan.components()
    a = comps[0].query_identity()
    assert a == comps[0].query_identity()
    assert a.startswith("dbnq1:")
    assert len({c.query_identity() for c in comps}) == len(comps)  # every request distinct


def test_manifest_semantic_identity_survives_a_cosmetic_path_move():
    base = {
        "databento_schema": "ohlcv-1m",
        "component_kind": ComponentKind.CONTINUOUS_FRONT,
        "root_symbol": "NQ",
        "split_role": SplitRole.RESEARCH,
        "symbol_request": ("NQ.v.0",),
        "stype_in": "continuous",
        "stype_out": "instrument_id",
        "start": "2018-01-01",
        "end": "2023-01-01",
        "query_identity": "dbnq1:abc",
        "price_domain": "raw_contract",
        "raw_or_continuous_role": "continuous_front",
        "raw_sha256": "deadbeef" * 8,
        "calendar_identity": "2026-09-phase04",
        "ingestion_version": "databento-source/1",
        "canonicalization_version": "1.0",
        "contract_definition_identity": "defs:xyz",
    }
    m1 = RealDatasetManifest(raw_artifact_relpath="data/raw/databento/GLBX.MDP3/x/a.dbn.zst", **base)
    m2 = RealDatasetManifest(raw_artifact_relpath="archive/2027/relocated/a.dbn.zst",
                             download_cost_usd=8.99, **base)
    assert m1.semantic_identity() == m2.semantic_identity()   # path + cost + timestamp excluded
    # a content-hash change DOES move the identity
    m3 = RealDatasetManifest(raw_artifact_relpath="x", **{**base, "raw_sha256": "0" * 64})
    assert m3.semantic_identity() != m1.semantic_identity()


def test_manifest_is_frozen():
    from pydantic import ValidationError

    m = RealDatasetManifest(
        databento_schema="definition", component_kind=ComponentKind.DEFINITIONS,
        root_symbol="CL", split_role=SplitRole.RESEARCH, symbol_request=("CL.FUT",),
        stype_in="parent", stype_out="instrument_id", start="2018-01-01", end="2018-01-08",
        query_identity="dbnq1:d", price_domain="n/a", raw_or_continuous_role="definitions",
        raw_artifact_relpath="x", raw_sha256="a" * 64, calendar_identity="v",
        ingestion_version="v", canonicalization_version="v", contract_definition_identity="v",
    )
    with pytest.raises(ValidationError):
        m.root_symbol = "ES"


def test_plan_windows_never_span_or_reveal_the_holdout_in_research_components():
    plan = load_plan(CONFIG, "B")
    ho_start = plan.holdout_span()[0]
    for c in plan.components(include_holdout=False):
        assert c.window.end <= ho_start          # no research/validation request reaches 2025
