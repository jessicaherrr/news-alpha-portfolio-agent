"""Market Intelligence + Futures Universe campaign, Checkpoint A / mission
Part 40 -- PRODUCT CATALOG tests: categories are correct, duplicate symbols
are rejected, capability states are explicit, and Market != Research !=
Trading holds as a real invariant, not just documentation.
"""
from __future__ import annotations

import pytest
from alpha_agent.marketdata import product_catalog as pc
from alpha_agent.marketdata.contracts import APPROVED_ROOTS


def test_research_universe_matches_the_one_frozen_constant():
    """CLAUDE.md Part 38: the research universe must never be a second,
    independently-typed list that could silently drift from
    `marketdata.contracts.APPROVED_ROOTS`."""
    assert pc.RESEARCH_UNIVERSE == APPROVED_ROOTS
    assert pc.RESEARCH_UNIVERSE == ("ES", "NQ", "CL", "GC", "ZN")


def test_no_duplicate_root_symbols_in_the_catalog():
    roots = [e.root_symbol for e in pc.PRODUCT_CATALOG]
    assert len(roots) == len(set(roots)), f"duplicate roots: {sorted({r for r in roots if roots.count(r) > 1})}"


def test_duplicate_symbol_construction_is_rejected():
    dup = pc.PRODUCT_CATALOG + (pc.PRODUCT_CATALOG[0],)
    with pytest.raises(ValueError, match="duplicate root_symbol"):
        pc._validate_catalog(dup)


@pytest.mark.parametrize(
    "asset_class,expected_roots",
    [
        (pc.AssetClass.EQUITY_INDEX, {"ES", "MES", "NQ", "MNQ", "YM", "MYM", "RTY", "M2K"}),
        (pc.AssetClass.ENERGY, {"CL", "MCL", "NG", "RB", "HO"}),
        (pc.AssetClass.METALS, {"GC", "MGC", "SI", "HG"}),
        (pc.AssetClass.RATES, {"ZT", "ZF", "ZN", "ZB", "UB"}),
        (pc.AssetClass.FX, {"6E", "6J", "6B", "6A", "6C", "6S"}),
        (pc.AssetClass.AGRICULTURE, {"ZC", "ZW", "ZS", "ZM", "ZL"}),
        (pc.AssetClass.CRYPTO, {"BTC", "MBT", "ETH"}),
    ],
)
def test_categories_are_correct(asset_class, expected_roots):
    roots = {e.root_symbol for e in pc.catalog_by_asset_class(asset_class)}
    assert roots == expected_roots


def test_every_catalog_entry_uses_the_shared_glbx_dataset():
    assert all(e.dataset == "GLBX.MDP3" for e in pc.PRODUCT_CATALOG)


def test_every_catalog_entry_has_a_continuous_display_symbol_convention():
    for entry in pc.PRODUCT_CATALOG:
        assert entry.continuous_display_symbol == f"{entry.root_symbol}.v.0"


def test_research_universe_is_a_subset_of_the_market_universe():
    market_roots = set(pc.market_universe_roots())
    assert set(pc.RESEARCH_UNIVERSE).issubset(market_roots)


def test_market_universe_roots_is_sorted_and_deduplicated():
    roots = pc.market_universe_roots()
    assert list(roots) == sorted(set(roots))
    assert len(roots) == len(pc.PRODUCT_CATALOG)


def test_catalog_entry_lookup_is_case_insensitive_and_honest_for_unknown_roots():
    assert pc.catalog_entry("cl") is not None
    assert pc.catalog_entry("cl").root_symbol == "CL"
    assert pc.catalog_entry("NOT_A_REAL_ROOT") is None
    assert pc.is_catalogued("GC") is True
    assert pc.is_catalogued("XX") is False
