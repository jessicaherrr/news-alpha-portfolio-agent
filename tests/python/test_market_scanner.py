"""Market Intelligence + Futures Universe campaign, Checkpoint A / mission
Part 40 -- SCANNER tests: real/fake injected provider values only (no
network, no hard-coded prices), deterministic sorting, and honest
degradation for missing products.
"""
from __future__ import annotations

from datetime import UTC, datetime

from alpha_agent.marketdata.capability import CapabilityState, FuturesProductCapability
from alpha_agent.marketdata.databento_schemas import (
    ContractResolution,
    MarketSnapshot,
    SessionSummary,
)
from alpha_agent.marketdata.product_catalog import AssetClass, catalog_entry
from alpha_agent.ui import market_scanner as ms

_CATALOG = (catalog_entry("NQ"), catalog_entry("CL"), catalog_entry("NG"))


def _cap(root: str, *, available: bool, research: bool, trading: bool) -> FuturesProductCapability:
    entry = catalog_entry(root)
    state = CapabilityState.AVAILABLE if available else CapabilityState.NOT_TESTED
    return FuturesProductCapability(
        root_symbol=root, display_name=entry.display_name, asset_class=entry.asset_class,
        venue=entry.venue, dataset=entry.dataset,
        market_data_available=state, historical_data_available=state,
        news_available=CapabilityState.DISABLED, event_calendar_available=CapabilityState.DISABLED,
        research_enabled=research, paper_trading_enabled=trading, challenge_eligible=False,
        contract_economics_verified=state, roll_logic_verified=state, cpp_execution_verified=state,
        capability_reason="test fixture",
    )


def _snapshot(root: str, *, last: float, change_pct: float, volume: float) -> MarketSnapshot:
    return MarketSnapshot(
        root_symbol=root,
        contract=ContractResolution(root_symbol=root, display_symbol=f"{root}.v.0", resolved_at=datetime.now(UTC)),
        last=last, change_pct=change_pct,
        session=SessionSummary(root_symbol=root, session_date="2026-09-11", volume=volume),
        as_of=datetime.now(UTC), capability="LATEST_AVAILABLE",
    )


_CAPS = {
    "NQ": _cap("NQ", available=True, research=True, trading=True),
    "CL": _cap("CL", available=True, research=True, trading=False),
    "NG": _cap("NG", available=False, research=False, trading=False),
}

_SNAPSHOTS = {
    "NQ": _snapshot("NQ", last=20000.0, change_pct=-0.5, volume=1_000_000.0),
    "CL": _snapshot("CL", last=80.0, change_pct=2.5, volume=500_000.0),
    # NG has no snapshot at all -- not wired this checkpoint.
}


def test_build_scanner_rows_never_fabricates_a_missing_snapshot():
    rows = ms.build_scanner_rows(_CATALOG, capabilities=_CAPS, snapshots=_SNAPSHOTS)
    by_root = {r.root_symbol: r for r in rows}
    assert by_root["NG"].last is None
    assert by_root["NG"].change_pct is None
    assert by_root["NG"].market_data_state is CapabilityState.NOT_TESTED
    assert by_root["NQ"].last == 20000.0
    assert by_root["CL"].change_pct == 2.5


def test_build_scanner_rows_carries_research_and_trading_status_honestly():
    rows = ms.build_scanner_rows(_CATALOG, capabilities=_CAPS, snapshots=_SNAPSHOTS)
    by_root = {r.root_symbol: r for r in rows}
    assert by_root["NQ"].research_status is ms.ResearchStatus.RESEARCH_ENABLED
    assert by_root["NQ"].trading_status is ms.TradingStatus.PAPER_ELIGIBLE
    assert by_root["CL"].research_status is ms.ResearchStatus.RESEARCH_ENABLED
    assert by_root["CL"].trading_status is ms.TradingStatus.NOT_ELIGIBLE
    assert by_root["NG"].research_status is ms.ResearchStatus.NOT_CERTIFIED


def test_build_scanner_rows_never_defaults_news_count_to_a_fabricated_zero():
    """Section 2: before a NewsStore is wired, every row must read
    NOT_CONNECTED with `news_count_recent=None` -- never a numeric `0`, which
    would falsely claim "we queried and found nothing"."""
    rows = ms.build_scanner_rows(_CATALOG, capabilities=_CAPS, snapshots=_SNAPSHOTS)
    assert all(r.news_state is CapabilityState.NOT_CONNECTED for r in rows)
    assert all(r.news_count_recent is None for r in rows)
    assert all(r.event_state is CapabilityState.NOT_CONNECTED for r in rows)
    assert all(r.high_impact_event_soon is False for r in rows)


def test_build_scanner_rows_reports_a_real_zero_news_count_only_when_available():
    news_by_root = {"NQ": (CapabilityState.AVAILABLE, 0), "CL": (CapabilityState.AVAILABLE, 3)}
    rows = ms.build_scanner_rows(_CATALOG, capabilities=_CAPS, snapshots=_SNAPSHOTS, news_by_root=news_by_root)
    by_root = {r.root_symbol: r for r in rows}
    assert by_root["NQ"].news_count_recent == 0
    assert by_root["CL"].news_count_recent == 3
    assert by_root["NG"].news_state is CapabilityState.NOT_CONNECTED
    assert by_root["NG"].news_count_recent is None


# ---------------------------------------------------------------------------
# sorting -- deterministic, same input => same output
# ---------------------------------------------------------------------------


def test_sort_by_symbol_is_alphabetical():
    rows = ms.build_scanner_rows(_CATALOG, capabilities=_CAPS, snapshots=_SNAPSHOTS)
    sorted_rows = ms.sort_rows(rows, "Symbol")
    assert [r.root_symbol for r in sorted_rows] == ["CL", "NG", "NQ"]


def test_sort_by_top_movers_ranks_by_absolute_change_missing_last():
    rows = ms.build_scanner_rows(_CATALOG, capabilities=_CAPS, snapshots=_SNAPSHOTS)
    sorted_rows = ms.sort_rows(rows, "Top Movers")
    assert [r.root_symbol for r in sorted_rows] == ["CL", "NQ", "NG"]  # |2.5| > |-0.5| > None


def test_sort_by_highest_volume_missing_last():
    rows = ms.build_scanner_rows(_CATALOG, capabilities=_CAPS, snapshots=_SNAPSHOTS)
    sorted_rows = ms.sort_rows(rows, "Highest Volume")
    assert [r.root_symbol for r in sorted_rows] == ["NQ", "CL", "NG"]


def test_sort_by_research_enabled_puts_certified_roots_first():
    rows = ms.build_scanner_rows(_CATALOG, capabilities=_CAPS, snapshots=_SNAPSHOTS)
    sorted_rows = ms.sort_rows(rows, "Research Enabled")
    assert sorted_rows[-1].root_symbol == "NG"
    assert {r.root_symbol for r in sorted_rows[:2]} == {"NQ", "CL"}


def test_sort_by_paper_trading_eligible_puts_eligible_roots_first():
    rows = ms.build_scanner_rows(_CATALOG, capabilities=_CAPS, snapshots=_SNAPSHOTS)
    sorted_rows = ms.sort_rows(rows, "Paper-Trading Eligible")
    assert sorted_rows[0].root_symbol == "NQ"


def test_sorting_is_deterministic_same_input_same_output():
    rows = ms.build_scanner_rows(_CATALOG, capabilities=_CAPS, snapshots=_SNAPSHOTS)
    a = [r.root_symbol for r in ms.sort_rows(rows, "Highest Volatility")]
    b = [r.root_symbol for r in ms.sort_rows(rows, "Highest Volatility")]
    assert a == b


# ---------------------------------------------------------------------------
# filtering
# ---------------------------------------------------------------------------


def test_filter_by_asset_class():
    rows = ms.build_scanner_rows(_CATALOG, capabilities=_CAPS, snapshots=_SNAPSHOTS)
    energy = ms.filter_rows(rows, asset_class=AssetClass.ENERGY)
    assert {r.root_symbol for r in energy} == {"CL", "NG"}


def test_filter_by_search_matches_symbol_or_display_name_case_insensitively():
    rows = ms.build_scanner_rows(_CATALOG, capabilities=_CAPS, snapshots=_SNAPSHOTS)
    assert {r.root_symbol for r in ms.filter_rows(rows, search="nq")} == {"NQ"}
    assert {r.root_symbol for r in ms.filter_rows(rows, search="crude")} == {"CL"}
    assert ms.filter_rows(rows, search="not-a-real-product") == ()


def test_filter_with_no_criteria_returns_everything():
    rows = ms.build_scanner_rows(_CATALOG, capabilities=_CAPS, snapshots=_SNAPSHOTS)
    assert ms.filter_rows(rows) == rows


# ---------------------------------------------------------------------------
# click-to-select translation (pure -- see root_for_selection's docstring for
# why this is tested independently of any Streamlit dataframe interaction)
# ---------------------------------------------------------------------------


def test_root_for_selection_maps_row_index_to_root_symbol():
    rows = ms.sort_rows(ms.build_scanner_rows(_CATALOG, capabilities=_CAPS, snapshots=_SNAPSHOTS), "Symbol")
    assert [r.root_symbol for r in rows] == ["CL", "NG", "NQ"]
    assert ms.root_for_selection(rows, [1]) == "NG"
    assert ms.root_for_selection(rows, [0]) == "CL"


def test_root_for_selection_none_when_nothing_selected():
    rows = ms.build_scanner_rows(_CATALOG, capabilities=_CAPS, snapshots=_SNAPSHOTS)
    assert ms.root_for_selection(rows, []) is None


def test_root_for_selection_never_indexes_past_the_visible_rows():
    rows = ms.build_scanner_rows(_CATALOG, capabilities=_CAPS, snapshots=_SNAPSHOTS)
    assert ms.root_for_selection(rows, [999]) is None
    assert ms.root_for_selection(rows, [-1]) is None


# ---------------------------------------------------------------------------
# Section 1/35 -- a blank price cell must never be ambiguous
# ---------------------------------------------------------------------------


def test_price_cell_shows_state_name_when_not_loaded_or_not_tested():
    rows = ms.build_scanner_rows(_CATALOG, capabilities=_CAPS, snapshots=_SNAPSHOTS)
    by_root = {r.root_symbol: r for r in rows}
    cell = ms._price_cell(by_root["NG"])
    assert "—" in cell
    assert "NOT TESTED" in cell.upper()


def test_price_cell_shows_the_real_price_when_available():
    rows = ms.build_scanner_rows(_CATALOG, capabilities=_CAPS, snapshots=_SNAPSHOTS)
    by_root = {r.root_symbol: r for r in rows}
    assert ms._price_cell(by_root["NQ"]) == "20,000.00"


def test_price_cell_shows_unavailable_state_with_no_dash_when_a_real_attempt_failed():
    from alpha_agent.marketdata.capability import CapabilityState

    row = ms.MarketScannerRow(
        root_symbol="XX", display_name="Test", asset_class=AssetClass.ENERGY,
        market_data_state=CapabilityState.UNAVAILABLE,
        research_status=ms.ResearchStatus.NOT_CERTIFIED, trading_status=ms.TradingStatus.NOT_ELIGIBLE,
    )
    assert ms._price_cell(row) == "UNAVAILABLE"


def test_news_cell_shows_state_not_a_number_before_connected():
    rows = ms.build_scanner_rows(_CATALOG, capabilities=_CAPS, snapshots=_SNAPSHOTS)
    by_root = {r.root_symbol: r for r in rows}
    assert ms._news_cell(by_root["NQ"]) == "NOT CONNECTED"


def test_news_cell_shows_a_real_zero_once_available():
    from alpha_agent.marketdata.capability import CapabilityState

    rows = ms.build_scanner_rows(
        _CATALOG, capabilities=_CAPS, snapshots=_SNAPSHOTS,
        news_by_root={"NQ": (CapabilityState.AVAILABLE, 0)},
    )
    by_root = {r.root_symbol: r for r in rows}
    assert ms._news_cell(by_root["NQ"]) == "0"


# ---------------------------------------------------------------------------
# category-scoped lazy loading (Section 3/36)
# ---------------------------------------------------------------------------


def test_loaded_roots_key_and_fetched_once_key_are_distinct_per_prefix():
    assert ms._loaded_roots_key("market") != ms._fetched_once_key("market")
    assert ms._loaded_roots_key("market") != ms._loaded_roots_key("home")


# ---------------------------------------------------------------------------
# Section 32 -- event state wiring
# ---------------------------------------------------------------------------


def test_event_cell_shows_state_when_not_available():
    row = ms.MarketScannerRow(
        root_symbol="NQ", display_name="Test", asset_class=AssetClass.EQUITY_INDEX,
        market_data_state=CapabilityState.NOT_LOADED, event_state=CapabilityState.NOT_CONNECTED,
        research_status=ms.ResearchStatus.RESEARCH_ENABLED, trading_status=ms.TradingStatus.NOT_ELIGIBLE,
    )
    assert ms._event_cell(row) == "NOT CONNECTED"


def test_event_cell_shows_none_scheduled_when_available_with_no_event():
    row = ms.MarketScannerRow(
        root_symbol="NQ", display_name="Test", asset_class=AssetClass.EQUITY_INDEX,
        market_data_state=CapabilityState.NOT_LOADED, event_state=CapabilityState.AVAILABLE,
        next_high_impact_event=None,
        research_status=ms.ResearchStatus.RESEARCH_ENABLED, trading_status=ms.TradingStatus.NOT_ELIGIBLE,
    )
    assert ms._event_cell(row) == "None scheduled"


# ---------------------------------------------------------------------------
# Product UI Polish pass -- primary/secondary split, capability summary, and
# featured coverage (task spec sections 3/4/5/6)
# ---------------------------------------------------------------------------


def test_split_primary_secondary_keeps_only_real_prices_in_primary():
    rows = ms.build_scanner_rows(_CATALOG, capabilities=_CAPS, snapshots=_SNAPSHOTS)
    primary, secondary = ms.split_primary_secondary(rows)
    assert {r.root_symbol for r in primary} == {"NQ", "CL"}
    assert {r.root_symbol for r in secondary} == {"NG"}


def test_split_primary_secondary_is_exhaustive_and_non_overlapping():
    rows = ms.build_scanner_rows(_CATALOG, capabilities=_CAPS, snapshots=_SNAPSHOTS)
    primary, secondary = ms.split_primary_secondary(rows)
    assert len(primary) + len(secondary) == len(rows)
    primary_roots = {r.root_symbol for r in primary}
    secondary_roots = {r.root_symbol for r in secondary}
    assert primary_roots | secondary_roots == {r.root_symbol for r in rows}
    assert not (primary_roots & secondary_roots)
    # order-preserving within each partition (relative catalog order kept)
    assert [r.root_symbol for r in primary] == [r.root_symbol for r in rows if r.root_symbol in primary_roots]


def test_secondary_reason_is_short_and_honest_never_the_long_capability_paragraph():
    rows = ms.build_scanner_rows(_CATALOG, capabilities=_CAPS, snapshots=_SNAPSHOTS)
    by_root = {r.root_symbol: r for r in rows}
    reason = ms._short_unavailable_reason(by_root["NG"])
    assert reason == "Not currently observed"
    assert reason != by_root["NG"].detail  # never the long capability_reason paragraph


def test_row_to_secondary_dict_has_no_research_or_trading_columns():
    rows = ms.build_scanner_rows(_CATALOG, capabilities=_CAPS, snapshots=_SNAPSHOTS)
    by_root = {r.root_symbol: r for r in rows}
    cell = ms._row_to_secondary_dict(by_root["NG"])
    assert set(cell) == {"Symbol", "Product", "Asset Class", "Reason"}


def test_row_to_table_dict_no_longer_carries_per_row_research_trading_columns():
    """Section 4: capability explanation moves to a once-per-page summary --
    the primary table's own row dict must never repeat Research/Trading."""
    rows = ms.build_scanner_rows(_CATALOG, capabilities=_CAPS, snapshots=_SNAPSHOTS)
    cell = ms._row_to_table_dict(rows[0])
    assert "Research" not in cell
    assert "Trading" not in cell


def test_featured_coverage_has_at_least_three_real_catalogued_roots_per_asset_class():
    from alpha_agent.marketdata.product_catalog import AssetClass, catalog_entry

    assert set(ms.FEATURED_COVERAGE) == set(AssetClass)
    for asset_class, roots in ms.FEATURED_COVERAGE.items():
        assert len(roots) >= 3, f"{asset_class} has fewer than 3 featured roots"
        for root in roots:
            entry = catalog_entry(root)
            assert entry is not None, f"featured root {root!r} is not in the product catalog"
            assert entry.asset_class is asset_class


def test_event_cell_shows_the_real_label_when_available():
    row = ms.MarketScannerRow(
        root_symbol="NQ", display_name="Test", asset_class=AssetClass.EQUITY_INDEX,
        market_data_state=CapabilityState.NOT_LOADED, event_state=CapabilityState.AVAILABLE,
        next_high_impact_event="FOMC Policy Statement (2026-09-16 18:00 UTC)",
        research_status=ms.ResearchStatus.RESEARCH_ENABLED, trading_status=ms.TradingStatus.NOT_ELIGIBLE,
    )
    assert ms._event_cell(row) == "FOMC Policy Statement (2026-09-16 18:00 UTC)"
