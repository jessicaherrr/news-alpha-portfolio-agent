"""Phase 9.1 -- Equities Foundation. No network, no data spend; proves the
new AssetDomain.EQUITY identity, the equities corporate-action/PIT-membership/
earnings-timestamp schemas, the instrument-economics adapter, and the fact
that no Equity data has been acquired and no Equity alpha experiment exists
yet.
"""
from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest
from alpha_agent.core.instrument import (
    AssetDomain,
    equity_instrument_identity,
    etf_instrument_identity,
    futures_instrument_identity,
)
from alpha_agent.equities import corporate_actions as ca
from alpha_agent.equities import data_availability as da
from alpha_agent.equities import earnings
from alpha_agent.equities import membership as mem
from alpha_agent.equities.calendar import equity_calendar
from alpha_agent.equities.data_source import (
    EQUITY_MULTIPLIER,
    EQUITY_TICK_SIZE,
    equity_contract_spec,
    load_primary_listing_bars,
)
from alpha_agent.equities.readiness import equities_foundation_readiness
from alpha_agent.equities.schemas import (
    CorporateActionCoverageStatus,
    CorporateActionSource,
    DelistingEvent,
    DelistingReason,
    EarningsTiming,
    EquityCashDistribution,
    EquityDataSourceProvenance,
    EquityDataSourceRole,
    EquityEarningsEvent,
    EquitySplitAction,
    EquityUniverseMembership,
    UniverseMembershipStatus,
)
from alpha_agent.equities.universe import (
    EQUITY_UNIVERSE,
    UNIVERSE_DECLARED_AT,
    equity_dsl_root_symbol,
    primary_listing_dataset,
)
from alpha_agent.registry.enums import AssetDomain as RegistryAssetDomain
from pydantic import ValidationError

_SOURCE = CorporateActionSource(
    name="test source", url="https://example.invalid/filing", retrieved_at=datetime(2026, 1, 1, tzinfo=UTC)
)


# ---------------------------------------------------------------------------
# Goal 2: AssetDomain.EQUITY at the correct shared identity boundary.
# ---------------------------------------------------------------------------


def test_asset_domain_equity_exists_and_is_distinct():
    assert RegistryAssetDomain.EQUITY.value == "EQUITY"
    assert RegistryAssetDomain.EQUITY != RegistryAssetDomain.FUTURES
    assert RegistryAssetDomain.EQUITY != RegistryAssetDomain.ETF
    # core.instrument re-exports the SAME object (`is`-identical), not a copy
    assert AssetDomain is RegistryAssetDomain


def test_equity_instrument_identity_is_structurally_isolated_from_futures_and_etf():
    equity_id = equity_instrument_identity("AAPL")
    etf_id = etf_instrument_identity("AAPL")  # hypothetical symbol collision
    futures_id = futures_instrument_identity("AAPL")
    assert equity_id.asset_domain == AssetDomain.EQUITY
    assert equity_id != etf_id
    assert equity_id != futures_id
    assert len({equity_id, etf_id, futures_id}) == 3  # frozen models are hashable; all distinct


def test_instrument_identity_is_frozen():
    ident = equity_instrument_identity("AAPL")
    with pytest.raises(ValidationError):
        ident.symbol = "MSFT"  # type: ignore[misc]


def test_alpha_graph_domain_loop_is_unchanged_by_the_new_enum_member():
    """Phase 9.3 scope, not 9.1: alpha_graph.builder's domain loop must still
    be exactly (FUTURES, ETF) today -- adding AssetDomain.EQUITY must not
    silently widen it."""
    import inspect

    from alpha_agent.alpha_graph import builder

    src = inspect.getsource(builder)
    assert "AssetDomain.FUTURES, AssetDomain.ETF" in src
    assert "AssetDomain.EQUITY" not in src


def test_ui_services_domain_tuple_is_unchanged_by_the_new_enum_member():
    """Same guard as above for ui/services.py's own hardcoded FUTURES/ETF
    tuple -- also explicitly Phase 9.3 scope."""
    import inspect

    from alpha_agent.ui import services

    src = inspect.getsource(services)
    assert '("FUTURES", futures_objs), ("ETF", etf_objs)' in src
    assert "EQUITY" not in src


# ---------------------------------------------------------------------------
# Universe: fixed, dated, 15-25 names.
# ---------------------------------------------------------------------------


def test_universe_size_matches_the_approved_15_to_25_range():
    assert 15 <= len(EQUITY_UNIVERSE) <= 25
    assert len(EQUITY_UNIVERSE) == len(set(EQUITY_UNIVERSE))  # no duplicates


def test_universe_declaration_date_is_fixed_not_derived_from_now():
    assert UNIVERSE_DECLARED_AT == date(2026, 9, 25)


def test_equity_dsl_root_symbol_never_collides_with_etf_or_a_raw_ticker():
    from alpha_agent.etf.universe import etf_dsl_root_symbol

    for ticker in EQUITY_UNIVERSE:
        root = equity_dsl_root_symbol(ticker)
        assert root != ticker
        assert root != etf_dsl_root_symbol(ticker)
        assert root.startswith("S")


def test_primary_listing_dataset_resolves_for_every_universe_member():
    for ticker in EQUITY_UNIVERSE:
        ds = primary_listing_dataset(ticker)
        assert ds in ("XNAS.ITCH", "XNYS.PILLAR")


def test_primary_listing_dataset_raises_for_unknown_ticker():
    with pytest.raises(KeyError):
        primary_listing_dataset("NOTATICKER")


# ---------------------------------------------------------------------------
# Typed schema requirements: every corporate action needs a real source.
# ---------------------------------------------------------------------------


def test_split_action_requires_a_source():
    with pytest.raises(ValidationError):
        EquitySplitAction.model_validate(
            {"raw_symbol": "AAPL", "effective_date": date(2020, 8, 31), "ratio": 4.0}
        )


def test_split_ratio_of_one_is_rejected():
    with pytest.raises(ValidationError):
        EquitySplitAction(raw_symbol="AAPL", effective_date=date(2020, 1, 1), ratio=1.0, source=_SOURCE)


def test_distribution_requires_pay_date_not_before_ex_date():
    with pytest.raises(ValidationError):
        EquityCashDistribution(
            raw_symbol="AAPL", ex_date=date(2024, 6, 1), pay_date=date(2024, 5, 1),
            amount_per_share_usd=0.24, source=_SOURCE,
        )


def test_distribution_requires_positive_amount():
    with pytest.raises(ValidationError):
        EquityCashDistribution(
            raw_symbol="AAPL", ex_date=date(2024, 6, 1), pay_date=date(2024, 6, 15),
            amount_per_share_usd=0.0, source=_SOURCE,
        )


# ---------------------------------------------------------------------------
# The known-actions store: real, cited, and honest about what isn't known.
# ---------------------------------------------------------------------------


def test_every_known_split_is_sourced_from_sec_gov():
    assert len(ca.KNOWN_SPLITS) == 7
    for split in ca.KNOWN_SPLITS:
        assert "sec.gov" in split.source.url.lower()
        assert split.raw_symbol in EQUITY_UNIVERSE


def test_aapl_split_matches_the_real_sec_filing():
    splits = ca.known_splits("AAPL")
    assert len(splits) == 1
    assert splits[0].effective_date == date(2020, 8, 31)
    assert splits[0].ratio == pytest.approx(4.0)


def test_no_fabricated_split_for_an_uninvestigated_ticker():
    for ticker in EQUITY_UNIVERSE:
        if ticker in {"AAPL", "TSLA", "NVDA", "GOOGL", "AMZN"}:
            continue
        assert ca.known_splits(ticker) == ()


def test_coverage_never_claims_distributions_are_sourced():
    for ticker in EQUITY_UNIVERSE:
        cov = ca.coverage(ticker)
        assert cov.distributions_status == CorporateActionCoverageStatus.NOT_YET_INVESTIGATED
        assert ca.known_distributions(ticker) == ()


def test_coverage_distinguishes_sourced_from_not_yet_investigated():
    aapl = ca.coverage("AAPL")
    msft = ca.coverage("MSFT")
    assert aapl.splits_status == CorporateActionCoverageStatus.SOURCED_FROM_OFFICIAL_RECORD
    assert msft.splits_status == CorporateActionCoverageStatus.NOT_YET_INVESTIGATED
    assert aapl.splits_status != msft.splits_status


def test_coverage_raises_for_a_ticker_outside_the_universe():
    with pytest.raises(KeyError):
        ca.coverage("NOTATICKER")


# ---------------------------------------------------------------------------
# Point-in-time universe membership / survivorship / delisting.
# ---------------------------------------------------------------------------


def test_every_universe_member_is_active_at_declaration():
    for ticker in EQUITY_UNIVERSE:
        m = mem.membership(ticker)
        assert m.status == UniverseMembershipStatus.ACTIVE
        assert m.delisting is None
        assert mem.is_active(ticker) is True
        assert mem.required_flat_by_date(ticker) is None


def test_membership_raises_for_unknown_ticker():
    with pytest.raises(KeyError):
        mem.membership("NOTATICKER")


def test_delisted_status_requires_a_delisting_event():
    with pytest.raises(ValidationError):
        EquityUniverseMembership(
            raw_symbol="ZZZZ", declared_at=UNIVERSE_DECLARED_AT,
            status=UniverseMembershipStatus.DELISTED, delisting=None,
        )


def test_active_status_forbids_a_delisting_event():
    delisting = DelistingEvent(
        raw_symbol="ZZZZ", last_trade_date=date(2023, 1, 1), delisting_date=date(2023, 1, 3),
        reason=DelistingReason.ACQUIRED_OR_MERGED, source=_SOURCE,
    )
    with pytest.raises(ValidationError):
        EquityUniverseMembership(
            raw_symbol="ZZZZ", declared_at=UNIVERSE_DECLARED_AT,
            status=UniverseMembershipStatus.ACTIVE, delisting=delisting,
        )


def test_delisting_event_symbol_mismatch_is_rejected():
    delisting = DelistingEvent(
        raw_symbol="OTHER", last_trade_date=date(2023, 1, 1), delisting_date=date(2023, 1, 3),
        reason=DelistingReason.ACQUIRED_OR_MERGED, source=_SOURCE,
    )
    with pytest.raises(ValidationError):
        EquityUniverseMembership(
            raw_symbol="ZZZZ", declared_at=UNIVERSE_DECLARED_AT,
            status=UniverseMembershipStatus.DELISTED, delisting=delisting,
        )


def test_delisting_date_before_last_trade_date_is_rejected():
    with pytest.raises(ValidationError):
        DelistingEvent(
            raw_symbol="ZZZZ", last_trade_date=date(2023, 1, 5), delisting_date=date(2023, 1, 1),
            reason=DelistingReason.BANKRUPTCY, source=_SOURCE,
        )


def test_truncate_bars_is_a_noop_for_an_active_name():
    bars = pd.DataFrame({"day": [date(2024, 1, 2), date(2024, 1, 3)], "close": [100.0, 101.0]})
    out = mem.truncate_bars_at_delisting(bars, "AAPL")
    assert len(out) == 2


def test_delisted_name_truncates_and_flags_post_delisting_bars(monkeypatch):
    delisting = DelistingEvent(
        raw_symbol="ZZZZ", last_trade_date=date(2023, 6, 1), delisting_date=date(2023, 6, 2),
        reason=DelistingReason.ACQUIRED_OR_MERGED, source=_SOURCE,
    )
    fake_membership = EquityUniverseMembership(
        raw_symbol="ZZZZ", declared_at=UNIVERSE_DECLARED_AT,
        status=UniverseMembershipStatus.DELISTED, delisting=delisting,
    )
    monkeypatch.setitem(mem.MEMBERSHIP, "ZZZZ", fake_membership)

    bars = pd.DataFrame({
        "day": [date(2023, 5, 30), date(2023, 6, 1), date(2023, 6, 2), date(2023, 6, 5)],
        "close": [10.0, 10.5, 10.6, 10.6],
    })
    assert mem.required_flat_by_date("ZZZZ") == date(2023, 6, 2)

    with pytest.raises(mem.PostDelistingDataError):
        mem.assert_no_post_delisting_bars(bars, "ZZZZ")

    truncated = mem.truncate_bars_at_delisting(bars, "ZZZZ")
    assert list(truncated["day"]) == [date(2023, 5, 30), date(2023, 6, 1), date(2023, 6, 2)]
    mem.assert_no_post_delisting_bars(truncated, "ZZZZ")  # now passes silently


# ---------------------------------------------------------------------------
# Earnings-event timestamps: PIT-honest, never fabricated.
# ---------------------------------------------------------------------------


def test_pit_available_earnings_event_requires_a_source():
    with pytest.raises(ValidationError):
        EquityEarningsEvent(
            raw_symbol="AAPL", fiscal_period="FY2099 Q1", announcement_date=date(2099, 1, 1),
            timing=EarningsTiming.AFTER_MARKET_CLOSE, point_in_time_available=True,
            point_in_time_note="no source attached", source=None,
        )


def test_pit_unknown_earnings_event_does_not_require_a_source():
    event = EquityEarningsEvent(
        raw_symbol="MSFT", fiscal_period="FY2024 Q1", announcement_date=date(2024, 1, 1),
        timing=EarningsTiming.UNKNOWN, point_in_time_available=None,
        point_in_time_note="not yet investigated", source=None,
    )
    assert event.point_in_time_available is None


def test_known_earnings_has_one_real_sourced_aapl_demonstration():
    events = earnings.known_earnings("AAPL")
    assert len(events) == 1
    assert events[0].announcement_date == date(2023, 11, 2)
    assert events[0].point_in_time_available is True
    assert "sec.gov" in events[0].source.url.lower()


def test_known_earnings_empty_for_uninvestigated_ticker():
    assert earnings.known_earnings("CAT") == ()


# ---------------------------------------------------------------------------
# Data-availability investigation: gates Phase 9.2's PEAD candidacy.
# ---------------------------------------------------------------------------


def test_pead_is_not_admissible_without_a_free_pit_consensus_source():
    assert da.pead_admissible_for_phase_9_2() is False


def test_earnings_timestamp_finding_is_free_and_self_serve():
    f = da.finding("earnings_announcement_timestamp")
    assert f.availability == da.DataSourceAvailability.AVAILABLE_FREE_SELF_SERVE


def test_unknown_concept_raises():
    with pytest.raises(KeyError):
        da.finding("not_a_real_concept")


# ---------------------------------------------------------------------------
# Data-source provenance + instrument-economics adapter.
# ---------------------------------------------------------------------------


def test_provenance_construction_refuses_whole_market_claim_on_a_single_venue():
    with pytest.raises(ValidationError):
        EquityDataSourceProvenance(
            dataset="XNAS.ITCH", role=EquityDataSourceRole.PRIMARY_LISTING_VENUE,
            listing_exchange="XNAS", volume_is_whole_market=True,
        )


def test_consolidated_role_may_claim_whole_market_volume():
    prov = EquityDataSourceProvenance(
        dataset="EQUS.SUMMARY", role=EquityDataSourceRole.CONSOLIDATED_EOD_SUMMARY,
        listing_exchange=None, volume_is_whole_market=True,
    )
    assert prov.volume_is_whole_market is True


def test_no_equity_market_data_has_been_acquired_yet():
    """Goal 5: Phase 9.1 must not have made a paid Equity data acquisition."""
    with pytest.raises(FileNotFoundError):
        load_primary_listing_bars("AAPL")


def test_equity_contract_spec_never_expires_within_the_research_window():
    spec = equity_contract_spec(1, "AAPL", activation_ns=1_000_000_000)
    window_end_ns = 1_735_689_600_000_000_000  # 2025-01-01, generous upper bound
    assert spec.expiration_ns > window_end_ns


def test_equity_contract_spec_multiplier_is_one_share_one_dollar():
    spec = equity_contract_spec(1, "AAPL", activation_ns=1_000_000_000)
    assert spec.multiplier == EQUITY_MULTIPLIER == 1.0
    assert spec.tick_size == EQUITY_TICK_SIZE == 0.01


def test_equity_contract_spec_root_symbol_is_the_dsl_virtual_root_not_the_ticker():
    spec = equity_contract_spec(1, "AAPL", activation_ns=1_000_000_000)
    assert spec.raw_symbol == "AAPL"
    assert spec.root_symbol == "SAAPL"
    assert spec.root_symbol != spec.raw_symbol


def test_equity_contract_spec_raises_for_unknown_ticker():
    with pytest.raises(KeyError):
        equity_contract_spec(1, "NOTATICKER", activation_ns=1_000_000_000)


# ---------------------------------------------------------------------------
# Calendar.
# ---------------------------------------------------------------------------


def test_equity_calendar_has_every_universe_root_and_weekends_closed():
    cal = equity_calendar()
    for ticker in EQUITY_UNIVERSE:
        assert cal.has_root(ticker)
    # Saturday 2024-06-01 should be weekly-closed under the default rule
    assert cal.calendar_name_for("AAPL") == "us_equity_daily"


# ---------------------------------------------------------------------------
# Readiness summary: honest, no fake PASS, no UI wiring required.
# ---------------------------------------------------------------------------


def test_readiness_reports_no_market_data_acquired():
    r = equities_foundation_readiness()
    assert r.market_data_acquired is False
    assert r.universe_size == len(EQUITY_UNIVERSE)
    assert r.active_members == len(EQUITY_UNIVERSE)
    assert r.delisted_members == 0
    assert r.splits_sourced_count == 5  # AAPL, TSLA, NVDA, GOOGL, AMZN
    assert r.earnings_events_sourced_count == 1
    assert r.pead_admissible_for_phase_9_2 is False


# ---------------------------------------------------------------------------
# Goal 4 / acceptance: no Equity alpha experiment exists yet.
# ---------------------------------------------------------------------------


def test_no_equity_alpha_memory_bridge_module_exists_yet():
    """Phase 9.1 explicitly does not wire Equities into Alpha Memory /
    Registry append (that is Phase 9.2/9.3 scope)."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("alpha_agent.equities.alpha_memory_bridge")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("alpha_agent.equities.registry_append")
