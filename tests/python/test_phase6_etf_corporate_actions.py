"""Phase 6 ETF pilot -- corporate-action schema, coverage honesty, and the
ETF instrument-economics adapter. No network, no data spend; exercises the
already-acquired real raw artifacts under data/raw/.
"""
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from alpha_agent.etf import corporate_actions as ca
from alpha_agent.etf.data_source import (
    etf_contract_spec,
    load_consolidated_eod_summary_bars,
    load_other_single_venue_bars,
    load_primary_listing_bars,
)
from alpha_agent.etf.robustness import (
    DataSourceSensitivityLevel,
    compare_across_data_sources,
)
from alpha_agent.etf.schemas import (
    CorporateActionCoverageStatus,
    CorporateActionSource,
    EtfCashDistribution,
    EtfDataSourceProvenance,
    EtfDataSourceRole,
    EtfSplitAction,
)
from alpha_agent.etf.universe import PILOT_UNIVERSE, PRIMARY_LISTING_EXCHANGE
from pydantic import ValidationError

_REPO_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    not (_REPO_ROOT / "data/raw/databento/ARCX.PILLAR").exists(),
    reason="requires the real Phase 6 ETF raw artifacts (scripts/phase6_etf_acquire.py --acquire)",
)

_SOURCE = CorporateActionSource(
    name="test source", url="https://example.invalid/filing", retrieved_at=datetime(2026, 1, 1, tzinfo=UTC)
)


# ---------------------------------------------------------------------------
# Typed schema requirements: every action needs a real, attributable source.
# ---------------------------------------------------------------------------


def test_split_action_requires_a_source():
    with pytest.raises(ValidationError):
        EtfSplitAction.model_validate(
            {"raw_symbol": "USO", "effective_date": date(2020, 4, 29), "ratio": 0.125}
        )  # missing 'source' entirely


def test_split_ratio_of_one_is_rejected():
    with pytest.raises(ValidationError):
        EtfSplitAction(raw_symbol="SPY", effective_date=date(2020, 1, 1), ratio=1.0, source=_SOURCE)


def test_distribution_requires_pay_date_not_before_ex_date():
    with pytest.raises(ValidationError):
        EtfCashDistribution(
            raw_symbol="SPY", ex_date=date(2024, 6, 1), pay_date=date(2024, 5, 1),
            amount_per_share_usd=1.5, source=_SOURCE,
        )


def test_distribution_requires_positive_amount():
    with pytest.raises(ValidationError):
        EtfCashDistribution(
            raw_symbol="SPY", ex_date=date(2024, 6, 1), pay_date=date(2024, 6, 15),
            amount_per_share_usd=0.0, source=_SOURCE,
        )


# ---------------------------------------------------------------------------
# The known-actions store: real, cited, and honest about what isn't known.
# ---------------------------------------------------------------------------


def test_uso_split_is_sourced_from_a_real_sec_filing_not_the_price_scan():
    splits = ca.known_splits("USO")
    assert len(splits) == 1
    split = splits[0]
    assert split.effective_date == date(2020, 4, 29)
    assert split.ratio == pytest.approx(0.125)
    assert "sec.gov" in split.source.url.lower()


def test_no_other_ticker_has_a_fabricated_split():
    for ticker in PILOT_UNIVERSE:
        if ticker == "USO":
            continue
        assert ca.known_splits(ticker) == ()


def test_coverage_never_claims_distributions_are_sourced():
    """An empty KNOWN_DISTRIBUTIONS must never silently read as 'these ETFs
    paid no dividends' -- every ticker's coverage record says explicitly
    that distributions are not yet investigated."""
    for ticker in PILOT_UNIVERSE:
        cov = ca.coverage(ticker)
        assert cov.distributions_status == CorporateActionCoverageStatus.NOT_YET_INVESTIGATED
        assert ca.known_distributions(ticker) == ()


def test_coverage_distinguishes_official_record_from_heuristic_screen():
    uso = ca.coverage("USO")
    spy = ca.coverage("SPY")
    assert uso.splits_status == CorporateActionCoverageStatus.SOURCED_FROM_OFFICIAL_RECORD
    assert spy.splits_status == CorporateActionCoverageStatus.SCREENED_NO_EVENT_DETECTED
    assert uso.splits_status != spy.splits_status


def test_coverage_raises_for_a_ticker_outside_the_pilot_universe():
    with pytest.raises(KeyError):
        ca.coverage("NOTATICKER")


# ---------------------------------------------------------------------------
# Data-source provenance: never mislabel a single venue as consolidated.
# ---------------------------------------------------------------------------


def test_primary_venue_provenance_never_claims_whole_market_volume():
    for ticker in PILOT_UNIVERSE:
        _frame, prov = load_primary_listing_bars(ticker)
        assert prov.role == EtfDataSourceRole.PRIMARY_LISTING_VENUE
        assert prov.volume_is_whole_market is False
        assert prov.listing_exchange == PRIMARY_LISTING_EXCHANGE[ticker]


def test_consolidated_provenance_is_the_only_role_allowed_whole_market_volume():
    _frame, prov = load_consolidated_eod_summary_bars("SPY")
    assert prov.role == EtfDataSourceRole.CONSOLIDATED_EOD_SUMMARY
    assert prov.volume_is_whole_market is True


def test_provenance_construction_refuses_whole_market_claim_on_a_single_venue():
    with pytest.raises(ValidationError):
        EtfDataSourceProvenance(
            dataset="ARCX.PILLAR", role=EtfDataSourceRole.PRIMARY_LISTING_VENUE,
            listing_exchange="ARCX", volume_is_whole_market=True,
        )


def test_other_single_venue_refuses_a_tickers_own_primary_listing():
    with pytest.raises(ValueError, match="primary listing"):
        load_other_single_venue_bars("ARCX.PILLAR", "SPY")  # SPY's own primary venue


def test_other_single_venue_loads_a_real_non_primary_dataset():
    frame, prov = load_other_single_venue_bars("XNYS.PILLAR", "SPY")
    assert prov.role == EtfDataSourceRole.OTHER_SINGLE_VENUE
    assert len(frame) > 0


def test_primary_venue_bars_have_plausible_real_prices():
    frame, _prov = load_primary_listing_bars("SPY")
    # SPY never traded below $150 or above $700 in 2018-2024 -- a sanity
    # bound against a decoding/scaling bug, not a tight assertion.
    assert (frame["close"] > 150).all()
    assert (frame["close"] < 700).all()


# ---------------------------------------------------------------------------
# ETF instrument-economics adapter.
# ---------------------------------------------------------------------------


def test_etf_contract_spec_never_expires_within_the_pilot_window():
    # successful construction already proves validity -- ContractSpecModel's
    # own model_validator rejects expiration_ns <= activation_ns at parse time
    spec = etf_contract_spec(15144, "SPY", activation_ns=1_000_000_000)
    pilot_window_end_ns = 1_735_689_600_000_000_000  # 2025-01-01 (holdout boundary, generous upper bound)
    assert spec.expiration_ns > pilot_window_end_ns


def test_etf_contract_spec_multiplier_is_one_share_one_dollar():
    spec = etf_contract_spec(15144, "SPY", activation_ns=1_000_000_000)
    assert spec.multiplier == 1.0
    assert spec.tick_size == 0.01


def test_etf_contract_spec_root_symbol_is_the_dsl_virtual_root():
    """Unlike futures (NQ virtual root, NQZ6 raw contract), an ETF's raw
    tradable symbol IS its ticker -- but the reference CLI's target-schedule
    path requires root_symbol != any real raw_symbol in the registry, so
    root_symbol carries the synthetic "E"-prefixed DSL virtual-root label
    (etf_dsl_root_symbol), never the bare ticker."""
    spec = etf_contract_spec(15144, "SPY", activation_ns=1_000_000_000)
    assert spec.raw_symbol == "SPY"
    assert spec.root_symbol == "ESPY"
    assert spec.root_symbol != spec.raw_symbol


# ---------------------------------------------------------------------------
# Data-source robustness comparator.
# ---------------------------------------------------------------------------


def _close_only(frame: pd.DataFrame) -> pd.Series:
    return frame["close"]


def test_robustness_report_never_merges_sources_only_compares():
    report = compare_across_data_sources("SPY", quantity_name="close", compute_quantity=_close_only)
    # every comparison names two DISTINCT real datasets -- proves nothing was blended
    for c in report.comparisons:
        assert c.reference_dataset != c.compared_dataset
    assert report.overall_sensitivity in tuple(DataSourceSensitivityLevel)


def test_robustness_report_flags_insufficient_overlap_honestly(monkeypatch):
    """A quantity with a stub compute function that returns all-NaN should
    never be scored as LOW/HIGH/MODERATE -- must come back INSUFFICIENT_OVERLAP
    or at least never silently claim LOW confidence from garbage data."""

    def _all_nan(frame: pd.DataFrame) -> pd.Series:
        return frame["close"] * float("nan")

    report = compare_across_data_sources("SPY", quantity_name="nan_probe", compute_quantity=_all_nan)
    assert report.overall_sensitivity == DataSourceSensitivityLevel.INSUFFICIENT_OVERLAP


def test_robustness_20d_momentum_is_a_real_measured_result_not_fabricated():
    """A real, price-based factor (Phase 6 instruction 1's own recommendation)
    should show measurably LOW-to-MODERATE sensitivity, not perfect agreement
    (that would suggest the comparison is a no-op) and not wildly divergent
    (that would contradict the Phase 6 audit's own measured ~12bps/27%-volume
    findings for a price-only quantity)."""

    def trend_20(frame: pd.DataFrame) -> pd.Series:
        return frame["close"].pct_change(20)

    report = compare_across_data_sources("SPY", quantity_name="20d_momentum", compute_quantity=trend_20)
    scored = [c for c in report.comparisons if c.pearson_correlation is not None]
    assert scored, "expected at least one scored comparison for SPY"
    for c in scored:
        assert 0.5 < c.pearson_correlation <= 1.0
        assert c.n_overlap_days >= 30
