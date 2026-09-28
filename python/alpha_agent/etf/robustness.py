"""Phase 6 instruction 2: a deterministic data-source robustness check.

Given any per-day scalar quantity computed from bars (a feature, a raw
signal value, a candidate factor reading -- this module does not care which),
compare it across every data source acquired for a ticker (its primary
listing venue, other acquired single venues, and EQUS.SUMMARY where their
windows overlap) and report whether the quantity is materially sensitive to
which market-data source computed it.

This is a REPORTING tool, not a merging tool -- Phase 6 instruction 4 ("do
not merge venue bars together ad hoc") is respected by construction: each
source's series is computed and compared independently, never combined into
one blended series.
"""
from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import numpy as np
import pandas as pd
from pydantic import BaseModel

from alpha_agent.etf.data_source import (
    load_consolidated_eod_summary_bars,
    load_other_single_venue_bars,
    load_primary_listing_bars,
)
from alpha_agent.etf.schemas import EtfDataSourceRole
from alpha_agent.etf.universe import primary_listing_dataset

__all__ = [
    "DataSourceSensitivityLevel",
    "DataSourceSensitivityReport",
    "PairwiseDataSourceComparison",
    "compare_across_data_sources",
]

#: Minimum overlapping trading days for a pairwise comparison to be scored at
#: all -- below this, a correlation estimate is too noisy to report as
#: evidence either way (an explicit, documented convention for this pilot,
#: not a derived statistical bound).
_MIN_OVERLAP_DAYS = 30
#: LOW: both thresholds met. HIGH: either threshold badly missed. MODERATE:
#: in between. Documented conventions, adjustable, never silently redefined.
_LOW_MIN_CORRELATION = 0.95
_LOW_MIN_SIGN_AGREEMENT = 0.90
_HIGH_MAX_CORRELATION = 0.80
_HIGH_MAX_SIGN_AGREEMENT = 0.70

_SINGLE_VENUE_DATASETS = ("ARCX.PILLAR", "XNAS.ITCH", "XNYS.PILLAR")


class DataSourceSensitivityLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    #: No comparison had enough overlapping days to score -- an honest "don't
    #: know" state, never silently reported as LOW.
    INSUFFICIENT_OVERLAP = "INSUFFICIENT_OVERLAP"


class PairwiseDataSourceComparison(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    reference_dataset: str
    reference_role: EtfDataSourceRole
    compared_dataset: str
    compared_role: EtfDataSourceRole
    n_overlap_days: int
    pearson_correlation: float | None
    sign_agreement_fraction: float | None


class DataSourceSensitivityReport(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    raw_symbol: str
    quantity_name: str
    comparisons: tuple[PairwiseDataSourceComparison, ...]
    overall_sensitivity: DataSourceSensitivityLevel
    overall_sensitivity_basis: str


def compare_across_data_sources(
    raw_symbol: str,
    *,
    quantity_name: str,
    compute_quantity: Callable[[pd.DataFrame], pd.Series],
) -> DataSourceSensitivityReport:
    """`compute_quantity` takes a bars frame (columns: day, open, high, low,
    close, volume, aligned/sorted by day) and returns a same-length Series
    (e.g. a rolling feature or raw signal reading). The PRIMARY LISTING VENUE
    is always the reference series every other source is compared against --
    it is the one series with the full 2018-2024 history (Phase 6
    instruction 1's chosen long-window research source)."""
    reference_frame, reference_prov = load_primary_listing_bars(raw_symbol)
    reference_series = pd.Series(
        compute_quantity(reference_frame).to_numpy(),
        index=pd.Index(reference_frame["day"].to_numpy()),
    )

    candidates: list[tuple[pd.DataFrame, object]] = []
    try:
        candidates.append(load_consolidated_eod_summary_bars(raw_symbol))
    except FileNotFoundError:
        pass
    for dataset in _SINGLE_VENUE_DATASETS:
        if dataset == primary_listing_dataset(raw_symbol):
            continue
        try:
            candidates.append(load_other_single_venue_bars(dataset, raw_symbol))
        except FileNotFoundError:
            pass

    comparisons: list[PairwiseDataSourceComparison] = []
    for frame, prov in candidates:
        series = pd.Series(
            compute_quantity(frame).to_numpy(), index=pd.Index(frame["day"].to_numpy())
        )
        joined = pd.concat([reference_series, series], axis=1, join="inner").dropna()
        n = len(joined)
        if n < _MIN_OVERLAP_DAYS:
            comparisons.append(
                PairwiseDataSourceComparison(
                    reference_dataset=reference_prov.dataset, reference_role=reference_prov.role,
                    compared_dataset=prov.dataset, compared_role=prov.role,
                    n_overlap_days=n, pearson_correlation=None, sign_agreement_fraction=None,
                )
            )
            continue
        a, b = joined.iloc[:, 0], joined.iloc[:, 1]
        corr = float(a.corr(b))
        sign_agreement = float((np.sign(a) == np.sign(b)).mean())
        comparisons.append(
            PairwiseDataSourceComparison(
                reference_dataset=reference_prov.dataset, reference_role=reference_prov.role,
                compared_dataset=prov.dataset, compared_role=prov.role,
                n_overlap_days=n, pearson_correlation=corr, sign_agreement_fraction=sign_agreement,
            )
        )

    scored = [c for c in comparisons if c.pearson_correlation is not None]
    if not scored:
        level = DataSourceSensitivityLevel.INSUFFICIENT_OVERLAP
        basis = f"no comparison had >= {_MIN_OVERLAP_DAYS} overlapping trading days"
    else:
        min_corr = min(c.pearson_correlation for c in scored)  # type: ignore[misc]
        min_sign = min(c.sign_agreement_fraction for c in scored)  # type: ignore[misc]
        if min_corr >= _LOW_MIN_CORRELATION and min_sign >= _LOW_MIN_SIGN_AGREEMENT:
            level = DataSourceSensitivityLevel.LOW
        elif min_corr < _HIGH_MAX_CORRELATION or min_sign < _HIGH_MAX_SIGN_AGREEMENT:
            level = DataSourceSensitivityLevel.HIGH
        else:
            level = DataSourceSensitivityLevel.MODERATE
        basis = (
            f"min pairwise correlation={min_corr:.3f}, min sign-agreement={min_sign:.3f} "
            f"across {len(scored)} comparison(s) with >= {_MIN_OVERLAP_DAYS} overlap days; "
            f"LOW: corr>={_LOW_MIN_CORRELATION} & sign>={_LOW_MIN_SIGN_AGREEMENT}; "
            f"HIGH: corr<{_HIGH_MAX_CORRELATION} or sign<{_HIGH_MAX_SIGN_AGREEMENT}; MODERATE otherwise"
        )

    return DataSourceSensitivityReport(
        raw_symbol=raw_symbol, quantity_name=quantity_name,
        comparisons=tuple(comparisons), overall_sensitivity=level,
        overall_sensitivity_basis=basis,
    )
