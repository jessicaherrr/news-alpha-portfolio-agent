"""The Phase 9.1 equities universe's known-corporate-actions store.

Every entry here is either a real, attributable, cited record (never
inferred from a price jump -- same rule as Phase 6 ETF) or an explicit
"not yet sourced" fact (never a silent empty-means-nothing-happened
assumption).

Splits: seven real splits across four of the twenty universe tickers were
sourced directly from SEC EDGAR Form 8-K filings (Item 5.03 / Item 8.01
exhibits announcing each split), each opened and quote-verified against the
live filing, not derived from a price scan and not guessed from public
knowledge alone. The remaining sixteen tickers are NOT_YET_INVESTIGATED for
splits -- several of them (e.g. WMT's 2024 3-for-1 split) are publicly known
to have split within the window but have not yet been individually sourced
and cited here; do not add one without a real citation.

Distributions: every ticker in this universe is a well-established dividend
payer, but -- exactly as the Phase 6 ETF pilot found -- Databento carries no
corporate-actions schema, and no free, self-serve, 2018-2024-depth dividend
source has yet been connected. NOT_YET_INVESTIGATED throughout; do not
populate without a real citation.

Earnings-announcement timestamps: see ``alpha_agent.equities.earnings`` --
handled separately since it is a distinct concept (event date, not a
position-adjusting action) with its own point-in-time honesty pattern.
"""
from __future__ import annotations

from datetime import UTC, date, datetime

from alpha_agent.equities.schemas import (
    CorporateActionCoverage,
    CorporateActionCoverageStatus,
    CorporateActionSource,
    EquityCashDistribution,
    EquitySplitAction,
)
from alpha_agent.equities.universe import EQUITY_UNIVERSE

__all__ = [
    "COVERAGE",
    "KNOWN_DISTRIBUTIONS",
    "KNOWN_SPLITS",
    "coverage",
    "known_distributions",
    "known_splits",
]


def _edgar(name: str, url: str, retrieved_at: datetime = datetime(2026, 9, 25, tzinfo=UTC)) -> CorporateActionSource:
    return CorporateActionSource(name=name, url=url, retrieved_at=retrieved_at)


#: Every split below was retrieved by opening the real SEC EDGAR document and
#: quote-matching the actual split language, not from search-result snippets
#: or public-knowledge recall alone (2026-09-25 sourcing pass).
KNOWN_SPLITS: tuple[EquitySplitAction, ...] = (
    EquitySplitAction(
        raw_symbol="AAPL", effective_date=date(2020, 8, 31), ratio=4.0,
        source=_edgar(
            "SEC EDGAR Form 8-K Ex-99.1, Apple Inc. (CIK 0000320193), Q3 FY2020 results, "
            "filed 2020-07-30: 'The Board of Directors has also approved a four-for-one stock split.'",
            "https://www.sec.gov/Archives/edgar/data/320193/000032019320000060/a8-kexhibit991q3202062.htm",
        ),
    ),
    EquitySplitAction(
        raw_symbol="TSLA", effective_date=date(2020, 8, 31), ratio=5.0,
        source=_edgar(
            "SEC EDGAR Form 8-K Ex-99.1, Tesla Inc. (CIK 0001318605), filed 2020-08-11 "
            "(accession 0001564590-20-039353): 'the Board of Directors has approved and "
            "declared a five-for-one split of Tesla's common stock.'",
            "https://www.sec.gov/Archives/edgar/data/1318605/000156459020039353/tsla-ex991_6.htm",
        ),
    ),
    EquitySplitAction(
        raw_symbol="TSLA", effective_date=date(2022, 8, 25), ratio=3.0,
        source=_edgar(
            "SEC EDGAR Form 8-K Ex-99.1, Tesla Inc. (CIK 0001318605), filed 2022-08-05 "
            "(accession 0001564590-22-028207): 'approved and declared a three-for-one split'; "
            "record date 2022-08-17, distribution after close 2022-08-24 (effective 2022-08-25).",
            "https://www.sec.gov/Archives/edgar/data/1318605/000156459022028207/tsla-ex991_6.htm",
        ),
    ),
    EquitySplitAction(
        raw_symbol="NVDA", effective_date=date(2021, 7, 20), ratio=4.0,
        source=_edgar(
            "SEC EDGAR Form 8-K, NVIDIA Corp (CIK 0001045810), filed 2021-05-21 "
            "(accession 0001045810-21-000056): board 'declared a four-for-one split' "
            "conditioned on stockholder approval (obtained at the 2021-06-03 annual meeting); "
            "effective/trading-adjusted date 2021-07-20 is NVIDIA's own public record.",
            "https://www.sec.gov/Archives/edgar/data/1045810/000104581021000056/nvda-20210521.htm",
        ),
    ),
    EquitySplitAction(
        raw_symbol="NVDA", effective_date=date(2024, 6, 10), ratio=10.0,
        source=_edgar(
            "SEC EDGAR Form 8-K, NVIDIA Corp (CIK 0001045810), filed 2024-05-22 "
            "(accession 0001045810-24-000113): record holders as of 2024-06-06 receive 9 "
            "additional shares per share held; 'Trading is expected to commence on a "
            "split-adjusted basis...Monday, June 10, 2024.'",
            "https://www.sec.gov/Archives/edgar/data/1045810/000104581024000113/nvda-20240522.htm",
        ),
    ),
    EquitySplitAction(
        raw_symbol="GOOGL", effective_date=date(2022, 7, 18), ratio=20.0,
        source=_edgar(
            "SEC EDGAR Form 8-K Ex-99.1, Alphabet Inc. (CIK 0001652044), filed 2022-02-01 "
            "(accession 0001652044-22-000015): 'approved and declared a 20-for-one stock "
            "split...in the form of a one-time special stock dividend.'",
            "https://www.sec.gov/Archives/edgar/data/1652044/000165204422000015/googexhibit991q42021.htm",
        ),
    ),
    EquitySplitAction(
        raw_symbol="AMZN", effective_date=date(2022, 6, 6), ratio=20.0,
        source=_edgar(
            "SEC EDGAR Form 8-K, Amazon.com Inc. (CIK 0001018724), filed 2022-03-09 "
            "(accession 0001018724-22-000009): 'On March 9, 2022, the Board of Directors... "
            "approved a 20-for-1 split'; record date 2022-05-27, split-adjusted trading "
            "began 2022-06-06.",
            "https://www.sec.gov/Archives/edgar/data/1018724/000101872422000009/amzn-20220309.htm",
        ),
    ),
)

#: Not yet sourced -- see module docstring. Deliberately empty.
KNOWN_DISTRIBUTIONS: tuple[EquityCashDistribution, ...] = ()

_SOURCED_TICKERS = frozenset(s.raw_symbol for s in KNOWN_SPLITS)

_SCREEN_NOTE = (
    "Not yet screened for this universe (no day-over-day close-ratio scan has been run here, "
    "unlike the Phase 6 ETF pilot's audit) and not yet individually sourced from a filing. "
    "NOT_YET_INVESTIGATED, not 'no split occurred' -- several tickers in this universe "
    "(e.g. WMT split 3-for-1 in 2024) are publicly known to have split within the window but "
    "have not yet been cited here."
)
_SOURCED_SPLIT_NOTE = (
    "Sourced from a real SEC EDGAR Form 8-K filing, opened and quote-verified against the "
    "live document (2026-09-25 sourcing pass) -- not inferred from a price series and not "
    "recalled from general knowledge without a citation."
)
_DISTRIBUTIONS_NOTE = (
    "No free/self-serve source with 2018-2024 depth has been connected for this universe "
    "(same finding as the Phase 6 ETF pilot's feasibility investigation: Databento carries no "
    "dividend/split schema). Not populated pending an explicit data-source decision."
)

COVERAGE: dict[str, CorporateActionCoverage] = {
    ticker: CorporateActionCoverage(
        raw_symbol=ticker,
        splits_status=(
            CorporateActionCoverageStatus.SOURCED_FROM_OFFICIAL_RECORD
            if ticker in _SOURCED_TICKERS
            else CorporateActionCoverageStatus.NOT_YET_INVESTIGATED
        ),
        splits_note=_SOURCED_SPLIT_NOTE if ticker in _SOURCED_TICKERS else _SCREEN_NOTE,
        distributions_status=CorporateActionCoverageStatus.NOT_YET_INVESTIGATED,
        distributions_note=_DISTRIBUTIONS_NOTE,
    )
    for ticker in EQUITY_UNIVERSE
}


def known_splits(raw_symbol: str) -> tuple[EquitySplitAction, ...]:
    return tuple(s for s in KNOWN_SPLITS if s.raw_symbol == raw_symbol)


def known_distributions(raw_symbol: str) -> tuple[EquityCashDistribution, ...]:
    return tuple(d for d in KNOWN_DISTRIBUTIONS if d.raw_symbol == raw_symbol)


def coverage(raw_symbol: str) -> CorporateActionCoverage:
    if raw_symbol not in COVERAGE:
        raise KeyError(
            f"{raw_symbol!r} is not in the Phase 9.1 equities universe; no coverage record exists"
        )
    return COVERAGE[raw_symbol]
