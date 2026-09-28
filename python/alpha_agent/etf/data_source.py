"""Load the Phase 6 pilot's acquired ETF bars with mandatory, honest
provenance, and the ETF instrument-economics adapter.

Every bar-loading function here returns an :class:`EtfDataSourceProvenance`
alongside the frame -- there is no path that returns bars without it, so a
caller can never accidentally treat a primary-listing-venue series as
"the market close" or "the market volume."

Instrument economics: an ETF is represented to the shared C++ event/order/
fill/portfolio machinery as a :class:`~alpha_agent.schemas.market_data.
ContractSpecModel` with a synthetic far-future ``expiration_ns`` (ETFs do
not expire, but ``ContractSpec.is_valid()`` requires ``expiration_ns >
activation_ns``), ``multiplier=1.0`` (1 share = $1 P&L per $1 move --
unlike a futures multiplier, this is not derived from a Databento
definition field; it is definitionally true for any single-share
security), and ``tick_size=0.01`` (the standard US equity minimum price
increment for any instrument priced above $1, true for every ticker in
this universe across the whole pilot window). No roll logic is ever
invoked -- a single synthetic contract per ticker, forever. This is the
"small correct instrument-economics adapter" Phase 6 instruction 4 asked
for: it reuses the existing Fill/Order/Portfolio machinery unchanged and
adds no futures-specific hack.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_agent.data import decode as dec
from alpha_agent.data.raw_store import raw_artifact_dir
from alpha_agent.etf.schemas import EtfDataSourceProvenance, EtfDataSourceRole
from alpha_agent.etf.universe import (
    DATASET_WINDOWS,
    PRIMARY_LISTING_EXCHANGE,
    etf_dsl_root_symbol,
    primary_listing_dataset,
)
from alpha_agent.schemas.market_data import ContractSpecModel

RAW_ROOT = "data/raw"

#: Real, measured Phase 6 divergence evidence (primary-listing-venue bar vs.
#: EQUS.SUMMARY, their real overlap 2024-07-01..2024-12-30, 1778 matched
#: rows) -- carried here so a consumer never has to re-derive it. See
#: docs/AGENTIC_ALPHA_ROADMAP_STATE.md's Phase 6 row for the full method.
MEASURED_MEAN_CLOSE_DIVERGENCE_BPS = 12.1
MEASURED_MEAN_VOLUME_CAPTURE_FRACTION = 0.27

#: US equity standard minimum price increment for any instrument priced
#: above $1 -- true for every ticker in this universe across the whole
#: pilot window (none ever traded below $1).
ETF_TICK_SIZE = 0.01
#: 1 share = $1 P&L per $1 of price movement -- definitionally true for a
#: single-share security, never guessed or derived from a Databento field.
ETF_MULTIPLIER = 1.0
#: A synthetic far-future expiration so ContractSpecModel's structural
#: validity check (expiration_ns > activation_ns) is satisfied for an
#: instrument that, in reality, never expires. No roll logic is ever
#: triggered because this instant is never reached during the pilot window.
_SYNTHETIC_EXPIRATION_NS = 4_102_444_800_000_000_000  # 2100-01-01T00:00:00Z


def _dataset_window(dataset: str) -> tuple[str, str]:
    for ds, start, end in DATASET_WINDOWS:
        if ds == dataset:
            return start, end
    raise KeyError(f"{dataset!r} is not one of the Phase 6 acquired datasets")


def _artifact_path(dataset: str, schema: str, raw_symbol: str) -> Path:
    from alpha_agent.etf.universe import PILOT_UNIVERSE

    start, end = _dataset_window(dataset)
    if raw_symbol not in PILOT_UNIVERSE:
        raise KeyError(f"{raw_symbol!r} is not in the Phase 6 pilot universe")
    d = raw_artifact_dir(
        vendor="databento", dataset=dataset, schema=schema, stype_in="raw_symbol",
        stype_out="instrument_id", symbols=list(PILOT_UNIVERSE), start=start, end=end,
        root_dir=RAW_ROOT,
    )
    for ext in ("dbn.zst", "dbn"):
        p = d / f"{schema}.{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"no acquired raw artifact for dataset={dataset!r} schema={schema!r} -- "
        "run scripts/phase6_etf_acquire.py --acquire first"
    )


def _load_ohlcv_frame(dataset: str, raw_symbol: str) -> pd.DataFrame:
    path = _artifact_path(dataset, "ohlcv-1d", raw_symbol)
    df = dec.load_dbn(path).to_df(price_type="fixed", pretty_ts=False, map_symbols=True).reset_index()
    df = df[df["symbol"] == raw_symbol].copy()
    df["day"] = pd.to_datetime(df["ts_event"], unit="ns", utc=True).dt.date
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype("float64") / 1e9
    return df.sort_values("ts_event").reset_index(drop=True)


def primary_listing_data_acquired(raw_symbol: str) -> bool:
    """Whether `raw_symbol`'s primary-listing daily bars exist in the raw
    store -- a cheap path-existence check (no decode, no network), for
    capability display only. Same resolution as `load_primary_listing_bars`."""
    try:
        _artifact_path(primary_listing_dataset(raw_symbol), "ohlcv-1d", raw_symbol)
    except FileNotFoundError:
        return False
    return True


def load_primary_listing_bars(raw_symbol: str) -> tuple[pd.DataFrame, EtfDataSourceProvenance]:
    """The one series Phase 6 instruction 1 designates for the long
    2018-2024 research window: `raw_symbol`'s own real primary-listing
    venue, explicitly labelled as such -- never "consolidated"."""
    dataset = primary_listing_dataset(raw_symbol)
    frame = _load_ohlcv_frame(dataset, raw_symbol)
    provenance = EtfDataSourceProvenance(
        dataset=dataset,
        role=EtfDataSourceRole.PRIMARY_LISTING_VENUE,
        listing_exchange=PRIMARY_LISTING_EXCHANGE[raw_symbol],
        volume_is_whole_market=False,
        measured_mean_close_divergence_bps=MEASURED_MEAN_CLOSE_DIVERGENCE_BPS,
        measured_mean_volume_capture_fraction=MEASURED_MEAN_VOLUME_CAPTURE_FRACTION,
    )
    return frame, provenance


def load_consolidated_eod_summary_bars(raw_symbol: str) -> tuple[pd.DataFrame, EtfDataSourceProvenance]:
    """EQUS.SUMMARY -- the one dataset empirically confirmed (Phase 6 audit:
    single publisher_id, 0% duplicate (instrument, ts) rows) to be a real
    pre-aggregated consolidated bar per instrument/day. Only available
    2024-07-01 onward pre-holdout."""
    frame = _load_ohlcv_frame("EQUS.SUMMARY", raw_symbol)
    provenance = EtfDataSourceProvenance(
        dataset="EQUS.SUMMARY",
        role=EtfDataSourceRole.CONSOLIDATED_EOD_SUMMARY,
        listing_exchange=None,
        volume_is_whole_market=True,
        measured_mean_close_divergence_bps=None,
        measured_mean_volume_capture_fraction=None,
    )
    return frame, provenance


def load_other_single_venue_bars(dataset: str, raw_symbol: str) -> tuple[pd.DataFrame, EtfDataSourceProvenance]:
    """A single-venue feed that is NOT `raw_symbol`'s own primary listing
    (e.g. XNAS.ITCH for an ARCX-listed ticker). Used only by the
    data-source robustness comparator, never as a research/execution
    source of truth on its own."""
    if dataset == primary_listing_dataset(raw_symbol):
        raise ValueError(
            f"{dataset!r} IS {raw_symbol!r}'s primary listing venue -- use "
            "load_primary_listing_bars instead"
        )
    frame = _load_ohlcv_frame(dataset, raw_symbol)
    provenance = EtfDataSourceProvenance(
        dataset=dataset,
        role=EtfDataSourceRole.OTHER_SINGLE_VENUE,
        listing_exchange=PRIMARY_LISTING_EXCHANGE[raw_symbol],
        volume_is_whole_market=False,
        measured_mean_close_divergence_bps=None,
        measured_mean_volume_capture_fraction=None,
    )
    return frame, provenance


def etf_contract_spec(instrument_id: int, raw_symbol: str, *, activation_ns: int) -> ContractSpecModel:
    """The Phase 6 instrument-economics adapter: represents `raw_symbol` to
    the shared C++ event/order/fill/portfolio machinery as a single,
    never-expiring synthetic contract. Reuses ContractSpecModel unchanged --
    no Futures economics are rewritten, no roll logic is ever engaged.
    `root_symbol` is the synthetic DSL virtual-root label (see
    `etf_dsl_root_symbol`), NOT the ticker -- matches the target schedule's
    own root_symbol so the reference CLI's ActiveContractResolver can
    resolve one to the other."""
    return ContractSpecModel(
        instrument_id=instrument_id,
        raw_symbol=raw_symbol,
        root_symbol=etf_dsl_root_symbol(raw_symbol),
        exchange=PRIMARY_LISTING_EXCHANGE[raw_symbol],
        tick_size=ETF_TICK_SIZE,
        multiplier=ETF_MULTIPLIER,
        activation_ns=activation_ns,
        expiration_ns=_SYNTHETIC_EXPIRATION_NS,
    )
