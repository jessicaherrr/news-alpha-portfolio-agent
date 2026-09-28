"""The Phase 9.1 equities data-source loader (mirrors
``alpha_agent.etf.data_source``) plus the equity instrument-economics
adapter.

No Equity market data has been acquired yet (goal 5 of the Phase 9.1 prompt:
"Do not make a paid or materially costly Equity data acquisition
automatically"). Every bar-loading function below therefore fails loudly with
``FileNotFoundError`` until a real, user-approved, cost-capped acquisition
happens -- there is no synthetic fallback and no silent empty frame. They are
written now, in the exact shape the ETF pilot proved out, so acquisition (a
separate MONEY/NETWORK decision) is the ONLY remaining step before real
research can start.

Instrument economics: identical adapter shape to
``alpha_agent.etf.data_source.etf_contract_spec`` -- a single-share security
represented to the shared C++ event/order/fill/portfolio machinery as a
:class:`~alpha_agent.schemas.market_data.ContractSpecModel` with a synthetic
far-future ``expiration_ns``, ``multiplier=1.0`` (1 share = $1 P&L per $1
move -- definitionally true, never derived from a Databento field), and
``tick_size=0.01`` (standard US equity minimum increment above $1, true for
every ticker in this universe). No roll logic is ever invoked.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_agent.data import decode as dec
from alpha_agent.data.raw_store import raw_artifact_dir
from alpha_agent.equities.schemas import EquityDataSourceProvenance, EquityDataSourceRole
from alpha_agent.equities.universe import (
    DATASET_WINDOWS,
    EQUITY_UNIVERSE,
    PRIMARY_LISTING_EXCHANGE,
    equity_dsl_root_symbol,
    primary_listing_dataset,
)
from alpha_agent.schemas.market_data import ContractSpecModel

RAW_ROOT = "data/raw"

#: US equity standard minimum price increment for any instrument priced
#: above $1 -- true for every ticker in this universe (none has ever traded
#: below $1 in 2018-2024).
EQUITY_TICK_SIZE = 0.01
#: 1 share = $1 P&L per $1 of price movement -- definitionally true for a
#: single-share security, never guessed or derived from a Databento field.
EQUITY_MULTIPLIER = 1.0
#: A synthetic far-future expiration so ContractSpecModel's structural
#: validity check (expiration_ns > activation_ns) is satisfied for an
#: instrument that, in reality, never expires. Byte-identical to the ETF
#: pilot's own constant.
_SYNTHETIC_EXPIRATION_NS = 4_102_444_800_000_000_000  # 2100-01-01T00:00:00Z


def _dataset_window(dataset: str) -> tuple[str, str]:
    for ds, start, end in DATASET_WINDOWS:
        if ds == dataset:
            return start, end
    raise KeyError(f"{dataset!r} is not one of the Phase 9.1 candidate datasets")


def _artifact_path(dataset: str, schema: str, raw_symbol: str) -> Path:
    start, end = _dataset_window(dataset)
    if raw_symbol not in EQUITY_UNIVERSE:
        raise KeyError(f"{raw_symbol!r} is not in the Phase 9.1 equities universe")
    d = raw_artifact_dir(
        vendor="databento", dataset=dataset, schema=schema, stype_in="raw_symbol",
        stype_out="instrument_id", symbols=list(EQUITY_UNIVERSE), start=start, end=end,
        root_dir=RAW_ROOT,
    )
    for ext in ("dbn.zst", "dbn"):
        p = d / f"{schema}.{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"no acquired raw artifact for dataset={dataset!r} schema={schema!r} -- no Phase "
        "9.1 Equities acquisition has happened yet (a separate, user-approved MONEY/NETWORK "
        "decision); run the capability/cost probe first "
        "(scripts/phase9_1_equity_capability_probe.py)"
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


def load_primary_listing_bars(raw_symbol: str) -> tuple[pd.DataFrame, EquityDataSourceProvenance]:
    """`raw_symbol`'s own real primary-listing venue, explicitly labelled as
    such -- never "consolidated." Raises ``FileNotFoundError`` until Equity
    data is actually acquired."""
    dataset = primary_listing_dataset(raw_symbol)
    frame = _load_ohlcv_frame(dataset, raw_symbol)
    provenance = EquityDataSourceProvenance(
        dataset=dataset,
        role=EquityDataSourceRole.PRIMARY_LISTING_VENUE,
        listing_exchange=PRIMARY_LISTING_EXCHANGE[raw_symbol],
        volume_is_whole_market=False,
        measured_mean_close_divergence_bps=None,
        measured_mean_volume_capture_fraction=None,
    )
    return frame, provenance


def load_consolidated_eod_summary_bars(raw_symbol: str) -> tuple[pd.DataFrame, EquityDataSourceProvenance]:
    """EQUS.SUMMARY -- the same consolidated-EOD dataset the ETF pilot
    confirmed to be a real pre-aggregated bar per instrument/day. Raises
    ``FileNotFoundError`` until Equity data is actually acquired."""
    frame = _load_ohlcv_frame("EQUS.SUMMARY", raw_symbol)
    provenance = EquityDataSourceProvenance(
        dataset="EQUS.SUMMARY",
        role=EquityDataSourceRole.CONSOLIDATED_EOD_SUMMARY,
        listing_exchange=None,
        volume_is_whole_market=True,
        measured_mean_close_divergence_bps=None,
        measured_mean_volume_capture_fraction=None,
    )
    return frame, provenance


def equity_contract_spec(instrument_id: int, raw_symbol: str, *, activation_ns: int) -> ContractSpecModel:
    """The Phase 9.1 instrument-economics adapter: represents `raw_symbol` to
    the shared C++ event/order/fill/portfolio machinery as a single,
    never-expiring synthetic contract. Reuses ContractSpecModel unchanged --
    no Futures/ETF economics are rewritten, no roll logic is ever engaged.
    `root_symbol` is the synthetic DSL virtual-root label (see
    `equity_dsl_root_symbol`), NOT the ticker."""
    return ContractSpecModel(
        instrument_id=instrument_id,
        raw_symbol=raw_symbol,
        root_symbol=equity_dsl_root_symbol(raw_symbol),
        exchange=PRIMARY_LISTING_EXCHANGE[raw_symbol],
        tick_size=EQUITY_TICK_SIZE,
        multiplier=EQUITY_MULTIPLIER,
        activation_ns=activation_ns,
        expiration_ns=_SYNTHETIC_EXPIRATION_NS,
    )
