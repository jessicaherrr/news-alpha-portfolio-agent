"""Dataset identity + chronological bar-window helpers for validation.

The validation framework never mutates market data. It only *slices* the frozen
Phase 02.5 ``bars`` frame into chronological windows (train / test folds /
holdout) and records where the data came from so a run is reproducible
(CLAUDE backtest rule 5, section 21).
"""
from __future__ import annotations

import hashlib

import pandas as pd
from pydantic import BaseModel, Field

from alpha_agent.validation.fingerprint import fingerprint


class DatasetIdentity(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    root_symbol: str
    price_domain: str
    adjustment_mode: str | None = None
    source_fingerprint: str                    # Phase 09 SourceSeries.fingerprint(), or a fixture hash
    bars_content_hash: str                     # sha256 over the exact bar rows used
    n_bars: int = Field(ge=0)
    first_ts_ns: int | None = None
    last_ts_ns: int | None = None
    contracts_content_hash: str | None = None
    # Phase 13.1: the canonical futures trading-day convention (calendar name +
    # version + timezone + local day boundary). Two day conventions must never
    # share one ValidationSpec identity (section 4).
    trading_day_convention: dict | None = None
    note: str = ""

    def identity(self) -> str:
        return fingerprint(
            "valdataset2",
            {
                "root_symbol": self.root_symbol,
                "price_domain": self.price_domain,
                "adjustment_mode": self.adjustment_mode,
                "source_fingerprint": self.source_fingerprint,
                "bars_content_hash": self.bars_content_hash,
                "contracts_content_hash": self.contracts_content_hash,
                "n_bars": self.n_bars,
                "trading_day_convention": self.trading_day_convention,
            },
        )


def frame_content_hash(df: pd.DataFrame) -> str:
    """Deterministic sha256 over a DataFrame's values + columns + dtypes."""
    h = hashlib.sha256()
    h.update(",".join(map(str, df.columns)).encode())
    h.update(",".join(map(str, df.dtypes)).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()


def dataset_identity_from_bars(
    bars: pd.DataFrame,
    *,
    root_symbol: str,
    price_domain: str,
    source_fingerprint: str,
    adjustment_mode: str | None = None,
    contracts: pd.DataFrame | None = None,
    trading_day_convention: dict | None = None,
    note: str = "",
) -> DatasetIdentity:
    ts = bars["ts_event_ns"] if "ts_event_ns" in bars.columns else None
    if trading_day_convention is None:
        from alpha_agent.data.calendars import default_calendar

        try:
            trading_day_convention = default_calendar().trading_day_convention(root_symbol)
        except Exception:  # noqa: BLE001 -- a root with no calendar records None, loudly caught later
            trading_day_convention = None
    return DatasetIdentity(
        root_symbol=root_symbol,
        price_domain=price_domain,
        adjustment_mode=adjustment_mode,
        source_fingerprint=source_fingerprint,
        bars_content_hash=frame_content_hash(bars),
        n_bars=len(bars),
        first_ts_ns=int(ts.min()) if ts is not None and len(ts) else None,
        last_ts_ns=int(ts.max()) if ts is not None and len(ts) else None,
        contracts_content_hash=frame_content_hash(contracts) if contracts is not None else None,
        trading_day_convention=trading_day_convention,
        note=note,
    )


def bars_in_window(bars: pd.DataFrame, start_ts_ns: int, end_ts_ns: int) -> pd.DataFrame:
    """Rows with ``start_ts_ns <= ts_event_ns < end_ts_ns`` (end exclusive),
    chronologically ordered. A pure slice -- never forward-fills, never drops
    rows inside the window."""
    m = (bars["ts_event_ns"] >= start_ts_ns) & (bars["ts_event_ns"] < end_ts_ns)
    return bars.loc[m].sort_values("ts_event_ns").reset_index(drop=True)


def sorted_unique_bar_ts(bars: pd.DataFrame) -> list[int]:
    return sorted(int(t) for t in pd.unique(bars["ts_event_ns"]))
