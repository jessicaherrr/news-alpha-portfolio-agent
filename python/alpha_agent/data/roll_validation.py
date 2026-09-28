"""Phase 04.5 -- real NQ roll + back-adjustment validation.

Stage A: NQ.v.0 continuous + NQ.FUT definitions over a completed historical
         quarterly-roll window -> detect the real instrument_id transition.
Stage B: both real raw contracts over a small overlap window around the
         transition -> the aligned same-timestamp roll basis.

No request runs here. Everything is cost-estimated and capped by the caller
(scripts/databento_roll_validate.py). After the two downloads, all further
validation is local replay.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from alpha_agent.data.databento_source import HistoricalRequest
from alpha_agent.data.definitions import DefinitionRegistry

DATASET = "GLBX.MDP3"
CONTINUOUS_SYMBOL = "NQ.v.0"
# June 2026 NQ quarterly roll -- completed before 2026-09-05. NQ.v.0 is
# volume-ranked; the real instrument_id transition (not the calendar) is the roll.
STAGE_A_START = "2026-06-08"
STAGE_A_END = "2026-06-19"


def stage_a_requests() -> tuple[HistoricalRequest, HistoricalRequest]:
    bars = HistoricalRequest(
        dataset=DATASET, symbols=(CONTINUOUS_SYMBOL,), schema="ohlcv-1m",
        start=STAGE_A_START, end=STAGE_A_END, stype_in="continuous", stype_out="instrument_id",
    )
    return bars, bars.definition_request()


@dataclass(frozen=True)
class InstrumentSpan:
    instrument_id: int
    raw_symbol: str
    first_ts_ns: int
    last_ts_ns: int
    n_bars: int


@dataclass(frozen=True)
class RollDetection:
    continuous_symbol: str
    from_instrument_id: int
    to_instrument_id: int
    from_raw_symbol: str
    to_raw_symbol: str
    transition_ts_ns: int          # first bar carrying the new instrument_id
    spans: tuple[InstrumentSpan, ...]

    def to_dict(self) -> dict:
        return {
            "continuous_symbol": self.continuous_symbol,
            "from_instrument_id": self.from_instrument_id,
            "to_instrument_id": self.to_instrument_id,
            "from_raw_symbol": self.from_raw_symbol,
            "to_raw_symbol": self.to_raw_symbol,
            "transition_ts_ns": self.transition_ts_ns,
            "spans": [vars(s) for s in self.spans],
        }


def detect_transition(
    continuous_ohlcv: pd.DataFrame, registry: DefinitionRegistry
) -> RollDetection | None:
    """From a continuous ohlcv vendor frame (``ts_event, instrument_id, ...``)
    return the single real instrument_id transition, or ``None`` if the feed
    never changes contract in this window."""
    b = continuous_ohlcv.rename(columns={"ts_event": "ts_event_ns"}).copy()
    b["ts_event_ns"] = b["ts_event_ns"].astype("int64")
    b["instrument_id"] = b["instrument_id"].astype("int64")
    b = b.sort_values("ts_event_ns", kind="stable").reset_index(drop=True)

    def _raw(iid: int) -> str:
        spec = registry.get(iid)
        return spec.raw_symbol if spec else f"<unknown:{iid}>"

    spans = tuple(
        InstrumentSpan(
            instrument_id=int(iid), raw_symbol=_raw(int(iid)),
            first_ts_ns=int(g["ts_event_ns"].min()), last_ts_ns=int(g["ts_event_ns"].max()),
            n_bars=len(g),
        )
        for iid, g in b.groupby("instrument_id", sort=True)
    )

    change = b["instrument_id"].ne(b["instrument_id"].shift())
    idx = [i for i in b.index[change] if i > 0]
    if not idx:
        return None
    i = idx[0]
    from_id = int(b.at[i - 1, "instrument_id"])
    to_id = int(b.at[i, "instrument_id"])
    return RollDetection(
        continuous_symbol=CONTINUOUS_SYMBOL,
        from_instrument_id=from_id, to_instrument_id=to_id,
        from_raw_symbol=_raw(from_id), to_raw_symbol=_raw(to_id),
        transition_ts_ns=int(b.at[i, "ts_event_ns"]), spans=spans,
    )


def stage_b_request(
    detection: RollDetection, *, pad_days_before: int = 1, pad_days_after: int = 2
) -> HistoricalRequest:
    """Both real raw contracts over a tight window around the transition.
    ``stype_in=raw_symbol`` (both are real contracts)."""
    t = pd.Timestamp(detection.transition_ts_ns, unit="ns", tz="UTC")
    start = (t.normalize() - pd.Timedelta(days=pad_days_before)).strftime("%Y-%m-%d")
    end = (t.normalize() + pd.Timedelta(days=pad_days_after)).strftime("%Y-%m-%d")
    return HistoricalRequest(
        dataset=DATASET,
        symbols=(detection.from_raw_symbol, detection.to_raw_symbol),
        schema="ohlcv-1m", start=start, end=end,
        stype_in="raw_symbol", stype_out="instrument_id",
    )
