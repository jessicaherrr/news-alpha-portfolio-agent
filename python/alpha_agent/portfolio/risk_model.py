"""News Alpha Phase G -- the RISK MODEL a plan is sized with: statistics only.

    InstrumentMarketSnapshot   one instrument's real daily bars over the
                               2018-2022 discovery window: research close
                               (roll/split adjusted), raw close, volume, the
                               contract actually traded, contract economics
    estimate_risk_model        -> RiskModel: annualized volatility, a joint-panel
                               correlation matrix, median daily traded notional,
                               the raw reference price and decision timestamp at
                               the as-of date

This module decides nothing: it never selects a signal, sizes a position or
applies a limit (the C++ allocator does). It estimates, and it refuses
loudly -- `InsufficientRiskData` -- rather than filling a gap.

RETURNS. Futures research closes are ADDITIVELY forward-adjusted, so a daily
return is ``d_adj / (raw_t - d_adj)``: the price change over the previous
close expressed in today's contract (exact across a roll, where the adjusted
change carries no roll gap). ETF research closes are backward split-adjusted
(multiplicative), so a return is the plain ratio. Price returns only: no
distribution record is sourced (the ETF measurement already says so).

JOINT PANEL. Volatility and correlation come from ONE panel: the trading days
every instrument in the plan traded, over the trailing ``risk_lookback_days``
up to the as-of date. A return spans the same calendar interval for every
instrument (a day one venue traded and another did not is absorbed, never
forward-filled), and a correlation matrix from one panel is positive
semi-definite by construction.

LIQUIDITY. Median over the same trailing days of ``volume x raw close x
multiplier``. Futures volume is the front contract's; ETF volume is the
primary listing venue's only -- measured at ~27% of consolidated volume
(`etf.data_source`), so ETF liquidity is a LOWER bound.

WINDOWS. A snapshot and a risk model carry the window they were built on
(`SnapshotWindow`). DISCOVERY (2018-2022, the default and the only window
portfolio CONSTRUCTION uses) keeps the as-of date before the 2023 validation
window. EVALUATION (2018-2024) exists for one caller: Phase H's causal
walk-forward of an already-frozen portfolio strategy, which re-sizes the book
at each rebalance date from data up to that date only (the estimator slices
at the as-of date; the window only bounds which days may exist). The 2025
locked holdout is in neither window, and every timestamp passes the holdout
guard. No vendor call, no download.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, model_validator

from alpha_agent.crypto.provenance import DataProvenanceRole
from alpha_agent.data.real_market_dataset import (
    RESEARCH_WINDOW,
    VALIDATION_WINDOW,
    assert_no_holdout_ts,
)
from alpha_agent.news_alpha.mandate import MandateDomain
from alpha_agent.portfolio.classification import RiskAssetClass, classify_instrument

__all__ = [
    "DISCOVERY_END",
    "DISCOVERY_START",
    "InstrumentMarketSnapshot",
    "InstrumentRisk",
    "InsufficientRiskData",
    "MarketSnapshotStore",
    "RiskModel",
    "SnapshotProvider",
    "SnapshotWindow",
    "common_as_of",
    "daily_returns",
    "estimate_risk_model",
    "instrument_key",
    "load_market_snapshot",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "data" / "news_alpha" / "market_snapshots"
SNAPSHOT_SCHEMA = "instrument-market-snapshot/1"
RISK_MODEL_SCHEMA = "risk-model/1"
RISK_ESTIMATOR = "trailing-joint-panel/1"
DISCOVERY_START = date.fromisoformat(RESEARCH_WINDOW[0])
#: Exclusive -- the first day of the 2023-2024 validation window.
DISCOVERY_END = date.fromisoformat(RESEARCH_WINDOW[1])
#: Exclusive -- the first day of the 2025 locked holdout.
EVALUATION_END = date.fromisoformat(VALIDATION_WINDOW[1])


class SnapshotWindow(str, Enum):
    """Which days a snapshot / risk model may hold. DISCOVERY is the default
    and the construction window; EVALUATION is only for replaying a frozen
    portfolio strategy causally through the validation window (Phase H)."""

    DISCOVERY = "DISCOVERY"
    EVALUATION = "EVALUATION"

    @property
    def end(self) -> date:
        """Exclusive end of the window."""
        return DISCOVERY_END if self is SnapshotWindow.DISCOVERY else EVALUATION_END


def fingerprint_payload(model: BaseModel) -> dict:
    """A DISCOVERY-window object fingerprints exactly as it did before windows
    existed (Phase G plans keep their fingerprints)."""
    payload = model.model_dump(mode="json")
    if payload.get("window") == SnapshotWindow.DISCOVERY.value:
        del payload["window"]
    return payload


def instrument_key(domain: MandateDomain, symbol: str) -> str:
    return f"{domain.value}:{symbol}"


class InsufficientRiskData(ValueError):
    """A risk statistic cannot be estimated honestly from the data present."""


class InstrumentMarketSnapshot(BaseModel):
    """One instrument's real daily bars over the discovery window -- enough
    to estimate risk and price a unit at any day inside it."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = SNAPSHOT_SCHEMA
    domain: MandateDomain
    symbol: str
    #: The root a target schedule names (futures root / the ETF's synthetic ``E<ticker>``).
    execution_root: str
    dataset: str
    unit: Literal["contract", "share"]
    #: USD per 1.0 price move per unit (ContractSpec semantics).
    multiplier: float
    tick_size: float
    adjustment: Literal["ADDITIVE", "MULTIPLICATIVE"]
    volume_scope: Literal["FRONT_CONTRACT", "PRIMARY_LISTING_ONLY"]
    data_role: DataProvenanceRole
    window: SnapshotWindow = SnapshotWindow.DISCOVERY
    days: tuple[str, ...]
    ts_event_ns: tuple[int, ...]
    research_close: tuple[float, ...]
    raw_close: tuple[float, ...]
    volume: tuple[float, ...]
    #: The contract actually traded that day (futures front) / the ticker.
    traded_symbol: tuple[str, ...]
    provenance: tuple[str, ...]

    @model_validator(mode="after")
    def _consistent(self) -> InstrumentMarketSnapshot:
        n = len(self.days)
        if not all(len(x) == n for x in (self.ts_event_ns, self.research_close, self.raw_close, self.volume,
                                         self.traded_symbol)):
            raise ValueError("one value per trading day in every column")
        if list(self.days) != sorted(set(self.days)):
            raise ValueError("trading days must be unique and ascending")
        if self.days and not (DISCOVERY_START <= date.fromisoformat(self.days[0])
                              and date.fromisoformat(self.days[-1]) < self.window.end):
            raise ValueError("a snapshot holds discovery-window days only" if self.window is SnapshotWindow.DISCOVERY
                             else f"an evaluation snapshot holds days before {self.window.end} only")
        if not self.multiplier > 0 or not self.tick_size > 0:
            raise ValueError("contract economics must be positive")
        return self

    @property
    def key(self) -> str:
        return instrument_key(self.domain, self.symbol)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "ts_event_ns": self.ts_event_ns, "research_close": self.research_close, "raw_close": self.raw_close,
            "volume": self.volume, "traded_symbol": self.traded_symbol,
        }, index=pd.Index(self.days, name="trading_day"))

    def fingerprint(self) -> str:
        payload = json.dumps(fingerprint_payload(self), sort_keys=True)
        return "mktsnap1:" + hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# real point-in-time loaders (offline, already-acquired bytes only)
# ---------------------------------------------------------------------------


def _window_slice(frame: pd.DataFrame, window: SnapshotWindow) -> pd.DataFrame:
    days = pd.to_datetime(frame["trading_day"]).dt.date
    out = frame[(days >= DISCOVERY_START) & (days < window.end)].reset_index(drop=True)
    assert_no_holdout_ts(out["ts_event_ns"])
    return out


def load_futures_snapshot(root: str, window: SnapshotWindow = SnapshotWindow.DISCOVERY) -> InstrumentMarketSnapshot:
    """~1 minute: rebuilds the root's layers offline from the raw store."""
    from alpha_agent.data.calendars import default_calendar
    from alpha_agent.data.real_market_dataset import reconstitute_root

    recon = reconstitute_root(root)
    calendar = default_calendar()
    raw = recon.continuous[["ts_event_ns", "active_instrument_id", "active_raw_symbol", "close", "volume"]]
    adj = recon.forward_adjusted[["ts_event_ns", "close"]].rename(columns={"close": "research_close"})
    minute = raw.merge(adj, on="ts_event_ns", how="inner", validate="one_to_one").sort_values(
        "ts_event_ns", kind="stable").reset_index(drop=True)
    if len(minute) != len(raw):
        raise InsufficientRiskData(f"{root}: adjusted and raw continuous series do not align bar for bar")
    trading_day, _session = calendar.classify_series(minute["ts_event_ns"], root)
    minute = minute.assign(trading_day=trading_day.astype(str))
    g = minute.groupby("trading_day", sort=True)
    daily = pd.DataFrame({
        "trading_day": g["trading_day"].last(), "ts_event_ns": g["ts_event_ns"].last().astype("int64"),
        "research_close": g["research_close"].last().astype("float64"),
        "raw_close": g["close"].last().astype("float64"), "volume": g["volume"].sum().astype("float64"),
        "traded_symbol": g["active_raw_symbol"].last().astype(str),
        "instrument_id": g["active_instrument_id"].last().astype("int64"),
    }).reset_index(drop=True)
    daily = _window_slice(daily, window)
    contracts = recon.contracts.set_index("instrument_id")
    used = contracts.loc[sorted(set(daily["instrument_id"]))]
    if used["multiplier"].nunique() != 1 or used["tick_size"].nunique() != 1:
        raise InsufficientRiskData(f"{root}: contract economics differ across the traded contracts")
    return InstrumentMarketSnapshot(
        domain=MandateDomain.FUTURES, symbol=root, execution_root=root, dataset="GLBX.MDP3", unit="contract",
        multiplier=float(used["multiplier"].iloc[0]), tick_size=float(used["tick_size"].iloc[0]),
        adjustment="ADDITIVE", volume_scope="FRONT_CONTRACT", data_role=DataProvenanceRole.REAL, window=window,
        days=tuple(daily["trading_day"]), ts_event_ns=tuple(int(x) for x in daily["ts_event_ns"]),
        research_close=tuple(float(x) for x in daily["research_close"]),
        raw_close=tuple(float(x) for x in daily["raw_close"]), volume=tuple(float(x) for x in daily["volume"]),
        traded_symbol=tuple(daily["traded_symbol"]),
        provenance=(
            (f"GLBX.MDP3 ohlcv-1m {root}.v.0 rebuilt offline from the raw store (data.real_market_dataset."
             "reconstitute_root); one bar per CME trading day, stamped at its last minute"),
            "research close: forward-adjusted (additive, causal); raw close: the front contract actually traded",
            "multiplier and tick size from the stored Databento definitions (docs/CONTRACT_ECONOMICS.md)",
        ),
    )


def load_etf_snapshot(symbol: str, window: SnapshotWindow = SnapshotWindow.DISCOVERY) -> InstrumentMarketSnapshot:
    from alpha_agent.etf.corporate_actions import known_splits
    from alpha_agent.etf.data_source import ETF_MULTIPLIER, ETF_TICK_SIZE, load_primary_listing_bars
    from alpha_agent.etf.universe import etf_dsl_root_symbol

    raw, source = load_primary_listing_bars(symbol)
    frame = pd.DataFrame({
        "trading_day": raw["day"].astype(str), "ts_event_ns": raw["ts_event"].astype("int64"),
        "raw_close": raw["close"].astype("float64"), "volume": raw["volume"].astype("float64"),
    }).sort_values("trading_day", kind="stable").reset_index(drop=True)
    research = frame["raw_close"].copy()
    notes = [f"{source.dataset} ohlcv-1d, primary listing only ({source.listing_exchange}); price return only"]
    for split in known_splits(symbol):
        before = pd.to_datetime(frame["trading_day"]).dt.date < split.effective_date
        research[before] = research[before] / split.ratio
        notes.append(f"research close split-adjusted backward for {split.ratio:g} on {split.effective_date} "
                     f"({split.source.name}); raw close unadjusted")
    frame = _window_slice(frame.assign(research_close=research), window)
    return InstrumentMarketSnapshot(
        domain=MandateDomain.ETF, symbol=symbol, execution_root=etf_dsl_root_symbol(symbol), dataset=source.dataset,
        unit="share", multiplier=ETF_MULTIPLIER, tick_size=ETF_TICK_SIZE, adjustment="MULTIPLICATIVE",
        volume_scope="PRIMARY_LISTING_ONLY", data_role=DataProvenanceRole.REAL, window=window,
        days=tuple(frame["trading_day"]), ts_event_ns=tuple(int(x) for x in frame["ts_event_ns"]),
        research_close=tuple(float(x) for x in frame["research_close"]),
        raw_close=tuple(float(x) for x in frame["raw_close"]), volume=tuple(float(x) for x in frame["volume"]),
        traded_symbol=tuple(symbol for _ in range(len(frame))),
        provenance=(*notes, "multiplier 1.0 (one share), tick 0.01 (etf.data_source)"),
    )


def load_market_snapshot(domain: MandateDomain, symbol: str,
                         window: SnapshotWindow = SnapshotWindow.DISCOVERY) -> InstrumentMarketSnapshot:
    if domain is MandateDomain.FUTURES:
        return load_futures_snapshot(symbol, window)
    if domain is MandateDomain.ETF:
        return load_etf_snapshot(symbol, window)
    raise InsufficientRiskData(f"no point-in-time daily bars for {domain.value} ({symbol})")


#: (domain, symbol) -> snapshot, or raise `InsufficientRiskData`.
SnapshotProvider = Callable[[MandateDomain, str], InstrumentMarketSnapshot]


class MarketSnapshotStore:
    """Operational cache (gitignored ``data/news_alpha/market_snapshots/``):
    a futures snapshot costs a minute to rebuild, so it is built once. A
    cached file of another schema, window, or of non-real data, is ignored.
    One store serves one window (DISCOVERY unless stated)."""

    def __init__(self, directory: Path | None = None, window: SnapshotWindow = SnapshotWindow.DISCOVERY) -> None:
        self.directory = directory or DEFAULT_SNAPSHOT_DIR
        self.window = window

    def _path(self, domain: MandateDomain, symbol: str) -> Path:
        return self.directory / f"{domain.value}__{symbol}__{DISCOVERY_START}__{self.window.end}.json"

    def load(self, domain: MandateDomain, symbol: str) -> InstrumentMarketSnapshot | None:
        path = self._path(domain, symbol)
        if not path.exists():
            return None
        try:
            snap = InstrumentMarketSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 -- a corrupt cache entry is rebuilt, never trusted
            return None
        if (snap.schema_version != SNAPSHOT_SCHEMA or snap.data_role is not DataProvenanceRole.REAL
                or snap.window is not self.window):
            return None
        return snap

    def save(self, snap: InstrumentMarketSnapshot) -> None:
        if snap.data_role is not DataProvenanceRole.REAL:
            raise ValueError("only real-data snapshots are cached")
        if snap.window is not self.window:
            raise ValueError(f"a {snap.window.value} snapshot does not belong in a {self.window.value} store")
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path(snap.domain, snap.symbol).write_text(snap.model_dump_json(), encoding="utf-8")

    def has(self, domain: MandateDomain, symbol: str) -> bool:
        return self.load(domain, symbol) is not None

    def provider(self, loader: SnapshotProvider | None = None) -> SnapshotProvider:
        window = self.window

        def default_loader(domain: MandateDomain, symbol: str) -> InstrumentMarketSnapshot:
            return load_market_snapshot(domain, symbol, window)

        load = loader or default_loader

        def get(domain: MandateDomain, symbol: str) -> InstrumentMarketSnapshot:
            snap = self.load(domain, symbol)
            if snap is None:
                snap = load(domain, symbol)
                self.save(snap)
            return snap
        return get


# ---------------------------------------------------------------------------
# estimation
# ---------------------------------------------------------------------------


class InstrumentRisk(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    key: str
    domain: MandateDomain
    symbol: str
    execution_root: str
    asset_class: RiskAssetClass
    sector: str
    unit: Literal["contract", "share"]
    multiplier: float
    tick_size: float
    #: The RAW close at the as-of date -- the reference price a unit is sized at.
    price: float
    traded_symbol: str
    decision_ts_ns: int
    annual_vol: float
    #: Median daily traded notional over the trailing window; ``None`` = not measured.
    adv_usd: float | None
    volume_scope: Literal["FRONT_CONTRACT", "PRIMARY_LISTING_ONLY"]
    snapshot_fingerprint: str


class RiskModel(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = RISK_MODEL_SCHEMA
    estimator: str = RISK_ESTIMATOR
    window: SnapshotWindow = SnapshotWindow.DISCOVERY
    as_of: date
    #: First and last day of the joint return panel.
    window_start: date
    window_end: date
    lookback_days: int
    n_observations: int
    annualization_days: int
    instruments: tuple[InstrumentRisk, ...]
    #: (a, b, rho) for every pair, a < b.
    correlations: tuple[tuple[str, str, float], ...]
    notes: tuple[str, ...]

    @model_validator(mode="after")
    def _guard(self) -> RiskModel:
        if self.as_of >= self.window.end:
            raise ValueError(f"a {self.window.value} risk model is estimated before {self.window.end} only")
        keys = [i.key for i in self.instruments]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("instruments are unique and ordered by key")
        return self

    def instrument(self, key: str) -> InstrumentRisk:
        return next(i for i in self.instruments if i.key == key)

    def correlation(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        lo, hi = sorted((a, b))
        return next(r for x, y, r in self.correlations if (x, y) == (lo, hi))

    def fingerprint(self) -> str:
        payload = json.dumps(fingerprint_payload(self), sort_keys=True)
        return "riskmodel1:" + hashlib.sha256(payload.encode()).hexdigest()


def common_as_of(snapshots: Sequence[InstrumentMarketSnapshot], *, on_or_before: date | None = None) -> date:
    """The last trading day every snapshot has a bar on (inside the snapshots' window)."""
    if not snapshots:
        raise InsufficientRiskData("no instruments to date")
    window = _one_window(snapshots)
    common = set(snapshots[0].days)
    for s in snapshots[1:]:
        common &= set(s.days)
    limit = min(on_or_before or window.end, window.end)
    usable = sorted(d for d in common if date.fromisoformat(d) <= limit and date.fromisoformat(d) < window.end)
    if not usable:
        raise InsufficientRiskData(f"the instruments share no trading day in the {window.value.lower()} window")
    return date.fromisoformat(usable[-1])


def _one_window(snapshots: Sequence[InstrumentMarketSnapshot]) -> SnapshotWindow:
    windows = {s.window for s in snapshots}
    if len(windows) != 1:
        raise InsufficientRiskData("snapshots of different windows cannot share one risk model")
    return windows.pop()


def _returns(snap: InstrumentMarketSnapshot, frame: pd.DataFrame) -> pd.Series:
    research = frame["research_close"]
    if snap.adjustment == "ADDITIVE":
        change = research.diff()
        denom = frame["raw_close"] - change
        out = change / denom
        out[denom <= 0] = np.nan  # a non-positive price makes a percentage return meaningless
        return out
    return research / research.shift(1) - 1.0


def daily_returns(snap: InstrumentMarketSnapshot) -> pd.Series:
    """The instrument's daily price return by trading day -- the estimator's
    own formula (exact across a futures roll), for callers that need the
    same returns outside a risk model (Phase H regime labels)."""
    return _returns(snap, snap.frame())


def estimate_risk_model(
    snapshots: Mapping[str, InstrumentMarketSnapshot],
    *,
    as_of: date,
    lookback_days: int = 252,
    min_observations: int = 200,
    annualization_days: int = 252,
) -> RiskModel:
    """``trailing-joint-panel/1`` over the snapshots named by instrument key.
    Reads nothing after ``as_of``."""
    keys = sorted(snapshots)
    if not keys:
        raise InsufficientRiskData("no instruments")
    window = _one_window([snapshots[k] for k in keys])
    if as_of >= window.end:
        raise InsufficientRiskData(
            "the as-of date must precede the 2023-2024 validation window" if window is SnapshotWindow.DISCOVERY
            else f"the as-of date must precede {window.end}")
    frames = {}
    for key in keys:
        snap = snapshots[key]
        if snap.data_role is not DataProvenanceRole.REAL:
            raise InsufficientRiskData(f"{key}: only real data may size a portfolio")
        f = snap.frame()
        f = f[pd.to_datetime(f.index).date <= as_of]
        if as_of.isoformat() not in f.index:
            raise InsufficientRiskData(f"{key}: no bar on the as-of date {as_of}")
        assert_no_holdout_ts(f["ts_event_ns"])
        frames[key] = f
    common = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    panel_days = common[-(lookback_days + 1):]
    returns = pd.DataFrame({key: _returns(snapshots[key], frames[key].loc[panel_days]) for key in keys},
                           index=panel_days).iloc[1:]
    bad = returns.isna().any(axis=1)
    returns = returns[~bad]
    n = len(returns)
    if n < min_observations:
        raise InsufficientRiskData(
            f"only {n} joint daily returns up to {as_of} (need {min_observations}) for {', '.join(keys)}")
    vols = returns.std(ddof=1) * np.sqrt(annualization_days)
    corr = returns.corr() if len(keys) > 1 else pd.DataFrame([[1.0]], index=keys, columns=keys)

    instruments = []
    for key in keys:
        snap, f = snapshots[key], frames[key]
        cls = classify_instrument(snap.domain, snap.symbol)
        if cls is None:
            raise InsufficientRiskData(f"{key}: no risk classification (risk-classification/1)")
        if not np.isfinite(vols[key]) or vols[key] <= 0:
            raise InsufficientRiskData(f"{key}: zero or undefined volatility over the window")
        tail = f.iloc[-lookback_days:]
        notional = (tail["volume"] * tail["raw_close"] * snap.multiplier).dropna()
        adv = float(notional.median()) if len(notional) else None
        last = f.loc[as_of.isoformat()]
        instruments.append(InstrumentRisk(
            key=key, domain=snap.domain, symbol=snap.symbol, execution_root=snap.execution_root,
            asset_class=cls.asset_class, sector=cls.sector, unit=snap.unit, multiplier=snap.multiplier,
            tick_size=snap.tick_size, price=float(last["raw_close"]), traded_symbol=str(last["traded_symbol"]),
            decision_ts_ns=int(last["ts_event_ns"]), annual_vol=round(float(vols[key]), 12), adv_usd=adv,
            volume_scope=snap.volume_scope, snapshot_fingerprint=snap.fingerprint(),
        ))
    pairs = tuple((a, b, round(float(corr.loc[a, b]), 12)) for i, a in enumerate(keys) for b in keys[i + 1:])
    notes = [f"{int(bad.sum())} panel day(s) dropped: a non-positive price made a return undefined"] if bad.any() else []
    if any(i.volume_scope == "PRIMARY_LISTING_ONLY" for i in instruments):
        notes.append("ETF liquidity counts the primary listing venue only (~27% of consolidated volume) -- a lower "
                     "bound")
    return RiskModel(
        window=window, as_of=as_of, window_start=date.fromisoformat(returns.index[0]), window_end=date.fromisoformat(returns.index[-1]),
        lookback_days=lookback_days, n_observations=n, annualization_days=annualization_days,
        instruments=tuple(instruments), correlations=pairs, notes=tuple(notes),
    )
