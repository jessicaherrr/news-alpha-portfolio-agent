"""Phase 13.5C -- strictly-offline reconstitution of the real CME research /
validation dataset from the already-acquired immutable raw DBN artifacts.

Phase 13.5B acquired the raw bytes but wrote the derived layer-3/4/6 parquet as
*single files per root*, so the 2023-2024 run overwrote the 2018-2022 layers.
This module rebuilds the full 2018-01-01 .. 2024-12-31 layers per root **without
any network access**:

* :func:`load_stored_ohlcv_or_fail` / :func:`load_stored_definitions_or_fail`
  resolve a :class:`HistoricalRequest` to its write-once raw artifact, verify the
  SHA-256, and decode it locally. A missing artifact raises
  :class:`MissingRawArtifact` naming the request -- there is **no download
  fallback**, no ``metadata.get_cost`` / ``get_range`` call, no Databento client.
* :func:`reconstitute_root` replays the Phase 13.5B canonicalize ->
  ``build_futures_history`` pipeline offline over the union of the research and
  validation windows.
* :func:`daily_signal_series` produces the causal one-bar-per-``(root,
  trading_day)`` SIGNAL series (baseline feature path). Execution stays on the
  native 1-minute raw-contract bars.
* :func:`degraded_trading_days` maps the five vendor-flagged degraded UTC source
  dates to the affected exchange ``trading_day``s (data-quality sensitivity).

Nothing here may touch 2025: :func:`assert_offline_window` refuses any timestamp
or window reaching ``2025-01-01``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pandas as pd

from alpha_agent.data import decode as dec
from alpha_agent.data.acquisition import (
    HOLDOUT_START,
    HoldoutViolation,
    continuous_request,
    definition_snapshot_requests,
    detect_all_transitions,
    roll_overlap_requests,
)
from alpha_agent.data.backadjust import AdjustmentMode, build_forward_adjusted_series
from alpha_agent.data.calendars import SessionCalendar, default_calendar
from alpha_agent.data.canonicalize import NormalizationPolicy, canonicalize
from alpha_agent.data.databento_source import HistoricalRequest
from alpha_agent.data.definitions import (
    DefinitionRegistry,
    contracts_frame,
    parse_definition_frame,
)
from alpha_agent.data.futures_history import build_futures_history
from alpha_agent.data.price_domain import descale_fixed_point
from alpha_agent.data.raw_store import raw_artifact_dir, verify_manifest
from alpha_agent.schemas.market_data import RollEvent

NS_PER_DAY = 86_400_000_000_000
HOLDOUT_START_NS = int(
    datetime(2025, 1, 1, tzinfo=UTC).timestamp() * 1_000_000_000
)
RAW_ROOT = "data/raw"

# Plan B frozen windows (configs/real_dataset.yaml). End-exclusive.
PLAN_B_WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("research", "2018-01-01", "2023-01-01"),
    ("validation", "2023-01-01", "2025-01-01"),
)
RESEARCH_WINDOW = ("2018-01-01", "2023-01-01")
VALIDATION_WINDOW = ("2023-01-01", "2025-01-01")

ROOTS: tuple[str, ...] = ("ES", "NQ", "CL", "GC", "ZN")
FRACTIONAL_ROOTS = {"ZN", "ZB", "ZF", "ZT", "UB", "TN"}

# Databento `metadata.get_dataset_condition` flagged these source days `degraded`
# (docs/REAL_DATASET_ACQUISITION.md section 3.1). Used AS DELIVERED in the
# canonical analysis; excluded only in the predeclared data-quality sensitivity.
DEGRADED_VENDOR_DATES: tuple[str, ...] = (
    "2020-02-27",
    "2020-07-01",
    "2021-12-05",
    "2022-01-02",
    "2024-09-18",
)


class MissingRawArtifact(RuntimeError):
    """A required raw DBN artifact is not in the write-once store. Phase 13.5C is
    strictly offline -- there is no download fallback."""

    def __init__(self, req: HistoricalRequest, path):
        self.request = req
        self.path = path
        super().__init__(
            f"missing immutable raw artifact for request "
            f"{list(req.symbols)} {req.schema} {req.start}..{req.end} "
            f"(stype_in={req.stype_in}); expected under {path}. "
            f"Phase 13.5C does not download -- acquire it in a dedicated 13.5B-style "
            f"run first."
        )


def assert_offline_window(start: str, end: str) -> None:
    """Refuse any request window reaching the prohibited 2025 LOCKED_HOLDOUT."""
    if end > HOLDOUT_START or start >= HOLDOUT_START:
        raise HoldoutViolation(
            f"window {start}..{end} reaches the prohibited LOCKED_HOLDOUT "
            f"(>= {HOLDOUT_START}); Phase 13.5C is 2018-2024 only"
        )


def assert_no_holdout_ts(ts_ns) -> None:
    """Refuse any timestamp on/after 2025-01-01 UTC (feature / execution / spec)."""
    bad = [int(t) for t in (ts_ns if hasattr(ts_ns, "__iter__") else [ts_ns]) if int(t) >= HOLDOUT_START_NS]
    if bad:
        raise HoldoutViolation(
            f"{len(bad)} timestamp(s) on/after 2025-01-01 reached a Phase 13.5C code path "
            f"(first = {bad[0]}); the 2025 holdout must never be loaded"
        )


# ---------------------------------------------------------------------------
# offline raw loading
# ---------------------------------------------------------------------------
def _artifact_path_or_none(req: HistoricalRequest):
    d = raw_artifact_dir(
        vendor="databento", dataset=req.dataset, schema=req.schema,
        stype_in=req.stype_in, stype_out=req.stype_out, symbols=list(req.symbols),
        start=req.start, end=req.end, root_dir=RAW_ROOT,
    )
    for ext in ("dbn.zst", "dbn"):
        p = d / f"{req.schema}.{ext}"
        if p.exists():
            return p
    return d


def load_stored_ohlcv_or_fail(req: HistoricalRequest) -> pd.DataFrame:
    """Decode an already-stored ``ohlcv-1m`` artifact to a vendor frame. Offline
    only -- verifies the SHA-256 and never contacts Databento."""
    assert_offline_window(req.start, req.end)
    p = _artifact_path_or_none(req)
    if p is None or not p.is_file():
        raise MissingRawArtifact(req, p)
    verify_manifest(p)
    df = dec.load_dbn(p).to_df(price_type="fixed", pretty_ts=False, map_symbols=True).reset_index()
    return dec.ohlcv_df_to_vendor_frame(df)


def load_stored_definitions_or_fail(
    reqs: list[HistoricalRequest], keep_ids: set[int]
) -> pd.DataFrame:
    frames = []
    for req in reqs:
        assert_offline_window(req.start, req.end)
        p = _artifact_path_or_none(req)
        if p is None or p.is_dir():
            raise MissingRawArtifact(req, p)
        verify_manifest(p)
        df = dec.load_dbn(p).to_df(pretty_ts=False, map_symbols=False).reset_index()
        frames.append(dec.definition_df_to_vendor_frame(df, keep_instrument_ids=keep_ids))
    if not frames:
        raise ValueError("no definition requests supplied")
    merged = pd.concat(frames, ignore_index=True).drop_duplicates("instrument_id", keep="last")
    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# per-root reconstitution
# ---------------------------------------------------------------------------
@dataclass
class ReconstitutedRoot:
    root: str
    continuous_symbol: str
    canonical_bars: pd.DataFrame          # layer 2, all contracts, 2018-2024, ts-sorted
    contracts: pd.DataFrame               # frozen CONTRACT_COLUMNS
    registry: DefinitionRegistry
    rolls: list[RollEvent]
    continuous: pd.DataFrame              # layer 3 raw continuous (front), 1m
    forward_adjusted: pd.DataFrame        # layer 4 FORWARD_ADJUSTED (causal signal), 1m
    n_transitions: int
    calendar_version: str
    # ADDITIVE (Phase 13.5C matrix): the raw both-contracts roll-overlap 1m bars
    # (native raw-contract prices around each observed .v.0 transition). Used
    # ONLY to inject the contemporaneous outgoing-contract bar at each roll
    # ``effective_ts_ns`` into the C++ execution feed so a position held across a
    # real futures roll prices on the Phase 04.5 same-timestamp basis rather than
    # deferring until the outgoing contract expires. Never a signal input.
    overlap_bars: pd.DataFrame | None = None
    paths: dict[str, str] = field(default_factory=dict)


def reconstitute_root(
    root: str,
    *,
    calendar: SessionCalendar | None = None,
) -> ReconstitutedRoot:
    """Rebuild the full 2018-2024 layers for ``root`` from stored raw bytes only."""
    calendar = calendar or default_calendar()
    cont_sym = f"{root}.v.0"
    allow_fractional = root in FRACTIONAL_ROOTS

    # 1. continuous fronts (both windows), concat + dedupe
    cont_frames = []
    for _role, w0, w1 in PLAN_B_WINDOWS:
        cont_frames.append(load_stored_ohlcv_or_fail(continuous_request(root, w0, w1)))
    vendor = (
        pd.concat(cont_frames, ignore_index=True)
        .drop_duplicates(["instrument_id", "ts_event"], keep="first")
        .sort_values("ts_event", kind="stable")
        .reset_index(drop=True)
    )
    assert_no_holdout_ts(vendor["ts_event"])
    seen_ids = {int(i) for i in vendor["instrument_id"].unique()}

    # 2. definition snapshots (both windows)
    def_reqs: list[HistoricalRequest] = []
    for _role, w0, w1 in PLAN_B_WINDOWS:
        def_reqs.extend(definition_snapshot_requests(root, w0, w1))
    def_vendor = load_stored_definitions_or_fail(def_reqs, seen_ids)
    specs = parse_definition_frame(def_vendor, allow_fractional=allow_fractional)
    registry = DefinitionRegistry(specs)
    unresolved = seen_ids - {s.instrument_id for s in specs}
    if unresolved:
        raise MissingRawArtifact(
            def_reqs[0],
            f"instrument_id(s) {sorted(unresolved)} have no stored definition",
        )

    # 3. roll overlaps -- from OBSERVED transitions, per window (as acquired)
    overlap_frames = []
    for _role, w0, w1 in PLAN_B_WINDOWS:
        wv = vendor[(vendor["ts_event"] >= _iso_ns(w0)) & (vendor["ts_event"] < _iso_ns(w1))]
        transitions = detect_all_transitions(wv, registry, continuous_symbol=cont_sym)
        for orq in roll_overlap_requests(transitions, window_start=w0, window_end=w1):
            of = descale_fixed_point(
                load_stored_ohlcv_or_fail(orq).rename(columns={"ts_event": "ts_event_ns"})
            )
            overlap_frames.append(of)
    overlap = pd.concat(overlap_frames, ignore_index=True) if overlap_frames else None

    n_transitions = len(
        detect_all_transitions(vendor, registry, continuous_symbol=cont_sym)
    )

    # 4. canonicalize (layer 2) over the whole 2018-2024 span
    canonical, report = canonicalize(vendor, registry, calendar, policy=NormalizationPolicy())
    if report.errors:
        raise ValueError(f"{root}: canonicalization reported {len(report.errors)} error(s)")
    canonical = canonical.sort_values("ts_event_ns", kind="stable").reset_index(drop=True)
    assert_no_holdout_ts(canonical["ts_event_ns"])

    # 5. layers 3/4/6 -- no write (13.5C owns its own derived path)
    fh = build_futures_history(
        canonical, continuous_symbol=cont_sym, registry=registry, calendar=calendar,
        overlap_bars=overlap, adjustment_mode=AdjustmentMode.RETROSPECTIVE_RESEARCH,
        write=False,
    )
    fwd, fwd_report = build_forward_adjusted_series(
        fh.continuous, fh.rolls, continuous_symbol=cont_sym
    )
    if fwd_report.errors:
        raise ValueError(f"{root}: forward-adjust reported {len(fwd_report.errors)} error(s)")

    return ReconstitutedRoot(
        root=root,
        continuous_symbol=cont_sym,
        canonical_bars=canonical,
        contracts=contracts_frame(specs),
        registry=registry,
        rolls=fh.rolls,
        continuous=fh.continuous,
        forward_adjusted=fwd,
        n_transitions=n_transitions,
        calendar_version=calendar.version,
        overlap_bars=(
            overlap[["ts_event_ns", "instrument_id", "open", "high", "low", "close", "volume"]]
            .dropna(subset=["instrument_id"])
            .astype({"instrument_id": "int64"})
            .drop_duplicates(["instrument_id", "ts_event_ns"])
            .sort_values(["ts_event_ns", "instrument_id"])
            .reset_index(drop=True)
            if overlap is not None else None
        ),
    )


def _iso_ns(iso: str) -> int:
    return int(datetime.fromisoformat(iso).replace(tzinfo=UTC).timestamp() * 1e9)


# ---------------------------------------------------------------------------
# execution bars (native 1m raw-contract) + daily SIGNAL series
# ---------------------------------------------------------------------------
_BOUNDARY_BAR_COLUMNS = ("ts_event_ns", "instrument_id", "open", "high", "low", "close", "volume")


def execution_bars(
    recon: ReconstitutedRoot, start_ns: int, end_ns: int
) -> pd.DataFrame:
    """Native 1-minute raw-contract bars in the frozen boundary schema, windowed
    ``[start_ns, end_ns)``. This is the ONLY series that reaches the C++
    execution path; raw-contract identity is preserved. It stays the clean
    continuous-front feed (one raw-contract bar per timestamp) -- the roll
    overlap bars go through :func:`roll_close_marks` instead."""
    assert_no_holdout_ts([end_ns - 1])
    b = recon.canonical_bars
    m = (b["ts_event_ns"] >= int(start_ns)) & (b["ts_event_ns"] < int(end_ns))
    out = b.loc[m, list(_BOUNDARY_BAR_COLUMNS)].sort_values("ts_event_ns").reset_index(drop=True)
    return out


_ROLL_CLOSE_MARK_COLUMNS = ("instrument_id", "ts_event_ns", "close")


def roll_close_marks(
    recon: ReconstitutedRoot, start_ns: int, end_ns: int
) -> pd.DataFrame:
    """Auxiliary same-timestamp OUTGOING-contract closes for the C++ roll
    close-leg (Phase 13.5C ``EngineConfig::roll.close_marks``).

    For every observed roll, the outgoing contract's real 1-minute roll-overlap
    bars from ``recon.effective_ts_ns`` up to (but not into) its own expiry and
    the window end. These are NEVER primary ``MarketEvent``s -- the engine
    consults them only to price a position held across the roll on the Phase 04.5
    same-timestamp basis instead of deferring under the frozen ``RejectDefer``
    rule. A missing mark simply leaves ``RejectDefer`` unchanged.

    Deterministic: sorted by ``(instrument_id, ts_event_ns)``, de-duplicated,
    integer/float dtypes fixed. Columns ``instrument_id, ts_event_ns, close``.
    """
    cols = list(_ROLL_CLOSE_MARK_COLUMNS)
    ov = recon.overlap_bars
    if ov is None or ov.empty or not recon.rolls:
        return pd.DataFrame({c: pd.Series(dtype="int64" if c != "close" else "float64") for c in cols})
    exp_by_id = {
        int(i): int(e)
        for i, e in zip(recon.contracts["instrument_id"], recon.contracts["expiration_ns"])
        if str(e) not in ("", "nan")
    }
    lo0, hi0 = int(start_ns), min(int(end_ns), HOLDOUT_START_NS)
    frames: list[pd.DataFrame] = []
    for r in recon.rolls:
        oid = int(r.from_instrument_id)
        hi = min(hi0, exp_by_id.get(oid, hi0))
        lo = max(lo0, int(r.effective_ts_ns))
        if hi <= lo:
            continue
        m = (
            (ov["instrument_id"].astype("int64") == oid)
            & (ov["ts_event_ns"] >= lo)
            & (ov["ts_event_ns"] < hi)
        )
        if m.any():
            frames.append(ov.loc[m, cols])
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype="int64" if c != "close" else "float64") for c in cols})
    out = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["instrument_id", "ts_event_ns"])
        .sort_values(["instrument_id", "ts_event_ns"], kind="stable")
        .reset_index(drop=True)
    )
    out["instrument_id"] = out["instrument_id"].astype("int64")
    out["ts_event_ns"] = out["ts_event_ns"].astype("int64")
    out["close"] = out["close"].astype("float64")
    assert_no_holdout_ts(out["ts_event_ns"])
    # a close may be <= 0 (CL front-month traded below zero on 2020-04-20); only a
    # non-finite price is malformed.
    import math

    if not out["close"].map(math.isfinite).all():
        raise ValueError(f"{recon.root}: roll_close_marks has a non-finite close price")
    return out[cols]


def write_roll_close_marks_csv(df: pd.DataFrame, path) -> tuple[str, str]:
    """Write the marks frame to ``path`` (header ``instrument_id,ts_event_ns,close``)
    and return ``(str(path), sha256_of_file_bytes)`` for provenance."""
    import hashlib
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    return str(p), digest


def daily_signal_series(
    minute_bars: pd.DataFrame,
    root: str,
    *,
    calendar: SessionCalendar | None = None,
) -> pd.DataFrame:
    """Causal one-bar-per-``(root, trading_day)`` SIGNAL series from a 1-minute
    CONTINUOUS (or forward-adjusted) front series.

    A roll inside a ``trading_day`` still yields exactly ONE row (the input is
    the single front series, never per-contract). Each daily bar is stamped at
    that trading day's **last** 1-minute ``ts_event_ns`` -- so a feature computed
    at the daily bar and executed at the next eligible event never uses a future
    1-minute bar.
    """
    calendar = calendar or default_calendar()
    df = minute_bars.sort_values("ts_event_ns", kind="stable").reset_index(drop=True)
    tday, sess = calendar.classify_series(df["ts_event_ns"], root)
    df = df.assign(trading_day=tday.astype(str), _session=sess)
    g = df.groupby("trading_day", sort=True)
    out = pd.DataFrame(
        {
            "ts_event_ns": g["ts_event_ns"].last().astype("int64"),
            "trading_day": g["trading_day"].last(),
            "open": g["open"].first().astype("float64"),
            "high": g["high"].max().astype("float64"),
            "low": g["low"].min().astype("float64"),
            "close": g["close"].last().astype("float64"),
            "volume": g["volume"].sum().astype("int64"),
        }
    ).reset_index(drop=True)
    out = out.sort_values("ts_event_ns").reset_index(drop=True)
    assert_no_holdout_ts(out["ts_event_ns"])
    return out


# ---------------------------------------------------------------------------
# data-quality sensitivity: degraded vendor day -> trading_day map
# ---------------------------------------------------------------------------
def degraded_trading_days(
    root: str,
    *,
    calendar: SessionCalendar | None = None,
    degraded_dates: tuple[str, ...] = DEGRADED_VENDOR_DATES,
) -> list[str]:
    """Deterministic map from the vendor-degraded UTC source dates to the
    exchange ``trading_day``(s) they touch for ``root`` (a UTC date's 1-minute
    bars can fall in two CME trading days; both are returned). ISO strings,
    sorted, de-duplicated."""
    calendar = calendar or default_calendar()
    days: set[str] = set()
    for iso in degraded_dates:
        d0 = datetime.fromisoformat(iso).replace(tzinfo=UTC)
        for minute in range(1440):
            ts_ns = int((d0 + timedelta(minutes=minute)).timestamp() * 1e9)
            if ts_ns >= HOLDOUT_START_NS:
                continue
            tday, _sess = calendar.classify(ts_ns, root)
            days.add(tday.isoformat())
    return sorted(days)
