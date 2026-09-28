"""Vendor OHLCV records -> canonical raw-contract bars (layer 2).

Produces exactly ``CANONICAL_BAR_COLUMNS``. UTC ``ts_event_ns`` is preserved
byte-for-byte; ``trading_day`` / ``session`` are added as derived labels. No
silent forward-fill, no silent reordering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from alpha_agent.data.calendars import SessionCalendar, SessionCalendarMissing
from alpha_agent.data.definitions import DefinitionRegistry
from alpha_agent.data.diagnostics import (
    DiagnosticKind,
    DiagnosticsReport,
    PipelineError,
    Severity,
)
from alpha_agent.data.price_domain import descale_fixed_point
from alpha_agent.schemas.market_data import (
    CANONICAL_BAR_COLUMNS,
    NORMALIZED_PRICE_ABS_LIMIT,
    PRICE_SCALE,
    UNDEF_PRICE,
    VENDOR_OHLCV_COLUMNS,
    is_tradable_contract_symbol,
)

_PRICE_COLS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class NormalizationPolicy:
    on_non_monotonic: Literal["fail", "sort"] = "fail"
    on_undefined_price: Literal["reject_row", "fail"] = "reject_row"
    on_unknown_instrument: Literal["reject_row", "fail"] = "reject_row"
    on_invalid_row: Literal["reject_row", "fail"] = "reject_row"
    on_gaps: Literal["record", "fail"] = "record"
    on_missing_calendar: Literal["reject_row", "fail"] = "fail"
    expected_bar_interval_ns: int = 60_000_000_000  # ohlcv-1m

    @property
    def price_scale_policy(self) -> str:
        return f"fixed_point/{PRICE_SCALE} -> float64; UNDEF_PRICE={UNDEF_PRICE} {self.on_undefined_price}"


def _fail(report: DiagnosticsReport, msg: str) -> None:
    raise PipelineError(msg, report)


def canonicalize(
    vendor_ohlcv: pd.DataFrame,
    registry: DefinitionRegistry,
    calendar: SessionCalendar,
    *,
    policy: NormalizationPolicy | None = None,
) -> tuple[pd.DataFrame, DiagnosticsReport]:
    policy = policy or NormalizationPolicy()
    missing = [c for c in VENDOR_OHLCV_COLUMNS if c not in vendor_ohlcv.columns]
    if missing:
        raise ValueError(f"vendor OHLCV frame missing columns: {missing}")

    df = vendor_ohlcv.loc[:, list(VENDOR_OHLCV_COLUMNS)].copy()
    df = df.rename(columns={"ts_event": "ts_event_ns"})
    df["ts_event_ns"] = df["ts_event_ns"].astype("int64")
    df["instrument_id"] = df["instrument_id"].astype("int64")
    for c in _PRICE_COLS:
        df[c] = df[c].astype("int64")
    df["volume"] = df["volume"].astype("int64")
    df["symbol"] = df["symbol"].astype("string").fillna("")

    report = DiagnosticsReport(n_input_rows=len(df))

    # --- duplicates: (instrument_id, ts_event_ns) must be unique. Always fatal.
    dup_mask = df.duplicated(subset=["instrument_id", "ts_event_ns"], keep=False)
    if dup_mask.any():
        offenders = (
            df.loc[dup_mask, ["instrument_id", "ts_event_ns"]]
            .drop_duplicates()
            .to_records(index=False)
            .tolist()
        )
        report.add(
            DiagnosticKind.DUPLICATE_BAR,
            f"{int(dup_mask.sum())} rows share an (instrument_id, ts_event_ns) key",
            count=int(dup_mask.sum()),
            offenders=[[int(a), int(b)] for a, b in offenders][:20],
        )
        _fail(report, "duplicate (instrument_id, ts_event_ns) bars -- refusing to normalize")

    reject = pd.Series(False, index=df.index)

    # --- undefined vendor price sentinel
    sentinel = (df[list(_PRICE_COLS)] == UNDEF_PRICE).any(axis=1)
    if sentinel.any():
        report.add(
            DiagnosticKind.UNDEFINED_PRICE_SENTINEL,
            f"{int(sentinel.sum())} rows contain the DBN UNDEF_PRICE sentinel",
            count=int(sentinel.sum()),
        )
        if policy.on_undefined_price == "fail":
            _fail(report, "undefined price sentinel present")
        reject |= sentinel

    # --- unknown instrument_id
    specs_by_id = {s.instrument_id: s for s in registry.specs()}
    unknown = ~df["instrument_id"].isin(specs_by_id)
    if unknown.any():
        report.add(
            DiagnosticKind.UNKNOWN_INSTRUMENT_ID,
            f"{int(unknown.sum())} rows reference an instrument_id not in the contract registry",
            count=int(unknown.sum()),
            instrument_ids=sorted({int(i) for i in df.loc[unknown, "instrument_id"].unique()})[:20],
        )
        if policy.on_unknown_instrument == "fail":
            _fail(report, "unknown instrument_id present")
        reject |= unknown

    resolvable = ~reject
    auth_raw = df["instrument_id"].map(
        lambda i: specs_by_id[i].raw_symbol if i in specs_by_id else ""
    )
    auth_root = df["instrument_id"].map(
        lambda i: specs_by_id[i].root_symbol if i in specs_by_id else ""
    )

    # --- vendor symbol is a LABEL, not the raw contract.
    #
    # Databento does not support continuous/parent -> raw_symbol, so our OHLCV
    # requests use stype_out=instrument_id. The vendor "symbol" column then holds
    # the requested smart symbol (e.g. "NQ.v.0"), which is expected and never a
    # rejection cause. The authoritative raw_symbol always comes from the
    # ContractRegistry via instrument_id (auth_raw below).
    #
    # Only an *independently resolved real contract symbol* that disagrees with
    # the registry is a genuine conflict worth flagging.
    vendor_sym = df["symbol"].astype(str)
    independent = resolvable & vendor_sym.map(is_tradable_contract_symbol)
    mismatch = independent & (vendor_sym != auth_raw)
    if mismatch.any():
        pairs = sorted(
            {(str(a), str(b)) for a, b in zip(vendor_sym[mismatch], auth_raw[mismatch])}
        )[:20]
        report.add(
            DiagnosticKind.RAW_SYMBOL_MISMATCH,
            f"{int(mismatch.sum())} rows: vendor contract symbol != contract-definition raw_symbol",
            count=int(mismatch.sum()),
            vendor_vs_definition=[list(p) for p in pairs],
        )
        if policy.on_invalid_row == "fail":
            _fail(report, "vendor symbol vs contract-definition mismatch")
        reject |= mismatch
        resolvable &= ~mismatch

    # --- root could not be determined (defensive; registry roots are always set)
    root_bad = resolvable & (auth_root == "")
    if root_bad.any():
        report.add(
            DiagnosticKind.ROOT_UNDETERMINED,
            f"{int(root_bad.sum())} rows have no determinable root symbol",
            count=int(root_bad.sum()),
        )
        reject |= root_bad
        resolvable &= ~root_bad

    # --- price / OHLC / volume checks on de-scaled prices (resolvable rows only)
    px = descale_fixed_point(df.loc[resolvable, list(_PRICE_COLS)], price_cols=_PRICE_COLS)
    o, h, low_, c = px["open"], px["high"], px["low"], px["close"]

    # A normalized futures price may be positive, ZERO, or NEGATIVE (WTI crude
    # traded through zero on 2020-04-20). The real failure modes are non-finite
    # values (bad decode) and an un-normalized fixed-point magnitude (skipped
    # descale). Sign is NOT a data-quality signal here. Product-specific "a
    # negative price is impossible for THIS root" QA, if wanted, belongs in a
    # dedicated per-root check, not this generic gate.
    implausible = (~np.isfinite(px) | (px.abs() >= NORMALIZED_PRICE_ABS_LIMIT)).any(axis=1)
    if implausible.any():
        idx = px.index[implausible]
        report.add(
            DiagnosticKind.IMPLAUSIBLE_NORMALIZED_PRICE,
            f"{len(idx)} rows have a non-finite or un-normalized price",
            count=len(idx),
        )
        if policy.on_invalid_row == "fail":
            _fail(report, "implausible normalized price present")
        reject.loc[idx] = True

    invalid_ohlc = (h < pd.concat([o, c, low_], axis=1).max(axis=1)) | (
        low_ > pd.concat([o, c, h], axis=1).min(axis=1)
    )
    if invalid_ohlc.any():
        idx = px.index[invalid_ohlc]
        report.add(
            DiagnosticKind.INVALID_OHLC,
            f"{len(idx)} rows violate high>=max(o,c,l) / low<=min(o,c,h)",
            count=len(idx),
        )
        if policy.on_invalid_row == "fail":
            _fail(report, "invalid OHLC present")
        reject.loc[idx] = True

    neg_vol = resolvable & (df["volume"] < 0)
    if neg_vol.any():
        report.add(
            DiagnosticKind.NEGATIVE_VOLUME,
            f"{int(neg_vol.sum())} rows have negative volume",
            count=int(neg_vol.sum()),
        )
        if policy.on_invalid_row == "fail":
            _fail(report, "negative volume present")
        reject |= neg_vol

    # --- materialise surviving rows (prices -> normalized points, one primitive)
    kept = descale_fixed_point(df.loc[~reject], price_cols=_PRICE_COLS)
    kept["raw_symbol"] = auth_raw[~reject].astype(str)
    kept["root_symbol"] = auth_root[~reject].astype(str)
    report.n_rejected_rows = int(reject.sum())

    # --- monotonicity per instrument (in arrival order)
    reordered_rows = 0
    for iid, grp in kept.groupby("instrument_id", sort=False):
        if not grp["ts_event_ns"].is_monotonic_increasing:
            reordered_rows += len(grp)
            report.add(
                DiagnosticKind.NON_MONOTONIC_TIMESTAMP,
                f"instrument_id {int(iid)} has non-monotonic ts_event_ns in arrival order",
                count=1,
                instrument_id=int(iid),
            )
            if policy.on_non_monotonic == "fail":
                _fail(report, "non-monotonic timestamps -- refusing to reorder silently")
    if reordered_rows and policy.on_non_monotonic == "sort":
        report.add(
            DiagnosticKind.TIMESTAMPS_REORDERED,
            f"sorted {reordered_rows} rows by (instrument_id, ts_event_ns) per policy",
            severity=Severity.WARNING,
            count=reordered_rows,
        )
    kept = kept.sort_values(["instrument_id", "ts_event_ns"], kind="stable").reset_index(drop=True)

    # --- trading_day / session per root
    kept["trading_day"] = pd.NaT
    kept["session"] = ""
    for root, grp in kept.groupby("root_symbol", sort=False):
        try:
            days, sessions = calendar.classify_series(grp["ts_event_ns"], root)
        except SessionCalendarMissing as exc:
            report.add(
                DiagnosticKind.SESSION_CALENDAR_MISSING,
                str(exc),
                count=len(grp),
            )
            if policy.on_missing_calendar == "fail":
                _fail(report, f"no session calendar for root {root!r}")
            kept = kept.drop(index=grp.index)
            report.n_rejected_rows += len(grp)
            continue
        kept.loc[grp.index, "trading_day"] = days.values
        kept.loc[grp.index, "session"] = sessions.values

    # --- missing-minute gaps (record, never fill)
    _detect_gaps(kept, report, policy, calendar)

    canonical = kept.loc[:, list(CANONICAL_BAR_COLUMNS)].reset_index(drop=True)
    canonical["ts_event_ns"] = canonical["ts_event_ns"].astype("int64")
    canonical["instrument_id"] = canonical["instrument_id"].astype("int64")
    canonical["volume"] = canonical["volume"].astype("int64")
    canonical["raw_symbol"] = canonical["raw_symbol"].astype(str)
    canonical["root_symbol"] = canonical["root_symbol"].astype(str)
    canonical["session"] = canonical["session"].astype(str)
    report.n_output_rows = len(canonical)

    if policy.on_gaps == "fail" and report.has_kind(DiagnosticKind.MISSING_MINUTE_GAP):
        _fail(report, "missing-minute gaps present and policy is on_gaps='fail'")

    return canonical, report


def _detect_gaps(
    kept: pd.DataFrame,
    report: DiagnosticsReport,
    policy: NormalizationPolicy,
    calendar: SessionCalendar,
) -> None:
    """A gap between two consecutive bars is only "missing minutes" for the
    slots that fall in a tradeable session -- slots inside MAINTENANCE / CLOSED
    are expected (the daily halt, the weekly halt) and are not flagged."""
    interval = policy.expected_bar_interval_ns
    tradeable = {"RTH", "ETH"}
    for iid, grp in kept.groupby("instrument_id", sort=False):
        g = grp.sort_values("ts_event_ns")
        ts = g["ts_event_ns"].to_numpy()
        root = str(g["root_symbol"].iloc[0])
        if len(ts) < 2:
            continue
        for i in np.where(np.diff(ts) > interval)[0]:
            slots = range(int(ts[i]) + interval, int(ts[i + 1]), interval)
            missing = sum(
                1 for s in slots if calendar.classify(s, root)[1].value in tradeable
            )
            if missing == 0:
                continue  # gap fully explained by a halt
            report.add(
                DiagnosticKind.MISSING_MINUTE_GAP,
                f"instrument_id {int(iid)}: {missing} missing tradeable bar(s) between "
                f"{int(ts[i])} and {int(ts[i + 1])}",
                severity=Severity.WARNING,
                count=missing,
                instrument_id=int(iid),
                ts_event_ns=int(ts[i]),
                gap_end_ns=int(ts[i + 1]),
                missing_bars=missing,
            )
