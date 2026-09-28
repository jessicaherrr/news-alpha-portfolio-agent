"""Contract lifecycle: `tradable_until` and the external first-notice join.

Databento instrument definitions carry no first-notice / last-trade field. This
module keeps those explicitly *missing* until an authoritative external source is
joined -- dates are never manufactured.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from alpha_agent.data.diagnostics import DiagnosticKind, DiagnosticsReport, Severity
from alpha_agent.schemas.market_data import ContractSpecModel

_NS_PER_DAY = 86_400_000_000_000

# Roots whose contracts are physically delivered -- holding past first notice in a
# simulation is unrealistic, so a missing first_notice_ns is a real risk.
PHYSICALLY_DELIVERED_ROOTS: frozenset[str] = frozenset({"CL", "GC", "ZN", "ZB", "ZF", "ZT"})


@dataclass(frozen=True)
class FirstNoticePolicy:
    """How first notice constrains the tradable window.

    ``safety_buffer_business_days`` (preferred, decision F) moves the cutoff back
    that many *weekday* sessions from first notice; ``safety_buffer_days`` is the
    simpler calendar-day version. Holidays are not yet modelled -- that is a
    later refinement of the weekday count.
    """

    apply_first_notice: bool = True
    safety_buffer_days: int = 0
    safety_buffer_business_days: int = 0


def _minus_business_days(ts_ns: int, n: int) -> int:
    if n <= 0:
        return ts_ns
    d = np.datetime64(int(ts_ns), "ns").astype("datetime64[D]")
    shifted = np.busday_offset(d, -n, roll="backward")
    keep_time = int(ts_ns) - int(d.astype("datetime64[ns]").astype("int64"))
    return int(shifted.astype("datetime64[ns]").astype("int64")) + keep_time


def tradable_until_ns(spec: ContractSpecModel, policy: FirstNoticePolicy | None = None) -> int:
    """The latest instant a contract may be traded / held in a simulation:

        min( expiration_ns,
             last_trade_ns                     (if known),
             first_notice_ns - safety buffer   (if known and policy.apply_first_notice) )
    """
    policy = policy or FirstNoticePolicy()
    candidates = [spec.expiration_ns]
    if spec.last_trade_ns is not None:
        candidates.append(spec.last_trade_ns)
    if policy.apply_first_notice and spec.first_notice_ns is not None:
        fn = spec.first_notice_ns - policy.safety_buffer_days * _NS_PER_DAY
        fn = _minus_business_days(fn, policy.safety_buffer_business_days)
        candidates.append(fn)
    return min(candidates)


def check_lifecycle_ordering(spec: ContractSpecModel) -> str | None:
    """Return an error message if the lifecycle timestamps are out of order."""
    if spec.expiration_ns <= spec.activation_ns:
        return f"{spec.raw_symbol}: expiration <= activation"
    if spec.last_trade_ns is not None and spec.last_trade_ns <= spec.activation_ns:
        return f"{spec.raw_symbol}: last_trade <= activation"
    if spec.first_notice_ns is not None:
        if spec.first_notice_ns <= spec.activation_ns:
            return f"{spec.raw_symbol}: first_notice <= activation"
        if spec.first_notice_ns > spec.expiration_ns:
            return f"{spec.raw_symbol}: first_notice > expiration"
    return None


# --- external lifecycle metadata join ---------------------------------------

class ContractLifecycleMeta(BaseModel):
    """One authoritative external lifecycle record for a real contract."""

    raw_symbol: str = Field(min_length=1)
    first_notice_ns: int | None = None
    last_trade_ns: int | None = None
    source: str = Field(min_length=1)     # e.g. "cme_calendar_2026", "manual"

    @classmethod
    def from_dates(
        cls, raw_symbol: str, *, source: str,
        first_notice: date | str | None = None, last_trade: date | str | None = None,
    ) -> ContractLifecycleMeta:
        def _ns(d):
            if d is None:
                return None
            d = date.fromisoformat(d) if isinstance(d, str) else d
            return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp() * 1e9)

        return cls(raw_symbol=raw_symbol, source=source,
                   first_notice_ns=_ns(first_notice), last_trade_ns=_ns(last_trade))


def load_lifecycle_meta(path: str | Path) -> dict[str, ContractLifecycleMeta]:
    """Load ``{raw_symbol: ContractLifecycleMeta}`` from a JSON list. Missing file
    -> empty dict (nothing enriched, nothing manufactured)."""
    p = Path(path)
    if not p.exists():
        return {}
    import json

    rows = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, ContractLifecycleMeta] = {}
    for r in rows:
        m = ContractLifecycleMeta.model_validate(r)
        out[m.raw_symbol] = m
    return out


def enrich_specs(
    specs: list[ContractSpecModel],
    meta: dict[str, ContractLifecycleMeta],
    *,
    report: DiagnosticsReport | None = None,
) -> list[ContractSpecModel]:
    """Apply external first-notice / last-trade to matching specs by raw_symbol.

    Specs with no matching metadata are returned unchanged (first_notice_ns stays
    ``None``). A physically-delivered contract still missing first notice is a
    warning -- its backtest tradable window falls back to expiration.
    """
    out: list[ContractSpecModel] = []
    for spec in specs:
        m = meta.get(spec.raw_symbol)
        if m is not None:
            spec = spec.model_copy(update={
                "first_notice_ns": m.first_notice_ns if m.first_notice_ns is not None
                else spec.first_notice_ns,
                "last_trade_ns": m.last_trade_ns if m.last_trade_ns is not None
                else spec.last_trade_ns,
            })
        err = check_lifecycle_ordering(spec)
        if err and report is not None:
            report.add(DiagnosticKind.INVALID_CONTRACT_LIFECYCLE_ORDERING, err,
                       instrument_id=spec.instrument_id)
        if (report is not None and spec.root_symbol in PHYSICALLY_DELIVERED_ROOTS
                and spec.first_notice_ns is None):
            report.add(
                DiagnosticKind.MISSING_FIRST_NOTICE_FOR_DELIVERABLE,
                f"{spec.raw_symbol} ({spec.root_symbol}) is physically delivered but has no "
                f"first_notice_ns; tradable window falls back to expiration",
                severity=Severity.WARNING, instrument_id=spec.instrument_id,
            )
        out.append(spec)
    return out
