"""Phase 9.1 point-in-time universe membership / survivorship-bias /
delisting handling -- the minimum-viable design the approved Phase 9 proposal
called for: a fixed, dated universe (never silently edited), an explicit
typed status per name, and a hard rule that a delisting is a typed event a
backtest must handle explicitly (flat the position, no silent gap-fill), not
full point-in-time index reconstruction (out of scope for V1).

Every function here is a pure function over already-loaded bars plus a typed
:class:`~alpha_agent.equities.schemas.EquityUniverseMembership` record -- no
network, no data spend, and (today) no real Equity bars exist to call this
against, since no acquisition has happened yet. It exists so that WHEN real
bars are acquired (a separate, user-approved MONEY/NETWORK decision), the
delisting-handling contract is already typed, tested, and ready.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from alpha_agent.equities.schemas import EquityUniverseMembership, UniverseMembershipStatus
from alpha_agent.equities.universe import EQUITY_UNIVERSE, UNIVERSE_DECLARED_AT

__all__ = [
    "MEMBERSHIP",
    "PostDelistingDataError",
    "is_active",
    "membership",
    "required_flat_by_date",
    "truncate_bars_at_delisting",
]


class PostDelistingDataError(ValueError):
    """Raised when a bar series for a DELISTED name carries rows after its
    delisting date -- the exact "silent gap-fill" failure mode the approved
    Phase 9 proposal forbids. A caller must truncate explicitly
    (:func:`truncate_bars_at_delisting`), never rely on the data simply
    stopping on its own."""


#: The Phase 9.1 fixed, dated membership snapshot -- every name in
#: ``EQUITY_UNIVERSE`` is ACTIVE as of ``UNIVERSE_DECLARED_AT`` (this V1
#: universe was deliberately chosen from currently-active large caps; see
#: ``alpha_agent.equities.universe``'s module docstring for the explicit
#: survivorship-selection caveat this does NOT solve). A future name that
#: delists inside the research window gets a real, cited
#: ``EquityUniverseMembership`` entry with ``status=DELISTED`` appended here --
#: never an in-place mutation of an existing record (mirrors the registry's
#: own append-only discipline).
MEMBERSHIP: dict[str, EquityUniverseMembership] = {
    ticker: EquityUniverseMembership(
        raw_symbol=ticker, declared_at=UNIVERSE_DECLARED_AT,
        status=UniverseMembershipStatus.ACTIVE, delisting=None,
    )
    for ticker in EQUITY_UNIVERSE
}


def membership(raw_symbol: str) -> EquityUniverseMembership:
    if raw_symbol not in MEMBERSHIP:
        raise KeyError(
            f"{raw_symbol!r} is not in the Phase 9.1 equities universe; no membership record exists"
        )
    return MEMBERSHIP[raw_symbol]


def is_active(raw_symbol: str) -> bool:
    return membership(raw_symbol).status == UniverseMembershipStatus.ACTIVE


def required_flat_by_date(raw_symbol: str) -> date | None:
    """The date by which any open position in `raw_symbol` MUST be flat --
    the delisting date, if delisted; ``None`` for an active name. A future
    target-schedule builder consults this to emit an explicit closing target,
    rather than letting a position ride into a gap in the data."""
    m = membership(raw_symbol)
    return m.delisting.delisting_date if m.delisting is not None else None


def truncate_bars_at_delisting(bars: pd.DataFrame, raw_symbol: str, *, date_column: str = "day") -> pd.DataFrame:
    """Returns `bars` unchanged for an ACTIVE name. For a DELISTED name,
    returns only rows on or before the delisting date -- the explicit,
    typed truncation the approved Phase 9 proposal requires in place of a
    silent gap-fill. Raises nothing; a caller that wants to detect
    "was there unexpected post-delisting data" should use
    :class:`PostDelistingDataError` via a direct comparison of
    ``len(bars)`` before/after, or call this defensively before any
    downstream feature computation."""
    m = membership(raw_symbol)
    if m.delisting is None:
        return bars
    return bars[bars[date_column] <= m.delisting.delisting_date].copy()


def assert_no_post_delisting_bars(bars: pd.DataFrame, raw_symbol: str, *, date_column: str = "day") -> None:
    """Fails loudly (never silently) if `bars` for a DELISTED name extends
    past its delisting date -- the guard a data-loading path should call
    immediately after truncation to prove the truncation actually worked, or
    call directly on a freshly-loaded frame to detect the failure mode before
    truncating."""
    m = membership(raw_symbol)
    if m.delisting is None:
        return
    bad = bars[bars[date_column] > m.delisting.delisting_date]
    if not bad.empty:
        raise PostDelistingDataError(
            f"{raw_symbol!r} is DELISTED as of {m.delisting.delisting_date} but its bar "
            f"series carries {len(bad)} row(s) after that date -- refusing to silently "
            "gap-fill or ignore this; truncate explicitly with truncate_bars_at_delisting()"
        )
