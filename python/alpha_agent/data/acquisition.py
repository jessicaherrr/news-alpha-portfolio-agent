"""Phase 13.5B -- real CME acquisition request planning (pure, no network).

Turns a :class:`~alpha_agent.data.real_dataset.RealDatasetPlan` plus observed
``.v.0`` roll transitions into the exact list of Databento ``HistoricalRequest``
objects, with two hard rules baked in:

* **2025 LOCKED_HOLDOUT is prohibited** -- every request is
  :func:`guard_no_holdout`-checked (``end <= HOLDOUT_START``); a violation raises.
* **budget cap** -- :class:`AcquisitionBudget` tracks running spend against the
  approved ceiling and refuses any download that would exceed it.

Roll-overlap raw symbols are resolved ONLY from observed transitions
(:func:`roll_overlap_requests`), never guessed.

Phase 23.2: the AUTHORITATIVE holdout guard now lives one layer down, in
:mod:`alpha_agent.data.databento_source` (``HistoricalRequest.__post_init__``)
-- every request this module builds is refused there before this module's own
:func:`guard_no_holdout` even runs, because holdout-violating windows can no
longer construct a :class:`HistoricalRequest` at all. ``HOLDOUT_START`` /
``HoldoutViolation`` are re-exported from there (not redeclared) so this
module and every caller of it keep working unchanged. ``guard_no_holdout`` is
kept as an explicit, now largely redundant, defense-in-depth layer -- this was
Phase 13.5B's original planner-level guard and Phase 23.2 does not weaken it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, timedelta

from alpha_agent.data.databento_source import HOLDOUT_START, HistoricalRequest, HoldoutViolation
from alpha_agent.data.roll_validation import RollDetection

DATASET = "GLBX.MDP3"                 # CLAUDE market-data rule 1
RESEARCH_SCHEMA = "ohlcv-1m"


class BudgetExceeded(RuntimeError):
    """A download would push cumulative spend past the approved cap."""


def guard_no_holdout(req: HistoricalRequest) -> HistoricalRequest:
    """Defense-in-depth re-check (Phase 13.5B original). By the time ``req``
    exists, ``HistoricalRequest.__post_init__`` (the authoritative guard, Phase
    23.2) has already refused any holdout-violating window, so this can only
    ever observe an already-safe request -- kept anyway, unweakened, in case a
    future refactor ever constructs a request some other way."""
    if req.end > HOLDOUT_START:
        raise HoldoutViolation(
            f"request {list(req.symbols)} {req.start}..{req.end} reaches the prohibited "
            f"LOCKED_HOLDOUT (>= {HOLDOUT_START}); Phase 13.5B is 2018-2024 only"
        )
    if req.start >= HOLDOUT_START:
        raise HoldoutViolation(f"request starts in the holdout: {req.start}")
    return req


@dataclass
class AcquisitionBudget:
    """Cumulative spend tracker. ``cap_usd`` is the approved combined ceiling."""

    cap_usd: float
    spent_usd: float = 0.0
    line_items: list[tuple[str, float]] = field(default_factory=list)

    def would_exceed(self, estimate_usd: float) -> bool:
        return (self.spent_usd + estimate_usd) > self.cap_usd + 1e-9

    def charge(self, label: str, actual_or_estimate_usd: float) -> None:
        if self.would_exceed(actual_or_estimate_usd):
            raise BudgetExceeded(
                f"{label}: ${actual_or_estimate_usd:.4f} would take cumulative spend to "
                f"${self.spent_usd + actual_or_estimate_usd:.4f} > cap ${self.cap_usd:.2f}"
            )
        self.spent_usd += actual_or_estimate_usd
        self.line_items.append((label, actual_or_estimate_usd))

    def remaining(self) -> float:
        return self.cap_usd - self.spent_usd


def continuous_request(root: str, start: str, end: str) -> HistoricalRequest:
    """``<ROOT>.v.0 ohlcv-1m`` volume-ranked front continuous (universe.yaml)."""
    return guard_no_holdout(HistoricalRequest(
        dataset=DATASET, symbols=(f"{root}.v.0",), schema=RESEARCH_SCHEMA,
        start=start, end=end, stype_in="continuous", stype_out="instrument_id",
    ))


DEFINITION_SNAPSHOT_MONTHS = 3           # quarterly (Jan/Apr/Jul/Oct 1) -- see below
DEFINITION_SNAPSHOT_WINDOW_DAYS = 4      # 4-day span so a holiday 1st still catches a trading day


def _period_first_days(start: str, end: str, months: int) -> list[str]:
    """First calendar day of every ``months``-th month, aligned to Jan, in
    ``[start, end)`` (end-exclusive)."""
    a = date.fromisoformat(start)
    b = date.fromisoformat(end)
    out: list[str] = []
    year, month = a.year, ((a.month - 1) // months) * months + 1
    while True:
        d = date(year, month, 1)
        if d >= b:
            break
        if d >= a:
            out.append(d.isoformat())
        month += months
        if month > 12:
            year += month // 12
            month = ((month - 1) % 12) + 1
    return out


def definition_snapshot_requests(
    root: str, start: str, end: str, *, months: int = DEFINITION_SNAPSHOT_MONTHS
) -> list[HistoricalRequest]:
    """One short ``<ROOT>.FUT`` parent definition snapshot every ``months``
    (default: quarterly), each spanning ``DEFINITION_SNAPSHOT_WINDOW_DAYS``.

    Definition cost scales with the requested SPAN, not the record count, so many
    short snapshots are far cheaper than one multi-year range request for CL.
    A futures contract is listed well before expiry and stays listed until
    expiry, so a quarterly parent snapshot captures every contract active
    anywhere near it -- and every ``instrument_id`` seen in the front / roll
    bars is a front-adjacent contract. Downstream ``keep_instrument_ids`` keeps
    only the contracts the bars actually reference.

    Roots that list only a few quarterly contracts at a time (CBOT Treasuries)
    have little snapshot-to-snapshot overlap, so the cadence is quarterly, not
    semi-annual, and the window is a few days wide -- a semi-annual snapshot that
    landed on a market holiday (e.g. Jan 1) returned an empty parent set and left
    a contract's front-month tenure with no definition (observed for ZNM2 2022).
    """
    reqs: list[HistoricalRequest] = []
    for d0 in _period_first_days(start, end, months):
        d1 = min(
            (date.fromisoformat(d0) + timedelta(days=DEFINITION_SNAPSHOT_WINDOW_DAYS)).isoformat(),
            end,
        )
        if d1 <= d0:
            continue
        reqs.append(guard_no_holdout(HistoricalRequest(
            dataset=DATASET, symbols=(f"{root}.FUT",), schema="definition",
            start=d0, end=d1, stype_in="parent", stype_out="instrument_id",
        )))
    return reqs


def roll_overlap_requests(
    transitions: list[RollDetection],
    *,
    window_start: str,
    window_end: str,
    pad_before_days: int = 4,
    pad_after_days: int = 8,
) -> list[HistoricalRequest]:
    """One ``raw_symbol`` request per observed transition for **both** the
    outgoing and incoming contract, over a tight window around the transition.
    The frozen ``SAME_TIMESTAMP_CLOSE_CLOSE`` roll-basis code
    (:func:`alpha_agent.data.rolls.build_roll_events` -> ``_aligned_gap``) needs
    a contemporaneous bar of BOTH contracts, so both are requested (matching the
    Phase 04.5 Stage B pattern). Clipped to the plan window and the holdout
    boundary. Never guessed -- the symbols come from the observed transition."""
    from datetime import datetime

    reqs: list[HistoricalRequest] = []
    for t in transitions:
        ts = datetime.fromtimestamp(t.transition_ts_ns / 1e9, tz=UTC).date()
        s = max(date.fromisoformat(window_start), ts - timedelta(days=pad_before_days)).isoformat()
        e = min(
            date.fromisoformat(window_end),
            date.fromisoformat(HOLDOUT_START),
            ts + timedelta(days=pad_after_days),
        ).isoformat()
        if e <= s:
            continue
        reqs.append(guard_no_holdout(HistoricalRequest(
            dataset=DATASET, symbols=tuple(sorted({t.from_raw_symbol, t.to_raw_symbol})),
            schema=RESEARCH_SCHEMA, start=s, end=e,
            stype_in="raw_symbol", stype_out="instrument_id",
        )))
    # de-duplicate identical (symbol,start,end)
    seen: set[tuple] = set()
    unique: list[HistoricalRequest] = []
    for r in reqs:
        key = (r.symbols, r.start, r.end)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def detect_all_transitions(
    vendor_ohlcv, registry, *, continuous_symbol: str
) -> list[RollDetection]:
    """Every ``instrument_id`` transition in a continuous vendor frame (not just
    the first -- a multi-year window has many rolls)."""
    from alpha_agent.data.roll_validation import InstrumentSpan

    b = vendor_ohlcv.rename(columns={"ts_event": "ts_event_ns"}).copy()
    b["ts_event_ns"] = b["ts_event_ns"].astype("int64")
    b["instrument_id"] = b["instrument_id"].astype("int64")
    b = b.sort_values("ts_event_ns", kind="stable").reset_index(drop=True)

    def _raw(iid: int) -> str:
        spec = registry.get(iid)
        return spec.raw_symbol if spec else f"<unknown:{iid}>"

    spans_by_id = {
        int(iid): InstrumentSpan(
            instrument_id=int(iid), raw_symbol=_raw(int(iid)),
            first_ts_ns=int(g["ts_event_ns"].min()), last_ts_ns=int(g["ts_event_ns"].max()),
            n_bars=len(g),
        )
        for iid, g in b.groupby("instrument_id", sort=True)
    }
    change = b["instrument_id"].ne(b["instrument_id"].shift())
    idx = [i for i in b.index[change] if i > 0]
    out: list[RollDetection] = []
    for i in idx:
        from_id = int(b.at[i - 1, "instrument_id"])
        to_id = int(b.at[i, "instrument_id"])
        out.append(RollDetection(
            continuous_symbol=continuous_symbol,
            from_instrument_id=from_id, to_instrument_id=to_id,
            from_raw_symbol=_raw(from_id), to_raw_symbol=_raw(to_id),
            transition_ts_ns=int(b.at[i, "ts_event_ns"]),
            spans=(spans_by_id[from_id], spans_by_id[to_id]),
        ))
    return out
