"""Phase 21.1 -- replay prefix consistency.

Each paper-trading step replays the FULL expanding window
``[window_start_ns, new_watermark_ns]`` from scratch through the deterministic
C++ engine (never an incremental / mutated Python-side state --
:mod:`alpha_agent.paper.engine`). A direct consequence: the historical PREFIX
of fills a later, longer replay produces must be byte-identical to the fills
an earlier, shorter replay of the SAME run already committed to the ledger.
If it is not -- a changed input file, a changed risk policy, a changed C++
build, a genuine non-determinism bug -- that is a real integrity violation and
must be refused loudly, never silently absorbed by slicing
``replay_fills[previous_fill_count:]`` and hoping the prefix still matches.

This module owns exactly one check, run BEFORE any ledger write for the step:
:func:`assert_replay_prefix_consistent`. On divergence it raises
:class:`~alpha_agent.paper.errors.PaperReplayDivergence` and the caller
(:mod:`alpha_agent.paper.engine`) writes nothing -- no step, no fill, no
position, no alert, no watermark/status change.

Phase 21.1b: comparison is now EXACT, with no float tolerance. This is safe,
not merely convenient: the C++ side formats every float here with a fixed
``%.12g`` (``detail::trade_g``, trade_export.hpp) before it ever reaches
Python, so two replays of IDENTICAL inputs through the deterministic engine
always produce the same double, the same decimal text, and -- IEEE-754
``float()`` parsing being deterministic -- the same bit pattern in Python;
SQLite's ``REAL`` column stores and returns that same double losslessly.
There is therefore no "harmless serialization noise" a tolerance could ever
be needed to absorb, only a real divergence a tolerance could hide. A
committed fill was previously round-tripped through this exact same
CSV-text -> ``float()`` -> SQLite ``REAL`` -> ``float`` path, so comparing it
by plain ``==`` against a freshly-parsed replay value is the correct exact
check, not an approximation.
"""
from __future__ import annotations

from collections.abc import Sequence

from alpha_agent.paper.errors import PaperReplayDivergence
from alpha_agent.paper.ledger import PaperFillRow

#: the C++ Fill-derived fields a committed fill and its replayed counterpart
#: must agree on. Deliberately excludes ledger-only bookkeeping (``fill_seq``,
#: ``step_ordinal``) -- those are POSITION-in-the-ledger metadata, not part of
#: the C++ Fill itself.
COMPARE_FIELDS: tuple[str, ...] = (
    "fill_id", "order_id", "ts_fill_ns", "instrument_id", "raw_symbol",
    "side", "quantity", "fill_price", "commission_usd", "slippage_ticks",
)


def fill_dict_from_csv_row(row: dict) -> dict:
    """Normalize one ``fills.csv`` (Phase 15A audit export) row's typed
    values -- the same coercions :meth:`alpha_agent.paper.engine
    .PaperTradingEngine` uses to build a :class:`PaperFillRow`."""
    return {
        "fill_id": int(row["fill_id"]),
        "order_id": int(row["order_id"]),
        "ts_fill_ns": int(row["ts_fill_ns"]),
        "instrument_id": int(row["instrument_id"]),
        "raw_symbol": str(row["raw_symbol"]),
        "side": str(row["side"]),
        "quantity": int(row["quantity"]),
        "fill_price": float(row["fill_price"]),
        "commission_usd": float(row["commission_usd"]),
        "slippage_ticks": float(row["slippage_ticks"]),
    }


def assert_replay_prefix_consistent(
    *, run_id: str, committed: Sequence[PaperFillRow], replayed_csv_rows: Sequence[dict],
) -> None:
    """``committed`` is the ledger's existing fills for ``run_id``, ordered by
    ``fill_seq`` ascending (i.e. :meth:`PaperLedger.list_fills`). ``replayed_csv_rows``
    is THIS step's full, freshly-replayed ``fills.csv``, parsed but otherwise
    untouched, in emission order.

    Raises :class:`PaperReplayDivergence` unless ``replayed_csv_rows`` begins
    with an EXACT, field-for-field (:data:`COMPARE_FIELDS`) copy of every
    ``committed`` fill, in order -- no tolerance, on any field, including the
    float ones. Never mutates or truncates either input.
    """
    if len(replayed_csv_rows) < len(committed):
        raise PaperReplayDivergence(
            f"paper run {run_id}: this replay produced {len(replayed_csv_rows)} fill(s), "
            f"fewer than the {len(committed)} already committed to the ledger -- refusing "
            "rather than silently discarding committed history"
        )
    for i, prev in enumerate(committed):
        new = fill_dict_from_csv_row(replayed_csv_rows[i])
        for field in COMPARE_FIELDS:
            a = getattr(prev, field)
            b = new[field]
            if a != b:
                raise PaperReplayDivergence(
                    f"paper run {run_id}: replay divergence at committed fill_seq="
                    f"{prev.fill_seq} (position {i} of the new replay) -- field "
                    f"{field!r} differs: committed={a!r} vs replayed={b!r}"
                )
