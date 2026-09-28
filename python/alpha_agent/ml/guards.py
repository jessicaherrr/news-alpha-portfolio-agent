"""Phase 15 fail-loud guards: the locked holdout, and network access.

The registry already guards its own write boundary
(:mod:`alpha_agent.registry.holdout_guard`). Phase 15 needs the same guarantee
*upstream* of the registry -- at every point where market timestamps enter the
ML layer -- because a 2025 bar must never be featured, labelled, trained on,
scored or inspected in the first place (prompt 15A s.17).
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from alpha_agent.ml.errors import NetworkAccessForbidden
from alpha_agent.registry.holdout_guard import (
    BOOKKEEPING_TIMESTAMP_KEYS,
    HOLDOUT_START,
    HOLDOUT_START_NS,
    HoldoutAccessError,
    assert_no_holdout_market_data,
)

#: 2018-01-01T00:00:00Z in epoch-ns -- the first instant of the Phase 15
#: development corpus. Nothing before this exists in the acquired dataset.
DEVELOPMENT_CORPUS_START_NS = 1_514_764_800_000_000_000
#: Exclusive end of the development corpus == HOLDOUT_START_NS.
DEVELOPMENT_CORPUS_END_NS = HOLDOUT_START_NS

__all__ = [
    "DEVELOPMENT_CORPUS_END_NS",
    "DEVELOPMENT_CORPUS_START_NS",
    "HOLDOUT_START",
    "HOLDOUT_START_NS",
    "HoldoutAccessError",
    "assert_exclusive_corpus_bound",
    "assert_half_open_window",
    "assert_no_holdout_in_config_payload",
    "assert_no_holdout_market_data",
    "assert_no_holdout_timestamps",
    "assert_within_development_corpus",
    "forbid_network",
]


def assert_no_holdout_timestamps(ts_ns: Iterable[int] | np.ndarray, *, what: str) -> None:
    """Refuse any market timestamp at or after ``2025-01-01``.

    This is the guard that must fire on ANY code path touching
    ``ts >= 2025-01-01`` (CLAUDE.md autonomous-execution STOP condition 2).
    """
    arr = np.asarray(list(ts_ns) if not isinstance(ts_ns, np.ndarray) else ts_ns, dtype="int64")
    if arr.size == 0:
        return
    bad = arr[arr >= HOLDOUT_START_NS]
    if bad.size:
        raise HoldoutAccessError(
            f"{what}: {bad.size} timestamp(s) at or after {HOLDOUT_START} "
            f"(first offender {int(bad[0])}). 2025 is the LOCKED_FINAL_HOLDOUT: it is "
            "never downloaded, queried, cost-fetched, loaded, featured, labelled, "
            "trained on, scored or inspected."
        )


def assert_within_development_corpus(
    ts_ns: Iterable[int] | np.ndarray, *, what: str
) -> None:
    """The stronger Phase 15 gate: inside ``2018-01-01 .. 2024-12-31`` inclusive."""
    arr = np.asarray(list(ts_ns) if not isinstance(ts_ns, np.ndarray) else ts_ns, dtype="int64")
    if arr.size == 0:
        return
    assert_no_holdout_timestamps(arr, what=what)
    early = arr[arr < DEVELOPMENT_CORPUS_START_NS]
    if early.size:
        raise HoldoutAccessError(
            f"{what}: {early.size} timestamp(s) before the Phase 15 development corpus "
            f"start (first offender {int(early[0])}); the acquired dataset begins 2018-01-01"
        )


def forbid_network(*_args: object, **_kwargs: object) -> None:
    """Phase 15A does no network I/O. Wire this in as a deliberate tripwire."""
    raise NetworkAccessForbidden(
        "Phase 15A performs no network access: no data acquisition, no cost query, "
        "no package installation, no model download. Any of those requires explicit "
        "human approval first."
    )


def assert_exclusive_corpus_bound(ts_ns: int, *, what: str) -> None:
    """The ONE legitimate appearance of ``2025-01-01`` in a Phase 15 config.

    CLAUDE.md's naming rule already says it: an END date of ``2025-01-01`` is an
    EXCLUSIVE boundary for data ending 2024-12-31 -- it does not mean holdout
    data exists or was loaded. A half-open window ``[start, 2025-01-01)`` is
    therefore the correct way to say "everything up to the holdout and nothing
    inside it", and it must be exactly that instant: one nanosecond later is a
    2025 timestamp, and a bound that merely *approaches* it is a bug worth
    failing on.
    """
    if int(ts_ns) != HOLDOUT_START_NS:
        raise HoldoutAccessError(
            f"{what}: {ts_ns} is not the exclusive locked-holdout boundary "
            f"{HOLDOUT_START_NS} ({HOLDOUT_START}). A Phase 15 corpus bound is either "
            "strictly inside the development corpus or exactly this exclusive boundary."
        )


def assert_half_open_window(start_ns: int, end_ns: int, *, what: str) -> None:
    """A ``[start, end)`` window that may end exactly at the exclusive boundary.

    ``start`` is guarded like any market timestamp -- it names a real instant of
    data. ``end`` is a bound, not an observation, so it may equal
    ``2025-01-01`` and may never exceed it.
    """
    assert_no_holdout_timestamps([start_ns], what=f"{what}.start")
    if int(end_ns) > HOLDOUT_START_NS:
        raise HoldoutAccessError(
            f"{what}.end {end_ns} reaches past the exclusive locked-holdout boundary "
            f"{HOLDOUT_START_NS} ({HOLDOUT_START})"
        )
    if int(end_ns) <= int(start_ns):
        raise ValueError(f"{what}: window end {end_ns} does not follow start {start_ns}")


def assert_no_holdout_in_config_payload(
    payload: object,
    *,
    exclusive_bound_keys: frozenset[str] = frozenset(),
    half_open_window_keys: frozenset[str] = frozenset(),
    path: str = "$",
) -> None:
    """Holdout guard for a CONFIG payload that legitimately carries end bounds.

    Identical to :func:`assert_no_holdout_market_data` except that:

    * a value under ``exclusive_bound_keys`` may be exactly the exclusive
      boundary -- and is checked to be exactly that, never merely "2025-ish";
    * a value under ``half_open_window_keys`` is a sequence of ``[start, end)``
      pairs whose ``end`` may reach the boundary but never pass it.

    Everything else is guarded unchanged. The allowance is deliberately narrow:
    it is granted by KEY, so a stray 2025 timestamp anywhere else still fails.
    """
    if isinstance(payload, dict):
        for k, v in payload.items():
            if k in BOOKKEEPING_TIMESTAMP_KEYS:
                continue
            if k in exclusive_bound_keys and isinstance(v, int) and not isinstance(v, bool):
                assert_exclusive_corpus_bound(v, what=f"{path}.{k}")
                continue
            if k in half_open_window_keys and isinstance(v, (list, tuple)):
                for i, window in enumerate(v):
                    if not (isinstance(window, (list, tuple)) and len(window) == 2):
                        raise ValueError(
                            f"{path}.{k}[{i}] is not a [start, end) pair: {window!r}"
                        )
                    assert_half_open_window(
                        int(window[0]), int(window[1]), what=f"{path}.{k}[{i}]"
                    )
                continue
            assert_no_holdout_in_config_payload(
                v,
                exclusive_bound_keys=exclusive_bound_keys,
                half_open_window_keys=half_open_window_keys,
                path=f"{path}.{k}",
            )
        return
    if isinstance(payload, (list, tuple)):
        for i, v in enumerate(payload):
            assert_no_holdout_in_config_payload(
                v,
                exclusive_bound_keys=exclusive_bound_keys,
                half_open_window_keys=half_open_window_keys,
                path=f"{path}[{i}]",
            )
        return
    assert_no_holdout_market_data(payload, path=path)
