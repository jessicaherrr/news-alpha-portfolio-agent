"""Databento historical adapter.

Cost-first and download-guarded (CLAUDE market-data rules 2-3). The vendor import
is lazy so the rest of the package tests without the SDK installed. Nothing here
runs automatically in tests -- a live download needs an explicit ``max_cost_usd``
cap and a real API key in the environment.

This module is also the AUTHORITATIVE, lowest-shared-boundary enforcement point
for the locked research holdout (Phase 23.1 finding C1; Phase 23.2 remediation).
Every acquisition entry point -- the guarded ``alpha_agent.data.acquisition``
planners, and any other caller, present or future -- must construct a
:class:`HistoricalRequest` before it can call :func:`estimate_cost_usd` or
:func:`fetch_and_store_raw`. Refusing an out-of-bounds window at construction
time therefore cannot be bypassed by a caller that forgets a separate guard
call; ``estimate_cost_usd``/``fetch_and_store_raw`` re-check anyway, as defense
in depth. ``alpha_agent.data.acquisition`` re-exports :data:`HOLDOUT_START` /
:class:`HoldoutViolation` from here rather than redeclaring them (its own
``guard_no_holdout`` is kept as an additional, now largely redundant,
defense-in-depth layer). ``alpha_agent.registry.holdout_guard`` carries an
independently-declared mirror of the same ``"2025-01-01"`` value for the
registry-write boundary, deliberately not importing across that layer -- see
its own docstring; this is not a second conflicting definition, both layers
agree on the identical frozen date.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from alpha_agent.data.raw_store import RAW_ROOT, RawArtifact, store_raw

# Databento's supported symbology matrix does NOT allow these combinations
# (422 symbology_invalid_request). instrument_id is the authoritative join key:
# bars carry instrument_id, and instrument_id -> raw_symbol is resolved from the
# definition table / ContractRegistry. See docs/DATABENTO_REALITY.md.
_UNSUPPORTED_SYMBOLOGY: frozenset[tuple[str, str]] = frozenset(
    {("continuous", "raw_symbol"), ("parent", "raw_symbol")}
)

#: Locked research holdout boundary (CLAUDE.md market-data rules / autonomous-
#: execution STOP condition 2). Nothing may request or acquire timestamps at or
#: after this date. Half-open request semantics: a window ``[start, end)`` is
#: SAFE iff ``start < HOLDOUT_START and end <= HOLDOUT_START``.
HOLDOUT_START = "2025-01-01"


class HoldoutViolation(RuntimeError):
    """A :class:`HistoricalRequest` window ``[start, end)`` reaches the locked
    ``>= 2025-01-01`` research holdout.

    Raised at :class:`HistoricalRequest` CONSTRUCTION time -- before any
    Databento client is built, before ``metadata.get_cost()``, before
    ``timeseries.get_range()``, before any network access, before any spend,
    and before any raw-file write. This is the authoritative,
    lowest-shared-boundary guard: it cannot be bypassed by a caller that
    forgets to invoke a separate, higher-level guard function.
    """


def _assert_before_holdout(start: str, end: str) -> None:
    """SAFE: ``start < HOLDOUT_START and end <= HOLDOUT_START``.
    REFUSE: ``start >= HOLDOUT_START or end > HOLDOUT_START``."""
    if start >= HOLDOUT_START or end > HOLDOUT_START:
        raise HoldoutViolation(
            f"request window [{start}, {end}) reaches the locked research holdout "
            f"(>= {HOLDOUT_START}); refused before any Databento client, cost "
            f"estimate, network access, spend, or raw-file write"
        )


@dataclass(frozen=True)
class HistoricalRequest:
    dataset: str
    symbols: tuple[str, ...]
    schema: str
    start: str
    end: str
    stype_in: str = "raw_symbol"
    # Output symbology. instrument_id is the only output that is valid for both
    # continuous and parent input, and it is our authoritative join key.
    stype_out: str = "instrument_id"

    def __post_init__(self) -> None:
        _assert_before_holdout(self.start, self.end)
        if (self.stype_in, self.stype_out) in _UNSUPPORTED_SYMBOLOGY:
            raise ValueError(
                f"Databento does not support {self.stype_in} -> {self.stype_out}; "
                f"use stype_out='instrument_id' and resolve raw_symbol from the "
                f"definition table / ContractRegistry"
            )

    def definition_request(self) -> HistoricalRequest:
        """The companion instrument-definition request for this bar request.

        Uses ``parent`` symbology (e.g. ``NQ.FUT``) so every contract of the root
        is described. Output stays ``instrument_id`` -- the definition *records*
        already carry ``raw_symbol``, ``asset`` (root) and contract metadata.
        """
        parents = tuple(sorted({_parent_symbol(s) for s in self.symbols}))
        return HistoricalRequest(
            dataset=self.dataset,
            symbols=parents,
            schema="definition",
            start=self.start,
            end=self.end,
            stype_in="parent",
            stype_out="instrument_id",
        )


def _parent_symbol(symbol: str) -> str:
    """``NQ.v.0`` / ``NQZ6`` / ``NQ.FUT`` -> ``NQ.FUT``."""
    root = symbol.split(".")[0]
    # strip a trailing month/year code from a bare contract symbol like NQZ6
    for i, ch in enumerate(root):
        if ch.isdigit():
            root = root[:i]
            break
    return f"{root}.FUT"


def _client(api_key: str | None = None):
    try:
        import databento as db
    except ImportError as exc:
        raise RuntimeError("Install data extras: pip install -e '.[data]'") from exc
    key = api_key or os.getenv("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError("DATABENTO_API_KEY is not set")
    return db.Historical(key)


def estimate_cost_usd(request: HistoricalRequest, api_key: str | None = None) -> float:
    _assert_before_holdout(request.start, request.end)  # defense in depth; __post_init__ already checked
    client = _client(api_key)
    return float(
        client.metadata.get_cost(
            dataset=request.dataset,
            symbols=list(request.symbols),
            schema=request.schema,
            stype_in=request.stype_in,
            start=request.start,
            end=request.end,
        )
    )


def _dbn_extension(store) -> str:
    """'.dbn.zst' when the streamed response is zstd-compressed, else '.dbn'."""
    comp = getattr(store, "compression", None)
    return "dbn.zst" if comp is not None and str(comp).lower().endswith(("zstd", "zst")) else "dbn"


def fetch_and_store_raw(
    request: HistoricalRequest,
    *,
    max_cost_usd: float,
    api_key: str | None = None,
    root_dir: str | Path = RAW_ROOT,
) -> RawArtifact:
    """Estimate cost, enforce the cap, download, and persist the ORIGINAL DBN
    bytes as the immutable raw artifact.

    NOT called by any test -- requires a live key and network. Decoded frames are
    produced later by ``alpha_agent.data.decode`` as staging artifacts.
    """
    _assert_before_holdout(request.start, request.end)  # defense in depth; __post_init__ already checked
    cost = estimate_cost_usd(request, api_key)
    if cost > max_cost_usd:
        raise RuntimeError(
            f"Estimated Databento cost ${cost:.4f} exceeds cap ${max_cost_usd:.4f}; aborting."
        )

    client = _client(api_key)
    with tempfile.TemporaryDirectory() as tmp:
        stream_path = Path(tmp) / f"{request.schema}.dbn.zst"
        store = client.timeseries.get_range(
            dataset=request.dataset,
            symbols=list(request.symbols),
            schema=request.schema,
            stype_in=request.stype_in,
            stype_out=request.stype_out,
            start=request.start,
            end=request.end,
            path=str(stream_path),  # stream true DBN bytes straight to disk
        )
        raw_bytes = stream_path.read_bytes()

    ext = _dbn_extension(store)
    try:
        n_rows: int | None = len(store.to_ndarray())
    except Exception:  # noqa: BLE001 -- row count is informational only
        n_rows = None

    return store_raw(
        raw_bytes,
        vendor="databento",
        dataset=request.dataset,
        schema=request.schema,
        stype_in=request.stype_in,
        stype_out=request.stype_out,
        symbols=list(request.symbols),
        start=request.start,
        end=request.end,
        artifact_format=ext,
        filename=f"{request.schema}.{ext}",
        row_count=n_rows,
        extra={
            "estimated_cost_usd": cost,
            "encoding": "dbn",
            "compression": str(getattr(store, "compression", "")),
        },
        root_dir=root_dir,
    )
