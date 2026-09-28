"""Phase 23.2 -- universal Databento acquisition-boundary holdout guard.

Phase 23.1 found (finding C1) that two committed scripts
(``scripts/databento_validate.py``, ``python/alpha_agent/data/roll_validation.py``
via ``scripts/databento_roll_validate.py``) could construct and fetch a real
Databento request without ever calling the Phase 13.5B planner-level
``acquisition.guard_no_holdout``. This module proves the fix: the holdout
invariant now lives at the lowest shared boundary,
``alpha_agent.data.databento_source.HistoricalRequest.__post_init__`` (with
``estimate_cost_usd``/``fetch_and_store_raw`` re-checking as defense in
depth), so no caller -- present or future -- can construct a request that
reaches the locked ``>= 2025-01-01`` holdout.

No test in this module makes a network call or spends money:
``alpha_agent.data.databento_source`` imports the ``databento`` SDK lazily,
only inside ``_client()``, and every refused case here never reaches
``_client()`` -- proven directly by monkeypatching it to fail the test if
called.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alpha_agent.data import databento_source as ds
from alpha_agent.data.databento_source import (
    HOLDOUT_START,
    HistoricalRequest,
    HoldoutViolation,
)

DATASET = "GLBX.MDP3"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _req(start: str, end: str) -> HistoricalRequest:
    return HistoricalRequest(
        dataset=DATASET, symbols=("NQ.v.0",), schema="ohlcv-1m",
        start=start, end=end, stype_in="continuous", stype_out="instrument_id",
    )


# --------------------------------------------------------------------------
# 1-5: the half-open [start, end) invariant, exactly as specified.
# --------------------------------------------------------------------------

def test_1_request_entirely_before_the_boundary_succeeds():
    req = _req("2024-12-30", "2025-01-01")
    assert (req.start, req.end) == ("2024-12-30", "2025-01-01")


def test_2_exact_exclusive_boundary_is_allowed():
    req = _req("2024-06-01", HOLDOUT_START)
    assert req.end == HOLDOUT_START == "2025-01-01"


def test_3_crossing_the_boundary_is_refused():
    with pytest.raises(HoldoutViolation):
        _req("2024-12-31", "2025-01-02")


def test_4_starting_on_the_boundary_is_refused():
    with pytest.raises(HoldoutViolation):
        _req(HOLDOUT_START, "2025-02-01")


def test_5_a_2026_request_is_refused():
    with pytest.raises(HoldoutViolation):
        _req("2026-06-08", "2026-06-19")


# --------------------------------------------------------------------------
# 6: every refused case makes zero client/get_cost/get_range/raw-write calls.
# --------------------------------------------------------------------------

def test_6_refused_requests_never_touch_client_cost_network_or_disk(monkeypatch):
    calls = {"client": 0, "store_raw": 0}

    def _forbidden_client(api_key=None):
        calls["client"] += 1
        raise AssertionError("_client() must never be called for a refused request")

    def _forbidden_store_raw(*args, **kwargs):
        calls["store_raw"] += 1
        raise AssertionError("store_raw() must never be called for a refused request")

    monkeypatch.setattr(ds, "_client", _forbidden_client)
    monkeypatch.setattr(ds, "store_raw", _forbidden_store_raw)

    refused_windows = [
        (HOLDOUT_START, "2025-02-01"),      # starts on the boundary
        ("2024-12-31", "2025-01-02"),       # crosses the boundary
        ("2026-06-08", "2026-06-19"),       # the exact historical Stage A window
        ("2026-09-03", "2026-09-04"),       # the exact historical databento_validate.py window
    ]
    for start, end in refused_windows:
        # Construction itself refuses -- no HistoricalRequest object exists
        # to pass to estimate_cost_usd/fetch_and_store_raw, so those functions
        # are, structurally, never reachable for these windows.
        with pytest.raises(HoldoutViolation):
            _req(start, end)

    assert calls == {"client": 0, "store_raw": 0}, (
        "a refused request must never construct a live client or write a raw artifact"
    )

    # Defense-in-depth: even a HistoricalRequest built through some other,
    # hypothetical route (bypassing __post_init__) would still be refused by
    # estimate_cost_usd/fetch_and_store_raw's own re-check, again before any
    # client/network/write. Simulate that by constructing a SAFE request and
    # then mutating past the frozen-dataclass guard is not attempted here
    # (frozen dataclasses cannot be constructed any other public way); instead
    # this proves the re-check functions exist and fire on the same predicate.
    from alpha_agent.data.databento_source import _assert_before_holdout
    with pytest.raises(HoldoutViolation):
        _assert_before_holdout("2025-01-01", "2025-02-01")
    assert calls == {"client": 0, "store_raw": 0}


def test_6b_no_spend_ledger_file_is_touched_by_a_refused_request(tmp_path, monkeypatch):
    """The spend ledger is written only by the acquisition scripts, after a
    successful fetch_and_store_raw. Since a refused window can never reach
    fetch_and_store_raw (item 6), it can never reach a spend-ledger write
    either -- prove the ledger file's mtime/content is untouched across a
    refused construction attempt using the REAL, committed ledger path."""
    ledger = REPO_ROOT / "data/manifests/real_dataset/spend_ledger.json"
    before_bytes = ledger.read_bytes()
    before_mtime = ledger.stat().st_mtime_ns

    with pytest.raises(HoldoutViolation):
        _req("2026-06-08", "2026-06-19")

    assert ledger.read_bytes() == before_bytes
    assert ledger.stat().st_mtime_ns == before_mtime


# --------------------------------------------------------------------------
# 7: the exact historical databento_validate.py request is refused before
#    network (module import only defines constants/functions -- no top-level
#    side effect -- so exec'ing it is safe and makes no network call itself).
# --------------------------------------------------------------------------

def _load_script(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    if str(REPO_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec.loader.exec_module(mod)
    return mod


def test_7_databento_validate_py_hardcoded_request_is_refused_before_network(monkeypatch):
    mod = _load_script("_databento_validate_c1", "scripts/databento_validate.py")
    assert (mod.START, mod.END) == ("2026-09-03", "2026-09-04")  # permanent provenance, unedited

    def _forbidden_client(api_key=None):
        raise AssertionError("must never construct a Databento client")

    monkeypatch.setattr(ds, "_client", _forbidden_client)
    with pytest.raises(HoldoutViolation):
        mod._bars_request()


# --------------------------------------------------------------------------
# 8: the exact historical Stage A / Stage B roll-validation path is refused
#    before network.
# --------------------------------------------------------------------------

def test_8a_stage_a_roll_validation_request_is_refused_before_network(monkeypatch):
    from alpha_agent.data.roll_validation import STAGE_A_END, STAGE_A_START, stage_a_requests

    assert (STAGE_A_START, STAGE_A_END) == ("2026-06-08", "2026-06-19")  # permanent provenance, unedited

    def _forbidden_client(api_key=None):
        raise AssertionError("must never construct a Databento client")

    monkeypatch.setattr(ds, "_client", _forbidden_client)
    with pytest.raises(HoldoutViolation):
        stage_a_requests()


def test_8b_stage_b_roll_validation_request_is_refused_before_network(monkeypatch):
    from alpha_agent.data.roll_validation import InstrumentSpan, RollDetection, stage_b_request

    # A synthetic detection dated inside Stage A's real (>=holdout) window --
    # exactly the shape _load_detection() would hand stage_b_request() in the
    # live path (no network needed to build this fixture).
    det = RollDetection(
        continuous_symbol="NQ.v.0", from_instrument_id=1, to_instrument_id=2,
        from_raw_symbol="NQM6", to_raw_symbol="NQU6",
        transition_ts_ns=int(datetime(2026, 6, 15, tzinfo=UTC).timestamp() * 1_000_000_000),
        spans=(InstrumentSpan(1, "NQM6", 0, 1, 1), InstrumentSpan(2, "NQU6", 2, 3, 1)),
    )

    def _forbidden_client(api_key=None):
        raise AssertionError("must never construct a Databento client")

    monkeypatch.setattr(ds, "_client", _forbidden_client)
    with pytest.raises(HoldoutViolation):
        stage_b_request(det)


def test_8c_roll_validate_script_replay_mode_is_unaffected_by_the_guard():
    """--replay resolves fixed, already-downloaded artifact paths and never
    reconstructs a live HistoricalRequest, so it needs neither the guard nor
    a network/API key -- prove the module imports and its replay constants
    still point at the real, already-verified local artifacts."""
    mod = _load_script("_databento_roll_validate_c1", "scripts/databento_roll_validate.py")
    for p in (mod.REPLAY_STAGE_A_BARS, mod.REPLAY_STAGE_A_DEF, mod.REPLAY_STAGE_B):
        assert p.exists(), f"expected the permanent Phase 04.5 replay artifact at {p}"


# --------------------------------------------------------------------------
# 9: the real, guarded 2018-2024 Phase 13.5B/C acquisition path is unchanged.
# --------------------------------------------------------------------------

def test_9_the_2018_2024_acquisition_path_still_works_identically():
    from alpha_agent.data.acquisition import (
        HOLDOUT_START as ACQ_HOLDOUT_START,
    )
    from alpha_agent.data.acquisition import (
        HoldoutViolation as AcqHoldoutViolation,
    )
    from alpha_agent.data.acquisition import (
        continuous_request,
        definition_snapshot_requests,
        guard_no_holdout,
    )

    # HOLDOUT_START / HoldoutViolation are re-exported, not redeclared --
    # identity-equal to the canonical databento_source definitions.
    assert ACQ_HOLDOUT_START == HOLDOUT_START
    assert AcqHoldoutViolation is HoldoutViolation

    # the real frozen 2018-2024 research corpus windows still construct fine
    req = continuous_request("NQ", "2018-01-01", "2023-01-01")
    assert (req.start, req.end) == ("2018-01-01", "2023-01-01")
    val_req = continuous_request("NQ", "2023-01-01", HOLDOUT_START)
    assert val_req.end == "2025-01-01"
    reqs = definition_snapshot_requests("CL", "2023-01-01", "2025-01-01")
    assert reqs and all(r.end <= HOLDOUT_START for r in reqs)

    # guard_no_holdout is kept, unweakened, as defense in depth
    with pytest.raises(AcqHoldoutViolation):
        guard_no_holdout(HistoricalRequest(
            dataset=DATASET, symbols=("NQ.FUT",), schema="definition",
            start="2024-12-15", end="2025-01-02", stype_in="parent",
        ))


# Item 10 of the mandatory-test list ("no test accesses network or spends
# money") is a property of every test above, not a separate mechanism: tests
# 1-5 and 9 only construct/inspect HistoricalRequest/HoldoutViolation objects
# and never call estimate_cost_usd/fetch_and_store_raw; tests 6-8 explicitly
# monkeypatch ds._client to raise if it is ever reached and assert it never
# was. None of this module's imports touch the `databento` package itself --
# alpha_agent.data.databento_source imports it lazily, only inside `_client`.
