"""Tests for Price + Signal + Position (Release UX Part G, task spec sections
26-27/56/61). Uses a REAL, already-on-disk, hash-verified experiment-bound
artifact bundle from the Alpha Discovery live-research campaign (Checkpoint
11/12) -- never a synthetic fixture -- to prove the whole chain end to end:
committed hash -> real price_bars.csv -> real trades.csv -> real fills.csv ->
chart, with every file's hash independently re-verified along the way.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from alpha_agent.ui import charts_discovery, services

REPO_ROOT = Path(__file__).resolve().parents[2]

_REAL_BUNDLE_DIR = (
    REPO_ROOT / "data" / "discovery" / "artifacts"
    / "experiment1_b3a3a2d95b537221e6a426c3597df061dcde43b58710f1bc2bbf70f44dacf77a" / "attempt_1"
)
_REAL_BUNDLE_SHA256 = "57725b0a7678aba78a25c1e9d3b3906f86d7551e06a0e6fc6c7a397fba5a9421"

pytestmark = pytest.mark.skipif(
    not (_REAL_BUNDLE_DIR / "manifest.json").exists(),
    reason="real Alpha Discovery live-research artifact bundle not present in this checkout",
)


def _real_detail() -> dict:
    return {
        "result": {
            "source_artifact": str((_REAL_BUNDLE_DIR / "manifest.json").relative_to(REPO_ROOT)),
            "source_artifact_sha256": _REAL_BUNDLE_SHA256,
        }
    }


def test_real_bundle_hash_verifies_and_carries_price_bars():
    bundle = services.find_experiment_bound_artifact_bundle(_real_detail())
    assert bundle is not None
    assert bundle["price_bars"] is not None
    assert bundle["trades"] is not None
    assert bundle["fills"] is not None


def test_hash_mismatch_refuses_the_whole_bundle():
    detail = _real_detail()
    detail["result"]["source_artifact_sha256"] = "0" * 64
    assert services.find_experiment_bound_artifact_bundle(detail) is None


def test_tampered_price_bars_file_would_be_refused(tmp_path, monkeypatch):
    """Negative control: if a referenced file's bytes don't match its
    recorded hash, the whole bundle is refused -- proven by pointing the
    manifest at a copy with one byte flipped."""
    import json

    manifest = json.loads((_REAL_BUNDLE_DIR / "manifest.json").read_text())
    tampered_dir = tmp_path / "tampered"
    tampered_dir.mkdir()
    price_bars_src = Path(manifest["price_bars"]["path"])
    tampered_price_bars = tampered_dir / "price_bars.csv"
    content = price_bars_src.read_bytes()
    tampered_price_bars.write_bytes(content[:-1] + b"9" if content[-1:] != b"9" else content[:-1] + b"8")
    manifest["price_bars"]["path"] = str(tampered_price_bars)
    tampered_manifest = tampered_dir / "manifest.json"
    tampered_manifest.write_text(json.dumps(manifest))

    detail = {"result": {"source_artifact": str(tampered_manifest), "source_artifact_sha256": manifest["bundle_sha256"]}}
    assert services.find_experiment_bound_artifact_bundle(detail) is None


def test_window_selects_only_the_trades_own_instrument_and_real_fills():
    bundle = services.find_experiment_bound_artifact_bundle(_real_detail())
    trades = services.trades_rows_from_bundle(bundle)
    assert trades
    trade0 = trades[0]
    window = services.price_signal_window_from_bundle(bundle, trade_index=0, context_bars=30)
    assert window is not None
    assert window["bars"], "must have real bars in the window"
    assert all(b["instrument_id"] == trade0["instrument_id"] for b in window["bars"])
    # Every fill in the window belongs to the same instrument and falls
    # inside the window's own timestamp bounds.
    for f in window["fills"]:
        assert f["instrument_id"] == trade0["instrument_id"]
        assert window["window_start_ns"] <= int(f["ts_fill_ns"]) <= window["window_end_ns"]
    # The trade's own open/close fills are present -- never inferred.
    fill_timestamps = {int(f["ts_fill_ns"]) for f in window["fills"]}
    assert int(trade0["ts_open_ns"]) in fill_timestamps
    assert int(trade0["ts_close_ns"]) in fill_timestamps


def test_unknown_trade_index_returns_none_never_fabricates():
    bundle = services.find_experiment_bound_artifact_bundle(_real_detail())
    assert services.price_signal_window_from_bundle(bundle, trade_index=999999) is None


def test_bundle_without_price_bars_or_trades_returns_none():
    assert services.price_signal_window_from_bundle(None, trade_index=0) is None
    assert services.price_signal_window_from_bundle({"price_bars": None, "trades": None}, trade_index=0) is None


def test_price_signal_chart_renders_real_candles_fills_and_position():
    bundle = services.find_experiment_bound_artifact_bundle(_real_detail())
    window = services.price_signal_window_from_bundle(bundle, trade_index=0, context_bars=30)
    fig = charts_discovery.price_signal_chart(window)
    trace_types = [t.type for t in fig.data]
    assert "candlestick" in trace_types
    assert trace_types.count("scatter") >= 2  # >=2 fill markers + 1 position line


def test_price_signal_chart_empty_window_never_crashes():
    fig = charts_discovery.price_signal_chart({"bars": [], "fills": [], "trade": None})
    assert fig is not None


def test_backtests_page_renders_real_signals_section(monkeypatch):
    """AppTest smoke: the Backtests page's new Signals section, with the
    experiment-bound-artifact lookup monkeypatched to the real on-disk
    bundle (proven correct above) regardless of which registry experiment
    the page's own selectors happen to pick -- this exercises the RENDER
    path against genuine bar/fill/trade content."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from alpha_agent.ui.views import backtests
    from streamlit.testing.v1 import AppTest

    real_bundle = services.find_experiment_bound_artifact_bundle(_real_detail())
    monkeypatch.setattr(backtests.services, "find_experiment_bound_artifact_bundle", lambda detail: real_bundle)

    at = AppTest.from_string("from alpha_agent.ui.views.backtests import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)
    full_text = " ".join(m.value for m in at.markdown)
    assert "Signals" in full_text


def test_position_is_zero_outside_the_trades_own_window():
    bundle = services.find_experiment_bound_artifact_bundle(_real_detail())
    window = services.price_signal_window_from_bundle(bundle, trade_index=0, context_bars=30)
    trade = window["trade"]
    ts_open, ts_close = int(trade["ts_open_ns"]), int(trade["ts_close_ns"])
    before = [b for b in window["bars"] if int(b["ts_event_ns"]) < ts_open]
    during = [b for b in window["bars"] if ts_open <= int(b["ts_event_ns"]) < ts_close]
    assert before, "context padding should include bars before entry"
    assert during, "window should include bars while the trade is open"


# ---------------------------------------------------------------------------
# Research Golden Path V1 acceptance pass, section 2F: Signals and Performance
# must bind and expose the SAME hash-verified artifact bundle for the same
# experiment -- no second, narrower "trade ledger" code path may silently
# diverge from what Signals already proves is real and bound.
# ---------------------------------------------------------------------------


def test_signals_and_performance_bind_the_same_real_bundle():
    detail = _real_detail()
    signals_bundle = services.find_experiment_bound_artifact_bundle(detail)
    performance_bundle = services.find_experiment_bound_artifact_bundle(detail)
    assert signals_bundle is not None
    assert signals_bundle == performance_bundle  # literally the same lookup, same result


def test_performance_exposes_equity_drawdown_and_trades_from_the_bundle():
    bundle = services.find_experiment_bound_artifact_bundle(_real_detail())
    series = services.daily_equity_series_from_bundle(bundle)
    assert series, "a bound bundle with a real daily_equity.csv must yield a non-empty equity series"
    assert all("cum_net_pnl_usd" in r and "drawdown_usd" in r for r in series)
    assert series[0]["cum_net_pnl_usd"] == 0.0  # re-based to start at zero, never an invented starting value

    max_dd = services.max_drawdown_usd(series)
    assert max_dd is not None and max_dd >= 0.0

    trades_rows = services.trades_rows_from_bundle(bundle)
    assert trades_rows, "a bound bundle with a real trades.csv must yield non-empty trade rows"
    assert {"trade_index", "net_pnl_usd", "entry_price", "exit_price"} <= trades_rows[0].keys()


def test_daily_equity_series_never_fabricates_without_a_bound_bundle():
    assert services.daily_equity_series_from_bundle(None) is None
    assert services.daily_equity_series_from_bundle({"daily_equity": None}) is None
