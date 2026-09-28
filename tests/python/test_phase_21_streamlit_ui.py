"""Phase 21 -- Paper Trading page (prompt
prompts/21_PAPER_TRADING_AND_DRIFT_MONITOR.md).

`alpha_agent.ui.services`'s Phase 21 additions are plain Python (no
`streamlit` import), tested unconditionally; the Streamlit rendering smoke
tests are gated behind `pytest.importorskip` (CLAUDE.md: "keep optional vendor
imports lazy").

Two states are proven against the REAL page:

1. today's real registry (0 authoritative PASS experiments) + no ledger runs
   -> the page renders the honest empty state, fabricating no metric, table,
   or number;
2. a seeded ledger fixture with one real run + step -> the page renders that
   run's REAL committed numbers (never a re-derived one).
"""
from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest
from alpha_agent.paper.ledger import PaperFillRow, PaperLedger, PaperStepRow
from alpha_agent.paper.risk_policy import PaperRiskPolicy
from alpha_agent.ui import services

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(),
    reason="Phase 14 registry sqlite not present in this checkout",
)

DAY_NS = 86_400_000_000_000


# ---------------------------------------------------------------------------
# services.py -- read-only boundary
# ---------------------------------------------------------------------------
def test_paper_services_module_never_calls_a_ledger_write_method():
    """Static regression guard, mirroring the Phase 20 registry-boundary test:
    the UI's Phase 21 read path must never call a PaperLedger write method."""
    src = services.__file__
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    forbidden = [
        r"\.create_run\(", r"\.append_step\(", r"\.append_fills\(",
        r"\.append_alerts\(", r"\.update_run_state\(",
        r"\.append_positions\(", r"\.commit_step\(",
    ]
    for pattern in forbidden:
        assert not re.search(pattern, text), f"ui/services.py must never call {pattern!r}"


def test_paper_trading_view_never_calls_a_ledger_write_method_directly():
    """Product UI Polish pass, section 11: the Paper Trading VIEW module may
    trigger a write (the "Start Paper Test" button), but only by calling
    `alpha_agent.ui.paper_actions.start_paper_run` -- it must never call a
    `PaperLedger`/`PaperTradingEngine` write method itself, mirroring the
    Phase 20/21 registry-boundary discipline `services.py` already has."""
    from alpha_agent.ui.views import paper_trading as pt_view

    with open(pt_view.__file__, encoding="utf-8") as fh:
        text = fh.read()
    forbidden = [
        r"\.create_run\(", r"\.append_step\(", r"\.append_fills\(",
        r"\.append_alerts\(", r"\.update_run_state\(",
        r"\.append_positions\(", r"\.commit_step\(", r"PaperTradingEngine\(",
    ]
    for pattern in forbidden:
        assert not re.search(pattern, text), f"paper_trading.py view must never call {pattern!r} directly"
    assert "paper_actions.start_paper_run(" in text


def test_paper_eligible_experiments_matches_real_registry_state():
    """Today's real registry has 0 authoritative PASS experiments (see
    phase-status notes); this must be reflected honestly, not hidden."""
    eligible = services.paper_eligible_experiments()
    assert eligible == []


def test_list_paper_runs_empty_when_no_ledger_seeded(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "PAPER_LEDGER_PATH", tmp_path / "does_not_exist.sqlite")
    assert services.list_paper_runs() == []
    assert services.paper_run_report("anything") is None


def test_list_paper_runs_and_report_reflect_a_seeded_run(tmp_path, monkeypatch):
    ledger_path = tmp_path / "paper_ledger.sqlite"
    ledger = PaperLedger(ledger_path)
    run_id = ledger.create_run(
        experiment_id="NQ__TEST__CANONICAL", experiment_identity="ident-1",
        strategy_family="mean_reversion", strategy_id="TEST", strategy_fingerprint="stratdsl1:x",
        root_symbol="NQ", params={"root_symbol": "NQ"},
        risk_policy=PaperRiskPolicy().model_dump(mode="json"), risk_policy_identity="paperrisk1:x",
        schedule_policy="no_decision", commission_per_contract_usd=0.0, slippage_ticks=0.0,
        spread_ticks=0.0, window_start_ns=0, window_end_ns=10 * DAY_NS,
        data_source="synthetic_test_fixture", validation_result={"backtest_daily_sharpe": 0.2},
    )
    ledger.append_step(
        PaperStepRow(
            run_id=run_id, step_ordinal=0, as_of_ts_ns=DAY_NS, created_at=datetime.now(UTC).isoformat(),
            bars_processed=5, orders_generated=1, fills_generated=1, trades_closed=0,
            gross_pnl_usd=0.0, costs_usd=0.0, net_pnl_usd=-1500.0, unrealized_pnl_usd=-1500.0,
            equity_usd=98500.0, peak_equity_usd=100000.0, drawdown_usd=1500.0, drawdown_pct=0.015,
            day_realized_pnl_usd=0.0, risk_rejects=0, risk_resizes=0, kill_switch_active=False,
            margin_complete=False, open_positions=1, daily_sharpe=None, annualized_sharpe=None,
            raw_result={"fills": 1},
        )
    )
    ledger.append_fills(
        [
            PaperFillRow(
                run_id=run_id, fill_seq=0, step_ordinal=0, fill_id=1, order_id=1, ts_fill_ns=DAY_NS,
                instrument_id=1, raw_symbol="NQZ6", side="buy", quantity=1, fill_price=20000.0,
                commission_usd=2.0, slippage_ticks=0.0,
            )
        ]
    )
    ledger.close()

    monkeypatch.setattr(services, "PAPER_LEDGER_PATH", ledger_path)
    runs = services.list_paper_runs()
    assert len(runs) == 1 and runs[0]["run_id"] == run_id

    report = services.paper_run_report(run_id)
    assert report is not None
    assert report["latest_step"]["equity_usd"] == 98500.0
    assert report["n_fills"] == 1
    assert report["fills"][0]["fill_price"] == 20000.0


# ---------------------------------------------------------------------------
# Streamlit rendering (skipped without the `ui` extra installed)
# ---------------------------------------------------------------------------
def test_paper_trading_page_renders_the_honest_empty_state(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    # Phase 21.1b: PaperLedger.__init__ now migrates whatever database it
    # opens. Point this test at an isolated, non-existent path so a real
    # local data/paper_trading/paper_ledger.sqlite (if one exists on this
    # machine) is never touched by running the test suite -- this test's
    # empty-state assertion is a fact about "no ledger seeded", not about
    # this machine's real ledger.
    monkeypatch.setattr(services, "PAPER_LEDGER_PATH", tmp_path / "does_not_exist.sqlite")

    at = AppTest.from_string("from alpha_agent.ui.views.paper_trading import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)
    full_text = " ".join(m.value for m in at.markdown)
    # Product UI Polish pass, section 10: a clean empty state, not a warning
    # wall -- real registry state today is 0 eligible, so this is the honest
    # branch, not a fabricated placeholder.
    assert "No active paper test" in full_text
    assert "No strategy currently satisfies the scientific eligibility policy" in full_text
    assert at.button(key="pt-empty-open-research").label == "Open Research"


def test_paper_trading_page_renders_real_seeded_run_data(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    ledger_path = tmp_path / "paper_ledger.sqlite"
    ledger = PaperLedger(ledger_path)
    run_id = ledger.create_run(
        experiment_id="NQ__TEST__CANONICAL", experiment_identity="ident-1",
        strategy_family="mean_reversion", strategy_id="TEST", strategy_fingerprint="stratdsl1:x",
        root_symbol="NQ", params={"root_symbol": "NQ"},
        risk_policy=PaperRiskPolicy().model_dump(mode="json"), risk_policy_identity="paperrisk1:x",
        schedule_policy="no_decision", commission_per_contract_usd=0.0, slippage_ticks=0.0,
        spread_ticks=0.0, window_start_ns=0, window_end_ns=10 * DAY_NS,
        data_source="synthetic_test_fixture", validation_result={"backtest_daily_sharpe": 0.2},
    )
    ledger.append_step(
        PaperStepRow(
            run_id=run_id, step_ordinal=0, as_of_ts_ns=DAY_NS, created_at=datetime.now(UTC).isoformat(),
            bars_processed=5, orders_generated=1, fills_generated=1, trades_closed=0,
            gross_pnl_usd=0.0, costs_usd=0.0, net_pnl_usd=-1500.0, unrealized_pnl_usd=-1500.0,
            equity_usd=98500.0, peak_equity_usd=100000.0, drawdown_usd=1500.0, drawdown_pct=0.015,
            day_realized_pnl_usd=0.0, risk_rejects=0, risk_resizes=0, kill_switch_active=False,
            margin_complete=False, open_positions=1, daily_sharpe=None, annualized_sharpe=None,
            raw_result={"fills": 1},
        )
    )
    ledger.close()

    monkeypatch.setattr(services, "PAPER_LEDGER_PATH", ledger_path)
    at = AppTest.from_string("from alpha_agent.ui.views.paper_trading import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)
    assert len(at.dataframe) >= 1  # the Steps tab's real dataframe
    full_text = " ".join(m.value for m in at.markdown)
    assert "$98,500" in full_text


# ---------------------------------------------------------------------------
# Phase 21.1 -- positions / drift / provenance tabs
# ---------------------------------------------------------------------------
def _seed_full_step(ledger_path, *, with_position: bool) -> str:
    from alpha_agent.paper.ledger import PaperPositionRow

    ledger = PaperLedger(ledger_path)
    run_id = ledger.create_run(
        experiment_id="NQ__TEST__CANONICAL", experiment_identity="ident-1",
        strategy_family="mean_reversion", strategy_id="TEST", strategy_fingerprint="stratdsl1:x",
        root_symbol="NQ", params={"root_symbol": "NQ"},
        risk_policy=PaperRiskPolicy().model_dump(mode="json"), risk_policy_identity="paperrisk1:x",
        schedule_policy="no_decision", commission_per_contract_usd=0.0, slippage_ticks=0.0,
        spread_ticks=0.0, window_start_ns=0, window_end_ns=10 * DAY_NS,
        data_source="synthetic_test_fixture", validation_result={"backtest_daily_sharpe": 0.2},
    )
    provenance = {
        "schema_version": "paper-step-provenance/1", "parent_experiment_identity": "ident-1",
        "strategy_fingerprint": "stratdsl1:x", "risk_policy_identity": "paperrisk1:x",
        "schedule_policy": "no_decision", "commission_per_contract_usd": 0.0, "slippage_ticks": 0.0,
        "spread_ticks": 0.0, "end_of_test": "leave_open", "bars_sha256": "a" * 64,
        "contracts_sha256": "b" * 64, "targets_sha256": "c" * 64,
        "target_schedule_hash": "targsched1:x", "validation_days_sha256": "d" * 64,
        "validation_day_plan_identity": "valdayplan1:x", "roll_close_marks_sha256": None,
        "executable_path": "build/cpp/cpp/quant_paper_trading_targets_csv",
        "executable_sha256": "e" * 64, "result_payload_sha256": "f" * 64,
    }
    drift = {
        "schema_version": "paper-drift-report/2", "run_id": run_id, "as_of_ts_ns": DAY_NS,
        "backtest_daily_sharpe": 0.2, "backtest_annualized_sharpe": 3.0,
        "backtest_net_pnl_usd": 500.0, "backtest_n_trades": 5,
        "paper_daily_sharpe": None, "paper_annualized_sharpe": None, "paper_net_pnl_usd": -1500.0,
        "paper_n_trades": 0, "paper_n_fills": 1, "orders_generated": 1, "fill_rate": 1.0,
        "configured_commission_per_contract_usd": 0.0, "configured_slippage_ticks": 0.0,
        "configured_spread_ticks": 0.0,
        "realized_slippage": {"status": "NOT_AVAILABLE", "n_fills": 0, "mean_slippage_ticks": None,
                              "mean_slippage_usd": None, "total_slippage_usd": None,
                              "n_fills_missing_contract_spec": 0,
                              "backtest_comparison_status": "NOT_AVAILABLE",
                              "backtest_mean_slippage_ticks": None},
        "trade_frequency": {"status": "AVAILABLE", "n_trades": 0, "elapsed_trading_days": 1,
                            "trades_per_day": 0.0, "backtest_comparison_status": "NOT_AVAILABLE",
                            "backtest_trades_per_day": None},
        "pnl_distribution": {"status": "NOT_AVAILABLE", "n_days": 0, "mean_daily_pnl_usd": None,
                             "std_daily_pnl_usd": None, "min_daily_pnl_usd": None,
                             "max_daily_pnl_usd": None, "n_positive_days": 0, "n_negative_days": 0,
                             "backtest_comparison_status": "NOT_AVAILABLE"},
        "comparison_availability": {"daily_sharpe": "AVAILABLE", "net_pnl_usd": "AVAILABLE",
                                    "n_trades": "AVAILABLE", "fill_rate": "NOT_AVAILABLE",
                                    "slippage": "NOT_AVAILABLE", "trade_frequency": "NOT_AVAILABLE",
                                    "pnl_distribution": "NOT_AVAILABLE"},
        "flags": ["PAPER_HAS_NO_CLOSED_TRADES_YET"],
    }
    ledger.append_step(
        PaperStepRow(
            run_id=run_id, step_ordinal=0, as_of_ts_ns=DAY_NS, created_at=datetime.now(UTC).isoformat(),
            bars_processed=5, orders_generated=1, fills_generated=1, trades_closed=0,
            gross_pnl_usd=0.0, costs_usd=0.0, net_pnl_usd=-1500.0, unrealized_pnl_usd=-1500.0,
            equity_usd=98500.0, peak_equity_usd=100000.0, drawdown_usd=1500.0, drawdown_pct=0.015,
            day_realized_pnl_usd=0.0, risk_rejects=0, risk_resizes=0, kill_switch_active=False,
            margin_complete=False, open_positions=1 if with_position else 0,
            elapsed_trading_days=1, daily_sharpe=None, annualized_sharpe=None,
            provenance=provenance, drift=drift, raw_result={"fills": 1},
        )
    )
    if with_position:
        ledger.append_positions(
            [
                PaperPositionRow(
                    run_id=run_id, step_ordinal=0, instrument_id=1, raw_symbol="NQZ6",
                    root_symbol="NQ", units=1, avg_entry_price=20000.0, multiplier=20.0,
                    mark_price=18500.0, mark_ts_ns=DAY_NS, mark_age_ns=0, mark_present=True,
                    mark_is_stale=False, valuation_is_estimated=False, gross_notional_usd=370000.0,
                    signed_notional_usd=370000.0, unrealized_pnl_usd=-30000.0,
                    initial_margin_usd=0.0, maintenance_margin_usd=0.0, margin_known=False,
                )
            ]
        )
    ledger.close()
    return run_id


def test_paper_run_report_includes_latest_positions_and_drift_and_provenance(tmp_path):
    run_id = _seed_full_step(tmp_path / "ledger.sqlite", with_position=True)
    ledger = PaperLedger(tmp_path / "ledger.sqlite")
    from alpha_agent.paper.report import build_paper_run_report

    report = build_paper_run_report(ledger, run_id)
    ledger.close()

    assert len(report["latest_positions"]) == 1
    assert report["latest_positions"][0]["raw_symbol"] == "NQZ6"
    assert report["latest_positions"][0]["unrealized_pnl_usd"] == -30000.0
    assert report["latest_step"]["provenance"]["parent_experiment_identity"] == "ident-1"
    assert report["latest_step"]["drift"]["schema_version"] == "paper-drift-report/2"


def test_paper_trading_page_renders_positions_execution_risk_monitoring_tabs(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    ledger_path = tmp_path / "ledger.sqlite"
    _seed_full_step(ledger_path, with_position=True)

    monkeypatch.setattr(services, "PAPER_LEDGER_PATH", ledger_path)
    at = AppTest.from_string("from alpha_agent.ui.views.paper_trading import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)

    tab_labels = [t.proto.label for t in at.tabs]
    assert {"Positions", "Execution", "Risk", "Monitoring"} <= set(tab_labels)

    # the position's raw_symbol / instrument identity is shown ONLY via the
    # committed C++ snapshot (never reconstructed) -- present somewhere on the page.
    all_dataframes_text = " ".join(str(df.value) for df in at.dataframe)
    assert "NQZ6" in all_dataframes_text
    warning_text = " ".join(w.body for w in at.warning)
    assert "PAPER_HAS_NO_CLOSED_TRADES_YET" in warning_text
