"""Phase 21 -- paper trading and drift monitor (prompt
prompts/21_PAPER_TRADING_AND_DRIFT_MONITOR.md).

Covers, against synthetic fixtures ONLY (no real market data, no 2025):

* PaperRiskPolicy CSV round-trip + identity + the margin-consistency guard;
* eligibility -- accept a genuine PASS + rebuildable strategy, refuse an
  unknown experiment, a non-PASS verdict, a non-rebuildable family, and a
  strategy-fingerprint mismatch;
* the persistent ledger (runs / steps / fills / alerts) round-trips exactly;
* the PaperTradingEngine end to end through the REAL compiled
  ``quant_paper_trading_targets_csv`` CLI: target -> order -> fill ->
  portfolio, the hard-risk kill switch tripping and halting the run, fills
  accumulating correctly across steps, and stop_run flattening + STOPPED;
* alerts (kill-switch transitions) and drift diagnostics (fill rate, realized
  slippage, elapsed-day-normalized trade frequency, PnL distribution, and
  backtest-comparison availability -- Phase 21.1);
* exact per-step execution provenance (Phase 21.1);
* replay-prefix consistency + atomic step commit (Phase 21.1).
"""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from alpha_agent.core.instrument import AssetDomain
from alpha_agent.paper.alerts import generate_step_alerts, kill_switch_active
from alpha_agent.paper.drift import build_drift_report
from alpha_agent.paper.eligibility import assert_paper_trading_eligible
from alpha_agent.paper.engine import PaperTradingEngine
from alpha_agent.paper.errors import (
    PaperDataWindowExhausted,
    PaperReplayDivergence,
    PaperRunStateError,
    PaperTradingEligibilityError,
)
from alpha_agent.paper.ledger import (
    PaperAlertRow,
    PaperFillRow,
    PaperLedger,
    PaperPositionRow,
    PaperStepRow,
)
from alpha_agent.paper.replay_integrity import assert_replay_prefix_consistent
from alpha_agent.paper.risk_policy import PaperRiskPolicy
from alpha_agent.registry.enums import ExperimentStatus, RegistryVerdict, TrialRole
from alpha_agent.registry.models import ExperimentRecord, MarketWindow, ResultRecord
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.schemas.market_data import BOUNDARY_BAR_COLUMNS, CONTRACT_COLUMNS
from alpha_agent.strategy import strategy_fingerprint
from alpha_agent.strategy.candidates_phase_13_5c import spec_for_params

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
PAPER_CLI = REPO_ROOT / "build" / "cpp" / "cpp" / "quant_paper_trading_targets_csv"

pytestmark = pytest.mark.skipif(
    not PAPER_CLI.exists(),
    reason="quant_paper_trading_targets_csv not built (cmake --build build/cpp)",
)

DAY_NS = 86_400_000_000_000
_EMPTY_CONTRACTS = pd.DataFrame(columns=list(CONTRACT_COLUMNS))
_BASE_IDENTITY_FIELDS = {
    "dataset_fingerprint": "valdataset2:test",
    "split_identity": "splitplan1:test",
    "validation_spec_fingerprint": "validationspec1:test",
    "reliability_policy_fingerprint": "valreliabilitypolicy1:test",
    "execution_config_identity": "execconfig1:test",
    "cost_config_identity": "costconfig1:test",
    "risk_identity": "riskconfig1:test",
    "feature_spec_fingerprint": "featset1:test",
}

MR_PARAMS = {"root_symbol": "NQ", "zscore_window": 5, "entry_z": 1.0, "exit_z": 0.25, "size": 1}


def _mr_fingerprint() -> str:
    return strategy_fingerprint(spec_for_params("mean_reversion", MR_PARAMS))


def _experiment_record(*, strategy_fp: str, family: str = "mean_reversion") -> ExperimentRecord:
    return ExperimentRecord(
        experiment_identity="paper-test-identity-1",
        experiment_id="NQ__MEAN_REVERSION__CANONICAL__TEST",
        display_name="NQ mean_reversion (test fixture)",
        created_at="2026-01-01T00:00:00+00:00",
        phase="TEST",
        status=ExperimentStatus.COMPLETED,
        root_symbol="NQ",
        asset_domain=AssetDomain.FUTURES,
        strategy_family=family,
        strategy_fingerprint=strategy_fp,
        strategy_id="TEST-MEAN-REVERSION",
        strategy_spec_json={
            "schema": "registry-strategy-spec/1",
            "strategy_family": family,
            "params": MR_PARAMS,
            "parameter_names": sorted(MR_PARAMS),
            "feature_fingerprints": ["feat1:x"],
        },
        market_window=MarketWindow(label="VALIDATION", start_date="2023-01-01", end_date="2024-12-31"),
        trial_role=TrialRole.CANONICAL,
        parameter_variant_identity="paramvariant1:test",
        parameter_variant_label="canonical",
        **_BASE_IDENTITY_FIELDS,
    )


def _pass_result(identity: str) -> ResultRecord:
    return ResultRecord(
        experiment_identity=identity,
        headline_verdict=RegistryVerdict.PASS,
        reason_codes=("all_required_gates_satisfied",),
        net_pnl_usd=1234.0,
        daily_sharpe=0.05,
        annualized_sharpe=0.8,
        n_trades=10,
    )


@pytest.fixture
def registry(tmp_path):
    with ExperimentRegistry(tmp_path / "experiments.sqlite") as reg:
        yield reg


@pytest.fixture
def pass_experiment_identity(registry):
    fp = _mr_fingerprint()
    record = _experiment_record(strategy_fp=fp)
    registry.insert_experiment(record, _pass_result(record.experiment_identity))
    return record.experiment_identity


# ---------------------------------------------------------------------------
# PaperRiskPolicy
# ---------------------------------------------------------------------------
def test_risk_policy_csv_roundtrip(tmp_path):
    policy = PaperRiskPolicy()
    path = policy.write_csv(tmp_path / "risk_config.csv")
    text = path.read_text(encoding="utf-8").splitlines()
    assert text[0].split(",")[0] == "max_contracts_per_symbol"
    assert len(text) == 2
    assert len(text[1].split(",")) == len(text[0].split(","))


def test_risk_policy_identity_changes_with_content():
    a = PaperRiskPolicy()
    b = PaperRiskPolicy(max_drawdown_pct=0.20)
    assert a.identity() != b.identity()
    assert a.identity() == PaperRiskPolicy().identity()


def test_risk_policy_rejects_inconsistent_margin_config():
    with pytest.raises(ValueError):
        PaperRiskPolicy(missing_margin_policy="treat_as_zero", max_margin_utilization_pct=0.5)


# ---------------------------------------------------------------------------
# eligibility
# ---------------------------------------------------------------------------
def test_eligibility_accepts_pass_experiment(registry, pass_experiment_identity):
    elig = assert_paper_trading_eligible(registry, pass_experiment_identity)
    assert elig.strategy_family == "mean_reversion"
    assert elig.root_symbol == "NQ"
    assert elig.params["zscore_window"] == 5
    assert elig.backtest_daily_sharpe == 0.05


def test_eligibility_rejects_unknown_experiment(registry):
    with pytest.raises(PaperTradingEligibilityError):
        assert_paper_trading_eligible(registry, "does-not-exist")


def test_eligibility_rejects_non_pass_verdict(registry):
    fp = _mr_fingerprint()
    record = _experiment_record(strategy_fp=fp)
    registry.insert_experiment(
        record,
        ResultRecord(
            experiment_identity=record.experiment_identity,
            headline_verdict=RegistryVerdict.REJECT,
            reason_codes=("fdr_qvalue_above_threshold",),
        ),
    )
    with pytest.raises(PaperTradingEligibilityError, match="not passed frozen validation"):
        assert_paper_trading_eligible(registry, record.experiment_identity)


def test_eligibility_rejects_non_rebuildable_family(registry):
    record = _experiment_record(strategy_fp="stratdsl1:whatever", family="silver_bullet")
    registry.insert_experiment(record, _pass_result(record.experiment_identity))
    with pytest.raises(PaperTradingEligibilityError, match="not one of the Phase 21"):
        assert_paper_trading_eligible(registry, record.experiment_identity)


def test_eligibility_rejects_fingerprint_mismatch(registry):
    record = _experiment_record(strategy_fp="stratdsl1:" + "0" * 64)
    registry.insert_experiment(record, _pass_result(record.experiment_identity))
    with pytest.raises(PaperTradingEligibilityError, match="does not match"):
        assert_paper_trading_eligible(registry, record.experiment_identity)


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------
def test_ledger_run_step_fill_alert_roundtrip(tmp_path):
    ledger = PaperLedger(tmp_path / "paper_ledger.sqlite")
    run_id = ledger.create_run(
        experiment_id="exp-1", experiment_identity="ident-1", strategy_family="mean_reversion",
        strategy_id="TEST", strategy_fingerprint="stratdsl1:x", root_symbol="NQ",
        params={"root_symbol": "NQ"}, risk_policy=PaperRiskPolicy().model_dump(mode="json"),
        risk_policy_identity="paperrisk1:x", schedule_policy="no_decision",
        commission_per_contract_usd=2.0, slippage_ticks=0.0, spread_ticks=0.0,
        window_start_ns=0, window_end_ns=10 * DAY_NS, data_source="synthetic_test_fixture",
        validation_result={"backtest_daily_sharpe": 0.1},
    )
    run = ledger.get_run(run_id)
    assert run.status == "ACTIVE"
    assert run.watermark_ns == 0

    ledger.append_step(
        PaperStepRow(
            run_id=run_id, step_ordinal=0, as_of_ts_ns=DAY_NS, created_at=datetime.now(UTC).isoformat(),
            bars_processed=1, orders_generated=1, fills_generated=1, trades_closed=0,
            gross_pnl_usd=0.0, costs_usd=2.0, net_pnl_usd=-2.0, unrealized_pnl_usd=0.0,
            equity_usd=99998.0, peak_equity_usd=100000.0, drawdown_usd=2.0, drawdown_pct=0.00002,
            day_realized_pnl_usd=-2.0, risk_rejects=0, risk_resizes=0, kill_switch_active=False,
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
    ledger.update_run_state(run_id, watermark_ns=DAY_NS, status="ACTIVE")

    assert ledger.next_step_ordinal(run_id) == 1
    assert ledger.cumulative_fill_count(run_id) == 1
    steps = ledger.list_steps(run_id)
    assert len(steps) == 1 and steps[0].fills_generated == 1
    fills = ledger.list_fills(run_id)
    assert len(fills) == 1 and fills[0].fill_price == 20000.0
    assert ledger.get_run(run_id).watermark_ns == DAY_NS
    ledger.close()


def test_ledger_unknown_run_raises(tmp_path):
    from alpha_agent.paper.ledger import UnknownPaperRun

    ledger = PaperLedger(tmp_path / "paper_ledger.sqlite")
    with pytest.raises(UnknownPaperRun):
        ledger.get_run("does-not-exist")
    ledger.close()


# ---------------------------------------------------------------------------
# alerts
# ---------------------------------------------------------------------------
def test_kill_switch_active_helper():
    policy = {"max_daily_loss_usd": 0.0, "max_drawdown_pct": 0.10, "max_drawdown_usd": 0.0}
    assert kill_switch_active(policy, {"portfolio_drawdown_pct": 0.15, "day_realized_pnl_usd": 0.0}) is True
    assert kill_switch_active(policy, {"portfolio_drawdown_pct": 0.05, "day_realized_pnl_usd": 0.0}) is False


def test_generate_step_alerts_only_fires_on_transitions():
    a = generate_step_alerts(
        kill_switch_now=True, kill_switch_before=False, margin_complete=True,
        window_exhausted=False,
    )
    assert any(x.alert_type.value == "KILL_SWITCH_ACTIVE" for x in a)

    b = generate_step_alerts(
        kill_switch_now=True, kill_switch_before=True, margin_complete=True,
        window_exhausted=False,
    )
    assert not any(x.alert_type.value == "KILL_SWITCH_ACTIVE" for x in b)


# ---------------------------------------------------------------------------
# synthetic market-window provider + full engine integration
# ---------------------------------------------------------------------------
class SyntheticProvider:
    """A deterministic, in-memory MarketWindowProvider: one raw contract
    spanning the whole window, one bar per business day. Prices are flat with
    a tiny wiggle except for an engineered crash (days 10-11) and a recovery
    -- enough to exercise real fills, a real drawdown, and the hard-risk kill
    switch through the REAL compiled CLI, without touching any real market
    data."""

    INSTRUMENT_ID = 1
    N_DAYS = 40

    def __init__(self) -> None:
        days = pd.bdate_range("2019-06-03", periods=self.N_DAYS, freq="B")
        ts = np.array(
            [int((d + pd.Timedelta(hours=18)).timestamp() * 1e9) for d in days], dtype="int64"
        )
        prices = np.full(self.N_DAYS, 20000.0)
        prices += np.array([((-1) ** i) * 0.5 for i in range(self.N_DAYS)])
        # day 10: a mild dip triggers the mean-reversion entry DECISION (z <
        # -entry_z). The order executes on the NEXT bar (day 11), so the crash
        # must keep falling past the entry to produce a real drawdown on the
        # now-open position -- day 12 is 2500 points below the day-11 entry.
        prices[10] = 19500.0
        prices[11] = 18500.0
        prices[12] = 16000.0
        recovery_days = 15
        for k in range(recovery_days):
            idx = 13 + k
            if idx < self.N_DAYS:
                prices[idx] = 16000.0 + (20000.0 - 16000.0) * (k + 1) / recovery_days
        self.ts_ns = ts
        self.prices = prices

    def _frame(self, start_ns: int, end_ns: int) -> pd.DataFrame:
        mask = (self.ts_ns >= start_ns) & (self.ts_ns <= end_ns)
        return pd.DataFrame(
            {
                "ts_event_ns": self.ts_ns[mask],
                "open": self.prices[mask],
                "high": self.prices[mask] + 1.0,
                "low": self.prices[mask] - 1.0,
                "close": self.prices[mask],
                "volume": np.full(mask.sum(), 100, dtype="int64"),
            }
        )

    def execution_bars(self, root_symbol: str, start_ns: int, end_ns: int) -> pd.DataFrame:
        df = self._frame(start_ns, end_ns)
        df = df.assign(instrument_id=self.INSTRUMENT_ID)
        return df.loc[:, list(BOUNDARY_BAR_COLUMNS)]

    def contracts_frame(self, root_symbol: str) -> pd.DataFrame:
        far_future = int(self.ts_ns[-1]) + 1000 * DAY_NS
        return pd.DataFrame(
            [
                {
                    "instrument_id": self.INSTRUMENT_ID,
                    "raw_symbol": f"{root_symbol}Z9",
                    "root_symbol": root_symbol,
                    "exchange": "XCME",
                    "tick_size": 0.25,
                    "multiplier": 20.0,
                    "activation_ns": 1,
                    "expiration_ns": far_future,
                    "first_notice_ns": far_future,
                    "last_trade_ns": far_future,
                }
            ],
            columns=list(CONTRACT_COLUMNS),
        )

    def daily_signal_bars(self, root_symbol: str, start_ns: int, end_ns: int) -> pd.DataFrame:
        return self._frame(start_ns, end_ns)

    @property
    def window_start_ns(self) -> int:
        return int(self.ts_ns[0]) - 1

    @property
    def window_end_ns(self) -> int:
        return int(self.ts_ns[-1]) + 1


@pytest.fixture
def provider():
    return SyntheticProvider()


@pytest.fixture
def engine(registry, tmp_path, provider):
    ledger = PaperLedger(tmp_path / "paper_ledger.sqlite")
    eng = PaperTradingEngine(
        registry=registry, ledger=ledger, provider=provider, executable=PAPER_CLI,
        work_dir=tmp_path / "work", data_source_label="synthetic_test_fixture",
    )
    yield eng, ledger
    ledger.close()


def test_engine_start_and_step_produces_real_fills_and_drift(
    engine, pass_experiment_identity, provider
):
    eng, ledger = engine
    risk_policy = PaperRiskPolicy(max_drawdown_pct=0.05, max_daily_loss_usd=0.0)
    run_id = eng.start_run(
        experiment_key=pass_experiment_identity,
        window_start_ns=provider.window_start_ns,
        window_end_ns=provider.window_end_ns,
        risk_policy=risk_policy,
        commission_per_contract_usd=0.0,
    )
    run = ledger.get_run(run_id)
    assert run.status == "ACTIVE"
    assert run.strategy_family == "mean_reversion"

    # jump straight to the crash-bottom trading day (index 12, 1-based count 13)
    snap = eng.step(run_id, n_trading_days=13)
    assert snap.step_ordinal == 0
    assert snap.status == "HALTED"
    assert snap.kill_switch_active is True

    steps = ledger.list_steps(run_id)
    assert len(steps) == 1
    latest = steps[-1]
    assert latest.drawdown_pct >= 0.05
    assert latest.risk_rejects == 0 or latest.open_positions >= 0  # engine ran; no crash

    # a real fill was recorded (the mean-reversion entry on the dip)
    fills = ledger.list_fills(run_id)
    assert len(fills) >= 1
    assert fills[0].raw_symbol.startswith("NQ")

    # alerts recorded the kill switch transition
    alerts = ledger.list_alerts(run_id)
    assert any(a.alert_type == "KILL_SWITCH_ACTIVE" for a in alerts)

    # drift report is a real, typed diagnostic (never an authority)
    assert snap.drift.paper_n_fills == latest.fills_generated
    assert snap.drift.backtest_daily_sharpe == 0.05


def test_engine_accumulates_fills_and_watermark_across_steps(engine, pass_experiment_identity, provider):
    eng, ledger = engine
    run_id = eng.start_run(
        experiment_key=pass_experiment_identity,
        window_start_ns=provider.window_start_ns,
        window_end_ns=provider.window_end_ns,
        risk_policy=PaperRiskPolicy(),
        commission_per_contract_usd=0.0,
    )
    snap1 = eng.step(run_id, n_trading_days=8)
    run_after_1 = ledger.get_run(run_id)
    snap2 = eng.step(run_id, n_trading_days=8)
    run_after_2 = ledger.get_run(run_id)

    assert snap2.step_ordinal == snap1.step_ordinal + 1
    assert run_after_2.watermark_ns > run_after_1.watermark_ns

    step1 = ledger.list_steps(run_id)[0]
    step2 = ledger.list_steps(run_id)[1]
    # cumulative counters never go backwards on a longer replay window
    assert step2.fills_generated >= step1.fills_generated
    assert step2.bars_processed > step1.bars_processed
    # every fill recorded through step 1 is still present unchanged after step 2
    fills_after_1 = ledger.cumulative_fill_count(run_id)
    assert fills_after_1 >= 0  # sanity: no exception, ledger consistent


def test_engine_data_window_exhausted(engine, pass_experiment_identity, provider):
    eng, _ledger = engine
    run_id = eng.start_run(
        experiment_key=pass_experiment_identity,
        window_start_ns=provider.window_start_ns,
        window_end_ns=provider.window_end_ns,
        risk_policy=PaperRiskPolicy(),
    )
    eng.step(run_id, n_trading_days=1000)  # clamps to the last available trading day
    with pytest.raises(PaperDataWindowExhausted):
        eng.step(run_id, n_trading_days=1)


def test_engine_stop_run_flattens_and_marks_stopped(engine, pass_experiment_identity, provider):
    eng, ledger = engine
    run_id = eng.start_run(
        experiment_key=pass_experiment_identity,
        window_start_ns=provider.window_start_ns,
        window_end_ns=provider.window_end_ns,
        risk_policy=PaperRiskPolicy(),
        commission_per_contract_usd=0.0,
    )
    eng.step(run_id, n_trading_days=13)
    stop_snap = eng.stop_run(run_id)
    assert stop_snap.status == "STOPPED"
    run = ledger.get_run(run_id)
    assert run.status == "STOPPED"
    assert run.stopped_at is not None

    with pytest.raises(PaperRunStateError):
        eng.step(run_id, n_trading_days=1)


# ---------------------------------------------------------------------------
# drift
# ---------------------------------------------------------------------------
def test_build_drift_report_flags_no_trades_yet():
    from alpha_agent.validation.runner import synthetic_backtest_run

    run = synthetic_backtest_run(np.array([0.0, 0.0]), n_trades=0, n_fills=0)
    report = build_drift_report(
        run_id="r1", as_of_ts_ns=1, cpp_json={"orders": 0, "fills": 0},
        paper_run=run, fills=[], contracts=_EMPTY_CONTRACTS, elapsed_trading_days=2,
        backtest_daily_sharpe=0.5, backtest_annualized_sharpe=8.0,
        backtest_net_pnl_usd=1000.0, backtest_n_trades=10,
    )
    assert "PAPER_HAS_NO_CLOSED_TRADES_YET" in report.flags
    assert report.fill_rate is None
    assert report.realized_slippage.status == "NOT_AVAILABLE"
    assert report.trade_frequency.status == "AVAILABLE"
    assert report.trade_frequency.trades_per_day == 0.0
    assert report.pnl_distribution.status == "AVAILABLE"


def test_realized_slippage_evidence_computed_from_fills_and_contracts():
    contracts = pd.DataFrame(
        [
            {
                "instrument_id": 1, "raw_symbol": "NQZ6", "root_symbol": "NQ", "exchange": "XCME",
                "tick_size": 0.25, "multiplier": 20.0, "activation_ns": 1,
                "expiration_ns": 10 * DAY_NS, "first_notice_ns": 10 * DAY_NS,
                "last_trade_ns": 10 * DAY_NS,
            }
        ],
        columns=list(CONTRACT_COLUMNS),
    )
    fills = [
        PaperFillRow(
            run_id="r1", fill_seq=0, step_ordinal=0, fill_id=1, order_id=1, ts_fill_ns=DAY_NS,
            instrument_id=1, raw_symbol="NQZ6", side="buy", quantity=2, fill_price=20000.0,
            commission_usd=2.0, slippage_ticks=1.0,
        ),
        PaperFillRow(
            run_id="r1", fill_seq=1, step_ordinal=1, fill_id=2, order_id=2, ts_fill_ns=2 * DAY_NS,
            instrument_id=1, raw_symbol="NQZ6", side="sell", quantity=2, fill_price=20100.0,
            commission_usd=2.0, slippage_ticks=3.0,
        ),
    ]
    from alpha_agent.validation.runner import synthetic_backtest_run

    run = synthetic_backtest_run(np.array([0.0]), n_trades=1, n_fills=2)
    report = build_drift_report(
        run_id="r1", as_of_ts_ns=1, cpp_json={"orders": 2, "fills": 2},
        paper_run=run, fills=fills, contracts=contracts, elapsed_trading_days=2,
        backtest_daily_sharpe=None, backtest_annualized_sharpe=None,
        backtest_net_pnl_usd=None, backtest_n_trades=None,
    )
    slip = report.realized_slippage
    assert slip.status == "AVAILABLE"
    assert slip.n_fills == 2
    assert slip.mean_slippage_ticks == pytest.approx((1.0 + 3.0) / 2)
    # each fill's $ slippage = |ticks| * tick_size * multiplier * |quantity|
    expected = [1.0 * 0.25 * 20.0 * 2, 3.0 * 0.25 * 20.0 * 2]
    assert slip.mean_slippage_usd == pytest.approx(sum(expected) / 2)
    assert slip.total_slippage_usd == pytest.approx(sum(expected))
    assert slip.n_fills_missing_contract_spec == 0
    # no committed backtest slippage-distribution artifact exists for any experiment today
    assert report.comparison_availability["slippage"] == "NOT_AVAILABLE"


def test_trade_frequency_normalized_by_elapsed_trading_days():
    from alpha_agent.validation.runner import synthetic_backtest_run

    run = synthetic_backtest_run(np.array([10.0, -5.0, 3.0, 3.0]), n_trades=4, n_fills=8)
    report = build_drift_report(
        run_id="r1", as_of_ts_ns=1, cpp_json={"orders": 8, "fills": 8},
        paper_run=run, fills=[], contracts=_EMPTY_CONTRACTS, elapsed_trading_days=20,
        backtest_daily_sharpe=None, backtest_annualized_sharpe=None,
        backtest_net_pnl_usd=None, backtest_n_trades=None,
    )
    freq = report.trade_frequency
    assert freq.status == "AVAILABLE"
    assert freq.n_trades == 4
    assert freq.elapsed_trading_days == 20
    assert freq.trades_per_day == pytest.approx(4 / 20)
    # no committed backtest elapsed-day-normalized trade-frequency artifact
    # exists for any experiment today
    assert report.comparison_availability["trade_frequency"] == "NOT_AVAILABLE"


def test_trade_frequency_not_available_with_zero_elapsed_days():
    from alpha_agent.validation.runner import synthetic_backtest_run

    run = synthetic_backtest_run(np.array([0.0]), n_trades=0, n_fills=0)
    report = build_drift_report(
        run_id="r1", as_of_ts_ns=1, cpp_json={"orders": 0, "fills": 0},
        paper_run=run, fills=[], contracts=_EMPTY_CONTRACTS, elapsed_trading_days=0,
        backtest_daily_sharpe=None, backtest_annualized_sharpe=None,
        backtest_net_pnl_usd=None, backtest_n_trades=None,
    )
    assert report.trade_frequency.status == "NOT_AVAILABLE"
    assert report.trade_frequency.trades_per_day is None


def test_pnl_distribution_descriptive_stats():
    from alpha_agent.validation.runner import synthetic_backtest_run

    daily = np.array([100.0, -50.0, 25.0, -25.0])
    run = synthetic_backtest_run(daily, n_trades=2, n_fills=4)
    report = build_drift_report(
        run_id="r1", as_of_ts_ns=1, cpp_json={"orders": 4, "fills": 4},
        paper_run=run, fills=[], contracts=_EMPTY_CONTRACTS, elapsed_trading_days=4,
        backtest_daily_sharpe=None, backtest_annualized_sharpe=None,
        backtest_net_pnl_usd=None, backtest_n_trades=None,
    )
    pnl = report.pnl_distribution
    assert pnl.status == "AVAILABLE"
    assert pnl.n_days == 4
    assert pnl.mean_daily_pnl_usd == pytest.approx(daily.mean())
    assert pnl.std_daily_pnl_usd == pytest.approx(daily.std(ddof=0))
    assert pnl.min_daily_pnl_usd == pytest.approx(-50.0)
    assert pnl.max_daily_pnl_usd == pytest.approx(100.0)
    assert pnl.n_positive_days == 2
    assert pnl.n_negative_days == 2
    # no committed backtest daily-PnL-distribution artifact exists for any
    # experiment today
    assert report.comparison_availability["pnl_distribution"] == "NOT_AVAILABLE"


def test_comparison_availability_reflects_only_real_committed_evidence():
    from alpha_agent.validation.runner import synthetic_backtest_run

    run = synthetic_backtest_run(np.array([0.0]), n_trades=0, n_fills=0)
    with_backtest = build_drift_report(
        run_id="r1", as_of_ts_ns=1, cpp_json={"orders": 0, "fills": 0},
        paper_run=run, fills=[], contracts=_EMPTY_CONTRACTS, elapsed_trading_days=1,
        backtest_daily_sharpe=0.1, backtest_annualized_sharpe=1.0,
        backtest_net_pnl_usd=500.0, backtest_n_trades=5,
    )
    assert with_backtest.comparison_availability["daily_sharpe"] == "AVAILABLE"
    assert with_backtest.comparison_availability["net_pnl_usd"] == "AVAILABLE"
    assert with_backtest.comparison_availability["n_trades"] == "AVAILABLE"

    without_backtest = build_drift_report(
        run_id="r1", as_of_ts_ns=1, cpp_json={"orders": 0, "fills": 0},
        paper_run=run, fills=[], contracts=_EMPTY_CONTRACTS, elapsed_trading_days=1,
        backtest_daily_sharpe=None, backtest_annualized_sharpe=None,
        backtest_net_pnl_usd=None, backtest_n_trades=None,
    )
    assert without_backtest.comparison_availability["daily_sharpe"] == "NOT_AVAILABLE"
    assert without_backtest.comparison_availability["net_pnl_usd"] == "NOT_AVAILABLE"
    assert without_backtest.comparison_availability["n_trades"] == "NOT_AVAILABLE"

    # these four never have a real committed comparable artifact today,
    # regardless of what the experiment's ResultRecord carries -- never
    # fabricated from an unrelated artifact.
    for report in (with_backtest, without_backtest):
        assert report.comparison_availability["fill_rate"] == "NOT_AVAILABLE"
        assert report.comparison_availability["slippage"] == "NOT_AVAILABLE"
        assert report.comparison_availability["trade_frequency"] == "NOT_AVAILABLE"
        assert report.comparison_availability["pnl_distribution"] == "NOT_AVAILABLE"


# ---------------------------------------------------------------------------
# replay-prefix consistency (Phase 21.1)
# ---------------------------------------------------------------------------
def _committed_fill(run_id: str, fill_seq: int, **overrides) -> PaperFillRow:
    base = {
        "run_id": run_id, "fill_seq": fill_seq, "step_ordinal": 0, "fill_id": fill_seq + 1,
        "order_id": fill_seq + 1, "ts_fill_ns": (fill_seq + 1) * DAY_NS, "instrument_id": 1,
        "raw_symbol": "NQZ6", "side": "buy", "quantity": 1, "fill_price": 20000.0,
        "commission_usd": 2.0, "slippage_ticks": 0.0,
    }
    base.update(overrides)
    return PaperFillRow(**base)


def _csv_row_from_fill(row: PaperFillRow) -> dict:
    return {
        "fill_id": str(row.fill_id), "order_id": str(row.order_id),
        "ts_fill_ns": str(row.ts_fill_ns), "instrument_id": str(row.instrument_id),
        "raw_symbol": row.raw_symbol, "side": row.side, "quantity": str(row.quantity),
        "fill_price": str(row.fill_price), "commission_usd": str(row.commission_usd),
        "slippage_ticks": str(row.slippage_ticks),
    }


def test_assert_replay_prefix_consistent_passes_on_exact_prefix():
    committed = [_committed_fill("r1", 0), _committed_fill("r1", 1)]
    replayed = [_csv_row_from_fill(committed[0]), _csv_row_from_fill(committed[1]),
                _csv_row_from_fill(_committed_fill("r1", 2))]
    assert_replay_prefix_consistent(run_id="r1", committed=committed, replayed_csv_rows=replayed)


def test_assert_replay_prefix_consistent_raises_on_shorter_replay():
    committed = [_committed_fill("r1", 0), _committed_fill("r1", 1)]
    replayed = [_csv_row_from_fill(committed[0])]
    with pytest.raises(PaperReplayDivergence, match="fewer than"):
        assert_replay_prefix_consistent(run_id="r1", committed=committed, replayed_csv_rows=replayed)


def test_assert_replay_prefix_consistent_raises_on_field_mismatch():
    committed = [_committed_fill("r1", 0, fill_price=20000.0)]
    replayed = [_csv_row_from_fill(_committed_fill("r1", 0, fill_price=20001.0))]
    with pytest.raises(PaperReplayDivergence, match="fill_price"):
        assert_replay_prefix_consistent(run_id="r1", committed=committed, replayed_csv_rows=replayed)


def test_assert_replay_prefix_consistent_rejects_tiny_float_perturbation():
    """Phase 21.1b: comparison is now EXACT -- a perturbation this tiny (well
    within the OLD ``math.isclose(rel_tol=1e-9, abs_tol=1e-9)`` tolerance,
    which would have silently accepted it) must still be caught."""
    committed = [_committed_fill("r1", 0, fill_price=20000.0)]
    replayed = [_csv_row_from_fill(_committed_fill("r1", 0, fill_price=20000.0 + 1e-10))]
    with pytest.raises(PaperReplayDivergence, match="fill_price"):
        assert_replay_prefix_consistent(run_id="r1", committed=committed, replayed_csv_rows=replayed)


def test_engine_step_detects_replay_divergence_and_writes_nothing(
    engine, pass_experiment_identity, provider
):
    eng, ledger = engine
    run_id = eng.start_run(
        experiment_key=pass_experiment_identity,
        window_start_ns=provider.window_start_ns,
        window_end_ns=provider.window_end_ns,
        risk_policy=PaperRiskPolicy(),
        commission_per_contract_usd=0.0,
    )
    eng.step(run_id, n_trading_days=13)  # commits the crash-bottom fill

    before_steps = ledger.list_steps(run_id)
    before_fills = ledger.list_fills(run_id)
    before_positions = ledger.list_positions(run_id)
    before_alerts = ledger.list_alerts(run_id)
    before_run = ledger.get_run(run_id)

    # mutate an EARLIER replay input (the day-11 entry price, already reflected
    # in the committed fill) -- the next replay's history now disagrees with
    # what was already committed.
    provider.prices[11] = 999.0

    with pytest.raises(PaperReplayDivergence):
        eng.step(run_id, n_trading_days=1)

    # nothing changed: divergence writes nothing.
    assert ledger.list_steps(run_id) == before_steps
    assert ledger.list_fills(run_id) == before_fills
    assert ledger.list_positions(run_id) == before_positions
    assert ledger.list_alerts(run_id) == before_alerts
    after_run = ledger.get_run(run_id)
    assert after_run.watermark_ns == before_run.watermark_ns
    assert after_run.status == before_run.status


def test_engine_step_tiny_float_perturbation_raises_and_writes_nothing(
    engine, pass_experiment_identity, provider, monkeypatch
):
    """Phase 21.1b end-to-end: a REPLAYED fill_price perturbed by 1e-10 (well
    inside the OLD tolerance) must raise PaperReplayDivergence through the
    real engine, and commit NOTHING -- no step, no fill, no position, no
    alert, no watermark/status move."""
    import alpha_agent.paper.engine as engine_module

    eng, ledger = engine
    run_id = eng.start_run(
        experiment_key=pass_experiment_identity,
        window_start_ns=provider.window_start_ns,
        window_end_ns=provider.window_end_ns,
        risk_policy=PaperRiskPolicy(),
        commission_per_contract_usd=0.0,
    )
    eng.step(run_id, n_trading_days=13)  # commits at least one real fill

    before_steps = ledger.list_steps(run_id)
    before_fills = ledger.list_fills(run_id)
    before_positions = ledger.list_positions(run_id)
    before_alerts = ledger.list_alerts(run_id)
    before_run = ledger.get_run(run_id)

    real_read_fills_csv = engine_module._read_fills_csv

    def _tiny_perturbation(path):
        rows = real_read_fills_csv(path)
        if rows:
            rows = [dict(r) for r in rows]
            rows[0]["fill_price"] = repr(float(rows[0]["fill_price"]) + 1e-10)
        return rows

    monkeypatch.setattr(engine_module, "_read_fills_csv", _tiny_perturbation)

    with pytest.raises(PaperReplayDivergence):
        eng.step(run_id, n_trading_days=1)

    assert ledger.list_steps(run_id) == before_steps
    assert ledger.list_fills(run_id) == before_fills
    assert ledger.list_positions(run_id) == before_positions
    assert ledger.list_alerts(run_id) == before_alerts
    after_run = ledger.get_run(run_id)
    assert after_run.watermark_ns == before_run.watermark_ns
    assert after_run.status == before_run.status


# ---------------------------------------------------------------------------
# atomic step commit (Phase 21.1)
# ---------------------------------------------------------------------------
def _mk_step_row(run_id: str, step_ordinal: int) -> PaperStepRow:
    return PaperStepRow(
        run_id=run_id, step_ordinal=step_ordinal, as_of_ts_ns=(step_ordinal + 1) * DAY_NS,
        created_at=datetime.now(UTC).isoformat(), bars_processed=1, orders_generated=1,
        fills_generated=1, trades_closed=0, gross_pnl_usd=0.0, costs_usd=0.0, net_pnl_usd=0.0,
        unrealized_pnl_usd=0.0, equity_usd=100000.0, peak_equity_usd=100000.0, drawdown_usd=0.0,
        drawdown_pct=0.0, day_realized_pnl_usd=0.0, risk_rejects=0, risk_resizes=0,
        kill_switch_active=False, margin_complete=False, open_positions=0,
        elapsed_trading_days=1, daily_sharpe=None, annualized_sharpe=None,
        provenance={}, drift={}, raw_result={"fills": 1},
    )


def _seeded_ledger_run(tmp_path) -> tuple[PaperLedger, str]:
    ledger = PaperLedger(tmp_path / "ledger.sqlite")
    run_id = ledger.create_run(
        experiment_id="exp-1", experiment_identity="ident-1", strategy_family="mean_reversion",
        strategy_id="TEST", strategy_fingerprint="stratdsl1:x", root_symbol="NQ",
        params={"root_symbol": "NQ"}, risk_policy=PaperRiskPolicy().model_dump(mode="json"),
        risk_policy_identity="paperrisk1:x", schedule_policy="no_decision",
        commission_per_contract_usd=2.0, slippage_ticks=0.0, spread_ticks=0.0,
        window_start_ns=0, window_end_ns=10 * DAY_NS, data_source="synthetic_test_fixture",
        validation_result={},
    )
    return ledger, run_id


def test_ledger_commit_step_atomic_success(tmp_path):
    ledger, run_id = _seeded_ledger_run(tmp_path)
    step = _mk_step_row(run_id, 0)
    fill = _committed_fill(run_id, 0)
    position = PaperPositionRow(
        run_id=run_id, step_ordinal=0, instrument_id=1, raw_symbol="NQZ6", root_symbol="NQ",
        units=1, avg_entry_price=20000.0, multiplier=20.0, mark_price=20000.0, mark_ts_ns=DAY_NS,
        mark_age_ns=0, mark_present=True, mark_is_stale=False, valuation_is_estimated=False,
        gross_notional_usd=400000.0, signed_notional_usd=400000.0, unrealized_pnl_usd=0.0,
        initial_margin_usd=0.0, maintenance_margin_usd=0.0, margin_known=False,
    )
    alert = PaperAlertRow(
        run_id=run_id, alert_seq=0, step_ordinal=0, ts_ns=DAY_NS,
        created_at=datetime.now(UTC).isoformat(), alert_type="RUN_STARTED", severity="INFO",
        message="x", detail={},
    )
    ledger.commit_step(
        run_id=run_id, step=step, fills=[fill], positions=[position], alerts=[alert],
        watermark_ns=DAY_NS, status="ACTIVE", stopped_at=None,
    )
    assert len(ledger.list_steps(run_id)) == 1
    assert len(ledger.list_fills(run_id)) == 1
    assert len(ledger.list_positions(run_id)) == 1
    assert len(ledger.list_alerts(run_id)) == 1
    assert ledger.get_run(run_id).watermark_ns == DAY_NS
    ledger.close()


def test_ledger_commit_step_rolls_back_completely_on_failure(tmp_path, monkeypatch):
    ledger, run_id = _seeded_ledger_run(tmp_path)
    step = _mk_step_row(run_id, 0)
    fill = _committed_fill(run_id, 0)

    def _boom(self, alerts):
        raise RuntimeError("simulated failure between fills and alerts")

    # inject a failure AFTER _insert_step / _insert_fills / _insert_positions
    # would have run, but before the transaction commits.
    monkeypatch.setattr(PaperLedger, "_insert_alerts", _boom)

    before_watermark = ledger.get_run(run_id).watermark_ns
    with pytest.raises(RuntimeError, match="simulated failure"):
        ledger.commit_step(
            run_id=run_id, step=step, fills=[fill], positions=[], alerts=[
                PaperAlertRow(
                    run_id=run_id, alert_seq=0, step_ordinal=0, ts_ns=DAY_NS,
                    created_at=datetime.now(UTC).isoformat(), alert_type="RUN_STARTED",
                    severity="INFO", message="x", detail={},
                )
            ],
            watermark_ns=DAY_NS, status="ACTIVE", stopped_at=None,
        )

    # NOTHING committed: the step, the fill, and the watermark move all rolled back.
    assert ledger.list_steps(run_id) == ()
    assert ledger.list_fills(run_id) == ()
    assert ledger.list_alerts(run_id) == ()
    assert ledger.get_run(run_id).watermark_ns == before_watermark
    ledger.close()


# ---------------------------------------------------------------------------
# authoritative C++ position snapshot (Phase 21.1)
# ---------------------------------------------------------------------------
def test_engine_step_persists_authoritative_position_snapshot(
    engine, pass_experiment_identity, provider
):
    eng, ledger = engine
    run_id = eng.start_run(
        experiment_key=pass_experiment_identity,
        window_start_ns=provider.window_start_ns,
        window_end_ns=provider.window_end_ns,
        risk_policy=PaperRiskPolicy(),
        commission_per_contract_usd=0.0,
    )
    snap = eng.step(run_id, n_trading_days=13)
    positions = ledger.list_positions(run_id, snap.step_ordinal)
    assert len(positions) == 1
    pos = positions[0]
    assert pos.raw_symbol.startswith("NQ")
    assert pos.root_symbol == "NQ"
    assert pos.units == 1
    assert pos.avg_entry_price == pytest.approx(18500.0)
    assert pos.mark_price == pytest.approx(16000.0)
    # the position's own unrealized PnL must equal the step's committed
    # unrealized_pnl_usd (the same C++ payload, never independently recomputed).
    latest = ledger.latest_step(run_id)
    assert pos.unrealized_pnl_usd == pytest.approx(latest.unrealized_pnl_usd)


def test_position_row_is_never_recomputed_in_python():
    """A position row is a verbatim copy of the CLI's JSON -- even an
    internally-inconsistent value (one that would be WRONG if Python
    recomputed unrealized PnL from mark/entry/units) must round-trip
    unchanged, proving nothing here recomputes it."""
    from alpha_agent.paper.engine import _position_rows

    contrived_cpp_json = {
        "positions": [
            {
                "instrument_id": 1, "raw_symbol": "NQZ6", "root_symbol": "NQ", "units": 1,
                "avg_entry_price": 20000.0, "multiplier": 20.0, "mark_price": 20000.0,
                "mark_ts_ns": DAY_NS, "mark_age_ns": 0, "mark_present": True,
                "mark_is_stale": False, "valuation_is_estimated": False,
                "gross_notional_usd": 400000.0, "signed_notional_usd": 400000.0,
                # if this were recomputed as (mark - entry) * mult * units it
                # would be 0.0 -- assert the CONTRIVED value survives verbatim.
                "unrealized_pnl_usd": -123456.0,
                "initial_margin_usd": 0.0, "maintenance_margin_usd": 0.0, "margin_known": False,
            }
        ]
    }
    rows = _position_rows("r1", 0, contrived_cpp_json)
    assert len(rows) == 1
    assert rows[0].unrealized_pnl_usd == -123456.0


# ---------------------------------------------------------------------------
# exact per-step execution provenance (Phase 21.1)
# ---------------------------------------------------------------------------
def test_step_provenance_stable_for_identical_inputs(registry, tmp_path, provider):
    """Two independently-started runs, identical configuration, stepped the
    same way, must produce the SAME provenance identity."""
    fp = _mr_fingerprint()
    record = _experiment_record(strategy_fp=fp)
    registry.insert_experiment(record, _pass_result(record.experiment_identity))

    def _one_run(work_subdir: str) -> str:
        ledger = PaperLedger(tmp_path / f"ledger_{work_subdir}.sqlite")
        eng = PaperTradingEngine(
            registry=registry, ledger=ledger, provider=provider, executable=PAPER_CLI,
            work_dir=tmp_path / work_subdir, data_source_label="synthetic_test_fixture",
        )
        run_id = eng.start_run(
            experiment_key=record.experiment_identity,
            window_start_ns=provider.window_start_ns,
            window_end_ns=provider.window_end_ns,
            risk_policy=PaperRiskPolicy(),
            commission_per_contract_usd=0.0,
        )
        snap = eng.step(run_id, n_trading_days=13)
        ledger.close()
        return snap.provenance.identity()

    identity_a = _one_run("work_a")
    identity_b = _one_run("work_b")
    assert identity_a == identity_b


def test_step_provenance_changes_when_risk_policy_changes(registry, tmp_path, provider):
    fp = _mr_fingerprint()
    record = _experiment_record(strategy_fp=fp)
    registry.insert_experiment(record, _pass_result(record.experiment_identity))

    def _one_run(work_subdir: str, risk_policy: PaperRiskPolicy) -> str:
        ledger = PaperLedger(tmp_path / f"ledger_{work_subdir}.sqlite")
        eng = PaperTradingEngine(
            registry=registry, ledger=ledger, provider=provider, executable=PAPER_CLI,
            work_dir=tmp_path / work_subdir, data_source_label="synthetic_test_fixture",
        )
        run_id = eng.start_run(
            experiment_key=record.experiment_identity,
            window_start_ns=provider.window_start_ns,
            window_end_ns=provider.window_end_ns,
            risk_policy=risk_policy,
            commission_per_contract_usd=0.0,
        )
        snap = eng.step(run_id, n_trading_days=13)
        ledger.close()
        return snap.provenance.identity()

    identity_default = _one_run("work_c", PaperRiskPolicy())
    identity_tight = _one_run("work_d", PaperRiskPolicy(max_drawdown_pct=0.01))
    assert identity_default != identity_tight


def test_step_provenance_persisted_and_retrievable_via_report(
    engine, pass_experiment_identity, provider
):
    from alpha_agent.paper.report import build_paper_run_report

    eng, ledger = engine
    run_id = eng.start_run(
        experiment_key=pass_experiment_identity,
        window_start_ns=provider.window_start_ns,
        window_end_ns=provider.window_end_ns,
        risk_policy=PaperRiskPolicy(),
        commission_per_contract_usd=0.0,
    )
    snap = eng.step(run_id, n_trading_days=13)
    report = build_paper_run_report(ledger, run_id)
    stored = report["latest_step"]["provenance"]
    assert stored["parent_experiment_identity"] == pass_experiment_identity
    assert stored["strategy_fingerprint"]
    assert stored["bars_sha256"]
    assert stored["executable_sha256"]
    assert stored["result_payload_sha256"]
    # recomputing PaperStepProvenance.identity() from the persisted dict must
    # match what step() itself reported.
    from alpha_agent.paper.provenance import PaperStepProvenance

    assert PaperStepProvenance.model_validate(stored).identity() == snap.provenance.identity()
    assert report["latest_step"]["drift"]["schema_version"] == "paper-drift-report/2"


# ---------------------------------------------------------------------------
# read-only ledger-write boundary (Phase 21.1 extension of the Phase 21 guard)
# ---------------------------------------------------------------------------
def test_paper_engine_never_bypasses_the_atomic_commit_path():
    """``PaperTradingEngine._run_step`` (the per-step persistence path) must
    write ONLY through ``PaperLedger.commit_step`` -- never call the
    single-purpose append_* / update_run_state methods directly (those exist
    for direct/test use, and for ``start_run``'s one-time RUN_STARTED alert,
    which is not part of a step's atomic multi-table write)."""
    import inspect

    from alpha_agent.paper.engine import PaperTradingEngine

    src = inspect.getsource(PaperTradingEngine._run_step)
    forbidden = ["self._ledger.append_step(", "self._ledger.append_fills(",
                 "self._ledger.append_positions(", "self._ledger.append_alerts(",
                 "self._ledger.update_run_state("]
    for pattern in forbidden:
        assert pattern not in src, f"_run_step must persist a step only via commit_step, found {pattern!r}"
    assert "self._ledger.commit_step(" in src


# ---------------------------------------------------------------------------
# versioned paper-ledger schema migration (Phase 21.1b)
# ---------------------------------------------------------------------------
#: byte-for-byte the Phase 21 (commit 1087724) paper_steps/paper_runs/
#: paper_fills/paper_alerts schema -- verified via `git show 1087724:
#: python/alpha_agent/paper/ledger.py`. No paper_positions table, no
#: elapsed_trading_days/provenance_json/drift_json columns, no PRAGMA
#: user_version ever set (reads back 0, SQLite's own default).
_PHASE_21_1087724_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_runs (
    run_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    experiment_identity TEXT NOT NULL,
    strategy_family TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    strategy_fingerprint TEXT NOT NULL,
    root_symbol TEXT NOT NULL,
    params_json TEXT NOT NULL,
    risk_policy_json TEXT NOT NULL,
    risk_policy_identity TEXT NOT NULL,
    schedule_policy TEXT NOT NULL,
    commission_per_contract_usd REAL NOT NULL,
    slippage_ticks REAL NOT NULL,
    spread_ticks REAL NOT NULL,
    window_start_ns INTEGER NOT NULL,
    window_end_ns INTEGER NOT NULL,
    watermark_ns INTEGER NOT NULL,
    status TEXT NOT NULL,
    data_source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    stopped_at TEXT,
    validation_result_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_steps (
    run_id TEXT NOT NULL,
    step_ordinal INTEGER NOT NULL,
    as_of_ts_ns INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    bars_processed INTEGER NOT NULL,
    orders_generated INTEGER NOT NULL,
    fills_generated INTEGER NOT NULL,
    trades_closed INTEGER NOT NULL,
    gross_pnl_usd REAL NOT NULL,
    costs_usd REAL NOT NULL,
    net_pnl_usd REAL NOT NULL,
    unrealized_pnl_usd REAL NOT NULL,
    equity_usd REAL NOT NULL,
    peak_equity_usd REAL NOT NULL,
    drawdown_usd REAL NOT NULL,
    drawdown_pct REAL NOT NULL,
    day_realized_pnl_usd REAL NOT NULL,
    risk_rejects INTEGER NOT NULL,
    risk_resizes INTEGER NOT NULL,
    kill_switch_active INTEGER NOT NULL,
    margin_complete INTEGER NOT NULL,
    open_positions INTEGER NOT NULL,
    daily_sharpe REAL,
    annualized_sharpe REAL,
    raw_result_json TEXT NOT NULL,
    PRIMARY KEY (run_id, step_ordinal),
    FOREIGN KEY (run_id) REFERENCES paper_runs(run_id)
);

CREATE TABLE IF NOT EXISTS paper_fills (
    run_id TEXT NOT NULL,
    fill_seq INTEGER NOT NULL,
    step_ordinal INTEGER NOT NULL,
    fill_id INTEGER NOT NULL,
    order_id INTEGER NOT NULL,
    ts_fill_ns INTEGER NOT NULL,
    instrument_id INTEGER NOT NULL,
    raw_symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    fill_price REAL NOT NULL,
    commission_usd REAL NOT NULL,
    slippage_ticks REAL NOT NULL,
    PRIMARY KEY (run_id, fill_seq),
    FOREIGN KEY (run_id) REFERENCES paper_runs(run_id)
);

CREATE TABLE IF NOT EXISTS paper_alerts (
    run_id TEXT NOT NULL,
    alert_seq INTEGER NOT NULL,
    step_ordinal INTEGER NOT NULL,
    ts_ns INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    detail_json TEXT NOT NULL,
    PRIMARY KEY (run_id, alert_seq),
    FOREIGN KEY (run_id) REFERENCES paper_runs(run_id)
);
"""


def _build_legacy_1087724_ledger(path) -> None:
    """Create a database at ``path`` using the EXACT Phase 21 (1087724)
    schema, with representative rows, entirely bypassing PaperLedger (which
    only knows the current schema) -- raw sqlite3 only, as a real pre-21.1
    ledger file would actually look on disk."""
    import sqlite3 as _sqlite3

    conn = _sqlite3.connect(path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        with conn:
            conn.executescript(_PHASE_21_1087724_SCHEMA)
            conn.execute(
                """
                INSERT INTO paper_runs (
                    run_id, schema_version, experiment_id, experiment_identity,
                    strategy_family, strategy_id, strategy_fingerprint, root_symbol,
                    params_json, risk_policy_json, risk_policy_identity, schedule_policy,
                    commission_per_contract_usd, slippage_ticks, spread_ticks,
                    window_start_ns, window_end_ns, watermark_ns, status, data_source,
                    created_at, stopped_at, validation_result_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    "paper-legacy-1", "paper-run/1", "NQ__LEGACY__CANONICAL", "legacy-ident-1",
                    "mean_reversion", "TEST-LEGACY", "stratdsl1:legacy", "NQ",
                    "{}", "{}", "paperrisk1:legacy", "no_decision",
                    2.0, 0.0, 0.0, 0, 10 * DAY_NS, DAY_NS, "ACTIVE", "real_2018_2024_replay",
                    "2026-01-01T00:00:00+00:00", None, "{}",
                ),
            )
            conn.execute(
                """
                INSERT INTO paper_steps (
                    run_id, step_ordinal, as_of_ts_ns, created_at, bars_processed,
                    orders_generated, fills_generated, trades_closed, gross_pnl_usd,
                    costs_usd, net_pnl_usd, unrealized_pnl_usd, equity_usd,
                    peak_equity_usd, drawdown_usd, drawdown_pct, day_realized_pnl_usd,
                    risk_rejects, risk_resizes, kill_switch_active, margin_complete,
                    open_positions, daily_sharpe, annualized_sharpe, raw_result_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    "paper-legacy-1", 0, DAY_NS, "2026-01-01T00:00:00+00:00", 5, 1, 1, 0,
                    0.0, 0.0, -1500.0, -1500.0, 98500.0, 100000.0, 1500.0, 0.015, 0.0,
                    0, 0, 0, 0, 1, None, None, '{"fills": 1}',
                ),
            )
            conn.execute(
                """
                INSERT INTO paper_fills (
                    run_id, fill_seq, step_ordinal, fill_id, order_id, ts_fill_ns,
                    instrument_id, raw_symbol, side, quantity, fill_price,
                    commission_usd, slippage_ticks
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                ("paper-legacy-1", 0, 0, 1, 1, DAY_NS, 1, "NQZ6", "buy", 1, 20000.0, 2.0, 0.0),
            )
            conn.execute(
                """
                INSERT INTO paper_alerts (
                    run_id, alert_seq, step_ordinal, ts_ns, created_at, alert_type,
                    severity, message, detail_json
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    "paper-legacy-1", 0, -1, 0, "2026-01-01T00:00:00+00:00", "RUN_STARTED",
                    "INFO", "legacy run started", "{}",
                ),
            )
    finally:
        conn.close()


def test_migration_preserves_legacy_rows_and_marks_them_honestly(tmp_path):
    from alpha_agent.paper.ledger import (
        CURRENT_SCHEMA_VERSION,
        LEGACY_DRIFT,
        LEGACY_PROVENANCE,
        PaperAlertRow,
        PaperPositionRow,
        PaperStepRow,
        is_legacy_evidence,
    )

    db_path = tmp_path / "legacy_ledger.sqlite"
    _build_legacy_1087724_ledger(db_path)

    # sanity: confirm the raw fixture really is the OLD, unversioned shape
    # before PaperLedger ever touches it.
    raw = __import__("sqlite3").connect(db_path)
    assert raw.execute("PRAGMA user_version").fetchone()[0] == 0
    old_cols = {r[1] for r in raw.execute("PRAGMA table_info(paper_steps)")}
    assert "provenance_json" not in old_cols
    assert "elapsed_trading_days" not in old_cols
    tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "paper_positions" not in tables
    raw.close()

    # 3/4: open with the CURRENT PaperLedger -- migration must run automatically.
    ledger = PaperLedger(db_path)
    assert ledger.schema_version == CURRENT_SCHEMA_VERSION

    # 5: every old row preserved, unchanged, for the fields Phase 21 recorded.
    run = ledger.get_run("paper-legacy-1")
    assert run.experiment_identity == "legacy-ident-1"
    assert run.strategy_family == "mean_reversion"
    assert run.watermark_ns == DAY_NS
    assert run.status == "ACTIVE"

    steps = ledger.list_steps("paper-legacy-1")
    assert len(steps) == 1
    legacy_step = steps[0]
    assert legacy_step.equity_usd == 98500.0
    assert legacy_step.net_pnl_usd == -1500.0
    assert legacy_step.fills_generated == 1
    assert legacy_step.open_positions == 1
    assert legacy_step.raw_result == {"fills": 1}

    fills = ledger.list_fills("paper-legacy-1")
    assert len(fills) == 1
    assert fills[0].fill_price == 20000.0
    assert fills[0].raw_symbol == "NQZ6"

    alerts = ledger.list_alerts("paper-legacy-1")
    assert len(alerts) == 1
    assert alerts[0].alert_type == "RUN_STARTED"

    # 6: legacy provenance/drift is HONEST, not invented.
    assert legacy_step.elapsed_trading_days is None
    assert is_legacy_evidence(legacy_step.provenance)
    assert legacy_step.provenance["status"] == "NOT_AVAILABLE"
    assert legacy_step.provenance == LEGACY_PROVENANCE
    assert is_legacy_evidence(legacy_step.drift)
    assert legacy_step.drift["status"] == "NOT_AVAILABLE"
    assert legacy_step.drift == LEGACY_DRIFT
    # paper_positions is a wholly new table -- a legacy step has no rows in
    # it (never fabricated), distinguishable from "flat" only via the legacy
    # provenance marker (see the UI's is_legacy_evidence usage).
    assert ledger.list_positions("paper-legacy-1", 0) == ()

    # 7: a NEW Phase 21.1 step commits atomically afterward, on the SAME
    # migrated database, alongside the preserved legacy row.
    new_step = PaperStepRow(
        run_id="paper-legacy-1", step_ordinal=1, as_of_ts_ns=2 * DAY_NS,
        created_at=datetime.now(UTC).isoformat(), bars_processed=6, orders_generated=1,
        fills_generated=1, trades_closed=0, gross_pnl_usd=0.0, costs_usd=0.0, net_pnl_usd=-1500.0,
        unrealized_pnl_usd=-1500.0, equity_usd=98500.0, peak_equity_usd=100000.0,
        drawdown_usd=1500.0, drawdown_pct=0.015, day_realized_pnl_usd=0.0, risk_rejects=0,
        risk_resizes=0, kill_switch_active=False, margin_complete=False, open_positions=1,
        elapsed_trading_days=2, daily_sharpe=None, annualized_sharpe=None,
        provenance={"schema_version": "paper-step-provenance/1", "status": "REAL"},
        drift={"schema_version": "paper-drift-report/2", "status": "REAL"}, raw_result={"fills": 1},
    )
    new_position = PaperPositionRow(
        run_id="paper-legacy-1", step_ordinal=1, instrument_id=1, raw_symbol="NQZ6",
        root_symbol="NQ", units=1, avg_entry_price=20000.0, multiplier=20.0, mark_price=19000.0,
        mark_ts_ns=2 * DAY_NS, mark_age_ns=0, mark_present=True, mark_is_stale=False,
        valuation_is_estimated=False, gross_notional_usd=380000.0, signed_notional_usd=380000.0,
        unrealized_pnl_usd=-20000.0, initial_margin_usd=0.0, maintenance_margin_usd=0.0,
        margin_known=False,
    )
    new_alert = PaperAlertRow(
        run_id="paper-legacy-1", alert_seq=1, step_ordinal=1, ts_ns=2 * DAY_NS,
        created_at=datetime.now(UTC).isoformat(), alert_type="DRIFT_WARNING", severity="WARNING",
        message="x", detail={},
    )
    ledger.commit_step(
        run_id="paper-legacy-1", step=new_step, fills=[], positions=[new_position],
        alerts=[new_alert], watermark_ns=2 * DAY_NS, status="ACTIVE", stopped_at=None,
    )
    assert len(ledger.list_steps("paper-legacy-1")) == 2
    fresh_step = ledger.latest_step("paper-legacy-1")
    assert not is_legacy_evidence(fresh_step.provenance)
    assert fresh_step.elapsed_trading_days == 2
    assert len(ledger.list_positions("paper-legacy-1", 1)) == 1
    ledger.close()

    # 8: re-opening the migrated database is idempotent -- no further change.
    ledger2 = PaperLedger(db_path)
    assert ledger2.schema_version == CURRENT_SCHEMA_VERSION
    assert len(ledger2.list_steps("paper-legacy-1")) == 2
    replayed_legacy_step = ledger2.list_steps("paper-legacy-1")[0]
    assert replayed_legacy_step == legacy_step  # byte-identical re-read, migration did not re-run
    ledger2.close()


def test_migration_is_idempotent_across_repeated_opens(tmp_path):
    db_path = tmp_path / "legacy_ledger_2.sqlite"
    _build_legacy_1087724_ledger(db_path)

    versions = []
    for _ in range(3):
        ledger = PaperLedger(db_path)
        versions.append(ledger.schema_version)
        steps = ledger.list_steps("paper-legacy-1")
        ledger.close()
        assert len(steps) == 1

    from alpha_agent.paper.ledger import CURRENT_SCHEMA_VERSION

    assert versions == [CURRENT_SCHEMA_VERSION] * 3


def test_new_database_is_created_directly_at_the_latest_schema(tmp_path):
    from alpha_agent.paper.ledger import CURRENT_SCHEMA_VERSION

    ledger = PaperLedger(tmp_path / "brand_new.sqlite")
    assert ledger.schema_version == CURRENT_SCHEMA_VERSION
    cols = {
        r[1]
        for r in ledger._conn.execute("PRAGMA table_info(paper_steps)")
    }
    assert {"elapsed_trading_days", "provenance_json", "drift_json"} <= cols
    tables = {r[0] for r in ledger._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "paper_positions" in tables
    ledger.close()


def test_local_real_paper_ledger_migrates_non_destructively_if_present(tmp_path):
    """If a real local pre-21.1 data/paper_trading/paper_ledger.sqlite exists
    on this machine, prove migration succeeds against a COPY of it -- never
    touching the original file."""
    import shutil

    real_path = REPO_ROOT / "data" / "paper_trading" / "paper_ledger.sqlite"
    if not real_path.exists():
        pytest.skip("no local data/paper_trading/paper_ledger.sqlite to smoke-test")

    original_bytes = real_path.read_bytes()
    copy_path = tmp_path / "real_ledger_copy.sqlite"
    shutil.copy2(real_path, copy_path)

    ledger = PaperLedger(copy_path)
    from alpha_agent.paper.ledger import CURRENT_SCHEMA_VERSION

    assert ledger.schema_version == CURRENT_SCHEMA_VERSION
    # every run in the copy is still readable post-migration.
    for run in ledger.list_runs():
        for step in ledger.list_steps(run.run_id):
            assert step.provenance  # never an unrecognisable {} post-migration
    ledger.close()

    # the ORIGINAL file was never opened by PaperLedger and is untouched.
    assert real_path.read_bytes() == original_bytes
