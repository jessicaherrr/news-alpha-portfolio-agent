// Phase 21 -- quant::run_paper_trading_backtest (quant_core/paper_trading_run.hpp).
//
//   A. equivalent to hand-wired BacktestEngine + PortfolioRiskManager when no
//      limit is breached (adds no execution/accounting semantics of its own);
//   B. the hard risk gate actually differs from the research reference path:
//      the SAME losing schedule that runs to completion under
//      run_targets_backtest (PassThroughRiskManager) trips the drawdown kill
//      switch under run_paper_trading_backtest (PortfolioRiskManager) --
//      opening orders are rejected/resized, the existing position is never
//      force-closed by risk, and the deficient margin schedule is visible
//      (never silently treated as satisfied);
//   C. EndOfTestPolicy defaults to LeaveOpen (a position survives to be
//      replayed again), never ForceLiquidateFinalClose, unless the caller asks;
//   D. the contract-resolution hard check fires exactly as the research path;
//   E. config guards (negative cost overrides, non-ascending validation days);
//   F. deterministic: identical inputs -> identical result.
//   G. (News Alpha Phase H) PaperTradingRunConfig::corporate_actions is a pure
//      pass-through: a split supplied here rebases the position exactly as the
//      same split supplied to a hand-wired BacktestEngine, and an empty config
//      is byte-identical to no corporate actions at all.

#include "quant_core/contract.hpp"
#include "quant_core/contract_registry.hpp"
#include "quant_core/contract_selector.hpp"
#include "quant_core/engine.hpp"
#include "quant_core/margin.hpp"
#include "quant_core/paper_trading_run.hpp"
#include "quant_core/risk_config.hpp"
#include "quant_core/risk_manager.hpp"
#include "quant_core/scheduled_target_strategy.hpp"
#include "quant_core/targets_run.hpp"
#include "quant_core/types.hpp"

#include "test_support.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace quant;

constexpr std::int64_t  kMin  = 60'000'000'000LL;
constexpr std::uint32_t kNq   = 1;
constexpr std::int64_t  kExp  = 100'000LL * kMin;
const std::string kFp = "stratdsl1:deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef";

ContractRegistry nq_registry() {
    ContractRegistry r;
    r.add(ContractSpec{.instrument_id = kNq, .raw_symbol = "NQZ6", .root_symbol = "NQ",
                       .exchange = "XCME", .tick_size = 0.25, .multiplier = 20.0,
                       .activation_ns = 1, .expiration_ns = kExp});
    return r;
}

MarketBar bar(std::int64_t idx, double open, double close) {
    const double hi = (open > close ? open : close) + 1.0;
    const double lo = (open < close ? open : close) - 1.0;
    return MarketBar{idx * kMin, kNq, open, hi, lo, close, 100};
}

// A dense +1 target every bar (never flat) so the strategy keeps trying to add
// to the position -- exactly the shape that exercises a kill switch.
std::vector<ScheduledTarget> dense_target(const std::vector<MarketBar>& bars,
                                          const std::vector<double>& targets) {
    std::vector<ScheduledTarget> rows;
    for (std::size_t i = 0; i + 1 < bars.size(); ++i) {
        rows.push_back(ScheduledTarget{.root_symbol = "NQ",
                                       .ts_event_ns = bars[i].ts_event_ns,
                                       .target_units = static_cast<int>(targets[i])});
    }
    return rows;
}

RiskConfig lenient_risk_config() {
    // Every hard limit disabled / generous: should behave like pass-through for
    // an order flow that never comes close to any of them.
    RiskConfig cfg;
    cfg.max_contracts_per_symbol = 1000;
    cfg.max_contracts_per_root = 1000;
    cfg.max_gross_contracts = 1000;
    cfg.max_gross_exposure_usd = 0.0;
    cfg.max_gross_leverage = 0.0;
    cfg.max_net_leverage = 0.0;
    cfg.max_margin_utilization_pct = 0.0;
    cfg.missing_margin = MissingMarginPolicy::TreatAsZero;
    cfg.max_daily_loss_usd = 0.0;
    cfg.max_drawdown_pct = 0.0;
    cfg.max_drawdown_usd = 0.0;
    cfg.stale_mark = StaleMarkPolicy::Ignore;
    cfg.portfolio.starting_capital_usd = 100'000.0;
    cfg.portfolio.mark_staleness_tolerance_ns = kMin;
    return cfg;
}

// A. equivalence to the hand-wired engine path when nothing is breached.
void test_equivalent_to_manual_engine() {
    std::vector<MarketBar> bars;
    for (int i = 1; i <= 8; ++i) bars.push_back(bar(i, 20000.0, 20000.0));
    const auto registry = nq_registry();
    const std::vector<double> targets(bars.size(), 1.0);

    const RegistryActiveContractResolver resolver(registry);
    const RiskConfig rc = lenient_risk_config();
    const PortfolioRiskManager risk(rc, nullptr);
    EngineConfig config;
    config.execution.commission_per_contract_usd = 2.0;
    config.end_of_test = EndOfTestPolicy::LeaveOpen;
    const ScheduledTargetStrategy strat(dense_target(bars, targets), kFp,
                                       ScheduledTargetStrategy::AbsentPolicy::NoDecision);
    const BacktestEngine engine(registry, resolver, risk, config);
    const auto manual = engine.run(bars, strat);

    PaperTradingRunConfig cfg;
    cfg.risk_config = rc;
    const auto out = run_paper_trading_backtest(bars, registry, dense_target(bars, targets), kFp, cfg);

    CHECK(out.result.fills_generated == manual.fills_generated);
    CHECK(out.result.closed_trades == manual.closed_trades);
    CHECK_CLOSE(out.result.net_realized_pnl_usd, manual.net_realized_pnl_usd, 1e-9);
    CHECK_CLOSE(out.result.net_equity_usd_at_end, manual.net_equity_usd_at_end, 1e-9);
    CHECK(out.result.risk_rejects == 0);
    CHECK(out.contracts_resolved == 1);
    CHECK(out.commission_per_contract_usd == 2.0);
}

// B. the hard risk gate is genuinely different from the research reference path.
void test_kill_switch_differs_from_research_reference_path() {
    // enter +1 at bar 2 open 20000; price then collapses; strategy keeps
    // asking to top up to +2 -- a risk-increasing order.
    std::vector<MarketBar> bars{
        bar(1, 20000.0, 20000.0),
        bar(2, 20000.0, 20000.0),   // BUY 1 @ 20000
        bar(3, 20000.0, 19000.0),
        bar(4, 18800.0, 18500.0),   // deep drawdown
        bar(5, 18500.0, 18500.0),   // attempted top-up executes here
        bar(6, 18500.0, 18500.0),
    };
    const std::vector<double> targets{1, 1, 1, 2, 2, 2};
    const auto registry = nq_registry();

    // (1) the research reference path -- PassThroughRiskManager -- never
    // rejects a well-formed order, no matter how large the drawdown, so the
    // top-up to +2 goes through unconditionally.
    TargetsRunConfig ref_cfg;
    ref_cfg.commission_per_contract_usd = 0.0;
    const auto reference = run_targets_backtest(bars, registry, dense_target(bars, targets), kFp, ref_cfg);
    CHECK(reference.result.risk_rejects == 0);  // pass-through never rejects a sized order
    CHECK(!reference.result.risk_decisions.empty());
    for (const auto& d : reference.result.risk_decisions) {
        CHECK(d.verdict == RiskVerdict::Approve);
    }

    // (2) the SAME schedule/bars under paper trading's hard risk manager.
    PaperTradingRunConfig cfg;
    cfg.commission_per_contract_usd = 0.0;
    cfg.risk_config = lenient_risk_config();
    cfg.risk_config.max_drawdown_pct = 0.10;  // the only limit this scenario should trip
    const auto paper = run_paper_trading_backtest(bars, registry, dense_target(bars, targets), kFp, cfg);

    CHECK(paper.result.risk_rejects >= 1);
    bool kill = false;
    for (const auto& d : paper.result.risk_decisions) {
        if (d.reason_code == "drawdown_kill_switch" && d.verdict == RiskVerdict::Reject) kill = true;
    }
    CHECK(kill);
    // the existing position is untouched -- a kill switch never force-closes it.
    CHECK(paper.result.final_positions.size() == 1);
    CHECK(paper.result.final_positions[0].units == 1);  // top-up BLOCKED, unlike the reference path
    // no MarginModel was supplied: the gap stays VISIBLE (margin_complete ==
    // false) rather than being silently treated as satisfied -- TreatAsZero
    // only controls whether the risk gate blocks on it, never the reported state.
    CHECK(paper.result.portfolio_at_end.margin_complete == false);
}

// C. EndOfTestPolicy defaults to LeaveOpen.
void test_default_end_of_test_is_leave_open() {
    std::vector<MarketBar> bars;
    for (int i = 1; i <= 4; ++i) bars.push_back(bar(i, 20000.0, 20000.0));
    const auto registry = nq_registry();
    const std::vector<double> targets(bars.size(), 1.0);

    PaperTradingRunConfig cfg;
    cfg.risk_config = lenient_risk_config();
    const auto leave_open = run_paper_trading_backtest(bars, registry, dense_target(bars, targets), kFp, cfg);
    CHECK(leave_open.result.final_positions.size() == 1);
    CHECK(leave_open.result.eot_liquidations.empty());

    PaperTradingRunConfig cfg2 = cfg;
    cfg2.end_of_test = EndOfTestPolicy::ForceLiquidateFinalClose;
    const auto flattened = run_paper_trading_backtest(bars, registry, dense_target(bars, targets), kFp, cfg2);
    CHECK(flattened.result.final_positions.empty());
    CHECK(flattened.result.eot_liquidations.size() == 1);
}

// D. contract-resolution hard check.
void test_unresolved_instrument_throws() {
    std::vector<MarketBar> bars{bar(1, 100.0, 100.0)};
    bars.back().instrument_id = 999999;
    const auto registry = nq_registry();
    PaperTradingRunConfig cfg;
    cfg.risk_config = lenient_risk_config();
    CHECK_THROWS(run_paper_trading_backtest(bars, registry, {}, kFp, cfg));
}

// E. config guards.
void test_config_guards() {
    std::vector<MarketBar> bars{bar(1, 100.0, 100.0), bar(2, 100.0, 100.0)};
    const auto registry = nq_registry();

    PaperTradingRunConfig neg;
    neg.risk_config = lenient_risk_config();
    neg.commission_per_contract_usd = -1.0;
    CHECK_THROWS(run_paper_trading_backtest(bars, registry, {}, kFp, neg));

    PaperTradingRunConfig unsorted;
    unsorted.risk_config = lenient_risk_config();
    unsorted.validation_day_boundaries_ns = {5 * kMin, 3 * kMin};
    CHECK_THROWS(run_paper_trading_backtest(bars, registry, {}, kFp, unsorted));
}

// F. determinism.
void test_deterministic() {
    std::vector<MarketBar> bars;
    for (int i = 1; i <= 6; ++i) bars.push_back(bar(i, 20000.0, 20000.0));
    const auto registry = nq_registry();
    const std::vector<double> targets(bars.size(), 1.0);
    PaperTradingRunConfig cfg;
    cfg.risk_config = lenient_risk_config();

    const auto a = run_paper_trading_backtest(bars, registry, dense_target(bars, targets), kFp, cfg);
    const auto b = run_paper_trading_backtest(bars, registry, dense_target(bars, targets), kFp, cfg);
    CHECK(a.result.fills_generated == b.result.fills_generated);
    CHECK_CLOSE(a.result.net_realized_pnl_usd, b.result.net_realized_pnl_usd, 0.0);
    CHECK_CLOSE(a.result.portfolio_at_end.equity_usd, b.result.portfolio_at_end.equity_usd, 0.0);
}

// G. corporate actions pass straight through to EngineConfig.
void test_corporate_actions_pass_through() {
    constexpr std::uint32_t kEtf = 7;
    ContractRegistry registry;
    registry.add(ContractSpec{.instrument_id = kEtf, .raw_symbol = "XLF", .root_symbol = "EXLF",
                              .exchange = "ARCX", .tick_size = 0.01, .multiplier = 1.0,
                              .activation_ns = 1, .expiration_ns = kExp});
    std::vector<MarketBar> bars;
    for (int i = 1; i <= 6; ++i) {
        const double px = i < 4 ? 40.0 : 20.0;  // a 2-for-1 split at bar 4 halves the price
        bars.push_back(MarketBar{i * kMin, kEtf, px, px, px, px, 1000});
    }
    const std::vector<ScheduledTarget> rows{{.root_symbol = "EXLF", .ts_event_ns = bars[0].ts_event_ns,
                                             .target_units = 100}};
    const SplitAction split{.instrument_id = kEtf, .effective_ts_ns = bars[3].ts_event_ns, .ratio = 2.0};

    PaperTradingRunConfig cfg;
    cfg.risk_config = lenient_risk_config();
    cfg.commission_per_contract_usd = 0.0;
    const auto plain = run_paper_trading_backtest(bars, registry, rows, kFp, cfg);
    cfg.corporate_actions.splits[{kEtf, split.effective_ts_ns}] = split;
    const auto split_run = run_paper_trading_backtest(bars, registry, rows, kFp, cfg);

    const RegistryActiveContractResolver resolver(registry);
    const PortfolioRiskManager risk(cfg.risk_config, nullptr);
    EngineConfig config;
    config.execution.commission_per_contract_usd = 0.0;
    config.end_of_test = EndOfTestPolicy::LeaveOpen;
    config.corporate_actions.splits[{kEtf, split.effective_ts_ns}] = split;
    const ScheduledTargetStrategy strat(rows, kFp, ScheduledTargetStrategy::AbsentPolicy::NoDecision);
    const auto manual = BacktestEngine(registry, resolver, risk, config).run(bars, strat);

    CHECK(plain.result.portfolio_at_end.positions.size() == 1);
    CHECK(split_run.result.portfolio_at_end.positions.size() == 1);
    CHECK(plain.result.portfolio_at_end.positions[0].units == 100);
    CHECK(split_run.result.portfolio_at_end.positions[0].units == 200);
    CHECK(manual.portfolio_at_end.positions[0].units == 200);
    CHECK_CLOSE(split_run.result.portfolio_at_end.equity_usd, manual.portfolio_at_end.equity_usd, 0.0);
    // A split carries no economic PnL; without it the halved price is a real loss.
    CHECK_CLOSE(split_run.result.portfolio_at_end.equity_usd, 100'000.0, 1e-9);
    CHECK_CLOSE(plain.result.portfolio_at_end.equity_usd, 100'000.0 - 100 * 20.0, 1e-9);
}

}  // namespace

int main() {
    test_equivalent_to_manual_engine();
    test_kill_switch_differs_from_research_reference_path();
    test_default_end_of_test_is_leave_open();
    test_unresolved_instrument_throws();
    test_config_guards();
    test_deterministic();
    test_corporate_actions_pass_through();
    return quant::test::summary("test_paper_trading_run");
}
