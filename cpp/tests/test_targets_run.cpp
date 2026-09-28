// Phase 19 -- quant::run_targets_backtest (quant_core/targets_run.hpp).
//
// The thin scheduled-target run helper the pybind11 fast boundary
// (bindings/pybind_module.cpp) calls. Its engine wiring MIRRORS the frozen
// reference CLI apps/backtest_targets_csv.cpp; it does NOT share code with the
// CLI, which is wired independently and unchanged. These checks pin the helper
// itself; the CLI-vs-pybind parity tests
// (tests/python/test_phase_19_pybind.py) are what hold the two transports in
// agreement.
//
//   A. it is EQUIVALENT to building BacktestEngine + PassThroughRiskManager +
//      EngineConfig{ForceLiquidateFinalClose, commission} by hand and calling
//      run() -- i.e. it adds no execution / accounting semantics of its own;
//   B. the contract-resolution hard check fires on an unknown / out-of-window
//      instrument_id;
//   C. cost overrides must be non-negative; validation-day boundaries must be
//      strictly ascending;
//   D. an empty validation-day vector => "utc_day" basis; a non-empty one =>
//      "trading_day" basis and one daily_equity point per boundary;
//   E. auxiliary roll close_marks reach the roll close-leg (accounted as
//      contemporaneous, from the auxiliary marks);
//   F. deterministic: identical inputs -> identical result.

#include "quant_core/contract.hpp"
#include "quant_core/contract_registry.hpp"
#include "quant_core/contract_selector.hpp"
#include "quant_core/engine.hpp"
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
constexpr std::uint32_t kM6   = 10;
constexpr std::uint32_t kU6   = 11;
constexpr std::int64_t  kExp6 = 5'000LL * kMin;
const std::string kFp = "stratdsl1:deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef";

ContractRegistry nq_registry() {
    ContractRegistry r;
    r.add(ContractSpec{.instrument_id = kM6, .raw_symbol = "NQM6", .root_symbol = "NQ",
                       .exchange = "XCME", .tick_size = 0.25, .multiplier = 20.0,
                       .activation_ns = 1, .expiration_ns = kExp6});
    r.add(ContractSpec{.instrument_id = kU6, .raw_symbol = "NQU6", .root_symbol = "NQ",
                       .exchange = "XCME", .tick_size = 0.25, .multiplier = 20.0,
                       .activation_ns = 1, .expiration_ns = kExp6 + 90LL * kMin});
    return r;
}

MarketBar bar(std::uint32_t id, std::int64_t idx, double close) {
    return MarketBar{.ts_event_ns = idx * kMin, .instrument_id = id,
                     .open = close, .high = close + 1.0, .low = close - 1.0,
                     .close = close, .volume = 100};
}

std::vector<MarketBar> single_contract_bars() {
    std::vector<MarketBar> b;
    for (int i = 1; i <= 12; ++i) b.push_back(bar(kU6, i, 100.0 + i));
    return b;
}

std::vector<ScheduledTarget> dense_long(const std::vector<MarketBar>& bars) {
    std::vector<ScheduledTarget> rows;
    for (std::size_t i = 0; i + 1 < bars.size(); ++i) {
        rows.push_back(ScheduledTarget{.root_symbol = "NQ",
                                       .ts_event_ns = bars[i].ts_event_ns,
                                       .target_units = 1});
    }
    return rows;
}

// A. equivalence to the hand-wired engine path.
void test_equivalent_to_manual_engine() {
    const auto bars = single_contract_bars();
    const auto registry = nq_registry();

    // hand-wired, mirroring apps/backtest_targets_csv.cpp
    const RegistryActiveContractResolver resolver(registry);
    const PassThroughRiskManager risk;
    EngineConfig config;
    config.execution.commission_per_contract_usd = 2.0;
    config.end_of_test = EndOfTestPolicy::ForceLiquidateFinalClose;
    const ScheduledTargetStrategy strat(dense_long(bars), kFp,
                                        ScheduledTargetStrategy::AbsentPolicy::NoDecision);
    const BacktestEngine engine(registry, resolver, risk, config);
    const auto manual = engine.run(bars, strat);

    TargetsRunConfig cfg;  // defaults == commission 2.0, no slippage/spread
    const auto out = run_targets_backtest(bars, registry, dense_long(bars), kFp, cfg);

    CHECK(out.result.fills_generated == manual.fills_generated);
    CHECK(out.result.closed_trades == manual.closed_trades);
    CHECK(out.result.signals_generated == manual.signals_generated);
    CHECK_CLOSE(out.result.gross_realized_pnl_usd, manual.gross_realized_pnl_usd, 1e-9);
    CHECK_CLOSE(out.result.costs_usd, manual.costs_usd, 1e-9);
    CHECK_CLOSE(out.result.net_realized_pnl_usd, manual.net_realized_pnl_usd, 1e-9);
    CHECK_CLOSE(out.result.net_equity_usd_at_end, manual.net_equity_usd_at_end, 1e-9);
    CHECK(out.result.daily_equity.size() == manual.daily_equity.size());
    CHECK(out.contracts_resolved == 1);
    CHECK(out.target_rows == dense_long(bars).size());
    CHECK(out.daily_equity_basis == "utc_day");
    CHECK(out.commission_per_contract_usd == 2.0);
    CHECK(out.n_bars == bars.size());
}

// B. contract-resolution hard check.
void test_unresolved_instrument_throws() {
    auto bars = single_contract_bars();
    bars.back().instrument_id = 999999;  // not in the registry
    const auto registry = nq_registry();
    CHECK_THROWS(run_targets_backtest(bars, registry, {}, kFp, TargetsRunConfig{}));

    // out-of-window: a bar after expiry of its contract
    auto bars2 = single_contract_bars();
    bars2.back().ts_event_ns = kExp6 + 10'000LL * kMin;
    CHECK_THROWS(run_targets_backtest(bars2, registry, {}, kFp, TargetsRunConfig{}));
}

// C. config guards.
void test_config_guards() {
    const auto bars = single_contract_bars();
    const auto registry = nq_registry();

    TargetsRunConfig neg;
    neg.commission_per_contract_usd = -1.0;
    CHECK_THROWS(run_targets_backtest(bars, registry, {}, kFp, neg));

    TargetsRunConfig unsorted;
    unsorted.validation_day_boundaries_ns = {5 * kMin, 3 * kMin};
    CHECK_THROWS(run_targets_backtest(bars, registry, {}, kFp, unsorted));
}

// D. daily-equity basis switches with the validation-day vector.
void test_daily_equity_basis() {
    const auto bars = single_contract_bars();
    const auto registry = nq_registry();

    TargetsRunConfig td;
    for (const auto& b : bars) td.validation_day_boundaries_ns.push_back(b.ts_event_ns);
    const auto out = run_targets_backtest(bars, registry, dense_long(bars), kFp, td);
    CHECK(out.daily_equity_basis == "trading_day");
    CHECK(out.result.daily_equity.size() == bars.size());
}

// E. auxiliary roll close_marks price a deferred roll contemporaneously.
void test_auxiliary_roll_marks() {
    std::vector<MarketBar> bars;
    for (int i = 1; i <= 3; ++i) bars.push_back(bar(kM6, i, 100.0 + i));
    for (int i = 4; i <= 7; ++i) bars.push_back(bar(kU6, i, 200.0 + i));
    const auto registry = nq_registry();

    std::vector<ScheduledTarget> rows;
    for (std::size_t i = 0; i + 1 < bars.size(); ++i)
        rows.push_back(ScheduledTarget{.root_symbol = "NQ", .ts_event_ns = bars[i].ts_event_ns,
                                       .target_units = 1});

    TargetsRunConfig base;
    for (const auto& b : bars) base.validation_day_boundaries_ns.push_back(b.ts_event_ns);

    auto no_marks = run_targets_backtest(bars, registry, rows, kFp, base);
    CHECK(no_marks.result.rolls_deferred >= 1);
    CHECK(no_marks.result.rolls_priced_contemporaneous == 0);

    TargetsRunConfig with_marks = base;
    // NQM6 close at exactly the first NQU6 bar's ts (the roll execution instant)
    with_marks.roll_close_marks[{kM6, bars[3].ts_event_ns}] = 102.5;
    auto marked = run_targets_backtest(bars, registry, rows, kFp, with_marks);
    CHECK(marked.result.rolls == 1);
    CHECK(marked.result.rolls_priced_contemporaneous == 1);
    CHECK(marked.result.rolls_priced_auxiliary_marks == 1);
}

// F. determinism.
void test_deterministic() {
    const auto bars = single_contract_bars();
    const auto registry = nq_registry();
    const auto a = run_targets_backtest(bars, registry, dense_long(bars), kFp, TargetsRunConfig{});
    const auto b = run_targets_backtest(bars, registry, dense_long(bars), kFp, TargetsRunConfig{});
    CHECK(a.result.fills_generated == b.result.fills_generated);
    CHECK_CLOSE(a.result.net_realized_pnl_usd, b.result.net_realized_pnl_usd, 0.0);
    CHECK_CLOSE(a.result.net_equity_usd_at_end, b.result.net_equity_usd_at_end, 0.0);
}

}  // namespace

int main() {
    test_equivalent_to_manual_engine();
    test_unresolved_instrument_throws();
    test_config_guards();
    test_daily_equity_basis();
    test_auxiliary_roll_marks();
    test_deterministic();
    return quant::test::summary("test_targets_run");
}
