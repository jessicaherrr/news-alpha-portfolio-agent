// News Alpha Phase H acceptance patch -- ExecutionConfig::commission_per_contract_usd_by_root.
//
//   A. an EMPTY schedule is the legacy scalar path, fill for fill;
//   B. a futures root is charged per contract, an ETF root per share, an
//      unlisted root the scalar;
//   C. a mixed futures + ETF book through the full engine charges each root its
//      own rate (and the old scalar would have charged shares a contract rate);
//   D. a futures roll never changes the rate: every raw contract of a root --
//      the entry, the roll close-leg and the re-open -- is charged the root's rate;
//   E. deterministic; F. a negative rate is refused.

#include "quant_core/contract.hpp"
#include "quant_core/contract_registry.hpp"
#include "quant_core/engine.hpp"
#include "quant_core/events.hpp"
#include "quant_core/execution_simulator.hpp"
#include "quant_core/paper_trading_run.hpp"
#include "quant_core/risk_config.hpp"
#include "quant_core/scheduled_target_strategy.hpp"
#include "quant_core/types.hpp"

#include "test_support.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace quant;

constexpr std::int64_t  kMin  = 60'000'000'000LL;
constexpr std::uint32_t kM6   = 10;
constexpr std::uint32_t kU6   = 11;
constexpr std::uint32_t kXlf  = 4'000'000'011;
constexpr std::int64_t  kExp6 = 5'000LL * kMin;
constexpr std::int64_t  kFar  = 1'000'000LL * kMin;
const std::string kFp = "portstrat1:deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef";

ContractRegistry mixed_registry() {
    ContractRegistry r;
    r.add(ContractSpec{.instrument_id = kM6, .raw_symbol = "NQM6", .root_symbol = "NQ", .exchange = "XCME",
                       .tick_size = 0.25, .multiplier = 20.0, .activation_ns = 1, .expiration_ns = kExp6});
    r.add(ContractSpec{.instrument_id = kU6, .raw_symbol = "NQU6", .root_symbol = "NQ", .exchange = "XCME",
                       .tick_size = 0.25, .multiplier = 20.0, .activation_ns = 1, .expiration_ns = kFar});
    r.add(ContractSpec{.instrument_id = kXlf, .raw_symbol = "XLF", .root_symbol = "EXLF", .exchange = "ARCX",
                       .tick_size = 0.01, .multiplier = 1.0, .activation_ns = 1, .expiration_ns = kFar});
    return r;
}

MarketBar bar(std::uint32_t id, std::int64_t idx, double px) {
    return MarketBar{.ts_event_ns = idx * kMin, .instrument_id = id, .open = px, .high = px, .low = px,
                     .close = px, .volume = 1'000'000};
}

Order order(const ContractRegistry& reg, std::uint32_t id, int qty) {
    Order o;
    o.order_id = 1;
    o.signal_id = 1;
    o.ts_created_ns = kMin;
    o.instrument_id = id;
    o.raw_symbol = reg.by_instrument_id(id).raw_symbol;
    o.side = Side::Buy;
    o.quantity = qty;
    o.order_type = OrderType::Market;
    o.tif = TimeInForce::Day;
    return o;
}

double fill_commission(const ContractRegistry& reg, const ExecutionConfig& cfg, std::uint32_t id, int qty,
                       double px) {
    const BarExecutionSimulator sim(reg, cfg);
    ExecutionRequest req;
    req.bar = bar(id, 2, px);
    req.ts_ns = 2 * kMin;
    const auto report = sim.execute(order(reg, id, qty), req, 1);
    if (!report.fill) quant::test::fail("expected a fill", __FILE__, __LINE__);
    return report.fill->commission_usd;
}

RiskConfig lenient() {
    RiskConfig cfg;
    cfg.max_contracts_per_symbol = 1'000'000;
    cfg.max_contracts_per_root = 1'000'000;
    cfg.max_gross_contracts = 1'000'000;
    cfg.missing_margin = MissingMarginPolicy::TreatAsZero;
    cfg.stale_mark = StaleMarkPolicy::Ignore;
    cfg.portfolio.starting_capital_usd = 1'000'000.0;
    cfg.portfolio.mark_staleness_tolerance_ns = 10 * kMin;
    return cfg;
}

// A. empty schedule == the legacy scalar, for every root.
void test_empty_schedule_is_the_scalar_path() {
    const auto reg = mixed_registry();
    ExecutionConfig scalar;
    scalar.commission_per_contract_usd = 2.0;
    ExecutionConfig empty = scalar;  // schedule present but empty
    CHECK(empty.commission_per_contract_usd_by_root.empty());
    for (const auto& [id, qty, px] : {std::tuple{kM6, 3, 20000.0}, std::tuple{kXlf, 1000, 34.25}}) {
        const double a = fill_commission(reg, scalar, id, qty, px);
        const double b = fill_commission(reg, empty, id, qty, px);
        CHECK(a == b);
        CHECK(a == 2.0 * qty);
    }
}

// B. per-contract futures, per-share ETF, scalar for an unlisted root.
void test_per_root_rates() {
    const auto reg = mixed_registry();
    ExecutionConfig cfg;
    cfg.commission_per_contract_usd = 7.0;
    cfg.commission_per_contract_usd_by_root = {{"NQ", 2.0}, {"EXLF", 0.005}};
    CHECK_CLOSE(fill_commission(reg, cfg, kM6, 3, 20000.0), 3 * 2.0, 1e-12);
    CHECK_CLOSE(fill_commission(reg, cfg, kXlf, 1000, 34.25), 1000 * 0.005, 1e-12);
    ExecutionConfig only_nq;
    only_nq.commission_per_contract_usd = 7.0;
    only_nq.commission_per_contract_usd_by_root = {{"NQ", 2.0}};
    CHECK_CLOSE(fill_commission(reg, only_nq, kXlf, 10, 34.25), 10 * 7.0, 1e-12);  // generic default
}

PaperTradingRunConfig run_config(std::map<std::string, double> schedule, double scalar) {
    PaperTradingRunConfig cfg;
    cfg.risk_config = lenient();
    cfg.commission_per_contract_usd = scalar;
    cfg.commission_per_contract_usd_by_root = std::move(schedule);
    return cfg;
}

// C. a mixed book through the full engine.
void test_mixed_book_charges_each_root_its_own_rate() {
    const auto reg = mixed_registry();
    std::vector<MarketBar> bars;
    for (int i = 1; i <= 4; ++i) {
        bars.push_back(bar(kU6, i, 20000.0));
        bars.push_back(bar(kXlf, i, 34.0));
    }
    const std::vector<ScheduledTarget> rows{{.root_symbol = "NQ", .ts_event_ns = kMin, .target_units = 2},
                                            {.root_symbol = "EXLF", .ts_event_ns = kMin, .target_units = 1000}};
    const auto priced = run_paper_trading_backtest(bars, reg, rows, kFp,
                                                   run_config({{"NQ", 2.0}, {"EXLF", 0.005}}, 0.0));
    CHECK(priced.result.fills_generated == 2);
    CHECK_CLOSE(priced.result.costs_usd, 2 * 2.0 + 1000 * 0.005, 1e-9);
    for (const auto& f : priced.result.fills) {
        CHECK_CLOSE(f.commission_usd, (f.instrument_id == kXlf ? 0.005 : 2.0) * f.quantity, 1e-12);
    }
    // the gap this patch closes: one scalar charges 1000 shares a contract rate
    const auto scalar = run_paper_trading_backtest(bars, reg, rows, kFp, run_config({}, 2.0));
    CHECK_CLOSE(scalar.result.costs_usd, 2 * 2.0 + 1000 * 2.0, 1e-9);
}

// D. a roll keeps the root's rate on every raw contract.
void test_roll_keeps_the_root_rate() {
    const auto reg = mixed_registry();
    std::vector<MarketBar> bars;
    for (int i = 1; i <= 3; ++i) bars.push_back(bar(kM6, i, 100.0 + i));
    for (int i = 4; i <= 7; ++i) bars.push_back(bar(kU6, i, 200.0 + i));
    std::vector<ScheduledTarget> rows;
    for (std::size_t i = 0; i + 1 < bars.size(); ++i) {
        rows.push_back(ScheduledTarget{.root_symbol = "NQ", .ts_event_ns = bars[i].ts_event_ns, .target_units = 3});
    }
    auto cfg = run_config({{"NQ", 1.25}}, 99.0);
    cfg.roll_close_marks[{kM6, bars[3].ts_event_ns}] = 102.5;
    const auto out = run_paper_trading_backtest(bars, reg, rows, kFp, cfg);
    CHECK(out.result.rolls == 1);
    CHECK(out.result.fills_generated >= 3);  // entry (M6), roll close-leg (M6), re-open (U6)
    bool saw_m6 = false, saw_u6 = false;
    for (const auto& f : out.result.fills) {
        CHECK_CLOSE(f.commission_usd, 1.25 * f.quantity, 1e-12);
        saw_m6 = saw_m6 || f.instrument_id == kM6;
        saw_u6 = saw_u6 || f.instrument_id == kU6;
    }
    CHECK(saw_m6 && saw_u6);
}

// E. determinism.
void test_deterministic() {
    const auto reg = mixed_registry();
    std::vector<MarketBar> bars;
    for (int i = 1; i <= 5; ++i) {
        bars.push_back(bar(kU6, i, 20000.0 + i));
        bars.push_back(bar(kXlf, i, 34.0 + 0.01 * i));
    }
    const std::vector<ScheduledTarget> rows{{.root_symbol = "NQ", .ts_event_ns = kMin, .target_units = 1},
                                            {.root_symbol = "EXLF", .ts_event_ns = 2 * kMin, .target_units = 500}};
    const auto cfg = run_config({{"NQ", 2.0}, {"EXLF", 0.005}}, 0.0);
    const auto a = run_paper_trading_backtest(bars, reg, rows, kFp, cfg);
    const auto b = run_paper_trading_backtest(bars, reg, rows, kFp, cfg);
    CHECK(a.result.fills_generated == b.result.fills_generated);
    CHECK_CLOSE(a.result.costs_usd, b.result.costs_usd, 0.0);
    CHECK_CLOSE(a.result.portfolio_at_end.equity_usd, b.result.portfolio_at_end.equity_usd, 0.0);
}

// F. a negative rate is refused before any bar is replayed.
void test_negative_rate_refused() {
    const auto reg = mixed_registry();
    const std::vector<MarketBar> bars{bar(kU6, 1, 20000.0), bar(kU6, 2, 20000.0)};
    const std::vector<ScheduledTarget> rows{{.root_symbol = "NQ", .ts_event_ns = kMin, .target_units = 1}};
    CHECK_THROWS(run_paper_trading_backtest(bars, reg, rows, kFp, run_config({{"NQ", -1.0}}, 0.0)));
}

}  // namespace

int main() {
    try {
        test_empty_schedule_is_the_scalar_path();
        test_per_root_rates();
        test_mixed_book_charges_each_root_its_own_rate();
        test_roll_keeps_the_root_rate();
        test_deterministic();
        test_negative_rate_refused();
    } catch (const std::exception& e) {
        quant::test::fail("uncaught exception", __FILE__, __LINE__, e.what());
    }
    return quant::test::summary("test_commission_schedule");
}
