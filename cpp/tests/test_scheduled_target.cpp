// Phase 11 / 11.1 -- ScheduledTargetStrategy + targets.csv boundary + explicit
// NO-DECISION semantics.
//
//   A. schedule row -> Signal at the decision bar
//   B. decision time != execution time: a row at T fills at the NEXT bar, not T
//   C. absent row -> NO DECISION (default): no Signal, no signal_id, no retry
//   D. absent row is NOT an implicit flat; an explicit target_units = 0 IS flat
//   E. NO DECISION consumes no signal_id and does not increment signals_generated
//   F. a risk-REJECTED target is NOT auto-retried after an absent row
//   G. an EXPLICIT repeated target row DOES create a new intent (a real retry)
//   H. a pending latency-delayed intent is not cancelled by a later NO DECISION
//   I. official realized PnL comes only from C++ Fill events
//   J. the mandatory RiskManager gate still applies (cannot be bypassed)
//   K. parse_targets_csv guards; RequireRow throws; deterministic replay
//   L. ReemitPreviousTarget is an explicit opt-in that DOES retry

#include "quant_core/contract.hpp"
#include "quant_core/contract_registry.hpp"
#include "quant_core/contract_selector.hpp"
#include "quant_core/engine.hpp"
#include "quant_core/events.hpp"
#include "quant_core/risk.hpp"
#include "quant_core/risk_manager.hpp"
#include "quant_core/scheduled_target_strategy.hpp"
#include "quant_core/types.hpp"

#include "test_support.hpp"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using namespace quant;

constexpr std::int64_t  kMin    = 60'000'000'000LL;
constexpr std::uint32_t kNqId   = 900001;
constexpr std::int64_t  kBigExp = 100'000LL * kMin;
const std::string kFp = "stratdsl1:deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef";
using AP = ScheduledTargetStrategy::AbsentPolicy;

ContractRegistry nq_registry() {
    ContractRegistry r;
    r.add(ContractSpec{.instrument_id = kNqId, .raw_symbol = "NQU6", .root_symbol = "NQ",
                       .exchange = "XCME", .tick_size = 0.25, .multiplier = 20.0,
                       .activation_ns = 1, .expiration_ns = kBigExp});
    return r;
}

MarketBar bar(std::int64_t idx, double open, double close) {
    const double hi = (open > close ? open : close) + 1.0;
    const double lo = (open < close ? open : close) - 1.0;
    return MarketBar{.ts_event_ns = idx * kMin, .instrument_id = kNqId,
                     .open = open, .high = hi, .low = lo, .close = close, .volume = 100};
}

std::vector<MarketBar> ramp(int n, double start) {
    std::vector<MarketBar> bars;
    for (int i = 1; i <= n; ++i) bars.push_back(bar(i, start + i, start + i));
    return bars;
}

std::string write_tmp(const std::string& name, const std::string& body) {
    static int counter = 0;
    const auto path = std::filesystem::temp_directory_path() /
                      ("qsched_" + std::to_string(++counter) + "_" + name);
    std::ofstream out(path);
    out << body;
    out.close();
    return path.string();
}

// A RiskManager that REJECTS every order (models a hard limit tripping).
struct RejectAllRisk final : RiskManager {
    RiskDecision review(const Order& order) const override {
        return RiskDecision{.ts_decision_ns = order.ts_created_ns, .order_id = order.order_id,
                            .instrument_id = order.instrument_id, .verdict = RiskVerdict::Reject,
                            .requested_quantity = order.quantity, .approved_quantity = 0,
                            .reason_code = "test_reject_all"};
    }
};

// ---- A / B: replay + decision-time vs execution-time --------------------------
void test_replay_and_execution_timing() {
    const auto reg = nq_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    const BacktestEngine engine(reg, resolver, risk, EngineConfig{});

    const auto bars = ramp(5, 100.0);  // ts = 1..5 minutes
    std::vector<ScheduledTarget> rows{{.root_symbol = "NQ", .ts_event_ns = 2 * kMin, .target_units = 1}};
    const ScheduledTargetStrategy strat(rows, kFp);  // default: NoDecision
    const auto r = engine.run(bars, strat);

    // Only the bar with a row makes a decision. The other 4 bars are NO DECISION.
    CHECK(r.signals_generated == 1);
    CHECK(strat.rows_applied() == 1);
    CHECK(r.fills_generated >= 1);
    // The Signal decided at ts==2min executes at the NEXT bar (ts==3min), never
    // at the decision bar and never pre-shifted earlier.
    CHECK(r.fills.front().ts_fill_ns == 3 * kMin);
    CHECK(r.fills.front().raw_symbol == "NQU6");
}

// ---- C / D: absent row is NO DECISION, not flat; explicit 0 is flat ----------
void test_absence_is_not_flat() {
    const auto reg = nq_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    const auto bars = ramp(6, 100.0);
    std::vector<ScheduledTarget> rows{{.root_symbol = "NQ", .ts_event_ns = 2 * kMin, .target_units = 1}};

    // NO DECISION (default): after the single +1 decision the strategy makes no
    // further decision. The position is held only because the engine has no new
    // intent -- there is no churn and only one strategy Signal total.
    {
        const BacktestEngine engine(reg, resolver, risk, EngineConfig{});
        const ScheduledTargetStrategy strat(rows, kFp);
        const auto r = engine.run(bars, strat);
        CHECK(r.signals_generated == 1);
        CHECK(r.orders_generated == 2);                // one open + one EOT close
        CHECK(r.fills_generated == 2);
        CHECK(r.fills.front().ts_fill_ns == 3 * kMin); // opened one bar after the row
        CHECK(r.fills.back().ts_fill_ns == 6 * kMin);  // held until the final engine event (EOT)
    }
    // Flat policy: a missing row is an explicit flat decision -> the position is
    // closed by a real signal-driven fill one bar after the last row, not at EOT.
    {
        const BacktestEngine engine(reg, resolver, risk, EngineConfig{});
        const ScheduledTargetStrategy strat(rows, kFp, AP::Flat);
        const auto r = engine.run(bars, strat);
        CHECK(r.signals_generated == 6);              // every bar decides (flat when absent)
        CHECK(r.fills_generated == 2);
        CHECK(r.fills.front().ts_fill_ns == 3 * kMin);
        CHECK(r.fills.back().ts_fill_ns == 4 * kMin); // closed well before EOT (ts=6min)
    }
    // explicit target_units = 0 row: an explicit FLAT that closes the long.
    {
        const BacktestEngine engine(reg, resolver, risk, EngineConfig{});
        std::vector<ScheduledTarget> flat_rows{
            {.root_symbol = "NQ", .ts_event_ns = 2 * kMin, .target_units = 1},
            {.root_symbol = "NQ", .ts_event_ns = 3 * kMin, .target_units = 0},
        };
        const ScheduledTargetStrategy strat(flat_rows, kFp);
        const auto r = engine.run(bars, strat);
        CHECK(r.signals_generated == 2);
        CHECK(r.closed_trades >= 1);
        CHECK(r.fills.back().ts_fill_ns == 4 * kMin);  // the explicit 0 closed the long at ts=4min
    }
}

// ---- E: NO DECISION consumes no signal_id / does not count -------------------
void test_no_decision_consumes_no_signal_id() {
    const auto reg = nq_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    const BacktestEngine engine(reg, resolver, risk, EngineConfig{});

    const auto bars = ramp(5, 100.0);
    // rows only on bars 1 and 3 (ts 1min, 3min); bars 2,4,5 are NO DECISION.
    std::vector<ScheduledTarget> rows{
        {.root_symbol = "NQ", .ts_event_ns = 1 * kMin, .target_units = 1},
        {.root_symbol = "NQ", .ts_event_ns = 3 * kMin, .target_units = -1},
    };
    const ScheduledTargetStrategy strat(rows, kFp);
    const auto r = engine.run(bars, strat);

    CHECK(r.signals_generated == 2);
    // The two strategy legs carry signal_id 1 then 2. If a NO-DECISION bar had
    // consumed an id, the bar-3 decision would be signal_id 3 (bars 1,2,3), not 2.
    CHECK(r.orders.size() >= 2);
    CHECK(r.orders[0].signal_id == 1);   // bar-1 decision, executes bar 2
    CHECK(r.orders[1].signal_id == 2);   // bar-3 decision, executes bar 4 (NOT id 3)
}

// ---- F: a risk-REJECTED target is not auto-retried after an absent row -------
void test_rejected_target_not_retried_on_absence() {
    const auto reg = nq_registry();
    const RegistryActiveContractResolver resolver(reg);
    const RejectAllRisk risk;
    const BacktestEngine engine(reg, resolver, risk, EngineConfig{});

    const auto bars = ramp(5, 100.0);
    // single decision at ts=2min; bars 3,4,5 have NO row.
    std::vector<ScheduledTarget> rows{{.root_symbol = "NQ", .ts_event_ns = 2 * kMin, .target_units = 1}};
    const ScheduledTargetStrategy strat(rows, kFp);
    const auto r = engine.run(bars, strat);

    CHECK(r.signals_generated == 1);       // exactly one strategy decision
    CHECK(r.orders_generated == 1);        // exactly one order attempt
    CHECK(r.risk_rejects == 1);            // ...which risk rejected
    CHECK(r.fills_generated == 0);         // no fill
    CHECK(r.risk_decisions.size() == 1);   // no SECOND attempt caused by schedule absence
}

// ---- G: an EXPLICIT repeated target row DOES create a new intent -------------
void test_explicit_repeat_target_does_retry() {
    const auto reg = nq_registry();
    const RegistryActiveContractResolver resolver(reg);
    const RejectAllRisk risk;
    const BacktestEngine engine(reg, resolver, risk, EngineConfig{});

    const auto bars = ramp(5, 100.0);
    // two EXPLICIT +1 rows: the strategy genuinely decides +1 twice.
    std::vector<ScheduledTarget> rows{
        {.root_symbol = "NQ", .ts_event_ns = 2 * kMin, .target_units = 1},
        {.root_symbol = "NQ", .ts_event_ns = 3 * kMin, .target_units = 1},
    };
    const ScheduledTargetStrategy strat(rows, kFp);
    const auto r = engine.run(bars, strat);

    CHECK(r.signals_generated == 2);
    CHECK(r.orders_generated == 2);        // a real second attempt (a new explicit decision)
    CHECK(r.risk_rejects == 2);
}

// ---- H: a pending latency-delayed intent is not cancelled by NO DECISION -----
void test_pending_intent_not_cancelled_by_no_decision() {
    const auto reg = nq_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    EngineConfig cfg;
    cfg.latency_bars = 2;  // a decision on bar i executes on bar i+3
    const BacktestEngine engine(reg, resolver, risk, cfg);

    const auto bars = ramp(8, 100.0);
    // one decision at ts=2min; it becomes eligible only at ts=5min. Bars 3 & 4
    // (between decision and execution) are NO DECISION -- they must not cancel it.
    std::vector<ScheduledTarget> rows{{.root_symbol = "NQ", .ts_event_ns = 2 * kMin, .target_units = 1}};
    const ScheduledTargetStrategy strat(rows, kFp);
    const auto r = engine.run(bars, strat);

    CHECK(r.signals_generated == 1);
    CHECK(r.fills_generated >= 1);
    CHECK(r.fills.front().ts_fill_ns == 5 * kMin);  // decided bar-2, +2 latency -> bar-5
}

// ---- I: official PnL is Fill-derived -------------------------------------
void test_official_pnl_from_fills() {
    const auto reg = nq_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    EngineConfig cfg;
    cfg.execution.commission_per_contract_usd = 2.0;
    const BacktestEngine engine(reg, resolver, risk, cfg);

    const auto bars = ramp(5, 100.0);  // opens 101..105
    std::vector<ScheduledTarget> rows{
        {.root_symbol = "NQ", .ts_event_ns = 2 * kMin, .target_units = 1},  // fill @ bar3 open = 103
        {.root_symbol = "NQ", .ts_event_ns = 4 * kMin, .target_units = 0},  // fill @ bar5 open = 105
    };
    const ScheduledTargetStrategy strat(rows, kFp);
    const auto r = engine.run(bars, strat);

    CHECK(r.fills_generated == 2);
    CHECK_CLOSE(r.gross_realized_pnl_usd, 40.0, 1e-6);  // (105-103)*20
    CHECK_CLOSE(r.costs_usd, 4.0, 1e-6);                // 2 fills * 2.0
    CHECK_CLOSE(r.net_realized_pnl_usd, 36.0, 1e-6);
}

// ---- J: the risk gate cannot be bypassed ----------------------------------
void test_risk_gate_applies() {
    const auto reg = nq_registry();
    const RegistryActiveContractResolver resolver(reg);
    RiskLimits limits;
    limits.max_contracts_per_symbol = 1;
    const MaxContractsRiskManager risk(limits);
    const BacktestEngine engine(reg, resolver, risk, EngineConfig{});

    const auto bars = ramp(5, 100.0);
    std::vector<ScheduledTarget> rows{{.root_symbol = "NQ", .ts_event_ns = 2 * kMin, .target_units = 5}};
    const ScheduledTargetStrategy strat(rows, kFp);
    const auto r = engine.run(bars, strat);

    CHECK(r.risk_resizes >= 1);
    CHECK(!r.risk_decisions.empty());
    for (const auto& f : r.fills) CHECK(f.quantity <= 1);
}

// ---- K: parsing guards + RequireRow + deterministic replay ------------------
void test_targets_csv_guards() {
    std::string fp;

    const std::string good =
        "ts_event_ns,root_symbol,target_units,strategy_fingerprint,matched_rule_id\n"
        "120000000000,NQ,1," + kFp + ",tsmom_long\n"
        "240000000000,NQ,0," + kFp + ",\n";
    const auto rows = parse_targets_csv(write_tmp("good.csv", good), fp);
    CHECK(rows.size() == 2);
    CHECK(fp == kFp);
    CHECK(rows[0].target_units == 1);

    const std::string with_price =
        "ts_event_ns,root_symbol,target_units,strategy_fingerprint,fill_price\n"
        "120000000000,NQ,1," + kFp + ",29000.0\n";
    CHECK_THROWS(parse_targets_csv(write_tmp("price.csv", with_price), fp));

    const std::string raw_root =
        "ts_event_ns,root_symbol,target_units,strategy_fingerprint,matched_rule_id\n"
        "120000000000,NQU6,1," + kFp + ",\n";
    CHECK_THROWS(parse_targets_csv(write_tmp("rawroot.csv", raw_root), fp));

    const std::string mixed =
        "ts_event_ns,root_symbol,target_units,strategy_fingerprint,matched_rule_id\n"
        "120000000000,NQ,1," + kFp + ",\n"
        "240000000000,NQ,-1,stratdsl1:0000000000000000000000000000000000000000000000000000000000000000,\n";
    CHECK_THROWS(parse_targets_csv(write_tmp("mixed.csv", mixed), fp));

    const std::string frac =
        "ts_event_ns,root_symbol,target_units,strategy_fingerprint,matched_rule_id\n"
        "120000000000,NQ,1.0," + kFp + ",\n";
    CHECK_THROWS(parse_targets_csv(write_tmp("frac.csv", frac), fp));

    // policy parsing
    CHECK(parse_absent_policy("") == AP::NoDecision);
    CHECK(parse_absent_policy("no_decision") == AP::NoDecision);
    CHECK(parse_absent_policy("flat") == AP::Flat);
    CHECK(parse_absent_policy("require_row") == AP::RequireRow);
    CHECK(parse_absent_policy("reemit_previous") == AP::ReemitPreviousTarget);
    CHECK_THROWS(parse_absent_policy("hold_previous"));  // the old default name is gone
}

void test_require_row_and_replay() {
    const auto reg = nq_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    const auto bars = ramp(4, 100.0);
    std::vector<ScheduledTarget> rows{{.root_symbol = "NQ", .ts_event_ns = 2 * kMin, .target_units = 1}};

    {
        const BacktestEngine engine(reg, resolver, risk, EngineConfig{});
        const ScheduledTargetStrategy strat(rows, kFp, AP::RequireRow);
        CHECK_THROWS(engine.run(bars, strat));  // bar 1 has no row
    }

    std::vector<ScheduledTarget> dense;
    for (int i = 1; i <= 4; ++i)
        dense.push_back({.root_symbol = "NQ", .ts_event_ns = i * kMin, .target_units = (i % 2 ? 1 : -1)});
    const BacktestEngine engine(reg, resolver, risk, EngineConfig{});
    const ScheduledTargetStrategy s1(dense, kFp);
    const ScheduledTargetStrategy s2(dense, kFp);
    const auto a = engine.run(bars, s1);
    const auto b = engine.run(bars, s2);
    CHECK(a.fills_generated == b.fills_generated);
    CHECK_CLOSE(a.net_realized_pnl_usd, b.net_realized_pnl_usd, 1e-9);
    CHECK(a.orders_generated == b.orders_generated);
    CHECK(a.signals_generated == b.signals_generated);
}

// ---- L: ReemitPreviousTarget is an explicit opt-in that DOES retry ----------
void test_reemit_previous_is_explicit_retry() {
    const auto reg = nq_registry();
    const RegistryActiveContractResolver resolver(reg);
    const RejectAllRisk risk;
    const BacktestEngine engine(reg, resolver, risk, EngineConfig{});

    const auto bars = ramp(5, 100.0);
    std::vector<ScheduledTarget> rows{{.root_symbol = "NQ", .ts_event_ns = 2 * kMin, .target_units = 1}};
    const ScheduledTargetStrategy strat(rows, kFp, AP::ReemitPreviousTarget);
    const auto r = engine.run(bars, strat);

    // every bar decides: bar 1 re-emits the seed (flat), bar 2 decides +1, bars
    // 3,4,5 re-emit +1 as NEW signals -> 5 decisions. Each +1 intent that reaches
    // an execution bar produces a fresh (rejected) order -- proof that
    // ReemitPreviousTarget retries, unlike the default NoDecision.
    CHECK(r.signals_generated == 5);
    CHECK(r.orders_generated >= 3);
    CHECK(r.risk_rejects == r.orders_generated);
    CHECK(r.fills_generated == 0);
}

}  // namespace

int main() {
    test_replay_and_execution_timing();
    test_absence_is_not_flat();
    test_no_decision_consumes_no_signal_id();
    test_rejected_target_not_retried_on_absence();
    test_explicit_repeat_target_does_retry();
    test_pending_intent_not_cancelled_by_no_decision();
    test_official_pnl_from_fills();
    test_risk_gate_applies();
    test_targets_csv_guards();
    test_require_row_and_replay();
    test_reemit_previous_is_explicit_retry();
    return quant::test::summary("test_scheduled_target");
}
