// Phase 11.2 -- execution-latency pending-intent queue invariant.
//
// EngineConfig.latency_bars > 0: a Signal decided on bar i of a root executes on
// bar i + 1 + latency_bars of that root. Every emitted Signal is an INDEPENDENT
// intent -- a later Signal must never silently overwrite an earlier not-yet-due
// one. The engine keeps a per-root FIFO std::deque<PendingIntent> (Signal +
// due-bar index only; NO execution price / contract identity / risk state) and
// drains it front-first when due. Contract resolution + order delta stay
// execution-time.
//
//   A. three consecutive DISTINCT Signals with latency 2 all survive
//   B. execution order is decision order, deterministically
//   C. a later Signal does not overwrite an earlier pending Signal
//   D. NO DECISION between queued Signals does not cancel them
//   E. signal ids stay decision-order monotonic (no phantom ids)
//   F. order delta uses the EXECUTION-TIME position, not a decision-time quantity
//   G. a pending Signal crossing a roll resolves the EXECUTION-TIME contract
//   H. risk rejection of one queued intent does not remove later queued intents
//   I. deterministic replay under latency
//   J. latency_bars == 0 behaviour is unchanged

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
#include <string>
#include <vector>

namespace {

using namespace quant;

constexpr std::int64_t  kMin    = 60'000'000'000LL;
constexpr std::uint32_t kNqM6   = 500001;
constexpr std::uint32_t kNqU6   = 500002;
constexpr std::int64_t  kBigExp = 1'000'000LL * kMin;
const std::string kFp = "stratdsl1:" + std::string(64, 'a');
using AP = ScheduledTargetStrategy::AbsentPolicy;

ContractRegistry one_nq() {
    ContractRegistry r;
    r.add(ContractSpec{.instrument_id = kNqM6, .raw_symbol = "NQM6", .root_symbol = "NQ",
                       .exchange = "XCME", .tick_size = 0.25, .multiplier = 20.0,
                       .activation_ns = 1, .expiration_ns = kBigExp});
    return r;
}

// NQM6 and NQU6 both live across the whole window -- a genuine roll overlap.
ContractRegistry roll_nq() {
    ContractRegistry r;
    r.add(ContractSpec{.instrument_id = kNqM6, .raw_symbol = "NQM6", .root_symbol = "NQ",
                       .exchange = "XCME", .tick_size = 0.25, .multiplier = 20.0,
                       .activation_ns = 1, .expiration_ns = kBigExp});
    r.add(ContractSpec{.instrument_id = kNqU6, .raw_symbol = "NQU6", .root_symbol = "NQ",
                       .exchange = "XCME", .tick_size = 0.25, .multiplier = 20.0,
                       .activation_ns = 1, .expiration_ns = kBigExp});
    return r;
}

MarketBar bar(std::int64_t idx, std::uint32_t instr, double px) {
    return MarketBar{.ts_event_ns = idx * kMin, .instrument_id = instr,
                     .open = px, .high = px + 1.0, .low = px - 1.0, .close = px, .volume = 100};
}

std::vector<MarketBar> ramp(int n, std::uint32_t instr, double start) {
    std::vector<MarketBar> b;
    for (int i = 1; i <= n; ++i) b.push_back(bar(i, instr, start + i));
    return b;
}

std::vector<ScheduledTarget> rows(std::initializer_list<std::pair<int, int>> ts_target) {
    std::vector<ScheduledTarget> out;
    for (const auto& [ts_idx, tgt] : ts_target)
        out.push_back({.root_symbol = "NQ", .ts_event_ns = ts_idx * kMin, .target_units = tgt});
    return out;
}

// Rejects the first N orders it sees, approves the rest unchanged.
struct RejectFirstN final : RiskManager {
    explicit RejectFirstN(int n) : n_(n) {}
    RiskDecision review(const Order& o) const override {
        const bool reject = seen_++ < n_;
        return RiskDecision{.ts_decision_ns = o.ts_created_ns, .order_id = o.order_id,
                            .instrument_id = o.instrument_id,
                            .verdict = reject ? RiskVerdict::Reject : RiskVerdict::Approve,
                            .requested_quantity = o.quantity,
                            .approved_quantity = reject ? 0 : o.quantity,
                            .reason_code = reject ? "test_reject_first_n" : "ok"};
    }
    int n_;
    mutable int seen_{0};
};

// ---- A / B / C / E / I: three distinct consecutive signals all survive -------
void test_three_consecutive_signals_all_survive() {
    const auto reg = one_nq();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    EngineConfig cfg;
    cfg.latency_bars = 2;
    cfg.end_of_test = EndOfTestPolicy::LeaveOpen;

    const auto bars = ramp(8, kNqM6, 100.0);  // opens 101..108, ts 1..8 min
    // decisions on bars 1,2,3 (ts 1,2,3 min): +1, -1, 0
    const ScheduledTargetStrategy strat(rows({{1, 1}, {2, -1}, {3, 0}}), kFp);

    const auto a = BacktestEngine(reg, resolver, risk, cfg).run(bars, strat);
    const ScheduledTargetStrategy strat2(rows({{1, 1}, {2, -1}, {3, 0}}), kFp);
    const auto b = BacktestEngine(reg, resolver, risk, cfg).run(bars, strat2);

    // latency 2: decided bar i -> executes bar i+3. i=1->4, i=2->5, i=3->6.
    CHECK(a.fills_generated == 3);                       // all three intents executed
    CHECK(a.fills[0].ts_fill_ns == 4 * kMin);
    CHECK(a.fills[1].ts_fill_ns == 5 * kMin);
    CHECK(a.fills[2].ts_fill_ns == 6 * kMin);
    // execution order == decision order; deltas: 0->+1 (Buy 1), +1->-1 (Sell 2),
    // -1->0 (Buy 1) -- a later signal did NOT overwrite an earlier one.
    CHECK(a.fills[0].side == Side::Buy  && a.fills[0].quantity == 1);
    CHECK(a.fills[1].side == Side::Sell && a.fills[1].quantity == 2);
    CHECK(a.fills[2].side == Side::Buy  && a.fills[2].quantity == 1);
    // signal ids: decision-order monotonic 1,2,3 (bars 4..8 make no decision -> no ids)
    CHECK(a.signals_generated == 3);
    CHECK(a.orders.size() == 3);
    CHECK(a.orders[0].signal_id == 1);
    CHECK(a.orders[1].signal_id == 2);
    CHECK(a.orders[2].signal_id == 3);
    // deterministic replay
    CHECK(a.fills_generated == b.fills_generated);
    CHECK_CLOSE(a.net_realized_pnl_usd, b.net_realized_pnl_usd, 1e-9);
    CHECK(a.fills[2].fill_id == b.fills[2].fill_id);
}

// ---- D: NO DECISION between queued signals does not cancel them --------------
void test_no_decision_between_queued_signals() {
    const auto reg = one_nq();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    EngineConfig cfg;
    cfg.latency_bars = 3;                     // wide gap between decision and execution
    cfg.end_of_test = EndOfTestPolicy::LeaveOpen;

    const auto bars = ramp(10, kNqM6, 100.0);
    // decisions only on bars 1 and 2; bars 3..10 are NO DECISION (default policy).
    const ScheduledTargetStrategy strat(rows({{1, 1}, {2, 2}}), kFp);
    const auto r = BacktestEngine(reg, resolver, risk, cfg).run(bars, strat);

    CHECK(r.signals_generated == 2);          // NO DECISION consumed no id / no count
    // both intents survive the NO-DECISION bars in between: i=1->exec bar5, i=2->exec bar6
    CHECK(r.fills_generated == 2);
    CHECK(r.fills[0].ts_fill_ns == 5 * kMin);
    CHECK(r.fills[1].ts_fill_ns == 6 * kMin);
    CHECK(r.fills[0].quantity == 1);          // 0 -> +1
    CHECK(r.fills[1].quantity == 1);          // +1 -> +2 (execution-time delta)
}

// ---- F: order delta uses the execution-time position ------------------------
void test_order_delta_is_execution_time() {
    const auto reg = one_nq();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    EngineConfig cfg;
    cfg.latency_bars = 2;
    cfg.end_of_test = EndOfTestPolicy::LeaveOpen;

    const auto bars = ramp(8, kNqM6, 100.0);
    // target +1 at bar 1, target +4 at bar 2.
    const ScheduledTargetStrategy strat(rows({{1, 1}, {2, 4}}), kFp);
    const auto r = BacktestEngine(reg, resolver, risk, cfg).run(bars, strat);

    CHECK(r.orders.size() == 2);
    // If the quantity were bound at decision time it would be 1 then 4. It is the
    // execution-time delta: 0->+1 == 1, then +1->+4 == 3.
    CHECK(r.orders[0].quantity == 1);
    CHECK(r.orders[1].quantity == 3);
}

// ---- G: a pending signal crossing a roll resolves the execution-time contract
void test_pending_signal_crosses_roll() {
    const auto reg = roll_nq();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    EngineConfig cfg;
    cfg.latency_bars = 2;
    cfg.end_of_test = EndOfTestPolicy::LeaveOpen;

    // bars 1..3 on NQM6, bars 4..7 on NQU6 (the feed rolls between bar 3 and 4).
    std::vector<MarketBar> bars;
    for (int i = 1; i <= 3; ++i) bars.push_back(bar(i, kNqM6, 100.0 + i));
    for (int i = 4; i <= 7; ++i) bars.push_back(bar(i, kNqU6, 200.0 + i));

    // one decision, on an NQM6 bar (ts 2 min); it becomes due at bar 5 (ts 5min),
    // by which time the feed is on NQU6.
    const ScheduledTargetStrategy strat(rows({{2, 1}}), kFp);
    const auto r = BacktestEngine(reg, resolver, risk, cfg).run(bars, strat);

    CHECK(r.fills_generated == 1);
    CHECK(r.fills.front().ts_fill_ns == 5 * kMin);
    CHECK(r.fills.front().raw_symbol == "NQU6");   // execution-time contract, not the decision-time NQM6
    CHECK(r.fills.front().instrument_id == kNqU6);
}

// ---- H: risk rejection of one queued intent does not remove later intents ----
void test_risk_rejection_does_not_drop_later_intents() {
    const auto reg = one_nq();
    const RegistryActiveContractResolver resolver(reg);
    const RejectFirstN risk(1);              // reject only the FIRST order
    EngineConfig cfg;
    cfg.latency_bars = 2;
    cfg.end_of_test = EndOfTestPolicy::LeaveOpen;

    const auto bars = ramp(8, kNqM6, 100.0);
    // distinct targets so every queued intent produces a non-zero order delta.
    const ScheduledTargetStrategy strat(rows({{1, 1}, {2, 2}, {3, 3}}), kFp);
    const auto r = BacktestEngine(reg, resolver, risk, cfg).run(bars, strat);

    CHECK(r.signals_generated == 3);
    CHECK(r.orders.size() == 3);              // all three queued intents reached the gate
    CHECK(r.risk_rejects == 1);              // only the first was rejected
    CHECK(r.risk_decisions.size() == 3);     // later intents were NOT deleted / mutated
    // the later intents still execute in order: intent 1 fills at bar 5, intent 2
    // at bar 6 -- the rejection of intent 0 did not disturb them.
    CHECK(r.fills_generated == 2);
    CHECK(r.fills[0].ts_fill_ns == 5 * kMin);
    CHECK(r.fills[1].ts_fill_ns == 6 * kMin);
}

// ---- J: latency_bars == 0 behaviour is unchanged ----------------------------
void test_latency_zero_unchanged() {
    const auto reg = one_nq();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    EngineConfig cfg;
    cfg.latency_bars = 0;
    cfg.end_of_test = EndOfTestPolicy::LeaveOpen;

    const auto bars = ramp(6, kNqM6, 100.0);
    const ScheduledTargetStrategy strat(rows({{1, 1}, {2, -1}, {3, 0}}), kFp);
    const auto r = BacktestEngine(reg, resolver, risk, cfg).run(bars, strat);

    // decided bar i -> executes bar i+1: i=1->2, i=2->3, i=3->4
    CHECK(r.fills_generated == 3);
    CHECK(r.fills[0].ts_fill_ns == 2 * kMin);
    CHECK(r.fills[1].ts_fill_ns == 3 * kMin);
    CHECK(r.fills[2].ts_fill_ns == 4 * kMin);
    CHECK(r.fills[1].quantity == 2);         // +1 -> -1
}

}  // namespace

int main() {
    test_three_consecutive_signals_all_survive();
    test_no_decision_between_queued_signals();
    test_order_delta_is_execution_time();
    test_pending_signal_crosses_roll();
    test_risk_rejection_does_not_drop_later_intents();
    test_latency_zero_unchanged();
    return quant::test::summary("test_latency_queue");
}
