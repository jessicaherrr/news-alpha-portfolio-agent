// Phase 06 -- deterministic event-driven backtest engine.
//
//   A. deterministic event ordering / replay
//   B. no same-bar execution
//   C. a signal on bar i executes on bar i+1 (at that bar's open)
//   D. strategy cannot see a future bar (propagated through the loop)
//   E. target-position delta semantics (0->+1, +1->+1, +1->-1, -1->-2, -2->0)
//   F. execution-time active-contract selection
//   G. real-style roll: NQM6 decision bar -> NQU6 execution bar
//   H. a held old contract is closed, never silently relabelled as the new one
//   I. mandatory RiskDecision gate (RESIZE honoured)
//   J. a Fill is only ever produced through the validated raw-contract path
//   K. tick / multiplier come from the ContractSpec
//   L. deterministic ids / traceability
//   M. equal-timestamp sequence determinism
//   N. end-of-test open-position policy (force-liquidate vs leave-open)
//   O. official PnL derives only from Fill events (not bar closes)

#include "quant_core/contract.hpp"
#include "quant_core/contract_registry.hpp"
#include "quant_core/contract_selector.hpp"
#include "quant_core/domain_errors.hpp"
#include "quant_core/engine.hpp"
#include "quant_core/events.hpp"
#include "quant_core/momentum_strategy.hpp"
#include "quant_core/risk.hpp"
#include "quant_core/risk_manager.hpp"
#include "quant_core/strategy.hpp"
#include "quant_core/strategy_context.hpp"
#include "quant_core/types.hpp"

#include "test_support.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

using namespace quant;

constexpr std::int64_t  kMin      = 60'000'000'000LL;
constexpr std::uint32_t kNqh6     = 1;
constexpr std::uint32_t kNqM6Id   = 42004058;
constexpr std::uint32_t kNqU6Id   = 42004177;
constexpr std::uint32_t kEsU6Id   = 500;
constexpr std::int64_t  kBigExp   = 10'000LL * kMin;

ContractSpec spec(std::uint32_t id, std::string raw, std::string root, double mult,
                  std::int64_t act, std::int64_t exp) {
    return ContractSpec{.instrument_id = id, .raw_symbol = std::move(raw),
                        .root_symbol = std::move(root), .exchange = "XCME",
                        .tick_size = 0.25, .multiplier = mult,
                        .activation_ns = act, .expiration_ns = exp};
}

ContractRegistry single_nq_registry() {
    ContractRegistry r;
    r.add(spec(kNqh6, "NQH6", "NQ", 20.0, 1, kBigExp));
    return r;
}

// NQM6 and NQU6 both live across the whole test window (a genuine roll overlap).
ContractRegistry roll_registry() {
    ContractRegistry r;
    r.add(spec(kNqM6Id, "NQM6", "NQ", 20.0, 1, kBigExp));
    r.add(spec(kNqU6Id, "NQU6", "NQ", 20.0, 1, kBigExp));
    r.add(spec(kEsU6Id, "ESU6", "ES", 50.0, 1, kBigExp));
    return r;
}

MarketBar bar(std::int64_t idx, std::uint32_t instr, double open, double close) {
    const double hi = (open > close ? open : close) + 1.0;
    const double lo = (open < close ? open : close) - 1.0;
    return MarketBar{.ts_event_ns = idx * kMin, .instrument_id = instr,
                     .open = open, .high = hi, .low = lo, .close = close, .volume = 100};
}

// Returns a fixed target position for its root on every decision bar.
struct ConstantTarget final : Strategy {
    explicit ConstantTarget(double t) : target_(t) {}
    std::optional<Signal> decide(const StrategyContext& ctx) const override {
        Signal s;
        s.ts_decision_ns = ctx.decision_ts_ns();
        s.root_symbol    = std::string(ctx.root_symbol());
        s.target_units   = target_;
        s.rationale_code = "const";
        return s;
    }
    double target_;
};

// Emits a scripted target per decision call (clamped to the last entry).
struct Scripted final : Strategy {
    explicit Scripted(std::vector<double> t) : targets_(std::move(t)) {}
    std::optional<Signal> decide(const StrategyContext& ctx) const override {
        const double t = targets_[idx_ < targets_.size() ? idx_ : targets_.size() - 1];
        ++idx_;
        Signal s;
        s.ts_decision_ns = ctx.decision_ts_ns();
        s.root_symbol    = std::string(ctx.root_symbol());
        s.target_units   = t;
        s.rationale_code = "scripted";
        return s;
    }
    std::vector<double> targets_;
    mutable std::size_t idx_{0};
};

struct Peeking final : Strategy {
    std::optional<Signal> decide(const StrategyContext& ctx) const override {
        const auto& h = ctx.history();
        Signal s;
        s.root_symbol  = std::string(ctx.root_symbol());
        s.target_units = h.at(h.size()).close;  // one past the decision bar -> throws
        return s;
    }
};

// ---- B / C: decision time vs execution time -------------------------------

void test_no_same_bar_execution() {
    const auto reg = single_nq_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    const BacktestEngine engine(reg, resolver, risk, EngineConfig{});

    std::vector<MarketBar> bars;
    for (int i = 1; i <= 5; ++i) bars.push_back(bar(i, kNqh6, 100.0 + i, 100.0 + i));
    const ConstantTarget strat(1.0);
    const auto r = engine.run(bars, strat);

    CHECK(r.signals_generated == 5);              // one decision per bar
    CHECK(r.fills_generated >= 1);
    // The first fill is at bar 2 (index 1), at that bar's OPEN, never bar 1.
    CHECK(r.fills.front().ts_fill_ns == bars[1].ts_event_ns);
    CHECK_CLOSE(r.fills.front().fill_price, bars[1].open, 1e-9);
    for (const auto& f : r.fills) {
        CHECK(f.ts_fill_ns > bars[0].ts_event_ns);  // nothing executes on the first bar
    }
}

// ---- D: no look-ahead ----------------------------------------------------

void test_strategy_cannot_see_future_bar() {
    const auto reg = single_nq_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    const BacktestEngine engine(reg, resolver, risk, EngineConfig{});

    std::vector<MarketBar> bars;
    for (int i = 1; i <= 4; ++i) bars.push_back(bar(i, kNqh6, 100.0 + i, 100.0 + i));
    const Peeking peeker;
    CHECK_THROWS_AS(engine.run(bars, peeker), std::out_of_range);
}

// ---- E: target-position delta semantics --------------------------------

void test_target_position_delta_semantics() {
    const auto reg = single_nq_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    EngineConfig cfg;
    cfg.end_of_test = EndOfTestPolicy::LeaveOpen;
    const BacktestEngine engine(reg, resolver, risk, cfg);

    // open price of bar i == 100 + i.
    std::vector<MarketBar> bars;
    for (int i = 1; i <= 6; ++i) bars.push_back(bar(i, kNqh6, 100.0 + i, 100.0 + i));

    // decision targets:  +1, +1, -1, -2, 0, 0
    const Scripted strat({1, 1, -1, -2, 0, 0});
    const auto r = engine.run(bars, strat);

    // executions land one bar later:
    //   bar2: 0 -> +1   BUY 1
    //   bar3: +1 -> +1   (no order)
    //   bar4: +1 -> -1   SELL 2
    //   bar5: -1 -> -2   SELL 1
    //   bar6: -2 -> 0    BUY 2
    CHECK(r.orders.size() == 4);
    CHECK(r.orders[0].side == Side::Buy  && r.orders[0].quantity == 1);
    CHECK(r.orders[1].side == Side::Sell && r.orders[1].quantity == 2);
    CHECK(r.orders[2].side == Side::Sell && r.orders[2].quantity == 1);
    CHECK(r.orders[3].side == Side::Buy  && r.orders[3].quantity == 2);
    CHECK(r.fills.size() == 4);

    // realized: close 1 long (101->103): +40 ; close 2 short (avg 103.5 -> 105): -60
    CHECK_CLOSE(r.gross_realized_pnl_usd, -20.0, 1e-6);
    CHECK_CLOSE(r.costs_usd, 0.0, 1e-9);
    CHECK_CLOSE(r.net_realized_pnl_usd, -20.0, 1e-6);
    CHECK(r.closed_trades == 2);
    CHECK(r.final_positions.empty());  // ended flat exactly at target 0
}

// ---- F / G: execution-time active contract + real-style roll ------------

void test_decision_vs_execution_time_roll() {
    // 23:59 the continuous feed carries NQM6; the strategy decides target +1.
    // 00:00 the feed has rolled to NQU6. The execution MUST create an NQU6
    // order/fill -- not the stale NQM6 from decision time.
    const auto reg = roll_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    EngineConfig cfg;
    cfg.end_of_test = EndOfTestPolicy::LeaveOpen;
    const BacktestEngine engine(reg, resolver, risk, cfg);

    std::vector<MarketBar> bars{
        bar(1, kNqM6Id, 20000.0, 20000.0),  // decision bar: NQM6
        bar(2, kNqU6Id, 20310.0, 20310.0),  // execution bar: NQU6 (feed rolled)
        bar(3, kNqU6Id, 20320.0, 20320.0),
    };
    const ConstantTarget strat(1.0);
    const auto r = engine.run(bars, strat);

    CHECK(r.fills.size() == 1);
    CHECK(r.fills[0].instrument_id == kNqU6Id);
    CHECK(r.fills[0].raw_symbol == "NQU6");
    CHECK_CLOSE(r.fills[0].fill_price, 20310.0, 1e-9);
    CHECK(r.orders[0].instrument_id == kNqU6Id);
    CHECK(r.rolls == 0);  // we were flat -- nothing to roll, just correct routing
    for (const auto& f : r.fills) CHECK(f.instrument_id != kNqM6Id);
    CHECK(r.final_positions.size() == 1);
    CHECK(r.final_positions[0].instrument_id == kNqU6Id);
    CHECK(r.final_positions[0].units == 1);
}

// ---- H: a position held across a roll is closed, not relabelled ---------

void test_held_contract_not_relabelled_across_roll() {
    const auto reg = roll_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    EngineConfig cfg;
    cfg.end_of_test = EndOfTestPolicy::LeaveOpen;
    // Bar 3 carries no CONTEMPORANEOUS NQM6 bar, so this test opts IN to the
    // StaleObservedClose approximation (the Phase 07.1 default is RejectDefer).
    cfg.roll.fallback = RollPriceFallback::StaleObservedClose;
    const BacktestEngine engine(reg, resolver, risk, cfg);

    std::vector<MarketBar> bars{
        bar(1, kNqM6Id, 20000.0, 20001.0),  // decide +1
        bar(2, kNqM6Id, 20002.0, 20003.0),  // execute: BUY 1 NQM6
        bar(3, kNqU6Id, 20310.0, 20311.0),  // roll: close NQM6, open NQU6
        bar(4, kNqU6Id, 20312.0, 20313.0),
    };
    const ConstantTarget strat(1.0);
    const auto r = engine.run(bars, strat);

    CHECK(r.rolls == 1);

    // The close-leg takes the opt-in fallback: last observed price, but the fill
    // is stamped at the EXECUTION timestamp -- never retroactive -- and its stale
    // source is auditable.
    CHECK(r.rolls_priced_stale == 1);
    CHECK(r.rolls_priced_contemporaneous == 0);
    CHECK(r.roll_fallback_audit.size() == 1);
    CHECK(r.roll_fallback_audit[0].instrument_id == kNqM6Id);
    CHECK(r.roll_fallback_audit[0].execution_ts_ns == bars[2].ts_event_ns);
    CHECK(r.roll_fallback_audit[0].reference_price_ts_ns == bars[1].ts_event_ns);
    CHECK(r.roll_fallback_audit[0].reference_age_ns == bars[2].ts_event_ns - bars[1].ts_event_ns);
    CHECK(r.roll_fallback_audit[0].execution_ts_ns != r.roll_fallback_audit[0].reference_price_ts_ns);

    // The old contract was really traded on both sides and ends flat.
    bool bought_nqm6 = false, sold_nqm6 = false, traded_nqu6 = false;
    for (const auto& f : r.fills) {
        if (f.instrument_id == kNqM6Id && f.side == Side::Buy)  bought_nqm6 = true;
        if (f.instrument_id == kNqM6Id && f.side == Side::Sell) sold_nqm6 = true;
        if (f.instrument_id == kNqU6Id) traded_nqu6 = true;
    }
    CHECK(bought_nqm6);
    CHECK(sold_nqm6);
    CHECK(traded_nqu6);

    // A roll-reason closed trade exists on the OLD contract.
    bool roll_trade = false;
    for (const auto& t : r.trades) {
        if (t.close_reason.rfind("roll", 0) == 0 && t.instrument_id == kNqM6Id) roll_trade = true;
    }
    CHECK(roll_trade);

    // The roll close uses the OLD contract's last observed price, but the fill
    // timestamp is the EXECUTION bar (bar 3), not the earlier bar 2.
    for (const auto& f : r.fills) {
        if (f.instrument_id == kNqM6Id && f.side == Side::Sell) {
            CHECK_CLOSE(f.fill_price, 20003.0, 1e-9);          // bar 2 close of NQM6 (last observed)
            CHECK(f.ts_fill_ns == bars[2].ts_event_ns);        // stamped at execution time
            CHECK(f.ts_fill_ns > bars[1].ts_event_ns);         // NOT retroactive
        }
    }

    // Final state: only NQU6 is held; NQM6 is gone (units 0, not carried).
    CHECK(r.final_positions.size() == 1);
    CHECK(r.final_positions[0].instrument_id == kNqU6Id);
    CHECK(r.final_positions[0].units == 1);
}

// ---- I: mandatory RiskDecision gate -----------------------------------

void test_risk_decision_gate_resizes() {
    const auto reg = single_nq_registry();
    const RegistryActiveContractResolver resolver(reg);
    const MaxContractsRiskManager risk(RiskLimits{});  // cap 5
    EngineConfig cfg;
    cfg.end_of_test = EndOfTestPolicy::LeaveOpen;
    const BacktestEngine engine(reg, resolver, risk, cfg);

    // Two bars: one decision, one execution -- so exactly one order is issued and
    // the RESIZE cap (per order) is observed without later top-up orders.
    std::vector<MarketBar> bars{bar(1, kNqh6, 100.0, 100.0), bar(2, kNqh6, 100.0, 100.0)};
    const ConstantTarget strat(9.0);  // wants +9; risk caps the ORDER to 5
    const auto r = engine.run(bars, strat);

    CHECK(r.risk_decisions.size() == r.orders.size());  // every order was reviewed
    CHECK(r.orders.size() == 1);
    CHECK(r.orders[0].quantity == 9);                   // requested delta
    CHECK(r.risk_decisions[0].verdict == RiskVerdict::Resize);
    CHECK(r.risk_decisions[0].approved_quantity == 5);
    CHECK(r.fills.size() == 1);
    CHECK(r.fills[0].quantity == 5);                    // gated down before the fill
    CHECK(r.final_positions.size() == 1);
    CHECK(r.final_positions[0].units == 5);
}

// ---- J / K: fills only through the validated path; spec tick/multiplier -

void test_fills_are_validated_raw_contract() {
    ContractRegistry reg;
    reg.add(spec(77, "CLZ6", "CL", 1000.0, 1, kBigExp));  // deliberately odd multiplier
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    const BacktestEngine engine(reg, resolver, risk, EngineConfig{});

    std::vector<MarketBar> bars;
    for (int i = 1; i <= 5; ++i) bars.push_back(bar(i, 77, 70.0 + i, 70.0 + i));
    const Scripted strat({1, 1, 0, 0, 0});
    const auto r = engine.run(bars, strat);

    CHECK(!r.fills.empty());
    for (const auto& f : r.fills) {
        CHECK(f.price_domain == PriceDomain::RawContract);
        CHECK(f.instrument_id == 77u);
        CHECK(f.raw_symbol == "CLZ6");
        CHECK_CLOSE(f.tick_size, 0.25, 1e-12);
        CHECK_CLOSE(f.multiplier, 1000.0, 1e-9);   // from ContractSpec, not a config
        // every fill traces back to a recorded order
        bool has_order = false;
        for (const auto& o : r.orders) if (o.order_id == f.order_id) has_order = true;
        CHECK(has_order);
    }
    CHECK(r.fills_generated <= r.orders_generated);
}

// ---- L / A / M: deterministic ids and replay --------------------------

void test_deterministic_ids_and_replay() {
    const auto reg = roll_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    const BacktestEngine engine(reg, resolver, risk, EngineConfig{});

    // Two roots, some SHARED timestamps -> exercise (ts_event_ns, seq) ordering.
    std::vector<MarketBar> bars{
        bar(1, kNqM6Id, 20000.0, 20000.0),
        bar(1, kEsU6Id,  5000.0,  5000.0),   // same ts as the NQ bar
        bar(2, kNqM6Id, 20010.0, 20010.0),
        bar(2, kEsU6Id,  5010.0,  5010.0),
        bar(3, kNqM6Id, 20020.0, 20020.0),
        bar(3, kEsU6Id,  5020.0,  5020.0),
    };
    const ConstantTarget strat(1.0);

    const auto a = engine.run(bars, strat);
    const auto b = engine.run(bars, strat);

    CHECK(a.orders.size() == b.orders.size());
    CHECK(a.fills.size() == b.fills.size());
    for (std::size_t i = 0; i < a.orders.size(); ++i) {
        CHECK(a.orders[i].order_id == b.orders[i].order_id);
        CHECK(a.orders[i].instrument_id == b.orders[i].instrument_id);
        CHECK(a.orders[i].signal_id == b.orders[i].signal_id);
    }
    for (std::size_t i = 0; i < a.fills.size(); ++i) {
        CHECK(a.fills[i].fill_id == b.fills[i].fill_id);
        CHECK_CLOSE(a.fills[i].fill_price, b.fills[i].fill_price, 1e-12);
    }
    CHECK_CLOSE(a.net_realized_pnl_usd, b.net_realized_pnl_usd, 1e-9);
    CHECK_CLOSE(a.max_drawdown_usd, b.max_drawdown_usd, 1e-9);

    // ids are monotonic from 1, no gaps in the order / fill streams.
    for (std::size_t i = 0; i < a.orders.size(); ++i) CHECK(a.orders[i].order_id == i + 1);
    for (std::size_t i = 0; i < a.fills.size(); ++i)  CHECK(a.fills[i].fill_id == i + 1);
    for (const auto& o : a.orders) CHECK(o.signal_id != 0);

    // Reordering simultaneous cross-root bars in the input does not change the
    // per-root result (behaviour is not input-container-order dependent).
    std::vector<MarketBar> swapped{bars[1], bars[0], bars[3], bars[2], bars[5], bars[4]};
    const auto c = engine.run(swapped, strat);
    CHECK_CLOSE(c.net_realized_pnl_usd, a.net_realized_pnl_usd, 1e-9);
    CHECK(c.fills.size() == a.fills.size());
}

// ---- deterministic multi-instrument ingestion (Phase 06 review fix) ------

constexpr std::uint32_t kEsZ6 = 700;
constexpr std::uint32_t kNqZ6 = 900;
constexpr std::uint32_t kClZ6 = 800;

ContractRegistry three_root_registry() {
    ContractRegistry r;
    r.add(spec(kEsZ6, "ESZ6", "ES", 50.0, 1, kBigExp));
    r.add(spec(kClZ6, "CLZ6", "CL", 1000.0, 1, kBigExp));
    r.add(spec(kNqZ6, "NQZ6", "NQ", 20.0, 1, kBigExp));
    return r;
}

// Per-root deterministic: alternates target by this root's own bar count.
struct AlternatingByHistory final : Strategy {
    std::optional<Signal> decide(const StrategyContext& ctx) const override {
        Signal s;
        s.ts_decision_ns = ctx.decision_ts_ns();
        s.root_symbol    = std::string(ctx.root_symbol());
        s.target_units   = (ctx.history().size() % 2 == 1) ? 1.0 : -1.0;
        s.rationale_code = "alt";
        return s;
    }
};

void check_results_equal(const BacktestResult& a, const BacktestResult& b) {
    CHECK(a.events_processed == b.events_processed);
    CHECK(a.bars_processed == b.bars_processed);
    CHECK(a.signals_generated == b.signals_generated);
    CHECK(a.orders_generated == b.orders_generated);
    CHECK(a.fills_generated == b.fills_generated);
    CHECK(a.closed_trades == b.closed_trades);
    CHECK(a.rolls == b.rolls);
    CHECK(a.rolls_priced_contemporaneous == b.rolls_priced_contemporaneous);
    CHECK(a.rolls_priced_stale == b.rolls_priced_stale);
    CHECK(a.rolls_deferred == b.rolls_deferred);
    CHECK(a.non_fills == b.non_fills);
    CHECK(a.roll_fallback_audit.size() == b.roll_fallback_audit.size());
    CHECK(a.eot_liquidations.size() == b.eot_liquidations.size());
    CHECK(a.final_position_marks.size() == b.final_position_marks.size());
    CHECK(a.eot_positions_left_open_stale == b.eot_positions_left_open_stale);
    CHECK_CLOSE(a.gross_realized_pnl_usd, b.gross_realized_pnl_usd, 1e-9);
    CHECK_CLOSE(a.costs_usd, b.costs_usd, 1e-9);
    CHECK_CLOSE(a.net_realized_pnl_usd, b.net_realized_pnl_usd, 1e-9);
    CHECK_CLOSE(a.max_drawdown_usd, b.max_drawdown_usd, 1e-9);
    CHECK_CLOSE(a.unrealized_pnl_usd_at_end, b.unrealized_pnl_usd_at_end, 1e-9);

    CHECK(a.orders.size() == b.orders.size());
    for (std::size_t i = 0; i < a.orders.size() && i < b.orders.size(); ++i) {
        const auto& x = a.orders[i];
        const auto& y = b.orders[i];
        CHECK(x.order_id == y.order_id);
        CHECK(x.signal_id == y.signal_id);
        CHECK(x.ts_created_ns == y.ts_created_ns);
        CHECK(x.instrument_id == y.instrument_id);
        CHECK(x.raw_symbol == y.raw_symbol);
        CHECK(x.side == y.side);
        CHECK(x.quantity == y.quantity);
    }
    CHECK(a.risk_decisions.size() == b.risk_decisions.size());
    for (std::size_t i = 0; i < a.risk_decisions.size() && i < b.risk_decisions.size(); ++i) {
        CHECK(a.risk_decisions[i].order_id == b.risk_decisions[i].order_id);
        CHECK(a.risk_decisions[i].verdict == b.risk_decisions[i].verdict);
        CHECK(a.risk_decisions[i].approved_quantity == b.risk_decisions[i].approved_quantity);
    }
    CHECK(a.fills.size() == b.fills.size());
    for (std::size_t i = 0; i < a.fills.size() && i < b.fills.size(); ++i) {
        const auto& x = a.fills[i];
        const auto& y = b.fills[i];
        CHECK(x.fill_id == y.fill_id);
        CHECK(x.order_id == y.order_id);
        CHECK(x.ts_fill_ns == y.ts_fill_ns);
        CHECK(x.instrument_id == y.instrument_id);
        CHECK(x.raw_symbol == y.raw_symbol);
        CHECK(x.side == y.side);
        CHECK(x.quantity == y.quantity);
        CHECK_CLOSE(x.fill_price, y.fill_price, 1e-9);
        CHECK_CLOSE(x.multiplier, y.multiplier, 1e-9);
        CHECK_CLOSE(x.commission_usd, y.commission_usd, 1e-9);
    }
    CHECK(a.trades.size() == b.trades.size());
    for (std::size_t i = 0; i < a.trades.size() && i < b.trades.size(); ++i) {
        const auto& x = a.trades[i];
        const auto& y = b.trades[i];
        CHECK(x.instrument_id == y.instrument_id);
        CHECK(x.ts_open_ns == y.ts_open_ns);
        CHECK(x.ts_close_ns == y.ts_close_ns);
        CHECK(x.quantity == y.quantity);
        CHECK(x.direction == y.direction);
        CHECK_CLOSE(x.gross_pnl_usd, y.gross_pnl_usd, 1e-9);
        CHECK_CLOSE(x.net_pnl_usd, y.net_pnl_usd, 1e-9);
        CHECK(x.close_reason == y.close_reason);
    }
    CHECK(a.final_positions.size() == b.final_positions.size());
    for (std::size_t i = 0; i < a.final_positions.size() && i < b.final_positions.size(); ++i) {
        CHECK(a.final_positions[i].instrument_id == b.final_positions[i].instrument_id);
        CHECK(a.final_positions[i].units == b.final_positions[i].units);
        CHECK_CLOSE(a.final_positions[i].avg_entry_price, b.final_positions[i].avg_entry_price, 1e-9);
    }
    CHECK(a.contracts_traded == b.contracts_traded);
}

void test_multi_root_same_timestamp_permutation_determinism() {
    const auto reg = three_root_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    EngineConfig cfg;
    cfg.execution.commission_per_contract_usd = 1.0;
    cfg.end_of_test = EndOfTestPolicy::LeaveOpen;
    const BacktestEngine engine(reg, resolver, risk, cfg);
    const AlternatingByHistory strat;

    // ES / NQ / CL, one bar per root at each of 6 SHARED timestamps. The base
    // vector is deliberately NOT in canonical (ts, instrument_id) order.
    std::vector<MarketBar> base;
    for (int i = 1; i <= 6; ++i) {
        base.push_back(bar(i, kNqZ6, 20000.0 + i, 20000.0 - i));
        base.push_back(bar(i, kClZ6, 70.0 + i, 70.0 + 2 * i));
        base.push_back(bar(i, kEsZ6, 5000.0 + i, 5000.0 + 3 * i));
    }

    const auto ord0 = order_bar_events(base, reg);
    const auto r0   = engine.run(base, strat);

    // canonical order really is (ts_event_ns, instrument_id) ascending.
    for (std::size_t i = 1; i < ord0.size(); ++i) {
        const bool ordered = (ord0[i - 1].ts_event_ns < ord0[i].ts_event_ns) ||
                             (ord0[i - 1].ts_event_ns == ord0[i].ts_event_ns &&
                              ord0[i - 1].instrument_id < ord0[i].instrument_id);
        CHECK(ordered);
        CHECK(ord0[i].seq == ord0[i - 1].seq + 1);  // dense monotonic seq
    }
    CHECK(ord0.front().seq == 0);

    // Several deterministic permutations of the SAME bar set.
    std::vector<std::vector<MarketBar>> perms;
    {
        auto rev = base; std::reverse(rev.begin(), rev.end()); perms.push_back(rev);
    }
    {
        auto rot = base; std::rotate(rot.begin(), rot.begin() + 7, rot.end()); perms.push_back(rot);
    }
    {  // group by root: all NQ, then all CL, then all ES
        std::vector<MarketBar> grp;
        for (const auto& b : base) if (b.instrument_id == kNqZ6) grp.push_back(b);
        for (const auto& b : base) if (b.instrument_id == kClZ6) grp.push_back(b);
        for (const auto& b : base) if (b.instrument_id == kEsZ6) grp.push_back(b);
        perms.push_back(grp);
    }
    {  // fixed pseudo-shuffle
        auto sh = base;
        for (std::size_t i = 0; i < sh.size(); ++i) {
            const std::size_t j = (i * 7 + 3) % sh.size();
            std::swap(sh[i], sh[j]);
        }
        perms.push_back(sh);
    }

    for (const auto& perm : perms) {
        CHECK(perm.size() == base.size());
        const auto ord = order_bar_events(perm, reg);
        CHECK(ord.size() == ord0.size());
        for (std::size_t i = 0; i < ord.size() && i < ord0.size(); ++i) {
            CHECK(ord[i].ts_event_ns == ord0[i].ts_event_ns);
            CHECK(ord[i].instrument_id == ord0[i].instrument_id);
            CHECK(ord[i].seq == ord0[i].seq);
        }
        check_results_equal(engine.run(perm, strat), r0);
    }

    // A duplicate (instrument_id, ts_event_ns) is rejected, not resolved by order.
    auto dup = base;
    dup.push_back(bar(3, kNqZ6, 21000.0, 21000.0));  // NQ already has a bar at ts 3
    CHECK_THROWS_AS(order_bar_events(dup, reg), InvalidMarketEvent);
    CHECK_THROWS_AS(engine.run(dup, strat), InvalidMarketEvent);

    // An unresolved instrument / non-positive ts is rejected up front.
    auto bad_instr = base; bad_instr.push_back(bar(7, 4242, 1.0, 1.0));
    CHECK_THROWS_AS(engine.run(bad_instr, strat), InvalidMarketEvent);
    auto bad_ts = base; bad_ts.push_back(MarketBar{0, kEsZ6, 1.0, 1.0, 1.0, 1.0, 1});
    CHECK_THROWS_AS(engine.run(bad_ts, strat), InvalidMarketEvent);
}

// ---- N: end-of-test policy ------------------------------------------

void test_end_of_test_policy() {
    const auto reg = single_nq_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;

    std::vector<MarketBar> bars{
        bar(1, kNqh6, 20000.0, 20000.0),
        bar(2, kNqh6, 20000.0, 20000.0),  // BUY 1 @ 20000
        bar(3, kNqh6, 20050.0, 20100.0),  // still long; close 20100
    };
    const ConstantTarget strat(1.0);

    {  // force-liquidate: flat at the end, realized includes the liquidation
        EngineConfig cfg;
        cfg.end_of_test = EndOfTestPolicy::ForceLiquidateFinalClose;
        const auto r = BacktestEngine(reg, resolver, risk, cfg).run(bars, strat);
        CHECK(r.final_positions.empty());
        CHECK_CLOSE(r.unrealized_pnl_usd_at_end, 0.0, 1e-9);
        bool eot = false;
        for (const auto& t : r.trades) if (t.close_reason == "eot") eot = true;
        CHECK(eot);
        // liquidated at bar 3 close (20100): (20100 - 20000) * 20 = 2000
        CHECK_CLOSE(r.net_realized_pnl_usd, 2000.0, 1e-6);
    }
    {  // leave-open: position survives, unrealized reported separately
        EngineConfig cfg;
        cfg.end_of_test = EndOfTestPolicy::LeaveOpen;
        const auto r = BacktestEngine(reg, resolver, risk, cfg).run(bars, strat);
        CHECK(r.final_positions.size() == 1);
        CHECK(r.final_positions[0].units == 1);
        CHECK_CLOSE(r.net_realized_pnl_usd, 0.0, 1e-9);
        CHECK_CLOSE(r.unrealized_pnl_usd_at_end, 2000.0, 1e-6);  // marked at 20100 close
        for (const auto& t : r.trades) CHECK(t.close_reason != "eot");
    }
}

// ---- O: official PnL comes only from Fill events -----------------------

void test_official_pnl_from_fills_only() {
    const auto reg = single_nq_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    EngineConfig cfg;
    cfg.end_of_test = EndOfTestPolicy::LeaveOpen;
    const BacktestEngine engine(reg, resolver, risk, cfg);

    // Bar closes are deliberately offset from opens by a NON-constant amount, so
    // a close-based PnL would give a different number than the fill-based one.
    std::vector<MarketBar> bars{
        bar(1, kNqh6, 20000.0, 20005.0),
        bar(2, kNqh6, 20000.0, 20005.0),   // BUY 1 @ open 20000
        bar(3, kNqh6, 20100.0, 20140.0),   // SELL 1 @ open 20100
        bar(4, kNqh6, 20100.0, 20140.0),
    };
    const Scripted strat({1, 1, 0, 0});
    const auto r = engine.run(bars, strat);

    CHECK(r.fills.size() == 2);
    // fill-based: (20100 - 20000) * 20 * 1 = 2000
    const double recomputed =
        (r.fills[1].fill_price - r.fills[0].fill_price) * r.fills[0].multiplier * 1.0;
    CHECK_CLOSE(recomputed, 2000.0, 1e-6);
    CHECK_CLOSE(r.gross_realized_pnl_usd, 2000.0, 1e-6);
    CHECK_CLOSE(r.net_realized_pnl_usd, 2000.0, 1e-6);
    // NOT the close-based number: (20140 - 20005) * 20 = 2700
    CHECK(r.gross_realized_pnl_usd < 2699.0 || r.gross_realized_pnl_usd > 2701.0);
    CHECK(r.trades.size() == 1);
    CHECK(r.trades[0].close_reason == "signal");
}

// ---- momentum still runs end to end ---------------------------------

void test_momentum_end_to_end() {
    const auto reg = single_nq_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    EngineConfig cfg;
    cfg.execution.commission_per_contract_usd = 2.0;
    const BacktestEngine engine(reg, resolver, risk, cfg);

    std::vector<MarketBar> bars;
    double p = 100.0;
    for (int i = 1; i <= 260; ++i) {
        const double open = p;
        p += (i < 150 ? 0.20 : -0.18);
        bars.push_back(bar(i, kNqh6, open, p));
    }
    const TimeSeriesMomentum strat(20, 0.01);
    const auto r = engine.run(bars, strat);
    CHECK(r.closed_trades > 0);
    CHECK(r.costs_usd > 0.0);
    CHECK(r.final_positions.empty());  // force-liquidate default
    CHECK(r.contracts_traded.size() == 1);
    CHECK(r.contracts_traded[0] == "NQH6");
}

// ---- Phase 08.2: negative CL price end-to-end -------------------------
//
// MarketBar -> MarketEvent -> Strategy -> Signal -> Order -> RiskDecision ->
// ExecutionSimulator -> Fill -> PositionLedger -> PortfolioAccountant ->
// BacktestResult, with prices that move through zero into negatives. No layer
// may reject a bar solely because a price is <= 0. Realized PnL is Fill-derived
// and carries NO absolute value.

void test_negative_cl_price_end_to_end() {
    ContractRegistry reg;
    reg.add(ContractSpec{.instrument_id = 55, .raw_symbol = "CLK0", .root_symbol = "CL",
                         .exchange = "XNYM", .tick_size = 0.01, .multiplier = 1000.0,
                         .activation_ns = 1, .expiration_ns = kBigExp});
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    EngineConfig cfg;
    cfg.end_of_test = EndOfTestPolicy::LeaveOpen;
    const BacktestEngine engine(reg, resolver, risk, cfg);

    // The prompt's price path: +5, +1, 0, -5, -20, -10 (bar opens).
    auto mkbar = [](std::int64_t i, double open) { return bar(i, 55, open, open); };

    {  // long from -20, exit at -10  ->  PnL = (+10) * 1000 = +10,000
        std::vector<MarketBar> bars{
            mkbar(1, -20.0), mkbar(2, -20.0),   // decide +1 @ b1 -> BUY 1 @ b2 open -20
            mkbar(3, -18.0), mkbar(4, -15.0),   // decide 0 @ b3 -> SELL 1 @ b4 open -15
            mkbar(5, -10.0), mkbar(6, -10.0),
        };
        const Scripted strat({1, 1, 0, 0, 0, 0});
        const auto r = engine.run(bars, strat);

        CHECK(r.bars_processed == 6);            // no bar rejected for price <= 0
        CHECK(r.fills.size() == 2);
        CHECK_CLOSE(r.fills[0].fill_price, -20.0, 1e-9);
        CHECK_CLOSE(r.fills[1].fill_price, -15.0, 1e-9);
        CHECK(r.fills[0].side == Side::Buy);
        CHECK(r.fills[1].side == Side::Sell);
        // (exit - entry) * multiplier * signed qty -- no abs()
        const double recomputed =
            (r.fills[1].fill_price - r.fills[0].fill_price) * r.fills[0].multiplier * 1.0;
        CHECK_CLOSE(recomputed, 5000.0, 1e-6);   // (-15 - -20) * 1000
        CHECK_CLOSE(r.gross_realized_pnl_usd, 5000.0, 1e-6);
        CHECK_CLOSE(r.net_realized_pnl_usd, 5000.0, 1e-6);
        CHECK(r.contracts_traded.size() == 1 && r.contracts_traded[0] == "CLK0");
        // portfolio view: long CL, gross exposure positive, direction not inverted
        const auto& pf = r.portfolio_at_end;
        CHECK(pf.positions.empty());             // flat after the exit
        CHECK_CLOSE(pf.net_realized_pnl_usd, 5000.0, 1e-6);

        // deterministic replay
        const auto r2 = engine.run(bars, Scripted({1, 1, 0, 0, 0, 0}));
        CHECK_CLOSE(r2.gross_realized_pnl_usd, r.gross_realized_pnl_usd, 1e-9);
        CHECK(r2.fills.size() == r.fills.size());
    }
    {  // long from +5, exit at -5  ->  PnL = (-10) * 1000 = -10,000
        std::vector<MarketBar> bars{
            mkbar(1, 5.0), mkbar(2, 5.0),
            mkbar(3, 0.0), mkbar(4, -5.0),
            mkbar(5, -5.0), mkbar(6, -5.0),
        };
        const Scripted strat({1, 1, 0, 0, 0, 0});
        const auto r = engine.run(bars, strat);
        CHECK(r.fills.size() == 2);
        CHECK_CLOSE(r.fills[0].fill_price, 5.0, 1e-9);
        CHECK_CLOSE(r.fills[1].fill_price, -5.0, 1e-9);
        CHECK_CLOSE(r.gross_realized_pnl_usd, -10000.0, 1e-6);   // (-5 - 5) * 1000
    }
    {  // a bar that opens EXACTLY at zero still executes
        std::vector<MarketBar> bars{
            mkbar(1, 2.0), mkbar(2, 0.0),   // BUY 1 @ 0.0
            mkbar(3, -1.0), mkbar(4, -1.0),
        };
        const Scripted strat({1, 1, 1, 1});
        const auto r = engine.run(bars, strat);
        CHECK(r.fills.size() == 1);
        CHECK_CLOSE(r.fills[0].fill_price, 0.0, 1e-9);
        CHECK(r.final_positions.size() == 1);
        CHECK(r.final_positions[0].units == 1);
    }
}

// ---- Phase 13.5C: auxiliary roll-close marks ----------------------------
//
//   P. an aux mark at EXACTLY the execution instant prices a would-be-deferred
//      held roll on the same-timestamp basis; rolls_priced_auxiliary_marks is a
//      subset of rolls_priced_contemporaneous.
//   Q. an EMPTY close_marks map is byte-identical to the frozen path; a mark for
//      a NON-matching (id, ts) is never consulted (no effect).
//   R. the aux mark changes ONLY roll pricing -- bars_seen / events / signals /
//      decision timing / target application are identical.
//   S. an aux mark for a contract already non-tradable at T still fails loudly.
//   T. a malformed close_marks entry (unknown instrument, non-positive price) is
//      rejected at run start.

// a held NQM6 -> NQU6 roll where the feed has NO contemporaneous NQM6 bar: the
// frozen default (RejectDefer) defers on every subsequent bar.
std::vector<MarketBar> held_roll_no_overlap_bars() {
    return {
        bar(1, kNqM6Id, 20000.0, 20001.0),   // decide +1
        bar(2, kNqM6Id, 20002.0, 20003.0),   // execute BUY 1 NQM6
        bar(3, kNqU6Id, 20310.0, 20311.0),   // feed rolls -> NQU6 (no NQM6 bar)
        bar(4, kNqU6Id, 20312.0, 20313.0),
        bar(5, kNqU6Id, 20314.0, 20315.0),
    };
}

void test_P_aux_mark_prices_a_deferred_roll() {
    const auto reg = roll_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    const auto bars = held_roll_no_overlap_bars();
    const ConstantTarget strat(1.0);

    // baseline: no marks -> the roll defers and no close-leg fill is produced
    EngineConfig base_cfg;
    base_cfg.end_of_test = EndOfTestPolicy::LeaveOpen;
    const auto base = BacktestEngine(reg, resolver, risk, base_cfg).run(bars, strat);
    CHECK(base.rolls_deferred >= 1);
    CHECK(base.rolls == 0);
    CHECK(base.rolls_priced_contemporaneous == 0);
    CHECK(base.rolls_priced_auxiliary_marks == 0);

    // with an auxiliary NQM6 close at EXACTLY bar 3's ts (the execution instant)
    EngineConfig aux_cfg;
    aux_cfg.end_of_test = EndOfTestPolicy::LeaveOpen;
    aux_cfg.roll.close_marks[{kNqM6Id, bars[2].ts_event_ns}] = 20050.0;
    const auto r = BacktestEngine(reg, resolver, risk, aux_cfg).run(bars, strat);

    CHECK(r.rolls == 1);
    CHECK(r.rolls_priced_contemporaneous == 1);
    CHECK(r.rolls_priced_auxiliary_marks == 1);
    CHECK(r.rolls_deferred == 0);
    CHECK(r.rolls_priced_stale == 0);
    CHECK(r.roll_fallback_audit.empty());
    // invariant: 0 <= aux <= contemporaneous
    CHECK(r.rolls_priced_auxiliary_marks <= r.rolls_priced_contemporaneous);

    bool closed_nqm6_at_aux = false;
    for (const auto& f : r.fills) {
        if (f.instrument_id == kNqM6Id && f.side == Side::Sell) {
            CHECK_CLOSE(f.fill_price, 20050.0, 1e-9);          // the aux close
            CHECK(f.ts_fill_ns == bars[2].ts_event_ns);        // stamped at T, never retroactive
            closed_nqm6_at_aux = true;
        }
    }
    CHECK(closed_nqm6_at_aux);
    // final state: only NQU6 held
    CHECK(r.final_positions.size() == 1);
    CHECK(r.final_positions[0].instrument_id == kNqU6Id);
}

void test_Q_empty_or_nonmatching_marks_are_a_noop() {
    const auto reg = roll_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    const auto bars = held_roll_no_overlap_bars();
    const ConstantTarget strat(1.0);

    EngineConfig empty_cfg;
    empty_cfg.end_of_test = EndOfTestPolicy::LeaveOpen;
    const auto a = BacktestEngine(reg, resolver, risk, empty_cfg).run(bars, strat);

    // a mark for the RIGHT contract but the WRONG ts, and the wrong contract at
    // the right ts -- neither key can ever match the roll close-leg
    EngineConfig nonmatch_cfg;
    nonmatch_cfg.end_of_test = EndOfTestPolicy::LeaveOpen;
    nonmatch_cfg.roll.close_marks[{kNqM6Id, bars[2].ts_event_ns + 1}] = 20050.0;
    nonmatch_cfg.roll.close_marks[{kNqU6Id, bars[2].ts_event_ns}]     = 20360.0;
    const auto b = BacktestEngine(reg, resolver, risk, nonmatch_cfg).run(bars, strat);

    CHECK(a.rolls == b.rolls);
    CHECK(a.rolls_deferred == b.rolls_deferred);
    CHECK(a.rolls_priced_contemporaneous == b.rolls_priced_contemporaneous);
    CHECK(a.rolls_priced_auxiliary_marks == 0 && b.rolls_priced_auxiliary_marks == 0);
    CHECK_CLOSE(a.net_realized_pnl_usd, b.net_realized_pnl_usd, 1e-9);
    CHECK(a.fills_generated == b.fills_generated);
    CHECK(a.events_processed == b.events_processed);
    CHECK(a.signals_generated == b.signals_generated);
}

void test_R_aux_mark_changes_only_roll_pricing() {
    const auto reg = roll_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    const auto bars = held_roll_no_overlap_bars();
    const ConstantTarget strat(1.0);

    EngineConfig base_cfg;  base_cfg.end_of_test = EndOfTestPolicy::LeaveOpen;
    EngineConfig aux_cfg;   aux_cfg.end_of_test  = EndOfTestPolicy::LeaveOpen;
    aux_cfg.roll.close_marks[{kNqM6Id, bars[2].ts_event_ns}] = 20050.0;

    const auto base = BacktestEngine(reg, resolver, risk, base_cfg).run(bars, strat);
    const auto aux  = BacktestEngine(reg, resolver, risk, aux_cfg).run(bars, strat);

    // the primary event stream / strategy loop is untouched
    CHECK(base.events_processed == aux.events_processed);
    CHECK(base.bars_processed == aux.bars_processed);
    CHECK(base.signals_generated == aux.signals_generated);
    CHECK(base.daily_equity.size() == aux.daily_equity.size());
    // only the previously-deferred roll now completes
    CHECK(base.rolls == 0 && aux.rolls == 1);
    CHECK(base.rolls_deferred >= 1 && aux.rolls_deferred == 0);
}

void test_S_aux_mark_for_dead_contract_still_fails_loud() {
    ContractRegistry reg;
    reg.add(spec(kNqM6Id, "NQM6", "NQ", 20.0, 1, 3 * kMin));   // NQM6 expires at ts 3*kMin
    reg.add(spec(kNqU6Id, "NQU6", "NQ", 20.0, 1, kBigExp));
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;

    std::vector<MarketBar> bars{
        bar(1, kNqM6Id, 20000.0, 20001.0),   // decide +1
        bar(2, kNqM6Id, 20002.0, 20003.0),   // execute BUY 1 NQM6
        bar(4, kNqU6Id, 20312.0, 20313.0),   // feed rolls AFTER NQM6 is already dead
        bar(5, kNqU6Id, 20314.0, 20315.0),
    };
    const ConstantTarget strat(1.0);
    EngineConfig cfg;
    // an aux mark stamped at the (post-expiry) execution instant must NOT rescue
    // a retroactive close -- the existing fail-loud guard still wins.
    cfg.roll.close_marks[{kNqM6Id, bars[2].ts_event_ns}] = 20050.0;
    CHECK_THROWS(BacktestEngine(reg, resolver, risk, cfg).run(bars, strat));
}

void test_T_malformed_close_marks_rejected_at_run_start() {
    const auto reg = roll_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    std::vector<MarketBar> bars{bar(1, kNqM6Id, 20000.0, 20001.0), bar(2, kNqM6Id, 20002.0, 20003.0)};
    const ConstantTarget strat(1.0);

    EngineConfig bad_id;
    bad_id.roll.close_marks[{999999u, 2 * kMin}] = 20050.0;       // unknown instrument
    CHECK_THROWS(BacktestEngine(reg, resolver, risk, bad_id).run(bars, strat));

    EngineConfig bad_px;
    bad_px.roll.close_marks[{kNqM6Id, 2 * kMin}] =
        std::numeric_limits<double>::quiet_NaN();                 // non-finite price
    CHECK_THROWS(BacktestEngine(reg, resolver, risk, bad_px).run(bars, strat));

    // a NEGATIVE finite price is legal (CL traded below zero on 2020-04-20)
    EngineConfig neg_px;
    neg_px.end_of_test = EndOfTestPolicy::LeaveOpen;
    neg_px.roll.close_marks[{kNqM6Id, held_roll_no_overlap_bars()[2].ts_event_ns}] = -12.5;
    const auto ok = BacktestEngine(reg, resolver, risk, neg_px).run(
        held_roll_no_overlap_bars(), ConstantTarget(1.0));
    CHECK(ok.rolls_priced_auxiliary_marks == 1);
}

}  // namespace

int main() {
    test_no_same_bar_execution();
    test_P_aux_mark_prices_a_deferred_roll();
    test_Q_empty_or_nonmatching_marks_are_a_noop();
    test_R_aux_mark_changes_only_roll_pricing();
    test_S_aux_mark_for_dead_contract_still_fails_loud();
    test_T_malformed_close_marks_rejected_at_run_start();
    test_strategy_cannot_see_future_bar();
    test_target_position_delta_semantics();
    test_decision_vs_execution_time_roll();
    test_held_contract_not_relabelled_across_roll();
    test_risk_decision_gate_resizes();
    test_fills_are_validated_raw_contract();
    test_deterministic_ids_and_replay();
    test_multi_root_same_timestamp_permutation_determinism();
    test_end_of_test_policy();
    test_official_pnl_from_fills_only();
    test_negative_cl_price_end_to_end();
    test_momentum_end_to_end();
    return quant::test::summary("quant_engine_tests");
}
