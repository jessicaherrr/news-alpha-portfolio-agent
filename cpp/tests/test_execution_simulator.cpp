// Phase 07 -- deterministic futures execution / fill simulator.
//
//   A. market buy / sell at the reference price
//   B. tick slippage + half-spread, adverse by side, from ContractSpec ticks
//   C. commission per contract per fill
//   D. limit orders: fill / no-fill / gap-through / fill price
//   E. stop orders: trigger / no-trigger / gap-through / slipped execution price
//   F. intrabar stop-vs-target ambiguity -> conservative Stop rule (never by PnL)
//   G. raw-contract-only domain + ContractSpec tick/multiplier authority
//   H. Phase 02.5 execution-domain guard still rejects BackAdjusted / RawContinuous
//   I. engine: roll close/open priced from EXECUTION-TIME raw contracts (NQM6->NQU6)
//   J. engine: a stale old-contract price is flagged, never a silent retroactive fill
//   K. engine: RejectDefer never fills at a stale price -- it defers execution
//   L. engine: deterministic N-bar latency
//   M. engine: risk gate still mandatory with the simulator in the path
//   N. engine: official PnL still derives only from Fill prices (incl. slippage)
//   O. deterministic replay

#include "quant_core/contract.hpp"
#include "quant_core/contract_registry.hpp"
#include "quant_core/contract_selector.hpp"
#include "quant_core/domain_errors.hpp"
#include "quant_core/engine.hpp"
#include "quant_core/events.hpp"
#include "quant_core/execution_simulator.hpp"
#include "quant_core/fill.hpp"
#include "quant_core/risk.hpp"
#include "quant_core/risk_manager.hpp"
#include "quant_core/strategy.hpp"
#include "quant_core/strategy_context.hpp"
#include "quant_core/types.hpp"

#include "test_support.hpp"

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace quant;

constexpr std::int64_t  kMin    = 60'000'000'000LL;
constexpr std::int64_t  kBigExp = 10'000LL * kMin;
constexpr std::uint32_t kNq     = 1;
constexpr std::uint32_t kNqM6   = 42004058;
constexpr std::uint32_t kNqU6   = 42004177;

ContractSpec spec(std::uint32_t id, std::string raw, std::string root, double tick, double mult,
                  std::int64_t act = 1, std::int64_t exp = kBigExp) {
    return ContractSpec{.instrument_id = id, .raw_symbol = std::move(raw),
                        .root_symbol = std::move(root), .exchange = "XCME",
                        .tick_size = tick, .multiplier = mult,
                        .activation_ns = act, .expiration_ns = exp};
}

ContractRegistry nq_registry(double tick = 0.25, double mult = 20.0) {
    ContractRegistry r;
    r.add(spec(kNq, "NQH6", "NQ", tick, mult));
    return r;
}

Order market_order(const ContractRegistry& reg, std::uint32_t instr, Side side, int qty) {
    Order o;
    o.order_id      = 7;
    o.signal_id     = 3;
    o.ts_created_ns = 5 * kMin;
    o.instrument_id = instr;
    o.raw_symbol    = reg.by_instrument_id(instr).raw_symbol;
    o.side          = side;
    o.quantity      = qty;
    o.order_type    = OrderType::Market;
    o.tif           = TimeInForce::Day;
    return o;
}

MarketBar exec_bar(double open, double high, double low, double close) {
    return MarketBar{5 * kMin, kNq, open, high, low, close, 1000};
}

ExecutionRequest req_for(const MarketBar& bar) {
    ExecutionRequest r;
    r.bar    = bar;
    r.ts_ns  = 5 * kMin;
    r.reason = "signal";
    return r;
}

// ---- A / B / C: market orders, slippage, spread, commission ---------------

void test_market_and_slippage() {
    const auto reg = nq_registry(0.25, 20.0);

    {  // no slippage: fill exactly at the bar open, both sides
        const BarExecutionSimulator sim(reg, ExecutionConfig{});
        const auto rep_b = sim.execute(market_order(reg, kNq, Side::Buy, 2), req_for(exec_bar(20000, 20010, 19990, 20005)), 1);
        CHECK(rep_b.outcome == ExecOutcome::Filled);
        CHECK(rep_b.fill.has_value());
        CHECK_CLOSE(rep_b.fill->fill_price, 20000.0, 1e-9);
        CHECK(rep_b.fill->price_domain == PriceDomain::RawContract);
        CHECK(rep_b.fill->quantity == 2);
        CHECK_CLOSE(rep_b.fill->tick_size, 0.25, 1e-12);   // from ContractSpec
        CHECK_CLOSE(rep_b.fill->multiplier, 20.0, 1e-12);  // from ContractSpec
        CHECK_CLOSE(rep_b.fill->commission_usd, 0.0, 1e-12);
        CHECK_CLOSE(rep_b.fill->slippage_ticks, 0.0, 1e-12);
        CHECK(rep_b.detail == "market");

        const auto rep_s = sim.execute(market_order(reg, kNq, Side::Sell, 1), req_for(exec_bar(20000, 20010, 19990, 20005)), 1);
        CHECK_CLOSE(rep_s.fill->fill_price, 20000.0, 1e-9);
    }
    {  // 3 slippage ticks + 1 spread tick == 4 ticks == 1.00 price unit, adverse
        ExecutionConfig cfg;
        cfg.slippage_ticks = 3.0;
        cfg.spread_ticks   = 1.0;
        cfg.commission_per_contract_usd = 2.5;
        const BarExecutionSimulator sim(reg, cfg);

        const auto buy = sim.execute(market_order(reg, kNq, Side::Buy, 2), req_for(exec_bar(20000, 20010, 19990, 20005)), 1);
        CHECK_CLOSE(buy.fill->fill_price, 20001.0, 1e-9);        // paid UP
        CHECK_CLOSE(buy.fill->slippage_ticks, 4.0, 1e-12);
        CHECK_CLOSE(buy.fill->commission_usd, 5.0, 1e-9);        // 2.5 * 2

        const auto sell = sim.execute(market_order(reg, kNq, Side::Sell, 2), req_for(exec_bar(20000, 20010, 19990, 20005)), 1);
        CHECK_CLOSE(sell.fill->fill_price, 19999.0, 1e-9);       // sold DOWN
        CHECK_CLOSE(sell.fill->commission_usd, 5.0, 1e-9);
    }
}

// ---- D: limit orders ----------------------------------------------------

Order limit_order(const ContractRegistry& reg, Side side, double limit) {
    Order o = market_order(reg, kNq, side, 1);
    o.order_type  = OrderType::Limit;
    o.limit_price = limit;
    return o;
}

void test_limit_orders() {
    const auto reg = nq_registry(0.25, 20.0);
    ExecutionConfig cfg;
    cfg.slippage_ticks = 5.0;  // deliberately large -- a limit order must ignore it
    const BarExecutionSimulator sim(reg, cfg);

    // BUY limit 19990
    {  // bar never trades down to the limit -> no fill
        const auto rep = sim.execute(limit_order(reg, Side::Buy, 19990.0), req_for(exec_bar(20000, 20010, 19995, 20005)), 1);
        CHECK(rep.outcome == ExecOutcome::NoFill);
        CHECK(rep.detail == "limit_unreachable");
        CHECK(!rep.fill.has_value());
    }
    {  // bar's low reaches the limit, open above -> fill AT the limit, no slippage
        const auto rep = sim.execute(limit_order(reg, Side::Buy, 19990.0), req_for(exec_bar(20000, 20010, 19985, 20005)), 1);
        CHECK(rep.outcome == ExecOutcome::Filled);
        CHECK_CLOSE(rep.fill->fill_price, 19990.0, 1e-9);
        CHECK_CLOSE(rep.fill->slippage_ticks, 0.0, 1e-12);
        CHECK(rep.detail == "limit_at_price");
    }
    {  // bar gaps open BELOW the limit -> price improvement, fill at the open
        const auto rep = sim.execute(limit_order(reg, Side::Buy, 19990.0), req_for(exec_bar(19980, 19985, 19970, 19975)), 1);
        CHECK_CLOSE(rep.fill->fill_price, 19980.0, 1e-9);
        CHECK(rep.detail == "limit_gap_through");
    }

    // SELL limit 20010
    {
        const auto no = sim.execute(limit_order(reg, Side::Sell, 20010.0), req_for(exec_bar(20000, 20005, 19990, 20002)), 1);
        CHECK(no.outcome == ExecOutcome::NoFill);
        const auto at = sim.execute(limit_order(reg, Side::Sell, 20010.0), req_for(exec_bar(20000, 20020, 19990, 20015)), 1);
        CHECK_CLOSE(at.fill->fill_price, 20010.0, 1e-9);
        const auto gap = sim.execute(limit_order(reg, Side::Sell, 20010.0), req_for(exec_bar(20030, 20040, 20025, 20035)), 1);
        CHECK_CLOSE(gap.fill->fill_price, 20030.0, 1e-9);
        CHECK(gap.detail == "limit_gap_through");
    }
}

// ---- E: stop orders ---------------------------------------------------

Order stop_order(const ContractRegistry& reg, Side side, double stop) {
    Order o = market_order(reg, kNq, side, 1);
    o.order_type = OrderType::Stop;
    o.stop_price = stop;
    return o;
}

void test_stop_orders() {
    const auto reg = nq_registry(0.25, 20.0);
    ExecutionConfig cfg;
    cfg.slippage_ticks = 2.0;  // a triggered stop IS marketable -> it slips
    const BarExecutionSimulator sim(reg, cfg);

    // BUY stop 20010
    {  // high never reaches the stop
        const auto rep = sim.execute(stop_order(reg, Side::Buy, 20010.0), req_for(exec_bar(20000, 20005, 19990, 20002)), 1);
        CHECK(rep.outcome == ExecOutcome::NoFill);
        CHECK(rep.detail == "stop_not_triggered");
    }
    {  // triggered intrabar: fill at stop + adverse slippage
        const auto rep = sim.execute(stop_order(reg, Side::Buy, 20010.0), req_for(exec_bar(20000, 20020, 19995, 20015)), 1);
        CHECK(rep.outcome == ExecOutcome::Filled);
        CHECK_CLOSE(rep.fill->fill_price, 20010.5, 1e-9);   // 20010 + 2 * 0.25
        CHECK(rep.detail == "stop_triggered");
    }
    {  // bar GAPS open above the stop: fill at the open + slippage (worse still)
        const auto rep = sim.execute(stop_order(reg, Side::Buy, 20010.0), req_for(exec_bar(20015, 20030, 20012, 20025)), 1);
        CHECK_CLOSE(rep.fill->fill_price, 20015.5, 1e-9);
        CHECK(rep.detail == "stop_gap_through");
    }

    // SELL stop 19990
    {
        const auto no = sim.execute(stop_order(reg, Side::Sell, 19990.0), req_for(exec_bar(20000, 20010, 19995, 20002)), 1);
        CHECK(no.outcome == ExecOutcome::NoFill);
        const auto hit = sim.execute(stop_order(reg, Side::Sell, 19990.0), req_for(exec_bar(20000, 20005, 19980, 19985)), 1);
        CHECK_CLOSE(hit.fill->fill_price, 19989.5, 1e-9);   // 19990 - 2 * 0.25
        const auto gap = sim.execute(stop_order(reg, Side::Sell, 19990.0), req_for(exec_bar(19985, 19988, 19970, 19975)), 1);
        CHECK_CLOSE(gap.fill->fill_price, 19984.5, 1e-9);   // 19985 - 2 * 0.25
        CHECK(gap.detail == "stop_gap_through");
    }
}

// ---- F: intrabar stop / target ambiguity ----------------------------

void test_intrabar_bracket_rule() {
    // Long position: protective stop at 19950 (below), profit target at 20050 (above).
    // exit_side is Sell.
    const double stop = 19950.0, target = 20050.0;

    // both reachable in one bar -> STOP wins (worst for the position), even though
    // the target would have been the more profitable outcome. NEVER decided by PnL.
    CHECK(resolve_intrabar_bracket(MarketBar{0, kNq, 20000, 20060, 19940, 20055, 1},
                                   Side::Sell, stop, target) == BracketExit::Stop);
    // only the target is reachable
    CHECK(resolve_intrabar_bracket(MarketBar{0, kNq, 20000, 20060, 19960, 20055, 1},
                                   Side::Sell, stop, target) == BracketExit::Target);
    // only the stop is reachable
    CHECK(resolve_intrabar_bracket(MarketBar{0, kNq, 20000, 20040, 19940, 19945, 1},
                                   Side::Sell, stop, target) == BracketExit::Stop);
    // neither
    CHECK(resolve_intrabar_bracket(MarketBar{0, kNq, 20000, 20040, 19960, 20010, 1},
                                   Side::Sell, stop, target) == BracketExit::None);
    // bar gaps open THROUGH the stop -> Stop (gap), regardless of the high
    CHECK(resolve_intrabar_bracket(MarketBar{0, kNq, 19930, 20060, 19920, 20055, 1},
                                   Side::Sell, stop, target) == BracketExit::Stop);

    // Short position: stop at 20050 (above), target at 19950 (below). exit_side Buy.
    CHECK(resolve_intrabar_bracket(MarketBar{0, kNq, 20000, 20060, 19940, 19945, 1},
                                   Side::Buy, 20050.0, 19950.0) == BracketExit::Stop);   // both -> stop
    CHECK(resolve_intrabar_bracket(MarketBar{0, kNq, 20070, 20080, 19940, 19945, 1},
                                   Side::Buy, 20050.0, 19950.0) == BracketExit::Stop);   // gap through stop
    CHECK(resolve_intrabar_bracket(MarketBar{0, kNq, 20000, 20040, 19940, 19945, 1},
                                   Side::Buy, 20050.0, 19950.0) == BracketExit::Target); // only target
}

// ---- G / H: raw-contract-only domain + spec authority ---------------

void test_domain_and_spec_authority() {
    // odd tick / multiplier: everything must come from the ContractSpec, and the
    // fill price must land on that contract's tick grid.
    ContractRegistry reg;
    reg.add(spec(kNq, "CLZ6", "CL", 0.5, 1000.0));
    const BarExecutionSimulator sim(reg, ExecutionConfig{});

    const auto rep = sim.execute(market_order(reg, kNq, Side::Buy, 1),
                                 req_for(MarketBar{5 * kMin, kNq, 70.1, 70.4, 69.9, 70.2, 10}), 1);
    CHECK(rep.outcome == ExecOutcome::Filled);
    CHECK_CLOSE(rep.fill->tick_size, 0.5, 1e-12);
    CHECK_CLOSE(rep.fill->multiplier, 1000.0, 1e-12);
    CHECK_CLOSE(rep.fill->fill_price, 70.0, 1e-9);          // 70.1 -> nearest 0.5
    CHECK(rep.fill->price_domain == PriceDomain::RawContract);
    CHECK(rep.fill->raw_symbol == "CLZ6");

    // Phase 02.5 execution-domain guard is untouched: make_fill still refuses a
    // non-RawContract price outright.
    FillRequest bad;
    bad.fill_id = 1; bad.order_id = 1; bad.ts_fill_ns = 5 * kMin;
    bad.instrument_id = kNq; bad.side = Side::Buy; bad.quantity = 1; bad.price = 70.0;
    bad.price_domain = PriceDomain::BackAdjusted;
    CHECK_THROWS_AS(make_fill(reg, bad), FillResolutionError);
    bad.price_domain = PriceDomain::RawContinuous;
    CHECK_THROWS_AS(make_fill(reg, bad), FillResolutionError);

    // The simulator itself refuses an un-normalized (Databento fixed-point ~1e13)
    // reference price before it can be priced into a fill.
    const auto fp = sim.execute(market_order(reg, kNq, Side::Buy, 1),
                                req_for(MarketBar{5 * kMin, kNq, 7.02e13, 7.1e13, 6.9e13, 7.0e13, 1}), 1);
    CHECK(fp.outcome == ExecOutcome::Rejected);
    CHECK(fp.detail == "implausible_raw_price");
    CHECK(!fp.fill.has_value());
}

// ---- H-L: signed / negative normalized prices (Phase 08.2) --------------
//
// WTI crude (CL) front-month futures traded below zero on 2020-04-20. The
// execution model must work across zero -- no comparison, rounding rule, or
// guard may assume a positive price.

void test_negative_price_execution() {
    ContractRegistry reg;
    reg.add(spec(kNq, "CLZ6", "CL", 0.01, 1000.0));   // real-style CL tick / multiplier

    {  // H. Market Buy / Sell with a negative reference; adverse slippage still adverse
        ExecutionConfig cfg;
        cfg.slippage_ticks = 2.0;
        const BarExecutionSimulator sim(reg, cfg);
        const MarketBar bar{5 * kMin, kNq, -20.00, -18.00, -25.00, -19.00, 500};

        const auto b = sim.execute(market_order(reg, kNq, Side::Buy, 1), req_for(bar), 1);
        CHECK(b.outcome == ExecOutcome::Filled);
        CHECK_CLOSE(b.fill->fill_price, -19.98, 1e-9);   // -20.00 + 2*0.01 (paid "up" == worse)
        const auto s = sim.execute(market_order(reg, kNq, Side::Sell, 1), req_for(bar), 1);
        CHECK_CLOSE(s.fill->fill_price, -20.02, 1e-9);   // -20.00 - 2*0.01 (received "down" == worse)
    }
    {  // I. Limit orders around negative values
        const BarExecutionSimulator sim(reg, ExecutionConfig{});
        // Buy limit -21: reachable (low -25 <= -21), no gap (open -20 > -21) -> fill at -21
        auto o = market_order(reg, kNq, Side::Buy, 1);
        o.order_type = OrderType::Limit; o.limit_price = -21.0;
        const auto lb = sim.execute(o, req_for(MarketBar{5 * kMin, kNq, -20, -18, -25, -19, 10}), 1);
        CHECK(lb.outcome == ExecOutcome::Filled);
        CHECK_CLOSE(lb.fill->fill_price, -21.0, 1e-9);
        CHECK(lb.detail == "limit_at_price");
        // gap-through: open -26 <= -21 -> price improvement, fill at the open
        const auto lg = sim.execute(o, req_for(MarketBar{5 * kMin, kNq, -26, -22, -30, -24, 10}), 1);
        CHECK_CLOSE(lg.fill->fill_price, -26.0, 1e-9);
        CHECK(lg.detail == "limit_gap_through");
        // Sell limit -19: reachable (high -15 >= -19), no gap (open -20 < -19) -> fill at -19
        auto o2 = market_order(reg, kNq, Side::Sell, 1);
        o2.order_type = OrderType::Limit; o2.limit_price = -19.0;
        const auto ls = sim.execute(o2, req_for(MarketBar{5 * kMin, kNq, -20, -15, -25, -17, 10}), 1);
        CHECK_CLOSE(ls.fill->fill_price, -19.0, 1e-9);
        // unreachable: Buy limit -30 with low -25 -> NoFill
        auto o3 = market_order(reg, kNq, Side::Buy, 1);
        o3.order_type = OrderType::Limit; o3.limit_price = -30.0;
        const auto un = sim.execute(o3, req_for(MarketBar{5 * kMin, kNq, -20, -18, -25, -19, 10}), 1);
        CHECK(un.outcome == ExecOutcome::NoFill);
        CHECK(un.detail == "limit_unreachable");
    }
    {  // J. Stop orders around negative values
        ExecutionConfig cfg;
        cfg.slippage_ticks = 1.0;
        const BarExecutionSimulator sim(reg, cfg);
        // Buy stop -18: triggers (high -15 >= -18), no gap (open -20 < -18) ->
        //   fill at -18 then +1 tick adverse == -17.99
        auto bs = market_order(reg, kNq, Side::Buy, 1);
        bs.order_type = OrderType::Stop; bs.stop_price = -18.0;
        const auto rb = sim.execute(bs, req_for(MarketBar{5 * kMin, kNq, -20, -14, -25, -16, 10}), 1);
        CHECK(rb.outcome == ExecOutcome::Filled);
        CHECK_CLOSE(rb.fill->fill_price, -17.99, 1e-9);
        CHECK(rb.detail == "stop_triggered");
        // Sell stop -22: triggers (low -25 <= -22) -> fill at -22 then -1 tick == -22.01
        auto ss = market_order(reg, kNq, Side::Sell, 1);
        ss.order_type = OrderType::Stop; ss.stop_price = -22.0;
        const auto rs = sim.execute(ss, req_for(MarketBar{5 * kMin, kNq, -20, -18, -25, -23, 10}), 1);
        CHECK_CLOSE(rs.fill->fill_price, -22.01, 1e-9);
        // not triggered: Buy stop -10 with high -15 -> NoFill
        auto ns = market_order(reg, kNq, Side::Buy, 1);
        ns.order_type = OrderType::Stop; ns.stop_price = -10.0;
        const auto nt = sim.execute(ns, req_for(MarketBar{5 * kMin, kNq, -20, -15, -25, -18, 10}), 1);
        CHECK(nt.outcome == ExecOutcome::NoFill);
        CHECK(nt.detail == "stop_not_triggered");
    }
    {  // K. crossing zero: a bar that straddles zero, market + limit
        const BarExecutionSimulator sim(reg, ExecutionConfig{});
        const MarketBar straddle{5 * kMin, kNq, -2.00, 3.00, -5.00, 1.00, 100};
        const auto m = sim.execute(market_order(reg, kNq, Side::Buy, 1), req_for(straddle), 1);
        CHECK_CLOSE(m.fill->fill_price, -2.00, 1e-9);
        auto lz = market_order(reg, kNq, Side::Buy, 1);
        lz.order_type = OrderType::Limit; lz.limit_price = 0.0;   // limit exactly at zero
        const auto rz = sim.execute(lz, req_for(straddle), 1);    // low -5 <= 0, open -2 <= 0 -> gap
        CHECK(rz.outcome == ExecOutcome::Filled);
        CHECK_CLOSE(rz.fill->fill_price, -2.00, 1e-9);            // fills at the open (improvement)
    }
    {  // L. negative tick-grid rounding uses the SAME round-half-away-from-zero rule
        const BarExecutionSimulator sim(reg, ExecutionConfig{});   // tick 0.01
        const auto a = sim.execute(market_order(reg, kNq, Side::Buy, 1),
                                   req_for(MarketBar{5 * kMin, kNq, -20.007, -19.0, -21.0, -20.0, 10}), 1);
        CHECK_CLOSE(a.fill->fill_price, -20.01, 1e-9);   // -2000.7 -> -2001
        const auto b = sim.execute(market_order(reg, kNq, Side::Buy, 1),
                                   req_for(MarketBar{5 * kMin, kNq, -20.004, -19.0, -21.0, -20.0, 10}), 1);
        CHECK_CLOSE(b.fill->fill_price, -20.00, 1e-9);   // -2000.4 -> -2000
    }
}

// ============================================================================
// engine-level execution behaviour
// ============================================================================

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

ContractRegistry roll_registry() {
    ContractRegistry r;
    r.add(spec(kNqM6, "NQM6", "NQ", 0.25, 20.0));
    r.add(spec(kNqU6, "NQU6", "NQ", 0.25, 20.0));
    return r;
}

MarketBar rbar(std::int64_t idx, std::uint32_t instr, double open, double close) {
    const double hi = (open > close ? open : close) + 1.0;
    const double lo = (open < close ? open : close) - 1.0;
    return MarketBar{idx * kMin, instr, open, hi, lo, close, 100};
}

// ---- I: NQM6 -> NQU6 roll priced from execution-time raw contracts ------

void test_roll_regression_contemporaneous() {
    const auto reg = roll_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    EngineConfig cfg;
    cfg.end_of_test = EndOfTestPolicy::LeaveOpen;
    const BacktestEngine engine(reg, resolver, risk, cfg);

    // The feed carries a CONTEMPORANEOUS NQM6 bar at the same timestamp as the
    // first NQU6 bar (a genuine roll overlap, Phase 04.5 same-timestamp basis).
    std::vector<MarketBar> bars{
        rbar(1, kNqM6, 20000.0, 20001.0),   // decide +1
        rbar(2, kNqM6, 20002.0, 20003.0),   // execute BUY 1 NQM6 @ 20002
        rbar(3, kNqM6, 20050.0, 20055.0),   // contemporaneous old-contract bar
        rbar(3, kNqU6, 20360.0, 20365.0),   // feed rolls: NQU6 at the SAME ts (lower id first)
        rbar(4, kNqU6, 20370.0, 20375.0),
    };
    const ConstantTarget strat(1.0);
    const auto r = engine.run(bars, strat);

    CHECK(r.rolls == 1);
    CHECK(r.rolls_priced_contemporaneous == 1);   // real same-instant close, not a fallback
    CHECK(r.rolls_priced_stale == 0);
    CHECK(r.roll_fallback_audit.empty());         // no approximation was used

    // Exactly: BUY NQM6, SELL NQM6 (roll close), BUY NQU6 (roll open).
    CHECK(r.fills.size() == 3);
    const Fill* buy_m6  = nullptr;
    const Fill* sell_m6 = nullptr;
    const Fill* buy_u6  = nullptr;
    for (const auto& f : r.fills) {
        // no fill is ever relabelled: raw_symbol always matches its instrument_id.
        CHECK(f.raw_symbol == reg.by_instrument_id(f.instrument_id).raw_symbol);
        CHECK(f.price_domain == PriceDomain::RawContract);
        if (f.instrument_id == kNqM6 && f.side == Side::Buy)  buy_m6  = &f;
        if (f.instrument_id == kNqM6 && f.side == Side::Sell) sell_m6 = &f;
        if (f.instrument_id == kNqU6 && f.side == Side::Buy)  buy_u6  = &f;
    }
    CHECK(buy_m6 != nullptr && sell_m6 != nullptr && buy_u6 != nullptr);

    // the OLD position is closed AS NQM6, at the contemporaneous NQM6 close, at
    // the execution timestamp -- never relabelled, never retroactive, never the
    // NQU6 price, never a back-adjusted price.
    CHECK(sell_m6->raw_symbol == "NQM6");
    CHECK_CLOSE(sell_m6->fill_price, 20055.0, 1e-9);          // NQM6 close at ts 3, NOT 20365
    CHECK(sell_m6->ts_fill_ns == 3 * kMin);                  // execution ts
    CHECK(sell_m6->ts_fill_ns > buy_m6->ts_fill_ns);         // strictly forward in time

    // the NEW target is opened AS NQU6, at NQU6's real execution-time price.
    CHECK(buy_u6->raw_symbol == "NQU6");
    CHECK_CLOSE(buy_u6->fill_price, 20360.0, 1e-9);
    CHECK(buy_u6->ts_fill_ns == 3 * kMin);

    // NQM6 ends flat; only NQU6 is held.
    CHECK(r.final_positions.size() == 1);
    CHECK(r.final_positions[0].instrument_id == kNqU6);
    CHECK(r.final_positions[0].units == 1);

    // official PnL from the fills only: NQM6 (20002 -> 20055) * 20 = 1060.
    CHECK_CLOSE(r.gross_realized_pnl_usd, (20055.0 - 20002.0) * 20.0, 1e-6);
}

// ---- J: an explicit StaleObservedClose is flagged + audited, never retroactive

void test_roll_stale_price_is_flagged() {
    const auto reg = roll_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    EngineConfig cfg;
    cfg.end_of_test = EndOfTestPolicy::LeaveOpen;
    cfg.roll.fallback = RollPriceFallback::StaleObservedClose;  // explicit opt-in approximation
    const BacktestEngine engine(reg, resolver, risk, cfg);

    std::vector<MarketBar> bars{
        rbar(1, kNqM6, 20000.0, 20001.0),
        rbar(2, kNqM6, 20002.0, 20003.0),   // BUY 1 NQM6
        rbar(3, kNqU6, 20360.0, 20365.0),   // roll: NO contemporaneous NQM6 bar
        rbar(4, kNqU6, 20370.0, 20375.0),
    };
    const auto r = engine.run(bars, ConstantTarget(1.0));

    CHECK(r.rolls == 1);
    CHECK(r.rolls_priced_stale == 1);                 // the staleness is RECORDED
    CHECK(r.rolls_priced_contemporaneous == 0);

    const Fill* sell_m6 = nullptr;
    for (const auto& f : r.fills) {
        if (f.instrument_id == kNqM6 && f.side == Side::Sell) sell_m6 = &f;
    }
    CHECK(sell_m6 != nullptr);
    CHECK_CLOSE(sell_m6->fill_price, 20003.0, 1e-9);  // last OBSERVED NQM6 close
    CHECK(sell_m6->ts_fill_ns == 3 * kMin);           // stamped at execution T, not bar 2

    // provenance of the stale reference is preserved without touching Fill.
    CHECK(r.roll_fallback_audit.size() == 1);
    const RollFallbackAudit& a = r.roll_fallback_audit[0];
    CHECK(a.instrument_id == kNqM6);
    CHECK(a.raw_symbol == "NQM6");
    CHECK(a.execution_ts_ns == 3 * kMin);
    CHECK(a.execution_ts_ns == sell_m6->ts_fill_ns);
    CHECK(a.reference_price_ts_ns == 2 * kMin);       // the bar the price came from
    CHECK(a.reference_price_ts_ns != a.execution_ts_ns);   // stale source ts != fill ts -- auditable
    CHECK(a.reference_age_ns == 1 * kMin);
    CHECK_CLOSE(a.reference_price, 20003.0, 1e-9);
    CHECK(a.fallback_policy == RollPriceFallback::StaleObservedClose);
}

// ---- K: RejectDefer (the Phase 07.1 DEFAULT) never fills at a stale price ----

void test_roll_reject_defer_is_the_default() {
    // RollExecutionPolicy{} must default to RejectDefer.
    CHECK(RollExecutionPolicy{}.fallback == RollPriceFallback::RejectDefer);
    CHECK(EngineConfig{}.roll.fallback == RollPriceFallback::RejectDefer);

    const auto reg = roll_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    EngineConfig cfg;
    cfg.end_of_test = EndOfTestPolicy::LeaveOpen;
    // deliberately NOT setting cfg.roll.fallback -- rely on the default.
    const BacktestEngine engine(reg, resolver, risk, cfg);

    std::vector<MarketBar> bars{
        rbar(1, kNqM6, 20000.0, 20001.0),
        rbar(2, kNqM6, 20002.0, 20003.0),   // BUY 1 NQM6
        rbar(3, kNqU6, 20360.0, 20365.0),   // roll attempt -> deferred (no NQM6 bar)
        rbar(4, kNqU6, 20370.0, 20375.0),   // deferred again
        rbar(5, kNqU6, 20380.0, 20385.0),   // deferred again
    };
    const auto r = engine.run(bars, ConstantTarget(1.0));

    CHECK(r.rolls == 0);                    // no roll close-leg was ever executed
    CHECK(r.rolls_deferred >= 1);
    CHECK(r.rolls_priced_stale == 0);
    CHECK(r.roll_fallback_audit.empty());
    // only the original NQM6 BUY filled; no stale close, no NQU6 entry.
    CHECK(r.fills.size() == 1);
    CHECK(r.fills[0].instrument_id == kNqM6);
    CHECK(r.final_positions.size() == 1);
    CHECK(r.final_positions[0].instrument_id == kNqM6);
    CHECK(r.final_positions[0].units == 1);
}

// ---- causality: no Fill ever precedes its engine event time -------------

void test_no_fill_precedes_engine_event_time() {
    const auto reg = roll_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    EngineConfig cfg;
    cfg.roll.fallback = RollPriceFallback::StaleObservedClose;  // the risky path
    cfg.execution.slippage_ticks = 1.0;
    const BacktestEngine engine(reg, resolver, risk, cfg);

    // a held-position roll (stale close-leg) plus a force-liquidate EOT.
    std::vector<MarketBar> bars{
        rbar(1, kNqM6, 20000.0, 20001.0),
        rbar(2, kNqM6, 20002.0, 20003.0),
        rbar(3, kNqM6, 20004.0, 20005.0),
        rbar(4, kNqU6, 20360.0, 20365.0),   // stale roll close-leg here
        rbar(5, kNqU6, 20370.0, 20375.0),
    };
    const auto r = engine.run(bars, ConstantTarget(1.0));

    // fills are produced in execution order, which is engine-time order.
    std::int64_t prev = 0;
    for (const auto& f : r.fills) {
        CHECK(f.ts_fill_ns >= prev);          // non-decreasing
        CHECK(f.ts_fill_ns >= 1 * kMin);      // >= the first engine event
        CHECK(f.ts_fill_ns <= 5 * kMin);      // never past the final engine event
        prev = f.ts_fill_ns;
    }
    // the stale roll close-leg is stamped at its execution event (bar 4), not the
    // earlier NQM6 bar it took the price from.
    for (const auto& f : r.fills) {
        if (f.instrument_id == kNqM6 && f.side == Side::Sell && f.ts_fill_ns < 5 * kMin) {
            CHECK(f.ts_fill_ns == 4 * kMin);
        }
    }
    // the EOT liquidation of NQU6 is stamped at the FINAL engine event time.
    for (const auto& f : r.fills) {
        if (f.instrument_id == kNqU6 && f.side == Side::Sell) {
            CHECK(f.ts_fill_ns == 5 * kMin);
        }
    }
}

// ---- an expired held contract cannot generate a retroactive fill --------

void test_expired_held_contract_fails_loudly() {
    // NQM6 expires at ts 3.5*kMin; NQU6 lives long. The feed rolls to NQU6 at ts
    // 5*kMin -- AFTER NQM6 is dead -- while an NQM6 position is still held.
    ContractRegistry reg;
    reg.add(spec(kNqM6, "NQM6", "NQ", 0.25, 20.0, 1, (7 * kMin) / 2));   // exp 3.5 min
    reg.add(spec(kNqU6, "NQU6", "NQ", 0.25, 20.0, 1, kBigExp));

    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    EngineConfig cfg;
    cfg.end_of_test = EndOfTestPolicy::LeaveOpen;
    cfg.roll.fallback = RollPriceFallback::StaleObservedClose;  // must STILL fail loudly
    const BacktestEngine engine(reg, resolver, risk, cfg);

    std::vector<MarketBar> bars{
        rbar(1, kNqM6, 20000.0, 20001.0),
        rbar(2, kNqM6, 20002.0, 20003.0),   // BUY 1 NQM6 (contract dies at ts 3.5)
        rbar(5, kNqU6, 20360.0, 20365.0),   // roll discovered here: NQM6 already expired
    };
    // No retroactive expiry fill is manufactured -- the engine fails loudly.
    CHECK_THROWS_AS(engine.run(bars, ConstantTarget(1.0)), ExecutionDomainError);
}

// ---- EOT liquidation stays causal; expired-contract EOT fails loudly ----

// Multi-asset ES / NQ / CL fixture -- roots stop producing bars at different
// times; the global final engine event is later than NQ's and CL's last quote.
ContractRegistry three_root_eot_registry() {
    ContractRegistry r;
    r.add(spec(700, "ESU6", "ES", 0.25, 50.0));     // long-lived, trades through ts 8
    r.add(spec(800, "NQU6", "NQ", 0.25, 20.0));     // stops at ts 3
    r.add(spec(900, "CLU6", "CL", 0.01, 1000.0));   // stops at ts 5
    return r;
}

std::vector<MarketBar> three_root_eot_bars() {
    std::vector<MarketBar> bars;
    for (int i = 1; i <= 8; ++i) bars.push_back(rbar(i, 700, 5000.0 + i, 5000.0 + i));
    for (int i = 1; i <= 3; ++i) bars.push_back(rbar(i, 800, 20000.0 + i, 20000.0 + i));
    for (int i = 1; i <= 5; ++i) bars.push_back(rbar(i, 900, 70.0 + i, 70.0 + i));
    return bars;
}

void test_multi_root_eot_stale_price_provenance() {
    const auto reg = three_root_eot_registry();
    const auto bars = three_root_eot_bars();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    const ConstantTarget strat(1.0);   // all three roots go long +1

    {  // DEFAULT: ForceLiquidateFinalClose + EotStalePolicy::LeaveOpen, tolerance 0
        const auto r = BacktestEngine(reg, resolver, risk, EngineConfig{}).run(bars, strat);

        // ES's last bar IS the final engine event (ts 8) -> contemporaneous -> liquidated.
        CHECK(r.eot_liquidations.size() == 1);
        const EotLiquidationAudit& es = r.eot_liquidations[0];
        CHECK(es.raw_symbol == "ESU6");
        CHECK(es.is_contemporaneous);
        CHECK(!es.is_approximation);
        CHECK(es.execution_ts_ns == 8 * kMin);
        CHECK(es.reference_price_ts_ns == 8 * kMin);
        CHECK(es.reference_age_ns == 0);

        // NQ (mark age 5m) and CL (mark age 3m) are NOT force-liquidated -- the
        // default reliability policy does not manufacture a fill from a stale mark.
        CHECK(r.final_positions.size() == 2);
        CHECK(r.eot_positions_left_open_stale == 2);
        CHECK(r.final_position_marks.size() == 2);

        const OpenPositionMark* nq = nullptr;
        const OpenPositionMark* cl = nullptr;
        for (const auto& m : r.final_position_marks) {
            if (m.raw_symbol == "NQU6") nq = &m;
            if (m.raw_symbol == "CLU6") cl = &m;
        }
        CHECK(nq != nullptr && cl != nullptr);
        CHECK(nq->is_stale);
        CHECK(nq->mark_ts_ns == 3 * kMin);
        CHECK(nq->mark_age_ns == 5 * kMin);          // 8 - 3
        CHECK_CLOSE(nq->mark_price, 20003.0, 1e-9);  // NQ's last observed close, not treated as current
        CHECK(cl->is_stale);
        CHECK(cl->mark_age_ns == 3 * kMin);          // 8 - 5

        // stale marks never enter realized PnL; unrealized is reported separately.
        CHECK_CLOSE(r.gross_realized_pnl_usd, (5008.0 - 5002.0) * 50.0, 1e-6);  // ES only
        CHECK(std::fabs(r.unrealized_pnl_usd_at_end) > 0.0);
        CHECK_CLOSE(r.net_equity_usd_at_end,
                    r.net_realized_pnl_usd + r.unrealized_pnl_usd_at_end, 1e-9);

        // every fill is causal (<= the final engine event, never before it).
        for (const auto& f : r.fills) {
            CHECK(f.ts_fill_ns >= 1 * kMin);
            CHECK(f.ts_fill_ns <= 8 * kMin);
        }
    }
    {  // EotStalePolicy::Fail -> refuse forced liquidation
        EngineConfig cfg;
        cfg.eot.stale = EotStalePolicy::Fail;
        CHECK_THROWS_AS(BacktestEngine(reg, resolver, risk, cfg).run(bars, strat),
                        ExecutionDomainError);
    }
    {  // EotStalePolicy::ForceApproximateStaleClose -> liquidate, flagged + audited
        EngineConfig cfg;
        cfg.eot.stale = EotStalePolicy::ForceApproximateStaleClose;
        const auto r = BacktestEngine(reg, resolver, risk, cfg).run(bars, strat);

        CHECK(r.eot_liquidations.size() == 3);
        CHECK(r.final_positions.empty());
        int approx = 0, contemp = 0;
        for (const auto& a : r.eot_liquidations) {
            CHECK(a.execution_ts_ns == 8 * kMin);                   // causal
            CHECK(a.reference_price_ts_ns <= a.execution_ts_ns);    // never from the future
            if (a.is_approximation) {
                ++approx;
                CHECK(!a.is_contemporaneous);
                CHECK(a.reference_age_ns > 0);
                CHECK(a.reference_price_ts_ns != a.execution_ts_ns);  // auditable staleness
            } else {
                ++contemp;
                CHECK(a.is_contemporaneous);
            }
        }
        CHECK(contemp == 1);   // ES
        CHECK(approx == 2);    // NQ, CL -- approximation is flagged, not silent
        for (const auto& f : r.fills) CHECK(f.ts_fill_ns <= 8 * kMin);
    }
    {  // configurable freshness tolerance: 3 minutes covers CL (age 3m) but not NQ (age 5m)
        EngineConfig cfg;
        cfg.eot.contemporaneous_tolerance_ns = 3 * kMin;
        const auto r = BacktestEngine(reg, resolver, risk, cfg).run(bars, strat);
        CHECK(r.eot_liquidations.size() == 2);          // ES + CL
        CHECK(r.final_positions.size() == 1);           // NQ still stale -> left open
        CHECK(r.final_positions[0].raw_symbol == "NQU6");
        for (const auto& a : r.eot_liquidations) {
            if (a.raw_symbol == "CLU6") {
                CHECK(a.is_contemporaneous);            // within tolerance
                CHECK(a.reference_age_ns == 3 * kMin);  // raw age still recorded, not hidden
            }
        }
    }
}

// ---- EOT of an expired contract: never a stale/illegal forced fill ----

void test_eot_expired_contract_policies() {
    ContractRegistry reg;
    reg.add(spec(700, "ESU6", "ES", 0.25, 50.0));                      // long-lived
    reg.add(spec(800, "NQM6", "NQ", 0.25, 20.0, 1, (7 * kMin) / 2));   // NQ expires 3.5 min

    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    const ConstantTarget strat(1.0);

    std::vector<MarketBar> bars;
    for (int i = 1; i <= 6; ++i) bars.push_back(rbar(i, 700, 5000.0 + i, 5000.0 + i));
    for (int i = 1; i <= 3; ++i) bars.push_back(rbar(i, 800, 20000.0 + i, 20000.0 + i));

    {  // DEFAULT (LeaveOpen): the expired NQ position is left open, marked + aged,
       // never liquidated at a stale or illegal price.
        const auto r = BacktestEngine(reg, resolver, risk, EngineConfig{}).run(bars, strat);
        CHECK(r.final_positions.size() == 1);
        CHECK(r.final_positions[0].raw_symbol == "NQM6");
        CHECK(r.eot_positions_left_open_stale == 1);
        for (const auto& m : r.final_position_marks) {
            if (m.raw_symbol == "NQM6") {
                CHECK(m.is_stale);
                CHECK(m.mark_age_ns == 3 * kMin);   // 6 - 3
            }
        }
        for (const auto& f : r.fills) CHECK(f.ts_fill_ns <= 6 * kMin);
    }
    {  // Fail
        EngineConfig cfg;
        cfg.eot.stale = EotStalePolicy::Fail;
        CHECK_THROWS_AS(BacktestEngine(reg, resolver, risk, cfg).run(bars, strat),
                        ExecutionDomainError);
    }
    {  // ForceApproximateStaleClose still cannot fill an expired contract
        EngineConfig cfg;
        cfg.eot.stale = EotStalePolicy::ForceApproximateStaleClose;
        CHECK_THROWS_AS(BacktestEngine(reg, resolver, risk, cfg).run(bars, strat),
                        ExecutionDomainError);
    }
}

// ---- L: deterministic N-bar latency ---------------------------------

void test_deterministic_latency() {
    const auto reg = nq_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;

    std::vector<MarketBar> bars;
    for (int i = 1; i <= 6; ++i) bars.push_back(rbar(i, kNq, 100.0 + i, 100.0 + i));

    {  // latency 0: a signal on bar i executes on bar i+1
        EngineConfig cfg;
        cfg.end_of_test = EndOfTestPolicy::LeaveOpen;
        cfg.latency_bars = 0;
        const auto r = BacktestEngine(reg, resolver, risk, cfg).run(bars, ConstantTarget(1.0));
        CHECK(!r.fills.empty());
        CHECK(r.fills.front().ts_fill_ns == bars[1].ts_event_ns);
    }
    {  // latency 2: a signal on bar i executes on bar i+3
        EngineConfig cfg;
        cfg.end_of_test = EndOfTestPolicy::LeaveOpen;
        cfg.latency_bars = 2;
        const auto a = BacktestEngine(reg, resolver, risk, cfg).run(bars, ConstantTarget(1.0));
        const auto b = BacktestEngine(reg, resolver, risk, cfg).run(bars, ConstantTarget(1.0));
        CHECK(!a.fills.empty());
        CHECK(a.fills.front().ts_fill_ns == bars[3].ts_event_ns);
        CHECK_CLOSE(a.fills.front().fill_price, bars[3].open, 1e-9);
        // deterministic replay under latency
        CHECK(a.fills.size() == b.fills.size());
        CHECK(a.fills.front().fill_id == b.fills.front().fill_id);
        CHECK_CLOSE(a.net_realized_pnl_usd, b.net_realized_pnl_usd, 1e-9);
    }
}

// ---- M: risk gate still mandatory with the simulator in the path ---------

void test_risk_gate_still_mandatory() {
    const auto reg = nq_registry();
    const RegistryActiveContractResolver resolver(reg);
    const MaxContractsRiskManager risk(RiskLimits{});  // cap 5
    EngineConfig cfg;
    cfg.end_of_test = EndOfTestPolicy::LeaveOpen;
    cfg.execution.slippage_ticks = 1.0;
    const BacktestEngine engine(reg, resolver, risk, cfg);

    std::vector<MarketBar> bars{rbar(1, kNq, 100.0, 100.0), rbar(2, kNq, 100.0, 100.0)};
    const auto r = engine.run(bars, ConstantTarget(9.0));   // wants +9; risk caps to 5

    CHECK(r.risk_decisions.size() == r.orders.size());       // every order reviewed
    CHECK(r.risk_decisions[0].verdict == RiskVerdict::Resize);
    CHECK(r.fills.size() == 1);
    CHECK(r.fills[0].quantity == 5);                          // gated BEFORE the simulator filled
    CHECK(r.final_positions[0].units == 5);
}

// ---- N: official PnL still derives only from Fill prices (incl. slippage) -

void test_pnl_from_fills_only_with_slippage() {
    const auto reg = nq_registry(0.25, 20.0);
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    EngineConfig cfg;
    cfg.end_of_test = EndOfTestPolicy::LeaveOpen;
    cfg.execution.slippage_ticks = 1.0;   // 0.25 price units, adverse
    const BacktestEngine engine(reg, resolver, risk, cfg);

    // closes deliberately offset from opens so a close-based PnL differs.
    std::vector<MarketBar> bars{
        rbar(1, kNq, 20000.0, 20009.0),
        rbar(2, kNq, 20000.0, 20009.0),   // BUY 1 @ 20000 + 0.25 = 20000.25
        rbar(3, kNq, 20100.0, 20140.0),   // SELL 1 @ 20100 - 0.25 = 20099.75
        rbar(4, kNq, 20100.0, 20140.0),
    };
    const auto r = engine.run(bars, Scripted({1, 1, 0, 0}));

    CHECK(r.fills.size() == 2);
    CHECK_CLOSE(r.fills[0].fill_price, 20000.25, 1e-9);
    CHECK_CLOSE(r.fills[1].fill_price, 20099.75, 1e-9);
    const double from_fills = (r.fills[1].fill_price - r.fills[0].fill_price) * r.fills[0].multiplier;
    CHECK_CLOSE(r.gross_realized_pnl_usd, from_fills, 1e-6);      // 99.5 * 20 = 1990
    CHECK_CLOSE(r.gross_realized_pnl_usd, 1990.0, 1e-6);
    // NOT the close-based number (20140 - 20009) * 20 = 2620
    CHECK(r.gross_realized_pnl_usd < 2619.0 || r.gross_realized_pnl_usd > 2621.0);
}

// ---- O: injected simulator + deterministic replay --------------------

void test_injected_simulator_and_replay() {
    const auto reg = nq_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    ExecutionConfig ec;
    ec.slippage_ticks = 2.0;
    ec.commission_per_contract_usd = 1.5;
    const BarExecutionSimulator sim(reg, ec);

    std::vector<MarketBar> bars;
    for (int i = 1; i <= 8; ++i) bars.push_back(rbar(i, kNq, 100.0 + i, 100.0 + i));
    const Scripted strat({1, 1, 0, -1, -1, 0, 0, 0});

    const BacktestEngine engine(reg, resolver, risk, sim, EngineConfig{});
    const auto a = engine.run(bars, strat);
    const BacktestEngine engine2(reg, resolver, risk, sim, EngineConfig{});
    const auto b = engine2.run(bars, Scripted({1, 1, 0, -1, -1, 0, 0, 0}));

    CHECK(a.fills.size() == b.fills.size());
    for (std::size_t i = 0; i < a.fills.size(); ++i) {
        CHECK(a.fills[i].fill_id == b.fills[i].fill_id);
        CHECK(a.fills[i].instrument_id == b.fills[i].instrument_id);
        CHECK_CLOSE(a.fills[i].fill_price, b.fills[i].fill_price, 1e-12);
        CHECK_CLOSE(a.fills[i].commission_usd, b.fills[i].commission_usd, 1e-12);
        CHECK(a.fills[i].fill_id == i + 1);           // dense monotonic ids
    }
    CHECK_CLOSE(a.net_realized_pnl_usd, b.net_realized_pnl_usd, 1e-9);
    CHECK(a.costs_usd > 0.0);
}

}  // namespace

int main() {
    test_market_and_slippage();
    test_limit_orders();
    test_stop_orders();
    test_intrabar_bracket_rule();
    test_domain_and_spec_authority();
    test_negative_price_execution();
    test_roll_regression_contemporaneous();
    test_roll_stale_price_is_flagged();
    test_roll_reject_defer_is_the_default();
    test_no_fill_precedes_engine_event_time();
    test_expired_held_contract_fails_loudly();
    test_multi_root_eot_stale_price_provenance();
    test_eot_expired_contract_policies();
    test_deterministic_latency();
    test_risk_gate_still_mandatory();
    test_pnl_from_fills_only_with_slippage();
    test_injected_simulator_and_replay();
    return quant::test::summary("quant_execution_simulator_tests");
}
