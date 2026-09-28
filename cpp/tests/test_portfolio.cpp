// Phase 08 -- deterministic portfolio accounting (PortfolioAccountant).
//
//   A. cash / equity accounting identities
//   B. realized vs unrealized PnL kept strictly separate
//   C. long and short positions
//   D. multiple contracts, multiple roots
//   E. gross / net / long / short exposure via ContractSpec economics
//      (price x multiplier x contracts, never quantity x price)
//   F. mark timestamp / age / staleness; an unmarked position is never "fresh"
//   G. Fill-only realized PnL; a mark never becomes a fill price
//   H. peak equity / drawdown on the equity curve
//   I. deterministic daily-loss buckets
//   J. deterministic replay of the engine's portfolio_at_end

#include "quant_core/contract.hpp"
#include "quant_core/contract_registry.hpp"
#include "quant_core/contract_selector.hpp"
#include "quant_core/engine.hpp"
#include "quant_core/fill.hpp"
#include "quant_core/margin.hpp"
#include "quant_core/portfolio.hpp"
#include "quant_core/risk_manager.hpp"
#include "quant_core/strategy.hpp"
#include "quant_core/strategy_context.hpp"

#include "test_support.hpp"

#include <string>
#include <vector>

namespace {

using namespace quant;

constexpr std::int64_t  kMin   = 60'000'000'000LL;
constexpr std::int64_t  kDay    = 86'400'000'000'000LL;
constexpr std::int64_t  kBigExp = 500'000LL * kMin;

constexpr std::uint32_t kEs = 10;   // multiplier 50
constexpr std::uint32_t kNq = 20;   // multiplier 20
constexpr std::uint32_t kCl = 30;   // multiplier 1000
constexpr std::uint32_t kGc = 40;   // multiplier 100
constexpr std::uint32_t kZn = 50;   // multiplier 1000

ContractSpec spec(std::uint32_t id, std::string sym, std::string root, double mult) {
    return ContractSpec{.instrument_id = id, .raw_symbol = std::move(sym), .root_symbol = std::move(root),
                        .exchange = "XCME", .tick_size = 0.25, .multiplier = mult,
                        .activation_ns = 1, .expiration_ns = kBigExp};
}

ContractRegistry five_root_registry() {
    ContractRegistry r;
    r.add(spec(kEs, "ESZ6", "ES", 50.0));
    r.add(spec(kNq, "NQZ6", "NQ", 20.0));
    r.add(spec(kCl, "CLZ6", "CL", 1000.0));
    r.add(spec(kGc, "GCZ6", "GC", 100.0));
    r.add(spec(kZn, "ZNZ6", "ZN", 1000.0));
    return r;
}

Fill mkfill(const ContractRegistry& reg, std::uint32_t id, Side side, int qty, double px,
            std::int64_t ts, double commission = 0.0) {
    FillRequest req;
    req.fill_id        = 1;
    req.order_id       = 1;
    req.ts_fill_ns     = ts;
    req.instrument_id  = id;
    req.side           = side;
    req.quantity       = qty;
    req.price          = px;
    req.price_domain   = PriceDomain::RawContract;
    req.commission_usd = commission;
    return make_fill(reg, req);
}

PortfolioConfig cfg(std::int64_t staleness = 1LL << 60) {
    PortfolioConfig c;
    c.starting_capital_usd = 100'000.0;
    c.mark_staleness_tolerance_ns = staleness;
    return c;
}

// ---- A / B: cash / equity / realized / unrealized -----------------------

void test_cash_equity_realized_unrealized() {
    const auto reg = five_root_registry();
    PortfolioAccountant pf(cfg(), nullptr);

    // BUY 2 NQ @ 20000, commission 4
    pf.apply_fill(mkfill(reg, kNq, Side::Buy, 2, 20000.0, 1 * kMin, 4.0), "NQ", "signal");
    pf.observe_mark(kNq, 20000.0, 1 * kMin);
    {
        const auto s = pf.snapshot(1 * kMin);
        CHECK_CLOSE(s.gross_realized_pnl_usd, 0.0, 1e-9);
        CHECK_CLOSE(s.costs_usd, 4.0, 1e-9);
        CHECK_CLOSE(s.net_realized_pnl_usd, -4.0, 1e-9);
        CHECK_CLOSE(s.unrealized_pnl_usd, 0.0, 1e-9);
        CHECK_CLOSE(s.cash_usd, 100'000.0 - 4.0, 1e-9);          // starting + net_realized
        CHECK_CLOSE(s.equity_usd, s.cash_usd + s.unrealized_pnl_usd, 1e-9);
    }

    // price moves up: unrealized only, cash unchanged
    pf.observe_mark(kNq, 20010.0, 2 * kMin);
    {
        const auto s = pf.snapshot(2 * kMin);
        CHECK_CLOSE(s.unrealized_pnl_usd, (20010.0 - 20000.0) * 20.0 * 2, 1e-9);  // +400
        CHECK_CLOSE(s.net_realized_pnl_usd, -4.0, 1e-9);          // realized untouched
        CHECK_CLOSE(s.cash_usd, 100'000.0 - 4.0, 1e-9);
        CHECK_CLOSE(s.equity_usd, s.cash_usd + 400.0, 1e-9);
    }

    // SELL 2 NQ @ 20010, commission 4 -> realize, flat
    pf.apply_fill(mkfill(reg, kNq, Side::Sell, 2, 20010.0, 3 * kMin, 4.0), "NQ", "signal");
    pf.observe_mark(kNq, 20010.0, 3 * kMin);
    {
        const auto s = pf.snapshot(3 * kMin);
        CHECK_CLOSE(s.gross_realized_pnl_usd, 400.0, 1e-9);
        CHECK_CLOSE(s.costs_usd, 8.0, 1e-9);
        CHECK_CLOSE(s.net_realized_pnl_usd, 392.0, 1e-9);
        CHECK_CLOSE(s.unrealized_pnl_usd, 0.0, 1e-9);             // flat -> no open-trade equity
        CHECK_CLOSE(s.cash_usd, 100'392.0, 1e-9);
        CHECK_CLOSE(s.equity_usd, 100'392.0, 1e-9);
        CHECK(s.positions.empty());
    }
}

// ---- C / D / E: long + short, multi-root, exposure economics -----------

void test_multi_root_exposure() {
    const auto reg = five_root_registry();
    PortfolioAccountant pf(cfg(), nullptr);

    pf.apply_fill(mkfill(reg, kEs, Side::Buy, 3, 5000.0, kMin), "ES", "signal");   // long
    pf.apply_fill(mkfill(reg, kNq, Side::Sell, 2, 20000.0, kMin), "NQ", "signal"); // short
    pf.apply_fill(mkfill(reg, kCl, Side::Buy, 1, 70.0, kMin), "CL", "signal");     // long
    pf.observe_mark(kEs, 5000.0, kMin);
    pf.observe_mark(kNq, 20000.0, kMin);
    pf.observe_mark(kCl, 70.0, kMin);

    const auto s = pf.snapshot(kMin);

    // exposure uses price x multiplier x contracts -- NOT quantity x price.
    const double es_notional = 5000.0 * 50.0 * 3;    // 750,000  (long)
    const double nq_notional = 20000.0 * 20.0 * -2;  // -800,000 (short)
    const double cl_notional = 70.0 * 1000.0 * 1;    // 70,000   (long)

    CHECK_CLOSE(s.gross_exposure_usd, 750'000.0 + 800'000.0 + 70'000.0, 1e-6);
    CHECK_CLOSE(s.net_exposure_usd, es_notional + nq_notional + cl_notional, 1e-6);  // 20,000
    CHECK_CLOSE(s.long_exposure_usd, 750'000.0 + 70'000.0, 1e-6);
    CHECK_CLOSE(s.short_exposure_usd, 800'000.0, 1e-6);
    CHECK(s.gross_contracts == 6);
    CHECK(s.net_contracts == 2);                       // +3 -2 +1
    CHECK(s.positions.size() == 3);
    // positions ordered by instrument_id
    CHECK(s.positions[0].instrument_id == kEs);
    CHECK(s.positions[1].instrument_id == kNq);
    CHECK(s.positions[2].instrument_id == kCl);
    CHECK(s.positions[1].units == -2);
    CHECK_CLOSE(s.positions[1].signed_notional_usd, nq_notional, 1e-6);
    CHECK_CLOSE(s.correlation_adjusted_gross_exposure_usd, s.gross_exposure_usd, 1e-9);
}

// ---- F: mark timestamp / age / staleness ------------------------------

void test_mark_age_and_staleness() {
    const auto reg = five_root_registry();
    PortfolioAccountant pf(cfg(/*staleness=*/30 * 1'000'000'000LL), nullptr);  // 30s tolerance

    pf.apply_fill(mkfill(reg, kCl, Side::Buy, 1, 70.0, kMin), "CL", "signal");
    pf.apply_fill(mkfill(reg, kGc, Side::Buy, 1, 2000.0, kMin), "GC", "signal");
    pf.apply_fill(mkfill(reg, kZn, Side::Buy, 1, 110.0, kMin), "ZN", "signal");
    pf.observe_mark(kCl, 71.0, 2 * kMin);   // fresh at t=2min
    pf.observe_mark(kGc, 2000.0, kMin);     // last seen at t=1min (60s old at t=2min)
    // ZN: never marked

    const auto s = pf.snapshot(2 * kMin);
    // positions ordered by instrument_id: CL(30), GC(40), ZN(50)
    CHECK(s.positions[0].instrument_id == kCl);
    CHECK(s.positions[0].mark_present);
    CHECK(s.positions[0].mark_ts_ns == 2 * kMin);
    CHECK(s.positions[0].mark_age_ns == 0);
    CHECK(!s.positions[0].mark_is_stale);

    CHECK(s.positions[1].instrument_id == kGc);
    CHECK(s.positions[1].mark_present);
    CHECK(s.positions[1].mark_age_ns == kMin);           // 60s
    CHECK(s.positions[1].mark_is_stale);                 // 60s > 30s tolerance

    CHECK(s.positions[2].instrument_id == kZn);
    CHECK(!s.positions[2].mark_present);
    CHECK(s.positions[2].mark_is_stale);                 // an unmarked position is never fresh
    CHECK(s.positions[2].valuation_is_estimated);        // notional is an ESTIMATE, not a market value
    CHECK_CLOSE(s.positions[2].unrealized_pnl_usd, 0.0, 1e-9);  // no valuation -> 0 unrealized
    CHECK_CLOSE(s.positions[2].gross_notional_usd, 110.0 * 1000.0 * 1, 1e-6);  // estimated: avg entry
    CHECK(!s.positions[0].valuation_is_estimated);       // CL is genuinely marked

    CHECK(s.has_stale_mark);
    CHECK(!s.valuation_complete);                        // GC stale + ZN unmarked
    CHECK(s.worst_mark_age_ns == kMin);
}

// ---- K: negative futures price -> positive gross exposure, direction kept --

void test_negative_price_exposure() {
    const auto reg = five_root_registry();
    PortfolioAccountant pf(cfg(), nullptr);

    // long 1 CL, mark goes NEGATIVE (this really happened in 2020).
    pf.apply_fill(mkfill(reg, kCl, Side::Buy, 1, 10.0, kMin), "CL", "signal");
    pf.observe_mark(kCl, -20.0, 2 * kMin);
    // short 2 GC alongside, marked normally.
    pf.apply_fill(mkfill(reg, kGc, Side::Sell, 2, 2000.0, kMin), "GC", "signal");
    pf.observe_mark(kGc, 2000.0, 2 * kMin);

    const auto s = pf.snapshot(2 * kMin);
    // CL(30) first, GC(40) second
    const auto& cl = s.positions[0];
    CHECK(cl.instrument_id == kCl);
    CHECK(cl.units == 1);
    CHECK_CLOSE(cl.mark_price, -20.0, 1e-9);
    CHECK_CLOSE(cl.gross_notional_usd, 20.0 * 1000.0 * 1, 1e-6);   // |1 * -20 * 1000| = +20,000
    CHECK(cl.gross_notional_usd > 0.0);
    CHECK_CLOSE(cl.signed_notional_usd, 20'000.0, 1e-6);           // LONG: direction from units, +ve
    CHECK_CLOSE(cl.unrealized_pnl_usd, (-20.0 - 10.0) * 1000.0 * 1, 1e-6);  // (mark-entry)*mult*units

    const auto& gc = s.positions[1];
    CHECK_CLOSE(gc.gross_notional_usd, 2000.0 * 100.0 * 2, 1e-6);  // 400,000
    CHECK_CLOSE(gc.signed_notional_usd, -400'000.0, 1e-6);         // SHORT

    // portfolio aggregates: gross is a sum of magnitudes, never negative
    CHECK(s.gross_exposure_usd > 0.0);
    CHECK_CLOSE(s.gross_exposure_usd, 20'000.0 + 400'000.0, 1e-6);
    CHECK_CLOSE(s.long_exposure_usd, 20'000.0, 1e-6);              // CL classified LONG despite -ve price
    CHECK_CLOSE(s.short_exposure_usd, 400'000.0, 1e-6);
    CHECK_CLOSE(s.net_exposure_usd, 20'000.0 - 400'000.0, 1e-6);
}

// ---- G: Fill-only realized; a mark is never an execution price ---------

void test_fill_only_realized() {
    const auto reg = five_root_registry();
    PortfolioAccountant pf(cfg(), nullptr);

    pf.apply_fill(mkfill(reg, kNq, Side::Buy, 1, 20000.0, kMin), "NQ", "signal");
    // A long series of marks well above entry: they must NOT create realized PnL.
    for (int i = 2; i <= 20; ++i) pf.observe_mark(kNq, 20000.0 + i * 100.0, i * kMin);

    const auto mid = pf.snapshot(20 * kMin);
    CHECK_CLOSE(mid.gross_realized_pnl_usd, 0.0, 1e-9);   // no closing Fill -> no realized
    CHECK(mid.unrealized_pnl_usd > 0.0);                  // it is all unrealized

    // Close with a real Fill at a price BELOW every mark -> realized is the Fill
    // price, not any mark.
    pf.apply_fill(mkfill(reg, kNq, Side::Sell, 1, 20050.0, 21 * kMin), "NQ", "signal");
    const auto end = pf.snapshot(21 * kMin);
    CHECK_CLOSE(end.gross_realized_pnl_usd, (20050.0 - 20000.0) * 20.0, 1e-9);  // +1000, from the Fill
    CHECK_CLOSE(end.unrealized_pnl_usd, 0.0, 1e-9);
}

// ---- H / I: peak equity, drawdown, daily buckets ---------------------

void test_drawdown_and_daily_buckets() {
    const auto reg = five_root_registry();
    PortfolioConfig c = cfg();
    c.day_boundary_offset_ns = 0;  // UTC midnight buckets
    PortfolioAccountant pf(c, nullptr);

    // Day 0: enter long, price rises then falls back.
    pf.apply_fill(mkfill(reg, kNq, Side::Buy, 1, 20000.0, 1 * kMin), "NQ", "signal");
    pf.observe_mark(kNq, 20000.0, 1 * kMin);
    pf.observe_mark(kNq, 20500.0, 2 * kMin);   // +10,000 unrealized -> peak 110,000
    pf.observe_mark(kNq, 20100.0, 3 * kMin);   // back to +2,000

    const auto s0 = pf.snapshot(3 * kMin);
    CHECK_CLOSE(s0.peak_equity_usd, 110'000.0, 1e-6);
    CHECK_CLOSE(s0.equity_usd, 102'000.0, 1e-6);
    CHECK_CLOSE(s0.drawdown_usd, 8'000.0, 1e-6);
    CHECK_CLOSE(s0.drawdown_pct, 8'000.0 / 110'000.0, 1e-9);
    CHECK(s0.session_day_index == 0);
    CHECK_CLOSE(s0.day_start_equity_usd, 100'000.0, 1e-6);   // first observation of day 0
    CHECK_CLOSE(s0.day_equity_change_usd, 2'000.0, 1e-6);

    // Cross into day 1: the day bucket rolls, day-start equity re-seeds.
    pf.observe_mark(kNq, 20100.0, kDay + 1 * kMin);
    pf.observe_mark(kNq, 19900.0, kDay + 2 * kMin);   // -2,000 vs day-1 start
    const auto s1 = pf.snapshot(kDay + 2 * kMin);
    CHECK(s1.session_day_index == 1);
    CHECK_CLOSE(s1.day_start_equity_usd, 102'000.0, 1e-6);   // equity at first obs of day 1
    CHECK_CLOSE(s1.day_equity_change_usd, -4'000.0, 1e-6);   // 98,000 - 102,000
    CHECK_CLOSE(s1.peak_equity_usd, 110'000.0, 1e-6);        // peak persists across days
}

// ---- J: deterministic replay of the engine portfolio view ------------

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

MarketBar bar(std::int64_t idx, std::uint32_t instr, double open, double close) {
    const double hi = (open > close ? open : close) + 1.0;
    const double lo = (open < close ? open : close) - 1.0;
    return MarketBar{idx * kMin, instr, open, hi, lo, close, 100};
}

void test_engine_portfolio_deterministic_replay() {
    const auto reg = five_root_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;   // reporting-only portfolio view
    EngineConfig ecfg;
    ecfg.end_of_test = EndOfTestPolicy::LeaveOpen;
    ecfg.execution.commission_per_contract_usd = 2.0;
    const BacktestEngine engine(reg, resolver, risk, ecfg);

    std::vector<MarketBar> bars;
    for (int i = 1; i <= 8; ++i) {
        bars.push_back(bar(i, kEs, 5000.0 + i, 5000.0 + 2 * i));
        bars.push_back(bar(i, kNq, 20000.0 - i, 20000.0 - 2 * i));
    }
    const ConstantTarget strat(1.0);

    const auto a = engine.run(bars, strat);
    const auto b = engine.run(bars, strat);

    // portfolio_at_end is a pure function of the inputs
    CHECK_CLOSE(a.portfolio_at_end.equity_usd, b.portfolio_at_end.equity_usd, 1e-9);
    CHECK_CLOSE(a.portfolio_at_end.gross_exposure_usd, b.portfolio_at_end.gross_exposure_usd, 1e-9);
    CHECK_CLOSE(a.portfolio_at_end.net_exposure_usd, b.portfolio_at_end.net_exposure_usd, 1e-9);
    CHECK_CLOSE(a.portfolio_at_end.peak_equity_usd, b.portfolio_at_end.peak_equity_usd, 1e-9);
    CHECK_CLOSE(a.portfolio_at_end.drawdown_usd, b.portfolio_at_end.drawdown_usd, 1e-9);

    // realized PnL in the portfolio view matches the Fill-derived ledger exactly
    CHECK_CLOSE(a.portfolio_at_end.net_realized_pnl_usd, a.net_realized_pnl_usd, 1e-9);
    CHECK_CLOSE(a.portfolio_at_end.gross_realized_pnl_usd, a.gross_realized_pnl_usd, 1e-9);
    CHECK_CLOSE(a.portfolio_at_end.costs_usd, a.costs_usd, 1e-9);

    // two roots held, exposure non-zero, cash == starting + net realized
    CHECK(a.portfolio_at_end.positions.size() == 2);
    CHECK(a.portfolio_at_end.gross_exposure_usd > 0.0);
    CHECK_CLOSE(a.portfolio_at_end.cash_usd,
               a.portfolio_at_end.starting_capital_usd + a.portfolio_at_end.net_realized_pnl_usd,
               1e-9);
}

}  // namespace

int main() {
    test_cash_equity_realized_unrealized();
    test_multi_root_exposure();
    test_mark_age_and_staleness();
    test_negative_price_exposure();
    test_fill_only_realized();
    test_drawdown_and_daily_buckets();
    test_engine_portfolio_deterministic_replay();
    return quant::test::summary("quant_portfolio_tests");
}
