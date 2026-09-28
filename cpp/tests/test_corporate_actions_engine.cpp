// ETF corporate actions (splits / cash distributions) wired into the
// deterministic BacktestEngine bar-replay loop -- EngineConfig::
// corporate_actions (engine.hpp), consulted inside engine.cpp's per-bar loop
// immediately before that bar's own open mark / queued-intent execution. This
// mirrors RollExecutionPolicy::close_marks's wiring pattern exactly (see its
// docs in engine.hpp and its own engine-level tests in test_engine.cpp).
//
// test_corporate_actions.cpp already proves the standalone PortfolioAccountant
// invariants (apply_split / record_distribution_ex_date / _payment) in
// isolation; this file proves the SAME invariants hold through the FULL engine
// replay, plus the engine-level wiring guarantees below:
//
//   1. an empty (or non-matching) CorporateActionsConfig produces a
//      byte-for-byte identical BacktestResult to the frozen reference path.
//   2. a 2-for-1 forward split applied mid-backtest to an open position
//      rebases units / avg_entry_price at the EXACT bar it is keyed to, with
//      zero realized-PnL / cash effect at that instant.
//   3. a cash distribution's ex-date entitlement reflects the position held
//      INTO that bar -- NOT a fill that executes ON that same bar (proven by
//      constructing exactly that same-bar-fill case) -- and its later
//      pay-date leg moves that (pre-fill-snapshotted) receivable into
//      distribution_income_usd without ever touching cash_usd / equity_usd /
//      net_realized_pnl_usd.

#include "quant_core/contract.hpp"
#include "quant_core/contract_registry.hpp"
#include "quant_core/contract_selector.hpp"
#include "quant_core/corporate_actions.hpp"
#include "quant_core/engine.hpp"
#include "quant_core/risk_manager.hpp"
#include "quant_core/strategy.hpp"
#include "quant_core/strategy_context.hpp"
#include "quant_core/types.hpp"

#include "test_support.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace quant;

constexpr std::int64_t  kMin    = 60'000'000'000LL;
constexpr std::uint32_t kSpy    = 100;  // pretend-ETF instrument id, multiplier 1
constexpr std::int64_t  kBigExp = 10'000LL * kMin;

ContractSpec etf_spec(std::uint32_t id, std::string sym, std::string root) {
    return ContractSpec{.instrument_id = id, .raw_symbol = std::move(sym), .root_symbol = std::move(root),
                        .exchange = "XNAS", .tick_size = 0.01, .multiplier = 1.0,
                        .activation_ns = 1, .expiration_ns = kBigExp};
}

ContractRegistry etf_registry() {
    ContractRegistry r;
    r.add(etf_spec(kSpy, "SPY", "SPY"));
    return r;
}

MarketBar bar(std::int64_t idx, std::uint32_t instr, double open, double close) {
    const double hi = (open > close ? open : close) + 0.5;
    const double lo = (open < close ? open : close) - 0.5;
    return MarketBar{.ts_event_ns = idx * kMin, .instrument_id = instr,
                     .open = open, .high = hi, .low = lo, .close = close, .volume = 100};
}

// Fires a decision (the scripted target) ONLY at the given 0-indexed decide()
// call ordinals; every other call returns std::nullopt (no decision, Phase
// 11.1 semantics). Deliberately NOT test_engine.cpp's `Scripted` fixture
// (which re-affirms a target on every bar): re-affirming a constant SHARE
// target after a split rebases the held unit count would itself queue a real
// corrective trade back to the pre-split share count on the very next bar --
// correct engine behaviour, but it would silently contaminate these tests'
// "no further orders after the corporate action" construction. Firing a
// decision only at explicit ordinals keeps every fill in each scenario
// intentional and accounted for.
struct DecideAt final : Strategy {
    explicit DecideAt(std::vector<std::pair<std::size_t, double>> schedule)
        : schedule_(std::move(schedule)) {}
    std::optional<Signal> decide(const StrategyContext& ctx) const override {
        const std::size_t call_idx = calls_++;
        for (const auto& [idx, target] : schedule_) {
            if (idx == call_idx) {
                Signal s;
                s.ts_decision_ns = ctx.decision_ts_ns();
                s.root_symbol    = std::string(ctx.root_symbol());
                s.target_units   = target;
                s.rationale_code = "decide_at";
                return s;
            }
        }
        return std::nullopt;
    }
    std::vector<std::pair<std::size_t, double>> schedule_;
    mutable std::size_t calls_{0};
};

// LeaveOpen so a surviving open position is reported verbatim in
// final_positions / portfolio_at_end without an end-of-test forced-liquidation
// fill muddying the numbers under test.
EngineConfig leave_open_config() {
    EngineConfig cfg;
    cfg.end_of_test = EndOfTestPolicy::LeaveOpen;
    return cfg;
}

// ---------------------------------------------------------------------------
// 1. empty / non-matching CorporateActionsConfig -> byte-for-byte frozen path
// ---------------------------------------------------------------------------

void assert_results_identical(const BacktestResult& a, const BacktestResult& b) {
    CHECK(a.events_processed == b.events_processed);
    CHECK(a.bars_processed == b.bars_processed);
    CHECK(a.signals_generated == b.signals_generated);
    CHECK(a.orders_generated == b.orders_generated);
    CHECK(a.fills_generated == b.fills_generated);
    CHECK(a.closed_trades == b.closed_trades);
    CHECK(a.rolls == b.rolls);
    CHECK(a.non_fills == b.non_fills);
    CHECK_CLOSE(a.gross_realized_pnl_usd, b.gross_realized_pnl_usd, 1e-12);
    CHECK_CLOSE(a.costs_usd, b.costs_usd, 1e-12);
    CHECK_CLOSE(a.net_realized_pnl_usd, b.net_realized_pnl_usd, 1e-12);
    CHECK_CLOSE(a.max_drawdown_usd, b.max_drawdown_usd, 1e-12);
    CHECK_CLOSE(a.unrealized_pnl_usd_at_end, b.unrealized_pnl_usd_at_end, 1e-12);
    CHECK_CLOSE(a.net_equity_usd_at_end, b.net_equity_usd_at_end, 1e-12);

    CHECK(a.final_positions.size() == b.final_positions.size());
    for (std::size_t i = 0; i < a.final_positions.size(); ++i) {
        CHECK(a.final_positions[i].instrument_id == b.final_positions[i].instrument_id);
        CHECK(a.final_positions[i].units == b.final_positions[i].units);
        CHECK_CLOSE(a.final_positions[i].avg_entry_price, b.final_positions[i].avg_entry_price, 1e-12);
    }

    CHECK(a.fills.size() == b.fills.size());
    for (std::size_t i = 0; i < a.fills.size(); ++i) {
        CHECK(a.fills[i].fill_id == b.fills[i].fill_id);
        CHECK(a.fills[i].instrument_id == b.fills[i].instrument_id);
        CHECK(a.fills[i].ts_fill_ns == b.fills[i].ts_fill_ns);
        CHECK_CLOSE(a.fills[i].fill_price, b.fills[i].fill_price, 1e-12);
    }

    CHECK_CLOSE(a.portfolio_at_end.cash_usd, b.portfolio_at_end.cash_usd, 1e-12);
    CHECK_CLOSE(a.portfolio_at_end.equity_usd, b.portfolio_at_end.equity_usd, 1e-12);
    CHECK_CLOSE(a.portfolio_at_end.distribution_receivable_usd,
               b.portfolio_at_end.distribution_receivable_usd, 1e-12);
    CHECK_CLOSE(a.portfolio_at_end.distribution_income_usd,
               b.portfolio_at_end.distribution_income_usd, 1e-12);
}

void test_empty_config_is_byte_identical_to_default() {
    const auto reg = etf_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;

    std::vector<MarketBar> bars;
    for (int i = 1; i <= 6; ++i) bars.push_back(bar(i, kSpy, 10.0 + i * 0.1, 10.0 + i * 0.1));
    const DecideAt strat_default({{0, 100.0}});

    // (a) EngineConfig{} left completely untouched -- the pre-existing
    //     construction pattern used everywhere else in this codebase.
    const BacktestEngine engine_default(reg, resolver, risk, EngineConfig{});
    const BacktestResult r_default = engine_default.run(bars, strat_default);

    // (b) explicit-but-empty CorporateActionsConfig.
    const DecideAt strat_explicit({{0, 100.0}});
    EngineConfig cfg_explicit_empty;
    cfg_explicit_empty.corporate_actions = CorporateActionsConfig{};
    const BacktestEngine engine_explicit(reg, resolver, risk, cfg_explicit_empty);
    const BacktestResult r_explicit = engine_explicit.run(bars, strat_explicit);

    // (c) a CorporateActionsConfig carrying REAL, well-formed entries keyed to
    //     timestamps that never occur in this bar set -- "no match" must be
    //     exactly as inert as "empty".
    const DecideAt strat_no_match({{0, 100.0}});
    EngineConfig cfg_no_match;
    cfg_no_match.corporate_actions.splits[{kSpy, 999 * kMin}] =
        SplitAction{.instrument_id = kSpy, .effective_ts_ns = 999 * kMin, .ratio = 2.0};
    cfg_no_match.corporate_actions.ex_date_distributions[{kSpy, 999 * kMin}] =
        CashDistribution{.instrument_id = kSpy, .ex_date_ts_ns = 999 * kMin,
                         .pay_date_ts_ns = 1000 * kMin, .amount_per_share_usd = 1.0};
    cfg_no_match.corporate_actions.payment_date_distributions[{kSpy, 1000 * kMin}] =
        CashDistribution{.instrument_id = kSpy, .ex_date_ts_ns = 999 * kMin,
                         .pay_date_ts_ns = 1000 * kMin, .amount_per_share_usd = 1.0};
    const BacktestEngine engine_no_match(reg, resolver, risk, cfg_no_match);
    const BacktestResult r_no_match = engine_no_match.run(bars, strat_no_match);

    assert_results_identical(r_default, r_explicit);
    assert_results_identical(r_default, r_no_match);
}

// ---------------------------------------------------------------------------
// 2. a 2-for-1 forward split rebases units / avg_entry_price at the exact bar
// ---------------------------------------------------------------------------

void test_split_rebases_units_and_avg_entry_zero_pnl_effect() {
    const auto reg = etf_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;

    std::vector<MarketBar> bars;
    for (int i = 1; i <= 5; ++i) bars.push_back(bar(i, kSpy, 20.0, 20.0));  // flat prices

    // decide 100 ONCE (at the bar1 decision) -> fills at bar2's open (20.0);
    // no further decisions are ever fired, so nothing re-affirms a pre-split
    // share count and no corrective trade can be queued after the split
    // (effective at bar3) rebases the held units out to bar5.
    const DecideAt strat_control({{0, 100.0}});
    const DecideAt strat_split({{0, 100.0}});

    // control: identical bars/strategy through an EMPTY corporate_actions
    // config.
    const BacktestEngine engine_control(reg, resolver, risk, leave_open_config());
    const BacktestResult r_control = engine_control.run(bars, strat_control);

    // treatment: a 2-for-1 split effective EXACTLY at bar 3's timestamp.
    EngineConfig cfg_split = leave_open_config();
    const SplitAction split{.instrument_id = kSpy, .effective_ts_ns = 3 * kMin, .ratio = 2.0};
    cfg_split.corporate_actions.splits[{kSpy, 3 * kMin}] = split;
    const BacktestEngine engine_split(reg, resolver, risk, cfg_split);
    const BacktestResult r_split = engine_split.run(bars, strat_split);

    // control: 100 units @ 20.0 entry, notional 2000.
    CHECK(r_control.final_positions.size() == 1);
    CHECK(r_control.final_positions[0].units == 100);
    CHECK_CLOSE(r_control.final_positions[0].avg_entry_price, 20.0, 1e-9);

    // treatment: 200 units @ 10.0 entry -- notional invariant (2000).
    CHECK(r_split.final_positions.size() == 1);
    CHECK(r_split.final_positions[0].units == 200);
    CHECK_CLOSE(r_split.final_positions[0].avg_entry_price, 10.0, 1e-9);
    CHECK_CLOSE(static_cast<double>(r_split.final_positions[0].units) *
                    r_split.final_positions[0].avg_entry_price,
                static_cast<double>(r_control.final_positions[0].units) *
                    r_control.final_positions[0].avg_entry_price,
                1e-9);

    // Zero economic effect at the split instant: realized PnL / costs / cash
    // are byte-identical to the no-split control.
    CHECK_CLOSE(r_split.gross_realized_pnl_usd, r_control.gross_realized_pnl_usd, 1e-12);
    CHECK_CLOSE(r_split.net_realized_pnl_usd, r_control.net_realized_pnl_usd, 1e-12);
    CHECK_CLOSE(r_split.costs_usd, r_control.costs_usd, 1e-12);
    CHECK_CLOSE(r_split.portfolio_at_end.cash_usd, r_control.portfolio_at_end.cash_usd, 1e-12);

    // Rebased identically on BOTH the engine's own ledger (final_positions,
    // above) and the PortfolioAccountant-derived snapshot -- proving the
    // engine's two ledgers (kept in sync via apply_fill / apply_split on every
    // corporate action, see engine.cpp) did not diverge.
    CHECK(r_split.portfolio_at_end.positions.size() == 1);
    CHECK(r_split.portfolio_at_end.positions[0].units == 200);
    CHECK_CLOSE(r_split.portfolio_at_end.positions[0].avg_entry_price, 10.0, 1e-9);
}

// ---------------------------------------------------------------------------
// 3. distribution ex-date entitlement (pre-fill) / pay-date settlement
// ---------------------------------------------------------------------------

void test_distribution_ex_date_uses_pre_fill_position_pay_date_moves_income() {
    const auto reg = etf_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;

    std::vector<MarketBar> bars;
    for (int i = 1; i <= 7; ++i) bars.push_back(bar(i, kSpy, 20.0, 20.0));

    // decide 100 at the bar1 decision (call idx 0, fills bar2); decide 500 at
    // the bar3 decision (call idx 2) -- this fills at bar4, EXACTLY the
    // ex-date bar below; no other decisions are ever fired.
    const DecideAt strat_full({{0, 100.0}, {2, 500.0}});

    const CashDistribution dist{.instrument_id = kSpy, .ex_date_ts_ns = 4 * kMin,
                                .pay_date_ts_ns = 6 * kMin, .amount_per_share_usd = 0.50};

    // ---- (a) truncate the run to end exactly at the ex-date bar: proves the
    //      receivable reflects the PRE-fill 100-unit position held coming INTO
    //      bar 4 -- NOT the 500 units that bar's own fill produces -- and that
    //      cash_usd / equity_usd are completely unaffected by the pending
    //      receivable.
    {
        const std::vector<MarketBar> bars4(bars.begin(), bars.begin() + 4);
        const DecideAt strat4_control({{0, 100.0}, {2, 500.0}});
        const DecideAt strat4_dist({{0, 100.0}, {2, 500.0}});

        const BacktestEngine engine_ctrl(reg, resolver, risk, leave_open_config());
        const BacktestResult r_ctrl = engine_ctrl.run(bars4, strat4_control);

        EngineConfig cfg_dist = leave_open_config();
        cfg_dist.corporate_actions.ex_date_distributions[{kSpy, dist.ex_date_ts_ns}] = dist;
        const BacktestEngine engine_dist(reg, resolver, risk, cfg_dist);
        const BacktestResult r_dist = engine_dist.run(bars4, strat4_dist);

        // The 100 -> 500 fill at bar 4 happened identically in BOTH runs.
        CHECK(r_ctrl.final_positions.size() == 1);
        CHECK(r_ctrl.final_positions[0].units == 500);
        CHECK(r_dist.final_positions.size() == 1);
        CHECK(r_dist.final_positions[0].units == 500);

        // Entitlement is the PRE-fill 100 units (100 * 0.50 = 50.0), never the
        // post-fill 500 (which would wrongly give 250.0).
        CHECK_CLOSE(r_dist.portfolio_at_end.distribution_receivable_usd, 50.0, 1e-9);
        CHECK_CLOSE(r_dist.portfolio_at_end.distribution_income_usd, 0.0, 1e-9);  // not paid yet

        // cash_usd / equity_usd / realized PnL are untouched by the pending
        // receivable.
        CHECK_CLOSE(r_dist.portfolio_at_end.cash_usd, r_ctrl.portfolio_at_end.cash_usd, 1e-9);
        CHECK_CLOSE(r_dist.portfolio_at_end.equity_usd, r_ctrl.portfolio_at_end.equity_usd, 1e-9);
        CHECK_CLOSE(r_dist.net_realized_pnl_usd, r_ctrl.net_realized_pnl_usd, 1e-12);
        CHECK_CLOSE(r_dist.costs_usd, r_ctrl.costs_usd, 1e-12);
    }

    // ---- (b) the full 7-bar run: the pay-date leg (bar 6) moves the SAME
    //      pre-fill-snapshotted 50.0 receivable into distribution_income_usd;
    //      cash / equity / realized PnL / costs stay byte-identical to a
    //      no-distribution control throughout.
    {
        const DecideAt strat_ctrl({{0, 100.0}, {2, 500.0}});
        const BacktestEngine engine_ctrl(reg, resolver, risk, leave_open_config());
        const BacktestResult r_ctrl = engine_ctrl.run(bars, strat_ctrl);

        EngineConfig cfg_dist = leave_open_config();
        cfg_dist.corporate_actions.ex_date_distributions[{kSpy, dist.ex_date_ts_ns}] = dist;
        cfg_dist.corporate_actions.payment_date_distributions[{kSpy, dist.pay_date_ts_ns}] = dist;
        const BacktestEngine engine_dist(reg, resolver, risk, cfg_dist);
        const BacktestResult r_dist = engine_dist.run(bars, strat_full);

        CHECK_CLOSE(r_dist.portfolio_at_end.distribution_income_usd, 50.0, 1e-9);
        CHECK_CLOSE(r_dist.portfolio_at_end.distribution_receivable_usd, 0.0, 1e-9);

        CHECK_CLOSE(r_dist.portfolio_at_end.cash_usd, r_ctrl.portfolio_at_end.cash_usd, 1e-9);
        CHECK_CLOSE(r_dist.portfolio_at_end.equity_usd, r_ctrl.portfolio_at_end.equity_usd, 1e-9);
        CHECK_CLOSE(r_dist.net_realized_pnl_usd, r_ctrl.net_realized_pnl_usd, 1e-12);
        CHECK_CLOSE(r_dist.costs_usd, r_ctrl.costs_usd, 1e-12);
        CHECK_CLOSE(r_dist.gross_realized_pnl_usd, r_ctrl.gross_realized_pnl_usd, 1e-12);
    }
}

}  // namespace

int main() {
    test_empty_config_is_byte_identical_to_default();
    test_split_rebases_units_and_avg_entry_zero_pnl_effect();
    test_distribution_ex_date_uses_pre_fill_position_pay_date_moves_income();
    return quant::test::summary("quant_corporate_actions_engine_tests");
}
