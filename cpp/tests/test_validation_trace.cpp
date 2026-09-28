// Phase 13 -- additive end-of-session-day equity trace (BacktestResult::daily_equity).
//
// The trace is the ONLY new export Phase 13 needs from the frozen C++ core. It
// changes no accounting formula: each point is the equity value
// PortfolioAccountant::snapshot() already computes, sampled at the last engine
// event of each session-day bucket. These checks pin:
//
//   A. one point per distinct session day; session_day_index strictly increasing
//   B. the final point equals portfolio_at_end (equity / net-realized / unrealized)
//   C. differencing the trace reproduces total PnL (final equity - starting cap)
//   D. a day with no economic PnL still emits a point (zero daily PnL, not skipped)
//   E. bars_cumulative accounts for every bar, monotonically (no silent gap)
//   F. deterministic replay: identical bytes across two runs

#include "quant_core/contract.hpp"
#include "quant_core/contract_registry.hpp"
#include "quant_core/contract_selector.hpp"
#include "quant_core/engine.hpp"
#include "quant_core/risk_manager.hpp"
#include "quant_core/scheduled_target_strategy.hpp"

#include "test_support.hpp"

#include <cstdint>
#include <vector>

namespace {

using namespace quant;

constexpr std::int64_t  kMin      = 60'000'000'000LL;
constexpr std::int64_t  kNsPerDay = 86'400'000'000'000LL;
constexpr std::uint32_t kNqId     = 950001;
const std::string kFp = "stratdsl1:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

ContractRegistry nq_registry() {
    ContractRegistry r;
    r.add(ContractSpec{.instrument_id = kNqId, .raw_symbol = "NQU6", .root_symbol = "NQ",
                       .exchange = "XCME", .tick_size = 0.25, .multiplier = 20.0,
                       .activation_ns = 1, .expiration_ns = 10'000LL * kNsPerDay});
    return r;
}

MarketBar bar(std::int64_t ts, double px) {
    return MarketBar{.ts_event_ns = ts, .instrument_id = kNqId,
                     .open = px, .high = px + 1.0, .low = px - 1.0, .close = px, .volume = 100};
}

MarketBar bar_oc(std::int64_t ts, double open, double close) {
    return MarketBar{.ts_event_ns = ts, .instrument_id = kNqId, .open = open,
                     .high = std::max(open, close) + 1.0, .low = std::min(open, close) - 1.0,
                     .close = close, .volume = 100};
}

// Three UTC days, 4 bars each (minute cadence early in the day). Prices ramp on
// day 0 up to 108, hold flat at 108 all of day 1 (incl. the overnight gap: day 0
// closes at 108 too), then ramp on day 2 -- so day 1 has genuinely zero economic
// PnL for a position held across it.
std::vector<MarketBar> three_day_bars() {
    std::vector<MarketBar> b;
    const double day_px[3] = {102.0, 108.0, 108.0};
    const double day_step[3] = {2.0, 0.0, 3.0};
    for (int d = 0; d < 3; ++d) {
        for (int i = 0; i < 4; ++i) {
            b.push_back(bar(d * kNsPerDay + (i + 1) * kMin, day_px[d] + i * day_step[d]));
        }
    }
    return b;
}

bool same_point(const DailyEquityPoint& a, const DailyEquityPoint& b) {
    return a.session_day_index == b.session_day_index && a.ts_ns == b.ts_ns &&
           a.equity_usd == b.equity_usd && a.net_realized_pnl_usd == b.net_realized_pnl_usd &&
           a.unrealized_pnl_usd == b.unrealized_pnl_usd && a.costs_usd == b.costs_usd &&
           a.fills_cumulative == b.fills_cumulative && a.bars_cumulative == b.bars_cumulative;
}

BacktestResult run_once(const std::vector<MarketBar>& bars,
                        const std::vector<ScheduledTarget>& rows,
                        std::vector<std::int64_t> vday = {}) {
    const auto reg = nq_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    EngineConfig cfg;
    cfg.validation_day_boundaries_ns = std::move(vday);
    const BacktestEngine engine(reg, resolver, risk, cfg);
    const ScheduledTargetStrategy strat(rows, kFp);
    return engine.run(bars, strat);
}

// Bars every 6h for 3 UTC days (12 bars). One CME trading day (17:00->17:00
// local) does NOT line up with a UTC date, so a boundary list that groups the
// bars in threes lands differently from the UTC-midnight buckets.
std::vector<MarketBar> six_hourly_bars(int n = 12, double start = 100.0) {
    std::vector<MarketBar> b;
    const std::int64_t six_h = 360 * kMin;
    for (int i = 0; i < n; ++i) b.push_back(bar(kMin + i * six_h, start + i * 0.5));
    return b;
}

void test_one_point_per_day_increasing() {
    const auto bars = three_day_bars();
    // go long on the 2nd bar of day 0; never decide again (held to EOT).
    const std::vector<ScheduledTarget> rows{
        {.root_symbol = "NQ", .ts_event_ns = 0 * kNsPerDay + 2 * kMin, .target_units = 1}};
    const auto r = run_once(bars, rows);

    CHECK(r.daily_equity.size() == 3);
    CHECK(r.daily_equity[0].session_day_index == 0);
    CHECK(r.daily_equity[1].session_day_index == 1);
    CHECK(r.daily_equity[2].session_day_index == 2);
    for (std::size_t i = 1; i < r.daily_equity.size(); ++i) {
        CHECK(r.daily_equity[i].session_day_index > r.daily_equity[i - 1].session_day_index);
        CHECK(r.daily_equity[i].ts_ns > r.daily_equity[i - 1].ts_ns);
    }
}

void test_final_point_matches_portfolio_at_end() {
    const auto bars = three_day_bars();
    const std::vector<ScheduledTarget> rows{
        {.root_symbol = "NQ", .ts_event_ns = 0 * kNsPerDay + 2 * kMin, .target_units = 1}};
    const auto r = run_once(bars, rows);
    const auto& last = r.daily_equity.back();
    CHECK_CLOSE(last.equity_usd, r.portfolio_at_end.equity_usd, 1e-9);
    CHECK_CLOSE(last.net_realized_pnl_usd, r.net_realized_pnl_usd, 1e-9);
    CHECK_CLOSE(last.unrealized_pnl_usd, r.unrealized_pnl_usd_at_end, 1e-9);
    CHECK(last.fills_cumulative == r.fills_generated);
    CHECK(last.bars_cumulative == r.bars_processed);
}

void test_differencing_reproduces_total_pnl() {
    const auto bars = three_day_bars();
    const std::vector<ScheduledTarget> rows{
        {.root_symbol = "NQ", .ts_event_ns = 0 * kNsPerDay + 2 * kMin, .target_units = 1}};
    const auto r = run_once(bars, rows);
    const double start_cap = r.portfolio_at_end.starting_capital_usd;
    double summed = r.daily_equity.front().equity_usd - start_cap;
    for (std::size_t i = 1; i < r.daily_equity.size(); ++i) {
        summed += r.daily_equity[i].equity_usd - r.daily_equity[i - 1].equity_usd;
    }
    CHECK_CLOSE(summed, r.portfolio_at_end.equity_usd - start_cap, 1e-6);
}

void test_flat_day_still_emits_a_point() {
    const auto bars = three_day_bars();
    // Long the whole time. Day 1 is price-flat -> zero economic PnL that day, but
    // it must still produce a trace point (a missing day never silently -> 0).
    const std::vector<ScheduledTarget> rows{
        {.root_symbol = "NQ", .ts_event_ns = 0 * kNsPerDay + 1 * kMin, .target_units = 1}};
    const auto r = run_once(bars, rows);
    CHECK(r.daily_equity.size() == 3);
    const double d1_pnl = r.daily_equity[1].equity_usd - r.daily_equity[0].equity_usd;
    CHECK_CLOSE(d1_pnl, 0.0, 1e-9);            // real observed zero, not a skipped day
    CHECK(r.daily_equity[1].session_day_index == 1);
}

void test_bars_cumulative_accounts_for_every_bar() {
    const auto bars = three_day_bars();
    const std::vector<ScheduledTarget> rows{
        {.root_symbol = "NQ", .ts_event_ns = 0 * kNsPerDay + 2 * kMin, .target_units = 1}};
    const auto r = run_once(bars, rows);
    std::size_t prev = 0;
    for (const auto& p : r.daily_equity) {
        CHECK(p.bars_cumulative >= prev);      // monotonic
        prev = p.bars_cumulative;
    }
    CHECK(r.daily_equity.back().bars_cumulative == bars.size());  // no bar unaccounted
    CHECK(r.daily_equity[0].bars_cumulative == 4);
    CHECK(r.daily_equity[1].bars_cumulative == 8);
}

void test_deterministic_replay() {
    const auto bars = three_day_bars();
    const std::vector<ScheduledTarget> rows{
        {.root_symbol = "NQ", .ts_event_ns = 0 * kNsPerDay + 2 * kMin, .target_units = 1}};
    const auto r1 = run_once(bars, rows);
    const auto r2 = run_once(bars, rows);
    CHECK(r1.daily_equity.size() == r2.daily_equity.size());
    for (std::size_t i = 0; i < r1.daily_equity.size(); ++i) {
        CHECK(same_point(r1.daily_equity[i], r2.daily_equity[i]));
    }
}

// ---- Phase 13.1: caller-supplied validation-day boundaries -------------------
void test_boundary_path_one_point_per_boundary() {
    const auto bars = six_hourly_bars(12);
    const std::int64_t six_h = 360 * kMin;
    // 4 validation days, 3 bars each -> boundary at bars 2, 5, 8, 11
    const std::vector<std::int64_t> vday{kMin + 2 * six_h, kMin + 5 * six_h,
                                         kMin + 8 * six_h, kMin + 11 * six_h};
    const std::vector<ScheduledTarget> rows{
        {.root_symbol = "NQ", .ts_event_ns = kMin + 1 * six_h, .target_units = 1}};
    const auto r = run_once(bars, rows, vday);

    CHECK(r.daily_equity.size() == 4);
    for (std::size_t i = 0; i < 4; ++i) {
        CHECK(r.daily_equity[i].session_day_index == static_cast<std::int64_t>(i));  // boundary ordinal
        CHECK(r.daily_equity[i].ts_ns == vday[i]);
    }
    // and the final point is the post-EOT equity
    CHECK_CLOSE(r.daily_equity.back().equity_usd, r.portfolio_at_end.equity_usd, 1e-9);
    CHECK_CLOSE(r.daily_equity.back().net_realized_pnl_usd, r.net_realized_pnl_usd, 1e-9);
}

void test_boundary_path_differs_from_utc_bucketing() {
    const auto bars = six_hourly_bars(12);
    const std::int64_t six_h = 360 * kMin;
    const std::vector<std::int64_t> vday{kMin + 2 * six_h, kMin + 5 * six_h,
                                         kMin + 8 * six_h, kMin + 11 * six_h};
    const std::vector<ScheduledTarget> rows{
        {.root_symbol = "NQ", .ts_event_ns = kMin + 1 * six_h, .target_units = 1}};

    const auto with_bounds = run_once(bars, rows, vday);
    const auto utc_bucket  = run_once(bars, rows, {});

    // 12 bars over 3 UTC days -> 3 UTC buckets; the trading-day boundary list -> 4.
    CHECK(utc_bucket.daily_equity.size() == 3);
    CHECK(with_bounds.daily_equity.size() == 4);
    // both reproduce the SAME total PnL (differencing the official trace)
    const double start_cap = with_bounds.portfolio_at_end.starting_capital_usd;
    auto total = [&](const BacktestResult& x) {
        double s = x.daily_equity.front().equity_usd - start_cap;
        for (std::size_t i = 1; i < x.daily_equity.size(); ++i)
            s += x.daily_equity[i].equity_usd - x.daily_equity[i - 1].equity_usd;
        return s;
    };
    CHECK_CLOSE(total(with_bounds), total(utc_bucket), 1e-6);
    // but they allocate that PnL to a DIFFERENT number of day observations
    CHECK(with_bounds.daily_equity.size() != utc_bucket.daily_equity.size());
}

void test_boundary_spanning_utc_midnight_is_one_observation() {
    // 4 bars at 18h, 24h, 30h, 36h -> straddles a UTC midnight (24h). One
    // boundary covering all four -> ONE validation observation.
    const std::int64_t h = 60 * kMin;
    std::vector<MarketBar> bars{bar(18 * h, 100.0), bar(24 * h, 101.0),
                                bar(30 * h, 102.0), bar(36 * h, 103.0)};
    const std::vector<std::int64_t> vday{36 * h};
    const std::vector<ScheduledTarget> rows{
        {.root_symbol = "NQ", .ts_event_ns = 18 * h, .target_units = 1}};
    const auto r = run_once(bars, rows, vday);
    CHECK(r.daily_equity.size() == 1);
    CHECK(r.daily_equity[0].ts_ns == 36 * h);
    // the legacy UTC bucketing of the same bars would be TWO days
    const auto utc = run_once(bars, rows, {});
    CHECK(utc.daily_equity.size() == 2);
}

void test_boundary_path_deterministic_replay() {
    const auto bars = six_hourly_bars(12);
    const std::int64_t six_h = 360 * kMin;
    const std::vector<std::int64_t> vday{kMin + 2 * six_h, kMin + 5 * six_h,
                                         kMin + 8 * six_h, kMin + 11 * six_h};
    const std::vector<ScheduledTarget> rows{
        {.root_symbol = "NQ", .ts_event_ns = kMin + 1 * six_h, .target_units = 1}};
    const auto a = run_once(bars, rows, vday);
    const auto b = run_once(bars, rows, vday);
    CHECK(a.daily_equity.size() == b.daily_equity.size());
    for (std::size_t i = 0; i < a.daily_equity.size(); ++i)
        CHECK(same_point(a.daily_equity[i], b.daily_equity[i]));
}

// ---- Phase 13.2: boundary-event economic effect lands on the RIGHT day -------
void test_boundary_bar_fill_and_mark_land_on_that_trading_day() {
    // multiplier 20, commission 0 (EngineConfig default). Three "trading days",
    // 4 bars each; the boundary is each day's 4th bar.
    std::vector<MarketBar> b;
    // day 0: flat @ 100 then the boundary bar OPENS 100, CLOSES 120
    b.push_back(bar_oc(1 * kMin, 100, 100));
    b.push_back(bar_oc(2 * kMin, 100, 100));
    b.push_back(bar_oc(3 * kMin, 100, 100));           // decision here -> +1
    b.push_back(bar_oc(4 * kMin, 100, 120));           // boundary: entry fill @ open 100, mark @ close 120
    // day 1: flat @ 120 then the boundary bar OPENS 130 (exit fill), CLOSES 130
    for (int i = 1; i <= 3; ++i) b.push_back(bar_oc(kNsPerDay + i * kMin, 120, 120));
    b.push_back(bar_oc(kNsPerDay + 4 * kMin, 130, 130));  // decision at day1 bar3 -> 0 -> exit @ open 130
    // day 2: flat, no position
    for (int i = 1; i <= 4; ++i) b.push_back(bar_oc(2 * kNsPerDay + i * kMin, 130, 130));

    const std::vector<std::int64_t> vday{4 * kMin, kNsPerDay + 4 * kMin, 2 * kNsPerDay + 4 * kMin};
    const std::vector<ScheduledTarget> rows{
        {.root_symbol = "NQ", .ts_event_ns = 3 * kMin, .target_units = 1},              // -> fill day0 bar3
        {.root_symbol = "NQ", .ts_event_ns = kNsPerDay + 3 * kMin, .target_units = 0},  // -> fill day1 bar3
    };
    const auto r = run_once(b, rows, vday);
    CHECK(r.daily_equity.size() == 3);
    const double start = r.portfolio_at_end.starting_capital_usd;

    // DAY 0: the boundary-bar entry fill @ 100 + mark to the boundary close 120
    //        -> unrealized (120-100)*20*1 = +400, ALL on day 0.
    CHECK_CLOSE(r.daily_equity[0].unrealized_pnl_usd, 400.0, 1e-6);
    CHECK_CLOSE(r.daily_equity[0].net_realized_pnl_usd, 0.0, 1e-6);
    CHECK_CLOSE(r.daily_equity[0].equity_usd - start, 400.0, 1e-6);   // daily_pnl[0]

    // DAY 1: the boundary-bar exit fill @ 130 -> realized (130-100)*20 = +600,
    //        unrealized back to 0. daily_pnl[1] = EOD[1]-EOD[0] = 600 - 400 = +200.
    CHECK_CLOSE(r.daily_equity[1].net_realized_pnl_usd, 600.0, 1e-6);
    CHECK_CLOSE(r.daily_equity[1].unrealized_pnl_usd, 0.0, 1e-6);
    CHECK_CLOSE(r.daily_equity[1].equity_usd - r.daily_equity[0].equity_usd, 200.0, 1e-6);

    // DAY 2: no position, flat -> zero daily PnL. The day-1 economics did NOT
    //        leak onto day 2.
    CHECK_CLOSE(r.daily_equity[2].equity_usd - r.daily_equity[1].equity_usd, 0.0, 1e-6);

    // total still reconciles
    CHECK_CLOSE(r.daily_equity.back().equity_usd - start,
                r.portfolio_at_end.equity_usd - start, 1e-6);
    CHECK_CLOSE(r.daily_equity.back().net_realized_pnl_usd, r.net_realized_pnl_usd, 1e-6);
}

void test_no_events_no_trace() {
    const auto reg = nq_registry();
    const RegistryActiveContractResolver resolver(reg);
    const PassThroughRiskManager risk;
    const BacktestEngine engine(reg, resolver, risk, EngineConfig{});
    const ScheduledTargetStrategy strat(std::vector<ScheduledTarget>{}, kFp);
    const auto r = engine.run(std::vector<MarketBar>{}, strat);
    CHECK(r.daily_equity.empty());
}

}  // namespace

int main() {
    test_one_point_per_day_increasing();
    test_final_point_matches_portfolio_at_end();
    test_differencing_reproduces_total_pnl();
    test_flat_day_still_emits_a_point();
    test_bars_cumulative_accounts_for_every_bar();
    test_deterministic_replay();
    test_boundary_path_one_point_per_boundary();
    test_boundary_path_differs_from_utc_bucketing();
    test_boundary_spanning_utc_midnight_is_one_observation();
    test_boundary_path_deterministic_replay();
    test_boundary_bar_fill_and_mark_land_on_that_trading_day();
    test_no_events_no_trace();
    return quant::test::summary("test_validation_trace");
}
