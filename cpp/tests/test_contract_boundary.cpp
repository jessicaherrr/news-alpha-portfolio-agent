// Phase 02.5 -- futures data / execution boundary contract.
//
// Proves the guarantees in docs/BOUNDARY_CONTRACT.md:
//   1. two different raw contracts can live in one continuous history
//   2. contract lookup by instrument_id and by timestamp
//   3. a roll transition is detectable and both sides resolve
//   4. a fill request without a valid raw contract is rejected
//   5. adjusted / continuous prices cannot enter the execution path
//   6. strategy code cannot read a bar at or after the decision bar
//   7. the contracts CSV boundary round-trips

#include "quant_core/bar_history.hpp"
#include "quant_core/contract.hpp"
#include "quant_core/contract_io.hpp"
#include "quant_core/contract_registry.hpp"
#include "quant_core/events.hpp"
#include "quant_core/fill.hpp"
#include "quant_core/momentum_strategy.hpp"
#include "quant_core/strategy.hpp"
#include "quant_core/strategy_context.hpp"
#include "quant_core/types.hpp"

#include "test_support.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace quant;

// Two overlapping NQ contracts: NQZ5 (front, then rolled out) and NQH6.
constexpr std::int64_t kNqZ5Activation = 1'600'000'000'000'000'000;
constexpr std::int64_t kNqZ5Expiration = 1'700'000'000'000'000'000;
constexpr std::int64_t kNqH6Activation = 1'690'000'000'000'000'000;  // overlaps NQZ5
constexpr std::int64_t kNqH6Expiration = 1'790'000'000'000'000'000;
constexpr std::int64_t kRollTs         = 1'695'000'000'000'000'000;  // inside the overlap

ContractSpec nq_z5() {
    return ContractSpec{
        .instrument_id = 101,
        .raw_symbol = "NQZ5",
        .root_symbol = "NQ",
        .exchange = "XCME",
        .tick_size = 0.25,
        .multiplier = 20.0,
        .activation_ns = kNqZ5Activation,
        .expiration_ns = kNqZ5Expiration,
    };
}

ContractSpec nq_h6() {
    return ContractSpec{
        .instrument_id = 102,
        .raw_symbol = "NQH6",
        .root_symbol = "NQ",
        .exchange = "XCME",
        .tick_size = 0.25,
        .multiplier = 20.0,
        .activation_ns = kNqH6Activation,
        .expiration_ns = kNqH6Expiration,
    };
}

ContractRegistry two_contract_registry() {
    ContractRegistry reg;
    reg.add(nq_z5());
    reg.add(nq_h6());
    return reg;
}

// A short continuous history that rolls NQZ5 -> NQH6 at kRollTs.
std::vector<MarketBar> rolling_history() {
    std::vector<MarketBar> bars;
    for (int k = 0; k < 6; ++k) {
        const std::int64_t ts = kRollTs - (6 - k) * 60'000'000'000LL;  // pre-roll minutes
        bars.push_back(MarketBar{.ts_event_ns = ts, .instrument_id = 101,
                                 .open = 20000.0 + k, .high = 20005.0 + k,
                                 .low = 19995.0 + k, .close = 20000.0 + k, .volume = 100});
    }
    for (int k = 0; k < 6; ++k) {
        const std::int64_t ts = kRollTs + k * 60'000'000'000LL;  // roll bar + after
        bars.push_back(MarketBar{.ts_event_ns = ts, .instrument_id = 102,
                                 .open = 20100.0 + k, .high = 20105.0 + k,
                                 .low = 20095.0 + k, .close = 20100.0 + k, .volume = 100});
    }
    return bars;
}

// ---- 1. two raw contracts in one continuous history -----------------------

void test_two_contracts_one_history() {
    const auto reg = two_contract_registry();
    const auto bars = rolling_history();

    int seen_101 = 0, seen_102 = 0;
    for (const auto& bar : bars) {
        const ContractSpec* spec = reg.find_by_instrument_id(bar.instrument_id);
        CHECK(spec != nullptr);
        CHECK(spec->is_live_at(bar.ts_event_ns));
        if (bar.instrument_id == 101) ++seen_101;
        if (bar.instrument_id == 102) ++seen_102;
    }
    CHECK(seen_101 == 6);
    CHECK(seen_102 == 6);
    CHECK(reg.by_instrument_id(101).raw_symbol != reg.by_instrument_id(102).raw_symbol);
}

// ---- 2. lookup by instrument_id and by timestamp --------------------------

void test_contract_lookup() {
    const auto reg = two_contract_registry();

    CHECK(reg.by_instrument_id(101).raw_symbol == "NQZ5");
    CHECK(reg.find_by_instrument_id(999) == nullptr);
    CHECK_THROWS_AS(reg.by_instrument_id(999), std::out_of_range);
    CHECK(reg.find_by_raw_symbol("NQH6") != nullptr);
    CHECK(reg.find_by_raw_symbol("NQH6")->instrument_id == 102);
    CHECK(reg.find_by_raw_symbol("NOPE") == nullptr);

    // latest_live_contract is a METADATA query (newest-activated contract of a
    // root live at ts), NOT the active-contract selector -- that follows market
    // state (see test_domain_model.cpp). Here it is exercised only for coverage.

    // Before NQH6 activates -> only NQZ5 is live.
    const ContractSpec* early = reg.latest_live_contract("NQ", kNqH6Activation - 1);
    CHECK(early != nullptr && early->instrument_id == 101);

    // After NQZ5 expires -> only NQH6.
    const ContractSpec* late = reg.latest_live_contract("NQ", kNqZ5Expiration + 1);
    CHECK(late != nullptr && late->instrument_id == 102);

    // In the overlap -> the later-activated contract is returned (heuristic only).
    const ContractSpec* overlap = reg.latest_live_contract("NQ", kRollTs);
    CHECK(overlap != nullptr && overlap->instrument_id == 102);

    // Unknown root, and a timestamp before anything is live.
    CHECK(reg.latest_live_contract("ES", kRollTs) == nullptr);
    CHECK(reg.latest_live_contract("NQ", kNqZ5Activation - 1) == nullptr);
}

// ---- 3. roll transition --------------------------------------------------

void test_roll_transition() {
    const auto reg = two_contract_registry();
    const auto bars = rolling_history();

    int boundaries = 0;
    std::size_t roll_index = 0;
    for (std::size_t i = 1; i < bars.size(); ++i) {
        if (bars[i].instrument_id != bars[i - 1].instrument_id) {
            ++boundaries;
            roll_index = i;
        }
    }
    CHECK(boundaries == 1);
    CHECK(bars[roll_index - 1].instrument_id == 101);
    CHECK(bars[roll_index].instrument_id == 102);

    // The additive roll gap is well defined and finite.
    const double gap = bars[roll_index].open - bars[roll_index - 1].close;
    CHECK(std::isfinite(gap));

    // Both sides of the roll resolve to real, live contracts and produce fills
    // against the correct raw contract.
    FillRequest pre{};
    pre.ts_fill_ns = bars[roll_index - 1].ts_event_ns;
    pre.instrument_id = bars[roll_index - 1].instrument_id;
    pre.quantity = 1;
    pre.price = bars[roll_index - 1].close;
    const Fill pre_fill = make_fill(reg, pre);
    CHECK(pre_fill.raw_symbol == "NQZ5");

    FillRequest post{};
    post.ts_fill_ns = bars[roll_index].ts_event_ns;
    post.instrument_id = bars[roll_index].instrument_id;
    post.quantity = 1;
    post.price = bars[roll_index].open;
    const Fill post_fill = make_fill(reg, post);
    CHECK(post_fill.raw_symbol == "NQH6");
}

// ---- 4. fill without a valid raw contract is rejected --------------------

void test_fill_requires_valid_contract() {
    const auto reg = two_contract_registry();

    auto good = []() {
        FillRequest r{};
        r.ts_fill_ns = kRollTs;
        r.instrument_id = 101;
        r.quantity = 1;
        r.price = 20000.10;  // not tick aligned on purpose
        return r;
    };

    // Baseline: a valid request resolves and aligns to the tick grid.
    const Fill f = make_fill(reg, good());
    CHECK(f.instrument_id == 101);
    CHECK(f.raw_symbol == "NQZ5");
    CHECK_CLOSE(f.tick_size, 0.25, 1e-12);
    CHECK_CLOSE(f.multiplier, 20.0, 1e-12);
    CHECK_CLOSE(f.fill_price, 20000.00, 1e-9);  // 20000.10 -> nearest 0.25

    {  // instrument_id 0
        FillRequest r = good();
        r.instrument_id = 0;
        CHECK_THROWS_AS(make_fill(reg, r), FillResolutionError);
    }
    {  // unknown instrument_id
        FillRequest r = good();
        r.instrument_id = 999;
        CHECK_THROWS_AS(make_fill(reg, r), FillResolutionError);
    }
    {  // timestamp outside the contract's tradable window
        FillRequest r = good();
        r.ts_fill_ns = kNqZ5Expiration + 1;
        CHECK_THROWS_AS(make_fill(reg, r), FillResolutionError);
    }
    {  // non-positive quantity
        FillRequest r = good();
        r.quantity = 0;
        CHECK_THROWS_AS(make_fill(reg, r), FillResolutionError);
    }
    {  // non-finite price
        FillRequest r = good();
        r.price = std::numeric_limits<double>::quiet_NaN();
        CHECK_THROWS_AS(make_fill(reg, r), FillResolutionError);
        r.price = std::numeric_limits<double>::infinity();
        CHECK_THROWS_AS(make_fill(reg, r), FillResolutionError);
    }
    {  // un-normalized vendor fixed-point price (magnitude tripwire)
        FillRequest r = good();
        r.price = 2.0e13;
        CHECK_THROWS_AS(make_fill(reg, r), FillResolutionError);
        r.price = -2.0e13;
        CHECK_THROWS_AS(make_fill(reg, r), FillResolutionError);
    }
    {  // a NORMALIZED signed price is valid -- zero and negative both fill and
       //   align to the tick grid by the same round-half-away-from-zero rule.
       //   (NQZ5 tick 0.25 here; the semantics, not the product, are the point.)
        FillRequest r = good();
        r.price = 0.0;
        const Fill z = make_fill(reg, r);
        CHECK_CLOSE(z.fill_price, 0.0, 1e-9);
        r.price = -20.10;                       // -> nearest 0.25 == -20.00
        const Fill n = make_fill(reg, r);
        CHECK_CLOSE(n.fill_price, -20.00, 1e-9);
        r.price = -20.13;                       // -> nearest 0.25 == -20.25
        CHECK_CLOSE(make_fill(reg, r).fill_price, -20.25, 1e-9);
    }
    {  // caller's expected symbol disagrees with the registry
        FillRequest r = good();
        r.expected_raw_symbol = "NQH6";
        CHECK_THROWS_AS(make_fill(reg, r), FillResolutionError);
    }
}

// ---- 5. adjusted / continuous prices cannot reach execution -------------

void test_no_adjusted_price_in_execution() {
    const auto reg = two_contract_registry();

    FillRequest base{};
    base.ts_fill_ns = kRollTs;
    base.instrument_id = 101;
    base.quantity = 1;
    base.price = 20000.00;

    {  // back-adjusted price domain is rejected even with a valid contract
        FillRequest r = base;
        r.price_domain = PriceDomain::BackAdjusted;
        CHECK_THROWS_AS(make_fill(reg, r), FillResolutionError);
    }
    {  // unadjusted-continuous domain is also rejected
        FillRequest r = base;
        r.price_domain = PriceDomain::RawContinuous;
        CHECK_THROWS_AS(make_fill(reg, r), FillResolutionError);
    }

    // Symbol classification.
    CHECK(is_continuous_symbol("NQ.v.0"));
    CHECK(is_continuous_symbol("ES.c.1"));
    CHECK(is_continuous_symbol("CL.n.0"));
    CHECK(!is_continuous_symbol("NQZ5"));
    CHECK(!is_continuous_symbol("NQH6"));

    CHECK(is_tradable_contract_symbol("NQZ5"));
    CHECK(!is_tradable_contract_symbol("NQ.v.0"));
    CHECK(!is_tradable_contract_symbol("NQ.FUT"));
    CHECK(!is_tradable_contract_symbol(""));

    // The registry refuses to hold a continuous symbol as a tradable contract.
    ContractSpec bad = nq_z5();
    bad.instrument_id = 500;
    bad.raw_symbol = "NQ.v.0";
    ContractRegistry local;
    CHECK_THROWS_AS(local.add(bad), std::invalid_argument);
}

// ---- 6. strategy cannot read a future bar ------------------------------

struct PeekingStrategy final : Strategy {
    std::optional<Signal> decide(const StrategyContext& ctx) const override {
        const BarHistoryView& h = ctx.history();
        Signal s;
        s.ts_decision_ns = ctx.decision_ts_ns();
        s.root_symbol = "NQ";
        // Deliberately reaches one past the decision bar -> std::out_of_range.
        s.target_units = h.at(h.size()).close > 0.0 ? 1.0 : 0.0;
        return s;
    }
};

// Build a minimal decision context for a bar index into `bars`.
StrategyContext make_ctx(const std::vector<MarketBar>& bars, std::size_t decision_index,
                         const BarHistoryView& view, MarketEvent& scratch) {
    const MarketBar& b = bars[decision_index];
    scratch.type = MarketEventType::Bar;
    scratch.ts_event_ns = b.ts_event_ns;
    scratch.seq = static_cast<SeqNum>(decision_index);
    scratch.instrument_id = b.instrument_id;
    scratch.open = b.open;
    scratch.high = b.high;
    scratch.low = b.low;
    scratch.close = b.close;
    scratch.volume = b.volume;
    return StrategyContext(b.ts_event_ns, scratch, view, "NQ", nullptr);
}

void test_strategy_cannot_see_future() {
    std::vector<MarketBar> bars;
    for (int k = 0; k < 10; ++k) {
        bars.push_back(MarketBar{.ts_event_ns = 1'000 + k, .instrument_id = 101,
                                 .open = 100.0 + k, .high = 100.5 + k,
                                 .low = 99.5 + k, .close = 100.0 + k, .volume = 1});
    }

    // Decision at index 3 -> only bars [0..3] visible.
    const BarHistoryView view(bars.data(), 4);
    CHECK(view.size() == 4);
    CHECK(view.latest().ts_event_ns == bars[3].ts_event_ns);
    CHECK(view.at(3).ts_event_ns == bars[3].ts_event_ns);
    CHECK(view.ago(0).ts_event_ns == bars[3].ts_event_ns);
    CHECK(view.ago(3).ts_event_ns == bars[0].ts_event_ns);

    CHECK_THROWS_AS(view.at(4), std::out_of_range);       // the very next bar
    CHECK_THROWS_AS(view.at(9), std::out_of_range);       // a far-future bar
    CHECK_THROWS_AS(view.ago(4), std::out_of_range);      // before history start

    const BarHistoryView empty(bars.data(), 0);
    CHECK_THROWS_AS(empty.latest(), std::out_of_range);

    // A strategy that tries to peek past the decision bar gets an exception.
    MarketEvent scratch;
    const StrategyContext ctx = make_ctx(bars, 3, view, scratch);
    const PeekingStrategy peeker;
    CHECK_THROWS_AS(peeker.decide(ctx), std::out_of_range);

    // A well-behaved strategy is unaffected and returns a root-level Signal.
    const TimeSeriesMomentum momentum(2, 0.0);
    const std::optional<Signal> maybe = momentum.decide(ctx);
    CHECK(maybe.has_value());
    const Signal sig = *maybe;
    CHECK(sig.root_symbol == "NQ");
    CHECK(sig.ts_decision_ns == bars[3].ts_event_ns);
    ::quant::test::pass();
}

// ---- 7. contracts CSV boundary round-trips -----------------------------

void test_contracts_csv_roundtrip() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "quant_boundary_contracts_test.csv";
    {
        std::ofstream out(path);
        out << "instrument_id,raw_symbol,root_symbol,exchange,tick_size,multiplier,"
               "activation_ns,expiration_ns,first_notice_ns,last_trade_ns\n";
        out << "101,NQZ5,NQ,XCME,0.25,20,"
            << kNqZ5Activation << "," << kNqZ5Expiration << ",,\n";
        out << "102,NQH6,NQ,XCME,0.25,20,"
            << kNqH6Activation << "," << kNqH6Expiration << ",," << (kNqH6Expiration - 1) << "\n";
    }

    const auto reg = load_contract_registry(path.string());
    CHECK(reg.size() == 2);
    CHECK(reg.by_instrument_id(101).raw_symbol == "NQZ5");
    CHECK(reg.by_instrument_id(102).last_trade_ns.has_value());
    CHECK(reg.by_instrument_id(102).last_trade_ns.value() == kNqH6Expiration - 1);
    CHECK(!reg.by_instrument_id(101).last_trade_ns.has_value());

    // A malformed header is rejected.
    const std::filesystem::path bad = std::filesystem::temp_directory_path() /
                                      "quant_boundary_contracts_bad.csv";
    {
        std::ofstream out(bad);
        out << "instrument_id,raw_symbol\n101,NQZ5\n";
    }
    CHECK_THROWS(load_contract_registry(bad.string()));

    std::filesystem::remove(path);
    std::filesystem::remove(bad);
}

}  // namespace

int main() {
    test_two_contracts_one_history();
    test_contract_lookup();
    test_roll_transition();
    test_fill_requires_valid_contract();
    test_no_adjusted_price_in_execution();
    test_strategy_cannot_see_future();
    test_contracts_csv_roundtrip();
    return quant::test::summary("quant_contract_boundary_tests");
}
