// Phase 05 -- typed deterministic C++ domain / event model.
//
// Covers the event flow types Phases 06-08 will execute:
//   A. MarketEvent validation
//   B. Strategy no-lookahead via StrategyContext
//   C. Signal validation
//   D. deterministic contract selection (root + ts -> raw contract)
//   E. Order invariants
//   F. RiskDecision invariants + APPROVE / RESIZE / REJECT
//   G. Fill model (raw contract only; tick / multiplier from ContractSpec)
//   H. deterministic event ordering (ts_event_ns, seq)
//   I. traceability: Signal -> Order -> RiskDecision -> Fill id chain
//   J. covered by quant_contract_boundary_tests (frozen Phase 02.5 boundary)

#include "quant_core/bar_history.hpp"
#include "quant_core/contract.hpp"
#include "quant_core/contract_registry.hpp"
#include "quant_core/contract_selector.hpp"
#include "quant_core/domain_errors.hpp"
#include "quant_core/domain_model.hpp"
#include "quant_core/events.hpp"
#include "quant_core/fill.hpp"
#include "quant_core/ids.hpp"
#include "quant_core/risk_manager.hpp"
#include "quant_core/strategy.hpp"
#include "quant_core/strategy_context.hpp"
#include "quant_core/types.hpp"

#include "test_support.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

using namespace quant;

constexpr std::int64_t kNqZ5Activation = 1'600'000'000'000'000'000;
constexpr std::int64_t kNqZ5Expiration = 1'700'000'000'000'000'000;
constexpr std::int64_t kNqH6Activation = 1'690'000'000'000'000'000;
constexpr std::int64_t kNqH6Expiration = 1'790'000'000'000'000'000;
constexpr std::int64_t kRollTs         = 1'695'000'000'000'000'000;

ContractSpec nq_z5() {
    return ContractSpec{
        .instrument_id = 101, .raw_symbol = "NQZ5", .root_symbol = "NQ", .exchange = "XCME",
        .tick_size = 0.25, .multiplier = 20.0,
        .activation_ns = kNqZ5Activation, .expiration_ns = kNqZ5Expiration};
}
ContractSpec nq_h6() {
    return ContractSpec{
        .instrument_id = 102, .raw_symbol = "NQH6", .root_symbol = "NQ", .exchange = "XCME",
        .tick_size = 0.25, .multiplier = 20.0,
        .activation_ns = kNqH6Activation, .expiration_ns = kNqH6Expiration};
}
ContractRegistry two_contract_registry() {
    ContractRegistry reg;
    reg.add(nq_z5());
    reg.add(nq_h6());
    return reg;
}

MarketEvent good_bar_event() {
    MarketEvent ev;
    ev.type = MarketEventType::Bar;
    ev.ts_event_ns = kRollTs;
    ev.seq = 0;
    ev.instrument_id = 102;
    ev.open = 20000.0;
    ev.high = 20010.0;
    ev.low = 19990.0;
    ev.close = 20005.0;
    ev.volume = 100;
    return ev;
}

// ---- A. MarketEvent ---------------------------------------------------

void test_market_event_validation() {
    const auto reg = two_contract_registry();
    const MarketEvent ev = good_bar_event();

    validate_market_event(ev, reg);            // valid -> no throw
    ::quant::test::pass();
    CHECK(ev.ts_event_ns == kRollTs);          // timestamp preserved, authoritative

    {  // instrument_id 0
        MarketEvent e = ev; e.instrument_id = 0;
        CHECK_THROWS_AS(validate_market_event(e, reg), InvalidMarketEvent);
    }
    {  // instrument_id not in the registry
        MarketEvent e = ev; e.instrument_id = 999;
        CHECK_THROWS_AS(validate_market_event(e, reg), InvalidMarketEvent);
    }
    {  // non-positive timestamp
        MarketEvent e = ev; e.ts_event_ns = 0;
        CHECK_THROWS_AS(validate_market_event(e, reg), InvalidMarketEvent);
    }
    {  // un-normalized vendor fixed-point price leaks in (magnitude tripwire,
       //   either sign)
        MarketEvent e = ev; e.close = 2.0005e13;
        CHECK_THROWS_AS(validate_market_event(e, reg), InvalidMarketEvent);
        MarketEvent en = ev; en.low = -2.0005e13;
        CHECK_THROWS_AS(validate_market_event(en, reg), InvalidMarketEvent);
    }
    {  // NaN / infinity rejected
        MarketEvent e = ev; e.open = std::numeric_limits<double>::quiet_NaN();
        CHECK_THROWS_AS(validate_market_event(e, reg), InvalidMarketEvent);
        MarketEvent ei = ev; ei.high = std::numeric_limits<double>::infinity();
        CHECK_THROWS_AS(validate_market_event(ei, reg), InvalidMarketEvent);
    }
    {  // SIGNED OHLC is valid: a negative-price CL-style bar (open -10, high -5,
       //   low -25, close -20) satisfies low <= open/close <= high.
        MarketEvent e = ev;
        e.open = -10.0; e.high = -5.0; e.low = -25.0; e.close = -20.0;
        validate_market_event(e, reg);
        ::quant::test::pass();
        MarketEvent z = ev;  // a bar that straddles zero
        z.open = 5.0; z.high = 6.0; z.low = -6.0; z.close = -5.0;
        validate_market_event(z, reg);
        ::quant::test::pass();
    }
    {  // OHLC inconsistent (high below the open) -- still rejected, signed too
        MarketEvent e = ev; e.high = 19999.0;
        CHECK_THROWS_AS(validate_market_event(e, reg), InvalidMarketEvent);
        MarketEvent en = ev;
        en.open = -10.0; en.high = -25.0; en.low = -30.0; en.close = -20.0;  // high below open
        CHECK_THROWS_AS(validate_market_event(en, reg), InvalidMarketEvent);
    }
    {  // a domain error is catchable as the base type
        MarketEvent e = ev; e.instrument_id = 0;
        CHECK_THROWS_AS(validate_market_event(e, reg), DomainError);
    }
}

// ---- B. Strategy no-lookahead --------------------------------------

struct PeekingStrategy final : Strategy {
    std::optional<Signal> decide(const StrategyContext& ctx) const override {
        const BarHistoryView& h = ctx.history();
        Signal s;
        s.ts_decision_ns = ctx.decision_ts_ns();
        s.root_symbol = "NQ";
        s.target_units = h.at(h.size()).close;  // one past the decision bar
        return s;
    }
};

void test_strategy_context_no_lookahead() {
    std::vector<MarketBar> bars;
    for (int k = 0; k < 10; ++k) {
        bars.push_back(MarketBar{.ts_event_ns = 1'000 + k, .instrument_id = 101,
                                 .open = 100.0 + k, .high = 100.5 + k,
                                 .low = 99.5 + k, .close = 100.0 + k, .volume = 1});
    }
    const BarHistoryView view(bars.data(), 4);  // decision at index 3

    MarketEvent ev = good_bar_event();
    ev.ts_event_ns = bars[3].ts_event_ns;
    ev.instrument_id = 101;
    const StrategyContext ctx(bars[3].ts_event_ns, ev, view, "NQ", nullptr);

    CHECK(ctx.decision_ts_ns() == bars[3].ts_event_ns);
    CHECK(ctx.root_symbol() == "NQ");
    CHECK(ctx.active_contract() == nullptr);            // legacy path -> no snapshot
    CHECK(ctx.history().size() == 4);
    CHECK(ctx.history().latest().ts_event_ns == bars[3].ts_event_ns);

    // The context exposes no way to read a bar at or after the decision bar.
    CHECK_THROWS_AS(ctx.history().at(ctx.history().size()), std::out_of_range);
    CHECK_THROWS_AS(ctx.history().at(4), std::out_of_range);

    // A strategy that tries anyway gets an exception, not silent future data.
    const PeekingStrategy peeker;
    CHECK_THROWS_AS(peeker.decide(ctx), std::out_of_range);
}

// ---- C. Signal -----------------------------------------------------

Signal good_signal() {
    Signal s;
    s.signal_id = 1;
    s.ts_decision_ns = kRollTs;
    s.root_symbol = "NQ";
    s.target_units = 1.0;
    s.rationale_code = "tsmom_up";
    return s;
}

void test_signal_validation() {
    const Signal s = good_signal();
    validate_signal(s);
    ::quant::test::pass();
    CHECK(s.direction() == SignalDirection::Long);
    CHECK(good_signal().direction() == SignalDirection::Long);
    { Signal x = s; x.target_units = -2.0; CHECK(x.direction() == SignalDirection::Short); }
    { Signal x = s; x.target_units = 0.0;  CHECK(x.direction() == SignalDirection::Flat); }

    { Signal x = s; x.signal_id = 0;       CHECK_THROWS_AS(validate_signal(x), InvalidSignal); }
    { Signal x = s; x.ts_decision_ns = 0;  CHECK_THROWS_AS(validate_signal(x), InvalidSignal); }
    { Signal x = s; x.root_symbol = "";    CHECK_THROWS_AS(validate_signal(x), InvalidSignal); }
    { Signal x = s; x.root_symbol = "NQ.v.0"; CHECK_THROWS_AS(validate_signal(x), InvalidSignal); }
    { Signal x = s; x.root_symbol = "NQ.FUT";  CHECK_THROWS_AS(validate_signal(x), InvalidSignal); }
    { Signal x = s; x.target_units = std::nan(""); CHECK_THROWS_AS(validate_signal(x), InvalidSignal); }

    // MVP invariant B: target_units is an integer contract target. A fractional
    // value is rejected, never silently rounded to a position.
    { Signal x = s; x.target_units = 0.51;  CHECK_THROWS_AS(validate_signal(x), InvalidSignal); }
    { Signal x = s; x.target_units = 1.5;   CHECK_THROWS_AS(validate_signal(x), InvalidSignal); }
    { Signal x = s; x.target_units = -2.4;  CHECK_THROWS_AS(validate_signal(x), InvalidSignal); }
    { Signal x = s; x.target_units = -3.0;  validate_signal(x); ::quant::test::pass(); }
    { Signal x = s; x.target_units = 2.0 + 1e-9; validate_signal(x); ::quant::test::pass(); }  // within tol
}

// ---- D. deterministic active-contract selection -------------------
//
// Selection follows the CURRENT MARKET STATE (the instrument_id on the feed),
// never an activation-date heuristic. NQZ5 and NQH6 are both live at kRollTs.

void test_contract_selection() {
    const auto reg = two_contract_registry();
    const RegistryActiveContractResolver resolver(reg);

    CHECK(reg.by_instrument_id(101).is_live_at(kRollTs));  // genuine overlap:
    CHECK(reg.by_instrument_id(102).is_live_at(kRollTs));  // both NQ contracts live

    // Feed on NQH6 -> NQH6.
    const ContractSpec& picked = resolver.resolve("NQ", MarketState{102, kRollTs});
    CHECK(picked.instrument_id == 102);
    CHECK(picked.raw_symbol == "NQH6");

    // Same timestamp, feed on NQZ5 -> NQZ5. The activation-date heuristic
    // (latest_live_contract) would wrongly return NQH6 for this same instant.
    CHECK(resolver.resolve("NQ", MarketState{101, kRollTs}).instrument_id == 101);
    CHECK(reg.latest_live_contract("NQ", kRollTs)->instrument_id == 102);

    // No market state -> deterministic, typed failure.
    CHECK_THROWS_AS(resolver.resolve("NQ", MarketState{}), ContractResolutionError);
    // Feed instrument_id not in the registry.
    CHECK_THROWS_AS(resolver.resolve("NQ", MarketState{999, kRollTs}), ContractResolutionError);
    // Root mismatch: the feed is on an NQ contract, the Signal asks for ES.
    CHECK_THROWS_AS(resolver.resolve("ES", MarketState{102, kRollTs}), ContractResolutionError);
    // Feed instrument not live at the state timestamp.
    CHECK_THROWS_AS(resolver.resolve("NQ", MarketState{101, kNqZ5Expiration + 1}),
                    ContractResolutionError);
}

// ---- D2. real roll semantics (modelled on Phase 04.5) ------------
//
// Before the roll the NQ.v.0 feed carries instrument_id 42004058 (NQM6); after,
// 42004177 (NQU6). BOTH are live across the overlap. The SAME Signal(root="NQ")
// must become Order(NQM6) before and Order(NQU6) after -- decided purely by the
// current MarketEvent.instrument_id: no contract-month parsing, no
// activation-date inference.

constexpr std::uint32_t kNqM6Id = 42004058;
constexpr std::uint32_t kNqU6Id = 42004177;
constexpr std::uint32_t kEsU6Id = 55501;
constexpr std::int64_t  kNqM6Act = 1'690'000'000'000'000'000;
constexpr std::int64_t  kNqM6Exp = 1'752'000'000'000'000'000;
constexpr std::int64_t  kNqU6Act = 1'745'000'000'000'000'000;  // overlaps NQM6
constexpr std::int64_t  kNqU6Exp = 1'760'000'000'000'000'000;
constexpr std::int64_t  kJuneRollTs = 1'750'000'000'000'000'000;  // both NQ contracts live

ContractRegistry roll_registry() {
    ContractRegistry reg;
    reg.add(ContractSpec{.instrument_id = kNqM6Id, .raw_symbol = "NQM6", .root_symbol = "NQ",
                         .exchange = "XCME", .tick_size = 0.25, .multiplier = 20.0,
                         .activation_ns = kNqM6Act, .expiration_ns = kNqM6Exp});
    reg.add(ContractSpec{.instrument_id = kNqU6Id, .raw_symbol = "NQU6", .root_symbol = "NQ",
                         .exchange = "XCME", .tick_size = 0.25, .multiplier = 20.0,
                         .activation_ns = kNqU6Act, .expiration_ns = kNqU6Exp});
    reg.add(ContractSpec{.instrument_id = kEsU6Id, .raw_symbol = "ESU6", .root_symbol = "ES",
                         .exchange = "XCME", .tick_size = 0.25, .multiplier = 50.0,
                         .activation_ns = kNqU6Act, .expiration_ns = kNqU6Exp});
    return reg;
}

MarketEvent feed_event(std::uint32_t instrument_id, std::int64_t ts) {
    MarketEvent ev = good_bar_event();
    ev.instrument_id = instrument_id;
    ev.ts_event_ns = ts;
    return ev;
}

void test_real_roll_contract_selection() {
    const auto reg = roll_registry();
    const RegistryActiveContractResolver resolver(reg);

    // Both NQ contracts are simultaneously live at the roll instant.
    CHECK(reg.by_instrument_id(kNqM6Id).is_live_at(kJuneRollTs));
    CHECK(reg.by_instrument_id(kNqU6Id).is_live_at(kJuneRollTs));

    Signal sig = good_signal();
    sig.signal_id = 1;
    sig.ts_decision_ns = kJuneRollTs;
    sig.root_symbol = "NQ";
    sig.target_units = 1.0;

    // Before the roll: feed carries NQM6 -> Order(NQM6). Flat -> target: BUY 1.
    const MarketState before = MarketState::from_event(feed_event(kNqM6Id, kJuneRollTs));
    const auto o_before = make_order_from_signal(sig, resolver, before, /*current=*/0, 10, kJuneRollTs);
    CHECK(o_before.has_value());
    CHECK(o_before->instrument_id == kNqM6Id);
    CHECK(o_before->raw_symbol == "NQM6");
    CHECK(o_before->quantity == 1);

    // After the roll: SAME Signal, feed now carries NQU6 -> Order(NQU6).
    const MarketState after = MarketState::from_event(feed_event(kNqU6Id, kJuneRollTs));
    const auto o_after = make_order_from_signal(sig, resolver, after, /*current=*/0, 11, kJuneRollTs);
    CHECK(o_after.has_value());
    CHECK(o_after->instrument_id == kNqU6Id);
    CHECK(o_after->raw_symbol == "NQU6");

    // The registry's activation-date heuristic cannot tell the two apart in the
    // overlap -- it always returns the later-activated NQU6. Market state is the
    // only thing that distinguishes "before" from "after".
    CHECK(reg.latest_live_contract("NQ", kJuneRollTs)->instrument_id == kNqU6Id);

    // A Signal for NQ must fail when the feed's active instrument is a different
    // root (here an ES contract).
    const MarketState es_feed = MarketState::from_event(feed_event(kEsU6Id, kJuneRollTs));
    CHECK_THROWS_AS(make_order_from_signal(sig, resolver, es_feed, /*current=*/0, 12, kJuneRollTs),
                    ContractResolutionError);
    CHECK_THROWS_AS(resolver.resolve("NQ", es_feed), ContractResolutionError);

    // Target-position delta: already at target -> no order.
    Signal hold = sig;
    hold.target_units = 1.0;
    CHECK(!make_order_from_signal(hold, resolver, before, /*current=*/1, 13, kJuneRollTs).has_value());
    // +1 held, target -1 -> SELL 2.
    Signal flip = sig;
    flip.target_units = -1.0;
    const auto o_flip = make_order_from_signal(flip, resolver, before, /*current=*/1, 14, kJuneRollTs);
    CHECK(o_flip.has_value());
    CHECK(o_flip->side == Side::Sell);
    CHECK(o_flip->quantity == 2);
}

// ---- E. Order ----------------------------------------------------

Order good_order() {
    Order o;
    o.order_id = 1;
    o.signal_id = 1;
    o.ts_created_ns = kRollTs;
    o.instrument_id = 102;
    o.raw_symbol = "NQH6";
    o.side = Side::Buy;
    o.quantity = 1;
    o.order_type = OrderType::Market;
    return o;
}

void test_order_invariants() {
    const auto reg = two_contract_registry();

    validate_order(good_order(), reg);
    ::quant::test::pass();

    { Order o = good_order(); o.order_id = 0;   CHECK_THROWS_AS(validate_order(o, reg), InvalidOrder); }
    { Order o = good_order(); o.signal_id = 0;  CHECK_THROWS_AS(validate_order(o, reg), InvalidOrder); }
    { Order o = good_order(); o.quantity = 0;   CHECK_THROWS_AS(validate_order(o, reg), InvalidOrder); }
    { Order o = good_order(); o.quantity = -3;  CHECK_THROWS_AS(validate_order(o, reg), InvalidOrder); }
    { Order o = good_order(); o.instrument_id = 0;  CHECK_THROWS_AS(validate_order(o, reg), InvalidOrder); }
    { Order o = good_order(); o.instrument_id = 999; CHECK_THROWS_AS(validate_order(o, reg), InvalidOrder); }
    {  // continuous raw_symbol is forbidden in an Order
        Order o = good_order(); o.raw_symbol = "NQ.v.0";
        CHECK_THROWS_AS(validate_order(o, reg), InvalidOrder);
    }
    {  // instrument_id / raw_symbol mismatch (102 is NQH6, not NQZ5)
        Order o = good_order(); o.raw_symbol = "NQZ5";
        CHECK_THROWS_AS(validate_order(o, reg), InvalidOrder);
    }
    {  // Limit order without a limit price
        Order o = good_order(); o.order_type = OrderType::Limit;
        CHECK_THROWS_AS(validate_order(o, reg), InvalidOrder);
    }
    {  // Limit order with a non-finite / un-normalized limit price (sign is NOT
       //   the rule -- a zero or negative limit is valid, see below)
        Order o = good_order(); o.order_type = OrderType::Limit;
        o.limit_price = std::numeric_limits<double>::quiet_NaN();
        CHECK_THROWS_AS(validate_order(o, reg), InvalidOrder);
        Order o2 = good_order(); o2.order_type = OrderType::Limit; o2.limit_price = 2.0e13;
        CHECK_THROWS_AS(validate_order(o2, reg), InvalidOrder);
    }
    {  // Stop order without a stop price
        Order o = good_order(); o.order_type = OrderType::Stop;
        CHECK_THROWS_AS(validate_order(o, reg), InvalidOrder);
    }
    {  // valid Limit and Stop orders -- including a NEGATIVE limit / stop price
        Order o = good_order(); o.order_type = OrderType::Limit; o.limit_price = 20100.25;
        validate_order(o, reg); ::quant::test::pass();
        Order s = good_order(); s.order_type = OrderType::Stop; s.stop_price = 19900.00;
        validate_order(s, reg); ::quant::test::pass();
        Order ln = good_order(); ln.order_type = OrderType::Limit; ln.limit_price = -21.0;
        validate_order(ln, reg); ::quant::test::pass();
        Order sn = good_order(); sn.order_type = OrderType::Stop; sn.stop_price = -25.0;
        validate_order(sn, reg); ::quant::test::pass();
        Order lz = good_order(); lz.order_type = OrderType::Limit; lz.limit_price = 0.0;
        validate_order(lz, reg); ::quant::test::pass();
    }
}

// ---- F. RiskDecision --------------------------------------------

void test_risk_decision_invariants() {
    // APPROVE
    { RiskDecision d; d.verdict = RiskVerdict::Approve; d.requested_quantity = 3;
      d.approved_quantity = 3; validate_risk_decision(d); ::quant::test::pass(); }
    { RiskDecision d; d.verdict = RiskVerdict::Approve; d.requested_quantity = 3;
      d.approved_quantity = 2; CHECK_THROWS_AS(validate_risk_decision(d), RiskInvariantError); }
    // RESIZE
    { RiskDecision d; d.verdict = RiskVerdict::Resize; d.requested_quantity = 9;
      d.approved_quantity = 5; validate_risk_decision(d); ::quant::test::pass(); }
    { RiskDecision d; d.verdict = RiskVerdict::Resize; d.requested_quantity = 9;
      d.approved_quantity = 9; CHECK_THROWS_AS(validate_risk_decision(d), RiskInvariantError); }
    { RiskDecision d; d.verdict = RiskVerdict::Resize; d.requested_quantity = 9;
      d.approved_quantity = 0; CHECK_THROWS_AS(validate_risk_decision(d), RiskInvariantError); }
    // REJECT
    { RiskDecision d; d.verdict = RiskVerdict::Reject; d.requested_quantity = 4;
      d.approved_quantity = 0; validate_risk_decision(d); ::quant::test::pass(); }
    { RiskDecision d; d.verdict = RiskVerdict::Reject; d.requested_quantity = 4;
      d.approved_quantity = 1; CHECK_THROWS_AS(validate_risk_decision(d), RiskInvariantError); }

    // The mandatory Order -> RiskManager -> RiskDecision path.
    const auto reg = two_contract_registry();
    Order o = good_order();
    o.quantity = 9;

    const PassThroughRiskManager pass_mgr;
    const RiskDecision approve = pass_mgr.review(o);
    CHECK(approve.verdict == RiskVerdict::Approve);
    CHECK(approve.approved_quantity == 9);
    CHECK(apply_risk_decision(o, approve).quantity == 9);

    const MaxContractsRiskManager cap_mgr(RiskLimits{});  // max_contracts_per_symbol == 5
    const RiskDecision resize = cap_mgr.review(o);
    CHECK(resize.verdict == RiskVerdict::Resize);
    CHECK(resize.approved_quantity == 5);
    CHECK(resize.reason_code == "max_contracts_per_symbol");
    CHECK(apply_risk_decision(o, resize).quantity == 5);

    Order zero = good_order();
    zero.quantity = 0;
    const RiskDecision reject = cap_mgr.review(zero);
    CHECK(reject.verdict == RiskVerdict::Reject);
    CHECK(reject.approved_quantity == 0);
    CHECK_THROWS_AS(apply_risk_decision(zero, reject), RiskInvariantError);

    // A decision that does not belong to the order is refused.
    RiskDecision alien = approve;
    alien.order_id = 999;
    CHECK_THROWS_AS(apply_risk_decision(o, alien), RiskInvariantError);
}

// ---- G. Fill model ---------------------------------------------

void test_fill_model() {
    const auto reg = two_contract_registry();

    FillRequest req{};
    req.fill_id = 7;
    req.order_id = 3;
    req.ts_fill_ns = kRollTs;
    req.instrument_id = 102;
    req.side = Side::Buy;
    req.quantity = 2;
    req.price = 20100.10;  // not tick aligned

    const Fill f = make_fill(reg, req);
    CHECK(f.fill_id == 7);
    CHECK(f.order_id == 3);
    CHECK(f.instrument_id == 102);
    CHECK(f.raw_symbol == "NQH6");
    CHECK(f.price_domain == PriceDomain::RawContract);
    CHECK_CLOSE(f.tick_size, 0.25, 1e-12);   // from ContractSpec, not the caller
    CHECK_CLOSE(f.multiplier, 20.0, 1e-12);  // from ContractSpec, not the caller
    CHECK_CLOSE(f.fill_price, 20100.00, 1e-9);

    // multiplier converts a price move to USD: one tick on NQ == $5.
    CHECK_CLOSE(f.tick_size * f.multiplier, 5.0, 1e-9);
    // 3-point favourable move on 2 contracts == 3 * 20 * 2 == $120.
    CHECK_CLOSE(3.0 * f.multiplier * f.quantity, 120.0, 1e-9);

    {  // back-adjusted price never reaches execution
        FillRequest r = req; r.price_domain = PriceDomain::BackAdjusted;
        CHECK_THROWS_AS(make_fill(reg, r), FillResolutionError);
        CHECK_THROWS_AS(make_fill(reg, r), ExecutionDomainError);  // and the category
    }
    {  // unadjusted continuous price never reaches execution
        FillRequest r = req; r.price_domain = PriceDomain::RawContinuous;
        CHECK_THROWS_AS(make_fill(reg, r), FillResolutionError);
    }
}

// ---- H. deterministic event ordering --------------------------

void test_event_ordering() {
    // Same timestamp, distinct seq -> ordered by seq. Different ts -> by ts.
    std::vector<MarketEvent> evs;
    auto mk = [](std::int64_t ts, SeqNum seq) {
        MarketEvent e = good_bar_event();
        e.ts_event_ns = ts;
        e.seq = seq;
        return e;
    };
    evs.push_back(mk(200, 9));
    evs.push_back(mk(100, 5));
    evs.push_back(mk(100, 2));
    evs.push_back(mk(200, 1));
    evs.push_back(mk(150, 7));

    // Shuffle with a fixed seed, then sort: the result must be identical every
    // run and must not depend on input / container order.
    std::mt19937 rng(12345);
    std::shuffle(evs.begin(), evs.end(), rng);
    std::sort(evs.begin(), evs.end(), MarketEventOrder{});

    const std::vector<std::pair<std::int64_t, SeqNum>> expected = {
        {100, 2}, {100, 5}, {150, 7}, {200, 1}, {200, 9}};
    for (std::size_t i = 0; i < evs.size(); ++i) {
        CHECK(evs[i].ts_event_ns == expected[i].first);
        CHECK(evs[i].seq == expected[i].second);
    }

    CHECK(event_before(mk(100, 1), mk(100, 2)));
    CHECK(!event_before(mk(100, 2), mk(100, 1)));
    CHECK(event_before(mk(100, 9), mk(101, 0)));
    // A strict weak order: equal events are not "before" each other.
    CHECK(!event_before(mk(100, 3), mk(100, 3)));
}

// ---- I. traceability -----------------------------------------

void test_traceability_chain() {
    const auto reg = two_contract_registry();
    const RegistryActiveContractResolver resolver(reg);
    const MarketState market_state{102, kRollTs};  // feed is on NQH6

    MonotonicId signal_ids;
    MonotonicId order_ids;
    MonotonicId fill_ids;

    Signal sig = good_signal();
    sig.signal_id = signal_ids.next();  // S1
    sig.target_units = 2.0;

    const auto maybe_order =
        make_order_from_signal(sig, resolver, market_state, /*current=*/0, order_ids.next(), kRollTs);
    CHECK(maybe_order.has_value());
    Order ord = *maybe_order;
    validate_order(ord, reg);

    CHECK(ord.order_id == 1);                 // O1
    CHECK(ord.signal_id == sig.signal_id);    // O1 -> S1
    CHECK(ord.instrument_id == 102);
    CHECK(ord.raw_symbol == "NQH6");
    CHECK(ord.side == Side::Buy);
    CHECK(ord.quantity == 2);

    const MaxContractsRiskManager risk(RiskLimits{});
    const RiskDecision decision = risk.review(ord);
    CHECK(decision.order_id == ord.order_id);  // decision -> O1
    CHECK(decision.verdict == RiskVerdict::Approve);
    const Order approved = apply_risk_decision(ord, decision);

    FillRequest req{};
    req.fill_id = fill_ids.next();  // F1
    req.order_id = approved.order_id;
    req.ts_fill_ns = kRollTs;
    req.instrument_id = approved.instrument_id;
    req.side = approved.side;
    req.quantity = approved.quantity;
    req.price = 20100.07;
    req.expected_raw_symbol = approved.raw_symbol;

    const Fill fill = make_fill(reg, req);
    CHECK(fill.fill_id == 1);                     // F1
    CHECK(fill.order_id == ord.order_id);         // F1 -> O1
    CHECK(fill.instrument_id == ord.instrument_id);
    CHECK(fill.raw_symbol == "NQH6");
    CHECK_CLOSE(fill.fill_price, 20100.00, 1e-9);

    // A Flat signal from a flat position produces no order (intent != execution).
    Signal flat = good_signal();
    flat.signal_id = signal_ids.next();
    flat.target_units = 0.0;
    CHECK(!make_order_from_signal(flat, resolver, market_state, /*current=*/0, order_ids.next(),
                                  kRollTs)
               .has_value());

    CHECK(trace_tag('S', sig.signal_id) == "S1");
    CHECK(trace_tag('O', ord.order_id) == "O1");
    CHECK(trace_tag('F', fill.fill_id) == "F1");
}

}  // namespace

int main() {
    test_market_event_validation();
    test_strategy_context_no_lookahead();
    test_signal_validation();
    test_contract_selection();
    test_real_roll_contract_selection();
    test_order_invariants();
    test_risk_decision_invariants();
    test_fill_model();
    test_event_ordering();
    test_traceability_chain();
    return quant::test::summary("quant_domain_model_tests");
}
