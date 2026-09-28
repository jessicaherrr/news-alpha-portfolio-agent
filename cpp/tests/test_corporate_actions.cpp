// Phase 6 ETF pilot -- corporate actions (SplitAction / CashDistribution).
//
//   1. Integer-preserving forward split rebases units/avg_entry_price, notional
//      invariant, realized PnL / costs untouched.
//   2. A fractional-unit split throws FractionalSplitQuantityError and leaves
//      the position completely unchanged (strong exception guarantee).
//   3. A split on a flat / never-seen position is a clean no-op.
//   4. record_distribution_ex_date creates the correct receivable; cash_usd is
//      unchanged immediately after.
//   5. A fill between ex-date and payment does not change the already-recorded
//      receivable (entitlement snapshotted at ex-date).
//   6. record_distribution_payment moves the receivable into
//      distribution_income_usd and never mixes with Fill-derived realized PnL.
//   7. record_distribution_payment with no matching ex-date record throws.

#include "quant_core/contract.hpp"
#include "quant_core/contract_registry.hpp"
#include "quant_core/corporate_actions.hpp"
#include "quant_core/fill.hpp"
#include "quant_core/portfolio.hpp"
#include "quant_core/position_ledger.hpp"

#include "test_support.hpp"

#include <string>

namespace {

using namespace quant;

constexpr std::int64_t kMin = 60'000'000'000LL;
constexpr std::int64_t kBigExp = 500'000LL * kMin;

constexpr std::uint32_t kSpy = 100;  // pretend-ETF instrument id, multiplier 1

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

PortfolioConfig cfg() {
    PortfolioConfig c;
    c.starting_capital_usd = 100'000.0;
    c.mark_staleness_tolerance_ns = 1LL << 60;
    return c;
}

// ---- 1: integer-preserving forward split --------------------------------

void test_split_forward_preserves_notional() {
    PositionLedger ledger;
    const auto reg = etf_registry();
    // Open 100 units @ 10.0 via apply() so realized_pnl_usd_/costs_usd_ carry a
    // real running total we can assert is untouched by the split.
    ledger.apply(mkfill(reg, kSpy, Side::Buy, 100, 10.0, kMin, 1.0), "SPY", "signal");

    const double realized_before = ledger.realized_pnl_usd();
    const double costs_before    = ledger.costs_usd();

    ledger.apply_split(kSpy, 2.0);  // 2-for-1 forward split

    CHECK(ledger.units(kSpy) == 200);
    const auto positions = ledger.open_positions();
    CHECK(positions.size() == 1);
    CHECK_CLOSE(positions[0].avg_entry_price, 5.0, 1e-9);
    // Cost-basis notional (units * avg_entry_price) invariant before/after.
    CHECK_CLOSE(static_cast<double>(positions[0].units) * positions[0].avg_entry_price, 1000.0, 1e-9);

    CHECK_CLOSE(ledger.realized_pnl_usd(), realized_before, 1e-12);
    CHECK_CLOSE(ledger.costs_usd(), costs_before, 1e-12);
}

// ---- 2: fractional split throws and leaves state unchanged --------------

void test_split_fractional_throws_and_unchanged() {
    PositionLedger ledger;
    const auto reg = etf_registry();
    ledger.apply(mkfill(reg, kSpy, Side::Buy, 100, 10.0, kMin, 1.0), "SPY", "signal");

    CHECK_THROWS_AS(ledger.apply_split(kSpy, 0.125), FractionalSplitQuantityError);  // 100 * 0.125 = 12.5

    // Completely unchanged.
    CHECK(ledger.units(kSpy) == 100);
    const auto positions = ledger.open_positions();
    CHECK(positions.size() == 1);
    CHECK_CLOSE(positions[0].avg_entry_price, 10.0, 1e-12);
}

// ---- 3: split on a flat / never-seen position is a clean no-op ----------

void test_split_flat_position_noop() {
    PositionLedger ledger;
    // Never seen at all.
    ledger.apply_split(kSpy, 2.0);
    CHECK(ledger.units(kSpy) == 0);

    const auto reg = etf_registry();
    // Open then fully close -> units == 0 but instrument known.
    ledger.apply(mkfill(reg, kSpy, Side::Buy, 10, 10.0, kMin, 0.0), "SPY", "signal");
    ledger.apply(mkfill(reg, kSpy, Side::Sell, 10, 11.0, 2 * kMin, 0.0), "SPY", "signal");
    CHECK(ledger.units(kSpy) == 0);
    const double realized_before = ledger.realized_pnl_usd();

    ledger.apply_split(kSpy, 3.0);  // no-op: flat
    CHECK(ledger.units(kSpy) == 0);
    CHECK_CLOSE(ledger.realized_pnl_usd(), realized_before, 1e-12);
}

// ---- PortfolioAccountant.apply_split propagates validation / errors -----

void test_portfolio_apply_split_delegates() {
    const auto reg = etf_registry();
    PortfolioAccountant pf(cfg(), nullptr);
    pf.apply_fill(mkfill(reg, kSpy, Side::Buy, 100, 10.0, kMin, 1.0), "SPY", "signal");

    SplitAction split{.instrument_id = kSpy, .effective_ts_ns = 2 * kMin, .ratio = 2.0};
    pf.apply_split(split);
    CHECK(pf.ledger().units(kSpy) == 200);

    SplitAction bad{.instrument_id = kSpy, .effective_ts_ns = 3 * kMin, .ratio = 0.001};  // 200*0.001=0.2
    CHECK_THROWS_AS(pf.apply_split(bad), FractionalSplitQuantityError);
    CHECK(pf.ledger().units(kSpy) == 200);  // unchanged after the throw

    SplitAction invalid{.instrument_id = kSpy, .effective_ts_ns = 4 * kMin, .ratio = 1.0};  // no-op ratio rejected
    CHECK(!invalid.is_valid());
    CHECK_THROWS(pf.apply_split(invalid));
}

// ---- 4/5/6/7: distributions ---------------------------------------------

void test_distribution_ex_date_creates_receivable_cash_unchanged() {
    const auto reg = etf_registry();
    PortfolioAccountant pf(cfg(), nullptr);
    pf.apply_fill(mkfill(reg, kSpy, Side::Buy, 100, 10.0, kMin, 0.0), "SPY", "signal");

    const auto before = pf.snapshot(2 * kMin);

    CashDistribution dist{.instrument_id = kSpy, .ex_date_ts_ns = 2 * kMin,
                          .pay_date_ts_ns = 5 * kMin, .amount_per_share_usd = 0.50};
    pf.record_distribution_ex_date(dist);

    const auto after = pf.snapshot(2 * kMin);
    CHECK_CLOSE(after.distribution_receivable_usd, 100.0 * 0.50, 1e-9);
    CHECK_CLOSE(after.cash_usd, before.cash_usd, 1e-9);           // unchanged: not spendable cash yet
    CHECK_CLOSE(after.equity_usd, before.equity_usd, 1e-9);
    CHECK_CLOSE(after.distribution_income_usd, 0.0, 1e-9);        // not paid yet
}

void test_distribution_entitlement_snapshotted_at_ex_date() {
    const auto reg = etf_registry();
    PortfolioAccountant pf(cfg(), nullptr);
    pf.apply_fill(mkfill(reg, kSpy, Side::Buy, 100, 10.0, kMin, 0.0), "SPY", "signal");

    CashDistribution dist{.instrument_id = kSpy, .ex_date_ts_ns = 2 * kMin,
                          .pay_date_ts_ns = 5 * kMin, .amount_per_share_usd = 0.50};
    pf.record_distribution_ex_date(dist);
    const double receivable_at_ex_date = pf.snapshot(2 * kMin).distribution_receivable_usd;
    CHECK_CLOSE(receivable_at_ex_date, 50.0, 1e-9);

    // Position changes AFTER the ex-date call, BEFORE payment.
    pf.apply_fill(mkfill(reg, kSpy, Side::Buy, 900, 10.0, 3 * kMin, 0.0), "SPY", "signal");  // now 1000 units
    CHECK(pf.ledger().units(kSpy) == 1000);

    const double receivable_after_more_buying = pf.snapshot(3 * kMin).distribution_receivable_usd;
    CHECK_CLOSE(receivable_after_more_buying, 50.0, 1e-9);  // unchanged: entitlement was snapshotted at ex-date
}

void test_distribution_payment_moves_income_never_mixes_with_realized() {
    const auto reg = etf_registry();
    PortfolioConfig c1 = cfg();
    PortfolioAccountant with_dist(c1, nullptr);
    PortfolioAccountant no_dist(cfg(), nullptr);

    // Identical fills on both accountants.
    with_dist.apply_fill(mkfill(reg, kSpy, Side::Buy, 100, 10.0, kMin, 1.0), "SPY", "signal");
    no_dist.apply_fill(mkfill(reg, kSpy, Side::Buy, 100, 10.0, kMin, 1.0), "SPY", "signal");

    CashDistribution dist{.instrument_id = kSpy, .ex_date_ts_ns = 2 * kMin,
                          .pay_date_ts_ns = 5 * kMin, .amount_per_share_usd = 0.50};
    with_dist.record_distribution_ex_date(dist);

    // A closing fill on both, simultaneous with the distribution's lifecycle.
    with_dist.apply_fill(mkfill(reg, kSpy, Side::Sell, 100, 11.0, 4 * kMin, 1.0), "SPY", "signal");
    no_dist.apply_fill(mkfill(reg, kSpy, Side::Sell, 100, 11.0, 4 * kMin, 1.0), "SPY", "signal");

    with_dist.record_distribution_payment(dist);

    const auto s_with = with_dist.snapshot(5 * kMin);
    const auto s_no    = no_dist.snapshot(4 * kMin);

    // Fill-derived PnL totals are byte-for-byte identical with or without the
    // distribution -- the two income streams never mix.
    CHECK_CLOSE(s_with.gross_realized_pnl_usd, s_no.gross_realized_pnl_usd, 1e-12);
    CHECK_CLOSE(s_with.net_realized_pnl_usd, s_no.net_realized_pnl_usd, 1e-12);
    CHECK_CLOSE(s_with.costs_usd, s_no.costs_usd, 1e-12);
    CHECK_CLOSE(s_with.cash_usd, s_no.cash_usd, 1e-12);

    // The distribution settled into its own separate total.
    CHECK_CLOSE(s_with.distribution_income_usd, 50.0, 1e-9);
    CHECK_CLOSE(s_with.distribution_receivable_usd, 0.0, 1e-9);   // moved out of pending
}

void test_distribution_payment_without_ex_date_throws() {
    const auto reg = etf_registry();
    PortfolioAccountant pf(cfg(), nullptr);
    pf.apply_fill(mkfill(reg, kSpy, Side::Buy, 100, 10.0, kMin, 0.0), "SPY", "signal");

    CashDistribution dist{.instrument_id = kSpy, .ex_date_ts_ns = 2 * kMin,
                          .pay_date_ts_ns = 5 * kMin, .amount_per_share_usd = 0.50};
    // No record_distribution_ex_date call first.
    CHECK_THROWS_AS(pf.record_distribution_payment(dist), NoPendingDistributionError);

    const auto s = pf.snapshot(kMin);
    CHECK_CLOSE(s.distribution_income_usd, 0.0, 1e-9);  // nothing fabricated
}

// ---- short-position distribution is a liability (sign-aware) ------------

void test_distribution_short_position_is_negative_receivable() {
    const auto reg = etf_registry();
    PortfolioAccountant pf(cfg(), nullptr);
    pf.apply_fill(mkfill(reg, kSpy, Side::Sell, 50, 10.0, kMin, 0.0), "SPY", "signal");  // short 50

    CashDistribution dist{.instrument_id = kSpy, .ex_date_ts_ns = 2 * kMin,
                          .pay_date_ts_ns = 5 * kMin, .amount_per_share_usd = 0.50};
    pf.record_distribution_ex_date(dist);

    const auto s = pf.snapshot(2 * kMin);
    CHECK_CLOSE(s.distribution_receivable_usd, -25.0, 1e-9);  // short owes the distribution
}

}  // namespace

int main() {
    test_split_forward_preserves_notional();
    test_split_fractional_throws_and_unchanged();
    test_split_flat_position_noop();
    test_portfolio_apply_split_delegates();
    test_distribution_ex_date_creates_receivable_cash_unchanged();
    test_distribution_entitlement_snapshotted_at_ex_date();
    test_distribution_payment_moves_income_never_mixes_with_realized();
    test_distribution_payment_without_ex_date_throws();
    test_distribution_short_position_is_negative_receivable();
    return quant::test::summary("quant_corporate_actions_tests");
}
