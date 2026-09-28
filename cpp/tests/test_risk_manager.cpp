// Phase 08 -- deterministic hard portfolio risk (PortfolioRiskManager).
//
//   A. APPROVE / RESIZE / REJECT invariants hold for every returned decision
//   B. per-symbol position cap -- RESIZE then REJECT
//   C. per-root position cap
//   D. gross-contracts cap
//   E. gross-exposure / leverage rejection (ContractSpec economics)
//   F. margin-utilisation rejection
//   G. missing-margin-metadata policy (Reject vs TreatAsZero)
//   H. daily-loss limit -- REJECT increasing, still APPROVE reducing
//   I. drawdown kill switch -- REJECT increasing, still APPROVE reducing
//   J. stale-mark protection (ref stale / portfolio stale / Ignore)
//   K. risk-reducing orders always clear (incl. a smaller-magnitude flip)
//   L. max_order_contracts, context-free review(order)
//   M. end-to-end: drawdown kill switch through a full BacktestEngine run

#include "quant_core/contract.hpp"
#include "quant_core/contract_registry.hpp"
#include "quant_core/contract_selector.hpp"
#include "quant_core/domain_model.hpp"
#include "quant_core/engine.hpp"
#include "quant_core/margin.hpp"
#include "quant_core/portfolio.hpp"
#include "quant_core/risk_config.hpp"
#include "quant_core/risk_manager.hpp"
#include "quant_core/strategy.hpp"
#include "quant_core/strategy_context.hpp"

#include "test_support.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace {

using namespace quant;

constexpr std::int64_t  kMin   = 60'000'000'000LL;
constexpr std::uint32_t kNq     = 1;
constexpr std::uint32_t kNq2    = 2;   // second NQ contract (same root)
constexpr std::int64_t  kBigExp = 100'000LL * kMin;

ContractSpec nq_spec(std::uint32_t id = kNq, std::string sym = "NQZ6") {
    return ContractSpec{.instrument_id = id, .raw_symbol = std::move(sym), .root_symbol = "NQ",
                        .exchange = "XCME", .tick_size = 0.25, .multiplier = 20.0,
                        .activation_ns = 1, .expiration_ns = kBigExp};
}

Order mkorder(Side side, int qty, std::uint32_t id = kNq, std::string sym = "NQZ6") {
    Order o;
    o.order_id      = 1;
    o.signal_id     = 1;
    o.ts_created_ns = 1000;
    o.instrument_id = id;
    o.raw_symbol    = std::move(sym);
    o.side          = side;
    o.quantity      = qty;
    o.order_type    = OrderType::Market;
    o.tif           = TimeInForce::Day;
    return o;
}

PortfolioState healthy_state() {
    PortfolioState s;
    s.as_of_ts_ns           = 1000;
    s.starting_capital_usd  = 100'000.0;
    s.cash_usd              = 100'000.0;
    s.equity_usd            = 100'000.0;
    s.peak_equity_usd       = 100'000.0;
    s.day_start_equity_usd  = 100'000.0;
    return s;
}

RiskReviewContext mkctx(const PortfolioState& pf, const ContractSpec& spec, int cur_units,
                        int root_units, double ref_price) {
    RiskReviewContext c;
    c.as_of_ts_ns              = 1000;
    c.portfolio                = &pf;
    c.spec                     = &spec;
    c.current_position_units   = cur_units;
    c.current_root_units       = root_units;
    c.reference_price          = ref_price;
    c.reference_price_ts_ns    = 1000;
    c.reference_price_age_ns   = 0;
    c.reference_price_stale    = false;
    c.reference_price_present  = true;
    return c;
}

MarginModel nq_margin(double initial, double maintenance) {
    MarginModel m;
    m.set(MarginRequirement{"NQ", initial, maintenance, "test", 1});
    return m;
}

// A portfolio holding exactly `units` NQ at `mark`, with `equity` on the books.
PortfolioState state_with_nq(int units, double mark, double equity) {
    PortfolioState s = healthy_state();
    s.equity_usd           = equity;
    s.cash_usd             = equity;
    s.day_start_equity_usd = equity;
    s.peak_equity_usd      = std::max(100'000.0, equity);

    PositionExposure e;
    e.instrument_id   = kNq;
    e.raw_symbol      = "NQZ6";
    e.root_symbol     = "NQ";
    e.units           = units;
    e.avg_entry_price = mark;
    e.multiplier      = 20.0;
    e.mark_present    = true;
    e.mark_price      = mark;
    e.mark_ts_ns      = 1000;
    const int ps            = (units > 0) - (units < 0);
    e.gross_notional_usd   = std::fabs(static_cast<double>(units) * mark * 20.0);
    e.signed_notional_usd  = ps * e.gross_notional_usd;
    s.positions.push_back(e);

    s.gross_exposure_usd = e.gross_notional_usd;
    s.net_exposure_usd   = e.signed_notional_usd;
    if (ps > 0) s.long_exposure_usd = e.gross_notional_usd;
    else        s.short_exposure_usd = e.gross_notional_usd;
    s.gross_contracts = std::abs(units);
    s.net_contracts   = units;
    return s;
}

// Every decision a hard-risk manager returns must satisfy the frozen invariants.
void assert_valid(const RiskDecision& d) { validate_risk_decision(d); ::quant::test::pass(); }

// ---- B: per-symbol cap ---------------------------------------------------

void test_per_symbol_cap() {
    RiskConfig cfg;
    cfg.max_contracts_per_symbol = 5;
    cfg.max_margin_utilization_pct = 0.0;     // isolate the position cap
    cfg.missing_margin = MissingMarginPolicy::TreatAsZero;
    cfg.stale_mark = StaleMarkPolicy::Ignore;
    const PortfolioRiskManager risk(cfg, nullptr);
    const ContractSpec spec = nq_spec();
    const PortfolioState pf = healthy_state();

    {  // 0 -> +9 : RESIZE down to 5
        const auto d = risk.review(mkorder(Side::Buy, 9), mkctx(pf, spec, 0, 0, 20000.0));
        CHECK(d.verdict == RiskVerdict::Resize);
        CHECK(d.approved_quantity == 5);
        CHECK(d.reason_code == "max_contracts_per_symbol");
        assert_valid(d);
    }
    {  // already at the cap, +1 more : REJECT
        const auto d = risk.review(mkorder(Side::Buy, 1), mkctx(pf, spec, 5, 5, 20000.0));
        CHECK(d.verdict == RiskVerdict::Reject);
        CHECK(d.approved_quantity == 0);
        CHECK(d.reason_code == "max_contracts_per_symbol");
        assert_valid(d);
    }
    {  // at the cap on the long side, SELL 3 (reduce) : APPROVE full
        const auto d = risk.review(mkorder(Side::Sell, 3), mkctx(pf, spec, 5, 5, 20000.0));
        CHECK(d.verdict == RiskVerdict::Approve);
        CHECK(d.approved_quantity == 3);
        assert_valid(d);
    }
}

// ---- C / D: per-root and gross-contracts caps ---------------------------

void test_root_and_gross_caps() {
    RiskConfig cfg;
    cfg.max_contracts_per_symbol = 100;
    cfg.max_contracts_per_root   = 10;
    cfg.max_gross_contracts      = 100;
    cfg.max_margin_utilization_pct = 0.0;
    cfg.missing_margin = MissingMarginPolicy::TreatAsZero;
    cfg.stale_mark = StaleMarkPolicy::Ignore;
    const PortfolioRiskManager risk(cfg, nullptr);
    const ContractSpec spec = nq_spec(kNq2, "NQH7");
    const PortfolioState pf = healthy_state();

    // this instrument flat, but the root already holds +8 elsewhere; BUY 5 -> root +13
    const auto d = risk.review(mkorder(Side::Buy, 5, kNq2, "NQH7"), mkctx(pf, spec, 0, 8, 20000.0));
    CHECK(d.verdict == RiskVerdict::Resize);
    CHECK(d.approved_quantity == 2);                     // root cap 10 - 8
    CHECK(d.reason_code == "max_contracts_per_root");
    assert_valid(d);

    // gross-contracts cap: portfolio already holds 10 elsewhere, gross cap 12
    RiskConfig g = cfg;
    g.max_contracts_per_root = 100;
    g.max_gross_contracts    = 12;
    PortfolioState pf2 = healthy_state();
    pf2.gross_contracts = 10;
    const PortfolioRiskManager risk2(g, nullptr);
    const auto d2 = risk2.review(mkorder(Side::Buy, 5), mkctx(pf2, spec, 0, 0, 20000.0));
    CHECK(d2.verdict == RiskVerdict::Resize);
    CHECK(d2.approved_quantity == 2);
    CHECK(d2.reason_code == "max_gross_contracts");
    assert_valid(d2);
}

// ---- E: leverage / gross-exposure rejection (multiplier economics) ------

void test_leverage_rejection() {
    RiskConfig cfg;
    cfg.max_contracts_per_symbol = 100;
    cfg.max_contracts_per_root   = 100;
    cfg.max_gross_contracts      = 100;
    cfg.max_margin_utilization_pct = 0.0;
    cfg.missing_margin = MissingMarginPolicy::TreatAsZero;
    cfg.stale_mark = StaleMarkPolicy::Ignore;
    cfg.max_gross_leverage = 2.0;
    const PortfolioRiskManager risk(cfg, nullptr);
    const ContractSpec spec = nq_spec();
    const PortfolioState pf = healthy_state();  // equity 100k

    // 1 NQ @ 20000 * multiplier 20 = 400k notional -> leverage 4.0 > 2.0.
    // Exposure is NOT quantity*price (that would be 20000).
    const auto d = risk.review(mkorder(Side::Buy, 1), mkctx(pf, spec, 0, 0, 20000.0));
    CHECK(d.verdict == RiskVerdict::Reject);
    CHECK(d.reason_code == "max_gross_leverage");
    assert_valid(d);

    // A gross-exposure ceiling that admits exactly 2 contracts -> RESIZE 5 -> 2.
    RiskConfig e = cfg;
    e.max_gross_leverage = 0.0;
    e.max_gross_exposure_usd = 850'000.0;  // 2 * 400k = 800k <= 850k < 1.2M
    const PortfolioRiskManager risk_e(e, nullptr);
    const auto d2 = risk_e.review(mkorder(Side::Buy, 5), mkctx(pf, spec, 0, 0, 20000.0));
    CHECK(d2.verdict == RiskVerdict::Resize);
    CHECK(d2.approved_quantity == 2);
    CHECK(d2.reason_code == "max_gross_exposure_usd");
    assert_valid(d2);
}

// ---- F / G: margin utilisation + missing-margin policy -----------------

void test_margin_rules() {
    RiskConfig cfg;
    cfg.max_contracts_per_symbol = 100;
    cfg.max_contracts_per_root   = 100;
    cfg.max_gross_contracts      = 100;
    cfg.max_gross_leverage       = 0.0;
    cfg.stale_mark = StaleMarkPolicy::Ignore;
    cfg.max_margin_utilization_pct = 0.50;

    const ContractSpec spec = nq_spec();
    const PortfolioState pf = healthy_state();  // equity 100k

    {  // known margin, 60k / contract -> util 0.60 > 0.50 -> REJECT
        const MarginModel m = nq_margin(60'000.0, 50'000.0);
        const PortfolioRiskManager risk(cfg, &m);
        const auto d = risk.review(mkorder(Side::Buy, 1), mkctx(pf, spec, 0, 0, 20000.0));
        CHECK(d.verdict == RiskVerdict::Reject);
        CHECK(d.reason_code == "max_margin_utilization_pct");
        assert_valid(d);
    }
    {  // known margin, 20k / contract -> 2 fit (0.40), 3 do not (0.60) -> RESIZE 5 -> 2
        const MarginModel m = nq_margin(20'000.0, 15'000.0);
        const PortfolioRiskManager risk(cfg, &m);
        const auto d = risk.review(mkorder(Side::Buy, 5), mkctx(pf, spec, 0, 0, 20000.0));
        CHECK(d.verdict == RiskVerdict::Resize);
        CHECK(d.approved_quantity == 2);
        CHECK(d.reason_code == "max_margin_utilization_pct");
        assert_valid(d);
    }
    {  // NO margin metadata + policy Reject -> REJECT missing_margin_metadata
        RiskConfig r = cfg;
        r.missing_margin = MissingMarginPolicy::Reject;
        const PortfolioRiskManager risk(r, nullptr);
        const auto d = risk.review(mkorder(Side::Buy, 1), mkctx(pf, spec, 0, 0, 20000.0));
        CHECK(d.verdict == RiskVerdict::Reject);
        CHECK(d.reason_code == "missing_margin_metadata");
        assert_valid(d);
    }
    {  // NO margin metadata + policy TreatAsZero -> utilisation check passes at 0
        RiskConfig r = cfg;
        r.missing_margin = MissingMarginPolicy::TreatAsZero;
        const PortfolioRiskManager risk(r, nullptr);
        const auto d = risk.review(mkorder(Side::Buy, 1), mkctx(pf, spec, 0, 0, 20000.0));
        CHECK(d.verdict == RiskVerdict::Approve);
        assert_valid(d);
    }
}

// ---- H: daily-loss limit ----------------------------------------------

void test_daily_loss_limit() {
    RiskConfig cfg;
    cfg.missing_margin = MissingMarginPolicy::TreatAsZero;
    cfg.max_margin_utilization_pct = 0.0;
    cfg.stale_mark = StaleMarkPolicy::Ignore;
    cfg.max_daily_loss_usd = 1500.0;
    const PortfolioRiskManager risk(cfg, nullptr);
    const ContractSpec spec = nq_spec();

    PortfolioState pf = healthy_state();
    pf.day_start_equity_usd = 100'000.0;
    pf.equity_usd           = 98'000.0;   // -2000 on the day, past the 1500 limit
    pf.peak_equity_usd      = 100'000.0;

    {  // risk-increasing order -> REJECT
        const auto d = risk.review(mkorder(Side::Buy, 1), mkctx(pf, spec, 0, 0, 20000.0));
        CHECK(d.verdict == RiskVerdict::Reject);
        CHECK(d.reason_code == "daily_loss_limit");
        assert_valid(d);
    }
    {  // still allowed to REDUCE / get flat
        const auto d = risk.review(mkorder(Side::Sell, 3), mkctx(pf, spec, 3, 3, 20000.0));
        CHECK(d.verdict == RiskVerdict::Approve);
        CHECK(d.approved_quantity == 3);
        assert_valid(d);
    }
}

// ---- I: drawdown kill switch -----------------------------------------

void test_drawdown_kill_switch() {
    RiskConfig cfg;
    cfg.missing_margin = MissingMarginPolicy::TreatAsZero;
    cfg.max_margin_utilization_pct = 0.0;
    cfg.stale_mark = StaleMarkPolicy::Ignore;
    cfg.max_daily_loss_usd = 0.0;    // isolate drawdown
    cfg.max_drawdown_pct = 0.10;
    const PortfolioRiskManager risk(cfg, nullptr);
    const ContractSpec spec = nq_spec();

    PortfolioState pf = healthy_state();
    pf.peak_equity_usd = 100'000.0;
    pf.equity_usd      = 88'000.0;
    pf.drawdown_usd    = 12'000.0;
    pf.drawdown_pct    = 0.12;         // past 10%
    pf.day_start_equity_usd = 88'000.0;

    {  // risk-increasing -> REJECT
        const auto d = risk.review(mkorder(Side::Buy, 2), mkctx(pf, spec, 1, 1, 20000.0));
        CHECK(d.verdict == RiskVerdict::Reject);
        CHECK(d.reason_code == "drawdown_kill_switch");
        assert_valid(d);
    }
    {  // reducing still allowed
        const auto d = risk.review(mkorder(Side::Sell, 1), mkctx(pf, spec, 1, 1, 20000.0));
        CHECK(d.verdict == RiskVerdict::Approve);
        assert_valid(d);
    }
    {  // absolute-usd variant
        RiskConfig a = cfg;
        a.max_drawdown_pct = 0.0;
        a.max_drawdown_usd = 10'000.0;
        const PortfolioRiskManager risk_a(a, nullptr);
        const auto d = risk_a.review(mkorder(Side::Buy, 1), mkctx(pf, spec, 0, 0, 20000.0));
        CHECK(d.verdict == RiskVerdict::Reject);
        CHECK(d.reason_code == "drawdown_kill_switch");
        assert_valid(d);
    }
}

// ---- J: stale-mark protection ---------------------------------------

void test_stale_mark_protection() {
    RiskConfig cfg;
    cfg.missing_margin = MissingMarginPolicy::TreatAsZero;
    cfg.max_margin_utilization_pct = 0.0;
    cfg.stale_mark = StaleMarkPolicy::RejectRiskIncreasing;
    const PortfolioRiskManager risk(cfg, nullptr);
    const ContractSpec spec = nq_spec();
    const PortfolioState healthy = healthy_state();

    {  // the order's own reference price is stale -> REJECT increasing
        auto ctx = mkctx(healthy, spec, 0, 0, 20000.0);
        ctx.reference_price_stale = true;
        const auto d = risk.review(mkorder(Side::Buy, 1), ctx);
        CHECK(d.verdict == RiskVerdict::Reject);
        CHECK(d.reason_code == "stale_mark");
        assert_valid(d);
    }
    {  // another held position has a stale mark -> REJECT increasing
        PortfolioState pf = healthy_state();
        pf.has_stale_mark = true;
        const auto d = risk.review(mkorder(Side::Buy, 1), mkctx(pf, spec, 0, 0, 20000.0));
        CHECK(d.verdict == RiskVerdict::Reject);
        CHECK(d.reason_code == "stale_mark");
        assert_valid(d);
    }
    {  // stale portfolio, but the order REDUCES risk -> APPROVE
        PortfolioState pf = healthy_state();
        pf.has_stale_mark = true;
        const auto d = risk.review(mkorder(Side::Sell, 2), mkctx(pf, spec, 2, 2, 20000.0));
        CHECK(d.verdict == RiskVerdict::Approve);
        assert_valid(d);
    }
    {  // policy Ignore -> staleness does not block
        RiskConfig ig = cfg;
        ig.stale_mark = StaleMarkPolicy::Ignore;
        const PortfolioRiskManager risk_ig(ig, nullptr);
        auto ctx = mkctx(healthy, spec, 0, 0, 20000.0);
        ctx.reference_price_stale = true;
        const auto d = risk_ig.review(mkorder(Side::Buy, 1), ctx);
        CHECK(d.verdict == RiskVerdict::Approve);
        assert_valid(d);
    }
}

// ---- K: reducing / flip always clears; L: order cap + context-free -----

void test_reducing_and_order_cap() {
    RiskConfig cfg;
    cfg.max_contracts_per_symbol = 5;
    cfg.missing_margin = MissingMarginPolicy::TreatAsZero;
    cfg.max_margin_utilization_pct = 0.0;
    cfg.stale_mark = StaleMarkPolicy::RejectRiskIncreasing;
    cfg.max_daily_loss_usd = 0.0;
    cfg.max_drawdown_pct   = 0.10;
    const ContractSpec spec = nq_spec();
    const PortfolioRiskManager risk(cfg, nullptr);

    // A portfolio in full kill-switch + stale-mark state.
    PortfolioState broke = healthy_state();
    broke.has_stale_mark    = true;
    broke.peak_equity_usd   = 100'000.0;
    broke.equity_usd        = 80'000.0;
    broke.drawdown_usd      = 20'000.0;
    broke.drawdown_pct      = 0.20;
    broke.day_start_equity_usd = 80'000.0;

    {  // pure reduce to flat -> APPROVE despite everything
        const auto d = risk.review(mkorder(Side::Sell, 5), mkctx(broke, spec, 5, 5, 20000.0));
        CHECK(d.verdict == RiskVerdict::Approve);
        CHECK(d.approved_quantity == 5);
        assert_valid(d);
    }
    {  // flip +4 -> would be -2: the close (4) clears, the new short (2) is
       //   blocked by the kill switch -> RESIZE to the flatten quantity
        const auto d = risk.review(mkorder(Side::Sell, 6), mkctx(broke, spec, 4, 4, 20000.0));
        CHECK(d.verdict == RiskVerdict::Resize);
        CHECK(d.approved_quantity == 4);
        CHECK(d.reason_code == "drawdown_kill_switch");
        assert_valid(d);
    }
    {  // flip +2 -> would be -6: RESIZE to SELL 2 (flatten only), never REJECT 8
        const auto d = risk.review(mkorder(Side::Sell, 8), mkctx(broke, spec, 2, 2, 20000.0));
        CHECK(d.verdict == RiskVerdict::Resize);
        CHECK(d.approved_quantity == 2);
        CHECK(d.reason_code == "drawdown_kill_switch");
        assert_valid(d);
    }

    // max_order_contracts via the context overload
    RiskConfig oc = cfg;
    oc.max_drawdown_pct = 0.0;
    oc.max_order_contracts = 3;
    const PortfolioRiskManager risk_oc(oc, nullptr);
    const auto d = risk_oc.review(mkorder(Side::Buy, 10), mkctx(healthy_state(), spec, 0, 0, 20000.0));
    CHECK(d.verdict == RiskVerdict::Resize);
    CHECK(d.approved_quantity == 3);
    CHECK(d.reason_code == "max_order_contracts");
    assert_valid(d);

    // context-free review(order): only structural + order cap
    const auto cf1 = risk_oc.review(mkorder(Side::Buy, 10));
    CHECK(cf1.verdict == RiskVerdict::Resize);
    CHECK(cf1.approved_quantity == 3);
    const auto cf2 = risk_oc.review(mkorder(Side::Buy, 2));
    CHECK(cf2.verdict == RiskVerdict::Approve);
    const auto cf3 = risk_oc.review(mkorder(Side::Buy, 0));
    CHECK(cf3.verdict == RiskVerdict::Reject);
    CHECK(cf3.reason_code == "non_positive_quantity");
    assert_valid(cf1);
    assert_valid(cf2);
    assert_valid(cf3);
}

// ---- flip decomposition (review points 1 & 2; tests A-I) --------------

void test_flip_decomposition() {
    const ContractSpec spec = nq_spec();

    RiskConfig base;
    base.max_contracts_per_symbol = 50;   // generous unless a case tightens it
    base.max_contracts_per_root   = 50;
    base.max_gross_contracts      = 50;
    base.max_margin_utilization_pct = 0.0;
    base.missing_margin = MissingMarginPolicy::TreatAsZero;
    base.stale_mark = StaleMarkPolicy::Ignore;
    base.max_daily_loss_usd = 0.0;
    base.max_drawdown_pct   = 0.0;
    base.max_gross_leverage = 0.0;

    const PortfolioRiskManager risk(base, nullptr);
    const PortfolioState pf = state_with_nq(2, 20000.0, 100'000.0);

    {  // A. long +2, SELL 1 -> approve close 1
        const auto d = risk.review(mkorder(Side::Sell, 1), mkctx(pf, spec, 2, 2, 20000.0));
        CHECK(d.verdict == RiskVerdict::Approve);
        CHECK(d.approved_quantity == 1);
        assert_valid(d);
    }
    {  // B. long +2, SELL 2 -> approve flatten
        const auto d = risk.review(mkorder(Side::Sell, 2), mkctx(pf, spec, 2, 2, 20000.0));
        CHECK(d.verdict == RiskVerdict::Approve);
        CHECK(d.approved_quantity == 2);
        assert_valid(d);
    }
    {  // C. long +2, SELL 3 -> close 2 + open short 1; healthy -> APPROVE the whole 3
        const auto d = risk.review(mkorder(Side::Sell, 3), mkctx(pf, spec, 2, 2, 20000.0));
        CHECK(d.verdict == RiskVerdict::Approve);
        CHECK(d.approved_quantity == 3);
        assert_valid(d);
    }
    {  // D. long +2, SELL 5, only 1 new short fits (gross leverage) -> RESIZE to SELL 3
        RiskConfig lev = base;
        lev.max_gross_leverage = 5.0;   // post |1| = 4x ok; post |2| = 8x fails
        const PortfolioRiskManager risk_lev(lev, nullptr);
        const auto d = risk_lev.review(mkorder(Side::Sell, 5), mkctx(pf, spec, 2, 2, 20000.0));
        CHECK(d.verdict == RiskVerdict::Resize);
        CHECK(d.approved_quantity == 3);           // close 2 + open 1
        CHECK(d.reason_code == "max_gross_leverage");
        assert_valid(d);
    }
    {  // E. mirror: short -2, BUY 5, only 1 new long fits -> RESIZE to BUY 3
        RiskConfig lev = base;
        lev.max_gross_leverage = 5.0;
        const PortfolioRiskManager risk_lev(lev, nullptr);
        const PortfolioState sp = state_with_nq(-2, 20000.0, 100'000.0);
        const auto d = risk_lev.review(mkorder(Side::Buy, 5), mkctx(sp, spec, -2, -2, 20000.0));
        CHECK(d.verdict == RiskVerdict::Resize);
        CHECK(d.approved_quantity == 3);
        CHECK(d.reason_code == "max_gross_leverage");
        assert_valid(d);
    }

    // F. drawdown kill switch: allow flatten, prohibit new opposite position
    {
        RiskConfig dd = base;
        dd.max_drawdown_pct = 0.10;
        const PortfolioRiskManager risk_dd(dd, nullptr);
        PortfolioState p = state_with_nq(2, 20000.0, 80'000.0);
        p.peak_equity_usd = 100'000.0;
        p.drawdown_usd = 20'000.0;
        p.drawdown_pct = 0.20;
        const auto flip = risk_dd.review(mkorder(Side::Sell, 5), mkctx(p, spec, 2, 2, 20000.0));
        CHECK(flip.verdict == RiskVerdict::Resize);
        CHECK(flip.approved_quantity == 2);        // flatten only, never SELL 5, never REJECT
        CHECK(flip.reason_code == "drawdown_kill_switch");
        assert_valid(flip);
        const auto flat = risk_dd.review(mkorder(Side::Sell, 2), mkctx(p, spec, 2, 2, 20000.0));
        CHECK(flat.verdict == RiskVerdict::Approve);
        assert_valid(flat);
    }

    // G. daily-loss kill switch: same behaviour
    {
        RiskConfig dl = base;
        dl.max_daily_loss_usd = 1500.0;
        const PortfolioRiskManager risk_dl(dl, nullptr);
        PortfolioState p = state_with_nq(2, 20000.0, 98'000.0);
        p.day_start_equity_usd = 100'000.0;        // -2000 on the day
        const auto flip = risk_dl.review(mkorder(Side::Sell, 5), mkctx(p, spec, 2, 2, 20000.0));
        CHECK(flip.verdict == RiskVerdict::Resize);
        CHECK(flip.approved_quantity == 2);
        CHECK(flip.reason_code == "daily_loss_limit");
        assert_valid(flip);
        const auto flat = risk_dl.review(mkorder(Side::Sell, 2), mkctx(p, spec, 2, 2, 20000.0));
        CHECK(flat.verdict == RiskVerdict::Approve);
        assert_valid(flat);
    }

    // H. stale mark: allow flatten, prohibit opposite opening exposure
    {
        RiskConfig st = base;
        st.stale_mark = StaleMarkPolicy::RejectRiskIncreasing;
        const PortfolioRiskManager risk_st(st, nullptr);
        PortfolioState p = state_with_nq(2, 20000.0, 100'000.0);
        p.has_stale_mark = true;
        const auto flip = risk_st.review(mkorder(Side::Sell, 5), mkctx(p, spec, 2, 2, 20000.0));
        CHECK(flip.verdict == RiskVerdict::Resize);
        CHECK(flip.approved_quantity == 2);
        CHECK(flip.reason_code == "stale_mark");
        assert_valid(flip);
        const auto flat = risk_st.review(mkorder(Side::Sell, 2), mkctx(p, spec, 2, 2, 20000.0));
        CHECK(flat.verdict == RiskVerdict::Approve);
        assert_valid(flat);
    }

    // I. missing margin: allow flatten, prohibit new opposite exposure
    {
        RiskConfig mm = base;
        mm.missing_margin = MissingMarginPolicy::Reject;   // and no MarginModel
        const PortfolioRiskManager risk_mm(mm, nullptr);
        const auto flip = risk_mm.review(mkorder(Side::Sell, 5), mkctx(pf, spec, 2, 2, 20000.0));
        CHECK(flip.verdict == RiskVerdict::Resize);
        CHECK(flip.approved_quantity == 2);
        CHECK(flip.reason_code == "missing_margin_metadata");
        assert_valid(flip);
        const auto flat = risk_mm.review(mkorder(Side::Sell, 2), mkctx(pf, spec, 2, 2, 20000.0));
        CHECK(flat.verdict == RiskVerdict::Approve);
        assert_valid(flat);
    }
}

// ---- J: net-leverage limit uses magnitude ----------------------------

void test_net_leverage_magnitude() {
    const ContractSpec spec = nq_spec();
    RiskConfig cfg;
    cfg.max_contracts_per_symbol = 50;
    cfg.max_contracts_per_root   = 50;
    cfg.max_gross_contracts      = 50;
    cfg.max_margin_utilization_pct = 0.0;
    cfg.missing_margin = MissingMarginPolicy::TreatAsZero;
    cfg.stale_mark = StaleMarkPolicy::Ignore;
    cfg.max_daily_loss_usd = 0.0;
    cfg.max_drawdown_pct   = 0.0;
    cfg.max_gross_leverage = 0.0;
    cfg.max_net_leverage   = 5.0;
    const PortfolioRiskManager risk(cfg, nullptr);

    {  // large POSITIVE net leverage: long 3 -> add 1 -> |net| 16x -> REJECT
        const PortfolioState p = state_with_nq(3, 20000.0, 100'000.0);
        const auto d = risk.review(mkorder(Side::Buy, 1), mkctx(p, spec, 3, 3, 20000.0));
        CHECK(d.verdict == RiskVerdict::Reject);
        CHECK(d.reason_code == "max_net_leverage");
        assert_valid(d);
    }
    {  // large NEGATIVE net leverage: short 3 -> add 1 short -> |net| 16x -> REJECT
       //   (without abs(), -16 > 5 is false and this would wrongly APPROVE)
        const PortfolioState p = state_with_nq(-3, 20000.0, 100'000.0);
        const auto d = risk.review(mkorder(Side::Sell, 1), mkctx(p, spec, -3, -3, 20000.0));
        CHECK(d.verdict == RiskVerdict::Reject);
        CHECK(d.reason_code == "max_net_leverage");
        assert_valid(d);
    }
    {  // sanity: a small net-short book with huge equity is fine
        PortfolioState p = state_with_nq(-1, 20000.0, 100'000'000.0);
        const auto d = risk.review(mkorder(Side::Sell, 1), mkctx(p, spec, -1, -1, 20000.0));
        CHECK(d.verdict == RiskVerdict::Approve);
        assert_valid(d);
    }
}

// ---- K (risk side): a negative futures price keeps exposure positive ---

void test_negative_price_exposure_risk() {
    // CL long 1 @ mark -20; multiplier 1000. gross notional = |1 * -20 * 1000| =
    // 20,000. A gross-exposure limit must see +20,000, not -20,000.
    ContractSpec cl{.instrument_id = 30, .raw_symbol = "CLZ6", .root_symbol = "CL",
                    .exchange = "XCME", .tick_size = 0.01, .multiplier = 1000.0,
                    .activation_ns = 1, .expiration_ns = kBigExp};
    RiskConfig cfg;
    cfg.max_contracts_per_symbol = 50;
    cfg.max_contracts_per_root   = 50;
    cfg.max_gross_contracts      = 50;
    cfg.max_margin_utilization_pct = 0.0;
    cfg.missing_margin = MissingMarginPolicy::TreatAsZero;
    cfg.stale_mark = StaleMarkPolicy::Ignore;
    cfg.max_daily_loss_usd = 0.0;
    cfg.max_drawdown_pct   = 0.0;
    cfg.max_gross_exposure_usd = 25'000.0;   // admits 1 contract (20k), not 2 (40k)
    const PortfolioRiskManager risk(cfg, nullptr);

    PortfolioState pf = healthy_state();     // flat CL
    auto ctx = mkctx(pf, cl, 0, 0, -20.0);   // negative reference price

    const auto d = risk.review(mkorder(Side::Buy, 3, 30, "CLZ6"), ctx);
    CHECK(d.verdict == RiskVerdict::Resize);
    CHECK(d.approved_quantity == 1);          // gross exposure treated as +20k/contract
    CHECK(d.reason_code == "max_gross_exposure_usd");
    assert_valid(d);
}

// ---- M: drawdown kill switch through a full engine run -----------------

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

MarketBar bar(std::int64_t idx, std::uint32_t instr, double open, double close) {
    const double hi = (open > close ? open : close) + 1.0;
    const double lo = (open < close ? open : close) - 1.0;
    return MarketBar{idx * kMin, instr, open, hi, lo, close, 100};
}

void test_engine_drawdown_kill_switch() {
    ContractRegistry reg;
    reg.add(nq_spec());
    const RegistryActiveContractResolver resolver(reg);

    RiskConfig cfg;
    cfg.missing_margin = MissingMarginPolicy::TreatAsZero;   // no margin metadata in this test
    cfg.max_margin_utilization_pct = 0.0;
    cfg.max_gross_leverage = 0.0;
    cfg.stale_mark = StaleMarkPolicy::RejectRiskIncreasing;
    cfg.max_daily_loss_usd = 0.0;                            // isolate drawdown
    cfg.max_drawdown_pct = 0.10;
    cfg.portfolio.starting_capital_usd = 100'000.0;
    cfg.portfolio.mark_staleness_tolerance_ns = kMin;        // one-bar cadence is fresh
    const PortfolioRiskManager risk(cfg, nullptr);

    EngineConfig ecfg;
    ecfg.end_of_test = EndOfTestPolicy::LeaveOpen;
    const BacktestEngine engine(reg, resolver, risk, ecfg);

    // enter +1 at bar 2 open 20000; price then collapses 1500 points (-30000 =
    // -30% of 100k). At bar 5 the strategy asks for +2 -> a risk-increasing
    // order that the drawdown kill switch must REJECT.
    std::vector<MarketBar> bars{
        bar(1, kNq, 20000.0, 20000.0),
        bar(2, kNq, 20000.0, 20000.0),   // BUY 1 @ 20000
        bar(3, kNq, 20000.0, 19000.0),
        bar(4, kNq, 18800.0, 18500.0),   // deep drawdown
        bar(5, kNq, 18500.0, 18500.0),   // attempted top-up executes here -> rejected
        bar(6, kNq, 18500.0, 18500.0),
    };
    const std::vector<double> script{1, 1, 1, 2, 2, 2};
    const auto r = engine.run(bars, Scripted(script));

    CHECK(r.risk_rejects >= 1);
    bool kill = false;
    for (const auto& d : r.risk_decisions) {
        if (d.reason_code == "drawdown_kill_switch" && d.verdict == RiskVerdict::Reject) kill = true;
    }
    CHECK(kill);
    // the position that was already open is untouched -- never force-closed by risk
    CHECK(r.final_positions.size() == 1);
    CHECK(r.final_positions[0].units == 1);

    // deterministic replay of the whole portfolio view
    const auto r2 = engine.run(bars, Scripted(script));
    CHECK(r2.risk_rejects == r.risk_rejects);
    CHECK_CLOSE(r2.portfolio_at_end.equity_usd, r.portfolio_at_end.equity_usd, 1e-9);
    CHECK_CLOSE(r2.portfolio_at_end.drawdown_pct, r.portfolio_at_end.drawdown_pct, 1e-12);
}

// A flip order through the engine while a kill switch is active: the close leg
// clears, the new opposite exposure does not -- the engine ends FLAT, not short.
void test_engine_flip_resizes_to_flatten_under_kill_switch() {
    ContractRegistry reg;
    reg.add(nq_spec());
    const RegistryActiveContractResolver resolver(reg);

    RiskConfig cfg;
    cfg.missing_margin = MissingMarginPolicy::TreatAsZero;
    cfg.max_margin_utilization_pct = 0.0;
    cfg.max_gross_leverage = 0.0;
    cfg.stale_mark = StaleMarkPolicy::Ignore;
    cfg.max_daily_loss_usd = 0.0;
    cfg.max_drawdown_pct = 0.10;
    cfg.portfolio.starting_capital_usd = 100'000.0;
    cfg.portfolio.mark_staleness_tolerance_ns = kMin;
    const PortfolioRiskManager risk(cfg, nullptr);

    EngineConfig ecfg;
    ecfg.end_of_test = EndOfTestPolicy::LeaveOpen;
    const BacktestEngine engine(reg, resolver, risk, ecfg);

    // long 2 by bar 3; price collapses (drawdown); strategy then targets -3.
    std::vector<MarketBar> bars{
        bar(1, kNq, 20000.0, 20000.0),
        bar(2, kNq, 20000.0, 20000.0),   // BUY 2 -> long 2
        bar(3, kNq, 20000.0, 20000.0),
        bar(4, kNq, 17000.0, 17000.0),   // -30% drawdown
        bar(5, kNq, 17000.0, 17000.0),   // flip order (SELL 5) executes here
        bar(6, kNq, 17000.0, 17000.0),
        bar(7, kNq, 17000.0, 17000.0),
    };
    const std::vector<double> script{2, 2, 2, -3, -3, -3, -3};
    const auto r = engine.run(bars, Scripted(script));

    bool resized_to_flatten = false;
    for (const auto& d : r.risk_decisions) {
        if (d.verdict == RiskVerdict::Resize && d.reason_code == "drawdown_kill_switch") {
            resized_to_flatten = true;
        }
        CHECK(d.verdict != RiskVerdict::Reject || d.reason_code == "drawdown_kill_switch" ||
              d.reason_code == "non_positive_quantity");
    }
    CHECK(resized_to_flatten);
    CHECK(r.final_positions.empty());   // flat -- the new short was never opened
}

}  // namespace

int main() {
    test_per_symbol_cap();
    test_root_and_gross_caps();
    test_leverage_rejection();
    test_margin_rules();
    test_daily_loss_limit();
    test_drawdown_kill_switch();
    test_stale_mark_protection();
    test_reducing_and_order_cap();
    test_flip_decomposition();
    test_net_leverage_magnitude();
    test_negative_price_exposure_risk();
    test_engine_drawdown_kill_switch();
    test_engine_flip_resizes_to_flatten_under_kill_switch();
    return quant::test::summary("quant_risk_manager_tests");
}
