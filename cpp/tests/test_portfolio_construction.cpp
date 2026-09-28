// News Alpha Phase G -- deterministic multi-asset portfolio construction.
//
//   A. unit conversion: futures multiplier vs ETF shares, truncation, below-one-unit
//   B. risk budgeting: inverse-vol legs within a cluster, ERC across clusters
//      (uncorrelated and correlated), target volatility
//   C. hard limits: short restriction, unit cap, leverage, liquidity (floor,
//      unmeasured, participation), sector / asset-class gross, per-cluster
//      risk share, volatility ceiling after a broken hedge, domain permission
//   D. rounding repair (net exposure), turnover (binding and overridden)
//   E. typed refusals (nets to zero) vs malformed inputs (throws)
//   F. determinism under input permutation
//   G. the allocator's executable exposure == PortfolioAccountant's exposure

#include "quant_core/contract.hpp"
#include "quant_core/contract_registry.hpp"
#include "quant_core/fill.hpp"
#include "quant_core/portfolio.hpp"
#include "quant_core/portfolio_construction.hpp"

#include "test_support.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace quant;

ConstructionInstrument inst(const std::string& key, const std::string& domain, double price, double mult,
                            double vol, const std::string& asset_class = "EQUITY",
                            const std::string& sector = "BROAD", double adv = 1e12, int max_units = 0) {
    ConstructionInstrument i;
    i.key = key;
    i.domain = domain;
    i.root_symbol = "R" + std::to_string(std::hash<std::string>{}(key) % 100000);
    i.asset_class = asset_class;
    i.sector = sector;
    i.price = price;
    i.multiplier = mult;
    i.annual_vol = vol;
    i.adv_usd = adv;
    i.max_units = max_units;
    i.decision_ts_ns = 1;
    return i;
}

ConstructionSignal sig(const std::string& id, const std::string& key, const std::string& cluster, int dir,
                       int priority) {
    return ConstructionSignal{id, key, cluster, dir, priority};
}

ConstructionLimits loose(double capital = 1e6, double target = 0.10) {
    ConstructionLimits l;
    l.capital_usd = capital;
    l.target_annual_vol = target;
    l.max_gross_leverage = 100.0;
    l.shorting_allowed = true;
    l.allowed_domains = {"ETF", "FUTURES"};
    return l;
}

void corr(ConstructionInput& in, const std::string& a, const std::string& b, double rho) {
    in.correlations[a < b ? std::make_pair(a, b) : std::make_pair(b, a)] = rho;
}

// Lookups fail the run cleanly (exit 1) when the item is missing -- never a
// null / end-iterator dereference.
const InstrumentAllocation& find_inst(const ConstructionResult& r, const std::string& key) {
    const auto it = std::find_if(r.instruments.begin(), r.instruments.end(),
                                 [&](const auto& a) { return a.input.key == key; });
    if (it == r.instruments.end()) quant::test::fail("instrument present", __FILE__, __LINE__, key);
    return *it;
}

const SignalAllocation& find_sig(const ConstructionResult& r, const std::string& id) {
    const auto it = std::find_if(r.signals.begin(), r.signals.end(),
                                 [&](const auto& s) { return s.input.signal_id == id; });
    if (it == r.signals.end()) quant::test::fail("signal present", __FILE__, __LINE__, id);
    return *it;
}

const ConstraintOutcome* find_con(const ConstructionResult& r, const std::string& name, const std::string& stage,
                                  const std::string& scope = "") {
    for (const auto& c : r.constraints) {
        if (c.name == name && c.stage == stage && (scope.empty() || c.scope == scope)) return &c;
    }
    quant::test::fail("constraint present", __FILE__, __LINE__, name + " / " + stage + " / " + scope);
    return nullptr;
}

template <class F>
bool throws_invalid(F f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

// ---- A. unit conversion ----------------------------------------------------

void test_etf_shares() {
    ConstructionInput in;
    in.instruments = {inst("ETF:AAA", "ETF", 100.0, 1.0, 0.20)};
    in.signals = {sig("s1", "ETF:AAA", "c1", 1, 1)};
    in.limits = loose();
    const auto r = construct_portfolio(in);
    const auto& a = find_inst(r, "ETF:AAA");
    CHECK(r.status == "CONSTRUCTED");
    CHECK_CLOSE(a.target_weight, 0.5, 1e-12);            // 0.10 / 0.20
    CHECK_CLOSE(a.input.unit_notional_usd(), 100.0, 1e-12);
    CHECK(a.units == 5000);                              // $500,000 / $100 per share
    CHECK_CLOSE(a.executable_notional_usd, 500000.0, 1e-6);
    CHECK_CLOSE(r.executable.annual_vol, 0.10, 1e-12);
    CHECK_CLOSE(find_sig(r, "s1").risk_contribution, 0.10, 1e-12);
}

void test_futures_multiplier_and_truncation() {
    ConstructionInput in;
    in.instruments = {inst("FUTURES:NQ", "FUTURES", 11000.0, 20.0, 0.25)};
    in.signals = {sig("s1", "FUTURES:NQ", "c1", 1, 1)};
    in.limits = loose();
    const auto r = construct_portfolio(in);
    const auto& a = find_inst(r, "FUTURES:NQ");
    CHECK_CLOSE(a.input.unit_notional_usd(), 220000.0, 1e-9);  // price x multiplier, never price alone
    CHECK_CLOSE(a.target_notional_usd, 400000.0, 1e-6);        // 0.10 / 0.25 x $1M
    CHECK(a.units == 1);                                       // 1.82 contracts -> 1 (toward zero)
    CHECK_CLOSE(a.executable_notional_usd, 220000.0, 1e-9);
    CHECK_CLOSE(a.rounding_residual_usd, 180000.0, 1e-6);
    CHECK_CLOSE(a.unit_risk_fraction, 0.055, 1e-12);           // one contract = 5.5% of capital in vol
    CHECK(!a.below_one_unit);
    CHECK(r.executable.annual_vol < r.target_annual_vol);      // rounding only ever lowers risk here
}

void test_below_one_unit() {
    ConstructionInput in;
    in.instruments = {inst("FUTURES:NQ", "FUTURES", 11000.0, 20.0, 0.25)};
    in.signals = {sig("s1", "FUTURES:NQ", "c1", 1, 1)};
    in.limits = loose(100000.0);
    const auto r = construct_portfolio(in);
    const auto& a = find_inst(r, "FUTURES:NQ");
    CHECK(a.units == 0);
    CHECK(a.below_one_unit);                                   // $40,000 target < one $220,000 contract
    CHECK(r.status == "NO_EXECUTABLE_POSITION");
    CHECK(find_sig(r, "s1").allocated);                        // allocated -- just not executable
}

// ---- B. risk budgeting -------------------------------------------------------

void test_inverse_vol_within_cluster() {
    ConstructionInput in;
    in.instruments = {inst("ETF:A", "ETF", 100.0, 1.0, 0.10), inst("ETF:B", "ETF", 100.0, 1.0, 0.20)};
    in.signals = {sig("sa", "ETF:A", "c1", 1, 1), sig("sb", "ETF:B", "c1", 1, 2)};
    corr(in, "ETF:A", "ETF:B", 0.5);
    in.limits = loose();
    const auto r = construct_portfolio(in);
    const double wa = find_inst(r, "ETF:A").target_weight;
    const double wb = find_inst(r, "ETF:B").target_weight;
    CHECK_CLOSE(wa * 0.10, wb * 0.20, 1e-12);                  // equal standalone risk per member
    CHECK_CLOSE(r.target.annual_vol, 0.10, 1e-12);
    CHECK(r.clusters.size() == 1 && r.clusters[0].signal_ids.size() == 2);
}

void test_erc_uncorrelated_clusters() {
    ConstructionInput in;
    in.instruments = {inst("ETF:A", "ETF", 100.0, 1.0, 0.10), inst("ETF:B", "ETF", 100.0, 1.0, 0.30)};
    in.signals = {sig("sa", "ETF:A", "ca", 1, 1), sig("sb", "ETF:B", "cb", 1, 2)};
    corr(in, "ETF:A", "ETF:B", 0.0);
    in.limits = loose();
    const auto r = construct_portfolio(in);
    CHECK_CLOSE(find_inst(r, "ETF:A").target_weight, 0.10 / (0.10 * std::sqrt(2.0)), 1e-9);
    CHECK_CLOSE(find_inst(r, "ETF:B").target_weight, 0.10 / (0.30 * std::sqrt(2.0)), 1e-9);
    CHECK_CLOSE(r.clusters[0].target_risk_contribution, r.clusters[1].target_risk_contribution, 1e-12);
    CHECK_CLOSE(r.target.effective_exposures, 2.0, 1e-9);
}

void test_erc_correlated_clusters() {
    ConstructionInput in;
    in.instruments = {inst("ETF:A", "ETF", 100.0, 1.0, 0.10), inst("FUTURES:B", "FUTURES", 4000.0, 50.0, 0.20),
                      inst("ETF:C", "ETF", 50.0, 1.0, 0.30)};
    in.signals = {sig("sa", "ETF:A", "ca", 1, 1), sig("sb", "FUTURES:B", "cb", -1, 2), sig("sc", "ETF:C", "cc", 1, 3)};
    corr(in, "ETF:A", "FUTURES:B", 0.6);
    corr(in, "ETF:A", "ETF:C", 0.2);
    corr(in, "ETF:C", "FUTURES:B", -0.3);
    in.limits = loose(1e8);
    const auto r = construct_portfolio(in);
    const double t = r.target.annual_vol;
    CHECK_CLOSE(t, 0.10, 1e-12);
    for (const auto& c : r.clusters) CHECK_CLOSE(c.target_risk_contribution / t, 1.0 / 3.0, 1e-9);
    // Covariance matters: the correlated pair gets less than inverse-vol would give.
    CHECK(find_inst(r, "FUTURES:B").target_weight < 0.0);      // direction preserved
    CHECK(r.erc_max_deviation <= 1e-9);
}

// ---- C. hard limits -------------------------------------------------------------

void test_short_restriction() {
    ConstructionInput in;
    in.instruments = {inst("ETF:A", "ETF", 100.0, 1.0, 0.10)};
    in.signals = {sig("s1", "ETF:A", "c1", -1, 1)};
    in.limits = loose();
    in.limits.shorting_allowed = false;
    auto r = construct_portfolio(in);
    CHECK(find_sig(r, "s1").reason == "SHORT_NOT_PERMITTED");
    CHECK(r.status == "NO_ALLOCATABLE_SIGNAL");
    in.limits.shorting_allowed = true;
    r = construct_portfolio(in);
    CHECK(find_inst(r, "ETF:A").units == -10000);
}

void test_gross_leverage_cap() {
    ConstructionInput in;
    in.instruments = {inst("ETF:SHY", "ETF", 80.0, 1.0, 0.02, "RATES", "RATES_FRONT")};
    in.signals = {sig("s1", "ETF:SHY", "c1", 1, 1)};
    in.limits = loose();
    in.limits.max_gross_leverage = 1.0;
    const auto r = construct_portfolio(in);
    CHECK_CLOSE(r.target.gross_exposure, 5.0, 1e-9);            // 10% / 2% vol -> 5x unconstrained
    CHECK_CLOSE(r.constrained.gross_exposure, 1.0, 1e-9);
    const auto* c = find_con(r, "gross_leverage", "allocation");
    CHECK(c != nullptr && c->status == ConstraintStatus::Binding);
    CHECK(find_inst(r, "ETF:SHY").units == 12500);              // $1M / $80
    CHECK_CLOSE(r.executable.annual_vol, 0.02, 1e-12);          // target volatility is not reached -- reported
}

void test_instrument_concentration() {
    ConstructionInput in;
    in.instruments = {inst("ETF:SHY", "ETF", 80.0, 1.0, 0.02, "RATES", "RATES_FRONT"),
                      inst("ETF:SPY", "ETF", 400.0, 1.0, 0.20)};
    in.signals = {sig("shy", "ETF:SHY", "c1", 1, 1), sig("spy", "ETF:SPY", "c2", 1, 2)};
    corr(in, "ETF:SHY", "ETF:SPY", 0.0);
    in.limits = loose();
    in.limits.max_gross_leverage = 2.0;
    in.limits.max_instrument_gross_share = 0.4;  // no instrument above 0.8x capital
    const auto r = construct_portfolio(in);
    CHECK(find_inst(r, "ETF:SHY").target_weight > 0.8);           // a 2%-vol fund wants a huge notional
    CHECK_CLOSE(find_inst(r, "ETF:SHY").constrained_weight, 0.8, 1e-12);
    CHECK(find_inst(r, "ETF:SHY").units == 10000);                // $800,000 / $80
    const auto* c = find_con(r, "instrument_concentration", "allocation");
    CHECK(c->status == ConstraintStatus::Binding && c->affected.size() == 1 && c->affected[0] == "ETF:SHY");
    CHECK_CLOSE(find_inst(r, "ETF:SPY").constrained_weight, find_inst(r, "ETF:SPY").target_weight, 1e-12);
}

void test_liquidity() {
    ConstructionInput in;
    in.instruments = {inst("ETF:THIN", "ETF", 100.0, 1.0, 0.10, "EQUITY", "BROAD", 5e6),
                      inst("ETF:UNMEASURED", "ETF", 100.0, 1.0, 0.10, "EQUITY", "BROAD", -1.0),
                      inst("ETF:OK", "ETF", 100.0, 1.0, 0.10, "EQUITY", "BROAD", 1e7)};
    in.signals = {sig("thin", "ETF:THIN", "c1", 1, 1), sig("unm", "ETF:UNMEASURED", "c2", 1, 2),
                  sig("ok", "ETF:OK", "c3", 1, 3)};
    corr(in, "ETF:OK", "ETF:THIN", 0.0);
    corr(in, "ETF:OK", "ETF:UNMEASURED", 0.0);
    corr(in, "ETF:THIN", "ETF:UNMEASURED", 0.0);
    in.limits = loose();
    in.limits.min_adv_usd = 6e6;
    in.limits.max_adv_participation = 0.01;
    const auto r = construct_portfolio(in);
    CHECK(find_inst(r, "ETF:THIN").reason == "BELOW_MIN_LIQUIDITY");
    CHECK(find_inst(r, "ETF:UNMEASURED").reason == "LIQUIDITY_NOT_MEASURED");  // fail closed
    CHECK(find_sig(r, "thin").reason == "INSTRUMENT_NOT_ADMITTED");
    const auto& ok = find_inst(r, "ETF:OK");
    CHECK(ok.units == 1000);                                     // 1% of $10M ADV = $100,000
    CHECK_CLOSE(ok.adv_participation, 0.01, 1e-12);
    const auto* c = find_con(r, "liquidity_participation", "allocation");
    CHECK(c != nullptr && c->status == ConstraintStatus::Binding && c->affected.size() == 1);
}

void test_sector_and_asset_class_caps() {
    ConstructionInput in;
    in.instruments = {inst("ETF:A", "ETF", 100.0, 1.0, 0.10, "EQUITY", "TECH"),
                      inst("ETF:B", "ETF", 100.0, 1.0, 0.10, "EQUITY", "TECH"),
                      inst("ETF:C", "ETF", 100.0, 1.0, 0.10, "RATES", "LONG")};
    in.signals = {sig("a", "ETF:A", "ca", 1, 1), sig("b", "ETF:B", "cb", 1, 2), sig("c", "ETF:C", "cc", 1, 3)};
    corr(in, "ETF:A", "ETF:B", 0.0);
    corr(in, "ETF:A", "ETF:C", 0.0);
    corr(in, "ETF:B", "ETF:C", 0.0);
    in.limits = loose();
    in.limits.max_gross_leverage = 2.0;
    in.limits.max_sector_gross_share = 0.4;        // TECH <= 0.8 of capital
    in.limits.max_asset_class_gross_share = 0.35;  // any class <= 0.7 of capital
    const auto r = construct_portfolio(in);
    const double tech = std::abs(find_inst(r, "ETF:A").constrained_weight) +
                        std::abs(find_inst(r, "ETF:B").constrained_weight);
    CHECK(tech <= 0.7 + 1e-9);                     // the tighter asset-class cap wins for EQUITY
    CHECK(std::abs(find_inst(r, "ETF:C").constrained_weight) <= 0.7 + 1e-9);
    CHECK(find_con(r, "sector_gross", "allocation", "sector:TECH")->status == ConstraintStatus::Binding);
    CHECK(find_con(r, "asset_class_gross", "allocation", "asset_class:EQUITY")->status == ConstraintStatus::Binding);
    CHECK_CLOSE(find_inst(r, "ETF:A").constrained_weight, find_inst(r, "ETF:B").constrained_weight, 1e-12);
}

void test_cluster_risk_share_single_exposure() {
    ConstructionInput in;
    in.instruments = {inst("ETF:A", "ETF", 100.0, 1.0, 0.20)};
    in.signals = {sig("a", "ETF:A", "c1", 1, 1)};
    in.limits = loose();
    in.limits.max_cluster_risk_share = 0.5;        // one exposure may use half the risk budget
    const auto r = construct_portfolio(in);
    CHECK_CLOSE(r.constrained.annual_vol, 0.05, 1e-12);
    CHECK(find_con(r, "cluster_risk_share", "allocation")->status == ConstraintStatus::Binding);
    CHECK(find_inst(r, "ETF:A").units == 2500);
}

void test_unit_cap() {
    ConstructionInput in;
    in.instruments = {inst("FUTURES:NQ", "FUTURES", 11000.0, 20.0, 0.25, "EQUITY", "GROWTH", 1e12, 3)};
    in.signals = {sig("s", "FUTURES:NQ", "c1", 1, 1)};
    in.limits = loose(1e7);
    const auto r = construct_portfolio(in);
    CHECK(find_inst(r, "FUTURES:NQ").units == 3);  // 18 contracts wanted, capped at 3 -- exactly, not 2
    CHECK(find_con(r, "unit_cap", "allocation")->status == ConstraintStatus::Binding);
}

void test_volatility_ceiling_after_broken_hedge() {
    ConstructionInput in;
    in.instruments = {inst("ETF:A", "ETF", 100.0, 1.0, 0.10, "EQUITY", "BROAD", 1e12, 5000),
                      inst("ETF:B", "ETF", 100.0, 1.0, 0.10)};
    in.signals = {sig("a", "ETF:A", "ca", 1, 1), sig("b", "ETF:B", "cb", -1, 2)};
    corr(in, "ETF:A", "ETF:B", 0.9);
    in.limits = loose();
    const auto r = construct_portfolio(in);
    CHECK_CLOSE(r.target.annual_vol, 0.10, 1e-12);
    // Capping only the long leg breaks the hedge; the ceiling pulls the book back to 10%.
    CHECK(find_con(r, "volatility_ceiling", "allocation")->status == ConstraintStatus::Binding);
    CHECK(r.constrained.annual_vol <= 0.10 + 1e-12);
    CHECK(r.executable.annual_vol <= 0.10 + 1e-12);
}

void test_domain_permission() {
    ConstructionInput in;
    in.instruments = {inst("FUTURES:CL", "FUTURES", 80.0, 1000.0, 0.35), inst("ETF:A", "ETF", 100.0, 1.0, 0.10)};
    in.signals = {sig("f", "FUTURES:CL", "c1", 1, 1), sig("e", "ETF:A", "c2", 1, 2)};
    corr(in, "ETF:A", "FUTURES:CL", 0.1);
    in.limits = loose();
    in.limits.allowed_domains = {"ETF"};
    const auto r = construct_portfolio(in);
    CHECK(find_inst(r, "FUTURES:CL").reason == "DOMAIN_NOT_ALLOWED");
    CHECK(find_inst(r, "FUTURES:CL").units == 0);
    CHECK(find_inst(r, "ETF:A").units > 0);
}

// ---- D. rounding repair, turnover ---------------------------------------------

void test_net_exposure_rounding_repair() {
    ConstructionInput in;
    in.instruments = {inst("ETF:A", "ETF", 1000.0, 1.0, 0.10), inst("FUTURES:B", "FUTURES", 3000.0, 100.0, 0.10)};
    in.signals = {sig("a", "ETF:A", "ca", 1, 1), sig("b", "FUTURES:B", "cb", -1, 2)};
    corr(in, "ETF:A", "FUTURES:B", 0.0);
    in.limits = loose();
    in.limits.max_net_exposure = 0.05;
    const auto r = construct_portfolio(in);
    // Continuous book is market-neutral (net 0); truncation leaves 707 shares vs -2 contracts (net +0.107).
    CHECK(find_con(r, "net_exposure", "allocation")->status == ConstraintStatus::Satisfied);
    const auto* repair = find_con(r, "net_exposure", "rounding");
    CHECK(repair != nullptr && repair->status == ConstraintStatus::Binding);
    CHECK_CLOSE(repair->before, 0.107, 1e-9);
    CHECK(find_inst(r, "ETF:A").units == 650);
    CHECK(find_inst(r, "FUTURES:B").units == -2);
    CHECK(r.repair_units_removed == 57);
    CHECK(std::abs(r.executable.net_exposure) <= 0.05 + 1e-12);
}

void test_turnover() {
    ConstructionInput in;
    in.instruments = {inst("ETF:A", "ETF", 100.0, 1.0, 0.20)};
    in.signals = {sig("a", "ETF:A", "c1", 1, 1)};
    in.limits = loose();
    in.limits.max_turnover = 0.2;
    in.previous_units = {{"ETF:A", 1000}};                     // 0.1 of capital held
    auto r = construct_portfolio(in);
    CHECK_CLOSE(r.turnover_target, 0.4, 1e-12);                // wants 0.5
    CHECK(find_inst(r, "ETF:A").units == 3000);                // moves 0.2 of capital toward it
    CHECK(find_con(r, "turnover", "allocation")->status == ConstraintStatus::Binding);

    in.previous_units = {{"ETF:A", 20000}};                    // 2.0x held, but the cap is 1.0x
    in.limits.max_gross_leverage = 1.0;
    r = construct_portfolio(in);
    CHECK(find_inst(r, "ETF:A").units == 5000);                // hard limits win over turnover (target 0.5)
    CHECK(find_con(r, "turnover", "allocation")->status == ConstraintStatus::Overridden);
}

// ---- E. refusals vs malformed input ----------------------------------------------

void test_nets_to_zero_is_a_typed_refusal() {
    ConstructionInput in;
    in.instruments = {inst("ETF:A", "ETF", 100.0, 1.0, 0.10)};
    in.signals = {sig("up", "ETF:A", "c1", 1, 1), sig("down", "ETF:A", "c1", -1, 2)};
    in.limits = loose();
    const auto r = construct_portfolio(in);
    CHECK(r.status == "NO_ALLOCATABLE_SIGNAL");
    CHECK(r.clusters[0].reason == "CLUSTER_NETS_TO_ZERO");
    CHECK(find_sig(r, "up").reason == "CLUSTER_NETS_TO_ZERO");
}

void test_malformed_inputs_throw() {
    ConstructionInput base;
    base.instruments = {inst("ETF:A", "ETF", 100.0, 1.0, 0.10), inst("ETF:B", "ETF", 100.0, 1.0, 0.10),
                        inst("ETF:C", "ETF", 100.0, 1.0, 0.10)};
    base.signals = {sig("a", "ETF:A", "ca", 1, 1), sig("b", "ETF:B", "cb", 1, 2), sig("c", "ETF:C", "cc", 1, 3)};
    corr(base, "ETF:A", "ETF:B", 0.9);
    corr(base, "ETF:A", "ETF:C", 0.9);
    corr(base, "ETF:B", "ETF:C", 0.0);
    base.limits = loose();
    CHECK(throws_invalid([&] { construct_portfolio(base); }));  // not positive semi-definite

    auto missing = base;
    missing.correlations.erase({"ETF:B", "ETF:C"});
    CHECK(throws_invalid([&] { construct_portfolio(missing); }));

    auto dir = base;
    corr(dir, "ETF:A", "ETF:C", 0.1);
    corr(dir, "ETF:A", "ETF:B", 0.1);
    CHECK(!throws_invalid([&] { construct_portfolio(dir); }));
    dir.signals[0].direction = 0;
    CHECK(throws_invalid([&] { construct_portfolio(dir); }));

    auto unknown = base;
    unknown.signals.push_back(sig("x", "ETF:NOPE", "cx", 1, 9));
    CHECK(throws_invalid([&] { construct_portfolio(unknown); }));

    auto dup = base;
    dup.instruments.push_back(inst("ETF:A", "ETF", 100.0, 1.0, 0.10));
    CHECK(throws_invalid([&] { construct_portfolio(dup); }));

    auto nolev = base;
    nolev.limits.max_gross_leverage = 0.0;                      // an unstated cap is resolved upstream
    CHECK(throws_invalid([&] { construct_portfolio(nolev); }));

    auto range = base;
    range.correlations[{"ETF:A", "ETF:B"}] = 1.5;
    CHECK(throws_invalid([&] { construct_portfolio(range); }));
}

void test_degenerate_risk_model_is_a_typed_refusal() {
    // Two exposures that hedge each other exactly: no ERC solution exists.
    ConstructionInput in;
    in.instruments = {inst("ETF:A", "ETF", 100.0, 1.0, 0.10), inst("ETF:B", "ETF", 100.0, 1.0, 0.10)};
    in.signals = {sig("a", "ETF:A", "ca", 1, 1), sig("b", "ETF:B", "cb", 1, 2)};
    corr(in, "ETF:A", "ETF:B", -1.0);
    in.limits = loose();
    auto r = construct_portfolio(in);
    CHECK(r.status == "DEGENERATE_RISK_MODEL");
    CHECK(find_sig(r, "a").reason == "DEGENERATE_RISK_MODEL" && !find_sig(r, "a").allocated);
    CHECK(find_inst(r, "ETF:A").units == 0 && find_inst(r, "ETF:B").units == 0);

    // The same hedge from two signal FAMILIES on one instrument, in different clusters.
    ConstructionInput one;
    one.instruments = {inst("FUTURES:NQ", "FUTURES", 11000.0, 20.0, 0.25)};
    one.signals = {sig("mom", "FUTURES:NQ", "momentum", 1, 1), sig("rev", "FUTURES:NQ", "reversion", -1, 2)};
    one.limits = loose();
    r = construct_portfolio(one);
    CHECK(r.status == "DEGENERATE_RISK_MODEL");
    CHECK(r.clusters.size() == 2 && !r.clusters[0].allocated && !r.clusters[1].allocated);
}

// ---- F. determinism ----------------------------------------------------------------

void test_permutation_invariance() {
    ConstructionInput in;
    in.instruments = {inst("ETF:A", "ETF", 101.0, 1.0, 0.11, "EQUITY", "TECH", 1e9),
                      inst("FUTURES:B", "FUTURES", 4012.25, 50.0, 0.19, "EQUITY", "BROAD", 1e10),
                      inst("ETF:C", "ETF", 47.3, 1.0, 0.07, "RATES", "LONG", 1e9),
                      inst("FUTURES:D", "FUTURES", 1820.0, 100.0, 0.16, "COMMODITY", "METALS", 1e10)};
    in.signals = {sig("s1", "ETF:A", "c1", 1, 1), sig("s2", "FUTURES:B", "c1", 1, 4), sig("s3", "ETF:C", "c2", 1, 2),
                  sig("s4", "FUTURES:D", "c3", 1, 3), sig("s5", "ETF:A", "c1", 1, 5)};
    corr(in, "ETF:A", "FUTURES:B", 0.85);
    corr(in, "ETF:A", "ETF:C", -0.2);
    corr(in, "ETF:A", "FUTURES:D", 0.1);
    corr(in, "ETF:C", "FUTURES:B", -0.25);
    corr(in, "FUTURES:B", "FUTURES:D", 0.05);
    corr(in, "ETF:C", "FUTURES:D", 0.3);
    in.limits = loose(5e6);
    in.limits.max_gross_leverage = 1.5;
    in.limits.max_instrument_gross_share = 0.5;
    in.limits.max_sector_gross_share = 0.6;
    in.limits.max_cluster_risk_share = 0.5;
    in.limits.max_adv_participation = 0.01;
    const auto a = construct_portfolio(in);

    auto shuffled = in;
    std::reverse(shuffled.instruments.begin(), shuffled.instruments.end());
    std::rotate(shuffled.signals.begin(), shuffled.signals.begin() + 2, shuffled.signals.end());
    const auto b = construct_portfolio(shuffled);

    CHECK(a.instruments.size() == b.instruments.size());
    for (std::size_t i = 0; i < a.instruments.size(); ++i) {
        CHECK(a.instruments[i].input.key == b.instruments[i].input.key);
        CHECK(a.instruments[i].units == b.instruments[i].units);
        CHECK(a.instruments[i].constrained_weight == b.instruments[i].constrained_weight);
    }
    for (std::size_t i = 0; i < a.signals.size(); ++i) {
        CHECK(a.signals[i].input.signal_id == b.signals[i].input.signal_id);
        CHECK(a.signals[i].risk_contribution == b.signals[i].risk_contribution);
    }
    CHECK(a.constraints.size() == b.constraints.size());
    CHECK(a.executable.annual_vol == b.executable.annual_vol);
    CHECK(a.status == "CONSTRUCTED");
    // Same-instrument members of one cluster net on the instrument; attribution sums back.
    double legs = 0.0;
    for (const auto& s : a.signals) {
        if (s.input.instrument_key == "ETF:A") legs += s.executable_weight;
    }
    CHECK_CLOSE(legs, find_inst(a, "ETF:A").executable_weight, 1e-12);
    double rc = 0.0;
    for (const auto& s : a.signals) rc += s.risk_contribution;
    CHECK_CLOSE(rc, a.executable.annual_vol, 1e-12);            // Euler: contributions sum to volatility
}

// ---- G. accounting agreement -------------------------------------------------------

void test_exposure_matches_portfolio_accountant() {
    ConstructionInput in;
    in.instruments = {inst("FUTURES:NQ", "FUTURES", 11000.0, 20.0, 0.25, "EQUITY", "GROWTH"),
                      inst("ETF:TLT", "ETF", 99.5, 1.0, 0.15, "RATES", "LONG")};
    in.signals = {sig("nq", "FUTURES:NQ", "c1", 1, 1), sig("tlt", "ETF:TLT", "c2", -1, 2)};
    corr(in, "ETF:TLT", "FUTURES:NQ", -0.3);
    in.limits = loose(2e6);
    const auto r = construct_portfolio(in);
    const auto& nq = find_inst(r, "FUTURES:NQ");
    const auto& tlt = find_inst(r, "ETF:TLT");
    CHECK(nq.units > 0 && tlt.units < 0);

    ContractRegistry reg;
    reg.add(ContractSpec{.instrument_id = 1, .raw_symbol = "NQH3", .root_symbol = "NQ", .exchange = "XCME",
                         .tick_size = 0.25, .multiplier = 20.0, .activation_ns = 1, .expiration_ns = 1LL << 62});
    reg.add(ContractSpec{.instrument_id = 2, .raw_symbol = "TLT", .root_symbol = "ETLT", .exchange = "XNAS",
                         .tick_size = 0.01, .multiplier = 1.0, .activation_ns = 1, .expiration_ns = 1LL << 62});
    PortfolioConfig pc;
    pc.starting_capital_usd = 2e6;
    pc.mark_staleness_tolerance_ns = 1LL << 60;
    PortfolioAccountant pf(pc, nullptr);
    auto fill = [&](std::uint32_t id, int units, double px) {
        FillRequest req;
        req.fill_id = id;
        req.order_id = id;
        req.ts_fill_ns = 10;
        req.instrument_id = id;
        req.side = units > 0 ? Side::Buy : Side::Sell;
        req.quantity = std::abs(units);
        req.price = px;
        return make_fill(reg, req);
    };
    pf.apply_fill(fill(1, nq.units, nq.input.price), "NQ", "signal");
    pf.apply_fill(fill(2, tlt.units, tlt.input.price), "ETLT", "signal");
    pf.observe_mark(1, nq.input.price, 10);
    pf.observe_mark(2, tlt.input.price, 10);
    const auto s = pf.snapshot(10);
    CHECK_CLOSE(s.gross_exposure_usd, std::abs(nq.executable_notional_usd) + std::abs(tlt.executable_notional_usd),
                1e-6);
    CHECK_CLOSE(s.net_exposure_usd, nq.executable_notional_usd + tlt.executable_notional_usd, 1e-6);
    CHECK_CLOSE(s.gross_leverage, r.executable.gross_exposure, 1e-12);
}

}  // namespace

int main() {
    // An allocator invariant (std::logic_error) or an unexpected exception is a
    // FAILED run, reported -- never an abort.
    try {
        test_etf_shares();
        test_futures_multiplier_and_truncation();
        test_below_one_unit();
        test_inverse_vol_within_cluster();
        test_erc_uncorrelated_clusters();
        test_erc_correlated_clusters();
        test_short_restriction();
        test_gross_leverage_cap();
        test_instrument_concentration();
        test_liquidity();
        test_sector_and_asset_class_caps();
        test_cluster_risk_share_single_exposure();
        test_unit_cap();
        test_volatility_ceiling_after_broken_hedge();
        test_domain_permission();
        test_net_exposure_rounding_repair();
        test_turnover();
        test_nets_to_zero_is_a_typed_refusal();
        test_malformed_inputs_throw();
        test_degenerate_risk_model_is_a_typed_refusal();
        test_permutation_invariance();
        test_exposure_matches_portfolio_accountant();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "CHECK FAILED: uncaught exception: %s\n", e.what());
        return 1;
    }
    return quant::test::summary("quant_portfolio_construction_tests");
}
