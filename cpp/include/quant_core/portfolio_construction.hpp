#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace quant {

// ============================================================================
// Deterministic multi-asset portfolio construction (News Alpha Phase G)
// ============================================================================
//
// Turns a set of already-SELECTED signals (each: one instrument, one direction,
// one exposure cluster) plus already-ESTIMATED risk statistics into integer
// target units per instrument, under explicit hard limits. Method
// `portfolio-construction/1`:
//
//   1. admission   -- domain permission, usable price / economics / volatility,
//                     liquidity floor, shorting (typed reasons, never throws)
//   2. legs        -- each signal of a cluster carries an equal standalone risk:
//                     leg = direction / (n_cluster * annual_vol)
//   3. ERC         -- equal risk contribution ACROSS clusters on the composite
//                     covariance (deterministic cyclical coordinate descent)
//   4. target vol  -- the whole book scaled to `target_annual_vol`
//   5. constraints -- all scale-downs, hard limits first, turnover last:
//                     short restriction, unit cap, instrument concentration,
//                     liquidity participation, sector gross, asset-class gross,
//                     gross leverage, net exposure, per-cluster risk share,
//                     volatility ceiling; then turnover (hard limits re-applied)
//   6. units       -- truncate toward zero (never exceeds a limit), then repair
//                     the limits truncation CAN break (net exposure, cluster risk
//                     share, volatility ceiling) by removing one unit at a time
//                     from the largest contributor.
//
// Ranking (`priority`) orders and breaks ties; it is never a weight. Every
// quantity is in capital fractions / USD: a futures contract and a share are
// never "one unit" of anything -- `unit_notional_usd = price * multiplier`
// (ContractSpec semantics: USD PnL per 1.0 price move per unit; 1.0 per share).
//
// What this does NOT do: choose signals, estimate volatility/correlation, fill,
// account. Selection and statistics are Python research orchestration; fills
// and accounting are the unchanged BacktestEngine / PortfolioAccountant. The
// executable exposure reported here is the same arithmetic the
// PortfolioAccountant applies (|units| x price x multiplier) -- pinned by
// cpp/tests/test_portfolio_construction.cpp.
//
// Determinism: inputs are canonicalised (instruments by key, signals by
// (priority, id), clusters by (best priority, id)); std::map everywhere; fixed
// iteration limits; no clock, RNG or hash order. The same input in any order
// yields the identical result.

inline constexpr const char* kPortfolioConstructionMethod = "portfolio-construction/1";

struct ConstructionInstrument {
    std::string  key;             // unique, e.g. "FUTURES:NQ", "ETF:QQQ"
    std::string  domain;          // FUTURES | ETF | EQUITY
    std::string  root_symbol;     // execution root the target schedule names
    std::string  asset_class;     // internal risk taxonomy (EQUITY, RATES, ...)
    std::string  sector;
    double       price{0.0};      // RAW reference price at the decision instant
    double       multiplier{0.0}; // USD per 1.0 price move per unit
    double       annual_vol{0.0}; // annualized volatility of the unit's return
    double       adv_usd{-1.0};   // median daily traded notional; < 0 == not measured
    int          max_units{0};    // |units| cap; 0 == none
    std::int64_t decision_ts_ns{0};

    double unit_notional_usd() const noexcept { return price * multiplier; }
};

struct ConstructionSignal {
    std::string signal_id;
    std::string instrument_key;
    std::string cluster_id;
    int         direction{0};  // +1 / -1 only
    int         priority{0};   // rank: ordering and tie-breaks only, never a weight
};

// All exposure caps are fractions of capital; 0 disables a cap (except the
// required capital, target volatility and gross leverage).
struct ConstructionLimits {
    double capital_usd{0.0};
    double target_annual_vol{0.0};
    double max_gross_leverage{0.0};           // sum |w|; REQUIRED > 0
    double max_net_exposure{0.0};             // |sum w|
    double max_instrument_gross_share{0.0};   // |w_k| <= share * max_gross_leverage
    double max_sector_gross_share{0.0};       // sum_sector |w| <= share * max_gross_leverage
    double max_asset_class_gross_share{0.0};  // sum_class |w| <= share * max_gross_leverage
    double max_cluster_risk_share{0.0};       // cluster risk contribution <= share * target vol
    double max_adv_participation{0.0};        // |notional| <= share * adv_usd
    double min_adv_usd{0.0};                  // admission floor
    double max_turnover{0.0};                 // sum |w - w_prev|
    bool   shorting_allowed{false};
    std::vector<std::string> allowed_domains;
};

struct ConstructionInput {
    std::vector<ConstructionInstrument> instruments;
    std::vector<ConstructionSignal>     signals;
    // Pairwise correlation of instrument returns, keyed by (a, b) with a < b.
    // Every pair of instruments referenced by a signal is required.
    std::map<std::pair<std::string, std::string>, double> correlations;
    ConstructionLimits limits;
    // Optional current holdings (instrument key -> signed units), for turnover.
    std::map<std::string, int> previous_units;
};

// OVERRIDDEN: only the turnover limit -- a hard risk limit required a larger
// trade than the turnover limit allows, and hard limits take precedence.
enum class ConstraintStatus : int { NotApplicable = 0, Satisfied = 1, Binding = 2, Overridden = 3 };

// One constraint's outcome. `before` / `after` / `limit` are in the
// constraint's own unit (capital fraction, ADV fraction, cap utilisation, ...).
struct ConstraintOutcome {
    std::string              name;
    std::string              scope;   // "portfolio" | "each instrument" | "sector:X" | "asset_class:X"
    std::string              stage;   // "allocation" | "rounding"
    ConstraintStatus         status{ConstraintStatus::NotApplicable};
    double                   limit{0.0};
    double                   before{0.0};
    double                   after{0.0};
    std::vector<std::string> affected;  // instrument keys changed by this constraint
};

struct InstrumentAllocation {
    ConstructionInstrument input;
    bool        admitted{false};
    std::string reason;                  // admission refusal, "" when admitted
    double target_weight{0.0};           // after ERC + target vol, before constraints
    double constrained_weight{0.0};      // after every constraint (continuous)
    int    previous_units{0};
    int    units{0};                     // executable, signed
    double executable_weight{0.0};       // units * unit_notional / capital
    double target_notional_usd{0.0};     // constrained_weight * capital
    double executable_notional_usd{0.0}; // units * unit_notional (signed)
    double rounding_residual_usd{0.0};   // target - executable
    double adv_participation{-1.0};      // |executable notional| / adv; < 0 not measured
    double risk_contribution{0.0};       // executable, annualized, capital fraction
    double unit_risk_fraction{0.0};      // one unit's standalone annual vol / capital
    bool   below_one_unit{false};        // targeted, but |target| < one unit
};

struct SignalAllocation {
    ConstructionSignal input;
    bool        allocated{false};
    std::string reason;                // "" when allocated
    double target_weight{0.0};         // leg after ERC + target vol, before constraints
    double executable_weight{0.0};     // leg attributed at executable units
    double risk_contribution{0.0};     // executable
};

struct ClusterAllocation {
    std::string              cluster_id;
    std::vector<std::string> signal_ids;  // allocated members, canonical order
    bool        allocated{false};
    std::string reason;
    double composite_vol{0.0};             // unit composite's annual vol
    double erc_weight{0.0};                // scaled to the target volatility
    double target_risk_contribution{0.0};  // before constraints
    double executable_risk_contribution{0.0};
};

struct PortfolioStats {
    double gross_exposure{0.0};  // sum |w| (capital fraction)
    double net_exposure{0.0};    // sum w
    double long_exposure{0.0};
    double short_exposure{0.0};  // positive magnitude
    double annual_vol{0.0};
    int    positions{0};
    // Risk-contribution concentration across clusters: 1 / sum(share^2).
    double effective_exposures{0.0};
    double max_cluster_risk_share{0.0};  // of this portfolio's own volatility
};

struct ConstructionResult {
    std::string method{kPortfolioConstructionMethod};
    // CONSTRUCTED | NO_ALLOCATABLE_SIGNAL | NO_EXECUTABLE_POSITION | DEGENERATE_RISK_MODEL
    std::string status;
    double      capital_usd{0.0};
    double      target_annual_vol{0.0};
    std::vector<InstrumentAllocation> instruments;  // by key
    std::vector<SignalAllocation>     signals;      // by (priority, id)
    std::vector<ClusterAllocation>    clusters;     // by (best priority, id)
    std::vector<ConstraintOutcome>    constraints;
    PortfolioStats target;       // after ERC + target vol
    PortfolioStats constrained;  // after every constraint (continuous)
    PortfolioStats executable;   // integer units
    int    erc_sweeps{0};
    double erc_max_deviation{0.0};
    double turnover_target{0.0};      // sum |w_target - w_prev| before the turnover limit
    double turnover_executable{0.0};  // sum |w_exec - w_prev|
    int    repair_units_removed{0};
};

// Throws std::invalid_argument on a malformed input (duplicate / unknown key,
// direction not +/-1, a missing or out-of-range correlation, a correlation
// matrix that is not positive semi-definite, invalid limits, non-finite
// numbers). Refusals that are a property of the portfolio are typed reasons
// in the result and never throw -- including DEGENERATE_RISK_MODEL, when no
// equal-risk-contribution solution exists (two clusters that hedge each other
// exactly, e.g. opposite signal families on one instrument).
ConstructionResult construct_portfolio(const ConstructionInput& input);

const char* to_string(ConstraintStatus status) noexcept;

}  // namespace quant
