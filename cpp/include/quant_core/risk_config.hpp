#pragma once

#include "quant_core/contract.hpp"
#include "quant_core/margin.hpp"
#include "quant_core/portfolio.hpp"

#include <cstdint>

namespace quant {

// ============================================================================
// Hard risk configuration + review context (Phase 08)
// ============================================================================
//
// CLAUDE.md risk rule 1: hard risk limits are deterministic and CANNOT be
// overridden by an LLM. The Research Agent may propose a position; the C++
// RiskManager owns the final admissibility decision. Every Order still resolves
// to exactly one of APPROVE / RESIZE / REJECT (docs/BOUNDARY_CONTRACT.md C).
//
// All limits are opt-out with 0 (or a non-positive value) meaning "disabled",
// except the position caps which ship with protective non-zero defaults.

enum class StaleMarkPolicy : int {
    // Do not consider mark freshness at all. NOT recommended -- leverage and
    // margin-utilisation checks then divide by an equity computed from stale
    // marks without knowing it.
    Ignore = 0,
    // DEFAULT: reject a risk-INCREASING order when its own reference price is
    // stale/missing, OR when any held position feeding portfolio equity/exposure
    // has a stale mark (PortfolioState.has_stale_mark). Risk-reducing orders are
    // never blocked -- you must always be able to get flat.
    RejectRiskIncreasing = 1,
};

struct RiskConfig {
    // ---- position limits (counted on the POST-trade position) ---------------
    int max_contracts_per_symbol{5};   // |units| in one raw contract
    int max_contracts_per_root{10};    // |sum of units| across a root (ES, NQ, ...)
    int max_gross_contracts{20};       // sum |units| across the whole portfolio
    int max_order_contracts{0};        // per-order quantity ceiling; 0 == disabled

    // ---- exposure / leverage (ContractSpec economics) ----------------------
    double max_gross_exposure_usd{0.0};  // sum |signed notional|; 0 == disabled
    double max_gross_leverage{0.0};      // gross_exposure / equity; 0 == disabled
    double max_net_leverage{0.0};        // |net_exposure| / equity; 0 == disabled

    // ---- margin -----------------------------------------------------------
    double              max_margin_utilization_pct{0.50};  // initial_margin / equity; 0 == disabled
    MissingMarginPolicy missing_margin{MissingMarginPolicy::Reject};

    // ---- loss / drawdown kill switches (REJECT-only, never RESIZE) ---------
    double max_daily_loss_usd{1500.0};  // trip when (day_start_equity - equity) >= this
    double max_drawdown_pct{0.10};      // trip when drawdown_pct >= this
    double max_drawdown_usd{0.0};       // absolute variant; 0 == disabled

    // ---- stale-mark protection (requirement 7) ----------------------------
    StaleMarkPolicy stale_mark{StaleMarkPolicy::RejectRiskIncreasing};

    // ---- portfolio valuation config (starting capital, staleness, day) ----
    PortfolioConfig portfolio{};

    // MVP invariant: historical research + paper trading only (CLAUDE risk rule
    // 2). Informational here; no live routing exists to gate.
    bool paper_only{true};
};

// Everything a hard-risk RiskManager needs beyond the Order itself. The engine
// builds this from PRE-trade state before the mandatory risk gate; `portfolio`
// and `spec` are never null on the engine path.
struct RiskReviewContext {
    std::int64_t         as_of_ts_ns{0};
    const PortfolioState* portfolio{nullptr};  // pre-trade snapshot
    const ContractSpec*  spec{nullptr};        // economics of order.instrument_id
    const MarginModel*   margin{nullptr};      // nullable

    // Best available valuation price for the order's instrument, for post-trade
    // notional / leverage / margin math. For a signal leg this is the execution
    // bar itself (age 0); for an administrative close it may be older.
    double       reference_price{0.0};
    std::int64_t reference_price_ts_ns{0};
    std::int64_t reference_price_age_ns{0};
    bool         reference_price_stale{false};
    bool         reference_price_present{false};

    int current_position_units{0};  // signed, in order.instrument_id (pre-trade)
    int current_root_units{0};      // signed, summed across the order's root (pre-trade)
};

}  // namespace quant
