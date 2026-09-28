#pragma once

namespace quant {

// Phase 05 placeholder limits, still used by MaxContractsRiskManager and the
// Phase 05/06/07 tests. The real Phase 08 hard-risk configuration is
// `quant::RiskConfig` (risk_config.hpp), enforced by `PortfolioRiskManager`.
struct RiskLimits {
    int max_contracts_per_symbol{5};
    int max_gross_contracts{12};
    double max_daily_loss_usd{1500.0};
    double max_drawdown_pct{0.10};
    double max_margin_utilization_pct{0.50};
};

inline bool validate_order_size(int requested_contracts, const RiskLimits& limits) {
    return requested_contracts >= 0 && requested_contracts <= limits.max_contracts_per_symbol;
}

}  // namespace quant
