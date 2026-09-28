#pragma once

#include <cstdint>

namespace quant {

// ============================================================================
// ETF corporate actions (Phase 6 pilot -- smallest additive capability)
// ============================================================================
//
// These two value types describe corporate actions that are NOT Fills: a split
// rebases an existing position's unit count / cost basis with zero economic PnL
// effect, and a cash distribution creates (at the ex-date) and later settles (at
// the pay date) an income obligation that is completely separate from Fill-
// derived realized PnL. Nothing here touches `Fill`/`Order` (events.hpp) or any
// Futures-path accounting; futures do not have splits and this repo's futures
// contracts are not distribution-bearing. See position_ledger.hpp::apply_split
// and portfolio.hpp's PortfolioAccountant::record_distribution_ex_date /
// record_distribution_payment for the accounting these actions drive.

// A share split (or reverse split). `ratio` maps old units to new units:
// new_units = old_units * ratio. 2.0 == a 2-for-1 forward split; 0.125 == a
// 1-for-8 reverse split. ratio == 1.0 is a no-op and is rejected by is_valid()
// so callers don't submit a vacuous action.
struct SplitAction {
    std::uint32_t instrument_id{0};
    std::int64_t  effective_ts_ns{0};  // the split's effective/ex date
    double        ratio{0.0};          // new_units = old_units * ratio

    bool is_valid() const noexcept {
        return instrument_id != 0
            && effective_ts_ns > 0
            && ratio > 0.0
            && ratio != 1.0;
    }
};

// A cash distribution (dividend). The ex-date determines entitlement (whoever
// holds the position as of the ex-date is owed the distribution); the pay date
// is when the cash actually settles. `pay_date_ts_ns >= ex_date_ts_ns` always.
struct CashDistribution {
    std::uint32_t instrument_id{0};
    std::int64_t  ex_date_ts_ns{0};
    std::int64_t  pay_date_ts_ns{0};    // must be >= ex_date_ts_ns
    double        amount_per_share_usd{0.0};

    bool is_valid() const noexcept {
        return instrument_id != 0
            && ex_date_ts_ns > 0
            && pay_date_ts_ns >= ex_date_ts_ns
            && amount_per_share_usd > 0.0;
    }
};

}  // namespace quant
