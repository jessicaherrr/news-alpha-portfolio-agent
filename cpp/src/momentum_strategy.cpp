#include "quant_core/momentum_strategy.hpp"

#include <string>

namespace quant {

// Temporary smoke strategy: root-level time-series-momentum intent, target +/-1
// contract. Used by the smokes and the reference CLI; the real strategy library
// is a later phase. `target_units` is a target POSITION (Phase 06), not an order
// size -- the engine computes the order delta.
std::optional<Signal> TimeSeriesMomentum::decide(const StrategyContext& ctx) const {
    const BarHistoryView& history = ctx.history();

    Signal sig;
    sig.ts_decision_ns = ctx.decision_ts_ns();
    sig.root_symbol    = std::string(ctx.root_symbol());
    sig.target_units   = 0.0;
    sig.rationale_code = "tsmom_flat";

    // Need `lookback_` bars strictly before the decision bar.
    if (lookback_ == 0 || history.size() <= lookback_) {
        return sig;
    }
    const double past = history.ago(lookback_).close;
    // A ratio-return is undefined when the base price is <= 0 (this smoke
    // strategy's signal math, NOT a domain rule -- the engine and execution layer
    // fully support signed prices). Stay flat rather than emit a garbage signal.
    if (past <= 0.0) {
        return sig;
    }
    const double ret = history.latest().close / past - 1.0;
    if (ret > threshold_return_) {
        sig.target_units   = 1.0;
        sig.rationale_code = "tsmom_up";
    } else if (ret < -threshold_return_) {
        sig.target_units   = -1.0;
        sig.rationale_code = "tsmom_down";
    }
    return sig;
}

}  // namespace quant
