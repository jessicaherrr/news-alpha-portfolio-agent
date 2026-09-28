#pragma once

#include "quant_core/events.hpp"
#include "quant_core/strategy_context.hpp"

#include <optional>

namespace quant {

// A strategy sees history only through the StrategyContext it is handed -- a
// bounded, look-ahead-safe view that ends at the decision bar. It cannot read a
// bar dated at or after the next execution point, cannot touch the registry,
// portfolio or execution internals, and cannot see future roll info.
//
// It returns EITHER a root-level Signal OR nothing (`std::nullopt` == NO
// DECISION). The engine (Phase 06) resolves a returned Signal to a concrete
// Order via an ActiveContractResolver (fed the current market state), runs it
// through the RiskManager, and only then simulates a Fill. `signal_id` may be
// left 0 -- the engine stamps the deterministic id.
//
// NO DECISION (Phase 11.1) is a first-class outcome, distinct from a flat Signal
// (`target_units == 0`). When `decide` returns `std::nullopt` the engine does
// NOT allocate a signal_id, does NOT increment signals_generated, does NOT
// validate or queue anything, does NOT cancel an already-pending earlier signal,
// and does NOT alter any target state -- it simply proceeds. It must never be
// represented by a NaN / magic / zero / previous-target "fake" Signal.
class Strategy {
public:
    virtual ~Strategy() = default;

    // Called once per decision bar. `ctx.history().latest()` is the decision bar.
    // Return `std::nullopt` to make NO decision on this bar.
    virtual std::optional<Signal> decide(const StrategyContext& ctx) const = 0;
};

}  // namespace quant
