#pragma once

#include "quant_core/strategy.hpp"

#include <cstddef>
#include <optional>

namespace quant {

class TimeSeriesMomentum final : public Strategy {
public:
    TimeSeriesMomentum(std::size_t lookback, double threshold_return)
        : lookback_(lookback), threshold_return_(threshold_return) {}

    // Always emits a Signal (flat when history is insufficient) -- this reference
    // / smoke strategy never returns NO DECISION.
    std::optional<Signal> decide(const StrategyContext& ctx) const override;

private:
    std::size_t lookback_;
    double threshold_return_;
};

}  // namespace quant
