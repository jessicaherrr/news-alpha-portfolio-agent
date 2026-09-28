#pragma once

#include "quant_core/bar_history.hpp"
#include "quant_core/contract.hpp"
#include "quant_core/events.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace quant {

// The ONLY thing a Strategy sees when it decides (Phase 05).
//
// Everything here is known AT decision_ts_ns. Deliberately excluded so a
// strategy cannot cheat:
//   * the full bar vector            -- only a bounded BarHistoryView is exposed
//   * any mutable market/engine state
//   * the ContractRegistry           -- it could resolve a *future* active contract
//   * future roll / back-adjustment info
//   * execution internals, portfolio internals
//
// No raw owning pointers, no containers that reopen future access. The one
// pointer (`active_contract`) is a const, non-owning snapshot of a single
// immutable ContractSpec.
class StrategyContext {
public:
    StrategyContext(std::int64_t decision_ts_ns,
                    const MarketEvent& event,
                    const BarHistoryView& history,
                    std::string_view root_symbol,
                    const ContractSpec* active_contract) noexcept
        : decision_ts_ns_(decision_ts_ns),
          event_(event),
          history_(history),
          root_symbol_(root_symbol),
          active_contract_(active_contract) {}

    // Non-copyable: it is a scoped view built by the engine for a single decide()
    // call and destroyed straight after.
    StrategyContext(const StrategyContext&) = delete;
    StrategyContext& operator=(const StrategyContext&) = delete;

    std::int64_t decision_ts_ns() const noexcept { return decision_ts_ns_; }

    // The MarketEvent being decided on (its bar payload is the decision bar).
    const MarketEvent& event() const noexcept { return event_; }

    // Look-ahead-safe: ends at the decision bar; every accessor is bounds-checked
    // and throws std::out_of_range rather than reading past the visible range.
    const BarHistoryView& history() const noexcept { return history_; }

    // The root / continuous series this strategy trades, e.g. "NQ".
    std::string_view root_symbol() const noexcept { return root_symbol_; }

    // Read-only metadata of the contract live at decision_ts_ns (may be null in
    // the legacy CSV path). This is a single immutable snapshot -- NOT the
    // registry -- so a strategy cannot look up a contract active at a later ts.
    const ContractSpec* active_contract() const noexcept { return active_contract_; }

private:
    std::int64_t          decision_ts_ns_;
    const MarketEvent&    event_;
    const BarHistoryView& history_;
    std::string           root_symbol_;
    const ContractSpec*   active_contract_;
};

}  // namespace quant
