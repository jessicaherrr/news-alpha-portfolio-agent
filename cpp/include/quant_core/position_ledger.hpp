#pragma once

#include "quant_core/events.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace quant {

// Thrown by PositionLedger::apply_split when `units * ratio` is not (within
// float noise) an integer. `LedgerPosition::units` is a plain signed int -- see
// the comment on `apply_split` below for why this is a deliberate, permanent
// representation limit rather than something to silently round/truncate.
class FractionalSplitQuantityError : public std::runtime_error {
public:
    explicit FractionalSplitQuantityError(const std::string& what) : std::runtime_error(what) {}
};

// ---- minimal deterministic position ledger (Phase 06) --------------------
//
// The SMALLEST state the event engine needs to compute official realized PnL
// from validated Fill events: per real raw contract, a signed unit count, an
// average entry price and an open timestamp. Realized PnL is always
//   (exit_price - entry_price) * multiplier * signed_units_closed
// with `multiplier` taken from the Fill (which copied it from the ContractSpec).
//
// This is NOT the Phase 08 portfolio -- it deliberately has no cash, margin,
// collateral, multi-currency, leverage policy or mark-to-market risk. Phase 08's
// PortfolioAccountant (portfolio.hpp) wraps a PositionLedger and adds all of
// that; this type stays the minimal Fill-derived realized-PnL core. Each raw
// contract is tracked independently -- a position that crosses a futures roll is
// represented as a real close of the old contract and a real open of the new
// one (the engine drives that with explicit Orders/Fills), never as one
// contract silently relabelled as another.

// One closed / reduced position event on a single raw contract.
struct ClosedTrade {
    std::uint32_t instrument_id{0};
    std::string   raw_symbol;
    std::string   root_symbol;
    std::int64_t  ts_open_ns{0};
    std::int64_t  ts_close_ns{0};
    int           quantity{0};          // contracts closed (> 0)
    int           direction{0};         // +1 == closed a long, -1 == closed a short
    double        entry_price{0.0};     // average entry of the closed units
    double        exit_price{0.0};      // this fill's price
    double        gross_pnl_usd{0.0};   // (exit-entry)*multiplier*direction*quantity
    double        costs_usd{0.0};       // commission attributed to this closing fill
    double        net_pnl_usd{0.0};     // gross - costs
    std::string   close_reason;         // "signal" | "roll" | "eot"
};

// Live state of one contract.
struct LedgerPosition {
    std::uint32_t instrument_id{0};
    std::string   raw_symbol;
    std::string   root_symbol;
    int           units{0};             // signed
    double        avg_entry_price{0.0}; // 0 when flat
    double        multiplier{0.0};
    std::int64_t  opened_ts_ns{0};
};

class PositionLedger {
public:
    struct ApplyResult {
        double realized_pnl_usd{0.0};   // this fill's realized PnL (0 if purely opening)
        double costs_usd{0.0};          // this fill's commission
        int    closed_units{0};         // > 0 when the fill reduced/closed/flipped a position
        std::optional<ClosedTrade> closed;  // present iff closed_units > 0
    };

    // Apply a validated raw-contract Fill. `root_symbol` and `close_reason` are
    // engine context (the Fill itself does not carry them).
    ApplyResult apply(const Fill& fill, const std::string& root_symbol,
                      const std::string& close_reason);

    int units(std::uint32_t instrument_id) const noexcept;

    // ETF corporate action: rebase an open position's unit count and average
    // entry price for a split, with zero economic PnL effect --
    // units * avg_entry_price (the position's cost-basis notional) is invariant
    // before/after. No-op if there is no open position for `instrument_id`
    // (units == 0 or the instrument was never seen): a split on a flat position
    // touches nothing.
    //
    // `LedgerPosition::units` is a plain signed `int` (the shared Futures
    // representation -- always a whole number of contracts), NOT a double. A
    // split ratio that does not divide the current unit count into a whole
    // number (e.g. a 1-for-8 reverse split on 100 shares -> 12.5) cannot be
    // represented and throws FractionalSplitQuantityError rather than silently
    // rounding/truncating a real economic quantity. This is a deliberate,
    // permanent limitation of `int units`; widening it to `double` would touch
    // the shared Futures type and is out of scope for this additive ETF path.
    // On throw, the position is left completely unchanged (strong exception
    // guarantee -- the mutation is computed and validated before anything is
    // written).
    //
    // Does NOT touch realized_pnl_usd_ or costs_usd_ -- a split is not a Fill.
    void apply_split(std::uint32_t instrument_id, double ratio);

    // Cumulative realized PnL and commission across every fill applied so far.
    double realized_pnl_usd() const noexcept { return realized_pnl_usd_; }
    double costs_usd() const noexcept { return costs_usd_; }

    // Unrealized PnL of all open positions marked at `mark_prices`
    // (instrument_id -> price). An instrument with no mark contributes 0.
    double unrealized_pnl_usd(
        const std::unordered_map<std::uint32_t, double>& mark_prices) const noexcept;

    // Open positions (units != 0), deterministically ordered by instrument_id.
    std::vector<LedgerPosition> open_positions() const;

private:
    // std::map: iteration order is by key (instrument_id) -- deterministic, never
    // hash order.
    std::map<std::uint32_t, LedgerPosition> positions_;
    double realized_pnl_usd_{0.0};
    double costs_usd_{0.0};
};

}  // namespace quant
