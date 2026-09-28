#include "quant_core/position_ledger.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace quant {
namespace {

int sign_of(int v) noexcept { return (v > 0) - (v < 0); }

}  // namespace

PositionLedger::ApplyResult PositionLedger::apply(const Fill& fill,
                                                 const std::string& root_symbol,
                                                 const std::string& close_reason) {
    const int signed_qty = (fill.side == Side::Buy) ? fill.quantity : -fill.quantity;

    LedgerPosition& p = positions_[fill.instrument_id];
    if (p.instrument_id == 0) {
        p.instrument_id = fill.instrument_id;
        p.raw_symbol    = fill.raw_symbol;
        p.root_symbol   = root_symbol;
        p.multiplier    = fill.multiplier;
    }

    ApplyResult out;
    out.costs_usd = fill.commission_usd;
    costs_usd_ += fill.commission_usd;

    const int old_units = p.units;
    const bool adding = (old_units == 0) || (sign_of(old_units) == sign_of(signed_qty));

    if (adding) {
        const double old_abs = std::fabs(static_cast<double>(old_units));
        const double add_abs = std::fabs(static_cast<double>(signed_qty));
        if (old_units == 0) {
            p.avg_entry_price = fill.fill_price;
            p.opened_ts_ns    = fill.ts_fill_ns;
        } else {
            p.avg_entry_price =
                (p.avg_entry_price * old_abs + fill.fill_price * add_abs) / (old_abs + add_abs);
        }
        p.units = old_units + signed_qty;
    } else {
        // Reducing / closing / flipping.
        const int closed_units = std::min(std::abs(old_units), std::abs(signed_qty));
        const int direction    = sign_of(old_units);  // +1 closed a long, -1 closed a short
        const double realized =
            (fill.fill_price - p.avg_entry_price) * fill.multiplier * direction * closed_units;

        out.realized_pnl_usd = realized;
        out.closed_units     = closed_units;
        realized_pnl_usd_ += realized;

        ClosedTrade t;
        t.instrument_id = p.instrument_id;
        t.raw_symbol    = p.raw_symbol;
        t.root_symbol   = p.root_symbol;
        t.ts_open_ns    = p.opened_ts_ns;
        t.ts_close_ns   = fill.ts_fill_ns;
        t.quantity      = closed_units;
        t.direction     = direction;
        t.entry_price   = p.avg_entry_price;
        t.exit_price    = fill.fill_price;
        t.gross_pnl_usd = realized;
        t.costs_usd     = fill.commission_usd;
        t.net_pnl_usd   = realized - fill.commission_usd;
        t.close_reason  = close_reason;
        out.closed      = t;

        const int new_units = old_units + signed_qty;
        p.units = new_units;
        if (new_units == 0) {
            p.avg_entry_price = 0.0;
            p.opened_ts_ns    = 0;
        } else if (sign_of(new_units) != sign_of(old_units)) {
            // Flipped through zero: the overshoot opens a fresh position.
            p.avg_entry_price = fill.fill_price;
            p.opened_ts_ns    = fill.ts_fill_ns;
        }
        // Partial reduction (same sign): avg_entry_price / opened_ts_ns unchanged.
    }

    return out;
}

int PositionLedger::units(std::uint32_t instrument_id) const noexcept {
    const auto it = positions_.find(instrument_id);
    return it == positions_.end() ? 0 : it->second.units;
}

void PositionLedger::apply_split(std::uint32_t instrument_id, double ratio) {
    const auto it = positions_.find(instrument_id);
    if (it == positions_.end() || it->second.units == 0) {
        return;  // flat / never-seen position: a split touches nothing.
    }
    LedgerPosition& p = it->second;

    const double raw_new_units = static_cast<double>(p.units) * ratio;
    const double rounded       = std::llround(raw_new_units);
    if (std::fabs(raw_new_units - rounded) > 1e-6) {
        throw FractionalSplitQuantityError(
            "apply_split: instrument_id=" + std::to_string(instrument_id) +
            " old_units=" + std::to_string(p.units) +
            " ratio=" + std::to_string(ratio) +
            " produced a non-integer unit count " + std::to_string(raw_new_units) +
            " (int units cannot represent a fractional share)");
    }

    // Cost-basis notional (units * avg_entry_price) is invariant before/after --
    // zero economic PnL effect. opened_ts_ns / instrument_id / raw_symbol /
    // root_symbol / multiplier are unchanged.
    p.units           = static_cast<int>(rounded);
    p.avg_entry_price = p.avg_entry_price / ratio;
}

double PositionLedger::unrealized_pnl_usd(
    const std::unordered_map<std::uint32_t, double>& mark_prices) const noexcept {
    double total = 0.0;
    for (const auto& [id, p] : positions_) {
        if (p.units == 0) continue;
        const auto it = mark_prices.find(id);
        if (it == mark_prices.end()) continue;
        total += (it->second - p.avg_entry_price) * p.multiplier * p.units;
    }
    return total;
}

std::vector<LedgerPosition> PositionLedger::open_positions() const {
    std::vector<LedgerPosition> out;
    for (const auto& [id, p] : positions_) {
        (void)id;
        if (p.units != 0) out.push_back(p);
    }
    return out;  // positions_ is ordered by instrument_id
}

}  // namespace quant
