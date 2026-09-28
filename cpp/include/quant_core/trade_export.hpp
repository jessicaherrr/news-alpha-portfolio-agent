#pragma once

// ADDITIVE AUDIT/EXPORT INTERFACE (Phase 15A). READ-ONLY.
//
// Serialises the already-recorded `BacktestResult::trades` audit trail -- the
// Fill-derived closed/reduced position events the engine has produced since
// Phase 06 -- to a deterministic CSV. It computes nothing: every value is
// copied verbatim out of a `quant::ClosedTrade`.
//
// It changes NO execution, accounting, fill, roll, risk or portfolio semantics.
// Nothing here is reachable from the event loop; the engine never calls it. It
// exists so the Python research layer can define a PRIMARY TRADE EPISODE label
// from official C++ Fill-derived economics instead of re-deriving PnL in
// Python (CLAUDE.md architecture boundary 3).
//
// Two exports are provided, and BOTH are needed to reconstruct an episode's
// economics honestly:
//
//   * closed trades -- gross realised PnL per closed/reduced position, priced by
//     the engine. Note that `ClosedTrade::costs_usd` is, by the ledger's own
//     definition, the commission of the CLOSING fill only; the opening fill's
//     commission is real and is in the engine's total but is attributed to no
//     ClosedTrade.
//   * fills -- every fill's commission, so a consumer can attribute the full
//     round-turn cost to the episode that incurred it by summing, never by
//     re-deriving the ledger's logic.
//
// Correcting the attribution inside `ClosedTrade` would be a change to the
// official C++ accounting path and is deliberately NOT done here.
//
// Column contract (frozen, mirrored by
// alpha_agent.ml.trade_export.CLOSED_TRADE_COLUMNS):
//   trade_index,instrument_id,raw_symbol,root_symbol,ts_open_ns,ts_close_ns,
//   quantity,direction,entry_price,exit_price,gross_pnl_usd,costs_usd,
//   net_pnl_usd,close_reason
//
// Fill column contract (frozen, mirrored by
// alpha_agent.ml.trade_export.FILL_COLUMNS):
//   fill_index,fill_id,order_id,ts_fill_ns,instrument_id,raw_symbol,side,
//   quantity,fill_price,commission_usd,slippage_ticks
//
// Rows are written in the engine's chronological audit order; `trade_index` /
// `fill_index` is that order, so a consumer never has to re-sort ties.

#include "quant_core/events.hpp"
#include "quant_core/position_ledger.hpp"

#include <cstdio>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace quant {

inline constexpr const char* kClosedTradeCsvHeader =
    "trade_index,instrument_id,raw_symbol,root_symbol,ts_open_ns,ts_close_ns,"
    "quantity,direction,entry_price,exit_price,gross_pnl_usd,costs_usd,"
    "net_pnl_usd,close_reason";

namespace detail {

// %.12g, matching the daily_equity trace precision, so Python differences the
// values without rounding noise.
inline std::string trade_g(double v) {
    char buf[40];
    std::snprintf(buf, sizeof buf, "%.12g", v);
    return std::string(buf);
}

// The export is a plain unquoted CSV. A symbol or close_reason containing a
// comma, quote or newline would silently corrupt the column contract, so it is
// a loud error instead. Real CME raw symbols never contain these.
inline void reject_csv_hostile(const std::string& field, const char* what) {
    if (field.find_first_of(",\"\n\r") != std::string::npos) {
        throw std::runtime_error(std::string("closed-trade export: ") + what +
                                 " contains a CSV-hostile character: '" + field + "'");
    }
}

}  // namespace detail

// Write the closed-trade audit trail as CSV. Deterministic and side-effect free
// with respect to the engine: `trades` is const.
inline void write_closed_trades_csv(std::ostream& out, const std::vector<ClosedTrade>& trades) {
    out << kClosedTradeCsvHeader << '\n';
    for (std::size_t i = 0; i < trades.size(); ++i) {
        const auto& t = trades[i];
        detail::reject_csv_hostile(t.raw_symbol, "raw_symbol");
        detail::reject_csv_hostile(t.root_symbol, "root_symbol");
        detail::reject_csv_hostile(t.close_reason, "close_reason");
        out << i << ',' << t.instrument_id << ',' << t.raw_symbol << ',' << t.root_symbol << ','
            << t.ts_open_ns << ',' << t.ts_close_ns << ',' << t.quantity << ',' << t.direction
            << ',' << detail::trade_g(t.entry_price) << ',' << detail::trade_g(t.exit_price) << ','
            << detail::trade_g(t.gross_pnl_usd) << ',' << detail::trade_g(t.costs_usd) << ','
            << detail::trade_g(t.net_pnl_usd) << ',' << t.close_reason << '\n';
    }
}

inline constexpr const char* kFillCsvHeader =
    "fill_index,fill_id,order_id,ts_fill_ns,instrument_id,raw_symbol,side,quantity,"
    "fill_price,commission_usd,slippage_ticks";

// Write the fill audit trail as CSV. Like the closed-trade export it computes
// nothing: it exists so a consumer can attribute the FULL round-turn commission
// to an episode, which `ClosedTrade::costs_usd` alone cannot express.
inline void write_fills_csv(std::ostream& out, const std::vector<Fill>& fills) {
    out << kFillCsvHeader << '\n';
    for (std::size_t i = 0; i < fills.size(); ++i) {
        const auto& f = fills[i];
        detail::reject_csv_hostile(f.raw_symbol, "raw_symbol");
        out << i << ',' << f.fill_id << ',' << f.order_id << ',' << f.ts_fill_ns << ','
            << f.instrument_id << ',' << f.raw_symbol << ','
            << (f.side == Side::Buy ? "buy" : "sell") << ',' << f.quantity << ','
            << detail::trade_g(f.fill_price) << ',' << detail::trade_g(f.commission_usd) << ','
            << detail::trade_g(f.slippage_ticks) << '\n';
    }
}

}  // namespace quant
