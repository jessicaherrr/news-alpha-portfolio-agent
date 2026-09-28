#pragma once

#include "quant_core/contract_registry.hpp"
#include "quant_core/events.hpp"
#include "quant_core/fill.hpp"
#include "quant_core/types.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <string>

namespace quant {

// ============================================================================
// Deterministic futures execution / fill simulator (Phase 07)
// ============================================================================
//
// The engine (Phase 06) turns strategy intent into a risk-approved Order. Phase
// 07 inserts a clean ExecutionSimulator between that approved Order and the
// PositionLedger:
//
//   Signal -> [ActiveContractResolver] -> Order -> [RiskManager] -> RiskDecision
//          -> [ExecutionSimulator] -> Fill | deterministic non-fill -> Ledger
//
// Invariants (docs/BOUNDARY_CONTRACT.md sections E + J, docs/EXECUTION_MODEL.md):
//   * the strategy NEVER chooses the official fill price -- the simulator does,
//     from raw-contract market data and an explicit ExecutionConfig;
//   * every Fill is built only through make_fill() -- so only real raw-contract
//     prices, tick_size / multiplier from the ContractSpec, tick-grid rounding;
//   * no RNG: the same Order + same ExecutionRequest + same ExecutionConfig
//     always produce an identical ExecReport (price, ts, ids, commission);
//   * OHLCV bars never gain invented bid/ask depth -- slippage / spread are
//     explicit configured tick offsets, not a synthesised order book.
//
// NOT here: margin / collateral / portfolio optimisation / production risk rules
// (Phase 08); the feature engine, strategy DSL, baseline strategies, ML,
// validation, the agent, the UI, pybind, broker routing (later phases).

// ---- configuration -------------------------------------------------------
//
// All fields are explicit and have inert defaults (a Market order fills at the
// reference price, no cost) so Phase 06 behaviour is recovered by ExecutionConfig{}.
struct ExecutionConfig {
    // Adverse price offset applied to a *marketable* order (a Market order, or a
    // Stop order once triggered). Expressed in CONTRACT TICKS; the tick size
    // itself always comes from the ContractSpec, never from here.
    //   Buy  fill = reference_price + (slippage_ticks + spread_ticks) * tick_size
    //   Sell fill = reference_price - (slippage_ticks + spread_ticks) * tick_size
    double slippage_ticks{0.0};

    // Optional half-spread assumption, also in ticks, added on top of slippage
    // for marketable orders. Kept separate from slippage_ticks purely for
    // attribution; the two are summed before being applied. A Limit order never
    // pays either -- it fills at its own limit price (or better on a gap).
    double spread_ticks{0.0};

    // Commission charged per contract, per fill, in USD. Applied to every Fill
    // the simulator produces (strategy fills, roll close-legs, end-of-test).
    double commission_per_contract_usd{0.0};

    // ADDITIVE (News Alpha Phase H acceptance patch). Optional per-ROOT
    // commission, USD per unit, keyed by ContractSpec::root_symbol -- the
    // execution boundary's own identity: every raw contract of a futures root
    // shares one entry (a roll never changes the rate), an ETF is its synthetic
    // `E<ticker>` root and a unit is one share. A root found here is charged its
    // own rate; any other root is charged commission_per_contract_usd. EMPTY
    // (the default, and every pre-existing caller) => the scalar path, byte for
    // byte. Whether a schedule must be complete is the caller's policy (the
    // portfolio replay CLI refuses an incomplete one); the simulator stays generic.
    std::map<std::string, double> commission_per_contract_usd_by_root{};

    // Phase 07 MVP fills are ALL-OR-NONE. A volume-cap partial-fill model is a
    // documented approximation left for a later market-data schema (Phase 08+);
    // see docs/EXECUTION_MODEL.md.
};

// ---- one execution attempt ---------------------------------------------
//
// Built by the engine for a single risk-approved Order. `bar` is the
// execution-time OHLCV bar of the Order's own instrument (the raw contract being
// traded), `ts_ns` the execution event timestamp.
//
// `ref_price_override` + `administrative` are how the engine expresses a forced
// close it priced itself -- a roll close-leg or an end-of-test liquidation. Such
// a fill takes NO slippage / spread (it is not a marketable order the strategy
// chose) and is priced at the supplied reference, which the engine derived
// deterministically from raw-contract data.
struct ExecutionRequest {
    MarketBar             bar{};              // execution-time bar of order.instrument_id
    std::int64_t          ts_ns{0};           // execution event timestamp (UTC ns)
    std::optional<double> ref_price_override{};  // forced-close reference price
    bool                  administrative{false}; // roll / eot: no slippage, no spread
    std::string           reason{"signal"};   // ledger close-reason tag
};

enum class ExecOutcome : int {
    Filled,   // a validated Fill was produced
    NoFill,   // deterministic non-fill (e.g. a Limit whose price the bar never reached)
    Pending,  // reserved: a working order awaiting a later bar (unused in Phase 07)
    Rejected, // the request could not be executed (should not happen post-validate_order)
};

struct ExecReport {
    ExecOutcome         outcome{ExecOutcome::NoFill};
    std::optional<Fill> fill{};                 // present iff outcome == Filled
    double              applied_slippage_ticks{0.0};
    // machine tag: "market" | "administrative" | "limit_at_price" |
    // "limit_gap_through" | "limit_unreachable" | "stop_triggered" |
    // "stop_gap_through" | "stop_not_triggered" | "unknown_instrument"
    std::string         detail;
};

// ---- interface --------------------------------------------------------
class ExecutionSimulator {
public:
    virtual ~ExecutionSimulator() = default;

    // `order` has ALREADY passed validate_order() and the RiskManager
    // (order.quantity == the approved quantity). The simulator decides whether
    // and at what raw-contract price it fills against `req`, builds the Fill only
    // through make_fill(), and stamps it with `fill_id`. It never consults the
    // strategy. Deterministic.
    virtual ExecReport execute(const Order& order,
                               const ExecutionRequest& req,
                               FillId fill_id) const = 0;

    virtual const ExecutionConfig& config() const noexcept = 0;
};

// ---- MVP: deterministic OHLCV-bar execution model -----------------------
//
// Market / Limit / Stop against a single OHLCV bar. Semantics (buy/sell rule,
// gap-through, fill price) are documented in docs/EXECUTION_MODEL.md and covered
// by cpp/tests/test_execution_simulator.cpp.
class BarExecutionSimulator final : public ExecutionSimulator {
public:
    explicit BarExecutionSimulator(const ContractRegistry& registry,
                                   ExecutionConfig config = {}) noexcept
        : registry_(registry), config_(config) {}

    ExecReport execute(const Order& order, const ExecutionRequest& req,
                       FillId fill_id) const override;

    const ExecutionConfig& config() const noexcept override { return config_; }

private:
    const ContractRegistry& registry_;
    ExecutionConfig         config_;
};

// ---- intrabar stop / target ambiguity (Phase 07 requirement 6) ----------
//
// An OHLC bar does not reveal the path between open/high/low/close. When an open
// position's protective stop AND its profit target are both potentially
// reachable inside ONE bar and the true order of events is unknowable, this is
// the deterministic conservative rule the platform uses:
//
//   * if the bar opens already through the stop            -> Stop  (gap)
//   * else if the bar opens already through the target     -> Target (gap)
//   * else if BOTH the stop and the target are reachable   -> Stop  (worst case)
//   * else whichever single one is reachable, or None
//
// It is NEVER resolved by which outcome gives better realised PnL. `exit_side` is
// the side of the closing order: Sell exits a long (stop below, target above),
// Buy exits a short (stop above, target below).
enum class BracketExit : int { None = 0, Stop = 1, Target = 2 };

BracketExit resolve_intrabar_bracket(const MarketBar& bar, Side exit_side,
                                     double stop_price, double target_price) noexcept;

}  // namespace quant
