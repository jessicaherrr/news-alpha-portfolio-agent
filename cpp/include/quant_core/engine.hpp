#pragma once

#include "quant_core/contract_registry.hpp"
#include "quant_core/contract_selector.hpp"
#include "quant_core/corporate_actions.hpp"
#include "quant_core/events.hpp"
#include "quant_core/execution_simulator.hpp"
#include "quant_core/portfolio.hpp"
#include "quant_core/position_ledger.hpp"
#include "quant_core/risk_manager.hpp"
#include "quant_core/strategy.hpp"
#include "quant_core/types.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace quant {

// ============================================================================
// Deterministic event-driven futures backtest engine (Phase 06)
// ============================================================================
//
// Replaces the Phase 01 ad-hoc entry/exit loop. Per ordered Bar MarketEvent:
//
//   1. update the per-ROOT market state from the event's instrument_id
//   2. execute any strategy intent that was queued at an EARLIER bar of this
//      root, at THIS event (never the bar it was decided on):
//        a. if the held contract != the execution-time active contract, close
//           the old raw contract and re-establish the position in the new one
//           (explicit roll -- real Orders/Fills, never a silent relabel)
//        b. resolve the active raw contract from the execution-time MarketState
//        c. order_delta = target_units - current_position_units
//        d. Order -> RiskManager -> RiskDecision -> apply -> make_fill -> ledger
//   3. append the event's bar to this root's bounded history
//   4. build a look-ahead-safe StrategyContext, call Strategy::decide
//   5. engine stamps signal_id, validate_signal, queue for the next bar of the root
//
// Only Bar events in Phase 06/07. No same-bar look-ahead. Official PnL derives
// ONLY from validated Fill events via ContractSpec tick/multiplier -- never from
// back-adjusted / continuous prices, a config multiplier, or strategy output.
//
// Phase 07 routes every risk-approved Order through an ExecutionSimulator
// (execution_simulator.hpp): the simulator owns the fill price (slippage /
// spread / limit / stop / gap-through), the engine owns timing (per-root
// execution bar, N-bar latency) and the roll close-leg pricing policy below.
//
// CAUSALITY INVARIANT (Phase 07.1): every Fill produced while processing engine
// event time T satisfies `fill.ts_fill_ns >= T` -- for the current bar-based
// model, `== T`. No strategy leg, roll close-leg, roll fallback, end-of-test
// liquidation, or simulator path may backdate a Fill. The engine asserts this on
// every fill and throws ExecutionDomainError otherwise.
//
// Phase 08 adds deterministic portfolio accounting (cash / equity / exposure /
// margin / leverage / drawdown -- portfolio.hpp) and hard risk controls
// (risk_manager.hpp / risk_config.hpp): the engine maintains a PortfolioAccountant
// alongside the ledger and builds a RiskReviewContext for the mandatory risk
// gate. See docs/PORTFOLIO_AND_RISK.md.
//
// NOT here: multi-currency, portfolio optimisation, SPAN / cross-margin /
// correlated exposure (explicit TODO in PortfolioState); the feature engine,
// strategy DSL, baseline strategies, ML, validation, the agent, the UI, pybind,
// broker routing. Scheduled expiry / settlement fills need typed lifecycle
// events -- a later phase.

enum class EndOfTestPolicy : int {
    ForceLiquidateFinalClose,  // liquidate positions with a fresh executable mark; stale ones follow EotStalePolicy
    LeaveOpen,                 // keep every open position; report unrealized PnL + mark age separately
};

// What ForceLiquidateFinalClose does with an open position whose last observed
// raw-contract bar is NOT a contemporaneous executable price at the final engine
// event time (mark age > EotExecutionPolicy.contemporaneous_tolerance_ns, or the
// contract is not tradable then). An old observed close must never be silently
// used as if it were a current executable price (Phase 07.2).
enum class EotStalePolicy : int {
    // DEFAULT, reliability-safe: leave the position open. It stays in
    // final_positions and is reported in final_position_marks with mark price,
    // mark timestamp and mark age; its unrealized PnL is reported separately and
    // is NEVER folded into realized PnL.
    LeaveOpen = 0,
    // Refuse: throw ExecutionDomainError rather than force a stale execution.
    Fail = 1,
    // EXPLICIT OPT-IN APPROXIMATION: liquidate at the last observed close, still
    // stamped at the final engine event time (never retroactive), recorded in
    // eot_liquidations with is_contemporaneous == false / is_approximation ==
    // true. Refused (throws) if the contract is not tradable at the final time.
    ForceApproximateStaleClose = 2,
};

struct EotExecutionPolicy {
    EotStalePolicy stale{EotStalePolicy::LeaveOpen};
    // Maximum mark age (final_engine_ts - last_bar_ts) at which a position's last
    // observed close still counts as a contemporaneous executable price for a
    // forced liquidation. 0 == require the position's last bar to be exactly the
    // final engine event. No product-specific hard-coded threshold.
    std::int64_t contemporaneous_tolerance_ns{0};
};

// Provenance for every forced end-of-test liquidation (Phase 07.2). One record
// per EOT Fill, whether it used a contemporaneous mark or an explicit
// approximation.
struct EotLiquidationAudit {
    std::uint32_t instrument_id{0};
    std::string   raw_symbol;
    int           units{0};                  // signed units liquidated
    std::int64_t  execution_ts_ns{0};        // == fill.ts_fill_ns == final engine event time
    std::int64_t  reference_price_ts_ns{0};  // ts of the bar the price came from
    std::int64_t  reference_age_ns{0};       // execution_ts_ns - reference_price_ts_ns (>= 0)
    double        reference_price{0.0};
    bool          is_contemporaneous{false}; // reference_age_ns <= eot.contemporaneous_tolerance_ns
    bool          is_approximation{false};   // true iff a stale mark was force-used (ForceApproximateStaleClose)
};

// Mark provenance for an open position that survives the end of the test (either
// EndOfTestPolicy::LeaveOpen, or a stale position under EotStalePolicy::LeaveOpen).
// Phase 08 risk consumes the mark age.
struct OpenPositionMark {
    std::uint32_t instrument_id{0};
    std::string   raw_symbol;
    int           units{0};                  // signed
    double        avg_entry_price{0.0};
    double        mark_price{0.0};            // last observed close of this contract
    std::int64_t  mark_ts_ns{0};             // ts of that bar
    std::int64_t  mark_age_ns{0};            // final_engine_ts - mark_ts_ns (>= 0)
    bool          is_stale{false};           // mark_age_ns > eot.contemporaneous_tolerance_ns
    double        unrealized_pnl_usd{0.0};   // (mark_price - avg_entry) * multiplier * units
};

// How to price the close-leg of a futures roll when the feed has moved to the
// new contract but there is no CONTEMPORANEOUS bar of the old raw contract at
// the execution instant. An execution decision at timestamp T must never create
// a retroactive fill at an earlier timestamp/price (CLAUDE.md backtest rule 1),
// so a stale price is never silently used.
enum class RollPriceFallback : int {
    // DEFAULT (Phase 07.1). Do not roll on this bar: keep the old-contract
    // position and retry on the next bar of the root
    // (BacktestResult::rolls_deferred). Missing execution-time raw-contract data
    // is NOT approximated by default in a reliability-oriented research engine --
    // the target is simply not reached until a priceable roll is possible.
    RejectDefer = 0,
    // EXPLICIT OPT-IN APPROXIMATION. Close the old contract at its LAST OBSERVED
    // price, still stamped at the execution timestamp T (never retroactive). Each
    // use appends a RollFallbackAudit (execution_ts, reference_price_ts,
    // reference_age_ns, policy), increments BacktestResult::rolls_priced_stale,
    // and is tagged "roll_stale_close" -- never presented as contemporaneous.
    StaleObservedClose = 1,
};

struct RollExecutionPolicy {
    RollPriceFallback fallback{RollPriceFallback::RejectDefer};
    // Maximum age (T - old_bar_ts) at which an old-contract bar still counts as a
    // genuine contemporaneous close (tagged "roll", not flagged). 0 == require an
    // exact-timestamp overlap bar, matching the Phase 04.5 same_timestamp basis.
    std::int64_t contemporaneous_tolerance_ns{0};

    // ADDITIVE (Phase 13.5C). Caller-supplied contemporaneous CLOSE of an
    // OUTGOING raw contract at a roll's execution instant, keyed
    // (instrument_id, ts_event_ns) -> close. Sourced from the already-acquired
    // Phase 04.5 Stage-B roll-overlap raw bars. Consulted ONLY by the roll
    // close-leg, ONLY when a key matches the outgoing contract at EXACTLY the
    // execution event time -- then that price is the same-timestamp close (age 0,
    // the existing `roll` / rolls_priced_contemporaneous path). No match => the
    // frozen RejectDefer / contemporaneous-tolerance logic is entirely unchanged.
    //
    // These entries are NOT MarketEvents: they never enter order_bar_events / the
    // event stream, never set MarketState.active_instrument_id, never advance
    // bars_seen, never reach Strategy::decide / StrategyContext / BarHistory / a
    // target row, never change latency or daily-equity sampling. Empty => the
    // engine is byte-for-byte the frozen reference path.
    std::map<std::pair<std::uint32_t, std::int64_t>, double> close_marks{};
};

// ADDITIVE (this phase). ETF corporate actions (corporate_actions.hpp) applied
// during a bar replay, mirroring RollExecutionPolicy::close_marks's shape and
// guarantees exactly:
//
//   * keyed EXACTLY like close_marks -- (instrument_id, ts_event_ns) -> action.
//     A key is consulted ONLY when it matches a PRIMARY event's
//     (instrument_id, ts_event_ns) exactly (never "first bar at/after"); a key
//     with no matching bar in the replayed window is simply never applied, the
//     same "no match -> untouched" contract close_marks already has.
//   * applied immediately BEFORE that event's own bar-open mark
//     (portfolio.observe_mark) and before that bar's own queued-intent
//     execution, so entitlement / rebasing always reflects the position held
//     coming INTO that timestamp, never contaminated by that same event's own
//     price action.
//   * order_bar_events already rejects a duplicate (instrument_id,
//     ts_event_ns) bar, so each key is matched against the primary event
//     stream at most once per run -- exactly-once application falls out of
//     that existing invariant and needs no separate "already applied"
//     tracking set.
//   * these entries are NOT MarketEvents: they never enter the event stream,
//     MarketState, bars_seen, Strategy::decide, BarHistory, latency, or
//     daily-equity sampling.
//   * empty (the default, on every pre-existing EngineConfig{}) => the engine
//     is byte-for-byte the frozen reference path -- the same "empty means
//     untouched" guarantee as RollExecutionPolicy::close_marks.
struct CorporateActionsConfig {
    std::map<std::pair<std::uint32_t, std::int64_t>, SplitAction> splits{};
    std::map<std::pair<std::uint32_t, std::int64_t>, CashDistribution> ex_date_distributions{};
    std::map<std::pair<std::uint32_t, std::int64_t>, CashDistribution> payment_date_distributions{};
};

// Provenance for a roll close-leg priced by the StaleObservedClose fallback. The
// stale reference came from an EARLIER bar than the fill; this record preserves
// that gap rather than overloading the frozen Fill schema (Phase 07.1).
struct RollFallbackAudit {
    std::uint32_t     instrument_id{0};      // the old raw contract that was closed
    std::string       raw_symbol;
    std::int64_t      execution_ts_ns{0};    // engine event time T == fill.ts_fill_ns
    std::int64_t      reference_price_ts_ns{0};  // ts of the bar the stale price came from
    std::int64_t      reference_age_ns{0};   // execution_ts_ns - reference_price_ts_ns (> 0)
    double            reference_price{0.0};
    RollPriceFallback fallback_policy{RollPriceFallback::StaleObservedClose};
};

struct EngineConfig {
    EndOfTestPolicy     end_of_test{EndOfTestPolicy::ForceLiquidateFinalClose};
    EotExecutionPolicy  eot{};
    RollExecutionPolicy roll{};

    // ADDITIVE (this phase). See CorporateActionsConfig above. Empty by default
    // on every EngineConfig{} -- zero blast radius on the frozen Futures path.
    CorporateActionsConfig corporate_actions{};

    // Deterministic execution latency: a Signal decided on bar i of a root
    // executes on bar i + 1 + latency_bars of that root. 0 == next bar (Phase
    // 06). No random latency, ever.
    int latency_bars{0};

    // Phase 13.1: caller-supplied canonical validation-day boundary timestamps,
    // STRICTLY ASCENDING. When non-empty, BacktestResult::daily_equity is emitted
    // as one PortfolioAccountant equity snapshot per boundary (see
    // DailyEquityPoint). Python derives these from the SessionCalendar
    // `trading_day` so one CME exchange trading day -> one validation
    // observation, even when its bars span two UTC dates. Empty => legacy
    // UTC-midnight bucketing (diagnostics only).
    std::vector<std::int64_t> validation_day_boundaries_ns;

    // Used ONLY by the convenience constructor (the one without an explicit
    // ExecutionSimulator) to build a default BarExecutionSimulator. Ignored when
    // a simulator is injected.
    ExecutionConfig execution{};
};

// ADDITIVE typed export (Phase 13 / 13.1). One end-of-day snapshot of the equity
// value the PortfolioAccountant ALREADY computes -- no accounting formula is
// added or changed. `equity_usd` == starting_capital + net_realized +
// unrealized, exactly PortfolioAccountant::snapshot().equity_usd.
//
// The day boundary is caller-controlled:
//   * EngineConfig::validation_day_boundaries_ns NON-EMPTY (Phase 13.1, the
//     canonical CME futures path): one point per supplied boundary timestamp,
//     sampled at the FIRST engine event whose ts_event_ns reaches that boundary;
//     `session_day_index` is the boundary ordinal (0,1,2,...). Python supplies
//     one boundary per canonical `trading_day` (the ts of that day's last
//     eligible event, from the SessionCalendar).
//   * EMPTY (legacy / diagnostics only -- NOT canonical for CME futures): one
//     point per UTC-midnight bucket
//     (session_day_index == session_day_index(ts_ns, day_boundary_offset_ns)).
// In both cases the FINAL day's point is (re)sampled AFTER end-of-test
// liquidation so its realized PnL is complete. Python differences consecutive
// points into a daily validation return series (docs/RELIABILITY_VALIDATION.md);
// it must not reconstruct fills / positions / equity itself.
struct DailyEquityPoint {
    std::int64_t session_day_index{0};      // boundary ordinal, or UTC day index (legacy)
    std::int64_t ts_ns{0};                  // engine event ts this snapshot was sampled at
    double       equity_usd{0.0};           // starting_capital + net_realized + unrealized
    double       net_realized_pnl_usd{0.0}; // gross_realized - costs (Fill-derived)
    double       unrealized_pnl_usd{0.0};   // mark-to-market on open positions
    double       costs_usd{0.0};            // cumulative commissions (Fill-derived)
    std::size_t  fills_cumulative{0};       // fills generated through this day
    std::size_t  bars_cumulative{0};        // bars processed through this day
};

// A "trade" in the result is one ClosedTrade == one close/reduce/flip event on a
// single raw contract (see position_ledger.hpp). A position flip is one closing
// ClosedTrade plus a new open position; a roll is a "roll"-reason ClosedTrade
// plus a re-open in the new contract.
struct BacktestResult {
    // counts
    std::size_t events_processed{0};
    std::size_t bars_processed{0};
    std::size_t signals_generated{0};
    std::size_t orders_generated{0};
    std::size_t fills_generated{0};
    std::size_t closed_trades{0};
    std::size_t rolls{0};                       // roll close-legs executed
    std::size_t rolls_priced_contemporaneous{0};// closed at a same-instant old-contract bar
    // ADDITIVE (Phase 13.5C). Subset of rolls_priced_contemporaneous whose
    // same-timestamp outgoing-contract close came from RollExecutionPolicy::
    // close_marks (the acquired roll-overlap raw bars) rather than a primary
    // MarketEvent. Invariant: 0 <= rolls_priced_auxiliary_marks
    //                            <= rolls_priced_contemporaneous.
    std::size_t rolls_priced_auxiliary_marks{0};
    std::size_t rolls_priced_stale{0};          // StaleObservedClose fallback: last observed price, ts stamped at T
    std::size_t rolls_deferred{0};              // RejectDefer: roll postponed to a later bar
    std::size_t non_fills{0};                   // deterministic non-fill execution reports
    std::size_t eot_positions_left_open_stale{0};  // ForceLiquidate declined these: stale mark, EotStalePolicy::LeaveOpen

    // official realized PnL (from Fills only)
    double gross_realized_pnl_usd{0.0};
    double costs_usd{0.0};
    double net_realized_pnl_usd{0.0};

    // realized + unrealized equity curve
    double max_drawdown_usd{0.0};
    double unrealized_pnl_usd_at_end{0.0};
    double net_equity_usd_at_end{0.0};  // net_realized + unrealized_at_end

    EndOfTestPolicy end_of_test_policy{EndOfTestPolicy::ForceLiquidateFinalClose};

    // audit trail (deterministic, chronological)
    std::vector<Order>              orders;
    std::vector<RiskDecision>       risk_decisions;
    std::vector<Fill>               fills;
    std::vector<ClosedTrade>        trades;
    std::vector<RollFallbackAudit>  roll_fallback_audit;  // one per StaleObservedClose roll close
    std::vector<EotLiquidationAudit> eot_liquidations;    // one per forced EOT liquidation

    // end state
    std::vector<LedgerPosition>  final_positions;      // units != 0, ordered by instrument_id
    std::vector<OpenPositionMark> final_position_marks; // mark provenance, 1:1 with final_positions
    std::vector<std::string>     contracts_traded;     // sorted unique raw symbols

    // Phase 08 portfolio accounting -- a valuation snapshot at the final engine
    // event time. Realized PnL / positions here are the SAME Fill-derived ledger
    // as the fields above; cash / equity / exposure / margin / leverage /
    // drawdown are the added portfolio view. Hard risk limits are enforced only
    // when the RiskManager is a PortfolioRiskManager (or otherwise opts in via
    // risk_config()); otherwise this block is reporting-only.
    PortfolioState portfolio_at_end{};
    std::size_t    risk_rejects{0};   // RiskDecisions with verdict == Reject
    std::size_t    risk_resizes{0};   // RiskDecisions with verdict == Resize

    // ADDITIVE (Phase 13): chronological end-of-session-day equity trace, one
    // point per distinct session-day bucket that had at least one engine event.
    // Strictly increasing in session_day_index. Empty iff no events.
    std::vector<DailyEquityPoint> daily_equity;
};

// ---- deterministic ingestion --------------------------------------------
//
// Turn raw input bars into the ordered MarketEvent stream the engine replays.
// The order is CANONICAL and independent of the caller's vector order, the file
// order, or any hash iteration:
//
//   1. every bar is validated (instrument_id resolves through `registry`,
//      ts_event_ns > 0);
//   2. a duplicate (instrument_id, ts_event_ns) is a hard error -- never
//      resolved by input order;
//   3. bars are ordered by (ts_event_ns, instrument_id) -- a total order for a
//      Bar-only stream;
//   4. a global monotonic `seq` (0, 1, 2, ...) is stamped AFTER ordering.
//
// `MarketEventOrder` stays (ts_event_ns, seq); after step 4 that order is
// identical to the canonical order. Throws `InvalidMarketEvent`.
std::vector<MarketEvent> order_bar_events(const std::vector<MarketBar>& bars,
                                          const ContractRegistry& registry);

class BacktestEngine {
public:
    // Convenience: build a default deterministic BarExecutionSimulator from
    // `config.execution`. Use this when you do not need to substitute the
    // simulator.
    BacktestEngine(const ContractRegistry& registry,
                   const ActiveContractResolver& resolver,
                   const RiskManager& risk,
                   EngineConfig config = {});

    // Explicit ExecutionSimulator injection (the sanctioned seam for a future
    // tick / MBP model). `config.execution` is ignored -- the simulator carries
    // its own ExecutionConfig. The reference must outlive the engine.
    BacktestEngine(const ContractRegistry& registry,
                   const ActiveContractResolver& resolver,
                   const RiskManager& risk,
                   const ExecutionSimulator& execution,
                   EngineConfig config = {});

    BacktestEngine(const BacktestEngine&) = delete;
    BacktestEngine& operator=(const BacktestEngine&) = delete;

    // Deterministic: `seq` is assigned by ingestion index, events are ordered by
    // (ts_event_ns, seq), and no step depends on hash / pointer / filesystem
    // order. Running the same `bars` twice yields an identical BacktestResult.
    BacktestResult run(const std::vector<MarketBar>& bars, const Strategy& strategy) const;

private:
    const ContractRegistry&       registry_;
    const ActiveContractResolver& resolver_;
    const RiskManager&            risk_;
    std::optional<BarExecutionSimulator> owned_execution_;  // set by the convenience ctor
    const ExecutionSimulator&     execution_;
    EngineConfig                  config_;
};

}  // namespace quant
