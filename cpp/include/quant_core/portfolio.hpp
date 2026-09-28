#pragma once

#include "quant_core/corporate_actions.hpp"
#include "quant_core/events.hpp"
#include "quant_core/margin.hpp"
#include "quant_core/position_ledger.hpp"

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace quant {

// Thrown by PortfolioAccountant::record_distribution_payment when no matching
// pending receivable (instrument_id + ex_date_ts_ns) exists -- a payment with no
// matching prior record_distribution_ex_date call is a caller bug, never a valid
// state to silently no-op or fabricate a receivable for.
class NoPendingDistributionError : public std::runtime_error {
public:
    explicit NoPendingDistributionError(const std::string& what) : std::runtime_error(what) {}
};

// ============================================================================
// Deterministic portfolio accounting (Phase 08)
// ============================================================================
//
// The Phase 06 PositionLedger is the minimal state for official realized PnL. It
// has no cash, equity, exposure, margin or mark-to-market. Phase 08 adds a
// PortfolioAccountant on top of a PositionLedger:
//
//   * it consumes ONLY validated Fill events (requirement 8) -- realized PnL and
//     positions come from the embedded ledger, so they are byte-identical to the
//     engine's ledger;
//   * it consumes valuation MARKS (bar closes) separately. A mark is NEVER a
//     Fill and can never become an execution price (requirement 2);
//   * realized and unrealized PnL are kept strictly separate (requirement 9);
//   * futures exposure uses ContractSpec economics: price x multiplier x
//     contracts, never quantity x price (requirement 4);
//   * every mark carries its timestamp, age and staleness so risk can refuse to
//     treat a stale value as fresh (requirement 7).
//
// Everything here is deterministic: std::map iteration is key-ordered, there is
// no clock, no RNG, no hash-order dependence.

// ---- cash / equity model (documented, explicit) -------------------------
//
// Futures post margin, they do not pay cash at entry. This models a standard
// futures account:
//
//   cash_usd   = starting_capital + net_realized_pnl            (variation margin
//                                                                on CLOSED trades)
//   equity_usd = cash_usd + unrealized_pnl                      (a.k.a. liquidating
//                                                                value / NLV)
//
// Unrealized PnL is open-trade equity; it is reported but NEVER folded into
// realized PnL or into cash.

struct PortfolioConfig {
    double       starting_capital_usd{100000.0};
    // A mark whose age (as_of_ts - mark_ts) exceeds this is `is_stale`. 0 == a
    // mark is stale unless it is exactly the as-of instant. Callers should set
    // this to their bar cadence / max acceptable session gap.
    std::int64_t mark_staleness_tolerance_ns{0};
    // Trading-day boundary for the daily-loss limit: day_index = floor((ts -
    // offset) / 86_400e9). 0 == UTC midnight. Set to the CME session reset
    // (~17:00 America/Chicago, i.e. 22:00/23:00 UTC) upstream where the session
    // calendar is known -- the deterministic C++ core does not carry a tz db.
    std::int64_t day_boundary_offset_ns{0};
};

// One held position, valued at its latest mark. `mark_present == false` means no
// valuation has been observed yet for a non-flat position; its notional then
// falls back to the average ENTRY price -- an ESTIMATE, not a current market
// value (`valuation_is_estimated == true`) -- unrealized PnL contributes 0, and
// the position is treated as stale.
//
// Exposure is direction-consistent for ANY finite mark price, including a
// negative one (CL has printed below zero). `gross_notional_usd` is always the
// non-negative magnitude; long/short is determined by POSITION DIRECTION
// (sign of `units`), NEVER by the sign of the price.
struct PositionExposure {
    std::uint32_t instrument_id{0};
    std::string   raw_symbol;
    std::string   root_symbol;
    int           units{0};                 // signed
    double        avg_entry_price{0.0};
    double        multiplier{0.0};          // USD PnL per 1.0 price move, from ContractSpec

    double        mark_price{0.0};
    std::int64_t  mark_ts_ns{0};
    std::int64_t  mark_age_ns{0};           // as_of_ts_ns - mark_ts_ns (>= 0)
    bool          mark_present{false};
    bool          mark_is_stale{false};     // !mark_present || mark_age_ns > tolerance
    bool          valuation_is_estimated{false};  // notional came from avg entry, not a mark

    double        gross_notional_usd{0.0};  // |units * valuation_price * multiplier|  (>= 0 always)
    double        signed_notional_usd{0.0}; // sign(units) * gross_notional_usd  (direction from units)
    double        unrealized_pnl_usd{0.0};  // (mark_price - avg_entry) * multiplier * units

    double        initial_margin_usd{0.0};      // per-contract initial x |units| (0 if unknown)
    double        maintenance_margin_usd{0.0};
    bool          margin_known{false};          // false => no MarginRequirement for this root
};

// Deterministic valuation snapshot of the whole portfolio at one instant.
struct PortfolioState {
    std::int64_t as_of_ts_ns{0};

    // cash / PnL / equity
    double starting_capital_usd{0.0};
    double gross_realized_pnl_usd{0.0};   // from Fills only
    double costs_usd{0.0};                // commissions from Fills only
    double net_realized_pnl_usd{0.0};     // gross_realized - costs
    double unrealized_pnl_usd{0.0};       // from marks; separate from realized, forever
    double cash_usd{0.0};                 // starting_capital + net_realized
    double equity_usd{0.0};               // cash + unrealized

    // exposure -- ContractSpec economics (|units| x |price| x multiplier), valid
    // for a negative futures price. Long/short is by POSITION DIRECTION, not the
    // sign of the price.
    double gross_exposure_usd{0.0};       // sum of per-position gross_notional (>= 0)
    double net_exposure_usd{0.0};         // long_exposure - short_exposure (== sum signed_notional)
    double long_exposure_usd{0.0};        // sum gross_notional over units > 0
    double short_exposure_usd{0.0};       // sum gross_notional over units < 0 (positive magnitude)
    int    gross_contracts{0};            // sum |units|
    int    net_contracts{0};              // sum units

    // leverage. gross_leverage uses the non-negative gross exposure; the
    // max_net_leverage check compares |net_leverage| against the limit so a large
    // net-SHORT book cannot slip under it on sign. `net_leverage` here is the
    // SIGNED ratio for reporting (negative == net short).
    double gross_leverage{0.0};           // gross_exposure / equity (0 when equity <= 0)
    double net_leverage{0.0};             // net_exposure / equity (signed; check uses |.|)

    // margin
    double initial_margin_usd{0.0};       // sum of per-position initial margin
    double maintenance_margin_usd{0.0};
    double margin_utilization_pct{0.0};   // initial_margin / equity; kUtilizationNoEquity if equity <= 0
    bool   margin_complete{true};         // false => a held position has no MarginRequirement

    // peak / drawdown on the (net_realized + unrealized) equity curve, peak
    // seeded at starting_capital.
    double peak_equity_usd{0.0};
    double drawdown_usd{0.0};             // peak_equity - equity (>= 0)
    double drawdown_pct{0.0};             // drawdown_usd / peak_equity (0 when peak <= 0)

    // deterministic daily-loss accounting
    std::int64_t session_day_index{0};
    double day_start_equity_usd{0.0};     // equity at the first observation of this day bucket
    double day_start_net_realized_usd{0.0};
    double day_realized_pnl_usd{0.0};     // net_realized - day_start_net_realized
    double day_equity_change_usd{0.0};    // equity - day_start_equity (mark-to-market, signed)

    // stale-mark surface for risk (requirement 7)
    bool         has_stale_mark{false};   // any held position with a stale/missing mark
    std::int64_t worst_mark_age_ns{0};
    // false <=> at least one open position has NO fresh mark (missing or stale),
    // so gross/net/long/short exposure includes an ESTIMATED leg (avg-entry
    // notional). Equal to !has_stale_mark; kept as a named "are these exposure
    // numbers a true current-market picture?" flag. Risk-increasing orders are
    // blocked while this is false under the default stale-mark policy.
    bool         valuation_complete{true};

    // TODO(phase-09+): correlated / cross-margin exposure. The MVP is additive by
    // root -- no netting between correlated roots (ES/NQ), no SPAN scanning, no
    // correlation haircut. A real portfolio risk model needs a correlation matrix
    // or SPAN-style risk arrays. This is deliberately NOT guessed here; the field
    // equals gross_exposure_usd until that model exists.
    double correlation_adjusted_gross_exposure_usd{0.0};

    // ---- ETF corporate-action income (Phase 6 pilot, additive) -----------
    //
    // Tracked completely separately from Fill-derived cash_usd / equity_usd.
    // cash_usd stays exactly starting_capital_usd + net_realized_pnl_usd and
    // equity_usd stays exactly cash_usd + unrealized_pnl_usd -- NEITHER folds in
    // distribution income or receivables. This is a deliberate, conservative
    // choice to keep the two income streams fully separable in the reported
    // state rather than silently merging them into one cash number; a future
    // phase can decide whether/how to compose them into a single "cash" figure.
    double distribution_receivable_usd{0.0};  // sum of pending (ex-date passed, unpaid) receivables
    double distribution_income_usd{0.0};      // cumulative distributions actually paid/received so far

    std::vector<PositionExposure> positions;  // units != 0, ordered by instrument_id
};

// margin_utilization_pct sentinel when equity <= 0 but margin is required: a
// finite, comparable, JSON-safe stand-in for "infinite utilisation".
inline constexpr double kUtilizationNoEquity = 1.0e9;

// day_index = floor((ts_ns - offset_ns) / 86_400e9), floor toward -inf.
std::int64_t session_day_index(std::int64_t ts_ns, std::int64_t day_boundary_offset_ns) noexcept;

class PortfolioAccountant {
public:
    PortfolioAccountant(PortfolioConfig config, const MarginModel* margin /* nullable */);

    // Consume a validated raw-contract Fill. Delegates position / realized-PnL
    // math to the embedded PositionLedger. `root_symbol` / `close_reason` are
    // engine context, exactly as PositionLedger::apply.
    void apply_fill(const Fill& fill, const std::string& root_symbol,
                    const std::string& close_reason);

    // Record a valuation observation for `instrument_id` (a bar open or close).
    // This NEVER creates a Fill and the price recorded here can never be used as
    // an execution price (requirement 2). Observations must be non-decreasing in
    // ts_ns; the latest observation for an instrument is its current mark.
    void observe_mark(std::uint32_t instrument_id, double price, std::int64_t ts_ns);

    // ETF corporate action: rebase an open position's units / avg_entry_price
    // for a split with zero economic PnL effect. Validates `action`, then
    // delegates to PositionLedger::apply_split, which is the sole owner of the
    // unit-count / cost-basis math (see its docs for the FractionalSplitQuantityError
    // representation limit). Any exception from the ledger propagates unchanged.
    void apply_split(const SplitAction& action);

    // ETF corporate action, ex-date leg: reads the CURRENT open position size
    // for `action.instrument_id` from the ledger at the moment this is called
    // and records a PENDING receivable (entitlement), sign-aware -- a short
    // position owes the distribution (a negative receivable). No-op if the
    // position is flat (units == 0): you cannot be entitled to a distribution
    // you do not hold. This does NOT change cash_usd.
    //
    // Causality is the CALLER's responsibility (Python orchestration), exactly
    // like observe_mark already documents: this is a deterministic replay
    // engine, not a causality-timing authority. Call this strictly before the
    // ex-date's own fills/marks are applied for the entitlement to reflect the
    // correct pre-ex-date holding.
    void record_distribution_ex_date(const CashDistribution& action);

    // ETF corporate action, pay-date leg: looks up the pending receivable keyed
    // by (action.instrument_id, action.ex_date_ts_ns) -- NOT the full struct
    // value, so a caller does not need to replay byte-identical
    // amount_per_share_usd / pay_date_ts_ns fields at payment time, only the
    // identifying instrument + ex-date. Moves its total_receivable_usd into the
    // cumulative distribution_income_usd_ (tracked completely separately from
    // Fill-derived realized PnL) and removes it from the pending set. Throws
    // NoPendingDistributionError if there is no matching prior
    // record_distribution_ex_date call -- a payment with nothing to settle is a
    // caller bug, never silently ignored.
    void record_distribution_payment(const CashDistribution& action);

    // Deterministic snapshot; marks are aged against `as_of_ts_ns`.
    PortfolioState snapshot(std::int64_t as_of_ts_ns) const;

    const PositionLedger& ledger() const noexcept { return ledger_; }

private:
    struct MarkObs {
        double       price{0.0};
        std::int64_t ts_ns{0};
    };

    // A recorded (ex-date passed, not yet paid) distribution entitlement.
    struct PendingReceivable {
        std::uint32_t instrument_id{0};
        std::int64_t  ex_date_ts_ns{0};
        int           units_entitled{0};        // signed; snapshotted at the ex-date call
        double        amount_per_share_usd{0.0};
        double        total_receivable_usd{0.0}; // units_entitled * amount_per_share_usd (sign-aware)
    };

    // Recompute running peak-equity and roll the day bucket. Called after every
    // fill and every mark, at the observation's timestamp.
    void refresh_running_stats(std::int64_t ts_ns);
    double equity_at(std::int64_t as_of_ts_ns) const;

    PortfolioConfig                   config_;
    const MarginModel*                margin_{nullptr};
    PositionLedger                    ledger_;
    std::map<std::uint32_t, MarkObs>  marks_;

    double       peak_equity_usd_{0.0};
    bool         day_initialised_{false};
    std::int64_t day_index_{0};
    double       day_start_equity_usd_{0.0};
    double       day_start_net_realized_usd_{0.0};
    std::int64_t last_stats_ts_ns_{0};

    // ETF corporate-action distribution income (Phase 6 pilot). std::map keyed
    // by (instrument_id, ex_date_ts_ns) -- deterministic key order, never
    // unordered_map, matching this file's existing determinism convention.
    std::map<std::pair<std::uint32_t, std::int64_t>, PendingReceivable> pending_distributions_;
    double distribution_income_usd_{0.0};  // cumulative paid distributions; Fill-PnL-independent
};

}  // namespace quant
