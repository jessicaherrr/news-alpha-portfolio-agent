#include "quant_core/engine.hpp"

#include "quant_core/bar_history.hpp"
#include "quant_core/contract.hpp"
#include "quant_core/domain_errors.hpp"
#include "quant_core/domain_model.hpp"
#include "quant_core/execution_simulator.hpp"
#include "quant_core/fill.hpp"
#include "quant_core/ids.hpp"
#include "quant_core/strategy_context.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace quant {
namespace {

struct MarkBar {
    std::int64_t ts_ns{0};
    double       close{0.0};
};

struct PendingIntent {
    Signal      sig;
    std::size_t ready_at_bars{0};  // execute once RootState::bars_seen reaches this
};

struct RootState {
    std::vector<MarketBar>   history;
    MarketState              market_state{};
    std::deque<PendingIntent> pending;      // FIFO; N-bar latency delays ready_at_bars
    std::size_t              bars_seen{0};
    std::uint32_t            held_instrument_id{0};  // contract this root holds a non-zero position in
};

}  // namespace

std::vector<MarketEvent> order_bar_events(const std::vector<MarketBar>& bars,
                                         const ContractRegistry& registry) {
    // 1. validate every input bar (order-independent checks).
    for (const MarketBar& b : bars) {
        if (b.instrument_id == 0 || registry.find_by_instrument_id(b.instrument_id) == nullptr) {
            throw InvalidMarketEvent("order_bar_events: bar instrument_id " +
                                     std::to_string(b.instrument_id) +
                                     " does not resolve through the ContractRegistry");
        }
        if (b.ts_event_ns <= 0) {
            throw InvalidMarketEvent(
                "order_bar_events: bar ts_event_ns must be a positive UTC nanosecond value");
        }
    }

    // 2. canonical order: (ts_event_ns, instrument_id) -- independent of the
    //    caller's vector order. Sort an index permutation so equal keys are
    //    detectable as adjacency (and so we never rely on std::sort stability).
    std::vector<std::size_t> order(bars.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t c) {
        if (bars[a].ts_event_ns != bars[c].ts_event_ns) {
            return bars[a].ts_event_ns < bars[c].ts_event_ns;
        }
        return bars[a].instrument_id < bars[c].instrument_id;
    });

    // 3. reject a duplicate (instrument_id, ts_event_ns) rather than letting
    //    input order decide which one wins.
    std::vector<MarketEvent> events;
    events.reserve(bars.size());
    for (std::size_t k = 0; k < order.size(); ++k) {
        const MarketBar& b = bars[order[k]];
        if (k > 0) {
            const MarketBar& prev = bars[order[k - 1]];
            if (prev.ts_event_ns == b.ts_event_ns && prev.instrument_id == b.instrument_id) {
                throw InvalidMarketEvent(
                    "order_bar_events: duplicate bar for instrument_id " +
                    std::to_string(b.instrument_id) + " at ts " + std::to_string(b.ts_event_ns) +
                    " -- ambiguous ingestion order");
            }
        }
        MarketEvent ev;
        ev.type          = MarketEventType::Bar;
        ev.ts_event_ns   = b.ts_event_ns;
        ev.seq           = static_cast<SeqNum>(k);  // 4. global monotonic seq AFTER ordering
        ev.instrument_id = b.instrument_id;
        ev.open = b.open;
        ev.high = b.high;
        ev.low  = b.low;
        ev.close = b.close;
        ev.volume = b.volume;
        events.push_back(ev);
    }
    return events;
}

// ---- constructors ------------------------------------------------------

BacktestEngine::BacktestEngine(const ContractRegistry& registry,
                               const ActiveContractResolver& resolver,
                               const RiskManager& risk,
                               EngineConfig config)
    : registry_(registry),
      resolver_(resolver),
      risk_(risk),
      owned_execution_(std::in_place, registry, config.execution),
      execution_(*owned_execution_),
      config_(std::move(config)) {}

BacktestEngine::BacktestEngine(const ContractRegistry& registry,
                               const ActiveContractResolver& resolver,
                               const RiskManager& risk,
                               const ExecutionSimulator& execution,
                               EngineConfig config)
    : registry_(registry),
      resolver_(resolver),
      risk_(risk),
      owned_execution_(std::nullopt),
      execution_(execution),
      config_(std::move(config)) {}

// ---- the deterministic run -------------------------------------------

BacktestResult BacktestEngine::run(const std::vector<MarketBar>& bars,
                                   const Strategy& strategy) const {
    BacktestResult result;
    result.end_of_test_policy = config_.end_of_test;

    // ---- deterministic ingestion: validate, canonical-order, stamp seq -------
    std::vector<MarketEvent> events = order_bar_events(bars, registry_);
    std::sort(events.begin(), events.end(), MarketEventOrder{});

    // ADDITIVE (Phase 13.5C): validate the auxiliary roll-close marks. Each key's
    // instrument_id must resolve through the registry, the timestamp must be
    // positive, and the price must be finite (it MAY be <= 0 -- CL front-month
    // traded below zero on 2020-04-20). The marks never enter the event stream.
    for (const auto& [key, px] : config_.roll.close_marks) {
        const auto& [inst_id, ts_ns] = key;
        if (inst_id == 0 || registry_.find_by_instrument_id(inst_id) == nullptr) {
            throw ContractResolutionError(
                "BacktestEngine: roll close_marks instrument_id " + std::to_string(inst_id) +
                " does not resolve through the ContractRegistry");
        }
        // The price may be <= 0 (CL front-month traded below zero on 2020-04-20);
        // only a non-finite price or a non-positive timestamp is malformed.
        if (ts_ns <= 0 || !std::isfinite(px)) {
            throw ExecutionDomainError(
                "BacktestEngine: roll close_marks entry for instrument_id " +
                std::to_string(inst_id) + " at ts " + std::to_string(ts_ns) +
                " has a non-positive ts or non-finite price");
        }
    }

    // ADDITIVE (this phase): validate the corporate-actions config up front,
    // exactly like the roll close_marks validation above -- each key's
    // instrument_id must resolve through the registry, and the action itself
    // must be well-formed AND its own fields must exactly match the map key it
    // is stored under (a caller cannot silently mismatch a key vs the struct's
    // embedded timestamp).
    for (const auto& [key, action] : config_.corporate_actions.splits) {
        const auto& [inst_id, ts_ns] = key;
        if (inst_id == 0 || registry_.find_by_instrument_id(inst_id) == nullptr) {
            throw ContractResolutionError(
                "BacktestEngine: corporate_actions.splits instrument_id " +
                std::to_string(inst_id) + " does not resolve through the ContractRegistry");
        }
        if (!action.is_valid() || action.instrument_id != inst_id ||
            action.effective_ts_ns != ts_ns) {
            throw ExecutionDomainError(
                "BacktestEngine: corporate_actions.splits entry for instrument_id " +
                std::to_string(inst_id) + " at ts " + std::to_string(ts_ns) +
                " is invalid or its fields do not match its map key");
        }
    }
    for (const auto& [key, action] : config_.corporate_actions.ex_date_distributions) {
        const auto& [inst_id, ts_ns] = key;
        if (inst_id == 0 || registry_.find_by_instrument_id(inst_id) == nullptr) {
            throw ContractResolutionError(
                "BacktestEngine: corporate_actions.ex_date_distributions instrument_id " +
                std::to_string(inst_id) + " does not resolve through the ContractRegistry");
        }
        if (!action.is_valid() || action.instrument_id != inst_id || action.ex_date_ts_ns != ts_ns) {
            throw ExecutionDomainError(
                "BacktestEngine: corporate_actions.ex_date_distributions entry for "
                "instrument_id " + std::to_string(inst_id) + " at ts " + std::to_string(ts_ns) +
                " is invalid or its fields do not match its map key");
        }
    }
    for (const auto& [key, action] : config_.corporate_actions.payment_date_distributions) {
        const auto& [inst_id, ts_ns] = key;
        if (inst_id == 0 || registry_.find_by_instrument_id(inst_id) == nullptr) {
            throw ContractResolutionError(
                "BacktestEngine: corporate_actions.payment_date_distributions instrument_id " +
                std::to_string(inst_id) + " does not resolve through the ContractRegistry");
        }
        if (!action.is_valid() || action.instrument_id != inst_id ||
            action.pay_date_ts_ns != ts_ns) {
            throw ExecutionDomainError(
                "BacktestEngine: corporate_actions.payment_date_distributions entry for "
                "instrument_id " + std::to_string(inst_id) + " at ts " + std::to_string(ts_ns) +
                " is invalid or its fields do not match its map key");
        }
    }

    const int latency_bars = std::max(0, config_.latency_bars);

    // ---- engine state (all deterministic containers / counters) -------------
    std::map<std::string, RootState>          roots;
    PositionLedger                            ledger;
    // Phase 08 portfolio accounting. Consumes ONLY validated Fills + bar-close
    // marks. Its embedded ledger mirrors `ledger` so realized PnL is identical.
    const PortfolioConfig portfolio_cfg =
        risk_.portfolio_config() != nullptr ? *risk_.portfolio_config() : PortfolioConfig{};
    PortfolioAccountant                       portfolio(portfolio_cfg, risk_.margin_model());
    MonotonicId                               signal_ids;
    MonotonicId                               order_ids;
    MonotonicId                               fill_ids;
    std::unordered_map<std::uint32_t, double>  mark_close;   // value lookup only
    std::unordered_map<std::uint32_t, MarkBar> last_bar;     // value lookup only
    std::set<std::string>                      traded;
    double                                     peak_equity = 0.0;

    // ADDITIVE (Phase 13 / 13.1): end-of-day equity trace. When the caller
    // supplies canonical validation-day boundaries (from the SessionCalendar
    // `trading_day`), emit one snapshot per boundary at the first event that
    // reaches it. Otherwise fall back to legacy UTC-midnight bucketing.
    const std::int64_t day_offset_ns = portfolio_cfg.day_boundary_offset_ns;
    const std::vector<std::int64_t>& vday = config_.validation_day_boundaries_ns;
    const bool use_vday = !vday.empty();
    std::size_t vb_ptr = 0;
    std::optional<DailyEquityPoint> current_day;  // legacy UTC path only
    auto make_point = [&](std::int64_t idx, std::int64_t ts_ns) {
        const PortfolioState ds = portfolio.snapshot(ts_ns);
        return DailyEquityPoint{idx, ts_ns, ds.equity_usd, ds.net_realized_pnl_usd,
                                ds.unrealized_pnl_usd, ds.costs_usd,
                                result.fills_generated, result.bars_processed};
    };
    auto sample_day = [&](std::int64_t ts_ns) {
        if (use_vday) {
            while (vb_ptr < vday.size() && vday[vb_ptr] <= ts_ns) {
                result.daily_equity.push_back(
                    make_point(static_cast<std::int64_t>(vb_ptr), ts_ns));
                ++vb_ptr;
            }
            return;
        }
        const std::int64_t d = session_day_index(ts_ns, day_offset_ns);
        if (current_day.has_value() && current_day->session_day_index != d) {
            result.daily_equity.push_back(*current_day);
            current_day.reset();
        }
        current_day = make_point(d, ts_ns);
    };

    // The engine event time currently being processed. Every Fill produced while
    // this is set must be timestamped >= it (Phase 07.1 causality invariant); for
    // the bar-based model it is always == it. Persists into the end-of-test block
    // (holds the final engine event time there).
    std::int64_t current_event_ts_ns = 0;

    auto mark_equity = [&]() {
        const double equity = ledger.realized_pnl_usd() - ledger.costs_usd() +
                              ledger.unrealized_pnl_usd(mark_close);
        peak_equity = std::max(peak_equity, equity);
        result.max_drawdown_usd = std::max(result.max_drawdown_usd, peak_equity - equity);
    };

    // Build one Market Order for `signed_delta`, gate it through the RiskManager,
    // then hand the approved order to the ExecutionSimulator. The simulator owns
    // the fill price; the engine only supplies the execution-time bar (or, for a
    // forced close, an explicit reference price via `base_req`). This is the ONLY
    // place the engine turns intent into an executed contract.
    auto submit = [&](const std::string& root, SignalId signal_id, const ContractSpec& spec,
                      int signed_delta, const ExecutionRequest& base_req) {
        if (signed_delta == 0) return;

        Order ord;
        ord.order_id      = order_ids.next();
        ord.signal_id     = signal_id;
        ord.ts_created_ns = base_req.ts_ns;
        ord.instrument_id = spec.instrument_id;
        ord.raw_symbol    = spec.raw_symbol;
        ord.side          = signed_delta > 0 ? Side::Buy : Side::Sell;
        ord.quantity      = std::abs(signed_delta);
        ord.order_type    = OrderType::Market;
        ord.tif           = TimeInForce::Day;
        validate_order(ord, registry_);
        result.orders.push_back(ord);
        ++result.orders_generated;

        // ---- mandatory risk gate. Build the Phase 08 review context from
        //      PRE-trade portfolio state so hard limits (daily loss, drawdown,
        //      leverage, margin utilisation, position caps, stale marks) can be
        //      enforced deterministically. A Phase 05/06 RiskManager ignores the
        //      context and behaves exactly as before.
        const PortfolioState pf_pre = portfolio.snapshot(current_event_ts_ns);
        RiskReviewContext ctx;
        ctx.as_of_ts_ns            = current_event_ts_ns;
        ctx.portfolio              = &pf_pre;
        ctx.spec                   = &spec;
        ctx.margin                 = risk_.margin_model();
        ctx.current_position_units = ledger.units(spec.instrument_id);
        int root_units = 0;
        for (const LedgerPosition& lp : ledger.open_positions()) {
            if (lp.root_symbol == root) root_units += lp.units;
        }
        ctx.current_root_units = root_units;
        {
            double       ref_px = base_req.bar.open;
            std::int64_t ref_ts = base_req.ts_ns;
            if (base_req.administrative) {
                ref_px = base_req.ref_price_override.value_or(base_req.bar.open);
                const auto it = last_bar.find(spec.instrument_id);
                ref_ts = (it != last_bar.end()) ? it->second.ts_ns : base_req.ts_ns;
            }
            ctx.reference_price          = ref_px;
            ctx.reference_price_ts_ns    = ref_ts;
            ctx.reference_price_age_ns   = std::max<std::int64_t>(0, current_event_ts_ns - ref_ts);
            ctx.reference_price_stale =
                ctx.reference_price_age_ns > portfolio_cfg.mark_staleness_tolerance_ns;
            ctx.reference_price_present  = true;
        }

        const RiskDecision decision = risk_.review(ord, ctx);  // mandatory gate
        result.risk_decisions.push_back(decision);
        if (decision.verdict == RiskVerdict::Reject) {
            ++result.risk_rejects;
            return;  // gated: no fill
        }
        if (decision.verdict == RiskVerdict::Resize) ++result.risk_resizes;
        const Order approved = apply_risk_decision(ord, decision);
        if (approved.quantity <= 0) return;

        const FillId fid = fill_ids.peek();
        const ExecReport rep = execution_.execute(approved, base_req, fid);
        if (rep.outcome != ExecOutcome::Filled || !rep.fill.has_value()) {
            ++result.non_fills;   // deterministic non-fill: recorded, no ledger change
            return;
        }
        fill_ids.next();          // commit the id only now -> fill_id stays dense
        const Fill& fill = *rep.fill;

        // Phase 07.1 causality invariant: no path -- strategy leg, roll close-leg,
        // roll fallback, end-of-test liquidation, or simulator -- may backdate a
        // Fill before the engine event time being processed.
        if (fill.ts_fill_ns < current_event_ts_ns) {
            throw ExecutionDomainError(
                "BacktestEngine: Fill " + trace_tag('F', fill.fill_id) + " on '" + fill.raw_symbol +
                "' is timestamped " + std::to_string(fill.ts_fill_ns) +
                " which precedes the current engine event time " +
                std::to_string(current_event_ts_ns) +
                " -- retroactive fills are forbidden (Phase 07.1 causality invariant)");
        }

        result.fills.push_back(fill);
        ++result.fills_generated;
        traded.insert(fill.raw_symbol);

        const auto applied = ledger.apply(fill, root, base_req.reason);
        portfolio.apply_fill(fill, root, base_req.reason);  // Fill-only accounting
        if (applied.closed.has_value()) {
            result.trades.push_back(*applied.closed);
            ++result.closed_trades;
        }
    };

    // Execute one queued intent at event `ev`. Returns false ONLY when a roll
    // close-leg could not be priced under RejectDefer -- the caller then requeues
    // the intent for the next bar of the root.
    auto execute_intent = [&](const std::string& root, RootState& rs, const MarketEvent& ev,
                              const Signal& sig) -> bool {
        const int target = static_cast<int>(std::llround(sig.target_units));
        const std::uint32_t held_id   = rs.held_instrument_id;
        const std::uint32_t active_id = rs.market_state.active_instrument_id;

        // ---- roll close-leg: the held contract differs from the execution-time
        //      active contract. Close the OLD raw contract explicitly -- never
        //      relabel it -- priced from EXECUTION-TIME market data for that raw
        //      contract, never a retroactive fill (CLAUDE.md backtest rule 1).
        if (held_id != 0 && held_id != active_id && ledger.units(held_id) != 0) {
            const ContractSpec& old_spec = registry_.by_instrument_id(held_id);
            const int          old_units = ledger.units(held_id);

            // The outgoing contract's reference close for the close-leg. Normally
            // its last primary bar; if an auxiliary same-timestamp overlap close
            // was supplied for EXACTLY this execution instant (Phase 13.5C), use
            // that -- it is a real outgoing-contract traded price at T, so the
            // close-leg prices on the Phase 04.5 same-timestamp basis (age 0)
            // instead of deferring. Absent => the frozen path below is unchanged.
            MarkBar    ob              = last_bar.at(held_id);  // present: held only after a bar
            bool       priced_from_aux = false;
            if (const auto it = config_.roll.close_marks.find(
                    std::pair<std::uint32_t, std::int64_t>{held_id, ev.ts_event_ns});
                it != config_.roll.close_marks.end()) {
                ob              = MarkBar{ev.ts_event_ns, it->second};
                priced_from_aux = true;
            }
            const std::int64_t age = ev.ts_event_ns - ob.ts_ns;

            // FAIL-SAFE (Phase 07.1): the held contract is already non-tradable at
            // this execution event and no earlier event closed it. Refuse to
            // manufacture a past Fill at the contract's expiry instant -- that is
            // retroactive and breaks event-sourced causality. Scheduled expiry /
            // settlement belongs to typed lifecycle events (Phase 08+).
            if (!old_spec.is_live_at(ev.ts_event_ns)) {
                throw ExecutionDomainError(
                    "BacktestEngine: held contract '" + old_spec.raw_symbol + "' (instrument_id " +
                    std::to_string(held_id) + ") is no longer tradable at engine event time " +
                    std::to_string(ev.ts_event_ns) + " (tradable_until " +
                    std::to_string(old_spec.tradable_until_ns()) +
                    ") and no earlier execution event closed it -- refusing to create a "
                    "retroactive expiry fill. Provide execution-time raw-contract data around "
                    "the roll, or add scheduled lifecycle events (Phase 08+).");
            }

            std::string reason;
            if (age >= 0 && age <= config_.roll.contemporaneous_tolerance_ns) {
                // a genuine contemporaneous old-contract bar (Phase 04.5 basis).
                reason = "roll";
                ++result.rolls_priced_contemporaneous;
                if (priced_from_aux) ++result.rolls_priced_auxiliary_marks;
            } else if (config_.roll.fallback == RollPriceFallback::RejectDefer) {
                // DEFAULT: missing execution-time raw-contract data is not
                // approximated -- postpone the roll to the next bar of the root.
                ++result.rolls_deferred;
                return false;  // caller requeues for the next bar of the root
            } else {
                // EXPLICIT opt-in approximation: last OBSERVED price, still
                // stamped at the execution time T (never retroactive). Record the
                // provenance so the stale source is auditable.
                reason = "roll_stale_close";
                ++result.rolls_priced_stale;
                result.roll_fallback_audit.push_back(RollFallbackAudit{
                    held_id, old_spec.raw_symbol, ev.ts_event_ns, ob.ts_ns,
                    ev.ts_event_ns - ob.ts_ns, ob.close,
                    RollPriceFallback::StaleObservedClose});
            }

            ExecutionRequest rreq;
            rreq.ts_ns              = ev.ts_event_ns;   // always the execution event time
            rreq.ref_price_override = ob.close;
            rreq.administrative     = true;
            rreq.reason             = reason;
            rreq.bar = MarketBar{ev.ts_event_ns, held_id, ob.close, ob.close,
                                 ob.close, ob.close, 0};
            submit(root, sig.signal_id, old_spec, -old_units, rreq);
            ++result.rolls;
        }

        // ---- resolve the active raw contract from EXECUTION-TIME market state,
        //      then the signal leg (Market order, at this bar).
        const ContractSpec& active  = resolver_.resolve(root, rs.market_state);
        const int           current = ledger.units(active.instrument_id);
        const int           delta   = target - current;
        if (delta != 0) {
            ExecutionRequest sreq;
            sreq.ts_ns  = ev.ts_event_ns;
            sreq.reason = "signal";
            sreq.bar = MarketBar{ev.ts_event_ns, ev.instrument_id, ev.open, ev.high,
                                 ev.low, ev.close, ev.volume};
            submit(root, sig.signal_id, active, delta, sreq);
        }

        // which real contract does this root hold now?
        if (ledger.units(active.instrument_id) != 0) {
            rs.held_instrument_id = active.instrument_id;
        } else if (held_id != 0 && ledger.units(held_id) != 0) {
            rs.held_instrument_id = held_id;  // residual (e.g. a risk-resized roll close)
        } else {
            rs.held_instrument_id = 0;
        }
        return true;
    };

    // ---- deterministic event loop -----------------------------------------
    for (const MarketEvent& ev : events) {
        ++result.events_processed;

        const ContractSpec* espec = registry_.find_by_instrument_id(ev.instrument_id);
        if (espec == nullptr) {
            throw ContractResolutionError(
                "BacktestEngine: bar instrument_id " + std::to_string(ev.instrument_id) +
                " does not resolve through the ContractRegistry");
        }
        const std::string root = espec->root_symbol;
        RootState& rs = roots[root];

        current_event_ts_ns = ev.ts_event_ns;  // events are sorted ascending

        // 1. update per-root execution-time market state
        rs.market_state = MarketState::from_event(ev);

        // ADDITIVE (this phase): corporate actions (splits / cash
        // distributions), consulted ONLY for an EXACT (instrument_id,
        // ts_event_ns) match against THIS primary event -- see
        // CorporateActionsConfig's docs (engine.hpp) for the full contract.
        // Applied before this bar's own open mark / queued-intent execution
        // (below) so entitlement/rebasing always reflects the position held
        // coming INTO this timestamp, never contaminated by this same event's
        // own price action. A split is applied to BOTH the engine's own
        // `ledger` (result.final_positions / gross_realized_pnl_usd) and
        // `portfolio`'s embedded ledger (result.portfolio_at_end) -- they have
        // received byte-identical Fills up to this point (every fill applies to
        // both together in `submit`), so `apply_split` behaves identically on
        // both. order_bar_events already rejects a duplicate (instrument_id,
        // ts_event_ns) bar, so each key here is matched at most once across the
        // whole run.
        {
            const auto ca_key =
                std::pair<std::uint32_t, std::int64_t>{ev.instrument_id, ev.ts_event_ns};
            if (const auto it = config_.corporate_actions.splits.find(ca_key);
                it != config_.corporate_actions.splits.end()) {
                portfolio.apply_split(it->second);
                ledger.apply_split(it->second.instrument_id, it->second.ratio);
            }
            if (const auto it = config_.corporate_actions.ex_date_distributions.find(ca_key);
                it != config_.corporate_actions.ex_date_distributions.end()) {
                portfolio.record_distribution_ex_date(it->second);
            }
            if (const auto it = config_.corporate_actions.payment_date_distributions.find(ca_key);
                it != config_.corporate_actions.payment_date_distributions.end()) {
                portfolio.record_distribution_payment(it->second);
            }
        }

        // Mark the instrument to THIS bar's open before executing intent, so the
        // pre-trade portfolio valuation the risk gate sees is contemporaneous
        // (no look-ahead: the open is a real print at ts_event_ns) rather than
        // carrying the previous bar's stale close. The close is observed in
        // step 3 once the bar is complete.
        portfolio.observe_mark(ev.instrument_id, ev.open, ev.ts_event_ns);

        // 2. execute intent decided at an earlier bar (never this one). With
        //    latency_bars == 0 exactly the intent decided on the previous bar of
        //    this root is ready; a larger latency holds it longer.
        while (!rs.pending.empty() && rs.pending.front().ready_at_bars <= rs.bars_seen) {
            const PendingIntent intent = rs.pending.front();
            rs.pending.pop_front();
            if (!execute_intent(root, rs, ev, intent.sig)) {
                // Roll deferred (RejectDefer, no priceable roll). Only the most
                // recent target matters, so collapse the queue to a single
                // re-dated intent -- bounded memory even if the roll never prices
                // -- and stop draining it this bar.
                const Signal keep = rs.pending.empty() ? intent.sig : rs.pending.back().sig;
                rs.pending.clear();
                rs.pending.push_back(PendingIntent{keep, rs.bars_seen + 1});
                break;
            }
        }

        // 3. append to the bounded per-root history; refresh marks
        rs.history.push_back(MarketBar{ev.ts_event_ns, ev.instrument_id, ev.open, ev.high,
                                       ev.low, ev.close, ev.volume});
        mark_close[ev.instrument_id] = ev.close;
        last_bar[ev.instrument_id]   = MarkBar{ev.ts_event_ns, ev.close};
        portfolio.observe_mark(ev.instrument_id, ev.close, ev.ts_event_ns);  // bar-close valuation, never a Fill
        ++rs.bars_seen;
        ++result.bars_processed;
        sample_day(ev.ts_event_ns);  // end-of-day equity trace (additive, Phase 13)

        // 4. look-ahead-safe decision: history ends at THIS bar
        const ContractSpec* active_snap =
            registry_.find_by_instrument_id(rs.market_state.active_instrument_id);
        const BarHistoryView view(rs.history.data(), rs.history.size());
        const StrategyContext ctx(ev.ts_event_ns, ev, view, root, active_snap);
        std::optional<Signal> maybe_sig = strategy.decide(ctx);

        // 5. a returned Signal: the ENGINE stamps the id / authoritative fields,
        //    validates, and queues it. NO DECISION (std::nullopt, Phase 11.1):
        //    the engine allocates no signal_id, counts nothing, validates and
        //    queues nothing, and -- crucially -- leaves `rs.pending` untouched so
        //    an earlier latency-delayed intent is NOT cancelled and no target
        //    state changes. It simply proceeds.
        if (maybe_sig.has_value()) {
            Signal sig         = *std::move(maybe_sig);
            sig.signal_id      = signal_ids.next();
            sig.ts_decision_ns = ev.ts_event_ns;
            sig.root_symbol    = root;
            validate_signal(sig);
            rs.pending.push_back(PendingIntent{
                sig, rs.bars_seen + static_cast<std::size_t>(latency_bars)});
            ++result.signals_generated;
        }

        mark_equity();
    }

    // ---- end-of-test policy ---------------------------------------------
    //
    // Liquidation is processed AFTER the final event, so it happens at the final
    // engine event time `final_ts`. It stays causal -- every EOT Fill is stamped
    // at `final_ts`, never at an earlier per-contract bar. A position is force-
    // liquidated only when its last observed raw-contract bar is a CONTEMPORANEOUS
    // executable price (mark age <= eot.contemporaneous_tolerance_ns AND the
    // contract is tradable at final_ts). Otherwise the explicit EotStalePolicy
    // decides: leave it open (default), fail, or -- opt-in only -- approximate at
    // the stale close (flagged + audited). An old observed close is never
    // silently used as a current executable price.
    const std::int64_t final_ts = current_event_ts_ns;
    if (config_.end_of_test == EndOfTestPolicy::ForceLiquidateFinalClose) {
        const std::vector<LedgerPosition> at_eot = ledger.open_positions();  // snapshot
        for (const LedgerPosition& p : at_eot) {  // ordered by instrument_id
            const ContractSpec& spec = registry_.by_instrument_id(p.instrument_id);
            const MarkBar      lb  = last_bar.at(p.instrument_id);
            const std::int64_t age = final_ts - lb.ts_ns;
            const bool tradable    = spec.is_live_at(final_ts);
            const bool contemporaneous =
                tradable && age >= 0 && age <= config_.eot.contemporaneous_tolerance_ns;

            auto record_and_fill = [&](bool approximation) {
                result.eot_liquidations.push_back(EotLiquidationAudit{
                    p.instrument_id, spec.raw_symbol, p.units, final_ts, lb.ts_ns, age,
                    lb.close, /*is_contemporaneous=*/!approximation, /*is_approximation=*/approximation});
                const SignalId sid = signal_ids.next();  // traceability only
                ExecutionRequest ereq;
                ereq.ts_ns              = final_ts;   // causal: the final engine event time
                ereq.ref_price_override = lb.close;   // final available executable bar
                ereq.administrative     = true;
                ereq.reason             = approximation ? "eot_stale" : "eot";
                ereq.bar = MarketBar{final_ts, p.instrument_id, lb.close, lb.close, lb.close, lb.close, 0};
                submit(p.root_symbol, sid, spec, -p.units, ereq);
            };

            if (contemporaneous) {
                record_and_fill(/*approximation=*/false);
                continue;
            }

            // Not a contemporaneous executable price -> explicit stale policy.
            switch (config_.eot.stale) {
                case EotStalePolicy::LeaveOpen:
                    // reliability-safe default: keep it open; reported below in
                    // final_position_marks with its mark age.
                    break;
                case EotStalePolicy::Fail:
                    throw ExecutionDomainError(
                        "BacktestEngine: end-of-test force-liquidation of '" + spec.raw_symbol +
                        "' has no contemporaneous executable observation at the final engine "
                        "time " + std::to_string(final_ts) + " (mark age " + std::to_string(age) +
                        " ns > tolerance " + std::to_string(config_.eot.contemporaneous_tolerance_ns) +
                        (tradable ? "" : "; contract not tradable at that time") +
                        ") -- refusing to force a stale execution (EotStalePolicy::Fail).");
                case EotStalePolicy::ForceApproximateStaleClose:
                    if (!tradable) {
                        throw ExecutionDomainError(
                            "BacktestEngine: EotStalePolicy::ForceApproximateStaleClose cannot "
                            "liquidate '" + spec.raw_symbol + "' -- it is not tradable at the final "
                            "engine time " + std::to_string(final_ts) + " (tradable_until " +
                            std::to_string(spec.tradable_until_ns()) + ").");
                    }
                    record_and_fill(/*approximation=*/true);
                    break;
            }
        }
        mark_equity();
    }

    // ---- finalise ------------------------------------------------------
    result.final_positions           = ledger.open_positions();
    for (const LedgerPosition& p : result.final_positions) {
        OpenPositionMark m;
        m.instrument_id   = p.instrument_id;
        m.raw_symbol      = p.raw_symbol;
        m.units           = p.units;
        m.avg_entry_price = p.avg_entry_price;
        const auto it = last_bar.find(p.instrument_id);
        if (it != last_bar.end()) {
            m.mark_price        = it->second.close;
            m.mark_ts_ns        = it->second.ts_ns;
            m.mark_age_ns       = final_ts - it->second.ts_ns;
            m.is_stale          = m.mark_age_ns > config_.eot.contemporaneous_tolerance_ns;
            m.unrealized_pnl_usd = (m.mark_price - p.avg_entry_price) * p.multiplier * p.units;
        }
        result.final_position_marks.push_back(m);
        // Count only positions the ForceLiquidate policy DECLINED to liquidate
        // because their mark was stale (under EndOfTestPolicy::LeaveOpen every
        // open position is intentionally kept regardless of mark age).
        if (m.is_stale && config_.end_of_test == EndOfTestPolicy::ForceLiquidateFinalClose) {
            ++result.eot_positions_left_open_stale;
        }
    }
    result.contracts_traded.assign(traded.begin(), traded.end());  // std::set -> sorted
    result.gross_realized_pnl_usd    = ledger.realized_pnl_usd();
    result.costs_usd                 = ledger.costs_usd();
    result.net_realized_pnl_usd      = result.gross_realized_pnl_usd - result.costs_usd;
    result.unrealized_pnl_usd_at_end = ledger.unrealized_pnl_usd(mark_close);
    result.net_equity_usd_at_end     = result.net_realized_pnl_usd + result.unrealized_pnl_usd_at_end;
    result.portfolio_at_end          = portfolio.snapshot(final_ts);

    // Flush the final day, re-sampled AFTER end-of-test liquidation so its
    // realized PnL is complete (additive, Phase 13/13.1). No formula changes --
    // this reads portfolio_at_end, the same snapshot reported above.
    auto resample_last = [&](DailyEquityPoint& p) {
        const PortfolioState& fs = result.portfolio_at_end;
        p.ts_ns                = final_ts;
        p.equity_usd           = fs.equity_usd;
        p.net_realized_pnl_usd = fs.net_realized_pnl_usd;
        p.unrealized_pnl_usd   = fs.unrealized_pnl_usd;
        p.costs_usd            = fs.costs_usd;
        p.fills_cumulative     = result.fills_generated;
        p.bars_cumulative      = result.bars_processed;
    };
    if (use_vday) {
        // any boundary at/after the final event still needs its point emitted...
        while (vb_ptr < vday.size() && vday[vb_ptr] <= final_ts) {
            result.daily_equity.push_back(make_point(static_cast<std::int64_t>(vb_ptr), final_ts));
            ++vb_ptr;
        }
        // ...and the last emitted boundary point carries post-EOT equity.
        if (!result.daily_equity.empty() && vb_ptr == vday.size() &&
            !vday.empty() && vday.back() <= final_ts) {
            resample_last(result.daily_equity.back());
        }
    } else if (current_day.has_value()) {
        resample_last(*current_day);
        result.daily_equity.push_back(*current_day);
        current_day.reset();
    }
    return result;
}

}  // namespace quant
