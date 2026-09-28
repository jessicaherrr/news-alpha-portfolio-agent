#include "quant_core/portfolio.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace quant {

namespace {
constexpr std::int64_t kNsPerDay = 86'400'000'000'000LL;
}  // namespace

std::int64_t session_day_index(std::int64_t ts_ns, std::int64_t day_boundary_offset_ns) noexcept {
    const std::int64_t shifted = ts_ns - day_boundary_offset_ns;
    // Floor division toward negative infinity (shifted can be negative for a
    // pre-epoch offset; ts_ns itself is always > 0 in this core).
    std::int64_t q = shifted / kNsPerDay;
    if ((shifted % kNsPerDay != 0) && ((shifted < 0) != (kNsPerDay < 0))) --q;
    return q;
}

PortfolioAccountant::PortfolioAccountant(PortfolioConfig config, const MarginModel* margin)
    : config_(config),
      margin_(margin),
      peak_equity_usd_(config.starting_capital_usd),
      day_start_equity_usd_(config.starting_capital_usd),
      day_start_net_realized_usd_(0.0) {}

double PortfolioAccountant::equity_at(std::int64_t /*as_of_ts_ns*/) const {
    const double net_realized = ledger_.realized_pnl_usd() - ledger_.costs_usd();
    double unrealized = 0.0;
    for (const LedgerPosition& p : ledger_.open_positions()) {
        const auto it = marks_.find(p.instrument_id);
        if (it == marks_.end()) continue;  // no valuation -> contributes 0
        unrealized += (it->second.price - p.avg_entry_price) * p.multiplier * p.units;
    }
    return config_.starting_capital_usd + net_realized + unrealized;
}

void PortfolioAccountant::refresh_running_stats(std::int64_t ts_ns) {
    const double equity = equity_at(ts_ns);
    peak_equity_usd_ = std::max(peak_equity_usd_, equity);

    const std::int64_t d = session_day_index(ts_ns, config_.day_boundary_offset_ns);
    if (!day_initialised_ || d != day_index_) {
        day_index_                  = d;
        day_start_equity_usd_       = equity;
        day_start_net_realized_usd_ = ledger_.realized_pnl_usd() - ledger_.costs_usd();
        day_initialised_            = true;
    }
    last_stats_ts_ns_ = ts_ns;
}

void PortfolioAccountant::apply_fill(const Fill& fill, const std::string& root_symbol,
                                     const std::string& close_reason) {
    ledger_.apply(fill, root_symbol, close_reason);
    refresh_running_stats(fill.ts_fill_ns);
}

void PortfolioAccountant::observe_mark(std::uint32_t instrument_id, double price,
                                       std::int64_t ts_ns) {
    marks_[instrument_id] = MarkObs{price, ts_ns};
    refresh_running_stats(ts_ns);
}

void PortfolioAccountant::apply_split(const SplitAction& action) {
    if (!action.is_valid()) {
        throw std::invalid_argument("PortfolioAccountant::apply_split: invalid SplitAction");
    }
    // PositionLedger::apply_split owns the unit-count / cost-basis math and is
    // the sole authority on the FractionalSplitQuantityError representation
    // limit; propagate any exception unchanged.
    ledger_.apply_split(action.instrument_id, action.ratio);
}

void PortfolioAccountant::record_distribution_ex_date(const CashDistribution& action) {
    if (!action.is_valid()) {
        throw std::invalid_argument(
            "PortfolioAccountant::record_distribution_ex_date: invalid CashDistribution");
    }
    const int units = ledger_.units(action.instrument_id);
    if (units == 0) {
        return;  // flat position: no entitlement, no receivable.
    }

    PendingReceivable r;
    r.instrument_id       = action.instrument_id;
    r.ex_date_ts_ns        = action.ex_date_ts_ns;
    r.units_entitled       = units;
    r.amount_per_share_usd = action.amount_per_share_usd;
    // Sign-aware: a short position (units < 0) owes the distribution, i.e. this
    // is negative -- it is a liability against the account, not free income.
    r.total_receivable_usd = static_cast<double>(units) * action.amount_per_share_usd;

    pending_distributions_[{action.instrument_id, action.ex_date_ts_ns}] = r;
}

void PortfolioAccountant::record_distribution_payment(const CashDistribution& action) {
    if (!action.is_valid()) {
        throw std::invalid_argument(
            "PortfolioAccountant::record_distribution_payment: invalid CashDistribution");
    }
    // Lookup key is (instrument_id, ex_date_ts_ns) only -- the simpler, less
    // error-prone option than requiring the caller replay a byte-identical
    // CashDistribution struct (amount_per_share_usd / pay_date_ts_ns are not
    // part of the key).
    const auto key = std::make_pair(action.instrument_id, action.ex_date_ts_ns);
    const auto it  = pending_distributions_.find(key);
    if (it == pending_distributions_.end()) {
        throw NoPendingDistributionError(
            "PortfolioAccountant::record_distribution_payment: no pending receivable for "
            "instrument_id=" + std::to_string(action.instrument_id) +
            " ex_date_ts_ns=" + std::to_string(action.ex_date_ts_ns));
    }
    // Fill-PnL-independent: never touches ledger_ / realized_pnl_usd_ / costs_usd_.
    distribution_income_usd_ += it->second.total_receivable_usd;
    pending_distributions_.erase(it);
}

PortfolioState PortfolioAccountant::snapshot(std::int64_t as_of_ts_ns) const {
    PortfolioState s;
    s.as_of_ts_ns          = as_of_ts_ns;
    s.starting_capital_usd = config_.starting_capital_usd;
    s.gross_realized_pnl_usd = ledger_.realized_pnl_usd();
    s.costs_usd              = ledger_.costs_usd();
    s.net_realized_pnl_usd   = s.gross_realized_pnl_usd - s.costs_usd;

    double unrealized = 0.0;
    for (const LedgerPosition& p : ledger_.open_positions()) {
        PositionExposure e;
        e.instrument_id   = p.instrument_id;
        e.raw_symbol      = p.raw_symbol;
        e.root_symbol     = p.root_symbol;
        e.units           = p.units;
        e.avg_entry_price = p.avg_entry_price;
        e.multiplier      = p.multiplier;

        const auto it = marks_.find(p.instrument_id);
        double valuation_price = p.avg_entry_price;  // fallback: no unrealized if unmarked
        if (it != marks_.end()) {
            e.mark_present = true;
            e.mark_price   = it->second.price;
            e.mark_ts_ns   = it->second.ts_ns;
            e.mark_age_ns  = as_of_ts_ns - it->second.ts_ns;
            if (e.mark_age_ns < 0) e.mark_age_ns = 0;
            e.mark_is_stale = e.mark_age_ns > config_.mark_staleness_tolerance_ns;
            valuation_price = it->second.price;
        } else {
            e.mark_present  = false;
            e.mark_price    = 0.0;
            e.mark_ts_ns    = 0;
            e.mark_age_ns   = 0;
            e.mark_is_stale = true;  // an unmarked open position is never "fresh"
        }
        // No mark -> notional is an ESTIMATE from the average entry price, not a
        // current market value. Flagged so risk never treats it as a live number.
        e.valuation_is_estimated = !e.mark_present;

        const int pos_sign = (p.units > 0) - (p.units < 0);
        e.unrealized_pnl_usd = (valuation_price - p.avg_entry_price) * p.multiplier * p.units;
        // Gross notional is the non-negative magnitude of the position value --
        // valid even when `valuation_price` is negative (CL below zero).
        e.gross_notional_usd =
            std::fabs(static_cast<double>(p.units) * valuation_price * p.multiplier);
        // Directional notional follows the POSITION, never the sign of the price:
        // long 1 CL @ -20 is a LONG with positive gross exposure.
        e.signed_notional_usd = pos_sign * e.gross_notional_usd;

        const std::optional<double> init_pc =
            margin_ ? margin_->initial_per_contract(p.root_symbol) : std::optional<double>{};
        const std::optional<double> main_pc =
            margin_ ? margin_->maintenance_per_contract(p.root_symbol) : std::optional<double>{};
        e.margin_known = init_pc.has_value();
        const int abs_units = std::abs(p.units);
        e.initial_margin_usd     = init_pc.value_or(0.0) * abs_units;
        e.maintenance_margin_usd = main_pc.value_or(0.0) * abs_units;

        unrealized += e.unrealized_pnl_usd;
        s.gross_exposure_usd += e.gross_notional_usd;
        s.net_exposure_usd   += e.signed_notional_usd;
        if (pos_sign > 0) {
            s.long_exposure_usd += e.gross_notional_usd;
        } else {
            s.short_exposure_usd += e.gross_notional_usd;
        }
        s.gross_contracts += abs_units;
        s.net_contracts   += p.units;
        s.initial_margin_usd     += e.initial_margin_usd;
        s.maintenance_margin_usd += e.maintenance_margin_usd;
        s.margin_complete = s.margin_complete && e.margin_known;
        if (e.mark_is_stale) {
            s.has_stale_mark = true;
            s.worst_mark_age_ns = std::max(s.worst_mark_age_ns, e.mark_age_ns);
        }
        s.valuation_complete = s.valuation_complete && e.mark_present && !e.mark_is_stale;
        s.positions.push_back(std::move(e));
    }

    s.unrealized_pnl_usd = unrealized;
    s.cash_usd           = s.starting_capital_usd + s.net_realized_pnl_usd;
    s.equity_usd         = s.cash_usd + s.unrealized_pnl_usd;

    const bool has_equity = s.equity_usd > 0.0;
    s.gross_leverage = has_equity ? s.gross_exposure_usd / s.equity_usd
                                  : (s.gross_exposure_usd > 0.0 ? kUtilizationNoEquity : 0.0);
    s.net_leverage = has_equity ? s.net_exposure_usd / s.equity_usd
                                : (s.net_exposure_usd != 0.0
                                       ? std::copysign(kUtilizationNoEquity, s.net_exposure_usd)
                                       : 0.0);
    s.margin_utilization_pct = has_equity
                                   ? s.initial_margin_usd / s.equity_usd
                                   : (s.initial_margin_usd > 0.0 ? kUtilizationNoEquity : 0.0);

    s.peak_equity_usd = std::max(peak_equity_usd_, s.equity_usd);
    s.drawdown_usd    = std::max(0.0, s.peak_equity_usd - s.equity_usd);
    s.drawdown_pct    = s.peak_equity_usd > 0.0 ? s.drawdown_usd / s.peak_equity_usd : 0.0;

    s.session_day_index           = day_initialised_ ? day_index_ : 0;
    s.day_start_equity_usd        = day_start_equity_usd_;
    s.day_start_net_realized_usd  = day_start_net_realized_usd_;
    s.day_realized_pnl_usd        = s.net_realized_pnl_usd - day_start_net_realized_usd_;
    s.day_equity_change_usd       = s.equity_usd - day_start_equity_usd_;

    s.correlation_adjusted_gross_exposure_usd = s.gross_exposure_usd;  // MVP: no offsets

    // ETF corporate-action income (Phase 6 pilot) -- deliberately NOT folded
    // into cash_usd / equity_usd above; see PortfolioState's field comments.
    double receivable = 0.0;
    for (const auto& [key, r] : pending_distributions_) {
        (void)key;
        receivable += r.total_receivable_usd;
    }
    s.distribution_receivable_usd = receivable;
    s.distribution_income_usd     = distribution_income_usd_;

    return s;
}

}  // namespace quant
