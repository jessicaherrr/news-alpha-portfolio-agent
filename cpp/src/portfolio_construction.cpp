#include "quant_core/portfolio_construction.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace quant {

namespace {

using Vec = std::vector<double>;
using Mat = std::vector<Vec>;

constexpr double kRelTol       = 1e-9;   // limit checks: value <= limit * (1 + kRelTol) + kAbsTol
constexpr double kAbsTol       = 1e-12;
constexpr double kNetsToZero   = 1e-9;   // unit-composite vol below which a cluster carries no risk
constexpr double kPsdTol       = 1e-10;  // most negative LDL pivot tolerated as rounding noise
constexpr double kCorrTol      = 1e-12;  // |rho| may exceed 1 by this much (then clamped)
constexpr double kUnitNudge    = 1e-9;   // units: truncate x + sign(x)*nudge (3 - 4e-16 is 3)
constexpr double kMaxUnits     = 2.0e9;
constexpr int    kMaxErcSweeps = 100000;
constexpr double kErcTarget    = 1e-12;  // max |risk share - 1/m| to stop
constexpr double kErcAccept    = 1e-9;   // max |risk share - 1/m| to accept

void require(bool ok, const std::string& what) {
    if (!ok) throw std::invalid_argument("portfolio construction: " + what);
}

bool exceeds(double value, double limit) {
    return value > limit * (1.0 + kRelTol) + kAbsTol;
}

double quad(const Mat& s, const Vec& w) {
    double v = 0.0;
    for (std::size_t i = 0; i < w.size(); ++i) {
        if (w[i] == 0.0) continue;
        for (std::size_t j = 0; j < w.size(); ++j) v += w[i] * s[i][j] * w[j];
    }
    return v;
}

Vec mul(const Mat& s, const Vec& w) {
    Vec out(w.size(), 0.0);
    for (std::size_t i = 0; i < w.size(); ++i) {
        for (std::size_t j = 0; j < w.size(); ++j) out[i] += s[i][j] * w[j];
    }
    return out;
}

double vol_of(const Mat& cov, const Vec& w) {
    return std::sqrt(std::max(0.0, quad(cov, w)));
}

// LDL^T pivots of a symmetric matrix; a pivot below -kPsdTol => not PSD.
bool positive_semidefinite(const Mat& a) {
    const std::size_t n = a.size();
    Mat l(n, Vec(n, 0.0));
    Vec d(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        double dj = a[j][j];
        for (std::size_t k = 0; k < j; ++k) dj -= l[j][k] * l[j][k] * d[k];
        if (dj < -kPsdTol) return false;
        d[j] = dj;
        l[j][j] = 1.0;
        for (std::size_t i = j + 1; i < n; ++i) {
            double v = a[i][j];
            for (std::size_t k = 0; k < j; ++k) v -= l[i][k] * l[j][k] * d[k];
            l[i][j] = dj > kPsdTol ? v / dj : 0.0;
        }
    }
    return true;
}

// Equal risk contribution: minimise 1/2 y'Sy - b * sum(ln y_j), b = 1/m, by
// cyclical coordinate descent. At the optimum y_j (Sy)_j = b for every j, so
// every cluster contributes the same share of variance. Fills y (any scale);
// false when no ERC solution exists (a degenerate risk model -- e.g. two
// clusters that hedge each other exactly, where the objective is unbounded).
bool solve_erc(const Mat& s, Vec& y, int& sweeps, double& max_dev) {
    const std::size_t m = s.size();
    const double b = 1.0 / static_cast<double>(m);
    y.assign(m, 0.0);
    for (std::size_t j = 0; j < m; ++j) y[j] = 1.0 / std::sqrt(s[j][j]);
    max_dev = std::numeric_limits<double>::infinity();
    for (sweeps = 1; sweeps <= kMaxErcSweeps; ++sweeps) {
        for (std::size_t j = 0; j < m; ++j) {
            double c = 0.0;
            for (std::size_t l = 0; l < m; ++l) {
                if (l != j) c += s[j][l] * y[l];
            }
            const double root = std::sqrt(c * c + 4.0 * s[j][j] * b);
            // Stable roots of s_jj y^2 + c y - b = 0 (no cancellation for c >= 0).
            y[j] = c >= 0.0 ? 2.0 * b / (c + root) : (root - c) / (2.0 * s[j][j]);
        }
        const Vec sy = mul(s, y);
        double total = 0.0;
        for (std::size_t j = 0; j < m; ++j) total += y[j] * sy[j];
        if (!std::isfinite(total) || total <= 0.0) break;
        max_dev = 0.0;
        for (std::size_t j = 0; j < m; ++j) max_dev = std::max(max_dev, std::abs(y[j] * sy[j] / total - b));
        if (max_dev <= kErcTarget) return true;
    }
    sweeps = std::min(sweeps, kMaxErcSweeps);
    return max_dev <= kErcAccept;
}

struct Leg {
    std::size_t signal{0};      // index into result.signals
    std::size_t instrument{0};  // index into result.instruments
    std::size_t cluster{0};     // index into result.clusters
    double      target{0.0};    // capital fraction after ERC + target vol
};

class Book {
public:
    Book(const Mat& cov, const Vec& base, const std::vector<Leg>& legs, std::size_t n_clusters)
        : cov_(cov), base_(base), legs_(legs), n_clusters_(n_clusters) {
        // Legs on one instrument may cancel; a base weight that is only rounding
        // noise relative to its legs is treated as zero (no attribution).
        Vec gross(base.size(), 0.0);
        for (const auto& leg : legs) gross[leg.instrument] += std::abs(leg.target);
        attributable_.assign(base.size(), false);
        for (std::size_t k = 0; k < base.size(); ++k) {
            attributable_[k] = std::abs(base[k]) > 1e-12 * gross[k] && base[k] != 0.0;
        }
    }

    // A leg's share of `w` on its instrument, pro rata to its target leg.
    double leg_weight(const Leg& leg, const Vec& w) const {
        return attributable_[leg.instrument] ? w[leg.instrument] * leg.target / base_[leg.instrument] : 0.0;
    }

    Vec instrument_rc(const Vec& w) const {
        Vec rc(w.size(), 0.0);
        const double sigma = vol_of(cov_, w);
        if (sigma <= 0.0) return rc;
        const Vec m = mul(cov_, w);
        for (std::size_t k = 0; k < w.size(); ++k) rc[k] = w[k] * m[k] / sigma;
        return rc;
    }

    Vec leg_rc(const Vec& w) const {
        Vec rc(legs_.size(), 0.0);
        const double sigma = vol_of(cov_, w);
        if (sigma <= 0.0) return rc;
        const Vec m = mul(cov_, w);
        for (std::size_t i = 0; i < legs_.size(); ++i) rc[i] = leg_weight(legs_[i], w) * m[legs_[i].instrument] / sigma;
        return rc;
    }

    Vec cluster_rc(const Vec& w) const {
        Vec rc(n_clusters_, 0.0);
        const Vec legs = leg_rc(w);
        for (std::size_t i = 0; i < legs_.size(); ++i) rc[legs_[i].cluster] += legs[i];
        return rc;
    }

    PortfolioStats stats(const Vec& w) const {
        PortfolioStats s;
        for (const double x : w) {
            s.gross_exposure += std::abs(x);
            s.net_exposure += x;
            if (x > 0.0) s.long_exposure += x;
            if (x < 0.0) s.short_exposure -= x;
            if (x != 0.0) ++s.positions;
        }
        s.annual_vol = vol_of(cov_, w);
        if (s.annual_vol > 0.0) {
            const Vec rc = cluster_rc(w);
            double sum_sq = 0.0;
            for (const double c : rc) {
                const double share = c / s.annual_vol;
                sum_sq += share * share;
                s.max_cluster_risk_share = std::max(s.max_cluster_risk_share, share);
            }
            s.effective_exposures = sum_sq > 0.0 ? 1.0 / sum_sq : 0.0;
        }
        return s;
    }

    const Mat& cov() const { return cov_; }
    const std::vector<Leg>& legs() const { return legs_; }

private:
    const Mat&              cov_;
    const Vec&              base_;
    const std::vector<Leg>& legs_;
    std::size_t             n_clusters_;
    std::vector<bool>       attributable_;
};

// Collects constraint outcomes; a second application of the same constraint
// (hard limits re-applied after the turnover limit) merges into the first.
class Outcomes {
public:
    ConstraintOutcome& touch(const std::string& name, const std::string& scope, const std::string& stage,
                             double limit, double before) {
        for (auto& o : items_) {
            if (o.name == name && o.scope == scope && o.stage == stage) return o;
        }
        ConstraintOutcome o;
        o.name = name;
        o.scope = scope;
        o.stage = stage;
        o.status = ConstraintStatus::Satisfied;
        o.limit = limit;
        o.before = before;
        o.after = before;
        items_.push_back(o);
        return items_.back();
    }

    void not_applicable(const std::string& name, const std::string& scope, const std::string& stage) {
        for (const auto& o : items_) {
            if (o.name == name && o.scope == scope && o.stage == stage) return;
        }
        ConstraintOutcome o;
        o.name = name;
        o.scope = scope;
        o.stage = stage;
        o.status = ConstraintStatus::NotApplicable;
        items_.push_back(o);
    }

    static void bind(ConstraintOutcome& o, const std::string& key) {
        if (o.status != ConstraintStatus::Overridden) o.status = ConstraintStatus::Binding;
        if (!key.empty() && std::find(o.affected.begin(), o.affected.end(), key) == o.affected.end()) {
            o.affected.push_back(key);
        }
    }

    std::vector<ConstraintOutcome> take() { return {items_.begin(), items_.end()}; }

private:
    // deque: references returned by touch() stay valid across later touches.
    std::deque<ConstraintOutcome> items_;
};

struct Context {
    const ConstructionLimits&                limits;
    const std::vector<InstrumentAllocation>& instruments;
    const Book&                              book;
    double                                   gross_cap;
};

void scale_all(Vec& w, double factor) {
    for (auto& x : w) x *= factor;
}

std::vector<std::string> keys_of(const std::vector<InstrumentAllocation>& inst, const Vec& before, const Vec& after) {
    std::vector<std::string> out;
    for (std::size_t k = 0; k < before.size(); ++k) {
        if (before[k] != after[k]) out.push_back(inst[k].input.key);
    }
    return out;
}

// Apply one uniform portfolio-level scale-down when `value(w) > limit`.
template <class ValueFn>
void uniform_cap(Vec& w, Outcomes& out, const std::string& name, double limit, ValueFn value,
                 const std::vector<InstrumentAllocation>& inst) {
    const double v = value(w);
    auto& o = out.touch(name, "portfolio", "allocation", limit, v);
    if (exceeds(v, limit) && v > 0.0) {
        const Vec before = w;
        scale_all(w, limit / v);
        for (const auto& key : keys_of(inst, before, w)) Outcomes::bind(o, key);
    }
    o.after = value(w);
}

// Group gross caps (sector / asset class): scale only the group's positions.
void group_caps(Vec& w, Outcomes& out, const Context& ctx, const std::string& name, double share,
                bool by_sector) {
    if (share <= 0.0) {
        out.not_applicable(name, "each group", "allocation");
        return;
    }
    std::map<std::string, std::vector<std::size_t>> groups;
    for (std::size_t k = 0; k < w.size(); ++k) {
        if (!ctx.instruments[k].admitted) continue;
        const auto& in = ctx.instruments[k].input;
        groups[by_sector ? in.sector : in.asset_class].push_back(k);
    }
    const double limit = share * ctx.gross_cap;
    for (const auto& [group, members] : groups) {
        double gross = 0.0;
        for (const auto k : members) gross += std::abs(w[k]);
        auto& o = out.touch(name, (by_sector ? "sector:" : "asset_class:") + group, "allocation", limit, gross);
        if (exceeds(gross, limit) && gross > 0.0) {
            const double f = limit / gross;
            for (const auto k : members) {
                if (w[k] != 0.0) {
                    w[k] *= f;
                    Outcomes::bind(o, ctx.instruments[k].input.key);
                }
            }
        }
        double after = 0.0;
        for (const auto k : members) after += std::abs(w[k]);
        o.after = after;
    }
}

// Every hard limit, in documented order. Each is a scale-down, so a later
// constraint never breaks an earlier one (uniform scalings shrink every
// position; group scalings touch only their group). Idempotent.
void apply_hard_limits(Vec& w, Outcomes& out, const Context& ctx) {
    const auto& lim = ctx.limits;
    const auto& inst = ctx.instruments;
    const std::size_t n = w.size();

    // 1. short restriction
    if (lim.shorting_allowed) {
        out.not_applicable("short_restriction", "each instrument", "allocation");
    } else {
        double shorts = 0.0;
        for (const double x : w) shorts += x < 0.0 ? -x : 0.0;
        auto& o = out.touch("short_restriction", "each instrument", "allocation", 0.0, shorts);
        for (std::size_t k = 0; k < n; ++k) {
            if (w[k] < 0.0) {
                w[k] = 0.0;
                Outcomes::bind(o, inst[k].input.key);
            }
        }
        o.after = 0.0;
    }

    // 2. per-instrument unit cap (utilisation of max_units)
    {
        bool any = false;
        double before = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            const auto& in = inst[k].input;
            if (!inst[k].admitted || in.max_units <= 0) continue;
            any = true;
            const double cap = in.max_units * in.unit_notional_usd() / lim.capital_usd;
            before = std::max(before, std::abs(w[k]) / cap);
        }
        if (!any) {
            out.not_applicable("unit_cap", "each instrument", "allocation");
        } else {
            auto& o = out.touch("unit_cap", "each instrument", "allocation", 1.0, before);
            double after = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                const auto& in = inst[k].input;
                if (!inst[k].admitted || in.max_units <= 0) continue;
                const double cap = in.max_units * in.unit_notional_usd() / lim.capital_usd;
                if (exceeds(std::abs(w[k]), cap)) {
                    w[k] = std::copysign(cap, w[k]);
                    Outcomes::bind(o, in.key);
                }
                after = std::max(after, std::abs(w[k]) / cap);
            }
            o.after = after;
        }
    }

    // 3. instrument concentration (share of the gross exposure the mandate allows)
    if (lim.max_instrument_gross_share <= 0.0) {
        out.not_applicable("instrument_concentration", "each instrument", "allocation");
    } else {
        const double cap = lim.max_instrument_gross_share * ctx.gross_cap;
        double before = 0.0;
        for (const double x : w) before = std::max(before, std::abs(x));
        auto& o = out.touch("instrument_concentration", "each instrument", "allocation", cap, before);
        for (std::size_t k = 0; k < n; ++k) {
            if (exceeds(std::abs(w[k]), cap)) {
                w[k] = std::copysign(cap, w[k]);
                Outcomes::bind(o, inst[k].input.key);
            }
        }
        double after = 0.0;
        for (const double x : w) after = std::max(after, std::abs(x));
        o.after = after;
    }

    // 4. liquidity participation (fraction of median daily traded notional)
    if (lim.max_adv_participation <= 0.0) {
        out.not_applicable("liquidity_participation", "each instrument", "allocation");
    } else {
        auto participation = [&](std::size_t k) {
            return inst[k].input.adv_usd > 0.0 ? std::abs(w[k]) * lim.capital_usd / inst[k].input.adv_usd : 0.0;
        };
        double before = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            if (inst[k].admitted) before = std::max(before, participation(k));
        }
        auto& o = out.touch("liquidity_participation", "each instrument", "allocation",
                            lim.max_adv_participation, before);
        double after = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            if (!inst[k].admitted) continue;
            const double adv = inst[k].input.adv_usd;
            const double cap = adv > 0.0 ? lim.max_adv_participation * adv / lim.capital_usd : 0.0;
            if (exceeds(std::abs(w[k]), cap)) {
                w[k] = std::copysign(cap, w[k]);
                Outcomes::bind(o, inst[k].input.key);
            }
            after = std::max(after, participation(k));
        }
        o.after = after;
    }

    // 5-6. sector and asset-class gross caps
    group_caps(w, out, ctx, "sector_gross", lim.max_sector_gross_share, true);
    group_caps(w, out, ctx, "asset_class_gross", lim.max_asset_class_gross_share, false);

    // 7. gross leverage (always on)
    uniform_cap(w, out, "gross_leverage", ctx.gross_cap, [](const Vec& v) {
        double g = 0.0;
        for (const double x : v) g += std::abs(x);
        return g;
    }, inst);

    // 8. net exposure
    if (lim.max_net_exposure <= 0.0) {
        out.not_applicable("net_exposure", "portfolio", "allocation");
    } else {
        uniform_cap(w, out, "net_exposure", lim.max_net_exposure, [](const Vec& v) {
            double s = 0.0;
            for (const double x : v) s += x;
            return std::abs(s);
        }, inst);
    }

    // 9. per-cluster risk share (absolute: share x target volatility)
    if (lim.max_cluster_risk_share <= 0.0) {
        out.not_applicable("cluster_risk_share", "portfolio", "allocation");
    } else {
        uniform_cap(w, out, "cluster_risk_share", lim.max_cluster_risk_share * lim.target_annual_vol,
                    [&](const Vec& v) {
                        double worst = 0.0;
                        for (const double c : ctx.book.cluster_rc(v)) worst = std::max(worst, c);
                        return worst;
                    }, inst);
    }

    // 10. volatility ceiling (a clipped hedge can raise volatility)
    uniform_cap(w, out, "volatility_ceiling", lim.target_annual_vol,
                [&](const Vec& v) { return vol_of(ctx.book.cov(), v); }, inst);
}

std::string instrument_reason(const ConstructionInstrument& in, const ConstructionLimits& lim,
                              const std::set<std::string>& domains) {
    if (domains.count(in.domain) == 0) return "DOMAIN_NOT_ALLOWED";
    if (!(in.price > 0.0)) return "NON_POSITIVE_PRICE";
    if (!(in.multiplier > 0.0)) return "INVALID_ECONOMICS";
    if (!(in.annual_vol > 0.0)) return "NO_VOLATILITY";
    const bool liquidity_used = lim.min_adv_usd > 0.0 || lim.max_adv_participation > 0.0;
    if (liquidity_used && in.adv_usd < 0.0) return "LIQUIDITY_NOT_MEASURED";
    if (lim.min_adv_usd > 0.0 && in.adv_usd < lim.min_adv_usd) return "BELOW_MIN_LIQUIDITY";
    return "";
}

void validate_limits(const ConstructionLimits& lim) {
    const double values[] = {lim.capital_usd, lim.target_annual_vol, lim.max_gross_leverage, lim.max_net_exposure,
                             lim.max_instrument_gross_share, lim.max_sector_gross_share,
                             lim.max_asset_class_gross_share, lim.max_cluster_risk_share,
                             lim.max_adv_participation, lim.min_adv_usd, lim.max_turnover};
    for (const double v : values) require(std::isfinite(v) && v >= 0.0, "limits must be finite and non-negative");
    require(lim.capital_usd > 0.0, "capital_usd must be > 0");
    require(lim.target_annual_vol > 0.0, "target_annual_vol must be > 0");
    require(lim.max_gross_leverage > 0.0, "max_gross_leverage must be > 0 (an unstated cap is resolved upstream)");
    require(lim.max_instrument_gross_share <= 1.0 && lim.max_sector_gross_share <= 1.0 &&
                lim.max_asset_class_gross_share <= 1.0 && lim.max_cluster_risk_share <= 1.0,
            "share limits must be <= 1");
    require(!lim.allowed_domains.empty(), "allowed_domains must not be empty");
}

}  // namespace

const char* to_string(ConstraintStatus status) noexcept {
    switch (status) {
        case ConstraintStatus::NotApplicable: return "NOT_APPLICABLE";
        case ConstraintStatus::Satisfied:     return "SATISFIED";
        case ConstraintStatus::Binding:       return "BINDING";
        case ConstraintStatus::Overridden:    return "OVERRIDDEN";
    }
    return "UNKNOWN";
}

ConstructionResult construct_portfolio(const ConstructionInput& input) {
    const ConstructionLimits& lim = input.limits;
    validate_limits(lim);
    const std::set<std::string> domains(lim.allowed_domains.begin(), lim.allowed_domains.end());

    // ---- canonical inputs -------------------------------------------------
    std::vector<ConstructionInstrument> instruments = input.instruments;
    std::sort(instruments.begin(), instruments.end(),
              [](const auto& a, const auto& b) { return a.key < b.key; });
    std::map<std::string, std::size_t> index;
    for (std::size_t k = 0; k < instruments.size(); ++k) {
        const auto& in = instruments[k];
        require(!in.key.empty(), "an instrument has an empty key");
        require(index.emplace(in.key, k).second, "duplicate instrument key '" + in.key + "'");
        require(!in.root_symbol.empty(), "instrument '" + in.key + "' has no execution root");
        require(std::isfinite(in.price) && std::isfinite(in.multiplier) && std::isfinite(in.annual_vol) &&
                    std::isfinite(in.adv_usd),
                "instrument '" + in.key + "' has a non-finite number");
        require(in.max_units >= 0, "instrument '" + in.key + "' has a negative max_units");
    }

    std::vector<ConstructionSignal> signals = input.signals;
    std::sort(signals.begin(), signals.end(), [](const auto& a, const auto& b) {
        return a.priority != b.priority ? a.priority < b.priority : a.signal_id < b.signal_id;
    });
    std::set<std::string> signal_ids;
    std::set<std::string> referenced;
    for (const auto& s : signals) {
        require(!s.signal_id.empty() && signal_ids.insert(s.signal_id).second,
                "duplicate or empty signal id '" + s.signal_id + "'");
        require(index.count(s.instrument_key) == 1,
                "signal '" + s.signal_id + "' names unknown instrument '" + s.instrument_key + "'");
        require(s.direction == 1 || s.direction == -1, "signal '" + s.signal_id + "' direction must be +1 or -1");
        require(!s.cluster_id.empty(), "signal '" + s.signal_id + "' has no cluster");
        referenced.insert(s.instrument_key);
    }
    for (const auto& [key, units] : input.previous_units) {
        require(index.count(key) == 1, "previous_units names unknown instrument '" + key + "'");
        (void)units;
    }

    ConstructionResult r;
    r.capital_usd = lim.capital_usd;
    r.target_annual_vol = lim.target_annual_vol;
    const std::size_t n = instruments.size();

    // ---- 1. instrument admission ------------------------------------------
    r.instruments.resize(n);
    for (std::size_t k = 0; k < n; ++k) {
        auto& a = r.instruments[k];
        a.input = instruments[k];
        a.reason = instrument_reason(a.input, lim, domains);
        a.admitted = a.reason.empty();
        const auto prev = input.previous_units.find(a.input.key);
        a.previous_units = prev == input.previous_units.end() ? 0 : prev->second;
        if (a.admitted) a.unit_risk_fraction = a.input.unit_notional_usd() * a.input.annual_vol / lim.capital_usd;
        if (a.admitted && a.input.adv_usd > 0.0) a.adv_participation = 0.0;
    }

    // ---- correlation / covariance over admitted, referenced instruments -----
    auto in_model = [&](std::size_t k) { return r.instruments[k].admitted && referenced.count(instruments[k].key); };
    Mat corr(n, Vec(n, 0.0));
    for (std::size_t k = 0; k < n; ++k) corr[k][k] = 1.0;
    Mat have(n, Vec(n, 0.0));
    for (const auto& [pair, rho] : input.correlations) {
        const auto& [a, b] = pair;
        require(a < b, "correlation key (" + a + ", " + b + ") must be ordered a < b");
        require(index.count(a) == 1 && index.count(b) == 1, "correlation names an unknown instrument");
        require(std::isfinite(rho) && std::abs(rho) <= 1.0 + kCorrTol,
                "correlation (" + a + ", " + b + ") must be finite and in [-1, 1]");
        const double c = std::clamp(rho, -1.0, 1.0);
        const std::size_t i = index.at(a);
        const std::size_t j = index.at(b);
        corr[i][j] = corr[j][i] = c;
        have[i][j] = have[j][i] = 1.0;
    }
    std::vector<std::size_t> model;
    for (std::size_t k = 0; k < n; ++k) {
        if (in_model(k)) model.push_back(k);
    }
    for (std::size_t x = 0; x < model.size(); ++x) {
        for (std::size_t y = x + 1; y < model.size(); ++y) {
            require(have[model[x]][model[y]] == 1.0, "missing correlation for (" + instruments[model[x]].key +
                                                         ", " + instruments[model[y]].key + ")");
        }
    }
    {
        Mat sub(model.size(), Vec(model.size(), 0.0));
        for (std::size_t x = 0; x < model.size(); ++x) {
            for (std::size_t y = 0; y < model.size(); ++y) sub[x][y] = corr[model[x]][model[y]];
        }
        require(positive_semidefinite(sub), "the correlation matrix is not positive semi-definite");
    }
    Mat cov(n, Vec(n, 0.0));
    for (const auto i : model) {
        for (const auto j : model) {
            cov[i][j] = instruments[i].annual_vol * instruments[j].annual_vol * corr[i][j];
        }
    }

    // ---- signal admission --------------------------------------------------
    r.signals.resize(signals.size());
    for (std::size_t i = 0; i < signals.size(); ++i) {
        auto& sa = r.signals[i];
        sa.input = signals[i];
        const auto& inst = r.instruments[index.at(sa.input.instrument_key)];
        if (!inst.admitted) {
            sa.reason = "INSTRUMENT_NOT_ADMITTED";
        } else if (sa.input.direction < 0 && !lim.shorting_allowed) {
            sa.reason = "SHORT_NOT_PERMITTED";
        }
    }

    // ---- 2. clusters and unit legs ------------------------------------------
    // Cluster order: best (lowest) priority of ANY member, then id -- stable
    // whatever is admitted.
    std::map<std::string, std::vector<std::size_t>> members;
    std::map<std::string, int> best;
    for (std::size_t i = 0; i < r.signals.size(); ++i) {
        const auto& s = r.signals[i].input;
        members[s.cluster_id].push_back(i);
        const auto it = best.find(s.cluster_id);
        if (it == best.end() || s.priority < it->second) best[s.cluster_id] = s.priority;
    }
    std::vector<std::string> cluster_order;
    for (const auto& [id, unused] : members) cluster_order.push_back(id);
    std::sort(cluster_order.begin(), cluster_order.end(), [&](const auto& a, const auto& b) {
        return best.at(a) != best.at(b) ? best.at(a) < best.at(b) : a < b;
    });

    std::vector<Vec> composites;       // unit composite per ALLOCATED cluster
    std::vector<std::size_t> allocated;  // result.clusters index of each composite
    std::vector<Leg> legs;               // unit legs (target filled after ERC)
    std::vector<std::size_t> leg_composite;
    for (const auto& cid : cluster_order) {
        ClusterAllocation ca;
        ca.cluster_id = cid;
        std::vector<std::size_t> live;
        for (const auto i : members.at(cid)) {
            if (r.signals[i].reason.empty()) live.push_back(i);
        }
        const std::size_t c = r.clusters.size();
        if (live.empty()) {
            ca.reason = "NO_ADMITTED_SIGNAL";
            r.clusters.push_back(ca);
            continue;
        }
        Vec composite(n, 0.0);
        std::vector<Leg> cluster_legs;
        for (const auto i : live) {
            const std::size_t k = index.at(r.signals[i].input.instrument_key);
            const double unit = r.signals[i].input.direction /
                                (static_cast<double>(live.size()) * instruments[k].annual_vol);
            composite[k] += unit;
            cluster_legs.push_back(Leg{i, k, c, unit});
        }
        ca.composite_vol = vol_of(cov, composite);
        if (ca.composite_vol < kNetsToZero) {
            ca.reason = "CLUSTER_NETS_TO_ZERO";
            for (const auto i : live) r.signals[i].reason = "CLUSTER_NETS_TO_ZERO";
            r.clusters.push_back(ca);
            continue;
        }
        ca.allocated = true;
        for (const auto i : live) ca.signal_ids.push_back(r.signals[i].input.signal_id);
        for (auto& leg : cluster_legs) {
            legs.push_back(leg);
            leg_composite.push_back(composites.size());
        }
        composites.push_back(composite);
        allocated.push_back(c);
        r.clusters.push_back(ca);
    }

    Vec base(n, 0.0);
    if (composites.empty()) {
        r.status = "NO_ALLOCATABLE_SIGNAL";
        return r;
    }

    // ---- 3-4. ERC across clusters, scaled to the target volatility ----------
    const std::size_t m = composites.size();
    Mat s(m, Vec(m, 0.0));
    for (std::size_t i = 0; i < m; ++i) {
        const Vec ci = mul(cov, composites[i]);
        for (std::size_t j = 0; j < m; ++j) {
            double v = 0.0;
            for (std::size_t k = 0; k < n; ++k) v += composites[j][k] * ci[k];
            s[i][j] = v;
        }
    }
    Vec y;
    if (!solve_erc(s, y, r.erc_sweeps, r.erc_max_deviation)) {
        // A property of the portfolio, not a malformed input: typed, never thrown.
        r.status = "DEGENERATE_RISK_MODEL";
        for (const auto c : allocated) {
            r.clusters[c].allocated = false;
            r.clusters[c].reason = "DEGENERATE_RISK_MODEL";
            r.clusters[c].signal_ids.clear();
        }
        for (const auto& leg : legs) r.signals[leg.signal].reason = "DEGENERATE_RISK_MODEL";
        return r;
    }
    const double sigma_y = std::sqrt(std::max(0.0, quad(s, y)));
    for (auto& v : y) v *= lim.target_annual_vol / sigma_y;
    for (std::size_t j = 0; j < m; ++j) r.clusters[allocated[j]].erc_weight = y[j];
    for (std::size_t i = 0; i < legs.size(); ++i) {
        legs[i].target *= y[leg_composite[i]];
        base[legs[i].instrument] += legs[i].target;
    }
    const Book target_book(cov, base, legs, r.clusters.size());
    for (std::size_t k = 0; k < n; ++k) r.instruments[k].target_weight = base[k];
    r.target = target_book.stats(base);
    {
        const Vec rc = target_book.cluster_rc(base);
        for (std::size_t c = 0; c < r.clusters.size(); ++c) r.clusters[c].target_risk_contribution = rc[c];
    }
    for (std::size_t i = 0; i < legs.size(); ++i) r.signals[legs[i].signal].target_weight = legs[i].target;

    // ---- 5. constraints (hard limits, then turnover, then hard limits again) --
    Outcomes out;
    const Context ctx{lim, r.instruments, target_book, lim.max_gross_leverage};
    Vec w = base;
    apply_hard_limits(w, out, ctx);

    Vec prev(n, 0.0);
    for (std::size_t k = 0; k < n; ++k) {
        const auto& a = r.instruments[k];
        prev[k] = a.previous_units * a.input.unit_notional_usd() / lim.capital_usd;
    }
    auto turnover = [&](const Vec& v) {
        double t = 0.0;
        for (std::size_t k = 0; k < n; ++k) t += std::abs(v[k] - prev[k]);
        return t;
    };
    r.turnover_target = turnover(w);
    if (lim.max_turnover <= 0.0) {
        out.not_applicable("turnover", "portfolio", "allocation");
    } else {
        auto& o = out.touch("turnover", "portfolio", "allocation", lim.max_turnover, r.turnover_target);
        if (exceeds(r.turnover_target, lim.max_turnover)) {
            const double lambda = lim.max_turnover / r.turnover_target;
            const Vec before = w;
            for (std::size_t k = 0; k < n; ++k) w[k] = prev[k] + lambda * (w[k] - prev[k]);
            for (const auto& key : keys_of(r.instruments, before, w)) Outcomes::bind(o, key);
            // Hard risk limits take precedence over the turnover limit.
            apply_hard_limits(w, out, ctx);
        }
        o.after = turnover(w);
        if (exceeds(o.after, lim.max_turnover)) o.status = ConstraintStatus::Overridden;
    }
    for (std::size_t k = 0; k < n; ++k) r.instruments[k].constrained_weight = w[k];
    r.constrained = target_book.stats(w);

    // ---- 6. integer units (truncate toward zero), then repair --------------
    std::vector<long long> units(n, 0);
    for (std::size_t k = 0; k < n; ++k) {
        const auto& a = r.instruments[k];
        if (w[k] == 0.0) continue;
        const double x = w[k] * lim.capital_usd / a.input.unit_notional_usd();
        if (!(std::abs(x) < kMaxUnits)) throw std::runtime_error("portfolio construction: unit count overflow for '" +
                                                                 a.input.key + "'");
        units[k] = static_cast<long long>(std::trunc(x + std::copysign(kUnitNudge, x)));
    }
    auto exec_weights = [&]() {
        Vec e(n, 0.0);
        for (std::size_t k = 0; k < n; ++k) {
            e[k] = static_cast<double>(units[k]) * r.instruments[k].input.unit_notional_usd() / lim.capital_usd;
        }
        return e;
    };
    auto remove_one = [&](std::size_t k, ConstraintOutcome& o) {
        units[k] -= units[k] > 0 ? 1 : -1;
        ++r.repair_units_removed;
        Outcomes::bind(o, r.instruments[k].input.key);
    };
    // Largest |executable weight| among `candidates` (ties: lowest key).
    auto pick = [&](const Vec& score, const std::vector<bool>& eligible) {
        std::size_t best_k = n;
        for (std::size_t k = 0; k < n; ++k) {
            if (!eligible[k] || units[k] == 0) continue;
            if (best_k == n || score[k] > score[best_k]) best_k = k;
        }
        return best_k;
    };

    Vec e = exec_weights();
    ConstraintOutcome* net_o = nullptr;
    ConstraintOutcome* cluster_o = nullptr;
    if (lim.max_net_exposure > 0.0) {
        double net = 0.0;
        for (const double x : e) net += x;
        net_o = &out.touch("net_exposure", "portfolio", "rounding", lim.max_net_exposure, std::abs(net));
    } else {
        out.not_applicable("net_exposure", "portfolio", "rounding");
    }
    if (lim.max_cluster_risk_share > 0.0) {
        double worst = 0.0;
        for (const double c : target_book.cluster_rc(e)) worst = std::max(worst, c);
        cluster_o = &out.touch("cluster_risk_share", "portfolio", "rounding",
                               lim.max_cluster_risk_share * lim.target_annual_vol, worst);
    } else {
        out.not_applicable("cluster_risk_share", "portfolio", "rounding");
    }
    ConstraintOutcome* vol_o =
        &out.touch("volatility_ceiling", "portfolio", "rounding", lim.target_annual_vol, vol_of(cov, e));

    for (;;) {
        e = exec_weights();
        // net exposure: shrink the dominant side's largest position
        if (net_o != nullptr) {
            double net = 0.0;
            for (const double x : e) net += x;
            if (exceeds(std::abs(net), lim.max_net_exposure)) {
                Vec score(n, 0.0);
                std::vector<bool> ok(n, false);
                for (std::size_t k = 0; k < n; ++k) {
                    ok[k] = (net > 0.0) == (e[k] > 0.0) && e[k] != 0.0;
                    score[k] = std::abs(e[k]);
                }
                const std::size_t k = pick(score, ok);
                if (k < n) { remove_one(k, *net_o); continue; }
            }
        }
        // cluster risk share: shrink the violating cluster's largest contributor
        if (cluster_o != nullptr) {
            const Vec rc = target_book.cluster_rc(e);
            const Vec lrc = target_book.leg_rc(e);
            std::size_t worst = rc.size();
            for (std::size_t c = 0; c < rc.size(); ++c) {
                if (exceeds(rc[c], lim.max_cluster_risk_share * lim.target_annual_vol) &&
                    (worst == rc.size() || rc[c] > rc[worst])) worst = c;
            }
            if (worst < rc.size()) {
                Vec score(n, -std::numeric_limits<double>::infinity());
                std::vector<bool> ok(n, false);
                for (std::size_t i = 0; i < legs.size(); ++i) {
                    if (legs[i].cluster != worst) continue;
                    ok[legs[i].instrument] = true;
                    score[legs[i].instrument] = std::max(score[legs[i].instrument], lrc[i]);
                }
                const std::size_t k = pick(score, ok);
                if (k < n) { remove_one(k, *cluster_o); continue; }
            }
        }
        // volatility ceiling: shrink the largest risk contributor
        if (exceeds(vol_of(cov, e), lim.target_annual_vol)) {
            const Vec rc = target_book.instrument_rc(e);
            std::vector<bool> ok(n, true);
            const std::size_t k = pick(rc, ok);
            if (k < n) { remove_one(k, *vol_o); continue; }
        }
        break;
    }
    e = exec_weights();
    if (net_o != nullptr) {
        double net = 0.0;
        for (const double x : e) net += x;
        net_o->after = std::abs(net);
    }
    if (cluster_o != nullptr) {
        double worst = 0.0;
        for (const double c : target_book.cluster_rc(e)) worst = std::max(worst, c);
        cluster_o->after = worst;
    }
    vol_o->after = vol_of(cov, e);

    // ---- invariants that truncation cannot break (a violation is a bug) -----
    {
        double gross = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            gross += std::abs(e[k]);
            if (std::abs(e[k]) > std::abs(w[k]) * (1.0 + kRelTol) + kAbsTol) {
                throw std::logic_error("portfolio construction: rounding increased a position");
            }
            if (!lim.shorting_allowed && units[k] < 0) {
                throw std::logic_error("portfolio construction: a short survived the short restriction");
            }
        }
        if (exceeds(gross, lim.max_gross_leverage)) {
            throw std::logic_error("portfolio construction: rounding breached gross leverage");
        }
    }
    r.constraints = out.take();

    // ---- outputs --------------------------------------------------------------
    const Vec irc = target_book.instrument_rc(e);
    const Vec lrc = target_book.leg_rc(e);
    const Vec crc = target_book.cluster_rc(e);
    for (std::size_t k = 0; k < n; ++k) {
        auto& a = r.instruments[k];
        a.units = static_cast<int>(units[k]);
        a.executable_weight = e[k];
        a.target_notional_usd = w[k] * lim.capital_usd;
        a.executable_notional_usd = static_cast<double>(units[k]) * a.input.unit_notional_usd();
        a.rounding_residual_usd = a.target_notional_usd - a.executable_notional_usd;
        if (a.admitted && a.input.adv_usd > 0.0) a.adv_participation = std::abs(a.executable_notional_usd) / a.input.adv_usd;
        a.risk_contribution = irc[k];
        a.below_one_unit = a.admitted && w[k] != 0.0 && units[k] == 0 &&
                           std::abs(w[k]) * lim.capital_usd < a.input.unit_notional_usd();
    }
    for (std::size_t i = 0; i < legs.size(); ++i) {
        auto& sa = r.signals[legs[i].signal];
        sa.allocated = true;
        sa.executable_weight = target_book.leg_weight(legs[i], e);
        sa.risk_contribution = lrc[i];
    }
    for (std::size_t c = 0; c < r.clusters.size(); ++c) r.clusters[c].executable_risk_contribution = crc[c];
    r.executable = target_book.stats(e);
    r.turnover_executable = turnover(e);
    r.status = r.executable.positions > 0 ? "CONSTRUCTED" : "NO_EXECUTABLE_POSITION";
    return r;
}

}  // namespace quant
