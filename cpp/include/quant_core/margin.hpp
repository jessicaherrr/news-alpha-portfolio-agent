#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace quant {

// ============================================================================
// Explicit, typed futures margin metadata (Phase 08)
// ============================================================================
//
// CLAUDE.md market-data rule 5 + Phase 08 requirement 5: margin is NEVER invented
// from Databento and NEVER guessed from notional. Exchange initial / maintenance
// margin is a separate, human-supplied, dated input. This is the typed interface
// that carries it into the deterministic risk core.
//
// A MarginRequirement is stated PER CONTRACT, PER ROOT (ES / NQ / CL / GC / ZN
// ...). The dollar requirement for a position is `per_contract * abs(units)` --
// a deterministic multiply, not a model. SPAN / cross-margin / correlation
// offsets are explicitly out of scope (see PortfolioState's correlated-exposure
// TODO); the MVP is additive by root.
//
// `source` + `as_of_ns` make every figure auditable: exchange margins change
// (often), so a backtest must record which schedule it used, exactly like it
// records the dataset version (CLAUDE backtest rule 5).

struct MarginRequirement {
    std::string  root_symbol;                 // bare root, e.g. "ES"
    double       initial_margin_usd{0.0};     // per contract, USD, > 0
    double       maintenance_margin_usd{0.0}; // per contract, USD, 0 < maint <= initial
    std::string  source;                      // provenance, e.g. "cme_2026_01_outright", "user_override"
    std::int64_t as_of_ns{0};                 // when this figure was published / entered (UTC ns)

    // Structural validity only. Says nothing about whether the figure is current.
    bool is_valid() const noexcept {
        return !root_symbol.empty() && initial_margin_usd > 0.0 &&
               maintenance_margin_usd > 0.0 && maintenance_margin_usd <= initial_margin_usd;
    }
};

// What the risk core does when it needs a margin figure for a root and there is
// no MarginRequirement for it. There is deliberately NO "estimate from notional"
// option -- an unknown requirement is a visible condition, handled by policy.
enum class MissingMarginPolicy : int {
    // DEFAULT, reliability-safe: a risk-INCREASING order in a root with no margin
    // metadata is REJECTed ("missing_margin_metadata"); a held position with no
    // metadata makes PortfolioState.margin_complete == false so it is visible.
    Reject = 0,
    // EXPLICIT OPT-IN (research only): treat the missing requirement as 0. The
    // portfolio is still flagged margin_complete == false so the assumption is
    // never silent.
    TreatAsZero = 1,
};

// Deterministic lookup table of margin requirements, keyed by bare root symbol.
// std::map -> iteration and lookup are ordered, never hash order.
class MarginModel {
public:
    // Overwrites any existing requirement for the same root. Throws
    // std::invalid_argument if `req` is structurally invalid.
    void set(MarginRequirement req);

    bool has(std::string_view root_symbol) const noexcept;
    const MarginRequirement* find(std::string_view root_symbol) const noexcept;

    // Per-contract initial / maintenance margin for `root_symbol`, or std::nullopt
    // when unknown (the caller applies MissingMarginPolicy).
    std::optional<double> initial_per_contract(std::string_view root_symbol) const noexcept;
    std::optional<double> maintenance_per_contract(std::string_view root_symbol) const noexcept;

    std::size_t size() const noexcept { return by_root_.size(); }
    bool empty() const noexcept { return by_root_.empty(); }

    std::vector<MarginRequirement> requirements() const;  // ordered by root

private:
    std::map<std::string, MarginRequirement, std::less<>> by_root_;
};

// Parse a margin CSV. Header (exact order):
//   root_symbol,initial_margin_usd,maintenance_margin_usd,source,as_of_ns
// Throws std::runtime_error on a missing file, header mismatch, unparseable row,
// or a structurally invalid requirement.
MarginModel load_margin_model(const std::string& path);

}  // namespace quant
