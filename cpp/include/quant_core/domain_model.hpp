#pragma once

#include "quant_core/contract_registry.hpp"
#include "quant_core/contract_selector.hpp"
#include "quant_core/domain_errors.hpp"
#include "quant_core/events.hpp"
#include "quant_core/ids.hpp"
#include "quant_core/types.hpp"

#include <optional>

namespace quant {

// Phase 05 domain model: structural validators, deterministic event ordering,
// and the deterministic Signal -> Order step. No event loop, no execution sim,
// no portfolio, no real risk rules here.

// ---- numeric / price-domain guard -------------------------------------
//
// The C++ core only ever sees NORMALIZED raw-contract prices (decimal points),
// e.g. NQ ~= 30000.00. Databento DBN fixed-point (1e-9) values -- e.g. 3.0e13 --
// are normalized at the Python boundary (alpha_agent.data.price_domain) and must
// never reach here. This is a coarse MAGNITUDE sanity bound (a fixed-point-scale
// tripwire), NOT the authoritative check; see docs/BOUNDARY_CONTRACT.md section C.
//
// A normalized futures price may be positive, ZERO, or NEGATIVE: WTI crude (CL)
// front-month futures settled at -$37.63 on 2020-04-20 and traded well below
// zero intraday. The core must not encode the false universal rule that a
// futures price is positive; product-specific "a negative price is bad data for
// THIS root" QA belongs in the Python/metadata layer, not here.
inline constexpr double kMaxPlausibleRawPrice = 1.0e9;

// A valid normalized raw-contract price: finite (not NaN, not +/-inf) and with a
// magnitude below the fixed-point tripwire. Sign is NOT constrained. Missing data
// is never encoded as a price value -- it travels the optional / rejection path.
bool is_plausible_raw_price(double price) noexcept;

// Signal.target_units is a signed integer contract target carried in a double
// (Phase 05 MVP invariant B). This is the tolerance within which a value is
// accepted as integral; anything outside it (e.g. 0.51) is a rejected Signal,
// never silently rounded.
inline constexpr double kTargetUnitsIntegralTol = 1.0e-6;

// True when `x` is a whole number within kTargetUnitsIntegralTol.
bool is_integral_within_tol(double x) noexcept;

// ---- structural validation (throws the matching domain error) --------
void validate_market_event(const MarketEvent& ev, const ContractRegistry& registry);
void validate_signal(const Signal& sig);
void validate_order(const Order& ord, const ContractRegistry& registry);
void validate_risk_decision(const RiskDecision& decision);

// ---- deterministic event ordering ---------------------------------
//
// Total order: (1) ts_event_ns, then (2) seq. `seq` is the ingestion sequence
// number; it breaks ties for identical timestamps and makes ordering independent
// of any unordered-container / hash iteration order.
inline bool event_before(const MarketEvent& a, const MarketEvent& b) noexcept {
    if (a.ts_event_ns != b.ts_event_ns) return a.ts_event_ns < b.ts_event_ns;
    return a.seq < b.seq;
}

struct MarketEventOrder {
    bool operator()(const MarketEvent& a, const MarketEvent& b) const noexcept {
        return event_before(a, b);
    }
};

// ---- Signal -> Order (deterministic active-contract selection) -----------
//
// Resolves a root-level Signal to a concrete Market Order against the contract
// the CURRENT market state says is active, via the ActiveContractResolver -- not
// an activation-date guess. `market_state` is the engine's current
// continuous-feed state (typically MarketState::from_event(execution_event));
// its instrument_id picks the contract and its as_of_ts_ns gates the live check.
// `ts_created_ns` stamps the Order.
//
// TARGET-POSITION delta semantics (Phase 06): `Signal.target_units` is a target
// position, not an order size. quantity = |target_units - current_position_units|
// and side follows the sign of that delta. A zero delta (already at target) or a
// Flat/zero-target Signal yields no order (std::nullopt). `current_position_units`
// is the signed size the caller currently holds in the resolved active contract.
// Throws InvalidSignal / ContractResolutionError.
//
// This is the sanctioned standalone Signal -> Order path. The returned Order
// still has to pass validate_order and the RiskManager before it can execute.
// (The Phase 06 BacktestEngine builds orders through the same validators and
// additionally emits the explicit close-leg of a roll.)
std::optional<Order> make_order_from_signal(const Signal& sig,
                                            const ActiveContractResolver& resolver,
                                            const MarketState& market_state,
                                            int current_position_units,
                                            OrderId order_id,
                                            std::int64_t ts_created_ns);

}  // namespace quant
