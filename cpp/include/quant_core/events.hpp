#pragma once

#include "quant_core/ids.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace quant {

// ---- strong enums -----------------------------------------------------------

enum class Side : int { Buy = 1, Sell = -1 };
enum class OrderType : int { Market, Limit, Stop };
enum class TimeInForce : int { Day, GTC, IOC };

// Phase 05 implements ONLY `Bar`. SessionOpen / SessionClose / Roll and
// quote / trade / settlement events are deferred to Phase 06+, where each gains a
// typed payload (extra optional members or a payload variant) AND a matching
// branch in validate_market_event. Declaring an enumerator before its payload can
// be represented and validated would let an un-checkable MarketEvent pass
// validation, so the reserved types are added together with their handling.
enum class MarketEventType : int { Bar = 0 };

// The price space a number lives in. Only RawContract prices may be used to
// simulate fills (see docs/BOUNDARY_CONTRACT.md section E).
enum class PriceDomain : int {
    RawContract   = 0,  // actual traded price of one specific real contract
    RawContinuous = 1,  // unadjusted stitched continuous series -- inspection only
    BackAdjusted  = 2,  // synthetic research series -- signals only, never fills
};

// ---- the deterministic event flow ---------------------------------------
//
//   MarketEvent -> Strategy -> Signal -> [ActiveContractResolver] -> Order
//               -> [RiskManager] -> RiskDecision -> [ExecutionSim] -> Fill
//               -> Portfolio update
//
// Phase 05 freezes the TYPES only. Phases 06-08 build the loop, the execution
// simulator, portfolio accounting and the real risk rules. The structural
// validators for these shapes live in domain_model.hpp.

// ---- MarketEvent --------------------------------------------------------
//
// Timestamp is authoritative UTC nanoseconds. `instrument_id` must resolve
// through the ContractRegistry (checked by validate_market_event). Phase 05
// carries only the Bar payload; quote / trade / settlement / session payloads
// attach later as extra optional members or a payload variant WITHOUT breaking a
// consumer that reads only {type, ts_event_ns, seq, instrument_id}. The
// `instrument_id` on the stream is also the current continuous-market state the
// engine feeds to the ActiveContractResolver (see contract_selector.hpp).
struct MarketEvent {
    MarketEventType type{MarketEventType::Bar};
    std::int64_t    ts_event_ns{0};   // authoritative UTC nanoseconds since epoch
    SeqNum          seq{0};           // deterministic tie-breaker for equal ts_event_ns
    std::uint32_t   instrument_id{0}; // real contract; must be in the ContractRegistry
    // Bar payload (Phase 05: type is always Bar); PriceDomain::RawContract units:
    double       open{0.0};
    double       high{0.0};
    double       low{0.0};
    double       close{0.0};
    std::int64_t volume{0};
};

// ---- Signal -----------------------------------------------------------
//
// Strategy intent, expressed against a ROOT symbol / continuous series -- never a
// raw contract, never a price. Signal intent and execution are SEPARATE layers:
// the engine (not the strategy) turns this into an Order against a concrete
// contract with a concrete quantity, then a RiskDecision, then a Fill.
//
// `target_units` is a SIGNED INTEGER CONTRACT TARGET carried in a double for a
// stable frozen shape (Phase 05 decision, MVP invariant B): validate_signal
// rejects a non-integral value (tolerance 1e-6) so 0.51 contracts can never
// silently round to 1. A strategy that wants fractional sizing must round to a
// whole contract count itself and own that decision.
//
// A Signal must NOT carry: fill price, commission, slippage, realised PnL, or an
// arbitrary raw contract chosen by an LLM.
enum class SignalDirection : int { Flat = 0, Long = 1, Short = -1 };

struct Signal {
    SignalId     signal_id{0};       // deterministic; stamped by the engine
    std::int64_t ts_decision_ns{0};  // timestamp of the newest information used
    std::string  root_symbol;        // e.g. "NQ" -- a bare root, not "NQU6" / "NQ.v.0"
    double       target_units{0.0};  // signed desired position, in contracts (MVP intent)
    std::string  rationale_code;     // short machine tag, e.g. "tsmom_up"

    // Direction implied by target_units. The magnitude (target_units) is intent;
    // the executable quantity is decided downstream.
    SignalDirection direction() const noexcept {
        if (target_units > 0.0) return SignalDirection::Long;
        if (target_units < 0.0) return SignalDirection::Short;
        return SignalDirection::Flat;
    }
};

// ---- Order ----------------------------------------------------------
//
// A concrete order against a real contract. Produced by the engine from a Signal
// via an ActiveContractResolver, never by an LLM. Invariants enforced by
// validate_order.
struct Order {
    OrderId       order_id{0};
    SignalId      signal_id{0};        // the Signal this order serves (traceability)
    std::int64_t  ts_created_ns{0};
    std::uint32_t instrument_id{0};    // resolved real contract; != 0
    std::string   raw_symbol;          // must equal registry[instrument_id].raw_symbol
    Side          side{Side::Buy};
    int           quantity{0};         // > 0
    OrderType     order_type{OrderType::Market};
    std::optional<double> limit_price{};  // required (finite, > 0) iff order_type == Limit
    std::optional<double> stop_price{};   // required (finite, > 0) iff order_type == Stop
    TimeInForce   tif{TimeInForce::Day};
};

// ---- Fill ---------------------------------------------------------
//
// The deterministic result of executing an order. Always fully contract-resolved
// and always PriceDomain::RawContract. Construct ONLY via make_fill() (fill.hpp).
struct Fill {
    FillId        fill_id{0};
    OrderId       order_id{0};        // the Order this fill settles (traceability)
    std::int64_t  ts_fill_ns{0};
    std::uint32_t instrument_id{0};   // real contract; never 0 in a valid Fill
    std::string   raw_symbol;         // real contract symbol; never empty, never continuous
    Side          side{Side::Buy};
    int           quantity{0};
    double        fill_price{0.0};    // raw traded price, tick-aligned
    double        tick_size{0.0};     // copied from ContractSpec, > 0
    double        multiplier{0.0};    // copied from ContractSpec, > 0 (USD per 1.0 price unit)
    double        commission_usd{0.0};
    double        slippage_ticks{0.0};
    PriceDomain   price_domain{PriceDomain::RawContract};
};

// ---- RiskDecision ----------------------------------------------------
//
// Immutable, deterministic risk ruling. Order -> RiskManager -> RiskDecision is
// MANDATORY before execution; no Strategy or Agent may bypass it and no LLM may
// override it (CLAUDE.md risk rule 1). Invariants enforced by
// validate_risk_decision:
//   APPROVE  approved_quantity == requested_quantity  (> 0)
//   RESIZE   0 < approved_quantity < requested_quantity
//   REJECT   approved_quantity == 0
enum class RiskVerdict : int { Approve, Resize, Reject };

struct RiskDecision {
    std::int64_t  ts_decision_ns{0};
    OrderId       order_id{0};
    std::uint32_t instrument_id{0};
    RiskVerdict   verdict{RiskVerdict::Approve};
    int           requested_quantity{0};
    int           approved_quantity{0};
    std::string   reason_code{"ok"};  // machine tag, e.g. "max_contracts_per_symbol"
};

}  // namespace quant
