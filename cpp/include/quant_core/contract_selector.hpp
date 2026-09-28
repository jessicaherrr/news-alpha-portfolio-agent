#pragma once

#include "quant_core/contract.hpp"
#include "quant_core/contract_registry.hpp"
#include "quant_core/events.hpp"

#include <cstdint>
#include <string_view>

namespace quant {

// ---- current continuous-market state ---------------------------------------
//
// Phase 04.5 established the source of active-contract truth for a futures
// continuous feed: whichever real contract that feed's bars currently carry. The
// roll is nothing more than the `instrument_id` transition of the ordered feed
// (docs/FUTURES_HISTORY.md) -- it is NEVER inferred from contract-month strings
// or activation dates, because multiple contracts of one root (e.g. NQM6 and
// NQU6) are simultaneously live around a roll.
//
// The engine (Phase 06) tracks this as it walks the event stream, updating it
// from each MarketEvent.instrument_id. Phase 05 only defines the value type and
// the resolver that consumes it.
struct MarketState {
    std::uint32_t active_instrument_id{0};  // real contract the feed is on right now; 0 == unknown
    std::int64_t  as_of_ts_ns{0};           // ts_event_ns of the event that set it

    static MarketState from_event(const MarketEvent& ev) noexcept {
        return MarketState{ev.instrument_id, ev.ts_event_ns};
    }
    bool has_active_contract() const noexcept { return active_instrument_id != 0; }
};

// ---- deterministic active-contract selection (Phase 05) --------------------
//
// A Signal is ROOT-level intent ("go long NQ"). This interface -- NOT the
// strategy, NOT an activation-date heuristic -- owns turning that intent into a
// concrete tradable contract.
//
//   MarketEvent(current instrument_id) -> StrategyContext -> Signal(root=NQ)
//       -> ActiveContractResolver(root, current MarketState)
//       -> current market-state instrument_id -> ContractRegistry validation
//       -> real raw contract
//
// The ContractRegistry stays authoritative for contract metadata and validity
// (does the instrument exist, does the root match, is it live at the timestamp,
// raw symbol / tick / multiplier), but it must NOT decide which contract is
// currently active from root + timestamp alone.
//
// Phase 06+ may add liquidity-aware or calendar-aware resolvers; the interface
// does not change. Roll-execution logic is out of scope here.
class ActiveContractResolver {
public:
    virtual ~ActiveContractResolver() = default;

    // Resolve `root_symbol` intent, against the CURRENT market state, to the real
    // raw contract to trade. Throws ContractResolutionError when:
    //   * the market state has no active instrument (id == 0);
    //   * the active instrument_id is not in the registry;
    //   * the resolved contract's root_symbol != `root_symbol`
    //     (a Signal for NQ must fail if the feed is currently on an ES contract);
    //   * the contract was not live at `state.as_of_ts_ns`.
    virtual const ContractSpec& resolve(std::string_view root_symbol,
                                        const MarketState& state) const = 0;
};

// MVP resolver: the current market-state instrument_id, validated against the
// registry. It performs NO activation-date inference.
class RegistryActiveContractResolver final : public ActiveContractResolver {
public:
    explicit RegistryActiveContractResolver(const ContractRegistry& registry) noexcept
        : registry_(registry) {}

    const ContractSpec& resolve(std::string_view root_symbol,
                                const MarketState& state) const override;

private:
    const ContractRegistry& registry_;
};

}  // namespace quant
