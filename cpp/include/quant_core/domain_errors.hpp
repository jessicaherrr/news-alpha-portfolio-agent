#pragma once

#include <stdexcept>

namespace quant {

// One small, flat family of domain errors (Phase 05). Each type names the layer
// that rejected a value, so a caller can tell a bad Signal from a bad Order from
// a bad Fill without matching on strings. Deliberately NOT a deep hierarchy.
//
//   DomainError
//     +-- InvalidMarketEvent       -- a MarketEvent failed structural validation
//     +-- InvalidSignal            -- a Signal failed structural validation
//     +-- InvalidOrder             -- an Order failed invariant checks
//     +-- ContractResolutionError  -- root/ts did not resolve to a live contract
//     +-- RiskInvariantError       -- a RiskDecision violated its own invariants
//     +-- ExecutionDomainError     -- the execution/fill layer rejected a value
//           +-- FillResolutionError (declared in fill.hpp)
class DomainError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class InvalidMarketEvent : public DomainError {
public:
    using DomainError::DomainError;
};

class InvalidSignal : public DomainError {
public:
    using DomainError::DomainError;
};

class InvalidOrder : public DomainError {
public:
    using DomainError::DomainError;
};

class ContractResolutionError : public DomainError {
public:
    using DomainError::DomainError;
};

class RiskInvariantError : public DomainError {
public:
    using DomainError::DomainError;
};

class ExecutionDomainError : public DomainError {
public:
    using DomainError::DomainError;
};

}  // namespace quant
