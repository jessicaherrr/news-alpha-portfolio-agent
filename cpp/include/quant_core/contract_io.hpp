#pragma once

#include "quant_core/contract_registry.hpp"

#include <string>
#include <vector>

namespace quant {

// Parse the boundary contracts CSV (docs/BOUNDARY_CONTRACT.md section G).
//
// Header (exact order):
//   instrument_id,raw_symbol,root_symbol,exchange,tick_size,multiplier,
//   activation_ns,expiration_ns,first_notice_ns,last_trade_ns
//
// `first_notice_ns` and `last_trade_ns` may be empty. Throws std::runtime_error
// on a missing file, a header mismatch, or an unparseable row.
std::vector<ContractSpec> parse_contracts_csv(const std::string& path);

// Convenience: parse and load into a registry (which re-validates every row).
ContractRegistry load_contract_registry(const std::string& path);

}  // namespace quant
