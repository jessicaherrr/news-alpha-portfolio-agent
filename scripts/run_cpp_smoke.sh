#!/usr/bin/env bash
set -euo pipefail
if [[ ! -x build/cpp/cpp/quant_core_smoke ]]; then
  bash scripts/build_cpp.sh
fi
build/cpp/cpp/quant_core_smoke
