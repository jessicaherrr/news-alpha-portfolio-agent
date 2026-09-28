#!/usr/bin/env bash
set -euo pipefail

# Use Ninja when available (the documented toolchain installs it); fall back to
# the default generator otherwise.
GENERATOR_ARGS=()
if command -v ninja >/dev/null 2>&1; then
  GENERATOR_ARGS=(-G Ninja)
fi

cmake -S . -B build/cpp -DCMAKE_BUILD_TYPE=Release "${GENERATOR_ARGS[@]}"
cmake --build build/cpp --parallel
