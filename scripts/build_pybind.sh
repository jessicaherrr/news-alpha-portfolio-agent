#!/usr/bin/env bash
set -euo pipefail

# Phase 19: build the optional pybind11 fast-boundary module (quant_core_py).
#
# The module is OPTIONAL -- the core C++ tests and the Python test suite run
# without it. Build it when you want the in-process research transport.
#
# pybind11 is a build-time dependency only. Install it into the active env with
#   pip install "pybind11>=2.12"
# (it is in the [dev] extra of pyproject.toml). No runtime dependency is added.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

# NB: `python -m pybind11 --cmakedir` shell-quotes a path containing spaces;
# get_cmake_dir() returns it raw, which is what CMake wants.
PYBIND11_DIR="$("$PYTHON_BIN" -c 'import pybind11; print(pybind11.get_cmake_dir())')"
echo "pybind11 cmake dir: $PYBIND11_DIR"

GENERATOR_ARGS=()
if command -v ninja >/dev/null 2>&1; then
  GENERATOR_ARGS=(-G Ninja)
fi

cmake -S . -B build/cpp -DCMAKE_BUILD_TYPE=Release -DBUILD_PYBIND=ON \
  -Dpybind11_DIR="$PYBIND11_DIR" \
  -DPYTHON_EXECUTABLE="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')" \
  "${GENERATOR_ARGS[@]}"
cmake --build build/cpp --parallel

MODULE="$(find build/cpp -name 'quant_core_py*.so' -o -name 'quant_core_py*.pyd' | head -1)"
echo "built module: $MODULE"
echo "import it with:  PYTHONPATH=\"$(dirname "$MODULE")\" $PYTHON_BIN -c 'import quant_core_py'"
