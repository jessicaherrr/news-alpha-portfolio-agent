#!/usr/bin/env bash
# macOS only. Clears the UF_HIDDEN file flag that some Desktop/iCloud/security
# tooling applies to files pip writes into the virtualenv. When an editable
# install's `__editable__*.pth` is UF_HIDDEN, CPython's site.py silently skips it
# and the `alpha_agent` package is not importable from a plain interpreter.
#
# Run this after any `pip install -e ...` if `python -c "import alpha_agent"`
# fails. The repo scripts and pytest do not depend on it (they add python/ to
# sys.path themselves), but it keeps the installed package usable directly.
set -euo pipefail

VENV_DIR="${1:-.venv}"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "no venv at $VENV_DIR" >&2
  exit 1
fi

if [[ "$(uname)" != "Darwin" ]]; then
  echo "not macOS; nothing to do"
  exit 0
fi

chflags -R nohidden "$VENV_DIR"
echo "cleared UF_HIDDEN under $VENV_DIR"
