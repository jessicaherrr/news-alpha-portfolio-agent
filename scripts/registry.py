"""Phase 14 -- researcher CLI wrapper for the experiment registry.

A thin entry point so the registry is inspectable without setting PYTHONPATH::

    python scripts/phase_14_registry.py summary
    python scripts/phase_14_registry.py list --family tsmom
    python scripts/phase_14_registry.py show NQ__TSMOM__CANONICAL__VALIDATION_2023_2024
    python scripts/phase_14_registry.py find-related --family tsmom --root NQ \
        --param fast_horizon=21 --param slow_horizon=120
    python scripts/phase_14_registry.py failures --class CONTRACT_ECONOMICS_FAILURE

See ``python -m alpha_agent.registry.cli --help`` for the full command set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from alpha_agent.registry.cli import main

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO / "data" / "registry" / "experiments.sqlite"

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--db" not in argv:
        argv = ["--db", str(DEFAULT_DB), *argv]
    sys.exit(main(argv))
