"""Phase 5 -- Hypothesis is already asset-neutral: ``HypothesisSpec`` has no
Futures-only field (no multiplier, tick size, or roll concept anywhere in its
shape). Re-exported verbatim, unchanged -- extending its shape is a FROZEN
RESEARCH SEMANTICS change (CLAUDE.md) this phase does not make.
"""
from __future__ import annotations

from alpha_agent.schemas.hypothesis import HypothesisSpec

__all__ = ["HypothesisSpec"]
