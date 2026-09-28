"""Phase 4 -- Learn: an explanation layer over the existing deterministic
research system (CLAUDE.md architecture boundaries), never a second source of
scientific truth.

Every submodule here reads real, already-committed state (the frozen Phase 11
family docs, the Phase 13/14 registry, `ReliabilityPolicy` reason codes) and
presents it -- it computes no PnL, fill, verdict, or gate decision of its own.
`alpha_agent.ui.views.learn` and `alpha_agent.ui.services` are the only
callers; this package has no Streamlit import and no UI concern.
"""
from __future__ import annotations
