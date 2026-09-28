"""Phase 6 -- the ETF Research Pilot.

Asset-specific market mechanics for the ``AssetDomain.ETF`` research
generation (``alpha_agent.core.instrument.AssetDomain``), mirroring the
"shared research intelligence, asset-specific market mechanics" principle:
this package owns ETF corporate actions, data-source provenance, and the
ETF instrument-economics adapter. It does not fork Mechanism/Factor/
Hypothesis/Validation/Registry concepts, which stay shared with Futures.

No Futures file is imported or modified by this package.
"""
from __future__ import annotations
