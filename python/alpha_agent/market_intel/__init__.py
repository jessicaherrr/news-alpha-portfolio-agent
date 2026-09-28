"""Market Intelligence Data Completion Pass, Checkpoints E-G -- MARKET NEWS
and SCHEDULED EVENTS. This is the OBSERVATION plane, exactly like
`alpha_agent.marketdata`: current/recent market color for display and
conversational context, never scientific evidence.

Deliberately a SEPARATE package from `alpha_agent.knowledge` (strategy-
mechanism research provenance -- `ResearchSource`) -- Section 15/31: "Market
News" and "Research Sources" are different concepts and must never be
merged, never share a store, and a `MarketNewsItem` must never enter
`ExperimentRegistry`, `FailureMemory`, strict validation, or Research
Promise. See `tests/python/test_market_intel_isolation.py`.
"""
from __future__ import annotations
