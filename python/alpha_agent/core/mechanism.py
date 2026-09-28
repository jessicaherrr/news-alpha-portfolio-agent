"""Phase 5 -- Mechanism is already asset-neutral (Phase 0 evolution plan
section 8.1): an economic mechanism name (``TREND``, ``CARRY``,
``TERM_STRUCTURE``, ...) is not itself a Futures concept, only its concrete
measurement/implementation is. This module re-exports the existing, unchanged
``EconomicMechanism`` enum rather than defining a second one -- a future
generation imports the SAME closed catalog, never a parallel copy.
"""
from __future__ import annotations

from alpha_agent.knowledge.models import EconomicMechanism

__all__ = ["EconomicMechanism"]
