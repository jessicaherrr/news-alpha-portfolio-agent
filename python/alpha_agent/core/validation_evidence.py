"""Phase 5 -- Validation evidence references. ``ExperimentEvidenceRef``
(``alpha_agent.alpha_memory.schemas``) is already a generic pointer into one
real registry experiment (``experiment_id`` / ``experiment_identity`` /
``headline_verdict`` / ``reason_codes``) -- never a duplicated PnL/fill
payload (CLAUDE.md: the registry, not this reference, is the source of
official truth). ``RegistryVerdict`` / ``Authority`` / ``TrialRole``
(``alpha_agent.registry.enums``) are the closed vocabularies that reference
resolves against. All four re-exported verbatim, unchanged.
"""
from __future__ import annotations

from alpha_agent.alpha_memory.schemas import ExperimentEvidenceRef
from alpha_agent.registry.enums import Authority, RegistryVerdict, TrialRole

__all__ = ["Authority", "ExperimentEvidenceRef", "RegistryVerdict", "TrialRole"]
