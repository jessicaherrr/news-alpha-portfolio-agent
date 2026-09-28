"""Phase 6 closure: the Factor Library's evidence-oriented status --
VALIDATED / UNDER_RESEARCH / RESEARCH_ARCHIVE -- computed purely from an
:class:`~alpha_agent.alpha_memory.schemas.AlphaResearchObject`'s ALREADY
COMPUTED :class:`~alpha_agent.alpha_memory.schemas.EvidenceProfile`. No new
scientific computation, no second database, no query the registry hasn't
already answered (``EvidenceProfile.scientific_evidence`` is built by
``alpha_memory.builder._scientific_evidence_for_family`` directly off real
``ExperimentView.verdict`` values).

Deliberately NOT a boolean ("successful"/"failed") -- a rejected StrategySpec
implementation does not invalidate the broader Factor or Mechanism (Phase 6
closure instruction 7); RESEARCH_ARCHIVE means "every tested implementation
so far came back REJECT", never "this economic idea is false".
"""
from __future__ import annotations

from enum import Enum

from alpha_agent.alpha_memory.schemas import AlphaResearchObject

__all__ = ["FactorStatus", "factor_status"]


class FactorStatus(str, Enum):
    """A Factor's CURRENT evidence-oriented research status. Recomputed
    fresh from the registry every time -- never stored, never cached, so a
    new experiment on the same Factor immediately changes which section it
    renders in."""

    #: at least one strategy variant's canonical trial genuinely PASSed
    #: under the frozen ReliabilityPolicy (never fabricated -- absent unless
    #: a real committed verdict says so).
    VALIDATED = "VALIDATED"
    #: no PASS yet, but real open research remains: a canonical trial is
    #: INCONCLUSIVE, NOT_ADJUDICATED, has no canonical trial at all yet, or
    #: disagrees across strategy variants (MIXED without a PASS).
    UNDER_RESEARCH = "UNDER_RESEARCH"
    #: every strategy variant with a canonical trial came back REJECT, and
    #: none is still open -- a real, informative negative result, not a
    #: claim that the underlying economic mechanism is false.
    RESEARCH_ARCHIVE = "RESEARCH_ARCHIVE"


def factor_status(obj: AlphaResearchObject) -> FactorStatus:
    values = list(obj.evidence_profile.scientific_evidence.values())
    if any("PASS" in v for v in values):
        return FactorStatus.VALIDATED
    if any(v != "REJECT" for v in values):
        return FactorStatus.UNDER_RESEARCH
    if values:
        return FactorStatus.RESEARCH_ARCHIVE
    # no strategy variant has a scientific_evidence entry at all -- cannot
    # happen for a real AlphaResearchObject (it always has >=1 variant with
    # >=1 real experiment), kept as a defensive, honest fallback rather than
    # an unreachable-code assumption.
    return FactorStatus.UNDER_RESEARCH
