"""News Alpha Phase C -- the MECHANISM-ADJUSTED IMPACT ASSESSMENT (second pass).

Phase A's Initial Impact Scan scored each (channel, market exposure) pair from
a STATIC prior -- the channel table's PRIMARY / SECONDARY / TERTIARY exposure
strength. Once the event has a mechanism graph and signal paths, this pass
revisits every one of those exposures with ``mechanism-adjusted-impact/2``.

TWO QUANTITIES, KEPT APART. Impact RELEVANCE (how much of an exposure's
economics the event moves) and MECHANISM SUPPORT (whether a reviewed route
from the event reaches the exposure's economic basis, and how confident its
weakest link is) are different things. A HIGH-confidence link says the
relationship probably holds -- not that its effect is large. So link
confidence is NEVER mapped onto exposure strength, and it never moves an
impact level. (Rule /1 did map HIGH/MEDIUM/LOW onto PRIMARY/SECONDARY/
TERTIARY; that was a semantic leak and is withdrawn.)

The rule, per exposure:

1. BASIS. The exposure's economic basis comes from the reviewed
   `signal_path_library.EXPOSURE_BASIS` (copper futures rest on copper
   demand). Only paths from the SAME channel's anchor count.
2. MECHANISM SUPPORT (`MechanismSupport`), reported, never scored:
   CORROBORATED (a RESEARCHABLE path reaches the basis), PROPOSED_ONLY
   (paths reach it, none researchable yet), NOT_REACHED, CHANNEL_NOT_SEEDED,
   NO_ECONOMIC_BASIS. The cited path's weakest-link confidence is carried
   beside it as `mechanism_confidence` -- a property of the mechanism.
3. LEVEL. Nothing in the graph establishes ECONOMIC MAGNITUDE (no exposure
   share, elasticity or size of effect), so the adjusted level EQUALS the
   initial level and `magnitude_established` is False, with that uncertainty
   stated on every revision. A future pass may move a level only from an
   established magnitude input -- and then by at most one step.

BOTH ASSESSMENTS ARE PRESERVED. This pass builds a SEPARATE
`MechanismAdjustedImpact` that carries the initial level next to the adjusted
one and the initial scan's fingerprint; the frozen `InitialImpactScan` is
never modified (test-enforced). Research triage only: no expected return,
probability, price direction, trade, weight or verdict -- the economic
direction of each basis consequence is shown, never a call on an instrument.
"""
from __future__ import annotations

import hashlib
from enum import Enum
from typing import Literal

from pydantic import BaseModel

from alpha_agent.news_alpha.channels import EconomicChannel, ExposureStrength, channel_definition
from alpha_agent.news_alpha.mandate import MandateDomain
from alpha_agent.news_alpha.mechanism_graph import ImpliedMovement
from alpha_agent.news_alpha.schemas import ExposureContribution, ImpactLevel, InitialImpactScan
from alpha_agent.news_alpha.signal_path_library import DEFAULT_PATH_LIBRARY, SignalPathLibrary
from alpha_agent.news_alpha.signal_paths import PathStatus, SignalPath, SignalPathDiscovery
from alpha_agent.news_alpha.transmission import LinkConfidence, SupportStatus
from alpha_agent.news_alpha.transmission_library import STATE_CATALOG

__all__ = [
    "ADJUSTMENT_RULE",
    "MAGNITUDE_NOTE",
    "MECHANISM_ADJUSTED_SCHEMA_VERSION",
    "BasisConsequence",
    "ExposureRevision",
    "LevelChange",
    "MechanismAdjustedImpact",
    "MechanismAdjustedImpactAssessment",
    "MechanismSupport",
    "adjust_impact",
    "scan_fingerprint",
]

MECHANISM_ADJUSTED_SCHEMA_VERSION = "mechanism-adjusted-impact/2"
ADJUSTMENT_RULE = "mechanism-adjusted-impact/2"
MAGNITUDE_NOTE = (
    "Economic magnitude not established: the mechanism shows a route to this exposure's economic basis, not how "
    "large the effect is, so the initial level is kept."
)

_STATE_LABELS = {s.state_id: s.label for s in STATE_CATALOG}


class MechanismSupport(str, Enum):
    """Whether the event's mechanism reaches an exposure's economic basis --
    mechanism evidence, never an exposure strength or an impact level."""

    #: At least one RESEARCHABLE path reaches the basis.
    CORROBORATED = "CORROBORATED"
    #: Paths reach the basis, but none is researchable yet.
    PROPOSED_ONLY = "PROPOSED_ONLY"
    #: The channel is seeded, but no path from its anchor reaches the basis.
    NOT_REACHED = "NOT_REACHED"
    #: The channel has no seeded transmission yet.
    CHANNEL_NOT_SEEDED = "CHANNEL_NOT_SEEDED"
    #: No reviewed economic basis is recorded for this exposure.
    NO_ECONOMIC_BASIS = "NO_ECONOMIC_BASIS"


class LevelChange(str, Enum):
    #: RAISED / LOWERED need an established economic magnitude; no current
    #: input provides one, so rule /2 only yields UNCHANGED / NOT_ASSESSED.
    RAISED = "RAISED"
    LOWERED = "LOWERED"
    UNCHANGED = "UNCHANGED"
    NOT_ASSESSED = "NOT_ASSESSED"


class BasisConsequence(BaseModel):
    """One economic state an exposure rests on, and how the event's paths
    move it -- an economic direction, never a price call."""

    model_config = {"frozen": True, "extra": "forbid"}

    state_id: str
    label: str
    #: ``None`` when no path from this channel's anchor reaches the state.
    movement: ImpliedMovement | None = None
    path_ids: tuple[str, ...] = ()


class ExposureRevision(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    channel: EconomicChannel
    exposure_label: str
    symbols: tuple[str, ...] = ()
    #: Phase A's static exposure strength -- unchanged by this pass.
    exposure_strength: ExposureStrength
    initial_level: ImpactLevel
    adjusted_level: ImpactLevel
    #: Always False today: no input establishes economic magnitude.
    magnitude_established: bool = False
    # -- mechanism evidence, kept apart from relevance -------------------------
    mechanism_support: MechanismSupport
    #: Weakest-link confidence of the cited path: confidence that the ROUTE
    #: holds, never the size of the effect.
    mechanism_confidence: LinkConfidence | None = None
    #: Weakest-link backing of the cited path (verified source or not).
    mechanism_backing: SupportStatus | None = None
    basis: tuple[BasisConsequence, ...] = ()
    cited_path_id: str | None = None
    rationale: str
    uncertainty: tuple[str, ...] = ()


class MechanismAdjustedImpactAssessment(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    assessment_pass: Literal["MECHANISM_ADJUSTED"] = "MECHANISM_ADJUSTED"
    event_id: str
    asset_domain: MandateDomain
    #: Copied from the initial assessment -- both passes stay side by side.
    initial_level: ImpactLevel | None
    adjusted_level: ImpactLevel | None
    change: LevelChange
    revisions: tuple[ExposureRevision, ...] = ()

    def corroborated(self) -> int:
        return sum(r.mechanism_support is MechanismSupport.CORROBORATED for r in self.revisions)


class MechanismAdjustedImpact(BaseModel):
    """The second pass for one scanned event. References the initial scan by
    fingerprint; never replaces it."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = MECHANISM_ADJUSTED_SCHEMA_VERSION
    assessment_pass: Literal["MECHANISM_ADJUSTED"] = "MECHANISM_ADJUSTED"
    rule: str = ADJUSTMENT_RULE
    event_id: str
    initial_scan_fingerprint: str
    mechanism_graph_id: str
    signal_path_discovery_id: str
    assessments: tuple[MechanismAdjustedImpactAssessment, ...]
    note: str = (
        "Mechanism-adjusted impact: each initial exposure checked against this event's signal paths. Mechanism "
        "support is reported beside the level, never converted into it; levels stay at the initial scan until an "
        "economic magnitude is established. The initial scan is preserved unchanged. Research triage only -- not an "
        "expected return, probability, trade instruction, portfolio weight, or scientific verdict."
    )

    def assessment(self, domain: MandateDomain) -> MechanismAdjustedImpactAssessment:
        return next(a for a in self.assessments if a.asset_domain is domain)

    @property
    def changed(self) -> tuple[MechanismAdjustedImpactAssessment, ...]:
        return tuple(a for a in self.assessments if a.change in (LevelChange.RAISED, LevelChange.LOWERED))

    def fingerprint(self) -> str:
        return "mechimpact1:" + hashlib.sha256(self.model_dump_json().encode()).hexdigest()


def scan_fingerprint(scan: InitialImpactScan) -> str:
    return "impactscan1:" + hashlib.sha256(scan.model_dump_json().encode()).hexdigest()


def _movement(paths: list[SignalPath]) -> ImpliedMovement | None:
    if not paths:
        return None
    directions = {p.expected_direction for p in paths}
    definite = directions & {ImpliedMovement.UP, ImpliedMovement.DOWN}
    if len(definite) == 2:
        return ImpliedMovement.MIXED
    if definite:
        return definite.pop()
    return ImpliedMovement.INDETERMINATE if ImpliedMovement.INDETERMINATE in directions else ImpliedMovement.UNANCHORED


_CONFIDENCE_RANK = {c: i for i, c in enumerate(LinkConfidence)}


def _revision(c: ExposureContribution, support: MechanismSupport, rationale: str, *,
              basis: tuple[BasisConsequence, ...] = (), cited: SignalPath | None = None) -> ExposureRevision:
    return ExposureRevision(
        channel=c.channel, exposure_label=c.exposure_label, symbols=c.symbols, exposure_strength=c.exposure_strength,
        initial_level=c.impact_level, adjusted_level=c.impact_level, mechanism_support=support,
        mechanism_confidence=cited.confidence if cited else None,
        mechanism_backing=cited.weakest_support if cited else None,
        basis=basis, cited_path_id=cited.path_id if cited else None, rationale=rationale,
        uncertainty=(MAGNITUDE_NOTE,),
    )


def _revise(
    c: ExposureContribution, discovery: SignalPathDiscovery, library: SignalPathLibrary,
) -> ExposureRevision:
    label = channel_definition(c.channel).label
    if c.channel in discovery.unseeded_channels:
        return _revision(c, MechanismSupport.CHANNEL_NOT_SEEDED, f"No transmission is seeded for {label} yet.")
    entry = library.basis(c.channel, c.exposure_label)
    if entry is None:
        return _revision(c, MechanismSupport.NO_ECONOMIC_BASIS,
                         "No reviewed economic basis is recorded for this exposure.")
    own = [p for p in discovery.paths if p.anchor_channel is c.channel]
    basis = []
    reaching: list[SignalPath] = []
    for sid in entry.state_ids:
        paths = [p for p in own if p.consequence_state == sid]
        reaching.extend(paths)
        state_label = paths[0].consequence_label if paths else _STATE_LABELS.get(sid, sid)
        basis.append(BasisConsequence(
            state_id=sid, label=state_label, movement=_movement(paths), path_ids=tuple(p.path_id for p in paths),
        ))
    basis_t = tuple(basis)
    if not reaching:
        return _revision(
            c, MechanismSupport.NOT_REACHED,
            f"The {label} graph does not reach {', '.join(b.label for b in basis_t)} from this event (a missing link "
            "is a library gap, not evidence against).", basis=basis_t,
        )
    # Cite the best-supported route: researchable first, then unopposed, then
    # the most confident weakest link. Depth never enters; nothing here
    # touches the level.
    cited = min(reaching, key=lambda p: (
        p.status is not PathStatus.RESEARCHABLE, bool(p.opposing_path_ids), _CONFIDENCE_RANK[p.confidence],
    ))
    support = (
        MechanismSupport.CORROBORATED if cited.status is PathStatus.RESEARCHABLE else MechanismSupport.PROPOSED_ONLY
    )
    verb = "corroborates" if support is MechanismSupport.CORROBORATED else "proposes (not yet researchable)"
    return _revision(
        c, support,
        f"The mechanism {verb} a route: {' → '.join(cited.state_labels)} (weakest link "
        f"{cited.confidence.value.lower()} confidence).", basis=basis_t, cited=cited,
    )


def adjust_impact(
    scan: InitialImpactScan, discovery: SignalPathDiscovery, *, library: SignalPathLibrary = DEFAULT_PATH_LIBRARY,
) -> MechanismAdjustedImpact:
    """The mechanism-adjusted second pass for one scanned event. Pure; the
    scan is read, never modified."""
    if discovery.event_id != scan.event.event_id:
        raise ValueError("signal paths were discovered for a different event than the one scanned")
    assessments = []
    for a in scan.assessments:
        if a.impact_level is None:
            assessments.append(MechanismAdjustedImpactAssessment(
                event_id=a.event_id, asset_domain=a.asset_domain, initial_level=None, adjusted_level=None,
                change=LevelChange.NOT_ASSESSED,
            ))
            continue
        revisions = tuple(_revise(c, discovery, library) for c in a.contributions)
        # Levels follow magnitude, and no magnitude is established: unchanged.
        adjusted = a.impact_level
        change = LevelChange.UNCHANGED
        assessments.append(MechanismAdjustedImpactAssessment(
            event_id=a.event_id, asset_domain=a.asset_domain, initial_level=a.impact_level, adjusted_level=adjusted,
            change=change, revisions=revisions,
        ))
    return MechanismAdjustedImpact(
        event_id=scan.event.event_id, initial_scan_fingerprint=scan_fingerprint(scan),
        mechanism_graph_id=discovery.mechanism_graph_id, signal_path_discovery_id=discovery.fingerprint(),
        assessments=tuple(assessments),
    )
