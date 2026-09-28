"""News Alpha Phase B -- the ECONOMIC MECHANISM GRAPH stage.

    ResearchMandate -> Allowed Asset Universe -> News/Event
        -> Initial Impact Scan -> ECONOMIC MECHANISM GRAPH  (this module)
        -> Signal Path Discovery (Phase C, `news_alpha.signal_paths`)

``build_economic_mechanism_graph(scan)`` is a pure, deterministic function:

1. ANCHOR. Each channel the Initial Impact Scan detected becomes an anchor on
   its seeded root state (`transmission_library.ANCHOR_RULES`); the channel's
   direction shift sets the root's movement. No direction cue, or opposing
   cues, leaves the movement UNKNOWN. A channel without a seed is reported
   in `unseeded_channels`, never silently skipped.
2. EXPAND. The event's graph is the part of the claim pool (the seeded
   library plus any ``extra_claims``, e.g. a `TransmissionProposal`)
   reachable from the anchors within ``max_hops``, resolved by
   `transmission.assemble_graph` with all its safety findings.
3. PROPAGATE (``sign-propagation/1``). Along every simple path from an
   anchor, a state's implied movement is the anchor's movement times the
   product of the link signs. Paths that disagree make the state MIXED --
   both paths are kept with their slowest lag and weakest link, because the
   disagreement usually IS the research question (capex is an immediate cost
   to the spender and a slow, uncertain revenue source). Feedback loops are
   reported by the graph, never walked twice.

THE MANDATE NEVER SHAPES THIS GRAPH. The builder reads only ``scan.event``
and ``scan.channels`` (both mandate-independent in Phase A) -- never the
assessments, candidate markets or universe. Economic states are not
tradable assets: an Equity-only mandate still sees electricity demand,
copper demand and the policy-rate path (test-enforced).

HYPOTHESIS PLANE ONLY. No expected return, probability, trade instruction,
weight, score or verdict; no registry, network or LLM access. The graph says
how an event COULD transmit and how well each link is backed. Phase C's
Signal Path Discovery classifies these paths (direct / supply-chain /
cross-sector); mapping states onto tradable markets is Asset Expression, a
later stage.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from enum import Enum
from typing import Literal

from pydantic import BaseModel

from alpha_agent.news_alpha.channels import EconomicChannel
from alpha_agent.news_alpha.schemas import DetectionStrength, InitialImpactScan
from alpha_agent.news_alpha.transmission import (
    EconomicState,
    GraphIssue,
    GraphIssueKind,
    IssueSeverity,
    LinkConfidence,
    Movement,
    Polarity,
    ProvenanceKind,
    StateKind,
    SupportStatus,
    TransmissionClaim,
    TransmissionEdge,
    TransmissionGraph,
    TransmissionLag,
    assemble_graph,
    combine_polarity,
    reachable_from,
)
from alpha_agent.news_alpha.transmission_library import DEFAULT_LIBRARY, TransmissionLibrary

__all__ = [
    "DEFAULT_MAX_HOPS",
    "MECHANISM_GRAPH_SCHEMA_VERSION",
    "PROPAGATION_RULE",
    "EconomicMechanismGraph",
    "GraphAnchor",
    "ImpliedMovement",
    "StateImplication",
    "TransmissionPath",
    "UnseededChannel",
    "build_economic_mechanism_graph",
    "trace_path",
]

MECHANISM_GRAPH_SCHEMA_VERSION = "economic-mechanism-graph/1"
PROPAGATION_RULE = "sign-propagation/1"
DEFAULT_MAX_HOPS = 6
_MAX_PATHS = 2000
_LIBRARY_SOURCE = "SEEDED_TRANSMISSION_LIBRARY"


class ImpliedMovement(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    #: Different paths imply opposite movements.
    MIXED = "MIXED"
    #: Every path runs through a link whose sign is ambiguous or unknown.
    INDETERMINATE = "INDETERMINATE"
    #: The anchor's own direction is unknown -- only the sign RELATIVE to the
    #: anchor is known (see `StateImplication.relative_sign`).
    UNANCHORED = "UNANCHORED"


class GraphAnchor(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    state_id: str
    label: str
    movement: Movement
    channel: EconomicChannel
    channel_label: str
    shift: str | None = None
    detection_strength: DetectionStrength
    anchor_rule: str
    basis: str


class TransmissionPath(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    anchor_state: str
    #: Anchor first, this state last.
    state_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    relative_sign: Polarity
    movement: ImpliedMovement
    #: The chain moves no faster than its slowest link.
    slowest_lag: TransmissionLag
    #: The chain is no more certain than its weakest link.
    weakest_confidence: LinkConfidence
    weakest_support: SupportStatus


class StateImplication(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    state_id: str
    label: str
    kind: StateKind
    is_anchor: bool = False
    hops: int
    movement: ImpliedMovement
    #: Sign relative to the anchor(s), whether or not the anchor's direction
    #: is known. AMBIGUOUS when paths disagree or a link is unsigned.
    relative_sign: Polarity
    paths: tuple[TransmissionPath, ...] = ()
    note: str | None = None


class UnseededChannel(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    channel: EconomicChannel
    channel_label: str
    note: str


class EconomicMechanismGraph(BaseModel):
    """One event's economic mechanism graph. `plane` is fixed to HYPOTHESIS:
    this object is never evidence."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = MECHANISM_GRAPH_SCHEMA_VERSION
    plane: Literal["HYPOTHESIS"] = "HYPOTHESIS"
    propagation_rule: str = PROPAGATION_RULE
    event_id: str
    event_headline: str
    anchors: tuple[GraphAnchor, ...]
    graph: TransmissionGraph
    implications: tuple[StateImplication, ...]
    unseeded_channels: tuple[UnseededChannel, ...] = ()
    #: Findings from anchoring/expansion/propagation (the graph's own
    #: findings live on `graph.issues`).
    stage_issues: tuple[GraphIssue, ...] = ()
    max_hops: int
    generated_by: tuple[str, ...]
    not_evidence_note: str = (
        "Economic mechanism graph: how this event could transmit through the economy -- economic states, not "
        "tradable assets, and never limited by your mandate. A hypothesis map, not evidence: not an expected "
        "return, probability, forecast, trade instruction, portfolio weight, or scientific verdict."
    )

    @property
    def is_empty(self) -> bool:
        return not self.anchors

    @property
    def issues(self) -> tuple[GraphIssue, ...]:
        return (*self.stage_issues, *self.graph.issues)

    def implication(self, state_id: str) -> StateImplication:
        return next(i for i in self.implications if i.state_id == state_id)

    def fingerprint(self) -> str:
        return "mechgraph1:" + hashlib.sha256(self.model_dump_json().encode()).hexdigest()


# ---------------------------------------------------------------------------
# anchoring
# ---------------------------------------------------------------------------


def _anchors(
    scan: InitialImpactScan, library: TransmissionLibrary,
) -> tuple[list[GraphAnchor], list[UnseededChannel], list[GraphIssue]]:
    labels = {s.state_id: s.label for s in library.states}
    anchors: dict[str, GraphAnchor] = {}
    unseeded: list[UnseededChannel] = []
    issues: list[GraphIssue] = []
    for ev in scan.channels:
        rule = library.anchor_rule(ev.channel)
        if rule is None:
            unseeded.append(UnseededChannel(
                channel=ev.channel, channel_label=ev.channel_label,
                note=f"No transmission library is seeded for {ev.channel_label} yet -- its downstream economic "
                     "states are not mapped (a reviewed seed or a model proposal can add them).",
            ))
            continue
        if ev.conflicting_shift_cues:
            movement, why = Movement.UNKNOWN, "opposing direction cues in the text"
        elif ev.shift is None:
            movement, why = Movement.UNKNOWN, "no direction cue in the text"
        else:
            movement = dict(rule.shift_movement)[ev.shift]
            why = f"direction cue: {ev.shift.replace('_', ' ').lower()}"
        anchor = GraphAnchor(
            state_id=rule.root_state, label=labels[rule.root_state], movement=movement, channel=ev.channel,
            channel_label=ev.channel_label, shift=ev.shift, detection_strength=ev.strength, anchor_rule=rule.rule_id,
            basis=f"{rule.basis} Movement from {why}; channel detected {ev.strength.value.lower()}ly.",
        )
        existing = anchors.get(rule.root_state)
        if existing is None:
            anchors[rule.root_state] = anchor
        elif existing.movement is not movement:
            anchors[rule.root_state] = existing.model_copy(update={"movement": Movement.UNKNOWN})
            issues.append(GraphIssue(
                kind=GraphIssueKind.ANCHOR_CONFLICT, severity=IssueSeverity.UNRESOLVED,
                message=f"Two detected channels move {existing.label} in different directions; its movement is "
                        "left UNKNOWN.",
                state_ids=(rule.root_state,),
            ))
    return list(anchors.values()), unseeded, issues


# ---------------------------------------------------------------------------
# propagation
# ---------------------------------------------------------------------------


def _path_movement(anchor: Movement, sign: Polarity) -> ImpliedMovement:
    if anchor is Movement.UNKNOWN:
        return ImpliedMovement.UNANCHORED
    if not sign.signed:
        return ImpliedMovement.INDETERMINATE
    up = (anchor is Movement.UP) == (sign is Polarity.POSITIVE)
    return ImpliedMovement.UP if up else ImpliedMovement.DOWN


def trace_path(
    anchor: GraphAnchor, state_ids: Iterable[str], edges: Iterable[TransmissionEdge],
) -> TransmissionPath:
    """The typed `TransmissionPath` for one walk from ``anchor`` over
    ``edges`` -- the single construction shared by the enumeration below and
    by Signal Path Discovery's validated proposals."""
    chain = tuple(edges)
    sign = combine_polarity(x.polarity for x in chain)
    return TransmissionPath(
        anchor_state=anchor.state_id, state_ids=tuple(state_ids), edge_ids=tuple(x.edge_id for x in chain),
        relative_sign=sign, movement=_path_movement(anchor.movement, sign),
        slowest_lag=max((x.lag for x in chain), key=lambda lag: lag.rank),
        weakest_confidence=max((x.confidence for x in chain), key=lambda c: c.rank),
        weakest_support=max((x.support for x in chain), key=lambda s: s.rank),
    )


def _enumerate_paths(
    graph: TransmissionGraph, anchors: list[GraphAnchor], max_hops: int,
) -> tuple[dict[str, list[TransmissionPath]], bool]:
    """Every simple path (no state revisited) from each anchor, up to
    ``max_hops`` links, capped at `_MAX_PATHS`. Deterministic: links are
    walked in the graph's canonical edge order."""
    adjacency = {s.state_id: graph.edges_from(s.state_id) for s in graph.states}
    by_state: dict[str, list[TransmissionPath]] = {}
    n_paths = 0
    truncated = False

    def walk(anchor: GraphAnchor, states: list[str], edges: list) -> None:
        nonlocal n_paths, truncated
        for e in adjacency[states[-1]]:
            if e.target in states:
                continue
            if n_paths >= _MAX_PATHS:
                truncated = True
                return
            chain = [*edges, e]
            by_state.setdefault(e.target, []).append(trace_path(anchor, (*states, e.target), chain))
            n_paths += 1
            if len(chain) < max_hops:
                walk(anchor, [*states, e.target], chain)

    for anchor in anchors:
        walk(anchor, [anchor.state_id], [])
    return by_state, truncated


_LAG_TEXT = {
    TransmissionLag.IMMEDIATE: "immediately", TransmissionLag.WEEKS: "within weeks",
    TransmissionLag.MONTHS: "within months", TransmissionLag.QUARTERS: "over quarters",
    TransmissionLag.YEARS: "over years", TransmissionLag.UNKNOWN: "at an unknown lag",
}


def _aggregate(state: EconomicState, paths: list[TransmissionPath]) -> StateImplication:
    movements = {p.movement for p in paths}
    definite = movements & {ImpliedMovement.UP, ImpliedMovement.DOWN}
    note = None
    if len(definite) == 2:
        movement = ImpliedMovement.MIXED
        fastest = {
            m: min((p for p in paths if p.movement is m), key=lambda p: (p.slowest_lag.rank, len(p.edge_ids)))
            for m in (ImpliedMovement.UP, ImpliedMovement.DOWN)
        }
        note = "Paths disagree: " + " vs ".join(
            f"{m.value} {_LAG_TEXT[p.slowest_lag]} (weakest link {p.weakest_confidence.value.lower()} confidence)"
            for m, p in sorted(fastest.items(), key=lambda kv: kv[1].slowest_lag.rank)
        ) + "."
    elif definite:
        movement = definite.pop()
        if len(movements) > 1:
            note = "Another path runs through an unsigned link or an anchor of unknown direction."
    elif ImpliedMovement.INDETERMINATE in movements:
        movement = ImpliedMovement.INDETERMINATE
    else:
        movement = ImpliedMovement.UNANCHORED
    relative = {p.relative_sign for p in paths}
    relative_sign = relative.pop() if len(relative) == 1 else Polarity.AMBIGUOUS
    ordered = sorted(paths, key=lambda p: (len(p.edge_ids), p.slowest_lag.rank, p.edge_ids))
    return StateImplication(
        state_id=state.state_id, label=state.label, kind=state.kind, hops=len(ordered[0].edge_ids),
        movement=movement, relative_sign=relative_sign, paths=tuple(ordered), note=note,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def build_economic_mechanism_graph(
    scan: InitialImpactScan,
    *,
    extra_claims: Iterable[TransmissionClaim] = (),
    library: TransmissionLibrary = DEFAULT_LIBRARY,
    max_hops: int = DEFAULT_MAX_HOPS,
) -> EconomicMechanismGraph:
    """The economic mechanism graph for one scanned event. Reads only
    ``scan.event`` and ``scan.channels`` -- never the mandate-scoped
    assessments. ``extra_claims`` (e.g. `claims_from_proposal(...)`) join the
    pool with their own provenance; they never change a seeded claim."""
    if max_hops < 1:
        raise ValueError("max_hops must be at least 1")
    extra = tuple(extra_claims)
    anchors, unseeded, stage_issues = _anchors(scan, library)
    anchor_ids = [a.state_id for a in anchors]

    pool = assemble_graph((*library.claims, *extra), catalog=library.states)
    _reached, edge_ids, beyond = reachable_from(pool, anchor_ids, max_hops=max_hops)
    event_claims = [c for e in pool.edges if e.edge_id in edge_ids for c in e.claims]
    graph = assemble_graph(event_claims, catalog=library.states, include_states=anchor_ids)

    if beyond:
        stage_issues.append(GraphIssue(
            kind=GraphIssueKind.HOP_LIMIT_REACHED, severity=IssueSeverity.INFO,
            message=f"The graph stops {max_hops} links from its anchors; further links continue from: "
                    f"{', '.join(pool.state(s).label for s in beyond)}.",
            state_ids=beyond,
        ))
    if extra:
        shown = set(event_claims)
        kept = sum(c in shown for c in extra)
        if kept < len(extra):
            stage_issues.append(GraphIssue(
                kind=GraphIssueKind.DISCONNECTED_CLAIMS, severity=IssueSeverity.INFO,
                message=f"{len(extra) - kept} of {len(extra)} extra claim(s) do not connect to this event's anchors "
                        "and are not shown.",
            ))

    by_state, truncated = _enumerate_paths(graph, anchors, max_hops)
    if truncated:
        stage_issues.append(GraphIssue(
            kind=GraphIssueKind.PATH_SEARCH_TRUNCATED, severity=IssueSeverity.CAUTION,
            message=f"Path enumeration stopped at {_MAX_PATHS} paths; some implications may be incomplete.",
        ))

    implications: list[StateImplication] = []
    for a in anchors:
        implications.append(StateImplication(
            state_id=a.state_id, label=a.label, kind=graph.state(a.state_id).kind, is_anchor=True, hops=0,
            movement=ImpliedMovement(a.movement.value) if a.movement is not Movement.UNKNOWN
            else ImpliedMovement.UNANCHORED,
            relative_sign=Polarity.POSITIVE, note=a.basis,
        ))
    anchor_set = set(anchor_ids)
    order = {s.state_id: i for i, s in enumerate(graph.states)}
    rest = [
        _aggregate(graph.state(sid), paths) for sid, paths in by_state.items() if sid not in anchor_set
    ]
    implications.extend(sorted(rest, key=lambda i: (i.hops, order[i.state_id])))

    generated_by = [_LIBRARY_SOURCE]
    generated_by.extend(dict.fromkeys(
        f"MODEL_PROPOSAL:{p.reference}" for c in extra for p in c.provenance if p.kind is ProvenanceKind.MODEL_OUTPUT
    ))
    return EconomicMechanismGraph(
        event_id=scan.event.event_id, event_headline=scan.event.headline, anchors=tuple(anchors), graph=graph,
        implications=tuple(implications), unseeded_channels=tuple(unseeded), stage_issues=tuple(stage_issues),
        max_hops=max_hops, generated_by=tuple(generated_by),
    )
