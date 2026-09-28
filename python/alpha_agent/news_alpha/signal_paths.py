"""News Alpha Phase C -- SIGNAL PATH DISCOVERY.

    ResearchMandate -> Allowed Asset Universe -> News/Event
        -> Initial Impact Scan -> Economic Mechanism Graph
        -> SIGNAL PATH DISCOVERY (this module) -> Economic Consequence
        -> (Asset Expression: a later stage, not here)

A `SignalPath` is ONE specific, inspectable hypothesis about how an event
could reach a market-relevant economic consequence: an ordered walk through
the event's `EconomicMechanismGraph` from an anchor to one economic state,
with its links, direction, horizon, confidence, provenance and a computed
status. It ends at an ECONOMIC CONSEQUENCE (copper demand up) and a candidate
target CONCEPT (copper as an industrial metal) -- never at an instrument,
ticker, trade, weight or expected return (no such field exists; test-enforced).

EXTRACTION (``path-extraction/1``). Deterministic and reused, not re-walked:
the paths are exactly the simple paths the Phase B graph already enumerated
(`StateImplication.paths`, built by `mechanism_graph.trace_path`). A model
may PROPOSE paths (`SignalPathProposal`, the only shape an LLM's path output
may take); each is validated against the graph's typed states and links and
is either a duplicate of an extracted path, a new walk over existing links,
or REJECTED with typed reasons -- a model can never add a link, a state, or a
tradable conclusion through a path proposal (links arrive only as Phase B
`TransmissionProposal` claims, with their own provenance).

PATH TYPE (``path-type/1``) -- HOW the effect propagates, from reviewed data
(`signal_path_library`), never keywords:

* CROSS_SECTOR -- the effect lands in one economic sector (the sector of the
  path's first consequence) and a later state lies in a different sector
  (AI capex -> data-center construction -> electricity demand).
* SUPPLY_CHAIN -- otherwise, when the path takes at least
  `SUPPLY_CHAIN_MIN_STEPS` supplier/customer steps (`CHANNEL_ROLES`): the
  effect travels past the event's first counterparty to that counterparty's
  own suppliers or customers (accelerators -> HBM -> foundry -> equipment).
* DIRECT -- otherwise: the event's own first-order targets and its first
  supplier or customer (AI capex -> compute -> accelerator demand; a rate
  hike -> bank margins -> bank earnings).

A state missing from the reviewed sector table (e.g. a model-proposed state)
leaves the type ``None`` -- never guessed.

`path_type` is the PRIMARY label, resolved by precedence (CROSS_SECTOR, then
SUPPLY_CHAIN, then DIRECT). The two characteristics behind it are
INDEPENDENT and both recorded on every path, so the label never erases one:
`value_chain_steps` (supplier/customer steps) and `sector_transitions`
(sector changes after the first consequence). A CROSS_SECTOR path can carry
two or more value-chain steps (AI capex -> data centers -> electricity ->
gas for power), and a SUPPLY_CHAIN path always has zero transitions.

DEPTH IS A RESEARCH VARIABLE, NOT A SCORE. `transmission_depth` (links) and
`value_chain_steps` are recorded on every path; neither enters status,
ordering beyond reading order, or relevance. Whether information diffuses
differently across direct / supply-chain / cross-sector paths and across
horizons is an open research question this stage records the inputs for --
it does not answer it.

STATUS (``path-status/1``), computed from typed `StatusReason`s:

* UNRESOLVED -- runs through an unresolved or unsigned link, starts from a
  conflicted anchor, touches an unclassified state, or is OPPOSED at the same
  horizon (another path reaches the same consequence with the opposite
  direction at the same economic lag, so the net is undetermined). Only a
  SOUND opposing path counts -- one through no unresolved, unsigned or
  unreviewed-model link -- so a single disputed link cannot unresolve the
  independent paths beside it (it is still listed as opposition);
* PROPOSED -- well-formed but not yet researchable: no event direction (only
  the sign relative to the start), a low/unknown-confidence link, an unknown
  lag, or a link only an unreviewed model proposed;
* RESEARCHABLE -- none of the above;
* REJECTED -- only a model path proposal that failed validation
  (`RejectedPathProposal`, preserved with its reasons).

Opposing paths at DIFFERENT horizons (spender cash flow DOWN within months via
capex, UP over years via services revenue) are both kept, cross-referenced,
and not downgraded: a horizon separates them, which is itself researchable.

HYPOTHESIS PLANE ONLY -- no registry, network or LLM access.
"""
from __future__ import annotations

import hashlib
import itertools
import re
from collections import defaultdict
from collections.abc import Iterable
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from alpha_agent.news_alpha.channels import EconomicChannel
from alpha_agent.news_alpha.mechanism_graph import (
    EconomicMechanismGraph,
    GraphAnchor,
    ImpliedMovement,
    TransmissionPath,
    trace_path,
)
from alpha_agent.news_alpha.signal_path_library import (
    DEFAULT_PATH_LIBRARY,
    SIGNAL_PATH_LIBRARY_VERSION,
    EconomicSector,
    SignalPathLibrary,
    ValueChainRole,
)
from alpha_agent.news_alpha.transmission import (
    EdgeOrigin,
    GraphIssueKind,
    LinkConfidence,
    Movement,
    Polarity,
    ProvenanceKind,
    StateKind,
    SupportStatus,
    TransmissionChannel,
    TransmissionEdge,
    TransmissionLag,
    canonical_state_id,
)

__all__ = [
    "EXTRACTION_RULE",
    "PATH_STATUS_RULE",
    "PATH_TYPE_RULE",
    "SIGNAL_PATH_SCHEMA_VERSION",
    "STATUS_REASON_TEXT",
    "SUPPLY_CHAIN_MIN_STEPS",
    "PathConflict",
    "PathFinding",
    "PathFindingKind",
    "PathLink",
    "PathOrigin",
    "PathProvenance",
    "PathStatus",
    "PathType",
    "ProposedSignalPath",
    "RejectedPathProposal",
    "RejectionReason",
    "SignalPath",
    "SignalPathDiscovery",
    "SignalPathProposal",
    "StatusReason",
    "discover_signal_paths",
]

SIGNAL_PATH_SCHEMA_VERSION = "signal-path-discovery/1"
EXTRACTION_RULE = "path-extraction/1"
PATH_TYPE_RULE = "path-type/1"
PATH_STATUS_RULE = "path-status/1"
#: Supplier/customer steps at which a same-sector path becomes SUPPLY_CHAIN:
#: the first counterparty is a direct target; ITS counterparties are not.
SUPPLY_CHAIN_MIN_STEPS = 2
_MAX_PROPOSAL_REALIZATIONS = 8


# ---------------------------------------------------------------------------
# vocabularies
# ---------------------------------------------------------------------------


class PathType(str, Enum):
    DIRECT = "DIRECT"
    SUPPLY_CHAIN = "SUPPLY_CHAIN"
    CROSS_SECTOR = "CROSS_SECTOR"


class PathStatus(str, Enum):
    PROPOSED = "PROPOSED"
    RESEARCHABLE = "RESEARCHABLE"
    UNRESOLVED = "UNRESOLVED"
    #: Only a model path proposal that failed validation.
    REJECTED = "REJECTED"


class PathOrigin(str, Enum):
    #: Enumerated from the mechanism graph (``path-extraction/1``).
    GRAPH_EXTRACTED = "GRAPH_EXTRACTED"
    #: A model-proposed walk over existing graph links, validated here.
    MODEL_PROPOSED = "MODEL_PROPOSED"


class StatusReason(str, Enum):
    # -- make a path UNRESOLVED ------------------------------------------------
    UNRESOLVED_LINK = "UNRESOLVED_LINK"
    UNSIGNED_LINK = "UNSIGNED_LINK"
    ANCHOR_CONFLICT = "ANCHOR_CONFLICT"
    UNCLASSIFIED_STATE = "UNCLASSIFIED_STATE"
    OPPOSED_AT_SAME_HORIZON = "OPPOSED_AT_SAME_HORIZON"
    # -- keep a path PROPOSED ----------------------------------------------------
    NO_EVENT_DIRECTION = "NO_EVENT_DIRECTION"
    WEAK_LINK = "WEAK_LINK"
    UNKNOWN_LAG = "UNKNOWN_LAG"
    UNREVIEWED_MODEL_LINK = "UNREVIEWED_MODEL_LINK"


_UNRESOLVING = frozenset({
    StatusReason.UNRESOLVED_LINK, StatusReason.UNSIGNED_LINK, StatusReason.ANCHOR_CONFLICT,
    StatusReason.UNCLASSIFIED_STATE, StatusReason.OPPOSED_AT_SAME_HORIZON,
})

STATUS_REASON_TEXT: dict[StatusReason, str] = {
    StatusReason.UNRESOLVED_LINK: "runs through an unresolved link (opposite-sign claims or missing provenance)",
    StatusReason.UNSIGNED_LINK: "a link's sign is ambiguous or unknown",
    StatusReason.ANCHOR_CONFLICT: "the event moves its starting state in conflicting directions",
    StatusReason.UNCLASSIFIED_STATE: "a state is not in the reviewed sector table",
    StatusReason.OPPOSED_AT_SAME_HORIZON: "an opposing path reaches the same consequence at the same horizon",
    StatusReason.NO_EVENT_DIRECTION: "the event text gives no direction -- only the sign relative to the start",
    StatusReason.WEAK_LINK: "a link has low or unknown confidence",
    StatusReason.UNKNOWN_LAG: "a link's lag is unknown",
    StatusReason.UNREVIEWED_MODEL_LINK: "a link was proposed by a model and is neither reviewed nor sourced",
}


class RejectionReason(str, Enum):
    UNKNOWN_STATE = "UNKNOWN_STATE"
    NOT_FROM_AN_ANCHOR = "NOT_FROM_AN_ANCHOR"
    REPEATS_A_STATE = "REPEATS_A_STATE"
    NOT_A_GRAPH_LINK = "NOT_A_GRAPH_LINK"
    TRADE_EXPRESSION = "TRADE_EXPRESSION"


class PathFindingKind(str, Enum):
    DUPLICATE_PROPOSAL = "DUPLICATE_PROPOSAL"
    MODEL_PATH_ADDED = "MODEL_PATH_ADDED"
    PATH_SEARCH_TRUNCATED = "PATH_SEARCH_TRUNCATED"


# ---------------------------------------------------------------------------
# typed path
# ---------------------------------------------------------------------------


class PathLink(BaseModel):
    """One link of a path, copied from the graph edge so the path can be
    inspected (and serialized) on its own."""

    model_config = {"frozen": True, "extra": "forbid"}

    edge_id: str
    source: str
    target: str
    polarity: Polarity
    channels: tuple[TransmissionChannel, ...]
    lag: TransmissionLag
    confidence: LinkConfidence
    support: SupportStatus
    origins: tuple[EdgeOrigin, ...]
    #: SUPPLIER / CUSTOMER when the link moves the effect to a new
    #: counterparty along a value chain (`CHANNEL_ROLES`); ``None`` otherwise.
    value_chain_role: ValueChainRole | None = None
    #: False when no claim carries the provenance its origin requires.
    provenance_complete: bool
    #: False when the link's only claims are unverified model proposals.
    reviewed: bool


class PathProvenance(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    origin: PathOrigin
    rules: tuple[str, ...]
    #: Reviewed-seed rule ids behind the path's links.
    seed_rules: tuple[str, ...] = ()
    #: Titles of the verified sources that back any of its links.
    verified_sources: tuple[str, ...] = ()
    #: Model outputs behind the path: model-proposed links on it, and path
    #: proposals that re-proposed this walk (a proposal never upgrades it).
    model_proposals: tuple[str, ...] = ()


class SignalPath(BaseModel):
    """One hypothesis: event -> ordered economic transmission -> consequence.
    Never an asset, direction on a price, trade, weight or expected return."""

    model_config = {"frozen": True, "extra": "forbid"}

    #: Event-scoped id: the same route from two events has two path ids.
    path_id: str
    #: Structural id of the route (its ordered links), independent of the
    #: event -- equal signatures are the same transmission hypothesis.
    signature: str
    origin_event_id: str
    mechanism_graph_id: str
    anchor_state: str
    #: The Phase A channel whose anchor the path starts from.
    anchor_channel: EconomicChannel
    anchor_movement: Movement
    #: Anchor first, consequence last -- the order is the hypothesis.
    state_ids: tuple[str, ...]
    state_labels: tuple[str, ...]
    links: tuple[PathLink, ...]
    #: Reviewed sector of each state (``None`` = not in the reviewed table).
    sectors: tuple[EconomicSector | None, ...]
    #: Primary label by precedence -- see `value_chain_steps` and
    #: `sector_transitions` for the independent characteristics behind it.
    path_type: PathType | None
    classification_basis: str
    #: Supplier/customer steps along the whole path (any sector).
    value_chain_steps: int
    #: Sector changes after the first consequence; ``None`` when a state's
    #: sector is not in the reviewed table.
    sector_transitions: int | None
    transmission_depth: int
    consequence_state: str
    consequence_label: str
    consequence_kind: StateKind
    candidate_target_concept: str | None
    #: Implied movement of the CONSEQUENCE given the event -- an economic
    #: quantity's direction, never a price call on any instrument.
    expected_direction: ImpliedMovement
    #: Sign relative to the anchor, known even when the event's own direction
    #: is not.
    relative_sign: Polarity
    #: Economic transmission horizon: the path's slowest link. Not a
    #: market-reaction horizon -- prices can react long before.
    expected_horizon: TransmissionLag
    #: Weakest link confidence -- a chain is no surer than its weakest link.
    confidence: LinkConfidence
    weakest_support: SupportStatus
    status: PathStatus
    status_reasons: tuple[StatusReason, ...] = ()
    #: Paths reaching the same consequence with the opposite direction.
    opposing_path_ids: tuple[str, ...] = ()
    provenance: PathProvenance

    @model_validator(mode="after")
    def _is_one_ordered_walk(self) -> SignalPath:
        n = len(self.links)
        if n < 1 or len(self.state_ids) != n + 1 or self.transmission_depth != n:
            raise ValueError("a signal path is one or more links and depth equals its link count")
        if len(self.state_labels) != n + 1 or len(self.sectors) != n + 1:
            raise ValueError("labels and sectors must align with states")
        for i, link in enumerate(self.links):
            if (link.source, link.target) != (self.state_ids[i], self.state_ids[i + 1]):
                raise ValueError("links must chain the ordered states")
        if self.anchor_state != self.state_ids[0] or self.consequence_state != self.state_ids[-1]:
            raise ValueError("a path starts at its anchor and ends at its consequence")
        if len(set(self.state_ids)) != len(self.state_ids):
            raise ValueError("a signal path never revisits a state")
        reasons = set(self.status_reasons)
        expected = (
            PathStatus.UNRESOLVED if reasons & _UNRESOLVING
            else PathStatus.PROPOSED if reasons else PathStatus.RESEARCHABLE
        )
        if self.status is not expected:
            raise ValueError(f"status {self.status.value} does not follow from its reasons ({expected.value})")
        if self.path_type is not None and (
            (self.path_type is PathType.CROSS_SECTOR) != bool(self.sector_transitions)
        ):
            raise ValueError("CROSS_SECTOR exactly when the path has a sector transition")
        return self

    @property
    def via_labels(self) -> tuple[str, ...]:
        return self.state_labels[1:-1]


class PathConflict(BaseModel):
    """Paths that reach one consequence with opposite directions -- kept,
    never averaged."""

    model_config = {"frozen": True, "extra": "forbid"}

    consequence_state: str
    consequence_label: str
    path_ids: tuple[str, ...]
    directions: tuple[str, ...]
    horizons: tuple[TransmissionLag, ...]
    #: True unless two SOUND opposing paths share a horizon -- the
    #: directions can then be told apart by WHEN they act (or the opposition
    #: rests on a disputed/unreviewed link), so no sound path is unresolved.
    separable_by_horizon: bool
    note: str


class RejectedPathProposal(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    status: Literal[PathStatus.REJECTED] = PathStatus.REJECTED
    proposer: str
    states: tuple[str, ...]
    rationale: str
    reasons: tuple[RejectionReason, ...]
    detail: str


class PathFinding(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    kind: PathFindingKind
    message: str
    path_ids: tuple[str, ...] = ()


class SignalPathDiscovery(BaseModel):
    """Every signal path for one event's mechanism graph. `plane` is fixed to
    HYPOTHESIS: this object is never evidence."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = SIGNAL_PATH_SCHEMA_VERSION
    plane: Literal["HYPOTHESIS"] = "HYPOTHESIS"
    rules: tuple[str, ...] = (EXTRACTION_RULE, PATH_TYPE_RULE, PATH_STATUS_RULE, SIGNAL_PATH_LIBRARY_VERSION)
    event_id: str
    event_headline: str
    mechanism_graph_id: str
    paths: tuple[SignalPath, ...]
    conflicts: tuple[PathConflict, ...] = ()
    rejected: tuple[RejectedPathProposal, ...] = ()
    findings: tuple[PathFinding, ...] = ()
    #: Detected channels with no seeded transmission -- they have no paths
    #: yet, which is a library gap, never a finding about the event.
    unseeded_channels: tuple[EconomicChannel, ...] = ()
    generated_by: tuple[str, ...]
    not_trade_note: str = (
        "Signal paths: hypotheses about how this event could reach an economic consequence -- economic states and "
        "target concepts, not tradable assets. Not an expected return, probability, forecast, trade instruction, "
        "portfolio weight, or scientific verdict; choosing how to express a consequence in a market is a later "
        "stage."
    )

    @property
    def is_empty(self) -> bool:
        return not self.paths

    def path(self, path_id: str) -> SignalPath:
        return next(p for p in self.paths if p.path_id == path_id)

    def of_type(self, path_type: PathType | None) -> tuple[SignalPath, ...]:
        return tuple(p for p in self.paths if p.path_type is path_type)

    def to(self, state_id: str) -> tuple[SignalPath, ...]:
        return tuple(p for p in self.paths if p.consequence_state == state_id)

    def count_by_type(self) -> dict[PathType, int]:
        return {t: sum(p.path_type is t for p in self.paths) for t in PathType}

    def count_by_status(self) -> dict[PathStatus, int]:
        counts = {s: sum(p.status is s for p in self.paths) for s in PathStatus}
        counts[PathStatus.REJECTED] = len(self.rejected)
        return counts

    def fingerprint(self) -> str:
        return "sigpaths1:" + hashlib.sha256(self.model_dump_json().encode()).hexdigest()


# ---------------------------------------------------------------------------
# model proposals -- the ONLY shape an LLM's path output may take
# ---------------------------------------------------------------------------


class ProposedSignalPath(BaseModel):
    """A walk a model proposes, by state name. No type, status, direction,
    consequence or instrument field exists (``extra="forbid"``): all of those
    are computed, and the consequence is simply the last state."""

    model_config = {"frozen": True, "extra": "forbid"}

    states: tuple[str, ...] = Field(min_length=2, max_length=12)
    rationale: str = Field(min_length=3)


class SignalPathProposal(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "signal-path-proposal/1"
    proposer: str = Field(min_length=1)
    prompt_fingerprint: str = Field(min_length=1)
    paths: tuple[ProposedSignalPath, ...] = Field(default=(), max_length=40)

    @property
    def reference(self) -> str:
        return f"{self.proposer}#{self.prompt_fingerprint}"


#: Unambiguous trade-instruction phrasing. Belt and braces only: the real
#: guard is structural -- a proposal can only name graph states, and the
#: schema has no instrument, direction or trade field.
_TRADE_EXPRESSION = re.compile(
    r"\b(BUY|SELL|LONG|SHORT)\b"
    r"|(?i:\b(go long|go short|long position|short position|short selling|strong buy|buy rating|sell rating"
    r"|call options?|put options?|call spread|put spread|overweight|underweight|price target|stop loss"
    r"|take profit)\b)"
    r"|\$[A-Za-z]{1,5}\b",
)


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def _signature(edge_ids: Iterable[str]) -> str:
    return "sigpath1:" + hashlib.sha256("|".join(edge_ids).encode()).hexdigest()[:20]


def _path_id(event_id: str, signature: str) -> str:
    return "sp-" + hashlib.sha256(f"{event_id}|{signature}".encode()).hexdigest()[:16]


def _role(edge: TransmissionEdge, library: SignalPathLibrary) -> ValueChainRole | None:
    roles = {library.channel_roles[c] for c in edge.channels} - {None}
    if ValueChainRole.SUPPLIER in roles:
        return ValueChainRole.SUPPLIER
    return ValueChainRole.CUSTOMER if roles else None


def _link(edge: TransmissionEdge, library: SignalPathLibrary) -> PathLink:
    return PathLink(
        edge_id=edge.edge_id, source=edge.source, target=edge.target, polarity=edge.polarity,
        channels=edge.channels, lag=edge.lag, confidence=edge.confidence, support=edge.support, origins=edge.origins,
        value_chain_role=_role(edge, library),
        provenance_complete=any(c.has_required_provenance for c in edge.claims),
        reviewed=edge.support is SupportStatus.SUPPORTED or any(o is not EdgeOrigin.MODEL_PROPOSED for o in edge.origins),
    )


def _sector_text(sector: EconomicSector) -> str:
    return sector.value.replace("_", " ").lower()


def _classify(
    labels: tuple[str, ...], sectors: tuple[EconomicSector | None, ...], links: tuple[PathLink, ...],
) -> tuple[PathType | None, int, int | None, str]:
    """``path-type/1``: (type, value-chain steps, sector transitions, basis)."""
    steps = sum(link.value_chain_role is not None for link in links)
    unknown = [labels[i] for i, s in enumerate(sectors) if i > 0 and s is None]
    if unknown:
        return None, steps, None, f"Not classified: {', '.join(unknown)} is not in the reviewed sector table."
    first = sectors[1]
    transitions = sum(sectors[i] is not sectors[i - 1] for i in range(2, len(sectors)))
    crossing = next((i for i in range(2, len(sectors)) if sectors[i] is not first), None)
    if crossing is not None:
        chain = f" It also takes {steps} supplier/customer steps." if steps >= SUPPLY_CHAIN_MIN_STEPS else ""
        return PathType.CROSS_SECTOR, steps, transitions, (
            f"Lands in {_sector_text(first)} at {labels[1]}, then crosses into {_sector_text(sectors[crossing])} "
            f"at {labels[crossing]}.{chain}"
        )
    if steps >= SUPPLY_CHAIN_MIN_STEPS:
        hops = [f"{labels[i]} → {labels[i + 1]}" for i, link in enumerate(links) if link.value_chain_role is not None]
        return PathType.SUPPLY_CHAIN, steps, transitions, (
            f"{steps} supplier/customer steps within {_sector_text(first)} ({'; '.join(hops)}) -- past the event's "
            "first counterparty."
        )
    if steps:
        return PathType.DIRECT, steps, transitions, (
            f"The event's first supplier/customer within {_sector_text(first)} -- a first-order target."
        )
    return PathType.DIRECT, steps, transitions, f"A first-order consequence of the event within {_sector_text(first)}."


class _Draft:
    """A path before conflicts and duplicates are settled."""

    def __init__(self, tp: TransmissionPath, origin: PathOrigin, sound: bool) -> None:
        self.tp = tp
        self.origin = origin
        #: No unresolved, unsigned or unreviewed-model link -- only a sound
        #: path's opposition can unresolve another path.
        self.sound = sound
        self.signature = _signature(tp.edge_ids)
        self.proposers: list[str] = []
        self.opposing: list[_Draft] = []
        self.same_horizon_opposed = False
        self.path_id = ""


def _sound(tp: TransmissionPath, mg: EconomicMechanismGraph) -> bool:
    edges = [mg.graph.edge(e) for e in tp.edge_ids]
    return all(
        e.support is not SupportStatus.UNRESOLVED and e.polarity.signed
        and (e.support is SupportStatus.SUPPORTED or any(o is not EdgeOrigin.MODEL_PROPOSED for o in e.origins))
        for e in edges
    )


def _direction_key(d: _Draft) -> tuple[str, str] | None:
    """What an opposing path must disagree on: the implied movement when the
    event's direction is known, else the sign relative to the SAME anchor."""
    tp = d.tp
    if tp.movement in (ImpliedMovement.UP, ImpliedMovement.DOWN):
        return ("movement", tp.movement.value)
    if tp.movement is ImpliedMovement.UNANCHORED and tp.relative_sign.signed:
        return (f"relative:{tp.anchor_state}", tp.relative_sign.value)
    return None


def _settle_conflicts(drafts: list[_Draft], labels: dict[str, str]) -> list[PathConflict]:
    by_consequence: dict[str, list[_Draft]] = defaultdict(list)
    for d in drafts:
        by_consequence[d.tp.state_ids[-1]].append(d)
    conflicts: list[PathConflict] = []
    for consequence, group in by_consequence.items():
        involved: list[_Draft] = []
        contested = False
        for a, b in itertools.combinations(group, 2):
            ka, kb = _direction_key(a), _direction_key(b)
            if ka is None or kb is None or ka[0] != kb[0] or ka[1] == kb[1]:
                continue
            a.opposing.append(b)
            b.opposing.append(a)
            if a.tp.slowest_lag is b.tp.slowest_lag or TransmissionLag.UNKNOWN in (a.tp.slowest_lag, b.tp.slowest_lag):
                a.same_horizon_opposed |= b.sound
                b.same_horizon_opposed |= a.sound
                contested |= a.sound and b.sound
            involved.extend(x for x in (a, b) if x not in involved)
        if not involved:
            continue
        separable = not contested
        unsound = [d for d in involved if not d.sound]
        sides = defaultdict(list)
        for d in involved:
            sides[_direction_key(d)[1]].append(d)
        parts = [
            f"{side.lower()} over {'/'.join(dict.fromkeys(d.tp.slowest_lag.value.lower() for d in ds))}"
            for side, ds in sides.items()
        ]
        conflicts.append(PathConflict(
            consequence_state=consequence, consequence_label=labels[consequence],
            path_ids=tuple(d.path_id for d in involved), directions=tuple(_direction_key(d)[1] for d in involved),
            horizons=tuple(d.tp.slowest_lag for d in involved), separable_by_horizon=separable,
            note=(
                f"{labels[consequence]}: paths disagree ({' vs '.join(parts)}). "
                + ("At least two sound opposing paths act at the same horizon, so the net direction there is "
                   "unresolved." if not separable
                   else "The opposing paths act at different horizons, so each stays researchable at its own horizon."
                   if not unsound
                   else "At least one side runs through a disputed, unsigned or unreviewed link, so this opposition "
                        "does not by itself unresolve a sound path.")
            ),
        ))
    return conflicts


def _build(
    d: _Draft, mg: EconomicMechanismGraph, graph_id: str, anchors: dict[str, GraphAnchor],
    conflicted_anchors: set[str], library: SignalPathLibrary,
) -> SignalPath:
    graph = mg.graph
    tp = d.tp
    edges = [graph.edge(e) for e in tp.edge_ids]
    links = tuple(_link(e, library) for e in edges)
    labels = tuple(graph.state(s).label for s in tp.state_ids)
    profiles = [library.profile(s) for s in tp.state_ids]
    sectors = tuple(p.sector if p else None for p in profiles)
    path_type, steps, transitions, basis = _classify(labels, sectors, links)
    anchor = anchors[tp.anchor_state]

    reasons: list[StatusReason] = []
    if any(link.support is SupportStatus.UNRESOLVED for link in links):
        reasons.append(StatusReason.UNRESOLVED_LINK)
    if any(not link.polarity.signed for link in links):
        reasons.append(StatusReason.UNSIGNED_LINK)
    if tp.anchor_state in conflicted_anchors:
        reasons.append(StatusReason.ANCHOR_CONFLICT)
    if path_type is None:
        reasons.append(StatusReason.UNCLASSIFIED_STATE)
    if d.same_horizon_opposed:
        reasons.append(StatusReason.OPPOSED_AT_SAME_HORIZON)
    if anchor.movement is Movement.UNKNOWN and tp.anchor_state not in conflicted_anchors:
        reasons.append(StatusReason.NO_EVENT_DIRECTION)
    if tp.weakest_confidence in (LinkConfidence.LOW, LinkConfidence.UNKNOWN):
        reasons.append(StatusReason.WEAK_LINK)
    if tp.slowest_lag is TransmissionLag.UNKNOWN:
        reasons.append(StatusReason.UNKNOWN_LAG)
    if any(not link.reviewed for link in links):
        reasons.append(StatusReason.UNREVIEWED_MODEL_LINK)
    status = (
        PathStatus.UNRESOLVED if set(reasons) & _UNRESOLVING
        else PathStatus.PROPOSED if reasons else PathStatus.RESEARCHABLE
    )

    sources = [p for e in edges for c in e.claims for p in c.provenance]
    provenance = PathProvenance(
        origin=d.origin,
        rules=(EXTRACTION_RULE, PATH_TYPE_RULE, PATH_STATUS_RULE, mg.propagation_rule, SIGNAL_PATH_LIBRARY_VERSION),
        seed_rules=tuple(dict.fromkeys(p.reference for p in sources if p.kind is ProvenanceKind.PLATFORM_RULE)),
        verified_sources=tuple(dict.fromkeys(p.title for p in sources if p.counts_as_support)),
        model_proposals=tuple(dict.fromkeys(
            [p.reference for p in sources if p.kind is ProvenanceKind.MODEL_OUTPUT] + d.proposers
        )),
    )
    consequence = graph.state(tp.state_ids[-1])
    return SignalPath(
        path_id=d.path_id, signature=d.signature, origin_event_id=mg.event_id, mechanism_graph_id=graph_id,
        anchor_state=tp.anchor_state, anchor_channel=anchor.channel, anchor_movement=anchor.movement,
        state_ids=tp.state_ids, state_labels=labels,
        links=links, sectors=sectors, path_type=path_type, classification_basis=basis, value_chain_steps=steps,
        sector_transitions=transitions, transmission_depth=len(links), consequence_state=consequence.state_id, consequence_label=consequence.label,
        consequence_kind=consequence.kind,
        candidate_target_concept=profiles[-1].target_concept if profiles[-1] else None,
        expected_direction=tp.movement, relative_sign=tp.relative_sign, expected_horizon=tp.slowest_lag,
        confidence=tp.weakest_confidence, weakest_support=tp.weakest_support, status=status,
        status_reasons=tuple(reasons), opposing_path_ids=tuple(o.path_id for o in d.opposing), provenance=provenance,
    )


def _state_index(mg: EconomicMechanismGraph) -> dict[str, str]:
    index: dict[str, str] = {}
    for s in mg.graph.states:
        for ref in (s.state_id, s.label, *s.aliases):
            index.setdefault(canonical_state_id(ref), s.state_id)
    return index


def _validate_proposal(
    proposed: ProposedSignalPath, proposer: str, mg: EconomicMechanismGraph, index: dict[str, str],
) -> tuple[list[list[TransmissionEdge]], RejectedPathProposal | None]:
    """Edge sequences realizing a proposed walk, or its typed rejection."""
    reasons: list[RejectionReason] = []
    details: list[str] = []
    if any(_TRADE_EXPRESSION.search(t) for t in (proposed.rationale, *proposed.states)):
        reasons.append(RejectionReason.TRADE_EXPRESSION)
        details.append("it expresses a trade; a signal path ends at an economic consequence")
    resolved: list[str | None] = []
    for ref in proposed.states:
        try:
            resolved.append(index.get(canonical_state_id(ref)))
        except ValueError:
            resolved.append(None)
    unknown = [ref for ref, sid in zip(proposed.states, resolved, strict=True) if sid is None]
    if unknown:
        reasons.append(RejectionReason.UNKNOWN_STATE)
        details.append(f"not an economic state in this event's mechanism graph: {', '.join(unknown)}")
    ids = [sid for sid in resolved if sid is not None]
    anchors = {a.state_id for a in mg.anchors}
    if resolved[0] is not None and resolved[0] not in anchors:
        reasons.append(RejectionReason.NOT_FROM_AN_ANCHOR)
        details.append(f"a path must start at one of the event's anchors ({', '.join(sorted(anchors))})")
    if len(set(ids)) != len(ids):
        reasons.append(RejectionReason.REPEATS_A_STATE)
        details.append("a path never revisits a state")
    realizations: list[list[TransmissionEdge]] = []
    if not unknown:
        per_pair = []
        for a, b in itertools.pairwise(ids):
            options = [e for e in mg.graph.edges if (e.source, e.target) == (a, b)]
            if not options:
                per_pair = []
                reasons.append(RejectionReason.NOT_A_GRAPH_LINK)
                details.append(
                    f"{mg.graph.state(a).label} → {mg.graph.state(b).label} is not a link in the graph; propose the "
                    "link itself first, where it joins with its own provenance"
                )
                break
            per_pair.append(options)
        if per_pair:
            realizations = [list(r) for r in itertools.islice(itertools.product(*per_pair), _MAX_PROPOSAL_REALIZATIONS)]
    if reasons:
        return [], RejectedPathProposal(
            proposer=proposer, states=proposed.states, rationale=proposed.rationale,
            reasons=tuple(dict.fromkeys(reasons)), detail="Rejected: " + "; ".join(details) + ".",
        )
    return realizations, None


_TYPE_ORDER = {PathType.DIRECT: 0, PathType.SUPPLY_CHAIN: 1, PathType.CROSS_SECTOR: 2, None: 3}


def discover_signal_paths(
    mg: EconomicMechanismGraph,
    *,
    proposals: Iterable[SignalPathProposal] = (),
    library: SignalPathLibrary = DEFAULT_PATH_LIBRARY,
) -> SignalPathDiscovery:
    """Every signal path in one event's mechanism graph (``path-extraction/1``),
    classified and statused, plus any validated model path proposals. Pure and
    deterministic: reads only the graph (hence never the mandate)."""
    graph = mg.graph
    labels = {s.state_id: s.label for s in graph.states}
    anchors = {a.state_id: a for a in mg.anchors}
    conflicted_anchors = {
        sid for issue in mg.stage_issues if issue.kind is GraphIssueKind.ANCHOR_CONFLICT for sid in issue.state_ids
    }

    drafts: dict[str, _Draft] = {}
    for implication in mg.implications:
        for tp in implication.paths:
            d = _Draft(tp, PathOrigin.GRAPH_EXTRACTED, _sound(tp, mg))
            drafts.setdefault(d.signature, d)

    rejected: list[RejectedPathProposal] = []
    findings: list[PathFinding] = []
    index = _state_index(mg)
    proposals = tuple(proposals)
    for proposal in proposals:
        for proposed in proposal.paths:
            realizations, rejection = _validate_proposal(proposed, proposal.reference, mg, index)
            if rejection is not None:
                rejected.append(rejection)
                continue
            for edges in realizations:
                tp = trace_path(anchors[edges[0].source], [edges[0].source, *(e.target for e in edges)], edges)
                sig = _signature(tp.edge_ids)
                if sig in drafts:
                    drafts[sig].proposers.append(proposal.reference)
                    findings.append(PathFinding(
                        kind=PathFindingKind.DUPLICATE_PROPOSAL,
                        message=f"{proposal.proposer} proposed a path the graph already yields "
                                f"({' → '.join(labels[s] for s in tp.state_ids)}); recorded as provenance, not added.",
                        path_ids=(_path_id(mg.event_id, sig),),
                    ))
                else:
                    d = _Draft(tp, PathOrigin.MODEL_PROPOSED, _sound(tp, mg))
                    d.proposers.append(proposal.reference)
                    drafts[sig] = d
                    findings.append(PathFinding(
                        kind=PathFindingKind.MODEL_PATH_ADDED,
                        message=f"{proposal.proposer} proposed a walk over existing links beyond the enumerated "
                                f"paths ({' → '.join(labels[s] for s in tp.state_ids)}); validated and added.",
                        path_ids=(_path_id(mg.event_id, sig),),
                    ))

    for d in drafts.values():
        d.path_id = _path_id(mg.event_id, d.signature)
    ordered_drafts = list(drafts.values())
    conflicts = _settle_conflicts(ordered_drafts, labels)
    graph_id = mg.fingerprint()
    paths = [_build(d, mg, graph_id, anchors, conflicted_anchors, library) for d in ordered_drafts]
    state_order = {s.state_id: i for i, s in enumerate(graph.states)}
    paths.sort(key=lambda p: (
        _TYPE_ORDER[p.path_type], p.transmission_depth, state_order[p.consequence_state], p.signature,
    ))

    if any(i.kind is GraphIssueKind.PATH_SEARCH_TRUNCATED for i in mg.stage_issues):
        findings.append(PathFinding(
            kind=PathFindingKind.PATH_SEARCH_TRUNCATED,
            message="The mechanism graph's path enumeration hit its bound; some signal paths may be missing.",
        ))

    generated_by = ["MECHANISM_GRAPH_EXTRACTION"]
    generated_by.extend(dict.fromkeys(f"MODEL_PATH_PROPOSAL:{p.reference}" for p in proposals))
    return SignalPathDiscovery(
        event_id=mg.event_id, event_headline=mg.event_headline, mechanism_graph_id=graph_id,
        paths=tuple(paths), conflicts=tuple(conflicts), rejected=tuple(rejected), findings=tuple(findings),
        unseeded_channels=tuple(u.channel for u in mg.unseeded_channels), generated_by=tuple(generated_by),
    )
