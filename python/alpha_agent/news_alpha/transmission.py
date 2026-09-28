"""News Alpha Phase B -- the typed ECONOMIC TRANSMISSION graph (hypothesis plane).

TWO WORDS, TWO CONCEPTS. Everywhere else in this repository "mechanism" means
a QUANT mechanism: `alpha_agent.knowledge.models.EconomicMechanism` (TREND,
MOMENTUM, CARRY, TERM_STRUCTURE, ...), re-exported by `alpha_agent.core
.mechanism`, keyed by `alpha_agent.translation.mechanism_library`, and
navigated by the evidence-plane `alpha_agent.alpha_graph` (`MechanismGraphView`).
Those name a *price pattern a strategy family exploits*. This module is about
ECONOMIC TRANSMISSION -- how a change in one economic state moves another
(AI capex up -> compute demand up -> accelerator demand up -> HBM demand up).
It never imports or extends `EconomicMechanism`, and `alpha_graph` never
imports it: the hypothesis plane and the evidence plane may be linked later,
never merged.

WHAT IS HERE -- immutable typed structures plus pure functions; no graph
database, no persistence, no network, no LLM:

* `EconomicState` -- a node. An economic quantity (electricity demand, copper
  demand, the policy-rate path, capacity utilization), NEVER a tradable
  instrument: it has no symbol, root, ticker or asset-domain field, so the
  user's allowed asset universe cannot constrain the graph by construction.
* `TransmissionClaim` -- one asserted relationship as supplied (by a reviewed
  seed, a model proposal, an external source or an observation), carrying its
  own `ProvenanceSource`s. A claim has NO status field: nobody asserts their
  own relationship into established fact.
* `TransmissionEdge` -- the graph-resolved relationship: duplicate claims
  merged (every claim preserved), and a `SupportStatus` COMPUTED by
  ``support-status/1``:

  - UNRESOLVED -- an opposite-sign link exists on the same (source, target)
    pair, or no claim carries the provenance its origin requires;
  - SUPPORTED  -- at least one claim cites a source whose statement was
    VERIFIED against the source itself (an external document, observed data,
    or a definitional identity);
  - PROPOSED   -- asserted (reasoned seed or model proposal), not yet backed.

  A model output can never be VERIFIED (validator-enforced) and
  `claims_from_proposal` records every model-cited reference as UNVERIFIED,
  so a model proposal is PROPOSED until a deterministic or human step
  verifies a source -- it never promotes itself.

* `assemble_graph` -- deterministic normalization with GRAPH SAFETY that
  exposes, never smooths over: alias/case-insensitive state resolution,
  undeclared states, self-loops (rejected, preserved), duplicate claims
  (merged, counted), lag disagreement, opposite-sign conflicts (both links
  kept, both UNRESOLVED), missing provenance, unverified citations, unsigned
  links, unknown lag/confidence, and feedback loops (every simple cycle,
  classified REINFORCING / BALANCING / INDETERMINATE).
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable
from datetime import date
from enum import Enum

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "MAX_LOOP_LENGTH",
    "SUPPORT_RULE",
    "TRANSMISSION_SCHEMA_VERSION",
    "EconomicState",
    "EdgeOrigin",
    "FeedbackLoop",
    "GraphIssue",
    "GraphIssueKind",
    "IssueSeverity",
    "LinkConfidence",
    "LoopType",
    "Movement",
    "Polarity",
    "ProposedCitation",
    "ProposedLink",
    "ProvenanceKind",
    "ProvenanceSource",
    "StateKind",
    "SupportStatus",
    "TransmissionChannel",
    "TransmissionClaim",
    "TransmissionEdge",
    "TransmissionGraph",
    "TransmissionLag",
    "TransmissionProposal",
    "Verification",
    "assemble_graph",
    "canonical_state_id",
    "claims_from_proposal",
    "combine_polarity",
    "reachable_from",
]

TRANSMISSION_SCHEMA_VERSION = "transmission-graph/1"
SUPPORT_RULE = "support-status/1"
#: Longest feedback loop enumerated; `LOOP_SEARCH_TRUNCATED` is raised when
#: the bound (or `_MAX_LOOP_SEQUENCES`) cuts the search short.
MAX_LOOP_LENGTH = 8
_MAX_LOOP_SEQUENCES = 256

_STATE_ID = r"^[a-z][a-z0-9_]{0,79}$"


def canonical_state_id(ref: str) -> str:
    """The one normalization every state reference goes through:
    ``"HBM demand"``, ``"hbm-demand"`` and ``" HBM  Demand "`` all become
    ``"hbm_demand"``."""
    slug = re.sub(r"[^a-z0-9]+", "_", ref.lower()).strip("_")
    if not slug:
        raise ValueError(f"state reference {ref!r} has no alphanumeric content")
    return slug if slug[0].isalpha() else f"s_{slug}"


# ---------------------------------------------------------------------------
# closed vocabularies
# ---------------------------------------------------------------------------


class StateKind(str, Enum):
    INVESTMENT = "INVESTMENT"
    DEMAND = "DEMAND"
    SUPPLY = "SUPPLY"
    CAPACITY = "CAPACITY"
    PRICE = "PRICE"
    RATE = "RATE"
    FINANCIAL = "FINANCIAL"
    ACTIVITY = "ACTIVITY"
    #: Referenced by a claim but absent from the state catalog.
    UNSPECIFIED = "UNSPECIFIED"


class Polarity(str, Enum):
    #: The target moves WITH the source.
    POSITIVE = "POSITIVE"
    #: The target moves AGAINST the source.
    NEGATIVE = "NEGATIVE"
    #: The sign depends on conditions the claim's rationale names.
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"

    @property
    def signed(self) -> bool:
        return self in (Polarity.POSITIVE, Polarity.NEGATIVE)


class Movement(str, Enum):
    """Which way a state moves. UNKNOWN when the event text establishes no
    direction (or establishes contradictory ones)."""

    UP = "UP"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"


def combine_polarity(signs: Iterable[Polarity]) -> Polarity:
    """Sign of a chain: the product of its links. Any unsigned link makes the
    chain unsigned (UNKNOWN dominates AMBIGUOUS)."""
    signs = list(signs)
    if Polarity.UNKNOWN in signs:
        return Polarity.UNKNOWN
    if Polarity.AMBIGUOUS in signs:
        return Polarity.AMBIGUOUS
    return Polarity.NEGATIVE if sum(s is Polarity.NEGATIVE for s in signs) % 2 else Polarity.POSITIVE


class TransmissionChannel(str, Enum):
    """HOW the source moves the target -- an economic route, never a quant
    mechanism (see module docstring)."""

    DEMAND_PULL = "DEMAND_PULL"
    INPUT_DEMAND = "INPUT_DEMAND"
    CAPACITY_UTILIZATION = "CAPACITY_UTILIZATION"
    INVESTMENT_RESPONSE = "INVESTMENT_RESPONSE"
    SUPPLY_RESPONSE = "SUPPLY_RESPONSE"
    SUPPLY_BALANCE = "SUPPLY_BALANCE"
    COST_PASS_THROUGH = "COST_PASS_THROUGH"
    COST_BURDEN = "COST_BURDEN"
    REVENUE = "REVENUE"
    POLICY_TRANSMISSION = "POLICY_TRANSMISSION"
    POLICY_REACTION = "POLICY_REACTION"
    DISCOUNT_RATE = "DISCOUNT_RATE"
    OPPORTUNITY_COST = "OPPORTUNITY_COST"
    FINANCING_CONDITIONS = "FINANCING_CONDITIONS"
    OTHER = "OTHER"


class TransmissionLag(str, Enum):
    """How long the TARGET takes to move once the source has -- an economic
    transmission lag, distinct from Phase A's market-reaction
    `ImpactHorizon` (prices can react at once to demand that arrives years
    later)."""

    IMMEDIATE = "IMMEDIATE"
    WEEKS = "WEEKS"
    MONTHS = "MONTHS"
    QUARTERS = "QUARTERS"
    YEARS = "YEARS"
    UNKNOWN = "UNKNOWN"

    @property
    def rank(self) -> int:
        """Slowness; UNKNOWN ranks slowest so a chain with an unknown link is
        never reported as faster than it is known to be."""
        return list(TransmissionLag).index(self)


class LinkConfidence(str, Enum):
    """Confidence that the RELATIONSHIP and its sign hold -- categorical,
    never a probability, and never confidence in any market outcome."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"

    @property
    def rank(self) -> int:
        """Weakness (HIGH = 0); UNKNOWN ranks weakest."""
        return list(LinkConfidence).index(self)


class EdgeOrigin(str, Enum):
    """Who introduced the relationship -- independent of whether anything
    supports it."""

    #: Measured in data by a deterministic platform process.
    OBSERVED = "OBSERVED"
    #: Ingested from an external source (a dataset, filing, report).
    EXTERNALLY_SOURCED = "EXTERNALLY_SOURCED"
    #: Proposed by an LLM.
    MODEL_PROPOSED = "MODEL_PROPOSED"
    #: A hand-reviewed platform seed (`news_alpha.transmission_library`).
    MANUALLY_SEEDED = "MANUALLY_SEEDED"


class SupportStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PROPOSED = "PROPOSED"
    UNRESOLVED = "UNRESOLVED"

    @property
    def rank(self) -> int:
        """Weakness (SUPPORTED = 0)."""
        return list(SupportStatus).index(self)


class ProvenanceKind(str, Enum):
    EXTERNAL_DOCUMENT = "EXTERNAL_DOCUMENT"
    OBSERVED_DATA = "OBSERVED_DATA"
    #: An accounting or definitional identity (e.g. FCF = CFO - capex).
    DEFINITIONAL_IDENTITY = "DEFINITIONAL_IDENTITY"
    #: A reviewed platform rule id -- a reasoned judgment, not evidence.
    PLATFORM_RULE = "PLATFORM_RULE"
    #: The model output a proposal came from -- never evidence.
    MODEL_OUTPUT = "MODEL_OUTPUT"


class Verification(str, Enum):
    #: The recorded ``claim`` was checked against the source itself.
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"


_SUPPORTING_KINDS = frozenset(
    {ProvenanceKind.EXTERNAL_DOCUMENT, ProvenanceKind.OBSERVED_DATA, ProvenanceKind.DEFINITIONAL_IDENTITY}
)
#: The provenance each origin must carry for its claim to be well-formed.
_REQUIRED_KIND = {
    EdgeOrigin.MANUALLY_SEEDED: ProvenanceKind.PLATFORM_RULE,
    EdgeOrigin.MODEL_PROPOSED: ProvenanceKind.MODEL_OUTPUT,
    EdgeOrigin.EXTERNALLY_SOURCED: ProvenanceKind.EXTERNAL_DOCUMENT,
    EdgeOrigin.OBSERVED: ProvenanceKind.OBSERVED_DATA,
}


# ---------------------------------------------------------------------------
# nodes, claims, provenance
# ---------------------------------------------------------------------------


class ProvenanceSource(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    kind: ProvenanceKind
    #: Rule id, URL, dataset id, or model id -- whatever locates the source.
    reference: str = Field(min_length=1)
    title: str = Field(min_length=1)
    publisher: str | None = None
    published: date | None = None
    #: What the source actually states, recorded verbatim or near-verbatim.
    claim: str | None = None
    verification: Verification = Verification.UNVERIFIED
    verified_on: date | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _verification_is_earned(self) -> ProvenanceSource:
        if self.verification is Verification.VERIFIED:
            if self.kind not in _SUPPORTING_KINDS:
                raise ValueError(f"a {self.kind.value} source is never verified evidence")
            if self.verified_on is None or not self.claim:
                raise ValueError("a VERIFIED source must record the verified claim and verified_on")
        elif self.verified_on is not None:
            raise ValueError("verified_on is set but the source is not VERIFIED")
        return self

    @property
    def counts_as_support(self) -> bool:
        return self.verification is Verification.VERIFIED and self.kind in _SUPPORTING_KINDS


class EconomicState(BaseModel):
    """An economic quantity -- never a tradable instrument. Deliberately no
    symbol / root / ticker / asset-domain field (test-enforced)."""

    model_config = {"frozen": True, "extra": "forbid"}

    state_id: str = Field(pattern=_STATE_ID)
    label: str = Field(min_length=1)
    kind: StateKind
    description: str = ""
    aliases: tuple[str, ...] = ()
    #: False for a state a claim referenced that the catalog does not declare.
    declared: bool = True


class TransmissionClaim(BaseModel):
    """One asserted relationship, exactly as supplied. No status field."""

    model_config = {"frozen": True, "extra": "forbid"}

    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    channel: TransmissionChannel
    polarity: Polarity
    lag: TransmissionLag
    confidence: LinkConfidence
    origin: EdgeOrigin
    rationale: str = Field(min_length=3)
    provenance: tuple[ProvenanceSource, ...] = ()

    @property
    def has_required_provenance(self) -> bool:
        return any(p.kind is _REQUIRED_KIND[self.origin] for p in self.provenance)

    @property
    def is_supported(self) -> bool:
        return self.has_required_provenance and any(p.counts_as_support for p in self.provenance)


class TransmissionEdge(BaseModel):
    """A graph-resolved relationship: every duplicate claim merged in and
    preserved; `support` computed by ``support-status/1``, never supplied."""

    model_config = {"frozen": True, "extra": "forbid"}

    edge_id: str
    source: str
    target: str
    polarity: Polarity
    channels: tuple[TransmissionChannel, ...]
    lag: TransmissionLag
    confidence: LinkConfidence
    origins: tuple[EdgeOrigin, ...]
    support: SupportStatus
    support_reason: str
    rationale: str
    claims: tuple[TransmissionClaim, ...]

    @property
    def sources(self) -> tuple[ProvenanceSource, ...]:
        seen: dict[tuple[str, str], ProvenanceSource] = {}
        for c in self.claims:
            for p in c.provenance:
                seen.setdefault((p.kind.value, p.reference), p)
        return tuple(seen.values())


# ---------------------------------------------------------------------------
# graph safety findings
# ---------------------------------------------------------------------------


class IssueSeverity(str, Enum):
    INFO = "INFO"
    CAUTION = "CAUTION"
    UNRESOLVED = "UNRESOLVED"


class GraphIssueKind(str, Enum):
    DUPLICATE_MERGED = "DUPLICATE_MERGED"
    DIRECTION_CONFLICT = "DIRECTION_CONFLICT"
    MISSING_PROVENANCE = "MISSING_PROVENANCE"
    UNVERIFIED_CITATION = "UNVERIFIED_CITATION"
    LAG_DISAGREEMENT = "LAG_DISAGREEMENT"
    UNSIGNED_LINK = "UNSIGNED_LINK"
    UNKNOWN_LAG = "UNKNOWN_LAG"
    UNKNOWN_CONFIDENCE = "UNKNOWN_CONFIDENCE"
    UNDECLARED_STATE = "UNDECLARED_STATE"
    STATE_DEFINITION_CONFLICT = "STATE_DEFINITION_CONFLICT"
    SELF_LOOP_REJECTED = "SELF_LOOP_REJECTED"
    FEEDBACK_LOOP = "FEEDBACK_LOOP"
    LOOP_SEARCH_TRUNCATED = "LOOP_SEARCH_TRUNCATED"
    # raised by the event-anchored stage (`news_alpha.mechanism_graph`)
    HOP_LIMIT_REACHED = "HOP_LIMIT_REACHED"
    DISCONNECTED_CLAIMS = "DISCONNECTED_CLAIMS"
    ANCHOR_CONFLICT = "ANCHOR_CONFLICT"
    PATH_SEARCH_TRUNCATED = "PATH_SEARCH_TRUNCATED"


class GraphIssue(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    kind: GraphIssueKind
    severity: IssueSeverity
    message: str
    state_ids: tuple[str, ...] = ()
    edge_ids: tuple[str, ...] = ()


class LoopType(str, Enum):
    #: An even number of negative links: a move feeds back on itself.
    REINFORCING = "REINFORCING"
    #: An odd number of negative links: the loop eventually counteracts it.
    BALANCING = "BALANCING"
    #: An unsigned link, or parallel links of opposite sign, on the loop.
    INDETERMINATE = "INDETERMINATE"


class FeedbackLoop(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    #: Canonical rotation: starts at the loop's earliest-declared state.
    state_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    loop_type: LoopType


class TransmissionGraph(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = TRANSMISSION_SCHEMA_VERSION
    support_rule: str = SUPPORT_RULE
    states: tuple[EconomicState, ...]
    edges: tuple[TransmissionEdge, ...]
    loops: tuple[FeedbackLoop, ...] = ()
    issues: tuple[GraphIssue, ...] = ()
    #: Claims refused outright (self-loops), preserved rather than dropped.
    rejected_claims: tuple[TransmissionClaim, ...] = ()

    def state(self, state_id: str) -> EconomicState:
        return next(s for s in self.states if s.state_id == state_id)

    def has_state(self, state_id: str) -> bool:
        return any(s.state_id == state_id for s in self.states)

    def edge(self, edge_id: str) -> TransmissionEdge:
        return next(e for e in self.edges if e.edge_id == edge_id)

    def edges_from(self, state_id: str) -> tuple[TransmissionEdge, ...]:
        return tuple(e for e in self.edges if e.source == state_id)

    def issues_of(self, kind: GraphIssueKind) -> tuple[GraphIssue, ...]:
        return tuple(i for i in self.issues if i.kind is kind)

    def support_counts(self) -> dict[SupportStatus, int]:
        return {s: sum(e.support is s for e in self.edges) for s in SupportStatus}

    def fingerprint(self) -> str:
        return "txgraph1:" + hashlib.sha256(self.model_dump_json().encode()).hexdigest()


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def _edge_id(source: str, target: str, polarity: Polarity) -> str:
    return "tx-" + hashlib.sha256(f"{source}>{target}:{polarity.value}".encode()).hexdigest()[:12]


class _Resolver:
    """Alias-, case- and punctuation-insensitive state resolution against a
    catalog. First declaration of an id wins; a conflicting redeclaration is
    reported, never silently merged."""

    def __init__(self, catalog: Iterable[EconomicState]) -> None:
        self.states: dict[str, EconomicState] = {}
        self.index: dict[str, str] = {}
        self.issues: list[GraphIssue] = []
        for state in catalog:
            existing = self.states.get(state.state_id)
            if existing is not None:
                if existing != state:
                    self.issues.append(GraphIssue(
                        kind=GraphIssueKind.STATE_DEFINITION_CONFLICT, severity=IssueSeverity.CAUTION,
                        message=f"State '{state.state_id}' is declared twice with different definitions; the first "
                                "declaration is used.",
                        state_ids=(state.state_id,),
                    ))
                continue
            self.states[state.state_id] = state
            for ref in (state.state_id, state.label, *state.aliases):
                key = canonical_state_id(ref)
                owner = self.index.setdefault(key, state.state_id)
                if owner != state.state_id:
                    self.issues.append(GraphIssue(
                        kind=GraphIssueKind.STATE_DEFINITION_CONFLICT, severity=IssueSeverity.CAUTION,
                        message=f"'{ref}' names both '{owner}' and '{state.state_id}'; it resolves to '{owner}'.",
                        state_ids=(owner, state.state_id),
                    ))
        self.undeclared: dict[str, EconomicState] = {}

    def resolve(self, ref: str) -> str:
        key = canonical_state_id(ref)
        if key in self.index:
            return self.index[key]
        if key not in self.undeclared:
            self.undeclared[key] = EconomicState(
                state_id=key, label=" ".join(ref.split()), kind=StateKind.UNSPECIFIED, declared=False,
            )
        return key

    def get(self, state_id: str) -> EconomicState:
        return self.states.get(state_id) or self.undeclared[state_id]


def _merge_group(key: tuple[str, str, Polarity], claims: list[TransmissionClaim]) -> tuple[dict, list[GraphIssue]]:
    source, target, polarity = key
    edge_id = _edge_id(source, target, polarity)
    issues: list[GraphIssue] = []
    known_lags = list(dict.fromkeys(c.lag for c in claims if c.lag is not TransmissionLag.UNKNOWN))
    if len(known_lags) > 1:
        issues.append(GraphIssue(
            kind=GraphIssueKind.LAG_DISAGREEMENT, severity=IssueSeverity.CAUTION,
            message=f"Claims for {source} -> {target} disagree on the lag "
                    f"({', '.join(lag.value.lower() for lag in known_lags)}); the link's lag is left UNKNOWN.",
            state_ids=(source, target), edge_ids=(edge_id,),
        ))
    lag = known_lags[0] if len(known_lags) == 1 else TransmissionLag.UNKNOWN
    known_conf = [c.confidence for c in claims if c.confidence is not LinkConfidence.UNKNOWN]
    confidence = max(known_conf, key=lambda c: c.rank) if known_conf else LinkConfidence.UNKNOWN
    if len(claims) > 1:
        issues.append(GraphIssue(
            kind=GraphIssueKind.DUPLICATE_MERGED, severity=IssueSeverity.INFO,
            message=f"{len(claims)} claims assert {source} -> {target} ({polarity.value.lower()}); merged into one link, "
                    "every claim and source preserved; confidence is the most conservative stated.",
            state_ids=(source, target), edge_ids=(edge_id,),
        ))
    fields = {
        "edge_id": edge_id, "source": source, "target": target, "polarity": polarity,
        "channels": tuple(dict.fromkeys(c.channel for c in claims)), "lag": lag, "confidence": confidence,
        "origins": tuple(dict.fromkeys(c.origin for c in claims)), "rationale": claims[0].rationale,
        "claims": tuple(claims),
    }
    return fields, issues


def _support(fields: dict, conflicted: bool) -> tuple[SupportStatus, str]:
    claims: tuple[TransmissionClaim, ...] = fields["claims"]
    if conflicted:
        return SupportStatus.UNRESOLVED, "An opposite-sign link is asserted on the same pair; the net sign is unresolved."
    if not any(c.has_required_provenance for c in claims):
        return SupportStatus.UNRESOLVED, "No claim carries the provenance its origin requires."
    supporting = [p for c in claims if c.has_required_provenance for p in c.provenance if p.counts_as_support]
    if supporting:
        return SupportStatus.SUPPORTED, f"Backed by a verified source: {supporting[0].title}."
    if all(c.origin is EdgeOrigin.MODEL_PROPOSED for c in claims):
        return SupportStatus.PROPOSED, "Model-proposed; no source has been verified."
    return SupportStatus.PROPOSED, "Reasoned, not yet backed by a verified source."


def _find_loops(
    order: list[str], edges: list[TransmissionEdge],
) -> tuple[tuple[FeedbackLoop, ...], bool]:
    """Every simple cycle up to `MAX_LOOP_LENGTH`, found exactly once from
    its earliest-ordered state. Parallel edges on one cycle are grouped; a
    group whose edge sequences disagree in type is INDETERMINATE."""
    index = {sid: i for i, sid in enumerate(order)}
    adjacency: dict[str, list[TransmissionEdge]] = defaultdict(list)
    for e in sorted(edges, key=lambda e: (index[e.target], e.edge_id)):
        adjacency[e.source].append(e)
    found: dict[tuple[str, ...], list[list[TransmissionEdge]]] = {}
    truncated = False
    n_sequences = 0

    def dfs(start: str, node: str, path: list[str], path_edges: list[TransmissionEdge]) -> None:
        nonlocal truncated, n_sequences
        for e in adjacency[node]:
            if n_sequences >= _MAX_LOOP_SEQUENCES:
                truncated = True
                return
            if e.target == start:
                found.setdefault(tuple(path), []).append([*path_edges, e])
                n_sequences += 1
            elif index[e.target] > index[start] and e.target not in path:
                if len(path) >= MAX_LOOP_LENGTH:
                    truncated = True
                    continue
                dfs(start, e.target, [*path, e.target], [*path_edges, e])

    for start in order:
        dfs(start, start, [start], [])

    loops = []
    for states, sequences in found.items():
        types = set()
        for seq in sequences:
            sign = combine_polarity(e.polarity for e in seq)
            types.add(
                LoopType.INDETERMINATE if not sign.signed
                else LoopType.BALANCING if sign is Polarity.NEGATIVE else LoopType.REINFORCING
            )
        loops.append(FeedbackLoop(
            state_ids=states,
            edge_ids=tuple(dict.fromkeys(e.edge_id for seq in sequences for e in seq)),
            loop_type=types.pop() if len(types) == 1 else LoopType.INDETERMINATE,
        ))
    return tuple(loops), truncated


def assemble_graph(
    claims: Iterable[TransmissionClaim],
    *,
    catalog: Iterable[EconomicState] = (),
    include_states: Iterable[str] = (),
) -> TransmissionGraph:
    """Deterministically resolve claims into a `TransmissionGraph`.

    ``catalog`` is the state vocabulary used for resolution (only states a
    link or ``include_states`` references appear in the graph). Claims are
    merged on (source, target, polarity); nothing is dropped silently -- see
    the module docstring for every safety finding raised."""
    resolver = _Resolver(catalog)
    issues: list[GraphIssue] = list(resolver.issues)
    groups: dict[tuple[str, str, Polarity], list[TransmissionClaim]] = {}
    rejected: list[TransmissionClaim] = []
    for claim in claims:
        source, target = resolver.resolve(claim.source), resolver.resolve(claim.target)
        if source == target:
            rejected.append(claim)
            issues.append(GraphIssue(
                kind=GraphIssueKind.SELF_LOOP_REJECTED, severity=IssueSeverity.CAUTION,
                message=f"A claim links '{source}' to itself; rejected and preserved.", state_ids=(source,),
            ))
            continue
        groups.setdefault((source, target, claim.polarity), []).append(claim)

    merged: list[dict] = []
    for key, group in groups.items():
        fields, group_issues = _merge_group(key, group)
        merged.append(fields)
        issues.extend(group_issues)

    signs_by_pair: dict[tuple[str, str], set[Polarity]] = defaultdict(set)
    for f in merged:
        signs_by_pair[(f["source"], f["target"])].add(f["polarity"])
    conflicted_pairs = {pair for pair, signs in signs_by_pair.items() if {Polarity.POSITIVE, Polarity.NEGATIVE} <= signs}

    edges = []
    for f in merged:
        support, reason = _support(f, (f["source"], f["target"]) in conflicted_pairs)
        edges.append(TransmissionEdge(**f, support=support, support_reason=reason))

    for source, target in sorted(conflicted_pairs):
        ids = tuple(e.edge_id for e in edges if (e.source, e.target) == (source, target))
        issues.append(GraphIssue(
            kind=GraphIssueKind.DIRECTION_CONFLICT, severity=IssueSeverity.UNRESOLVED,
            message=f"{source} -> {target} is asserted with opposite signs; both links are kept and marked "
                    "UNRESOLVED -- the contradiction is not averaged away.",
            state_ids=(source, target), edge_ids=ids,
        ))
    for kind, severity, predicate, message in (
        (GraphIssueKind.MISSING_PROVENANCE, IssueSeverity.UNRESOLVED,
         lambda e: not all(c.has_required_provenance for c in e.claims),
         "claims lack the provenance their origin requires"),
        (GraphIssueKind.UNVERIFIED_CITATION, IssueSeverity.INFO,
         lambda e: any(p.kind is ProvenanceKind.EXTERNAL_DOCUMENT and p.verification is Verification.UNVERIFIED
                       for c in e.claims for p in c.provenance),
         "cite an external source nobody has verified yet"),
        (GraphIssueKind.UNSIGNED_LINK, IssueSeverity.CAUTION, lambda e: not e.polarity.signed,
         "have an ambiguous or unknown sign"),
        (GraphIssueKind.UNKNOWN_LAG, IssueSeverity.INFO, lambda e: e.lag is TransmissionLag.UNKNOWN,
         "have an unknown lag"),
        (GraphIssueKind.UNKNOWN_CONFIDENCE, IssueSeverity.INFO,
         lambda e: e.confidence is LinkConfidence.UNKNOWN, "have unknown confidence"),
    ):
        hits = [e for e in edges if predicate(e)]
        if hits:
            issues.append(GraphIssue(
                kind=kind, severity=severity, message=f"{len(hits)} link(s) {message}.",
                state_ids=tuple(dict.fromkeys(s for e in hits for s in (e.source, e.target))),
                edge_ids=tuple(e.edge_id for e in hits),
            ))

    referenced = list(dict.fromkeys(
        [resolver.resolve(s) for s in include_states] + [s for e in edges for s in (e.source, e.target)]
    ))
    declared_order = {sid: i for i, sid in enumerate(resolver.states)}
    first_seen = {sid: i for i, sid in enumerate(referenced)}
    order = sorted(referenced, key=lambda sid: (declared_order.get(sid, len(declared_order)), first_seen[sid]))
    position = {sid: i for i, sid in enumerate(order)}
    states = tuple(resolver.get(sid) for sid in order)
    undeclared = [s.state_id for s in states if not s.declared]
    if undeclared:
        issues.append(GraphIssue(
            kind=GraphIssueKind.UNDECLARED_STATE, severity=IssueSeverity.CAUTION,
            message=f"{len(undeclared)} state(s) are not in the reviewed state catalog and were added as "
                    f"UNSPECIFIED: {', '.join(undeclared)}.",
            state_ids=tuple(undeclared),
        ))

    loops, truncated = _find_loops(order, edges)
    for loop in loops:
        issues.append(GraphIssue(
            kind=GraphIssueKind.FEEDBACK_LOOP, severity=IssueSeverity.INFO,
            message=f"{loop.loop_type.value.title()} feedback loop: {' -> '.join((*loop.state_ids, loop.state_ids[0]))}.",
            state_ids=loop.state_ids, edge_ids=loop.edge_ids,
        ))
    if truncated:
        issues.append(GraphIssue(
            kind=GraphIssueKind.LOOP_SEARCH_TRUNCATED, severity=IssueSeverity.CAUTION,
            message=f"Feedback-loop search stopped at its bound ({MAX_LOOP_LENGTH} states / {_MAX_LOOP_SEQUENCES} "
                    "cycles); longer loops may exist.",
        ))

    return TransmissionGraph(
        states=states, edges=tuple(sorted(edges, key=lambda e: (position[e.source], position[e.target], e.edge_id))),
        loops=loops, issues=tuple(issues), rejected_claims=tuple(rejected),
    )


def reachable_from(
    graph: TransmissionGraph, starts: Iterable[str], *, max_hops: int,
) -> tuple[frozenset[str], frozenset[str], tuple[str, ...]]:
    """Breadth-first reach from ``starts``: (state ids, edge ids, frontier
    states that have further links beyond ``max_hops``)."""
    depth = {s: 0 for s in starts if graph.has_state(s)}
    frontier = list(depth)
    edge_ids: set[str] = set()
    beyond: list[str] = []
    while frontier:
        nxt = []
        for sid in frontier:
            for e in graph.edges_from(sid):
                if e.target in depth:
                    # a link between two reached states (a cross-link or a
                    # loop closing back) is always kept, even at the boundary
                    edge_ids.add(e.edge_id)
                elif depth[sid] >= max_hops:
                    if sid not in beyond:
                        beyond.append(sid)
                else:
                    edge_ids.add(e.edge_id)
                    depth[e.target] = depth[sid] + 1
                    nxt.append(e.target)
        frontier = nxt
    return frozenset(depth), frozenset(edge_ids), tuple(beyond)


# ---------------------------------------------------------------------------
# model proposals -- the ONLY shape an LLM's transmission output may take
# ---------------------------------------------------------------------------


class ProposedCitation(BaseModel):
    """A source a model says supports its link. Carries no verification
    field: the model cannot vouch for its own citation."""

    model_config = {"frozen": True, "extra": "forbid"}

    title: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    publisher: str | None = None
    claim: str | None = None


class ProposedLink(BaseModel):
    """One model-proposed relationship. No status, origin or verification
    field exists (``extra="forbid"``), so a model output that tries to mark
    its own link SUPPORTED fails validation."""

    model_config = {"frozen": True, "extra": "forbid"}

    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    channel: TransmissionChannel = TransmissionChannel.OTHER
    polarity: Polarity
    lag: TransmissionLag = TransmissionLag.UNKNOWN
    confidence: LinkConfidence = LinkConfidence.UNKNOWN
    rationale: str = Field(min_length=3)
    citations: tuple[ProposedCitation, ...] = ()


class TransmissionProposal(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "transmission-proposal/1"
    #: The model id that produced the proposal.
    proposer: str = Field(min_length=1)
    #: Hash of the prompt/context the model saw, so the output is traceable.
    prompt_fingerprint: str = Field(min_length=1)
    links: tuple[ProposedLink, ...] = Field(default=(), max_length=40)


def claims_from_proposal(proposal: TransmissionProposal) -> tuple[TransmissionClaim, ...]:
    """Model links -> MODEL_PROPOSED claims. Each carries the MODEL_OUTPUT
    record, and each model citation is kept as an UNVERIFIED external
    document -- so the resulting link is PROPOSED, never SUPPORTED."""
    output = ProvenanceSource(
        kind=ProvenanceKind.MODEL_OUTPUT, reference=f"{proposal.proposer}#{proposal.prompt_fingerprint}",
        title=f"Model proposal ({proposal.proposer})",
        note="A model's proposal is a hypothesis, never evidence.",
    )
    claims = []
    for link in proposal.links:
        cited = tuple(
            ProvenanceSource(
                kind=ProvenanceKind.EXTERNAL_DOCUMENT, reference=c.reference, title=c.title, publisher=c.publisher,
                claim=c.claim, note="Cited by the model; not verified.",
            )
            for c in link.citations
        )
        claims.append(TransmissionClaim(
            source=link.source, target=link.target, channel=link.channel, polarity=link.polarity, lag=link.lag,
            confidence=link.confidence, origin=EdgeOrigin.MODEL_PROPOSED, rationale=link.rationale,
            provenance=(output, *cited),
        ))
    return tuple(claims)
