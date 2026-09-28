"""Builds the Phase 7 Cross-Asset Alpha Graph purely by READING existing
sources of truth: the experiment registry (via
:mod:`alpha_agent.alpha_memory.builder`, the SAME machinery the Factor
Library already uses for Futures) plus Phase 1's frozen, deterministic
``EconomicMechanism`` -> candidate-family bridges (Futures:
:mod:`alpha_agent.translation.research_memory`; ETF:
:mod:`alpha_agent.etf.alpha_memory_bridge`) and ``NewsCategory`` -> Mechanism
mechanism library (:mod:`alpha_agent.translation.mechanism_library`). Nothing
here writes a registry row, calls Claude, calls a market-data API, or runs a
backtest (prompt section 1/9/10).

No second scientific database, no graph database, no caching: every call
recomputes straight from the registry, exactly like
:mod:`alpha_agent.alpha_memory.builder` itself does.

Instrument universe (prompt section 1: "use existing sources of truth", never
a broadened research universe): Futures = the frozen
:data:`alpha_agent.translation.schemas.CERTIFIED_ROOTS`; ETF = the frozen
Phase 6 :data:`alpha_agent.etf.universe.PILOT_UNIVERSE`. Mechanism universe:
the union of keys in :data:`alpha_agent.translation.research_memory.MECHANISM_TO_KNOWN_FAMILIES`
(Futures) and :data:`alpha_agent.etf.alpha_memory_bridge.ETF_MECHANISM_TO_FAMILIES`
(ETF) -- a mechanism absent from BOTH bridges has no representation in either
domain's candidate manifest today and is out of scope, exactly like Phase 1's
own "a mechanism absent from this table has no representation yet" stance.
"""
from __future__ import annotations

from alpha_agent.alpha_graph.schemas import (
    CrossAssetSynthesis,
    EventCategoryRef,
    EvidenceCoverage,
    FactorUnderMechanism,
    InstrumentEvidence,
    InstrumentRef,
    MechanismGraphSummary,
    MechanismGraphView,
    ResearchGapRow,
)
from alpha_agent.alpha_memory.builder import build_alpha_research_objects_for_mechanism
from alpha_agent.alpha_memory.factor_identity import (
    compute_factor_identity,
    group_families_for_factor_identity,
)
from alpha_agent.alpha_memory.schemas import AlphaResearchObject, ResearchMaturity
from alpha_agent.etf.alpha_memory_bridge import ETF_MECHANISM_TO_FAMILIES
from alpha_agent.etf.universe import PILOT_UNIVERSE
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.registry.enums import AssetDomain
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.translation.mechanism_library import CATEGORY_MECHANISM_TEMPLATES
from alpha_agent.translation.research_memory import MECHANISM_TO_KNOWN_FAMILIES
from alpha_agent.translation.schemas import CERTIFIED_ROOTS

__all__ = [
    "MECHANISM_UNIVERSE",
    "build_alpha_graph",
    "build_mechanism_graph",
    "cross_asset_synthesis",
    "list_considered_instruments",
    "list_research_gaps",
    "summarize",
]

#: every mechanism either real bridge maps today -- see module docstring.
MECHANISM_UNIVERSE: tuple[EconomicMechanism, ...] = tuple(
    sorted(set(MECHANISM_TO_KNOWN_FAMILIES) | set(ETF_MECHANISM_TO_FAMILIES), key=lambda m: m.value)
)

_MATURITY_RANK: dict[ResearchMaturity, int] = {
    ResearchMaturity.IDEA: 0, ResearchMaturity.FORMALIZED: 1, ResearchMaturity.TESTED: 2,
    ResearchMaturity.REPLICATED: 3, ResearchMaturity.ADJUDICATED: 4,
}
_MATURITY_RESEARCHED = (ResearchMaturity.REPLICATED, ResearchMaturity.ADJUDICATED)


def list_considered_instruments() -> tuple[InstrumentRef, ...]:
    """Every Instrument node the graph considers -- the SAME frozen
    universes the Factor Library already uses for each domain, never a
    broadened one."""
    out = [InstrumentRef(root_symbol=r, asset_domain=AssetDomain.FUTURES) for r in CERTIFIED_ROOTS]
    out += [InstrumentRef(root_symbol=r, asset_domain=AssetDomain.ETF) for r in PILOT_UNIVERSE]
    return tuple(out)


def _family_groups(mechanism: EconomicMechanism, asset_domain: AssetDomain) -> tuple[tuple[str, ...], ...]:
    if asset_domain is AssetDomain.FUTURES:
        return group_families_for_factor_identity(mechanism)
    families = ETF_MECHANISM_TO_FAMILIES.get(mechanism, ())
    return tuple((f,) for f in families)


def _event_categories_for_mechanism(mechanism: EconomicMechanism) -> tuple[EventCategoryRef, ...]:
    """Reverse-indexes Phase 1's frozen, no-network
    ``CATEGORY_MECHANISM_TEMPLATES`` -- a real, already-committed Event ->
    Mechanism edge, never a live news fetch."""
    out: list[EventCategoryRef] = []
    for category, templates in CATEGORY_MECHANISM_TEMPLATES.items():
        matches = [t for t in templates if t.mechanism is mechanism]
        if matches:
            out.append(EventCategoryRef(category=category, evidence_basis=matches[0].evidence_basis))
    return tuple(sorted(out, key=lambda e: e.category.value))


def _coverage(objects: tuple[AlphaResearchObject, ...], *, mapped: bool) -> EvidenceCoverage:
    if not objects:
        return EvidenceCoverage.UNDEREXPLORED if mapped else EvidenceCoverage.NO_EVIDENCE
    if any(o.research_maturity in _MATURITY_RESEARCHED for o in objects):
        return EvidenceCoverage.RESEARCHED
    return EvidenceCoverage.WEAKLY_RESEARCHED


def _instrument_evidence(instrument: InstrumentRef, objects: tuple[AlphaResearchObject, ...], *, mapped: bool) -> InstrumentEvidence:
    coverage = _coverage(objects, mapped=mapped)
    scientific_evidence: dict[str, str] = {}
    reason_codes: dict[str, int] = {}
    failure_classes: dict[str, int] = {}
    best_maturity: ResearchMaturity | None = None
    for obj in objects:
        scientific_evidence.update(obj.evidence_profile.scientific_evidence)
        for code, n in obj.repeated_failure_reason_codes.items():
            reason_codes[code] = reason_codes.get(code, 0) + n
        for cls, n in obj.repeated_failure_classes.items():
            failure_classes[cls] = failure_classes.get(cls, 0) + n
        if best_maturity is None or _MATURITY_RANK[obj.research_maturity] > _MATURITY_RANK[best_maturity]:
            best_maturity = obj.research_maturity

    return InstrumentEvidence(
        instrument=instrument,
        coverage=coverage,
        alpha_ids=tuple(sorted(o.alpha_id for o in objects)),
        research_maturity=best_maturity.value if best_maturity is not None else None,
        scientific_evidence=scientific_evidence,
        repeated_failure_reason_codes=reason_codes,
        repeated_failure_classes=failure_classes,
    )


def _factors_for_mechanism(
    mechanism: EconomicMechanism, factor_instruments: dict[tuple[AssetDomain, str], list[InstrumentRef]],
) -> tuple[FactorUnderMechanism, ...]:
    """Every real, deterministically-computable Factor under this mechanism,
    in BOTH domains -- a pure function of (mechanism, family), so it is
    listed even for a family with zero current experiments (an honest
    "researchable, not yet researched" Factor node). ``factor_instruments``
    ((asset_domain, factor_identity) -> real instruments carrying >=1
    AlphaResearchObject for it) is ground truth collected directly off real
    materialized objects while building `instrument_evidence` -- never
    re-derived or guessed. Keyed by domain AND identity together so a
    FUTURES family and an ETF family can never merge instruments even if
    their factor_identity hashes ever collided (see `build_mechanism_graph`)."""
    out: list[FactorUnderMechanism] = []
    for domain in (AssetDomain.FUTURES, AssetDomain.ETF):
        for families in _family_groups(mechanism, domain):
            factor = compute_factor_identity(mechanism, families)
            instruments = tuple(
                sorted(factor_instruments.get((domain, factor.factor_identity), ()), key=lambda i: i.root_symbol)
            )
            out.append(FactorUnderMechanism(factor=factor, asset_domain=domain, instruments_with_evidence=instruments))
    return tuple(out)


def build_mechanism_graph(registry: ExperimentRegistry, mechanism: EconomicMechanism) -> MechanismGraphView:
    """The full navigable view for one Mechanism -- every considered
    Instrument's evidence across BOTH asset domains, the real Factor(s) that
    represent it, and cross-asset aggregate facts. Never merges or transfers
    a verdict across instruments (prompt section 4)."""
    evidence: list[InstrumentEvidence] = []
    #: keyed by (asset_domain, factor_identity) -- NEVER factor_identity
    #: alone. `compute_factor_identity` hashes only (mechanism, strategy
    #: family names); it does not include asset_domain, so a FUTURES family
    #: and an ETF family that ever shared a name would otherwise collide on
    #: the same factor_identity string and silently merge two different
    #: domains' instruments onto one Factor node -- a real asset-domain
    #: isolation violation (prompt section 4/9), not merely a display bug.
    factor_instruments: dict[tuple[AssetDomain, str], list[InstrumentRef]] = {}

    for inst in list_considered_instruments():
        groups = _family_groups(mechanism, inst.asset_domain)
        objects = build_alpha_research_objects_for_mechanism(
            registry, root_symbol=inst.root_symbol, mechanism=mechanism,
            asset_domain=inst.asset_domain, family_groups_override=groups,
        )
        for obj in objects:
            factor_instruments.setdefault((inst.asset_domain, obj.factor.factor_identity), []).append(inst)
        evidence.append(_instrument_evidence(inst, objects, mapped=bool(groups)))

    domains_with_evidence = tuple(
        sorted(
            {
                e.instrument.asset_domain
                for e in evidence
                if e.coverage in (EvidenceCoverage.RESEARCHED, EvidenceCoverage.WEAKLY_RESEARCHED)
            },
            key=lambda d: d.value,
        )
    )
    reason_codes: dict[str, int] = {}
    failure_classes: dict[str, int] = {}
    for e in evidence:
        for code, n in e.repeated_failure_reason_codes.items():
            reason_codes[code] = reason_codes.get(code, 0) + n
        for cls, n in e.repeated_failure_classes.items():
            failure_classes[cls] = failure_classes.get(cls, 0) + n

    return MechanismGraphView(
        mechanism=mechanism,
        event_categories=_event_categories_for_mechanism(mechanism),
        factors=_factors_for_mechanism(mechanism, factor_instruments),
        instrument_evidence=tuple(evidence),
        domains_with_evidence=domains_with_evidence,
        repeated_failure_reason_codes=reason_codes,
        repeated_failure_classes=failure_classes,
    )


def build_alpha_graph(
    registry: ExperimentRegistry, mechanisms: tuple[EconomicMechanism, ...] | None = None,
) -> tuple[MechanismGraphView, ...]:
    """Every considered Mechanism's graph view, sorted for a stable landing
    order. ``mechanisms`` defaults to :data:`MECHANISM_UNIVERSE`."""
    return tuple(build_mechanism_graph(registry, m) for m in (mechanisms or MECHANISM_UNIVERSE))


def summarize(view: MechanismGraphView) -> MechanismGraphSummary:
    counts = dict.fromkeys(EvidenceCoverage, 0)
    for e in view.instrument_evidence:
        counts[e.coverage] += 1
    return MechanismGraphSummary(
        mechanism=view.mechanism,
        n_researched=counts[EvidenceCoverage.RESEARCHED],
        n_weakly_researched=counts[EvidenceCoverage.WEAKLY_RESEARCHED],
        n_underexplored=counts[EvidenceCoverage.UNDEREXPLORED],
        n_no_evidence=counts[EvidenceCoverage.NO_EVIDENCE],
        domains_with_evidence=view.domains_with_evidence,
        is_cross_asset=len(view.domains_with_evidence) >= 2,
    )


def list_research_gaps(views: tuple[MechanismGraphView, ...]) -> tuple[ResearchGapRow, ...]:
    """Every (Mechanism, Instrument) pair at UNDEREXPLORED or NO_EVIDENCE
    coverage, flattened -- first-class, never hidden (prompt section 6)."""
    out: list[ResearchGapRow] = []
    for view in views:
        for e in view.instrument_evidence:
            if e.coverage in (EvidenceCoverage.UNDEREXPLORED, EvidenceCoverage.NO_EVIDENCE):
                out.append(ResearchGapRow(mechanism=view.mechanism, instrument=e.instrument, coverage=e.coverage))
    return tuple(out)


def cross_asset_synthesis(view: MechanismGraphView) -> CrossAssetSynthesis:
    """The deterministic "what is common / what differs / repeated failures
    / what remains underexplored" synthesis for one Mechanism (prompt
    section 5/8). Every sentence is built from already-computed, per-
    instrument fields -- never a merged verdict, never a trade instruction
    (see ``retrieval.py``'s dedicated regression tests for the exact banned
    vocabulary)."""
    researched = [
        e for e in view.instrument_evidence
        if e.coverage in (EvidenceCoverage.RESEARCHED, EvidenceCoverage.WEAKLY_RESEARCHED)
    ]
    common: list[str] = []
    differs: list[str] = []
    if len(view.domains_with_evidence) >= 2:
        by_domain: dict[AssetDomain, list[InstrumentEvidence]] = {}
        for e in researched:
            by_domain.setdefault(e.instrument.asset_domain, []).append(e)
        domain_bits = ", ".join(
            f"{d.value} ({', '.join(sorted(e.instrument.root_symbol for e in es))})"
            for d, es in sorted(by_domain.items(), key=lambda kv: kv[0].value)
        )
        common.append(f"{view.mechanism.value} has real registry evidence in more than one asset domain: {domain_bits}.")
    elif researched:
        common.append(
            f"{view.mechanism.value} has real registry evidence in exactly one asset domain so far: "
            f"{researched[0].instrument.asset_domain.value}."
        )
    else:
        common.append(f"{view.mechanism.value} has no real registry evidence yet in any asset domain.")

    if len(researched) >= 2:
        verdict_bits = ", ".join(
            f"{e.instrument.root_symbol} ({e.instrument.asset_domain.value}): "
            + ("; ".join(f"{fam}={v}" for fam, v in sorted(e.scientific_evidence.items())) or "NOT_ADJUDICATED")
            for e in sorted(researched, key=lambda e: (e.instrument.asset_domain.value, e.instrument.root_symbol))
        )
        differs.append(
            "Each instrument keeps its own, separately-adjudicated scientific verdict -- never merged or "
            f"transferred: {verdict_bits}."
        )

    repeated_failures = tuple(
        f"`{code}` recorded {n} time(s) across researched instruments"
        for code, n in sorted(view.repeated_failure_reason_codes.items(), key=lambda kv: -kv[1])
    )

    underexplored = tuple(
        f"{e.instrument.root_symbol} ({e.instrument.asset_domain.value}): {e.coverage.value}"
        for e in sorted(view.instrument_evidence, key=lambda e: (e.instrument.asset_domain.value, e.instrument.root_symbol))
        if e.coverage in (EvidenceCoverage.UNDEREXPLORED, EvidenceCoverage.NO_EVIDENCE)
    )

    return CrossAssetSynthesis(
        mechanism=view.mechanism, common=tuple(common), differs=tuple(differs),
        repeated_failures=repeated_failures, underexplored=underexplored,
    )
