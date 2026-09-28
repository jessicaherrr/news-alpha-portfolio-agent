"""News Alpha Phase D -- ASSET EXPRESSION.

    ResearchMandate -> Allowed Asset Universe -> News/Event -> Initial Impact Scan
        -> Economic Mechanism Graph -> Signal Paths -> Economic Consequence
        -> ASSET EXPRESSION (this module) -> Measurement Spec -> PIT Data Field Resolution
        -> (quantitative Candidate Signals: a later stage, not here)

Phase C ends each signal path at an ECONOMIC CONSEQUENCE (copper demand up).
This stage answers where that consequence could be expressed in tradable
markets, whether the user's mandate lets it continue there, and -- through
`news_alpha.measurement` -- how each surviving expression would be measured
and which measurements actually existed at the historical point in time.

ECONOMIC RELEVANCE IS NOT USER-ALLOWED EXPRESSION. The expressions of a
consequence come from the reviewed `expression_library` alone, so they are
identical under every mandate (test-enforced); the mandate then decides, per
expression, whether it CONTINUES (``expression-admission/1``):

* EXCLUDED_BY_MANDATE -- the domain is not allowed. Kept visible, never
  enumerated: no instrument, no measurement;
* DOMAIN_UNAVAILABLE -- allowed, but the platform has no real data or
  execution path for the domain (Options: no universe; Crypto: synthetic
  only). Its measurements are resolved -- so the user sees how it WOULD be
  measured -- and every one is DOMAIN_UNAVAILABLE;
* NO_INSTRUMENT -- a real tradable concept (copper miners, HBM makers) with
  no member in the platform's declared universe. Reported, never filled with
  a nearest-looking ticker;
* REMOVED_BY_CONSTRAINTS -- the mandate's instrument allow/deny list removed
  every instrument that carries it;
* CONTINUES -- at least one in-scope instrument; its measurements are
  resolved field by field.

`execution_capable` is True only for a CONTINUES expression with an
instrument the platform can actually research and simulate today (a
certified Futures root, an acquired ETF) -- never for Options, Crypto, or an
Equity foundation name, whatever the mandate says. It is a PLATFORM fact,
not candidate readiness: `AssetExpressionPlan.candidate_signal_basis` is the
gate a later Candidate Signal stage must pass, and it additionally requires
at least one real, point-in-time safe, executable, historically usable
measurement on a research-ready instrument of the expression.

FIDELITY. Each expression carries its rule's `ExpressionFidelity` (direct
underlying ... macro proxy) beside its form, so a copper future and a basket
that only sits near the actors never read as the same claim. Descriptive
only: no status, order, count or gate reads it.

GAPS. A measurement may be blocked by several gaps at once; `blockers()`
counts, per gap, where it is the primary blocker, the only blocker, and an
additional one -- closing a gap removes one blocker, and only unlocks the
measurements it was the last blocker for.

PRESSURE IS A HYPOTHESIS. Each expression carries, per distinct (direction,
horizon) of the paths reaching its consequence, the first-order pressure on
the instrument's value that the relation implies (consequence up x moves
with -> up). Never a forecast, trade, weight or expected return; a
downward pressure under a no-shorting mandate is noted, not dropped --
research may still study it.

HYPOTHESIS PLANE ONLY -- no registry, network or LLM access.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from enum import Enum
from typing import Literal

from pydantic import BaseModel, model_validator

from alpha_agent.crypto.provenance import DataProvenanceRole
from alpha_agent.features.registry import FeatureRegistry
from alpha_agent.news_alpha.channels import PressureSign
from alpha_agent.news_alpha.expression_library import (
    DEFAULT_EXPRESSION_LIBRARY,
    EXPRESSION_LIBRARY_VERSION,
    DataRequirement,
    ExpressionFidelity,
    ExpressionForm,
    ExpressionLibrary,
    ExpressionRule,
)
from alpha_agent.news_alpha.mandate import DOMAIN_LABELS, MandateDomain, ResearchMandate
from alpha_agent.news_alpha.measurement import (
    FIELD_RESOLUTION_RULE,
    MeasurementSpec,
    ResolutionIssue,
    ResolutionStatus,
    build_measurement_spec,
    status_for,
)
from alpha_agent.news_alpha.mechanism_graph import ImpliedMovement
from alpha_agent.news_alpha.signal_paths import SignalPath, SignalPathDiscovery
from alpha_agent.news_alpha.transmission import Polarity, TransmissionLag
from alpha_agent.news_alpha.universe import (
    AllowedAssetUniverse,
    DomainCapabilities,
    DomainScope,
    DomainSupport,
    UniverseInstrument,
)

__all__ = [
    "EXPRESSION_ADMISSION_RULE",
    "EXPRESSION_PLAN_SCHEMA_VERSION",
    "AssetExpression",
    "AssetExpressionPlan",
    "CandidateSignalPrerequisiteError",
    "DomainExpressionSummary",
    "ExpressionPressure",
    "ExpressionStatus",
    "MeasurementBlocker",
    "build_asset_expressions",
]

EXPRESSION_PLAN_SCHEMA_VERSION = "asset-expression-plan/1"
EXPRESSION_ADMISSION_RULE = "expression-admission/1"


class ExpressionStatus(str, Enum):
    CONTINUES = "CONTINUES"
    EXCLUDED_BY_MANDATE = "EXCLUDED_BY_MANDATE"
    DOMAIN_UNAVAILABLE = "DOMAIN_UNAVAILABLE"
    NO_INSTRUMENT = "NO_INSTRUMENT"
    REMOVED_BY_CONSTRAINTS = "REMOVED_BY_CONSTRAINTS"


_UNAVAILABLE_SUPPORT = frozenset({DomainSupport.NOT_SUPPORTED, DomainSupport.SYNTHETIC_ONLY})
#: Statuses whose measurements are resolved.
_MEASURED = frozenset({ExpressionStatus.CONTINUES, ExpressionStatus.DOMAIN_UNAVAILABLE})


class ExpressionPressure(BaseModel):
    """The first-order pressure one group of paths implies for an
    expression's value. A hypothesis to test, never a forecast."""

    model_config = {"frozen": True, "extra": "forbid"}

    path_ids: tuple[str, ...]
    consequence_direction: ImpliedMovement
    horizon: TransmissionLag
    #: ``None`` when the consequence's direction or the relation is not signed.
    pressure: PressureSign | None


class AssetExpression(BaseModel):
    """One way one economic consequence could be expressed in one domain."""

    model_config = {"frozen": True, "extra": "forbid"}

    #: Structural: the same consequence/domain/concept from any event has
    #: the same id (like `SignalPath.signature`).
    expression_id: str
    rule_id: str
    consequence_state: str
    consequence_label: str
    target_concept: str | None
    #: Every signal path ending at this consequence.
    path_ids: tuple[str, ...]
    domain: MandateDomain
    concept: str
    form: ExpressionForm
    #: How directly the instrument carries the consequence -- descriptive
    #: only; never a score, rank or strength (`ExpressionFidelity`).
    fidelity: ExpressionFidelity
    #: The instrument's value moves WITH (POSITIVE) or AGAINST (NEGATIVE) the
    #: consequence; AMBIGUOUS when the same move helps and hurts the concept.
    relation: Polarity
    rationale: str
    status: ExpressionStatus
    status_note: str
    #: In scope after the mandate. Always ``()`` unless CONTINUES.
    instruments: tuple[UniverseInstrument, ...] = ()
    #: Symbols the mandate's instrument constraints removed.
    removed_by_constraints: tuple[str, ...] = ()
    #: The PLATFORM can research and simulate at least one in-scope
    #: instrument (a certified Futures root, an acquired ETF). It is NOT
    #: "ready for a candidate signal": that additionally needs a real,
    #: point-in-time safe, executable, historically usable measurement --
    #: see `candidate_signal_basis`, the gate Phase E must pass through.
    execution_capable: bool = False
    pressures: tuple[ExpressionPressure, ...] = ()
    mandate_notes: tuple[str, ...] = ()
    measurement_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> AssetExpression:
        if self.status is not ExpressionStatus.CONTINUES and self.instruments:
            raise ValueError("only a continuing expression enumerates instruments")
        if self.status not in _MEASURED and self.measurement_ids:
            raise ValueError("only a continuing or domain-unavailable expression is measured")
        if self.execution_capable and not (
            self.status is ExpressionStatus.CONTINUES and any(i.research_ready for i in self.instruments)
        ):
            raise ValueError("execution_capable needs a continuing expression with a research-ready instrument")
        return self

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(i.symbol for i in self.instruments)


class DomainExpressionSummary(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    domain: MandateDomain
    in_mandate: bool
    support: DomainSupport
    expressions: int
    continuing: int
    execution_capable: int
    note: str


class MeasurementBlocker(BaseModel):
    """One named gap and the measurements it blocks. Closing a gap REMOVES
    ONE BLOCKER from each of them; only `only_blocker_for` of them would
    become usable by that alone -- the rest still wait on another gap.
    Counts, never a score or priority."""

    model_config = {"frozen": True, "extra": "forbid"}

    gap: str
    issue: ResolutionIssue
    #: The status the gap forces where it is the primary blocker.
    status: ResolutionStatus
    domains: tuple[MandateDomain, ...]
    requirements: tuple[DataRequirement, ...]
    #: Measurements whose status this gap decides.
    primary_for: int
    #: Of those, the ones with no other blocker -- usable once it closes.
    only_blocker_for: int
    #: Measurements where it is an ADDITIONAL blocker behind another gap.
    also_blocks: int
    targets: tuple[str, ...]

    @property
    def blocks(self) -> int:
        return self.primary_for + self.also_blocks

    @model_validator(mode="after")
    def _counts(self) -> MeasurementBlocker:
        if not 0 <= self.only_blocker_for <= self.primary_for or self.blocks < 1:
            raise ValueError("inconsistent blocker counts")
        return self


class CandidateSignalPrerequisiteError(ValueError):
    """An expression does not meet the Phase E candidate-signal
    prerequisites (see `AssetExpressionPlan.candidate_signal_basis`)."""


class AssetExpressionPlan(BaseModel):
    """Every asset expression for one event's signal paths, under one
    mandate, with the measurements behind them. `plane` is fixed to
    HYPOTHESIS: this object is never evidence and never a trade."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = EXPRESSION_PLAN_SCHEMA_VERSION
    plane: Literal["HYPOTHESIS"] = "HYPOTHESIS"
    rules: tuple[str, ...] = (EXPRESSION_ADMISSION_RULE, FIELD_RESOLUTION_RULE, EXPRESSION_LIBRARY_VERSION)
    event_id: str
    event_headline: str
    signal_paths_id: str
    mandate_fingerprint: str
    capabilities: DomainCapabilities
    expressions: tuple[AssetExpression, ...]
    measurements: tuple[MeasurementSpec, ...]
    domains: tuple[DomainExpressionSummary, ...]
    #: Consequences a path reaches with no expression rule -- a library gap.
    unmapped_consequences: tuple[str, ...] = ()
    not_trade_note: str = (
        "Asset expression: where an economic consequence could be expressed in a market and how it would be "
        "measured -- not an expected return, probability, forecast, trade instruction, portfolio weight, or "
        "scientific verdict. Pressure signs are hypotheses to test."
    )

    @model_validator(mode="after")
    def _references_resolve(self) -> AssetExpressionPlan:
        ids = {m.spec_id for m in self.measurements}
        for e in self.expressions:
            if not set(e.measurement_ids) <= ids:
                raise ValueError(f"{e.expression_id} references a measurement the plan does not hold")
        return self

    @property
    def is_empty(self) -> bool:
        return not self.expressions

    def of_domain(self, domain: MandateDomain) -> tuple[AssetExpression, ...]:
        return tuple(e for e in self.expressions if e.domain is domain)

    def for_path(self, path_id: str) -> tuple[AssetExpression, ...]:
        return tuple(e for e in self.expressions if path_id in e.path_ids)

    def with_status(self, status: ExpressionStatus) -> tuple[AssetExpression, ...]:
        return tuple(e for e in self.expressions if e.status is status)

    def measurement(self, spec_id: str) -> MeasurementSpec:
        return next(m for m in self.measurements if m.spec_id == spec_id)

    def measurements_for(self, expression: AssetExpression) -> tuple[MeasurementSpec, ...]:
        return tuple(self.measurement(i) for i in expression.measurement_ids)

    def usable_measurements(self) -> tuple[MeasurementSpec, ...]:
        return tuple(m for m in self.measurements if m.resolution.historically_usable)

    def measurable_expressions(self) -> tuple[AssetExpression, ...]:
        """Continuing expressions with at least one historically usable
        measurement -- what could be researched on past data today."""
        usable = {m.spec_id for m in self.usable_measurements()}
        return tuple(
            e for e in self.expressions if e.status is ExpressionStatus.CONTINUES and usable & set(e.measurement_ids)
        )

    def count_by_resolution(self) -> dict[ResolutionStatus, int]:
        return {s: sum(m.resolution.status is s for m in self.measurements) for s in ResolutionStatus}

    def blockers(self) -> tuple[MeasurementBlocker, ...]:
        """Every named gap across the unusable measurements, with where it
        is the primary blocker, where it is the ONLY blocker, and where it
        is an additional one -- largest primary count first. Every typed
        blocker of every measurement is counted exactly once."""
        primary: dict[str, list[MeasurementSpec]] = defaultdict(list)
        secondary: dict[str, list[MeasurementSpec]] = defaultdict(list)
        issue_of: dict[str, ResolutionIssue] = {}
        for m in self.measurements:
            for i, g in enumerate(m.resolution.gaps):
                issue_of.setdefault(g.label, g.issue)
                (primary if i == 0 else secondary)[g.label].append(m)
        order = {d: i for i, d in enumerate(MandateDomain)}
        out = []
        for gap in dict.fromkeys([*primary, *secondary]):
            ms = primary[gap] + secondary[gap]
            out.append(MeasurementBlocker(
                gap=gap, issue=issue_of[gap], status=status_for((issue_of[gap],)),
                domains=tuple(sorted({m.domain for m in ms}, key=order.__getitem__)),
                requirements=tuple(dict.fromkeys(m.requirement for m in ms)),
                primary_for=len(primary[gap]),
                only_blocker_for=sum(len(m.resolution.gaps) == 1 for m in primary[gap]),
                also_blocks=len(secondary[gap]),
                targets=tuple(dict.fromkeys(m.target for m in ms)),
            ))
        return tuple(sorted(out, key=lambda b: (-b.primary_for, -b.blocks, order[b.domains[0]], b.gap)))

    def candidate_signal_basis(self, expression: AssetExpression) -> tuple[MeasurementSpec, ...]:
        """THE GATE Phase E must pass through before any Candidate Signal.
        `execution_capable` only says the platform can research and simulate
        an instrument; a candidate additionally needs at least one REAL,
        point-in-time safe, executable, historically usable measurement ON a
        research-ready instrument of this expression. Returns those
        measurements, or raises `CandidateSignalPrerequisiteError` with the
        reason -- never an empty basis."""
        if expression not in self.expressions:
            raise CandidateSignalPrerequisiteError(f"{expression.expression_id} is not an expression of this plan")
        if expression.status is not ExpressionStatus.CONTINUES:
            raise CandidateSignalPrerequisiteError(
                f"{expression.concept}: status {expression.status.value} -- only a continuing expression can "
                "carry a candidate signal"
            )
        if not expression.execution_capable:
            raise CandidateSignalPrerequisiteError(
                f"{expression.concept}: no in-scope instrument the platform can research and simulate"
            )
        ready = {i.symbol for i in expression.instruments if i.research_ready}
        basis = tuple(
            m for m in self.measurements_for(expression)
            if m.symbol in ready and m.resolution.historically_usable
            and m.resolution.pit_safe is True and m.resolution.executable
            and m.resolution.data_role is DataProvenanceRole.REAL
        )
        if not basis:
            raise CandidateSignalPrerequisiteError(
                f"{expression.concept}: execution-capable, but no real, point-in-time safe, executable, "
                "historically usable measurement exists on a research-ready instrument -- not candidate-ready"
            )
        return basis

    def fingerprint(self) -> str:
        return "assetexpr1:" + hashlib.sha256(self.model_dump_json().encode()).hexdigest()


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------


def _expression_id(rule: ExpressionRule) -> str:
    key = f"{rule.state_id}|{rule.domain.value}|{rule.concept}|{rule.form.value}"
    return "expr-" + hashlib.sha256(key.encode()).hexdigest()[:16]


def _matches(rule: ExpressionRule, inst: UniverseInstrument) -> bool:
    return inst.symbol in rule.symbols or inst.group in rule.groups


def _pressure(direction: ImpliedMovement, relation: Polarity) -> PressureSign | None:
    if direction not in (ImpliedMovement.UP, ImpliedMovement.DOWN) or not relation.signed:
        return None
    up = (direction is ImpliedMovement.UP) == (relation is Polarity.POSITIVE)
    return PressureSign.UP if up else PressureSign.DOWN


def _pressures(paths: list[SignalPath], relation: Polarity) -> tuple[ExpressionPressure, ...]:
    groups: dict[tuple[ImpliedMovement, TransmissionLag], list[str]] = defaultdict(list)
    for p in paths:
        groups[(p.expected_direction, p.expected_horizon)].append(p.path_id)
    lag_order = {lag: i for i, lag in enumerate(TransmissionLag)}
    return tuple(
        ExpressionPressure(path_ids=tuple(ids), consequence_direction=d, horizon=h, pressure=_pressure(d, relation))
        for (d, h), ids in sorted(groups.items(), key=lambda kv: (lag_order[kv[0][1]], kv[0][0].value))
    )


def _admit(
    rule: ExpressionRule, scope: DomainScope,
) -> tuple[ExpressionStatus, str, tuple[UniverseInstrument, ...], tuple[str, ...]]:
    label = DOMAIN_LABELS[rule.domain]
    if not scope.in_mandate:
        return ExpressionStatus.EXCLUDED_BY_MANDATE, f"{label} is excluded by your research mandate.", (), ()
    if scope.support in _UNAVAILABLE_SUPPORT:
        return ExpressionStatus.DOMAIN_UNAVAILABLE, scope.support_note, (), ()
    kept = tuple(i for i in scope.instruments if _matches(rule, i))
    removed = tuple(i.symbol for i in scope.removed_by_constraints if _matches(rule, i))
    if kept:
        return ExpressionStatus.CONTINUES, scope.support_note, kept, removed
    if removed:
        note = f"Your mandate's instrument constraints removed {', '.join(removed)}."
        return ExpressionStatus.REMOVED_BY_CONSTRAINTS, note, (), removed
    note = (
        f"No {rule.concept.lower()} in the platform's declared {label} universe -- a tradable concept without an "
        "instrument here."
    )
    return ExpressionStatus.NO_INSTRUMENT, note, (), ()


def _mandate_notes(pressures: tuple[ExpressionPressure, ...], mandate: ResearchMandate) -> tuple[str, ...]:
    down = [p for p in pressures if p.pressure is PressureSign.DOWN]
    if not down or mandate.shorting_allowed:
        return ()
    horizons = "/".join(dict.fromkeys(p.horizon.value.lower() for p in down))
    note = (
        f"Shorting not allowed: the downward-pressure hypothesis (over {horizons}) can be researched but not held "
        "as a net short under your mandate."
    )
    return (note,)


def build_asset_expressions(
    discovery: SignalPathDiscovery,
    universe: AllowedAssetUniverse,
    mandate: ResearchMandate,
    *,
    library: ExpressionLibrary = DEFAULT_EXPRESSION_LIBRARY,
    registry: FeatureRegistry | None = None,
) -> AssetExpressionPlan:
    """``expression-admission/1`` + ``field-resolution/1`` for every
    consequence one event's signal paths reach. Pure and deterministic for a
    fixed (discovery, universe, mandate, library, registry)."""
    if universe.mandate_fingerprint != mandate.fingerprint():
        raise ValueError("the universe was resolved for a different mandate")
    by_consequence: dict[str, list[SignalPath]] = defaultdict(list)
    for p in discovery.paths:
        by_consequence[p.consequence_state].append(p)

    expressions: list[AssetExpression] = []
    measured: dict[str, tuple[MeasurementSpec, list[str]]] = {}
    unmapped: list[str] = []
    caps = universe.capabilities
    for state_id, paths in by_consequence.items():
        rules = library.rules_for(state_id)
        if not rules:
            unmapped.append(state_id)
            continue
        first = paths[0]
        for rule in rules:
            scope = universe.scope(rule.domain)
            status, note, kept, removed = _admit(rule, scope)
            expression_id = _expression_id(rule)
            pressures = _pressures(paths, rule.relation)
            spec_ids: list[str] = []
            if status in _MEASURED:
                # A domain with no universe measures the underlying the rule names.
                targets = [i.symbol for i in kept] or list(rule.symbols) or [None]
                for template in library.templates_for(rule):
                    for symbol in targets:
                        spec = build_measurement_spec(
                            template, symbol, support=scope.support, capabilities=caps, registry=registry,
                        )
                        measured.setdefault(spec.spec_id, (spec, []))[1].append(expression_id)
                        spec_ids.append(spec.spec_id)
            expressions.append(AssetExpression(
                expression_id=expression_id, rule_id=rule.rule_id, consequence_state=state_id,
                consequence_label=first.consequence_label, target_concept=first.candidate_target_concept,
                path_ids=tuple(p.path_id for p in paths), domain=rule.domain, concept=rule.concept, form=rule.form,
                fidelity=rule.fidelity,
                relation=rule.relation, rationale=rule.rationale, status=status, status_note=note, instruments=kept,
                removed_by_constraints=removed,
                execution_capable=status is ExpressionStatus.CONTINUES and any(i.research_ready for i in kept),
                pressures=pressures,
                mandate_notes=_mandate_notes(pressures, mandate) if status is ExpressionStatus.CONTINUES else (),
                measurement_ids=tuple(dict.fromkeys(spec_ids)),
            ))

    measurements = tuple(
        spec.model_copy(update={"expression_ids": tuple(dict.fromkeys(ids))}) for spec, ids in measured.values()
    )
    return AssetExpressionPlan(
        event_id=discovery.event_id, event_headline=discovery.event_headline,
        signal_paths_id=discovery.fingerprint(), mandate_fingerprint=mandate.fingerprint(), capabilities=caps,
        expressions=tuple(expressions), measurements=measurements,
        domains=tuple(_domain_summary(d, universe.scope(d), expressions) for d in MandateDomain),
        unmapped_consequences=tuple(unmapped),
    )


def _domain_summary(
    domain: MandateDomain, scope: DomainScope, expressions: list[AssetExpression],
) -> DomainExpressionSummary:
    mine = [e for e in expressions if e.domain is domain]
    continuing = [e for e in mine if e.status is ExpressionStatus.CONTINUES]
    if not scope.in_mandate:
        note = "Excluded by your mandate" + (f" ({len(mine)} economically relevant expressions not pursued)."
                                              if mine else ".")
    elif scope.support in _UNAVAILABLE_SUPPORT:
        note = f"Allowed, but {scope.support_note[0].lower()}{scope.support_note[1:]}"
    elif not mine:
        note = "Allowed; no consequence of this event is expressed here by the reviewed library."
    else:
        note = scope.support_note
    return DomainExpressionSummary(
        domain=domain, in_mandate=scope.in_mandate, support=scope.support, expressions=len(mine),
        continuing=len(continuing), execution_capable=sum(e.execution_capable for e in mine), note=note,
    )
