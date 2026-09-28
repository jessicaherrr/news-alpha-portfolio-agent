"""News Alpha Phase E -- CANDIDATE SIGNALS: measurable consequences as typed,
testable factor hypotheses.

    ... -> Asset Expression -> Measurement Spec -> PIT Data Field Resolution  (Phase D)
        -> CANDIDATE SIGNAL SPEC -> FACTOR EXPRESSION (this module)
        -> factor series -> factor diagnostics  (`alpha_agent.screening.candidate_signal_screen`)

A `CandidateSignal` is a TESTABLE quantitative hypothesis: one registered
factor expression on one instrument, predicting that instrument's forward
return over a declared horizon with a declared sign, plus the complete
economic lineage that says why it exists. It is not a validated factor, an
alpha, a recommendation, a trade, a weight or a verdict.

THE GATE. Candidates are built ONLY from `AssetExpressionPlan.
candidate_signal_basis` -- the Phase D gate that returns the real,
point-in-time safe, executable, historically usable measurements of a
research-ready instrument. `execution_capable=True` alone never yields a
formula, and nothing here reads news text, an impact scan, a graph or a
path in isolation.

RULES (``candidate-signal-rules/2``, reviewed, predeclared -- never fitted):

* price momentum measurement -> ``ts_return(close, L)``, expected
  relationship POSITIVE (continuation: a consequence that diffuses into
  price gradually keeps moving it the same way);
* trading-activity measurement -> NO standalone candidate: volume says the
  market is paying attention, not which way. Kept as a typed refusal
  (CONFIRMATION_ONLY_MEASUREMENT) for a later event-conditioned test;
* TRANSMISSION HORIZON IS NOT PREDICTION HORIZON. A path's economic lag
  ("AI capex reaches HBM demand over months") says when the ECONOMICS
  arrive; prices can move long before or after. So no lag is converted
  into a market horizon: the lag stays on each origin as provenance, and
  the prediction horizons come from a predeclared GENERATION GRID
  (``candidate-generation/1``: (20 bars, 20D) and (60 bars, 60D)) applied
  to every directional measurement alike -- whatever the lag, including
  UNKNOWN. The diagnostics report 1/5/20/60D decay for every candidate;
* `formation_lookback` and `prediction_horizon` are INDEPENDENT fields.
  The grid starts with them matched -- an initial generation policy, not a
  semantic rule: (60 bars, 20D) is an equally valid spec with its own
  identity.

IDENTITY. `factor_identity` is STRUCTURAL: instrument + dataset + the
expression's canonical form (registered FeatureSpec, feature-engine
version; the lookback is inside the expression). `candidate_signal_id`
adds the prediction horizon, relationship, execution lag, measurement, rule
and generation policy. Neither contains an event, path or timestamp, so
several paths -- or several events -- reaching NQ ``ts_return(close, 20)``
share one candidate and keep every originating hypothesis in `origins`.
`alpha_memory.FactorIdentity` is a different object (quant mechanism x
strategy family, an economic idea with no expression or instrument plane);
it is not reused here and nothing maps a candidate onto a strategy family.

MAGNITUDES STAY UNKNOWN. The conceptual event-conditioned model
EventStrength x Exposure x Confirmation x TransmissionWeight is recorded per
candidate (`EventConditioning`) as separately testable components: the
factor IS the confirmation term; event strength and exposure have no
quantitative magnitude on the platform (Phase C `magnitude_established`,
Phase D fidelity is a category), and transmission weight is never derived
from link confidence. `ExpressionFidelity` travels in each origin as
provenance and is never read by any rule, identity or order here.

HYPOTHESIS PLANE ONLY -- no data is loaded, no registry, network or LLM.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, model_validator

from alpha_agent.crypto.provenance import DataProvenanceRole
from alpha_agent.features.expression import EXPRESSION_VOCABULARY_VERSION, FactorExpression
from alpha_agent.features.registry import FEATURE_ENGINE_VERSION
from alpha_agent.news_alpha.asset_expression import (
    AssetExpression,
    AssetExpressionPlan,
    CandidateSignalPrerequisiteError,
    ExpressionPressure,
)
from alpha_agent.news_alpha.channels import PressureSign
from alpha_agent.news_alpha.expression_library import (
    ExpressionFidelity,
    ExpressionForm,
    MeasurementRole,
)
from alpha_agent.news_alpha.mandate import MandateDomain
from alpha_agent.news_alpha.measurement import (
    DataFrequency,
    MeasurementSpec,
    PublicationLag,
    ResolutionStatus,
)
from alpha_agent.news_alpha.mechanism_graph import ImpliedMovement
from alpha_agent.news_alpha.signal_paths import (
    PathStatus,
    PathType,
    SignalPath,
    SignalPathDiscovery,
)
from alpha_agent.news_alpha.transmission import Polarity, TransmissionLag
from alpha_agent.registry.holdout_guard import HOLDOUT_START

__all__ = [
    "CANDIDATE_SET_SCHEMA_VERSION",
    "DEFAULT_GENERATION_GRID",
    "GENERATION_POLICY",
    "SIGNAL_RULES",
    "SIGNAL_RULES_VERSION",
    "CandidateNote",
    "CandidateOrigin",
    "CandidateRefusal",
    "CandidateSignal",
    "CandidateSignalSet",
    "ConditioningStatus",
    "EventConditioning",
    "ExpectedRelationship",
    "GenerationPoint",
    "MeasurementSignalRole",
    "RefusalReason",
    "SignalDataBinding",
    "SignalHorizon",
    "SignalRationale",
    "SignalRule",
    "SignalSpec",
    "build_candidate_signals",
]

CANDIDATE_SET_SCHEMA_VERSION = "candidate-signal-set/2"
SIGNAL_RULES_VERSION = "candidate-signal-rules/2"
GENERATION_POLICY = "candidate-generation/1"
_HOLDOUT = date.fromisoformat(HOLDOUT_START)


class SignalHorizon(str, Enum):
    D1 = "1D"
    D5 = "5D"
    D20 = "20D"
    D60 = "60D"

    @property
    def days(self) -> int:
        return int(self.value[:-1])


class ExpectedRelationship(str, Enum):
    """The predeclared sign between the factor and the forward return it
    predicts. Never re-signed after diagnostics: an opposite sign is a
    different hypothesis with a different identity."""

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"

    @property
    def sign(self) -> int:
        return 1 if self is ExpectedRelationship.POSITIVE else -1


@dataclass(frozen=True)
class GenerationPoint:
    """One (formation lookback, prediction horizon) pair the generation
    policy proposes. The two are independent; the default grid matches them
    only as a starting policy."""

    formation_lookback: int
    prediction_horizon: SignalHorizon


#: ``candidate-generation/1`` -- an initial GENERATION policy, not a semantic
#: rule and not derived from any economic lag: the same small grid for every
#: directional measurement, so the multiple-testing family stays small and
#: explicit. Changing it is a new policy with new candidate identities.
DEFAULT_GENERATION_GRID: tuple[GenerationPoint, ...] = (
    GenerationPoint(20, SignalHorizon.D20),
    GenerationPoint(60, SignalHorizon.D60),
)


class MeasurementSignalRole(str, Enum):
    #: The measurement becomes a standalone directional factor.
    DIRECTIONAL = "DIRECTIONAL"
    #: The measurement can confirm, but carries no direction of its own.
    CONFIRMATION_ONLY = "CONFIRMATION_ONLY"


@dataclass(frozen=True)
class SignalRule:
    rule_id: str
    measurement_ids: tuple[str, ...]
    role: MeasurementSignalRole
    rationale: str
    operator: str | None = None
    field: str | None = None
    relationship: ExpectedRelationship | None = None
    transform_label: str | None = None
    why_sign: str | None = None


SIGNAL_RULES: tuple[SignalRule, ...] = (
    SignalRule(
        "price-momentum-continuation", ("futures.momentum", "etf.momentum", "equity.momentum"),
        MeasurementSignalRole.DIRECTIONAL,
        rationale=(
            "The measurement names the instrument's own price response; its trailing return IS that response "
            "(the registered `return` feature -- nothing is added to it)."
        ),
        operator="ts_return", field="close", relationship=ExpectedRelationship.POSITIVE,
        transform_label="Momentum (trailing return)",
        why_sign=(
            "Continuation: if the consequence diffuses into the price gradually, the response already visible over "
            "the trailing window keeps going over the next window. Reversal (overreaction) would be a different, "
            "negative-sign hypothesis -- the diagnostics would show it; the sign is never flipped after the fact."
        ),
    ),
    SignalRule(
        "activity-confirmation", ("futures.activity", "etf.activity", "equity.activity"),
        MeasurementSignalRole.CONFIRMATION_ONLY,
        rationale=(
            "Trading activity says whether the market is paying attention, not which way: it is kept as a "
            "confirmation input for a later event-conditioned test, never a standalone directional candidate."
        ),
    ),
)
_RULE_BY_MEASUREMENT = {mid: r for r in SIGNAL_RULES for mid in r.measurement_ids}


class ConditioningStatus(str, Enum):
    #: The candidate's factor IS this component.
    IN_FACTOR = "IN_FACTOR"
    #: No valid quantitative magnitude exists on the platform -- kept UNKNOWN,
    #: never invented.
    UNKNOWN_MAGNITUDE = "UNKNOWN_MAGNITUDE"
    #: Deliberately not derived (transmission weight is never read off link
    #: confidence).
    NOT_DERIVED = "NOT_DERIVED"


class EventConditioning(BaseModel):
    """Where the candidate sits in the conceptual event-conditioned model
    Signal = EventStrength x Exposure x Confirmation x TransmissionWeight --
    each a separately testable ablation dimension, none assumed to help."""

    model_config = {"frozen": True, "extra": "forbid"}

    variant: Literal["UNCONDITIONAL_MEASUREMENT"] = "UNCONDITIONAL_MEASUREMENT"
    event_strength: ConditioningStatus = ConditioningStatus.UNKNOWN_MAGNITUDE
    exposure: ConditioningStatus = ConditioningStatus.UNKNOWN_MAGNITUDE
    confirmation: ConditioningStatus = ConditioningStatus.IN_FACTOR
    transmission_weight: ConditioningStatus = ConditioningStatus.NOT_DERIVED
    note: str = (
        "Unconditional: the factor is the market's own price response (the confirmation term) on every day of "
        "history -- it does not know about this event. Event strength has no quantitative magnitude yet (the "
        "mechanism pass reports magnitude_established=False); exposure is only a fidelity category; transmission "
        "weight is never derived from link confidence. An event-conditioned test needs a history of comparable "
        "events and is a later stage."
    )

    @model_validator(mode="after")
    def _no_invented_magnitude(self) -> EventConditioning:
        if ConditioningStatus.IN_FACTOR in (self.event_strength, self.exposure, self.transmission_weight):
            raise ValueError("no event-strength, exposure or transmission-weight magnitude exists to put in a factor")
        if self.transmission_weight is not ConditioningStatus.NOT_DERIVED:
            raise ValueError("transmission weight is never derived (link confidence is not magnitude)")
        return self


class SignalRationale(BaseModel):
    """The answers to "why this transform / sign / lookback / horizon?"."""

    model_config = {"frozen": True, "extra": "forbid"}

    transform: str
    sign: str
    formation_lookback: str
    prediction_horizon: str


class SignalDataBinding(BaseModel):
    """The Phase D `DataFieldResolution` the factor reads -- copied, never
    re-resolved, and re-checked: only real, point-in-time safe, executable,
    usable data may back a candidate."""

    model_config = {"frozen": True, "extra": "forbid"}

    measurement_spec_id: str
    resolution_status: ResolutionStatus
    dataset: str
    data_schema: str
    field: str
    frequency: DataFrequency
    coverage_start: date
    coverage_end_exclusive: date
    coverage_note: str
    pit_safe: Literal[True]
    pit_rule: str
    publication_lag: PublicationLag
    data_role: Literal[DataProvenanceRole.REAL]
    feature_kinds: tuple[str, ...]
    proxy_note: str | None = None
    provenance: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _usable_and_before_holdout(self) -> SignalDataBinding:
        if self.resolution_status not in (
            ResolutionStatus.AVAILABLE, ResolutionStatus.AVAILABLE_WITH_PROXY, ResolutionStatus.PARTIAL,
        ):
            raise ValueError(f"a {self.resolution_status.value} measurement can never back a candidate")
        if self.coverage_end_exclusive > _HOLDOUT:
            raise ValueError("a candidate's data may never reach into the locked holdout")
        return self


class SignalSpec(BaseModel):
    """The typed quantitative specification a factor expression is compiled
    from. No return, probability, weight or verdict field exists."""

    model_config = {"frozen": True, "extra": "forbid"}

    instrument: str
    domain: MandateDomain
    measurement_id: str
    measurement_label: str
    measurement_role: MeasurementRole
    signal_rule_id: str
    transform_label: str
    expression: FactorExpression
    #: Bars the factor looks back over -- independent of the horizon.
    formation_lookback: int
    #: Bars between the information and the first permitted execution: the
    #: factor at bar t's close is acted on no earlier than bar t+1's open.
    execution_lag_bars: Literal[1] = 1
    expected_relationship: ExpectedRelationship
    #: Trading days of forward return the factor is asked to predict --
    #: never derived from an economic transmission lag.
    prediction_horizon: SignalHorizon
    #: The generation policy that proposed this (lookback, horizon) pair.
    generation_policy: str = GENERATION_POLICY
    data: SignalDataBinding
    rationale: SignalRationale

    @model_validator(mode="after")
    def _consistent(self) -> SignalSpec:
        if self.expression.cross_sectional:
            raise ValueError("a single-instrument candidate cannot carry a cross-sectional operator")
        if self.expression.kind not in self.data.feature_kinds:
            raise ValueError("the expression must compute a feature kind the measurement resolved as executable")
        if dict(self.expression.params).get("n", self.formation_lookback) != self.formation_lookback:
            raise ValueError("the declared formation lookback must equal the expression's")
        return self

    @property
    def expression_text(self) -> str:
        return self.expression.render()


class CandidateOrigin(BaseModel):
    """One originating hypothesis: event -> path -> consequence -> asset
    expression -> measurement. A candidate keeps every one."""

    model_config = {"frozen": True, "extra": "forbid"}

    event_id: str
    event_headline: str
    mechanism_graph_id: str
    path_id: str
    path_signature: str
    path_type: PathType | None
    transmission_depth: int
    value_chain_steps: int
    sector_transitions: int | None
    path_status: PathStatus
    route: tuple[str, ...]
    consequence_state: str
    consequence_label: str
    consequence_direction: ImpliedMovement
    #: The path's ECONOMIC transmission lag -- provenance only, never
    #: converted into the candidate's prediction horizon.
    transmission_lag: TransmissionLag
    expression_id: str
    expression_concept: str
    #: Provenance only -- never a score, rank, strength or filter.
    expression_fidelity: ExpressionFidelity
    expression_form: ExpressionForm
    expression_relation: Polarity
    #: The first-order pressure the path implies on the instrument (a
    #: hypothesis); ``None`` when unsigned.
    pressure: PressureSign | None
    measurement_spec_id: str
    measurement_id: str


class CandidateNote(str, Enum):
    """Typed qualifications of a candidate -- never blockers, never scores."""

    PROXY_MEASUREMENT = "PROXY_MEASUREMENT"
    PARTIAL_HISTORY = "PARTIAL_HISTORY"
    OPPOSING_PRESSURES = "OPPOSING_PRESSURES"
    UNRESOLVED_PATH_ORIGIN = "UNRESOLVED_PATH_ORIGIN"
    NO_RESEARCHABLE_PATH = "NO_RESEARCHABLE_PATH"


NOTE_TEXT: dict[CandidateNote, str] = {
    CandidateNote.PROXY_MEASUREMENT: "the measured quantity is a proxy for the ideal one",
    CandidateNote.PARTIAL_HISTORY: "the data covers only part of the research window",
    CandidateNote.OPPOSING_PRESSURES: "origins imply opposite pressure on this instrument",
    CandidateNote.UNRESOLVED_PATH_ORIGIN: "at least one originating path is UNRESOLVED",
    CandidateNote.NO_RESEARCHABLE_PATH: "no originating path is RESEARCHABLE (all proposed or unresolved)",
}


class CandidateSignal(BaseModel):
    """A testable factor hypothesis with its complete lineage. Status is
    always CANDIDATE here: screening is a separate object
    (`screening.candidate_signal_screen`) and validation belongs to the
    registry alone."""

    model_config = {"frozen": True, "extra": "forbid"}

    candidate_signal_id: str
    factor_identity: str
    name: str
    spec: SignalSpec
    origins: tuple[CandidateOrigin, ...]
    conditioning: EventConditioning = EventConditioning()
    status: Literal["CANDIDATE"] = "CANDIDATE"
    notes: tuple[CandidateNote, ...] = ()

    @model_validator(mode="after")
    def _lineage(self) -> CandidateSignal:
        if not self.origins:
            raise ValueError("a candidate signal exists only with at least one originating hypothesis")
        if {o.measurement_spec_id for o in self.origins} != {self.spec.data.measurement_spec_id}:
            raise ValueError("every origin must reach this candidate's own measurement")
        if self.factor_identity != factor_identity(self.spec) or self.candidate_signal_id != candidate_signal_id(
            self.spec,
        ):
            raise ValueError("identity must be computed from the spec")
        return self

    @property
    def expression(self) -> str:
        return self.spec.expression_text

    @property
    def consequences(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(o.consequence_label for o in self.origins))

    @property
    def path_types(self) -> tuple[PathType | None, ...]:
        order = {t: i for i, t in enumerate(PathType)}
        return tuple(sorted({o.path_type for o in self.origins}, key=lambda t: order.get(t, 99)))

    @property
    def depths(self) -> tuple[int, ...]:
        return tuple(sorted({o.transmission_depth for o in self.origins}))

    @property
    def fidelities(self) -> tuple[ExpressionFidelity, ...]:
        return tuple(dict.fromkeys(o.expression_fidelity for o in self.origins))

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(o.event_id for o in self.origins))


class RefusalReason(str, Enum):
    #: The Phase D gate refused the expression (not continuing, no
    #: research-ready instrument, or no usable measurement on it).
    NOT_CANDIDATE_READY = "NOT_CANDIDATE_READY"
    CONFIRMATION_ONLY_MEASUREMENT = "CONFIRMATION_ONLY_MEASUREMENT"
    NO_SIGNAL_RULE = "NO_SIGNAL_RULE"


class CandidateRefusal(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    expression_id: str
    concept: str
    domain: MandateDomain
    reason: RefusalReason
    note: str
    instrument: str | None = None
    measurement_id: str | None = None


class CandidateSignalSet(BaseModel):
    """Every candidate signal for one event's asset expressions under one
    mandate, plus every typed refusal. `plane` is fixed to HYPOTHESIS."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = CANDIDATE_SET_SCHEMA_VERSION
    plane: Literal["HYPOTHESIS"] = "HYPOTHESIS"
    rules: tuple[str, ...] = (SIGNAL_RULES_VERSION, GENERATION_POLICY, EXPRESSION_VOCABULARY_VERSION)
    event_id: str
    event_headline: str
    signal_paths_id: str
    asset_expression_id: str
    mandate_fingerprint: str
    candidates: tuple[CandidateSignal, ...]
    refusals: tuple[CandidateRefusal, ...] = ()
    not_validated_note: str = (
        "Candidate signals are testable factor hypotheses -- not validated factors, alpha, recommendations, "
        "trades, portfolio weights, or scientific verdicts. Screening diagnostics can only say whether one "
        "deserves deeper validation; only the registry's validation can validate."
    )

    @model_validator(mode="after")
    def _unique(self) -> CandidateSignalSet:
        ids = [c.candidate_signal_id for c in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("one candidate per structural identity -- origins are merged, never duplicated")
        return self

    @property
    def is_empty(self) -> bool:
        return not self.candidates

    def candidate(self, candidate_signal_id: str) -> CandidateSignal:
        return next(c for c in self.candidates if c.candidate_signal_id == candidate_signal_id)

    def instruments(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(c.spec.instrument for c in self.candidates))

    def refusals_by_reason(self) -> dict[RefusalReason, tuple[CandidateRefusal, ...]]:
        return {r: tuple(x for x in self.refusals if x.reason is r) for r in RefusalReason
                if any(x.reason is r for x in self.refusals)}

    def fingerprint(self) -> str:
        return "candsig1:" + hashlib.sha256(self.model_dump_json().encode()).hexdigest()


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def _sha(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def factor_identity(spec: SignalSpec) -> str:
    """STRUCTURAL identity of the executable factor: the same instrument,
    data and expression (registered feature + engine version) give the same
    identity, whichever event or path proposed it."""
    return "factor1:" + _sha({
        "instrument": spec.instrument, "domain": spec.domain.value, "dataset": spec.data.dataset,
        "data_schema": spec.data.data_schema, "expression": json.loads(spec.expression.canonical_json()),
    })


def candidate_signal_id(spec: SignalSpec) -> str:
    """The factor plus what it is asked to predict. Excludes events, paths,
    timestamps and display text."""
    return "csig2:" + _sha({
        "factor_identity": factor_identity(spec), "measurement_id": spec.measurement_id,
        "formation_lookback": spec.formation_lookback, "prediction_horizon": spec.prediction_horizon.value,
        "expected_relationship": spec.expected_relationship.value,
        "execution_lag_bars": spec.execution_lag_bars, "signal_rule": spec.signal_rule_id,
        "generation_policy": spec.generation_policy, "rules": SIGNAL_RULES_VERSION,
        "feature_engine_version": FEATURE_ENGINE_VERSION,
    })


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------

_HORIZON_TEXT = {
    SignalHorizon.D1: "1 trading day", SignalHorizon.D5: "5 trading days", SignalHorizon.D20: "20 trading days",
    SignalHorizon.D60: "60 trading days",
}


def _binding(m: MeasurementSpec) -> SignalDataBinding:
    r = m.resolution
    return SignalDataBinding(
        measurement_spec_id=m.spec_id, resolution_status=r.status, dataset=r.dataset, data_schema=r.data_schema,
        field=r.field, frequency=r.frequency, coverage_start=r.coverage_start,
        coverage_end_exclusive=r.coverage_end_exclusive, coverage_note=r.coverage_note, pit_safe=r.pit_safe,
        pit_rule=r.pit_rule, publication_lag=r.publication_lag, data_role=r.data_role,
        feature_kinds=r.feature_kinds, proxy_note=r.proxy_note, provenance=r.provenance,
    )


def _spec(rule: SignalRule, m: MeasurementSpec, point: GenerationPoint) -> SignalSpec:
    lookback, horizon = point.formation_lookback, point.prediction_horizon
    return SignalSpec(
        instrument=m.symbol, domain=m.domain, measurement_id=m.measurement_id, measurement_label=m.label,
        measurement_role=m.role, signal_rule_id=rule.rule_id, transform_label=rule.transform_label,
        expression=FactorExpression.of(rule.operator, rule.field, lookback), formation_lookback=lookback,
        expected_relationship=rule.relationship, prediction_horizon=horizon, data=_binding(m),
        rationale=SignalRationale(
            transform=rule.rationale, sign=rule.why_sign,
            formation_lookback=(
                f"{lookback} bars, from the generation grid ({GENERATION_POLICY}). The grid starts with lookback "
                "matched to horizon as an initial policy, not a rule: the lookback is an independent field, and a "
                "different lookback for the same horizon is a different candidate."
            ),
            prediction_horizon=(
                f"{_HORIZON_TEXT[horizon]}, from the predeclared generation grid -- NOT from the paths' economic "
                "transmission lag, which says when the economics arrive, not when prices react (it is kept on each "
                "origin). The diagnostics report 1D-60D decay rather than assuming either."
            ),
        ),
    )


def _origin(
    path: SignalPath, e: AssetExpression, m: MeasurementSpec, p: ExpressionPressure, *, headline: str,
) -> CandidateOrigin:
    return CandidateOrigin(
        event_id=path.origin_event_id, event_headline=headline, mechanism_graph_id=path.mechanism_graph_id,
        path_id=path.path_id, path_signature=path.signature, path_type=path.path_type,
        transmission_depth=path.transmission_depth, value_chain_steps=path.value_chain_steps,
        sector_transitions=path.sector_transitions, path_status=path.status, route=path.state_labels,
        consequence_state=path.consequence_state, consequence_label=path.consequence_label,
        consequence_direction=p.consequence_direction, transmission_lag=p.horizon, expression_id=e.expression_id,
        expression_concept=e.concept, expression_fidelity=e.fidelity, expression_form=e.form,
        expression_relation=e.relation, pressure=p.pressure, measurement_spec_id=m.spec_id,
        measurement_id=m.measurement_id,
    )


def _notes(spec: SignalSpec, origins: list[CandidateOrigin]) -> tuple[CandidateNote, ...]:
    notes = []
    if spec.data.resolution_status is ResolutionStatus.AVAILABLE_WITH_PROXY:
        notes.append(CandidateNote.PROXY_MEASUREMENT)
    if spec.data.resolution_status is ResolutionStatus.PARTIAL:
        notes.append(CandidateNote.PARTIAL_HISTORY)
    if {PressureSign.UP, PressureSign.DOWN} <= {o.pressure for o in origins}:
        notes.append(CandidateNote.OPPOSING_PRESSURES)
    statuses = {o.path_status for o in origins}
    if PathStatus.UNRESOLVED in statuses:
        notes.append(CandidateNote.UNRESOLVED_PATH_ORIGIN)
    if PathStatus.RESEARCHABLE not in statuses:
        notes.append(CandidateNote.NO_RESEARCHABLE_PATH)
    return tuple(notes)


def _name(spec: SignalSpec) -> str:
    return f"{spec.instrument} {spec.formation_lookback}-day price momentum → {spec.prediction_horizon.value}"


def build_candidate_signals(
    plan: AssetExpressionPlan,
    discovery: SignalPathDiscovery,
    *,
    grid: tuple[GenerationPoint, ...] = DEFAULT_GENERATION_GRID,
) -> CandidateSignalSet:
    """``candidate-signal-rules/2`` over every expression of ``plan``, through
    the Phase D gate only, at every point of the generation ``grid``. Pure
    and deterministic."""
    if plan.signal_paths_id != discovery.fingerprint():
        raise ValueError("the asset-expression plan was built from different signal paths")
    paths = {p.path_id: p for p in discovery.paths}
    specs: dict[str, SignalSpec] = {}
    origins: dict[str, list[CandidateOrigin]] = defaultdict(list)
    refusals: dict[tuple, CandidateRefusal] = {}

    def refuse(e: AssetExpression, reason: RefusalReason, note: str, m: MeasurementSpec | None = None) -> None:
        key = (e.expression_id, m.spec_id if m else None, reason)
        refusals.setdefault(key, CandidateRefusal(
            expression_id=e.expression_id, concept=e.concept, domain=e.domain, reason=reason, note=note,
            instrument=m.symbol if m else None, measurement_id=m.measurement_id if m else None,
        ))

    for e in plan.expressions:
        try:
            basis = plan.candidate_signal_basis(e)
        except CandidateSignalPrerequisiteError as err:
            refuse(e, RefusalReason.NOT_CANDIDATE_READY, str(err))
            continue
        for m in basis:
            rule = _RULE_BY_MEASUREMENT.get(m.measurement_id)
            if rule is None:
                refuse(e, RefusalReason.NO_SIGNAL_RULE, f"no reviewed signal rule turns {m.label.lower()} into a "
                                                        "factor yet", m)
                continue
            if rule.role is MeasurementSignalRole.CONFIRMATION_ONLY:
                refuse(e, RefusalReason.CONFIRMATION_ONLY_MEASUREMENT, rule.rationale, m)
                continue
            for point in grid:
                spec = _spec(rule, m, point)
                cid = candidate_signal_id(spec)
                specs.setdefault(cid, spec)
                origins[cid] += [
                    _origin(paths[pid], e, m, pressure, headline=plan.event_headline)
                    for pressure in e.pressures for pid in pressure.path_ids
                ]

    domain_order = {d: i for i, d in enumerate(MandateDomain)}
    candidates = sorted(
        (
            CandidateSignal(
                candidate_signal_id=cid, factor_identity=factor_identity(spec), name=_name(spec), spec=spec,
                origins=tuple({(o.path_id, o.expression_id): o for o in origins[cid]}.values()),
                notes=_notes(spec, origins[cid]),
            )
            for cid, spec in specs.items()
        ),
        key=lambda c: (domain_order[c.spec.domain], c.spec.instrument, c.spec.prediction_horizon.days,
                       c.spec.formation_lookback),
    )
    return CandidateSignalSet(
        event_id=plan.event_id, event_headline=plan.event_headline, signal_paths_id=plan.signal_paths_id,
        asset_expression_id=plan.fingerprint(), mandate_fingerprint=plan.mandate_fingerprint,
        candidates=tuple(candidates), refusals=tuple(refusals.values()),
    )
