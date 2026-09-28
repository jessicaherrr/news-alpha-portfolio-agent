"""Research Angles (Release UX Part C, task spec sections 8-11).

A `ResearchAnglePlan` is built BEFORE hypothesis generation and states, for
each planned generation slot, WHY that idea is being pursued -- not just
WHAT economic mechanism it targets (`alpha_agent.discovery.mechanisms`
already guarantees mechanism diversity within one campaign; this module adds
the orthogonal "research angle" dimension the mission asks for: baseline
replication vs. failure-guided vs. source-inspired vs. user-constraint-driven,
etc.).

Integration is deliberately ADDITIVE and zero-risk to the frozen Phase 18
orchestrator: `generate_candidate_pool` (`alpha_agent.discovery.
candidate_pool`) already takes a plain `objective: str` the caller fully
controls, so a `ResearchAnglePlan` is consumed by building an augmented
objective string per slot via :func:`augment_objective_for_angle` -- nothing
in `ResearchOrchestrator.plan_family` or `generate_candidate_pool` itself is
touched, subclassed, or monkeypatched.

Diversity is measured, never gamed: :func:`mechanism_diversity_report` reads
an already-produced `CandidatePoolResult` and reports how many DISTINCT
mechanisms actually produced a surviving member -- a pool of "MA 20/50, MA
30/80, MA 40/100" is one mechanism with three parameter variants, and this
function calls that what it is (task spec section 9/53: "trivial parameter
variations do not count as diversity").
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.recommendation.profile import InvestorProfile, OvernightPreference


class ResearchAngle(str, Enum):
    """The ten lanes task spec section 8 names. Purely descriptive metadata
    -- an angle never changes scientific validity or the BH/FDR family; it
    only shapes WHICH economically-motivated idea gets proposed and WHY."""

    BASELINE_REPLICATION = "BASELINE_REPLICATION"
    FAILURE_GUIDED = "FAILURE_GUIDED"
    MARKET_SPECIFIC_ADAPTATION = "MARKET_SPECIFIC_ADAPTATION"
    REGIME_CONDITIONED = "REGIME_CONDITIONED"
    SESSION_MICROSTRUCTURE = "SESSION_MICROSTRUCTURE"
    CROSS_MARKET = "CROSS_MARKET"
    RELATIVE_VALUE = "RELATIVE_VALUE"
    SOURCE_INSPIRED = "SOURCE_INSPIRED"
    USER_CONSTRAINT_DRIVEN = "USER_CONSTRAINT_DRIVEN"
    NOVEL_COMBINATION = "NOVEL_COMBINATION"


#: Fixed default ordering when no failure memory / profile signal favors a
#: specific lane -- documented and identical for every run (mirrors
#: `alpha_agent.discovery.mechanisms.DEFAULT_MECHANISM_ORDER`'s own
#: "fixed, documented, identical for every run" discipline).
DEFAULT_ANGLE_ORDER: tuple[ResearchAngle, ...] = (
    ResearchAngle.BASELINE_REPLICATION,
    ResearchAngle.FAILURE_GUIDED,
    ResearchAngle.MARKET_SPECIFIC_ADAPTATION,
    ResearchAngle.SOURCE_INSPIRED,
    ResearchAngle.USER_CONSTRAINT_DRIVEN,
    ResearchAngle.REGIME_CONDITIONED,
    ResearchAngle.SESSION_MICROSTRUCTURE,
    ResearchAngle.CROSS_MARKET,
    ResearchAngle.RELATIVE_VALUE,
    ResearchAngle.NOVEL_COMBINATION,
)

#: Angle -> the mechanisms it most naturally draws from. Guidance for the
#: objective text only -- never a gate on what the ResearchAgent may
#: actually propose (the frozen orchestrator's own market-universe /
#: feature-catalog guardrails remain the only enforcement).
ANGLE_MECHANISM_HINTS: dict[ResearchAngle, tuple[EconomicMechanism, ...]] = {
    ResearchAngle.BASELINE_REPLICATION: (EconomicMechanism.TREND, EconomicMechanism.MOMENTUM,
                                          EconomicMechanism.MEAN_REVERSION),
    ResearchAngle.FAILURE_GUIDED: (),  # derived from the actual prior failure, not a fixed list
    ResearchAngle.MARKET_SPECIFIC_ADAPTATION: (EconomicMechanism.SESSION_EFFECTS, EconomicMechanism.OVERNIGHT_GAP,
                                                EconomicMechanism.VOLUME_LIQUIDITY),
    ResearchAngle.REGIME_CONDITIONED: (EconomicMechanism.REGIME_CONDITIONED_TREND,
                                        EconomicMechanism.REGIME_CONDITIONED_MEAN_REVERSION,
                                        EconomicMechanism.VOLATILITY_TRANSITION),
    ResearchAngle.SESSION_MICROSTRUCTURE: (EconomicMechanism.OPENING_RANGE, EconomicMechanism.SESSION_EFFECTS,
                                            EconomicMechanism.VOLUME_LIQUIDITY),
    ResearchAngle.CROSS_MARKET: (EconomicMechanism.CROSS_MARKET_LEAD_LAG, EconomicMechanism.CORRELATION_SPREAD),
    ResearchAngle.RELATIVE_VALUE: (EconomicMechanism.RELATIVE_VALUE, EconomicMechanism.CARRY,
                                    EconomicMechanism.TERM_STRUCTURE),
    ResearchAngle.SOURCE_INSPIRED: (),  # derived from the actual cited knowledge-base item, not a fixed list
    ResearchAngle.USER_CONSTRAINT_DRIVEN: (),  # derived from the actual stated constraint
    ResearchAngle.NOVEL_COMBINATION: (EconomicMechanism.MULTI_SIGNAL_ENSEMBLE, EconomicMechanism.HYBRID_TREND_REVERSAL),
}


class ResearchAngleAssignment(BaseModel):
    """One planned generation slot's angle + why (task spec section 11: a
    successor must state what changed, why, what prior failure it addresses,
    and how it can be falsified -- `rationale`/`addresses_failure`/
    `falsification_hint` carry exactly those, deterministically, never a new
    LLM judgment)."""

    model_config = {"frozen": True, "extra": "forbid"}

    angle: ResearchAngle
    rationale: str
    addresses_failure: str | None = None
    falsification_hint: str | None = None
    mechanism_hints: tuple[EconomicMechanism, ...] = ()


class ResearchAnglePlan(BaseModel):
    """The full, ordered plan -- built BEFORE any hypothesis is proposed
    (task spec section 8)."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "research-angle-plan/1"
    market: str
    assignments: tuple[ResearchAngleAssignment, ...]

    @property
    def angles(self) -> tuple[ResearchAngle, ...]:
        return tuple(a.angle for a in self.assignments)


def _user_constraint_rationale(profile: InvestorProfile) -> str | None:
    """Only fires on an ACTUAL stated constraint -- mirrors
    `alpha_agent.recommendation.fit`'s "no preference" exclusion discipline,
    never inventing a constraint from an untouched default profile."""
    clauses: list[str] = []
    if profile.overnight == OvernightPreference.AVOID:
        clauses.append("avoid overnight exposure")
    if profile.max_drawdown.value not in ("15%",):  # narrower than the platform default
        clauses.append(f"keep drawdown within {profile.max_drawdown.value}")
    if profile.strategy_preference.value != "Mixed / No Preference":
        clauses.append(f"prefer a {profile.strategy_preference.value.lower()} mechanism")
    if not clauses:
        return None
    return "User-stated constraint: " + "; ".join(clauses) + "."


#: Angles that only ever appear via their explicit evidence-gated
#: conditional in `build_research_angle_plan` -- never as a generic
#: fallback-fill choice (see the loop below).
_CONDITIONAL_ONLY_ANGLES: frozenset[ResearchAngle] = frozenset(
    {ResearchAngle.FAILURE_GUIDED, ResearchAngle.SOURCE_INSPIRED, ResearchAngle.USER_CONSTRAINT_DRIVEN}
)


def build_research_angle_plan(
    *,
    market: str,
    profile: InvestorProfile | None = None,
    has_prior_failures: bool = False,
    failure_lesson: str | None = None,
    has_source_material: bool = False,
    max_angles: int = 6,
) -> ResearchAnglePlan:
    """Deterministic construction (task spec section 8): profile/failure-
    memory signals may REORDER/INCLUDE lanes, never invent scientific
    evidence. `max_angles` bounds the plan -- never an uncontrolled sweep."""
    if max_angles < 1:
        raise ValueError("max_angles must be >= 1")

    assignments: list[ResearchAngleAssignment] = []

    assignments.append(
        ResearchAngleAssignment(
            angle=ResearchAngle.BASELINE_REPLICATION,
            rationale=f"Establish a known baseline mechanism for {market} before exploring variations.",
            mechanism_hints=ANGLE_MECHANISM_HINTS[ResearchAngle.BASELINE_REPLICATION],
        )
    )

    if has_prior_failures:
        assignments.append(
            ResearchAngleAssignment(
                angle=ResearchAngle.FAILURE_GUIDED,
                rationale="A prior related hypothesis has committed registry evidence -- propose an "
                          "economically distinct successor, not a parameter retune.",
                addresses_failure=failure_lesson,
                falsification_hint="Falsified if the successor mechanism repeats the same weak-regime "
                                    "behaviour the prior failure exhibited.",
            )
        )

    assignments.append(
        ResearchAngleAssignment(
            angle=ResearchAngle.MARKET_SPECIFIC_ADAPTATION,
            rationale=f"Consider {market}'s own session structure and liquidity pattern rather than a "
                      "generic cross-market template.",
            mechanism_hints=ANGLE_MECHANISM_HINTS[ResearchAngle.MARKET_SPECIFIC_ADAPTATION],
        )
    )

    if has_source_material:
        assignments.append(
            ResearchAngleAssignment(
                angle=ResearchAngle.SOURCE_INSPIRED,
                rationale="External knowledge-base material exists for this market/mechanism neighbourhood "
                          "-- ground a proposal in it (lineage only, never treated as our own evidence).",
            )
        )

    if profile is not None:
        constraint = _user_constraint_rationale(profile)
        if constraint is not None:
            assignments.append(
                ResearchAngleAssignment(angle=ResearchAngle.USER_CONSTRAINT_DRIVEN, rationale=constraint)
            )

    for angle in DEFAULT_ANGLE_ORDER:
        if len(assignments) >= max_angles:
            break
        if angle in {a.angle for a in assignments}:
            continue
        if angle in _CONDITIONAL_ONLY_ANGLES:
            # These three require REAL evidence (a prior failure, source
            # material, a stated constraint) -- handled by the explicit
            # conditionals above only. Filling them in generically here
            # would fabricate a rationale ("diversify along the
            # failure-guided lane") for evidence that does not exist.
            continue
        assignments.append(
            ResearchAngleAssignment(
                angle=angle,
                rationale=f"Diversify the candidate pool along the {angle.value.replace('_', ' ').lower()} lane.",
                mechanism_hints=ANGLE_MECHANISM_HINTS.get(angle, ()),
            )
        )

    return ResearchAnglePlan(market=market, assignments=tuple(assignments[:max_angles]))


def augment_objective_for_angle(objective: str, assignment: ResearchAngleAssignment) -> str:
    """Builds the per-slot objective string a caller passes to
    `alpha_agent.discovery.candidate_pool.generate_candidate_pool` (whose
    `objective` parameter is already a plain, caller-controlled string) --
    the one sanctioned integration point, never a change to the frozen
    orchestrator itself."""
    lines = [objective.rstrip("."), f"Research angle: {assignment.angle.value}. {assignment.rationale}"]
    if assignment.addresses_failure:
        lines.append(f"Addresses prior finding: {assignment.addresses_failure}")
    if assignment.falsification_hint:
        lines.append(f"Falsification framing: {assignment.falsification_hint}")
    return " ".join(lines)


class ResearchDiversityReport(BaseModel):
    """Task spec sections 9/53: measures REAL diversity, never gamed by
    trivial parameter variation within one mechanism."""

    model_config = {"frozen": True, "extra": "forbid"}

    mechanisms_attempted: int
    mechanisms_supported: int
    distinct_mechanisms_with_members: tuple[str, ...]
    total_members: int
    trivial_parameter_variation_only: bool

    @property
    def mechanism_coverage_ratio(self) -> float:
        return self.mechanisms_supported / self.mechanisms_attempted if self.mechanisms_attempted else 0.0


def mechanism_diversity_report(pool) -> ResearchDiversityReport:
    """`pool` is a `alpha_agent.discovery.candidate_pool.CandidatePoolResult`
    (duck-typed here to avoid a hard import-time dependency in either
    direction). `trivial_parameter_variation_only` is True iff every
    surviving member, across ALL mechanisms, shares one identical feature-kind
    set -- the "MA 20/50, MA 30/80, MA 40/100" anti-pattern the mission names
    explicitly."""
    distinct_mechanisms = tuple(o.mechanism.value for o in pool.per_mechanism if o.members)
    feature_kind_sets: set[frozenset[str]] = set()
    for member in pool.members:
        kinds = frozenset(
            f.get("spec", {}).get("kind", "?") for f in (member.strategy_spec_json.get("features") or [])
        )
        feature_kind_sets.add(kinds)
    trivial_only = len(feature_kind_sets) <= 1 and len(pool.members) > 1
    return ResearchDiversityReport(
        mechanisms_attempted=pool.mechanisms_attempted,
        mechanisms_supported=pool.mechanisms_supported,
        distinct_mechanisms_with_members=distinct_mechanisms,
        total_members=len(pool.members),
        trivial_parameter_variation_only=trivial_only,
    )


__all__ = [
    "ANGLE_MECHANISM_HINTS",
    "DEFAULT_ANGLE_ORDER",
    "ResearchAngle",
    "ResearchAngleAssignment",
    "ResearchAnglePlan",
    "ResearchDiversityReport",
    "augment_objective_for_angle",
    "build_research_angle_plan",
    "mechanism_diversity_report",
]
