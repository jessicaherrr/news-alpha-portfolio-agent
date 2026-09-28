"""Builds :class:`AlphaResearchObject` instances purely by READING existing
sources of truth (prompt 2 section 3/4): the experiment registry and its
:class:`FailureMemory` facade. Nothing here writes a registry row, calls
Claude, calls a market-data API, or runs a backtest -- building or listing
Alpha Research Objects is a pure, offline aggregation over already-committed
evidence (section 18: "opening or viewing an AlphaResearchObject must never
run a backtest / call Databento / call Claude / write Registry / start
Paper").

No caching, no second store: every call recomputes straight from the
registry, exactly like :mod:`alpha_agent.translation.research_memory` and
`alpha_agent.ui.services.list_experiments` already do (CLAUDE.md: "no second
scientific database").

Identity-hardening patch: a mechanism may legitimately map to MORE THAN ONE
Factor (V1's default is one Factor per mapped family -- see
:mod:`alpha_agent.alpha_memory.factor_identity`), so
:func:`build_alpha_research_objects_for_mechanism` returns a TUPLE, never a
single object. Final Phase 2 semantic fix: every real mechanism in today's
registry that maps to more than one family (e.g. TREND -> tsmom, ma_trend)
now produces that many SEPARATE objects, one per family, never merged on the
strength of a shared data-shape requirement alone (see this module's own
tests).
"""
from __future__ import annotations

from alpha_agent.agents.context import build_failure_memory_digest
from alpha_agent.alpha_memory.factor_identity import (
    compute_factor_identity,
    group_families_for_factor_identity,
)
from alpha_agent.alpha_memory.schemas import (
    ALPHA_OBJECT_SCHEMA,
    AlphaResearchObject,
    EvidenceProfile,
    ExperimentEvidenceRef,
    FactorIdentity,
    ResearchMaturity,
    StrategyVariantEvidence,
)
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.registry.enums import AssetDomain, RegistryVerdict, TrialRole
from alpha_agent.registry.failure_memory import FailureMemory
from alpha_agent.registry.sqlite_registry import ExperimentRegistry, ExperimentView
from alpha_agent.strategy.baselines.families import BASELINE_FAMILIES
from alpha_agent.strategy.baselines.silver_bullet import SILVER_BULLET_FAMILY
from alpha_agent.translation.research_memory import MECHANISM_TO_KNOWN_FAMILIES
from alpha_agent.translation.schemas import CERTIFIED_ROOTS
from alpha_agent.validation.fingerprint import fingerprint

__all__ = [
    "MECHANISM_PROVENANCE_NOTE",
    "alpha_research_object_id",
    "build_alpha_research_objects_for_mechanism",
    "get_alpha_research_object",
    "list_alpha_research_objects",
]

_FAMILY_DISPLAY_NAMES: dict[str, str] = {d.key: d.name for d in (*BASELINE_FAMILIES, SILVER_BULLET_FAMILY)}

MECHANISM_PROVENANCE_NOTE = (
    "This AlphaResearchObject's mechanism is a PARENT semantic dimension, read from Phase 1's frozen "
    "EconomicMechanism -> candidate-family bridge (MECHANISM_TO_KNOWN_FAMILIES) -- never proposed or "
    "altered by an LLM. The object's own identity is root + Factor, not root + mechanism: V1 bundles "
    "exactly one strategy family per Factor by default -- see alpha_agent.alpha_memory.factor_identity "
    "for why equal declared data requirements are not treated as proof two families share one Factor."
)

#: Reason codes the frozen ReliabilityPolicy emits when it explicitly ran a
#: cost-stress / parameter-stability check and it came back weak -- the same
#: codes `alpha_agent.ui.services.GATE_DEFINITIONS` uses for its gate grid.
#: Reused here (as plain strings, not imported, to keep this module free of a
#: `streamlit`-adjacent UI dependency) so the Evidence Profile's categorical
#: read matches the Validation tab's own vocabulary exactly.
_COST_STRESS_FAIL_CODE = "cost_stress_degradation_exceeds_limit"
_PARAMETER_INSTABILITY_CODE = "parameter_neighbourhood_unstable"


def alpha_research_object_id(root_symbol: str, factor_identity: str) -> str:
    """Deterministic AlphaResearchObject identity: composed from the stable
    ``root_symbol`` and the real ``FactorIdentity.factor_identity`` -- never
    an ``experiment_id`` (section 6), and never ``root_symbol`` + mechanism
    alone (identity-hardening patch, section 2)."""
    return fingerprint(
        "alpharesearchobject3",
        {"schema": ALPHA_OBJECT_SCHEMA, "root_symbol": root_symbol, "factor_identity": factor_identity},
    )


def _research_maturity(*, n_mapped_families: int, n_experiments: int, n_adjudicated: int) -> ResearchMaturity:
    if n_mapped_families == 0:
        return ResearchMaturity.IDEA
    if n_experiments == 0:
        return ResearchMaturity.FORMALIZED
    if n_adjudicated >= 1:
        return ResearchMaturity.ADJUDICATED
    if n_experiments >= 2:
        return ResearchMaturity.REPLICATED
    return ResearchMaturity.TESTED


def _scientific_evidence_for_family(experiments: list[ExperimentView], family: str) -> str:
    canonical = [v for v in experiments if v.experiment.strategy_family == family and v.experiment.trial_role is TrialRole.CANONICAL]
    if not canonical:
        return "NO_CANONICAL_TRIAL"
    verdicts = {(v.verdict.value if v.verdict is not None else "NOT_ADJUDICATED") for v in canonical}
    if len(verdicts) == 1:
        return verdicts.pop()
    return "MIXED(" + ",".join(sorted(verdicts)) + ")"


def _evidence_profile(experiments: list[ExperimentView], families: tuple[str, ...]) -> EvidenceProfile:
    scientific = {f: _scientific_evidence_for_family(experiments, f) for f in families}
    related_experiment_count = len(experiments)

    canonical_results = [
        v.result for v in experiments if v.experiment.trial_role is TrialRole.CANONICAL and v.result is not None
    ]
    cost_robustness = "NOT_EVALUATED"
    parameter_stability = "NOT_EVALUATED"
    regime_breadth = "NOT_EVALUATED"
    if canonical_results:
        if any(_COST_STRESS_FAIL_CODE in r.reason_codes for r in canonical_results):
            cost_robustness = "FAILED_STRESS"
        elif any(r.cost_stress for r in canonical_results):
            cost_robustness = "EVALUATED"

        if any(_PARAMETER_INSTABILITY_CODE in r.reason_codes for r in canonical_results):
            parameter_stability = "WEAK"
        elif any(r.parameter_stability for r in canonical_results):
            parameter_stability = "EVALUATED"

        if any((r.regime_evidence or {}).get("status") == "evaluated" for r in canonical_results):
            regime_breadth = "EVALUATED"

    return EvidenceProfile(
        scientific_evidence=scientific,
        related_experiment_count=related_experiment_count,
        related_experiment_label=f"{related_experiment_count} related experiment(s)",
        cost_robustness=cost_robustness,
        parameter_stability=parameter_stability,
        regime_breadth=regime_breadth,
    )


def _strategy_variant_evidence(
    fm: FailureMemory, *, family: str, root_symbol: str, family_experiments: list[ExperimentView],
    asset_domain: AssetDomain = AssetDomain.FUTURES,
) -> StrategyVariantEvidence:
    canonical = [v for v in family_experiments if v.experiment.trial_role is TrialRole.CANONICAL]
    canonical_params = dict(canonical[0].experiment.strategy_spec_json.get("params", {})) if canonical else None
    digest = build_failure_memory_digest(
        fm.lookup(strategy_family=family, root_symbol=root_symbol, asset_domain=asset_domain)
    )
    return StrategyVariantEvidence(
        strategy_family=family,
        strategy_display_name=_FAMILY_DISPLAY_NAMES.get(family, family),
        n_experiments=len(family_experiments),
        canonical_experiment_ids=tuple(sorted(v.experiment_id for v in canonical)),
        strategy_fingerprints=tuple(sorted({v.experiment.strategy_fingerprint for v in family_experiments})),
        canonical_params=canonical_params,
        digest=digest,
    )


def _build_summary(
    *,
    root_symbol: str,
    mechanism: EconomicMechanism,
    variants: tuple[StrategyVariantEvidence, ...],
    evidence_profile: EvidenceProfile,
    maturity: ResearchMaturity,
) -> str:
    fam_list = ", ".join(v.strategy_display_name for v in variants)
    verdict_bits = "; ".join(f"{fam}: {v}" for fam, v in evidence_profile.scientific_evidence.items())
    return (
        f"{mechanism.value} on {root_symbol}: {len(variants)} strategy implementation(s) ({fam_list}), "
        f"{evidence_profile.related_experiment_label}, research maturity {maturity.value}. "
        f"Scientific evidence -- {verdict_bits}."
    )


def _build_for_family_group(
    registry: ExperimentRegistry,
    *,
    root_symbol: str,
    mechanism: EconomicMechanism,
    families: tuple[str, ...],
    asset_domain: AssetDomain = AssetDomain.FUTURES,
) -> AlphaResearchObject | None:
    """One (root, Factor) AlphaResearchObject for the families in ONE
    already-grouped family group (a V1 singleton by default -- see
    :func:`alpha_agent.alpha_memory.factor_identity.group_families_for_factor_identity`),
    or ``None`` when there is no real registry evidence to ground it in yet.

    Deliberately returns ``None`` (rather than an empty IDEA/FORMALIZED-level
    object) whenever zero real registry experiments have ever been attempted
    for any family in this group on this root -- the Personal Alpha Library
    only materializes objects with real evidence (section 9); a "have I
    researched this before" lookup with no match is an honest ``None``, not
    a fabricated empty entry.
    """
    factor: FactorIdentity = compute_factor_identity(mechanism, families)

    experiments: list[ExperimentView] = []
    for family in families:
        experiments.extend(
            registry.experiments(
                strategy_family=family, root_symbol=root_symbol,
                asset_domain=asset_domain, authoritative_only=True,
            )
        )
    if not experiments:
        return None

    fm = FailureMemory(registry)
    variants = tuple(
        _strategy_variant_evidence(
            fm, family=family, root_symbol=root_symbol, asset_domain=asset_domain,
            family_experiments=[v for v in experiments if v.experiment.strategy_family == family],
        )
        for family in families
    )

    exp_refs = tuple(
        ExperimentEvidenceRef(
            experiment_id=v.experiment_id,
            experiment_identity=v.experiment_identity,
            strategy_family=v.experiment.strategy_family,
            trial_role=v.experiment.trial_role,
            parameter_variant_label=v.experiment.parameter_variant_label,
            headline_verdict=v.verdict,
            reason_codes=v.result.reason_codes if v.result else (),
            n_valid_attempts=v.n_valid_attempts,
            n_invalid_attempts=v.n_invalid_attempts,
            has_valid_authoritative_result=v.has_valid_authoritative_result,
            holdout_eligible=v.result.holdout_eligible if v.result else False,
            authority=v.authority,
            market_window=f"{v.experiment.market_window.start_date}..{v.experiment.market_window.end_date}",
        )
        for v in sorted(experiments, key=lambda v: v.experiment_id)
    )

    evidence_profile = _evidence_profile(experiments, families)

    n_adjudicated = sum(
        1
        for v in experiments
        if v.experiment.trial_role is TrialRole.CANONICAL
        and v.verdict is not None
        and v.verdict is not RegistryVerdict.NOT_ADJUDICATED
    )
    maturity = _research_maturity(
        n_mapped_families=len(families), n_experiments=len(experiments), n_adjudicated=n_adjudicated
    )

    reason_codes: dict[str, int] = {}
    failure_classes: dict[str, int] = {}
    for family in families:
        resp = fm.lookup(strategy_family=family, root_symbol=root_symbol, asset_domain=asset_domain)
        for code, n in resp.reason_code_counts.items():
            reason_codes[code] = reason_codes.get(code, 0) + n
        for cls, n in resp.failure_class_counts.items():
            failure_classes[cls] = failure_classes.get(cls, 0) + n
    engineering_lessons = tuple(
        f"{f.failure_code}: {f.summary}" for f in fm.related_engineering_failures(root_symbol=root_symbol)
    )

    return AlphaResearchObject(
        alpha_id=alpha_research_object_id(root_symbol, factor.factor_identity),
        root_symbol=root_symbol,
        mechanism=mechanism,
        mechanism_provenance=MECHANISM_PROVENANCE_NOTE,
        factor=factor,
        strategy_variants=variants,
        experiments=exp_refs,
        evidence_profile=evidence_profile,
        research_maturity=maturity,
        repeated_failure_reason_codes=reason_codes,
        repeated_failure_classes=failure_classes,
        engineering_lessons=engineering_lessons,
        summary=_build_summary(
            root_symbol=root_symbol, mechanism=mechanism, variants=variants,
            evidence_profile=evidence_profile, maturity=maturity,
        ),
    )


def build_alpha_research_objects_for_mechanism(
    registry: ExperimentRegistry, *, root_symbol: str, mechanism: EconomicMechanism,
    asset_domain: AssetDomain = AssetDomain.FUTURES,
    family_groups_override: tuple[tuple[str, ...], ...] | None = None,
) -> tuple[AlphaResearchObject, ...]:
    """Every real, evidence-grounded AlphaResearchObject for one (root,
    mechanism) pair -- ZERO, ONE, or (typically, whenever the mechanism maps
    to more than one family) MORE than one, since V1 never merges different
    families into one Factor automatically (a mechanism must never be
    assumed to be exactly one Factor).

    ``family_groups_override`` -- every existing caller omits it, so
    ``group_families_for_factor_identity`` (Phase 1's frozen Futures
    mechanism->family bridge) still drives grouping exactly as before. Phase 6
    (ETF Research Pilot) passes its OWN, separate, small mechanism->family map
    (``alpha_agent.etf.alpha_memory_bridge.ETF_MECHANISM_TO_FAMILIES``) here
    instead -- shared Mechanism VOCABULARY, never shared Factor grouping
    logic or evidence.
    """
    groups = family_groups_override if family_groups_override is not None else group_families_for_factor_identity(mechanism)
    out: list[AlphaResearchObject] = []
    for families in groups:
        obj = _build_for_family_group(
            registry, root_symbol=root_symbol, mechanism=mechanism, families=families, asset_domain=asset_domain,
        )
        if obj is not None:
            out.append(obj)
    return tuple(out)


def _link_related(objs: list[AlphaResearchObject]) -> list[AlphaResearchObject]:
    """Cross-reference AlphaResearchObjects on the SAME root whose mapped
    strategy families overlap -- an honest, structural fact (e.g. BREAKOUT /
    FAILED_BREAKOUT / VOLATILITY_BREAKOUT all resolve to the single
    "breakout" family today), never a similarity score or a merge."""
    linked: list[AlphaResearchObject] = []
    for obj in objs:
        own_families = set(obj.factor.related_strategy_families)
        related = tuple(
            sorted(
                other.alpha_id
                for other in objs
                if other.alpha_id != obj.alpha_id
                and other.root_symbol == obj.root_symbol
                and own_families & set(other.factor.related_strategy_families)
            )
        )
        linked.append(obj.model_copy(update={"related_alpha_ids": related}) if related else obj)
    return linked


def list_alpha_research_objects(
    registry: ExperimentRegistry, *, root_symbol: str | None = None,
    asset_domain: AssetDomain = AssetDomain.FUTURES,
    roots_override: tuple[str, ...] | None = None,
    mechanism_family_map: dict[EconomicMechanism, tuple[str, ...]] | None = None,
) -> tuple[AlphaResearchObject, ...]:
    """Every real, evidence-grounded AlphaResearchObject -- one per (root,
    Factor) with at least one real registry experiment. Only mechanisms
    Phase 1's frozen bridge maps to a real candidate family are even
    considered; an unmapped mechanism (e.g. CARRY) or an unmapped strategy
    family (e.g. ``silver_bullet``, the Phase 15 ``ml_meta_label.*``
    wrappers) is an honest gap, never force-matched.

    ``asset_domain``/``roots_override``/``mechanism_family_map`` -- every
    existing (Futures) caller omits all three, so behaviour is byte-for-byte
    unchanged: ``CERTIFIED_ROOTS`` x ``MECHANISM_TO_KNOWN_FAMILIES``, exactly
    as before. Phase 6 (ETF Research Pilot) is the one caller that supplies
    its own small, separate root list and mechanism->family map."""
    roots = (root_symbol,) if root_symbol else (roots_override or CERTIFIED_ROOTS)
    fam_map = mechanism_family_map if mechanism_family_map is not None else MECHANISM_TO_KNOWN_FAMILIES
    objs: list[AlphaResearchObject] = []
    for root in roots:
        for mechanism, families in fam_map.items():
            objs.extend(
                build_alpha_research_objects_for_mechanism(
                    registry, root_symbol=root, mechanism=mechanism, asset_domain=asset_domain,
                    family_groups_override=tuple((f,) for f in families) if mechanism_family_map is not None else None,
                )
            )
    objs = _link_related(objs)
    return tuple(sorted(objs, key=lambda o: (o.root_symbol, o.mechanism.value, o.alpha_id)))


def get_alpha_research_object(registry: ExperimentRegistry, alpha_id: str) -> AlphaResearchObject | None:
    """Lookup by the object's own deterministic identity."""
    for obj in list_alpha_research_objects(registry):
        if obj.alpha_id == alpha_id:
            return obj
    return None
